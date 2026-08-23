"""Gate 7 artifact audit; preview outputs with --validate-only or --dry-run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import (
    CANONICAL_EXECUTION_MODE,
    GateArtifactError,
    add_validation_arguments,
    gate_failure_artifact_path,
    validation_requested,
    verify_collector_gate_artifact,
    verify_guided_gate_artifact,
    verify_oracle_gate_artifact,
    verify_preflight_gate_artifact,
    verify_seed_gate_artifact,
    verify_trainer_gate_artifact,
)

REQUIRED_GATE_FILES = {
    "preflight": "gate01_preflight.json",
    "seed": "gate02_seed.json",
    "oracle_replay": "gate03_oracle.json",
    "collector": "gate04_collector.json",
    "guided": "gate05_guided_group.json",
    "trainer": "gate06_trainer.json",
}
GATE_VERIFIERS = {
    "preflight": verify_preflight_gate_artifact,
    "seed": verify_seed_gate_artifact,
    "oracle_replay": verify_oracle_gate_artifact,
    "collector": verify_collector_gate_artifact,
    "guided": verify_guided_gate_artifact,
    "trainer": verify_trainer_gate_artifact,
}


def _load_gate_artifact_snapshot(
    path: Path, gate: str
) -> tuple[Mapping[str, Any], str]:
    """Parse and hash one unchanged regular file from the same opened bytes."""

    if path.is_symlink():
        raise GateArtifactError(f"gate {gate} artifact must not be a symlink: {path}")
    try:
        with path.open("rb") as stream:
            opened_before = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened_before.st_mode):
                raise GateArtifactError(
                    f"gate {gate} artifact must be a regular file: {path}"
                )
            raw_bytes = stream.read()
            opened_after = os.fstat(stream.fileno())
        lexical_after = path.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise GateArtifactError(f"missing gate artifact for {gate}: {path}") from error
    except OSError as error:
        raise GateArtifactError(f"cannot read gate {gate} artifact {path}: {error}") from error
    identities = {
        (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
        )
        for value in (opened_before, opened_after, lexical_after)
    }
    if len(identities) != 1 or len(raw_bytes) != opened_after.st_size:
        raise GateArtifactError(f"gate {gate} artifact changed while it was read: {path}")
    try:
        payload = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateArtifactError(f"gate {gate} artifact is not valid UTF-8 JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise GateArtifactError(f"gate {gate} artifact root must be a JSON mapping")
    return payload, hashlib.sha256(raw_bytes).hexdigest()


def _record_cleanup_error(primary: BaseException, cleanup: BaseException) -> None:
    try:
        existing = getattr(primary, "cleanup_errors", ())
        errors = existing if isinstance(existing, tuple) else ()
        setattr(primary, "cleanup_errors", (*errors, cleanup))
    except BaseException:
        pass


def _remove_owned_staged_link(path: Path, staging: Path, expected: bytes) -> bool:
    """Remove a partial output only while it is still our unchanged hard link."""

    try:
        if path.is_symlink():
            return False
        path_before = path.stat(follow_symlinks=False)
        staging_stat = staging.stat(follow_symlinks=False)
    except FileNotFoundError:
        return True
    expected_identity = (staging_stat.st_dev, staging_stat.st_ino)
    if (path_before.st_dev, path_before.st_ino) != expected_identity:
        return False
    try:
        content = path.read_bytes()
        path_after = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return True
    if (
        (path_after.st_dev, path_after.st_ino) != expected_identity
        or content != expected
    ):
        return False
    path.unlink()
    return True


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _learning_group(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    direct = payload.get("members")
    if isinstance(direct, list):
        return payload
    typed_group = _mapping(payload.get("learning_group"))
    if typed_group is not None and isinstance(typed_group.get("members"), list):
        return typed_group
    assembly = _mapping(payload.get("assembly"))
    group = _mapping(assembly.get("group")) if assembly else None
    return group if group is not None and isinstance(group.get("members"), list) else None


def _members(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    group = _learning_group(payload)
    members = group.get("members") if group is not None else None
    return [item for item in members or [] if isinstance(item, Mapping)]


def _learning_group_identity(
    group: Mapping[str, Any], members: list[Mapping[str, Any]]
) -> tuple[str, str]:
    group_uid = group.get("group_uid")
    if isinstance(group_uid, str) and group_uid:
        context = json.dumps(
            [group.get("task_id"), group.get("environment_seed"), group_uid],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return "uid", context
    canonical = json.dumps(
        {
            "task_id": group.get("task_id"),
            "environment_seed": group.get("environment_seed"),
            "initial_state_sha256": group.get("initial_state_sha256"),
            "members": members,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "content", canonical


def _repair_attempts(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    direct = payload.get("repair_attempts")
    if isinstance(direct, list):
        return [item for item in direct if isinstance(item, Mapping)]
    assembly = _mapping(payload.get("assembly"))
    nested = assembly.get("repair_attempts") if assembly else None
    return [item for item in nested or [] if isinstance(item, Mapping)]


def _attempt_succeeded(
    attempt: Mapping[str, Any], flat_field: str, result_field: str
) -> bool:
    if attempt.get(flat_field) == "success":
        return True
    result = _mapping(attempt.get(result_field))
    return result is not None and result.get("outcome") == "success"


def analyze_directory(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"artifact directory does not exist: {root}")
    summary: dict[str, Any] = {
        "artifact_files": 0,
        "learning_groups": 0,
        "base_members": 0,
        "base_successes": 0,
        "guided_members": 0,
        "guided_successes": 0,
        "repair_attempts": 0,
        "pt_successes": 0,
        "p_hat_successes": 0,
        "retry_count": 0,
        "infra_failures": 0,
        "optimizer_steps": 0,
    }
    seen_learning_groups: set[tuple[str, str]] = set()
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        summary["artifact_files"] += 1
        members = _members(payload)
        group = _learning_group(payload)
        unseen_group = False
        if members and group is not None:
            group_identity = _learning_group_identity(group, members)
            unseen_group = group_identity not in seen_learning_groups
            seen_learning_groups.add(group_identity)
        if unseen_group:
            summary["learning_groups"] += 1
        for member in members if unseen_group else ():
            member_type = member.get("member_type")
            reward = member.get("reward")
            if member_type == "base":
                summary["base_members"] += 1
                summary["base_successes"] += int(reward == 1 or reward == 1.0)
            elif member_type == "critique_guided_revision":
                summary["guided_members"] += 1
                summary["guided_successes"] += int(reward == 1 or reward == 1.0)
        attempts = _repair_attempts(payload)
        summary["repair_attempts"] += len(attempts)
        summary["pt_successes"] += sum(
            _attempt_succeeded(attempt, "pt_outcome", "pt_result") for attempt in attempts
        )
        summary["p_hat_successes"] += sum(
            _attempt_succeeded(attempt, "p_hat_outcome", "revision_result")
            for attempt in attempts
        )
        retry_count = payload.get("retry_count", 0)
        infra_failures = payload.get("infra_failures", 0)
        optimizer_steps = payload.get("optimizer_steps", 0)
        if isinstance(retry_count, int) and not isinstance(retry_count, bool):
            summary["retry_count"] += retry_count
        if isinstance(infra_failures, int) and not isinstance(infra_failures, bool):
            summary["infra_failures"] += infra_failures
        if isinstance(optimizer_steps, int) and not isinstance(optimizer_steps, bool):
            summary["optimizer_steps"] += optimizer_steps
    return summary


def audit_gate_directory(directory: str | Path) -> dict[str, Any]:
    """Verify the complete Gate 1--6 evidence chain and return an auditable manifest."""

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"artifact directory does not exist: {root}")
    artifacts: list[tuple[str, Path, Mapping[str, Any], str]] = []
    for gate, filename in REQUIRED_GATE_FILES.items():
        path = root / filename
        if path.is_symlink():
            raise GateArtifactError(
                f"gate {gate} artifact must not be a symlink: {path}"
            )
        if not path.is_file():
            raise GateArtifactError(f"missing gate artifact for {gate}: {path}")
        failure_path = gate_failure_artifact_path(path)
        if failure_path.exists() or failure_path.is_symlink():
            raise GateArtifactError(
                f"gate {gate} has both success and failure evidence: {path}, {failure_path}"
            )
        payload, snapshot_sha256 = _load_gate_artifact_snapshot(path, gate)
        GATE_VERIFIERS[gate](payload)
        if payload.get("execution_mode") != CANONICAL_EXECUTION_MODE:
            raise GateArtifactError(
                f"gate {gate} is noncanonical; runtime verification only accepts the "
                "repository server adapter"
            )
        artifacts.append((gate, path, payload, snapshot_sha256))

    identity_fields = ("run_id", "config_sha256", "git_sha")
    identity = tuple(artifacts[0][2][field] for field in identity_fields)
    for gate, _path, payload, _sha256 in artifacts[1:]:
        current = tuple(payload[field] for field in identity_fields)
        if current != identity:
            raise GateArtifactError(
                f"gate {gate} does not share run_id, config SHA, and Git SHA with preflight"
            )

    dataset_sha256 = artifacts[0][2]["dataset_sha256"]
    for gate, _path, payload, _sha256 in artifacts[1:]:
        if payload["dataset_sha256"] != dataset_sha256:
            raise GateArtifactError(
                f"gate {gate} does not share the preflight dataset SHA-256"
            )

    dependency_labels = {
        "resolved_environment_sha256": "resolved environment SHA",
        "verl_resolved_config_sha256": "resolved VeRL config SHA",
    }
    runtime_dependency_sha256s = {
        field_name: artifacts[0][2][field_name]
        for field_name in dependency_labels
    }
    for gate, _path, payload, _sha256 in artifacts[1:]:
        for field_name, label in dependency_labels.items():
            if payload[field_name] != runtime_dependency_sha256s[field_name]:
                raise GateArtifactError(
                    f"gate {gate} does not share the preflight {label}-256"
                )

    payload_by_gate = {
        gate: payload for gate, _path, payload, _sha256 in artifacts
    }
    guided_group = payload_by_gate["guided"]["learning_group"]
    trainer_group = payload_by_gate["trainer"]["learning_group"]
    if trainer_group != guided_group:
        raise GateArtifactError("trainer gate must consume the exact verified guided group")
    guided_sha256 = next(
        sha256
        for gate, _path, _payload, sha256 in artifacts
        if gate == "guided"
    )
    if (
        payload_by_gate["trainer"].get("guided_artifact_sha256")
        != guided_sha256
    ):
        raise GateArtifactError(
            "trainer gate guided_artifact_sha256 does not match the persisted Gate 5 artifact"
        )
    preflight_verl = payload_by_gate["preflight"]["checks"]
    trainer_verl = payload_by_gate["trainer"]["verl_provenance_after"]
    if (
        trainer_verl.get("source_path") != preflight_verl.get("verl_source_path")
        or trainer_verl.get("expected_sha") != preflight_verl.get("verl_expected_sha")
        or trainer_verl.get("actual_sha") != preflight_verl.get("verl_actual_sha")
        or trainer_verl.get("clean") is not True
    ):
        raise GateArtifactError(
            "trainer VeRL provenance does not match the verified preflight checkout"
        )

    oracle_result = payload_by_gate["oracle_replay"]["replays"][0]["result"]
    collector_result = payload_by_gate["collector"]["base_results"][0]
    guided_task = payload_by_gate["guided"]["task_instance"]
    typed_seed5_identities = {
        (
            evidence["task_id"],
            evidence["environment_seed"],
            evidence["initial_state_sha256"],
        )
        for evidence in (oracle_result, collector_result, guided_task)
    }
    if (
        len(typed_seed5_identities) != 1
        or next(iter(typed_seed5_identities))[1] != 5
    ):
        raise GateArtifactError(
            "oracle, collector, and guided gates must share one typed seed-5 task identity"
        )
    seed5_task_id, seed5_environment_seed, _seed5_initial_state = next(
        iter(typed_seed5_identities)
    )
    preflight_dataset_identities = {
        (record["task_id"], record["environment_seed"])
        for record in payload_by_gate["preflight"]["checks"][
            "dataset_task_identities"
        ]
    }
    if (seed5_task_id, seed5_environment_seed) not in preflight_dataset_identities:
        raise GateArtifactError(
            "typed seed-5 task identity does not exist in the preflight dataset"
        )

    seed5_state = payload_by_gate["seed"]["initial_state_sha256"][0]
    replay_states = [
        *[
            record["result"]["initial_state_sha256"]
            for record in payload_by_gate["oracle_replay"]["replays"]
        ],
        *[
            result["initial_state_sha256"]
            for result in payload_by_gate["collector"]["base_results"]
        ],
        payload_by_gate["guided"]["learning_group"]["initial_state_sha256"],
    ]
    if any(state != seed5_state for state in replay_states):
        raise GateArtifactError(
            "seed-5 initial state must match oracle, collector, and guided gates"
        )

    summary = analyze_directory(root)
    for gate, path, _payload, snapshot_sha256 in artifacts:
        _current_payload, current_sha256 = _load_gate_artifact_snapshot(path, gate)
        if current_sha256 != snapshot_sha256:
            raise GateArtifactError(
                f"gate {gate} artifact changed during Gate 7 audit: {path}"
            )

    chain: list[dict[str, Any]] = []
    previous_sha256: str | None = None
    for gate, path, _payload, sha256 in artifacts:
        chain.append(
            {
                "gate": gate,
                "path": str(path),
                "sha256": sha256,
                "previous_sha256": previous_sha256,
            }
        )
        previous_sha256 = sha256

    trainer_payload = artifacts[-1][2]
    summary.update(
        {
            "runtime_verified": True,
            "run_id": identity[0],
            "config_sha256": identity[1],
            "git_sha": identity[2],
            "dataset_sha256": dataset_sha256,
            **runtime_dependency_sha256s,
            "typed_task_identities": [
                {
                    "task_id": seed5_task_id,
                    "environment_seed": seed5_environment_seed,
                    "initial_state_sha256": seed5_state,
                }
            ],
            "gate_statuses": {gate: "passed" for gate in REQUIRED_GATE_FILES},
            "gate_chain": chain,
            "gradient_norm": trainer_payload["gradient_norm"],
            "trainer_metrics": dict(trainer_payload["metrics"]),
        }
    )
    return summary


def _markdown(summary: Mapping[str, Any]) -> str:
    rows = ["# Capsule-RL artifact audit", "", "| Metric | Value |", "|---|---:|"]
    rows.extend(f"| `{key}` | {value} |" for key, value in summary.items())
    rows.extend(
        [
            "",
            "This report audits persisted evidence only; it does not claim simulator or training ",
            "runtime verification unless gates 1 through 6 all passed.",
            "",
        ]
    )
    return "\n".join(rows)


def _publish_audit_outputs(
    output_json: Path,
    output_report: Path,
    summary: Mapping[str, Any],
) -> None:
    """Publish the JSON and Markdown pair together, rolling back a partial pair."""

    json_path = output_json.expanduser().resolve()
    report_path = output_report.expanduser().resolve()
    if json_path == report_path:
        raise ValueError("audit JSON and Markdown outputs must be different paths")
    for path in (json_path, report_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"artifact already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    transaction_id = uuid.uuid4().hex
    json_staging = json_path.with_name(f".{json_path.name}.{transaction_id}.tmp")
    report_staging = report_path.with_name(f".{report_path.name}.{transaction_id}.tmp")
    staged_payloads = (
        (
            json_staging,
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
        ),
        (report_staging, _markdown(summary)),
    )
    published: list[tuple[Path, Path, bytes]] = []
    primary_error: BaseException | None = None
    try:
        for staging, content in staged_payloads:
            with staging.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        os.link(json_staging, json_path)
        published.append((json_path, json_staging, staged_payloads[0][1].encode("utf-8")))
        os.link(report_staging, report_path)
        published.append(
            (report_path, report_staging, staged_payloads[1][1].encode("utf-8"))
        )
    except BaseException as error:
        primary_error = error
        for path, staging, expected in reversed(published):
            try:
                _remove_owned_staged_link(path, staging, expected)
            except BaseException as cleanup_error:
                _record_cleanup_error(error, cleanup_error)
        raise
    finally:
        for staging in (json_staging, report_staging):
            try:
                staging.unlink(missing_ok=True)
            except BaseException as cleanup_error:
                if primary_error is not None:
                    _record_cleanup_error(primary_error, cleanup_error)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Capsule-RL server artifacts.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    add_validation_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_only = validation_requested(args)
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"artifact directory does not exist: {args.input_dir.resolve()}")
    for output in (args.output_json, args.output_report):
        if output.exists():
            raise FileExistsError(f"artifact already exists: {output.resolve()}")
    summary = audit_gate_directory(args.input_dir)
    plan = {
        "mode": "VALIDATION ONLY" if validate_only else "ANALYZE",
        "input": str(args.input_dir.resolve()),
        "json": str(args.output_json.resolve()),
        "report": str(args.output_report.resolve()),
        "verified_run_id": summary["run_id"],
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if validate_only:
        return 0
    _publish_audit_outputs(args.output_json, args.output_report, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
