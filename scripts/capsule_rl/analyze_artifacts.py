"""Gate 7 artifact audit; preview outputs with --validate-only or --dry-run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import uuid
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import yaml

from capx.rl.capsule.stable_io import (
    MutationWatch,
    PathMutationGuard,
    StablePathError,
    read_stable_regular_file,
    sys_platform_linux,
)

from .common import (
    CANONICAL_EXECUTION_MODE,
    GateArtifactError,
    add_validation_arguments,
    gate_failure_artifact_path,
    validation_requested,
    verify_adapter_reload_artifact,
    verify_collector_gate_artifact,
    verify_guided_gate_artifact,
    verify_oracle_gate_artifact,
    verify_preflight_gate_artifact,
    verify_seed_gate_artifact,
    verify_trainer_gate_artifact,
)
from .llama_attestation import (
    LlamaRuntimeAttestationError,
    attest_llama_cpp_runtime,
)

REQUIRED_GATE_FILES = {
    "preflight": "gate01_preflight.json",
    "seed": "gate02_seed.json",
    "oracle_replay": "gate03_oracle.json",
    "collector": "gate04_collector.json",
    "guided": "gate05_guided_group.json",
    "trainer": "gate06_trainer.json",
    "adapter_reload": "adapter_reload_smoke.json",
}
RUNTIME_VERIFICATION_PENDING = (
    "launcher_continuous_memory",
    "owned_service_cleanup",
    "controller_runtime_attestation",
)
_AUDIT_META_FILENAMES = {
    "gate07_audit.candidate.json",
    "gate07_audit.json",
    "launcher_continuous_memory.json",
    "launcher_controller_attestation.json",
    "launcher_owned_cleanup.json",
    "launcher_initial_audit.json",
    "launcher_memory_00_post-controller.json",
}
_OOM_PROFILES = {
    "base_dynamic_fp32",
    "vllm_util_026",
    "fixed_microbatch_1",
    "fsdp_base_bf16",
    "fsdp_base_bf16_vllm_util_045",
}
_LLAMA_ARCHIVE_SHA256 = (
    "f263a91280471b4c33c4999d7c76259c0f3a0a53a0b3e692b2c0b84380137a35"
)
_CONTROLLER_GGUF_SHA256 = (
    "509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c"
)
_EXTERNAL_CONTROLLER_ATTESTATION_TYPE = "external_controller_runtime_attestation"
_EXTERNAL_CONTROLLER_ENDPOINT = "https://coding.dashscope.aliyuncs.com/v1"
_EXTERNAL_CONTROLLER_MODEL = "qwen3.7-plus"
GATE_VERIFIERS = {
    "preflight": verify_preflight_gate_artifact,
    "seed": verify_seed_gate_artifact,
    "oracle_replay": verify_oracle_gate_artifact,
    "collector": verify_collector_gate_artifact,
    "guided": verify_guided_gate_artifact,
    "trainer": verify_trainer_gate_artifact,
    "adapter_reload": verify_adapter_reload_artifact,
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
        raise GateArtifactError(
            f"gate {gate} artifact is not valid UTF-8 JSON: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise GateArtifactError(f"gate {gate} artifact root must be a JSON mapping")
    return payload, hashlib.sha256(raw_bytes).hexdigest()


def _gate7_mutation_context(
    root: Path, *, finalizing: bool
) -> PathMutationGuard | Any:
    if not sys_platform_linux():
        if finalizing:
            raise GateArtifactError(
                "final runtime verification requires Linux inotify mutation guards"
            )
        return nullcontext()
    gate_paths = [root / filename for filename in REQUIRED_GATE_FILES.values()]
    trainer_payload, trainer_sha256 = _load_gate_artifact_snapshot(
        root / REQUIRED_GATE_FILES["trainer"], "trainer_guard_setup"
    )
    checkpoint_value = trainer_payload.get("checkpoint")
    checkpoint = (
        Path(checkpoint_value).expanduser()
        if isinstance(checkpoint_value, str) and checkpoint_value
        else None
    )
    if checkpoint is None or not checkpoint.is_absolute():
        raise GateArtifactError("trainer checkpoint path is invalid during guard setup")
    watched_paths = list(gate_paths)
    if finalizing:
        watched_paths.extend(
            root / filename
            for filename in (
                "gate07_audit.candidate.json",
                "launcher_continuous_memory.json",
                "launcher_controller_attestation.json",
                "launcher_owned_cleanup.json",
                "launcher_initial_audit.json",
                "launcher_memory_00_post-controller.json",
            )
        )
        watched_paths.append(root / "resolved" / "verl.yaml")
    watches = [
        MutationWatch(path=path, label=f"Gate 7 evidence {path.name}")
        for path in watched_paths
    ]
    watches.append(
        MutationWatch(
            path=checkpoint,
            label="Gate 6 checkpoint tree",
            recursive=True,
        )
    )
    try:
        guard = PathMutationGuard.open(watches)
        try:
            _current, current_sha256 = _load_gate_artifact_snapshot(
                root / REQUIRED_GATE_FILES["trainer"], "trainer_guard_recheck"
            )
            if current_sha256 != trainer_sha256:
                raise GateArtifactError(
                    "trainer artifact changed while Gate 7 mutation guards were installed"
                )
        except BaseException:
            guard.close()
            raise
        return guard
    except StablePathError as error:
        raise GateArtifactError(f"cannot guard Gate 7 evidence: {error}") from error


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
        if path.name in _AUDIT_META_FILENAMES:
            continue
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
    """Verify Gate 1--6 plus the independent adapter reload evidence chain."""

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
        "program_model_sha256": "Program model SHA",
        "actor_binding_sha256": "actor binding SHA",
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
        current_payload, current_sha256 = _load_gate_artifact_snapshot(path, gate)
        if current_sha256 != snapshot_sha256:
            raise GateArtifactError(
                f"gate {gate} artifact changed during Gate 7 audit: {path}"
            )
        GATE_VERIFIERS[gate](current_payload)

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

    trainer_payload = payload_by_gate["trainer"]
    reload_payload = payload_by_gate["adapter_reload"]
    summary.update(
        {
            "schema_version": 1,
            "artifact_type": "capsule_rl_gate07_candidate_audit",
            "runtime_verified": False,
            "runtime_verification_pending": list(RUNTIME_VERIFICATION_PENDING),
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
            "adapter_reload_artifact_sha256": next(
                sha256
                for gate, _path, _payload, sha256 in artifacts
                if gate == "adapter_reload"
            ),
            "adapter_reload_max_abs_logit_diff": reload_payload[
                "max_abs_logit_diff"
            ],
            "adapter_model_sha256": reload_payload["adapter_model_sha256"],
            "adapter_config_sha256": reload_payload["adapter_config_sha256"],
        }
    )
    return summary


def _verify_continuous_memory_artifact(
    path: Path,
) -> tuple[Mapping[str, Any], str]:
    payload, sha256 = _load_gate_artifact_snapshot(path, "launcher_continuous_memory")
    expected_fields = {
        "schema_version",
        "artifact_type",
        "interval_s",
        "required_mib",
        "sample_count",
        "minimum_available_mib",
        "maximum_sample_gap_ms",
        "maximum_allowed_gap_ms",
        "passed",
        "probe_error",
        "samples",
    }
    if set(payload) != expected_fields:
        raise GateArtifactError("continuous memory artifact schema is incomplete or unexpected")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_type") != "single_a800_continuous_memory"
        or payload.get("required_mib") != 12288
        or payload.get("passed") is not True
        or payload.get("probe_error") is not None
    ):
        raise GateArtifactError("continuous memory artifact did not pass the 12 GiB contract")
    interval_s = payload.get("interval_s")
    if (
        isinstance(interval_s, bool)
        or not isinstance(interval_s, (int, float))
        or not math.isfinite(float(interval_s))
        or interval_s <= 0
    ):
        raise GateArtifactError("continuous memory interval must be positive and finite")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise GateArtifactError("continuous memory artifact requires at least two samples")
    normalized: list[tuple[int, int]] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise GateArtifactError("continuous memory sample must be a mapping")
        elapsed_ms = sample.get("elapsed_ms")
        available_mib = sample.get("available_mib")
        if (
            isinstance(elapsed_ms, bool)
            or not isinstance(elapsed_ms, int)
            or elapsed_ms < 0
            or isinstance(available_mib, bool)
            or not isinstance(available_mib, int)
            or available_mib < 12288
        ):
            raise GateArtifactError("continuous memory sample violates the 12 GiB contract")
        normalized.append((elapsed_ms, available_mib))
    if any(later[0] < earlier[0] for earlier, later in zip(normalized, normalized[1:])):
        raise GateArtifactError("continuous memory sample times must be monotonic")
    minimum = min(available for _elapsed, available in normalized)
    maximum_gap = max(
        (
            later[0] - earlier[0]
            for earlier, later in zip(normalized, normalized[1:])
        ),
        default=0,
    )
    maximum_allowed_gap = int(max(5.0, float(interval_s) * 5) * 1000)
    if (
        payload.get("sample_count") != len(normalized)
        or payload.get("minimum_available_mib") != minimum
        or payload.get("maximum_sample_gap_ms") != maximum_gap
        or payload.get("maximum_allowed_gap_ms") != maximum_allowed_gap
        or maximum_gap > maximum_allowed_gap
    ):
        raise GateArtifactError("continuous memory aggregate evidence is inconsistent")
    return payload, sha256


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_controller_attestation_artifact(
    path: Path,
) -> tuple[Mapping[str, Any], str]:
    payload, sha256 = _load_gate_artifact_snapshot(
        path, "launcher_controller_attestation"
    )
    if payload.get("artifact_type") == _EXTERNAL_CONTROLLER_ATTESTATION_TYPE:
        config_fields = {
            "mode",
            "endpoint",
            "model",
            "api_key_env",
            "request_timeout_s",
            "max_output_tokens",
            "stream",
            "enable_thinking",
            "temperature",
        }
        expected_fields = {
            "schema_version",
            "artifact_type",
            "ownership",
            *config_fields,
            "credential_present",
            "controller_binding_sha256",
        }
        if set(payload) != expected_fields:
            raise GateArtifactError(
                "external Controller attestation schema is incomplete or unexpected"
            )
        expected_config = {
            "mode": "external",
            "endpoint": _EXTERNAL_CONTROLLER_ENDPOINT,
            "model": _EXTERNAL_CONTROLLER_MODEL,
            "api_key_env": "CAPX_CONTROLLER_API_KEY",
            "request_timeout_s": 300.0,
            "max_output_tokens": 4096,
            "stream": False,
            "enable_thinking": False,
            "temperature": 0.7,
        }
        actual_config = {field_name: payload.get(field_name) for field_name in config_fields}
        encoded = json.dumps(
            actual_config,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        binding_sha256 = hashlib.sha256(encoded).hexdigest()
        if (
            payload.get("schema_version") != 1
            or payload.get("ownership") != "external"
            or payload.get("credential_present") is not True
            or actual_config != expected_config
            or payload.get("controller_binding_sha256") != binding_sha256
        ):
            raise GateArtifactError(
                "external Controller attestation does not bind the fixed request contract"
            )
        return payload, sha256

    expected_fields = {
        "schema_version",
        "artifact_type",
        "version_tag",
        "archive_path",
        "archive_sha256",
        "binary_path",
        "binary_archive_member",
        "binary_sha256",
        "gguf_path",
        "gguf_sha256",
        "build_number",
        "runtime_tree_sha256",
        "regular_file_count",
        "symlink_count",
    }
    if set(payload) != expected_fields:
        raise GateArtifactError(
            "Controller runtime attestation schema is incomplete or unexpected"
        )
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_type")
        != "llama_cpp_b10516_runtime_attestation"
        or payload.get("version_tag") != "b10516"
        or payload.get("build_number") != 10516
        or payload.get("archive_sha256") != _LLAMA_ARCHIVE_SHA256
        or payload.get("gguf_sha256") != _CONTROLLER_GGUF_SHA256
    ):
        raise GateArtifactError(
            "Controller runtime attestation does not bind the fixed b10516/Q4_K_M inputs"
        )
    for field_name in ("archive_path", "binary_path", "gguf_path"):
        value = payload.get(field_name)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise GateArtifactError(
                f"Controller runtime attestation {field_name} must be absolute"
            )
    member = payload.get("binary_archive_member")
    if not isinstance(member, str) or not member or Path(member).name != "llama-server":
        raise GateArtifactError(
            "Controller runtime attestation must identify the archive llama-server"
        )
    for field_name in ("binary_sha256", "runtime_tree_sha256"):
        if not _is_lower_sha256(payload.get(field_name)):
            raise GateArtifactError(
                f"Controller runtime attestation {field_name} must be SHA-256"
            )
    regular_count = payload.get("regular_file_count")
    symlink_count = payload.get("symlink_count")
    if (
        isinstance(regular_count, bool)
        or not isinstance(regular_count, int)
        or regular_count < 1
        or isinstance(symlink_count, bool)
        or not isinstance(symlink_count, int)
        or symlink_count < 0
    ):
        raise GateArtifactError(
            "Controller runtime attestation member counts are invalid"
        )
    try:
        recomputed = attest_llama_cpp_runtime(
            archive_path=payload["archive_path"],
            expected_archive_sha256=_LLAMA_ARCHIVE_SHA256,
            binary_path=payload["binary_path"],
            gguf_path=payload["gguf_path"],
            expected_gguf_sha256=_CONTROLLER_GGUF_SHA256,
            expected_build_number=10516,
            version_tag="b10516",
        )
    except LlamaRuntimeAttestationError as error:
        raise GateArtifactError(
            f"Controller runtime changed before final Gate 7 publication: {error}"
        ) from error
    if recomputed != dict(payload):
        raise GateArtifactError(
            "Controller runtime attestation changed before final Gate 7 publication"
        )
    return payload, sha256


def _verify_owned_cleanup_artifact(
    path: Path, *, expected_run_id: str
) -> tuple[Mapping[str, Any], str]:
    payload, sha256 = _load_gate_artifact_snapshot(path, "launcher_owned_cleanup")
    if set(payload) != {
        "schema_version",
        "artifact_type",
        "run_id",
        "cleanup_completed",
        "services",
    }:
        raise GateArtifactError("owned-service cleanup schema is incomplete or unexpected")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_type") != "single_a800_owned_service_cleanup"
        or payload.get("run_id") != expected_run_id
        or payload.get("cleanup_completed") is not True
    ):
        raise GateArtifactError("owned-service cleanup did not complete for this run")
    services = payload.get("services")
    if not isinstance(services, list) or len(services) != 3:
        raise GateArtifactError("owned-service cleanup must bind exactly three services")
    expected_names = {"controller", "program", "pyroki"}
    seen_names: set[str] = set()
    seen_processes: set[tuple[int, int]] = set()
    ownership_by_name: dict[str, str] = {}
    for service in services:
        if not isinstance(service, Mapping):
            raise GateArtifactError("owned-service cleanup entry schema is invalid")
        name = service.get("name")
        ownership = service.get("ownership")
        if name == "controller" and ownership == "external":
            if set(service) != {
                "name",
                "ownership",
                "termination_confirmed",
            } or service.get("termination_confirmed") is not None:
                raise GateArtifactError("external Controller cleanup entry is invalid")
            if name in seen_names:
                raise GateArtifactError(
                    "owned-service cleanup entry is invalid or duplicated"
                )
            seen_names.add(name)
            ownership_by_name[name] = ownership
            continue
        if set(service) != {
            "name",
            "ownership",
            "pid",
            "starttime_ticks",
            "termination_confirmed",
        }:
            raise GateArtifactError("owned-service cleanup entry schema is invalid")
        pid = service.get("pid")
        starttime_ticks = service.get("starttime_ticks")
        if (
            name not in expected_names
            or name in seen_names
            or ownership != "owned"
            or isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid < 1
            or isinstance(starttime_ticks, bool)
            or not isinstance(starttime_ticks, int)
            or starttime_ticks < 1
            or service.get("termination_confirmed") is not True
            or (pid, starttime_ticks) in seen_processes
        ):
            raise GateArtifactError("owned-service cleanup entry is invalid or duplicated")
        proc_stat = Path(f"/proc/{pid}/stat")
        if Path("/proc").is_dir():
            try:
                stat_text = proc_stat.read_text(encoding="ascii")
            except FileNotFoundError:
                pass
            except OSError as error:
                raise GateArtifactError(
                    f"cannot verify cleaned owned process {pid}: {error}"
                ) from error
            else:
                closing = stat_text.rfind(")")
                if closing < 0:
                    raise GateArtifactError(
                        f"cannot parse cleaned owned process identity for PID {pid}"
                    )
                try:
                    current_starttime = int(stat_text[closing + 2 :].split()[19])
                except (IndexError, ValueError):
                    raise GateArtifactError(
                        f"cannot parse cleaned owned process identity for PID {pid}"
                    ) from None
                if current_starttime == starttime_ticks:
                    raise GateArtifactError(
                        f"owned service {name} is still running at final Gate 7"
                    )
        seen_names.add(name)
        ownership_by_name[str(name)] = str(ownership)
        seen_processes.add((pid, starttime_ticks))
    if seen_names != expected_names:
        raise GateArtifactError("owned-service cleanup is missing a required service")
    if ownership_by_name.get("program") != "owned" or ownership_by_name.get("pyroki") != "owned":
        raise GateArtifactError("Program and PyRoKi cleanup entries must be owned")
    return payload, sha256


def _non_negative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _verify_initial_audit_artifact(
    path: Path, *, expected_run_id: str, expected_git_sha: str
) -> tuple[Mapping[str, Any], str]:
    payload, sha256 = _load_gate_artifact_snapshot(path, "launcher_initial_audit")
    if set(payload) != {
        "schema_version",
        "artifact_type",
        "run_id",
        "retry_name",
        "profile_sha256",
        "snapshot",
    }:
        raise GateArtifactError("initial hardware audit schema is incomplete or unexpected")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_type") != "single_a800_initial_audit"
        or payload.get("run_id") != expected_run_id
        or payload.get("retry_name") not in _OOM_PROFILES
        or not _is_lower_sha256(payload.get("profile_sha256"))
    ):
        raise GateArtifactError("initial hardware audit identity is invalid")
    snapshot = payload.get("snapshot")
    expected_snapshot_fields = {
        "gpu_name",
        "gpu_count",
        "gpu_total_vram_mib",
        "gpu_free_vram_mib",
        "other_gpu_processes_mib",
        "host_memory_mib",
        "mem_available_before_controller_mib",
        "mem_available_after_controller_mib",
        "mem_available_during_run_mib",
        "shm_available_mib",
        "disk_free_mib",
        "cuda_version",
        "nvidia_driver",
        "repo_head",
        "repo_is_dirty",
        "system_version",
    }
    if not isinstance(snapshot, Mapping) or set(snapshot) != expected_snapshot_fields:
        raise GateArtifactError("initial hardware audit snapshot schema is invalid")
    gpu_name = snapshot.get("gpu_name")
    normalized_gpu_name = (
        gpu_name.upper().replace("-", "").replace(" ", "")
        if isinstance(gpu_name, str)
        else ""
    )
    threshold_fields = {
        "gpu_count": 1,
        "gpu_total_vram_mib": 81920,
        "gpu_free_vram_mib": 77824,
        "host_memory_mib": 122880,
        "shm_available_mib": 12288,
        "disk_free_mib": 81920,
    }
    for field_name, minimum in threshold_fields.items():
        value = snapshot.get(field_name)
        if not _non_negative_int(value):
            raise GateArtifactError(f"initial hardware audit {field_name} is invalid")
        if field_name == "gpu_count":
            if value != minimum:
                raise GateArtifactError("initial hardware audit requires exactly one GPU")
        elif value < minimum:
            raise GateArtifactError(
                f"initial hardware audit {field_name} is below the fixed threshold"
            )
    if "A800" not in normalized_gpu_name or "80GB" not in normalized_gpu_name:
        raise GateArtifactError("initial hardware audit GPU must be an A800 80GB")
    other_processes = snapshot.get("other_gpu_processes_mib")
    if (
        not isinstance(other_processes, list)
        or any(
            not _non_negative_int(value) or value > 512
            for value in other_processes
        )
    ):
        raise GateArtifactError("initial hardware audit has an oversized GPU process")
    for field_name in (
        "mem_available_before_controller_mib",
        "mem_available_after_controller_mib",
        "mem_available_during_run_mib",
    ):
        if not _non_negative_int(snapshot.get(field_name)):
            raise GateArtifactError(f"initial hardware audit {field_name} is invalid")
    for field_name in ("cuda_version", "nvidia_driver", "system_version"):
        value = snapshot.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise GateArtifactError(f"initial hardware audit {field_name} is empty")
    if snapshot.get("repo_head") != expected_git_sha:
        raise GateArtifactError("initial hardware audit Git SHA does not match Gate 1")
    if not isinstance(snapshot.get("repo_is_dirty"), bool):
        raise GateArtifactError("initial hardware audit repo_is_dirty must be boolean")
    return payload, sha256


def _verify_post_controller_memory_artifact(
    path: Path,
) -> tuple[Mapping[str, Any], str]:
    payload, sha256 = _load_gate_artifact_snapshot(
        path, "launcher_memory_00_post-controller"
    )
    if set(payload) != {
        "schema_version",
        "artifact_type",
        "stage",
        "available_mib",
        "required_mib",
        "passed",
    }:
        raise GateArtifactError("post-Controller memory artifact schema is invalid")
    available = payload.get("available_mib")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_type") != "single_a800_memory_check"
        or payload.get("stage") != "post-controller"
        or payload.get("required_mib") != 92160
        or payload.get("passed") is not True
        or not _non_negative_int(available)
        or available < 92160
    ):
        raise GateArtifactError("post-Controller MemAvailable did not satisfy 90 GiB")
    return payload, sha256


def _verify_resolved_verl_profile(
    path: Path,
    *,
    initial_profile_sha256: object,
    candidate_profile_sha256: object,
    retry_name: object,
) -> str:
    try:
        snapshot = read_stable_regular_file(path, label="resolved VeRL profile")
    except StablePathError as error:
        raise GateArtifactError(f"cannot read resolved VeRL profile: {error}") from error
    if snapshot.sha256 != initial_profile_sha256:
        raise GateArtifactError(
            "resolved VeRL profile SHA does not match the initial launcher audit"
        )
    if snapshot.sha256 != candidate_profile_sha256:
        raise GateArtifactError(
            "resolved VeRL profile SHA does not match the Gate 7 candidate"
        )
    try:
        payload = yaml.safe_load(snapshot.raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise GateArtifactError(f"resolved VeRL profile is invalid YAML: {error}") from error
    capsule_runtime = payload.get("capsule_runtime") if isinstance(payload, Mapping) else None
    oom_profile = (
        capsule_runtime.get("oom_profile")
        if isinstance(capsule_runtime, Mapping)
        else None
    )
    if oom_profile != retry_name:
        raise GateArtifactError(
            "resolved VeRL capsule_runtime.oom_profile does not match initial audit retry_name"
        )
    return snapshot.sha256


def finalize_runtime_audit(
    directory: str | Path,
    *,
    candidate_artifact: str | Path,
    continuous_memory_artifact: str | Path,
    controller_attestation_artifact: str | Path,
    owned_cleanup_artifact: str | Path,
) -> dict[str, Any]:
    """Bind the monitored Gate 7 candidate and successful owned-service cleanup proof."""

    root = Path(directory).expanduser().resolve()
    candidate_path = Path(candidate_artifact).expanduser().resolve()
    memory_path = Path(continuous_memory_artifact).expanduser().resolve()
    controller_path = Path(controller_attestation_artifact).expanduser().resolve()
    cleanup_path = Path(owned_cleanup_artifact).expanduser().resolve()
    initial_audit_path = root / "launcher_initial_audit.json"
    post_controller_memory_path = root / "launcher_memory_00_post-controller.json"
    if candidate_path != root / "gate07_audit.candidate.json":
        raise GateArtifactError("Gate 7 candidate must use the owned launcher path")
    if memory_path != root / "launcher_continuous_memory.json":
        raise GateArtifactError("continuous memory evidence must use the owned launcher path")
    if controller_path != root / "launcher_controller_attestation.json":
        raise GateArtifactError(
            "Controller runtime attestation must use the owned launcher path"
        )
    if cleanup_path != root / "launcher_owned_cleanup.json":
        raise GateArtifactError("owned-service cleanup must use the owned launcher path")
    launcher_failure = root / "launcher_failure.json"
    if launcher_failure.exists() or launcher_failure.is_symlink():
        raise GateArtifactError("launcher failure evidence forbids runtime verification")

    candidate, candidate_sha256 = _load_gate_artifact_snapshot(
        candidate_path, "gate07_candidate"
    )
    if (
        candidate.get("runtime_verified") is not False
        or candidate.get("runtime_verification_pending")
        != list(RUNTIME_VERIFICATION_PENDING)
    ):
        raise GateArtifactError("Gate 7 candidate is not pending launcher finalization")
    memory, memory_sha256 = _verify_continuous_memory_artifact(memory_path)
    controller, controller_sha256 = _verify_controller_attestation_artifact(
        controller_path
    )
    cleanup, cleanup_sha256 = _verify_owned_cleanup_artifact(
        cleanup_path, expected_run_id=str(candidate.get("run_id"))
    )
    initial_audit, initial_audit_sha256 = _verify_initial_audit_artifact(
        initial_audit_path,
        expected_run_id=str(candidate.get("run_id")),
        expected_git_sha=str(candidate.get("git_sha")),
    )
    post_controller_memory, post_controller_memory_sha256 = (
        _verify_post_controller_memory_artifact(post_controller_memory_path)
    )
    resolved_profile_sha256 = _verify_resolved_verl_profile(
        root / "resolved" / "verl.yaml",
        initial_profile_sha256=initial_audit.get("profile_sha256"),
        candidate_profile_sha256=candidate.get("verl_resolved_config_sha256"),
        retry_name=initial_audit.get("retry_name"),
    )
    summary = audit_gate_directory(root)
    if resolved_profile_sha256 != summary.get("verl_resolved_config_sha256"):
        raise GateArtifactError(
            "resolved VeRL profile SHA does not match the recomputed gate summary"
        )
    expected_candidate = dict(summary)
    actual_candidate = dict(candidate)
    expected_candidate.pop("artifact_files", None)
    actual_candidate.pop("artifact_files", None)
    if actual_candidate != expected_candidate:
        raise GateArtifactError("Gate 7 candidate does not match the final gate chain")

    if controller.get("artifact_type") == _EXTERNAL_CONTROLLER_ATTESTATION_TYPE:
        controller_summary = {
            "controller_mode": "external",
            "controller_endpoint": controller["endpoint"],
            "controller_model": controller["model"],
            "controller_binding_sha256": controller["controller_binding_sha256"],
        }
    else:
        controller_summary = {
            "controller_mode": "local",
            "controller_endpoint": "http://127.0.0.1:8102/v1",
            "controller_model": "qwen2.5-coder-7b-controller-q4_k_m",
            "controller_binding_sha256": controller["runtime_tree_sha256"],
        }
    summary.pop("runtime_verification_pending", None)
    summary.update(
        {
            "artifact_type": "capsule_rl_gate07_runtime_audit",
            "runtime_verified": True,
            "gate07_candidate_sha256": candidate_sha256,
            "launcher_continuous_memory_sha256": memory_sha256,
            "launcher_controller_attestation_sha256": controller_sha256,
            "launcher_owned_cleanup_sha256": cleanup_sha256,
            "launcher_initial_audit_sha256": initial_audit_sha256,
            "launcher_post_controller_memory_sha256": (
                post_controller_memory_sha256
            ),
            "continuous_memory_required_mib": memory["required_mib"],
            "minimum_mem_available_mib": memory["minimum_available_mib"],
            "continuous_memory_sample_count": memory["sample_count"],
            "continuous_memory_maximum_sample_gap_ms": memory[
                "maximum_sample_gap_ms"
            ],
            **controller_summary,
            "owned_service_cleanup_completed": cleanup["cleanup_completed"],
            "owned_service_cleanup_count": len(cleanup["services"]),
            "oom_profile": initial_audit["retry_name"],
            "resolved_profile_sha256": resolved_profile_sha256,
            "initial_hardware": {
                "gpu_name": initial_audit["snapshot"]["gpu_name"],
                "gpu_count": initial_audit["snapshot"]["gpu_count"],
                "gpu_total_vram_mib": initial_audit["snapshot"][
                    "gpu_total_vram_mib"
                ],
                "gpu_free_vram_mib": initial_audit["snapshot"][
                    "gpu_free_vram_mib"
                ],
                "host_memory_mib": initial_audit["snapshot"]["host_memory_mib"],
                "shm_available_mib": initial_audit["snapshot"]["shm_available_mib"],
                "disk_free_mib": initial_audit["snapshot"]["disk_free_mib"],
                "repo_is_dirty": initial_audit["snapshot"]["repo_is_dirty"],
            },
            "mem_available_after_controller_mib": post_controller_memory[
                "available_mib"
            ],
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
            "runtime verification unless gates 1 through 6 and the adapter reload all passed.",
            "",
        ]
    )
    return "\n".join(rows)


def _publish_audit_outputs(
    output_json: Path,
    output_report: Path,
    summary: Mapping[str, Any],
    *,
    post_publish_verify: Callable[[], None] | None = None,
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
        if post_publish_verify is not None:
            post_publish_verify()
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
    parser.add_argument("--candidate-artifact", type=Path)
    parser.add_argument("--continuous-memory-artifact", type=Path)
    parser.add_argument("--controller-attestation-artifact", type=Path)
    parser.add_argument("--owned-cleanup-artifact", type=Path)
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
    finalization_values = (
        args.candidate_artifact,
        args.continuous_memory_artifact,
        args.controller_attestation_artifact,
        args.owned_cleanup_artifact,
    )
    provided = sum(value is not None for value in finalization_values)
    if provided not in {0, len(finalization_values)}:
        raise ValueError(
            "all Gate 7 finalization artifacts must be provided together"
        )
    finalizing = provided == len(finalization_values)
    root = args.input_dir.expanduser().resolve()

    def compute_summary() -> dict[str, Any]:
        if not finalizing:
            return audit_gate_directory(root)
        return finalize_runtime_audit(
            root,
            candidate_artifact=args.candidate_artifact,
            continuous_memory_artifact=args.continuous_memory_artifact,
            controller_attestation_artifact=args.controller_attestation_artifact,
            owned_cleanup_artifact=args.owned_cleanup_artifact,
        )

    with _gate7_mutation_context(root, finalizing=finalizing) as mutation_guard:
        summary = compute_summary()
        plan = {
            "mode": "VALIDATION ONLY" if validate_only else "ANALYZE",
            "input": str(root),
            "json": str(args.output_json.resolve()),
            "report": str(args.output_report.resolve()),
            "verified_run_id": summary["run_id"],
        }
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        if validate_only:
            if isinstance(mutation_guard, PathMutationGuard):
                mutation_guard.assert_unchanged(
                    context="during Gate 7 validation"
                )
            return 0

        def verify_after_publish() -> None:
            if isinstance(mutation_guard, PathMutationGuard):
                mutation_guard.assert_unchanged(
                    context="during Gate 7 publication"
                )
            if finalizing:
                recomputed = compute_summary()
                expected = dict(summary)
                actual = dict(recomputed)
                expected.pop("artifact_files", None)
                actual.pop("artifact_files", None)
                if actual != expected:
                    raise GateArtifactError(
                        "Gate 7 evidence changed during final publication"
                    )
            if isinstance(mutation_guard, PathMutationGuard):
                mutation_guard.assert_unchanged(
                    context="during Gate 7 post-publication verification"
                )

        _publish_audit_outputs(
            args.output_json,
            args.output_report,
            summary,
            post_publish_verify=verify_after_publish,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
