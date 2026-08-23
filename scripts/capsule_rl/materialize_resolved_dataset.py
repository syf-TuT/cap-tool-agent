"""Materialize seed-resolved typed tasks; preview safely with --validate-only/--dry-run."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from capx.rl.capsule.provenance import project_path
from capx.rl.capsule.schema import TaskInstanceV1

from .common import (
    ConfigValidationError,
    add_validation_arguments,
    artifact_file_sha256,
    load_and_validate_server_config,
    runtime_dataset_path,
    runtime_dependency_hashes,
    validation_requested,
)


TaskResolver = Callable[[Mapping[str, Any]], Sequence[TaskInstanceV1]]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INITIAL_STATE_PLACEHOLDER = "0" * 64


@dataclass(frozen=True)
class MaterializationResult:
    record_count: int | None
    dataset_path: Path
    config_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class _Gate7Audit:
    source_path: Path
    raw_bytes: bytes
    sha256: str
    run_id: str
    config_sha256: str
    dataset_sha256: str
    resolved_environment_sha256: str
    verl_resolved_config_sha256: str
    typed_task_identities: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class _SourceTaskExpectation:
    task: TaskInstanceV1
    initial_state_may_resolve: bool


def _required_sha256(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ConfigValidationError(f"Gate7 audit {field_name} must be lowercase SHA-256")
    return value


def _load_gate7_audit(path: str | Path) -> _Gate7Audit:
    audit_path = Path(path).expanduser().resolve()
    if not audit_path.is_file():
        raise ConfigValidationError(f"Gate7 audit must be an existing JSON file: {audit_path}")
    raw_bytes = audit_path.read_bytes()
    try:
        payload = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfigValidationError(f"Gate7 audit is not valid UTF-8 JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise ConfigValidationError("Gate7 audit root must be a JSON mapping")
    if payload.get("runtime_verified") is not True:
        raise ConfigValidationError("Gate7 audit must record runtime_verified=true")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ConfigValidationError("Gate7 audit run_id must be non-empty")
    identities = payload.get("typed_task_identities")
    if not isinstance(identities, list) or not identities:
        raise ConfigValidationError("Gate7 audit typed_task_identities must be non-empty")
    normalized_identities: list[dict[str, object]] = []
    observed: set[tuple[str, int, str]] = set()
    for index, identity in enumerate(identities):
        if not isinstance(identity, Mapping):
            raise ConfigValidationError(
                f"Gate7 audit typed_task_identities[{index}] must be a mapping"
            )
        task_id = identity.get("task_id")
        environment_seed = identity.get("environment_seed")
        initial_state_sha256 = identity.get("initial_state_sha256")
        if not isinstance(task_id, str) or not task_id:
            raise ConfigValidationError(
                f"Gate7 audit typed_task_identities[{index}].task_id must be non-empty"
            )
        if (
            isinstance(environment_seed, bool)
            or not isinstance(environment_seed, int)
            or environment_seed < 0
        ):
            raise ConfigValidationError(
                f"Gate7 audit typed_task_identities[{index}].environment_seed is invalid"
            )
        if (
            not isinstance(initial_state_sha256, str)
            or _SHA256_RE.fullmatch(initial_state_sha256) is None
        ):
            raise ConfigValidationError(
                f"Gate7 audit typed_task_identities[{index}].initial_state_sha256 is invalid"
            )
        typed_identity = (task_id, environment_seed, initial_state_sha256)
        if typed_identity in observed:
            raise ConfigValidationError("Gate7 audit contains duplicate typed task identities")
        observed.add(typed_identity)
        normalized_identities.append(
            {
                "task_id": task_id,
                "environment_seed": environment_seed,
                "initial_state_sha256": initial_state_sha256,
            }
        )
    return _Gate7Audit(
        source_path=audit_path,
        raw_bytes=raw_bytes,
        sha256=artifact_file_sha256(audit_path),
        run_id=run_id,
        config_sha256=_required_sha256(payload, "config_sha256"),
        dataset_sha256=_required_sha256(payload, "dataset_sha256"),
        resolved_environment_sha256=_required_sha256(
            payload, "resolved_environment_sha256"
        ),
        verl_resolved_config_sha256=_required_sha256(
            payload, "verl_resolved_config_sha256"
        ),
        typed_task_identities=tuple(normalized_identities),
    )


def _validate_gate7_bindings(
    audit: _Gate7Audit,
    *,
    source_config_sha256: str,
    source_dataset_sha256: str,
    dependency_hashes: Mapping[str, str],
) -> None:
    if audit.config_sha256 != source_config_sha256:
        raise ConfigValidationError("Gate7 audit config SHA does not match the source config")
    if audit.dataset_sha256 != source_dataset_sha256:
        raise ConfigValidationError("Gate7 audit dataset SHA does not match the source dataset")
    if audit.resolved_environment_sha256 != dependency_hashes[
        "resolved_environment_sha256"
    ]:
        raise ConfigValidationError(
            "Gate7 audit resolved environment SHA does not match the source config"
        )
    if audit.verl_resolved_config_sha256 != dependency_hashes[
        "verl_resolved_config_sha256"
    ]:
        raise ConfigValidationError(
            "Gate7 audit resolved VeRL config SHA does not match the source config"
        )


def _source_rows(dataset_path: Path) -> list[tuple[int, Mapping[str, Any]]]:
    suffix = dataset_path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        rows: list[tuple[int, Mapping[str, Any]]] = []
        for line_number, line in enumerate(
            dataset_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ConfigValidationError(
                    f"runtime.dataset_path line {line_number} is invalid JSON: {error}"
                ) from error
            if not isinstance(payload, Mapping):
                raise ConfigValidationError(
                    f"runtime.dataset_path line {line_number} must be a JSON object"
                )
            rows.append((line_number, payload))
        return rows
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ModuleNotFoundError as error:
            raise ConfigValidationError(
                "reading a Parquet task dataset requires pandas"
            ) from error
        return [
            (index, dict(row))
            for index, row in enumerate(
                pd.read_parquet(dataset_path).to_dict("records"), start=1
            )
        ]
    raise ConfigValidationError(
        "runtime.dataset_path must be JSONL, JSON, or Parquet"
    )


def _source_task_expectations(
    config: Mapping[str, Any], dataset_path: Path
) -> tuple[_SourceTaskExpectation, ...]:
    task_config = config["task"]
    identities: set[tuple[str, int]] = set()
    expectations: list[_SourceTaskExpectation] = []
    for line_number, payload in _source_rows(dataset_path):
        data = dict(payload)
        data.setdefault("schema_version", 1)
        for field_name in ("environment", "api", "privilege"):
            expected = str(task_config[field_name])
            data.setdefault(field_name, expected)
            if data[field_name] != expected:
                raise ConfigValidationError(
                    f"runtime.dataset_path line {line_number} {field_name} does not match "
                    "task config"
                )
        # Prepared rows intentionally omit the server-produced initial-state hash.  A local
        # placeholder lets the schema validate every other field without manufacturing state
        # evidence or creating an environment during validate-only.
        source_initial_state = data.get("initial_state_sha256")
        initial_state_may_resolve = (
            "initial_state_sha256" not in data
            or source_initial_state == _INITIAL_STATE_PLACEHOLDER
        )
        data.setdefault("initial_state_sha256", _INITIAL_STATE_PLACEHOLDER)
        try:
            task = TaskInstanceV1.from_dict(data)
        except (KeyError, TypeError, ValueError) as error:
            raise ConfigValidationError(
                f"runtime.dataset_path line {line_number} is not a valid TaskInstanceV1: {error}"
            ) from error
        if task.environment_seed < 0:
            raise ConfigValidationError(
                f"runtime.dataset_path line {line_number} environment_seed must be non-negative"
            )
        identity = (task.task_id, task.environment_seed)
        if identity in identities:
            raise ConfigValidationError(
                f"runtime.dataset_path contains duplicate task identity {identity!r}"
            )
        identities.add(identity)
        expectations.append(
            _SourceTaskExpectation(
                task=task,
                initial_state_may_resolve=initial_state_may_resolve,
            )
        )
    if not expectations:
        raise ConfigValidationError("runtime.dataset_path contains no task rows")
    return tuple(expectations)


def _validate_resolved_tasks(
    tasks: Sequence[TaskInstanceV1],
    source_expectations: tuple[_SourceTaskExpectation, ...],
) -> tuple[TaskInstanceV1, ...]:
    resolved = tuple(tasks)
    if not resolved:
        raise ConfigValidationError("task state resolver returned no tasks")
    if len(resolved) != len(source_expectations):
        raise ConfigValidationError(
            "task state resolver changed the dataset row count: "
            f"expected {len(source_expectations)}, got {len(resolved)}"
        )
    identities: set[tuple[str, int]] = set()
    immutable_fields = (
        "schema_version",
        "task_id",
        "environment_seed",
        "prompt",
        "environment",
        "api",
        "privilege",
        "metadata",
    )
    for index, (task, expectation) in enumerate(zip(resolved, source_expectations)):
        if not isinstance(task, TaskInstanceV1):
            raise ConfigValidationError(
                f"task state resolver item {index} is not TaskInstanceV1"
            )
        if task.environment_seed < 0:
            raise ConfigValidationError(f"resolved task {index} has a negative environment seed")
        identity = (task.task_id, task.environment_seed)
        if identity in identities:
            raise ConfigValidationError(f"resolved tasks contain duplicate identity {identity!r}")
        identities.add(identity)
        actual_payload = task.to_dict()
        expected_payload = expectation.task.to_dict()
        for field_name in immutable_fields:
            if actual_payload[field_name] != expected_payload[field_name]:
                raise ConfigValidationError(
                    f"resolved task {index} changed immutable source field {field_name}"
                )
        if expectation.initial_state_may_resolve:
            if task.initial_state_sha256 == _INITIAL_STATE_PLACEHOLDER:
                raise ConfigValidationError(
                    f"resolved task {index} did not replace the initial-state placeholder"
                )
        elif task.initial_state_sha256 != expectation.task.initial_state_sha256:
            raise ConfigValidationError(
                f"resolved task {index} initial_state_sha256 changed immutable source value"
            )
    return resolved


def _write_synced_text(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _write_synced_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _record_cleanup_error(primary_error: BaseException, cleanup_error: BaseException) -> None:
    try:
        existing = getattr(primary_error, "cleanup_errors", ())
        errors = existing if isinstance(existing, tuple) else ()
        setattr(primary_error, "cleanup_errors", (*errors, cleanup_error))
    except BaseException:
        pass


def _remove_owned_destination(
    destination: Path, owned_identity: tuple[int, int]
) -> None:
    try:
        current = destination.stat(follow_symlinks=False)
    except OSError:
        return
    if (
        not stat.S_ISDIR(current.st_mode)
        or destination.is_symlink()
        or (current.st_dev, current.st_ino) != owned_identity
    ):
        return
    shutil.rmtree(destination)


def _publish_bundle(
    destination: Path,
    *,
    tasks: tuple[TaskInstanceV1, ...],
    config: Mapping[str, Any],
    source_config_path: Path,
    source_config_sha256: str,
    source_dataset_path: Path,
    source_dataset_sha256: str,
    dependency_hashes: Mapping[str, str],
    gate7_audit: _Gate7Audit,
) -> MaterializationResult:
    dataset_name = "capsule_rl.seed_resolved.dataset.jsonl"
    config_name = "capsule_rl.seed_resolved.yaml"
    manifest_name = "bundle_manifest.json"
    audit_name = "gate07_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
    )
    owned_destination_identity: tuple[int, int] | None = None
    primary_error: BaseException | None = None
    publication_complete = False
    try:
        final_dataset = destination / dataset_name
        final_config = destination / config_name
        final_manifest = destination / manifest_name
        final_audit = destination / audit_name
        dataset_text = "".join(f"{task.to_json()}\n" for task in tasks)
        _write_synced_text(staging / dataset_name, dataset_text)

        resolved_config = deepcopy(dict(config))
        resolved_config["runtime"] = dict(resolved_config["runtime"])
        resolved_config["runtime"]["dataset_path"] = str(final_dataset)
        resolved_config["runtime"]["bundle_manifest_path"] = str(final_manifest)
        resolved_config["runtime"]["gate7_audit_path"] = str(final_audit)
        config_text = yaml.safe_dump(resolved_config, sort_keys=False, allow_unicode=True)
        _write_synced_text(staging / config_name, config_text)
        _write_synced_bytes(staging / audit_name, gate7_audit.raw_bytes)

        output_task_identities = [
            {
                "task_id": task.task_id,
                "environment_seed": task.environment_seed,
                "initial_state_sha256": task.initial_state_sha256,
            }
            for task in tasks
        ]
        resolved_verl_config_path = project_path(
            config,
            config["runtime"]["verl_resolved_config_path"],
            "runtime.verl_resolved_config_path",
        )
        output_dataset_sha256 = artifact_file_sha256(staging / dataset_name)
        output_config_sha256 = artifact_file_sha256(staging / config_name)

        manifest = {
            "schema_version": 1,
            "artifact_type": "capsule_seed_resolved_dataset",
            "record_count": len(tasks),
            "dataset_path": str(final_dataset),
            "config_path": str(final_config),
            "dataset_sha256": output_dataset_sha256,
            "config_sha256": output_config_sha256,
            "source_config_path": str(source_config_path),
            "source_config_sha256": source_config_sha256,
            "source_dataset_path": str(source_dataset_path),
            "source_dataset_sha256": source_dataset_sha256,
            "gate7_audit_path": str(final_audit),
            "gate7_audit_sha256": gate7_audit.sha256,
            "gate7_run_id": gate7_audit.run_id,
            "gate7_typed_task_identities": list(gate7_audit.typed_task_identities),
            "output_dataset_path": str(final_dataset),
            "output_dataset_sha256": output_dataset_sha256,
            "output_config_path": str(final_config),
            "output_config_sha256": output_config_sha256,
            "output_task_identities": output_task_identities,
            "resolved_environment_sha256": dependency_hashes[
                "resolved_environment_sha256"
            ],
            "verl_resolved_config_path": str(resolved_verl_config_path),
            "verl_resolved_config_sha256": dependency_hashes[
                "verl_resolved_config_sha256"
            ],
        }
        _write_synced_text(
            staging / manifest_name,
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
        )
        _fsync_directory(staging)

        # mkdir is the exclusive publication lock. We never rename over an existing directory.
        destination.mkdir(exist_ok=False)
        destination_stat = destination.stat(follow_symlinks=False)
        owned_destination_identity = (
            destination_stat.st_dev,
            destination_stat.st_ino,
        )
        for filename in (dataset_name, config_name, audit_name, manifest_name):
            os.replace(staging / filename, destination / filename)
        _fsync_directory(destination)
        _fsync_directory(destination.parent)
        publication_complete = True
        return MaterializationResult(len(tasks), final_dataset, final_config, final_manifest)
    except BaseException as error:
        primary_error = error
        if owned_destination_identity is not None:
            try:
                _remove_owned_destination(destination, owned_destination_identity)
            except BaseException as cleanup_error:
                _record_cleanup_error(error, cleanup_error)
        raise
    finally:
        try:
            shutil.rmtree(staging)
        except FileNotFoundError:
            pass
        except BaseException as cleanup_error:
            if primary_error is None and not publication_complete:
                raise
            if primary_error is not None:
                _record_cleanup_error(primary_error, cleanup_error)


def materialize(
    *,
    config_path: str | Path,
    gate7_audit_path: str | Path,
    output_dir: str | Path,
    validate_only: bool,
    task_resolver: TaskResolver | None = None,
) -> MaterializationResult:
    config_file = Path(config_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    config = load_and_validate_server_config(config_file, check_runtime_paths=True)
    source_config_sha256 = artifact_file_sha256(config_file)
    source_dataset_path = runtime_dataset_path(config)
    source_dataset_sha256 = artifact_file_sha256(source_dataset_path)
    dependency_hashes = runtime_dependency_hashes(config)
    gate7_audit = _load_gate7_audit(gate7_audit_path)
    _validate_gate7_bindings(
        gate7_audit,
        source_config_sha256=source_config_sha256,
        source_dataset_sha256=source_dataset_sha256,
        dependency_hashes=dependency_hashes,
    )
    if destination.exists():
        raise FileExistsError(f"materialization output already exists: {destination}")
    nearest_parent = destination.parent
    while not nearest_parent.exists() and nearest_parent != nearest_parent.parent:
        nearest_parent = nearest_parent.parent
    if not nearest_parent.is_dir():
        raise ConfigValidationError(f"output directory has no existing parent: {destination}")

    source_expectations = _source_task_expectations(config, source_dataset_path)
    source_count = len(source_expectations)
    result = MaterializationResult(
        source_count,
        destination / "capsule_rl.seed_resolved.dataset.jsonl",
        destination / "capsule_rl.seed_resolved.yaml",
        destination / "bundle_manifest.json",
    )
    print(
        json.dumps(
            {
                "mode": "VALIDATION ONLY" if validate_only else "RESOLVE AND WRITE",
                "source_config": str(config_file),
                "source_dataset": str(source_dataset_path),
                "gate7_audit": str(gate7_audit.source_path),
                "gate7_run_id": gate7_audit.run_id,
                "source_records": source_count,
                "output_dataset": str(result.dataset_path),
                "output_config": str(result.config_path),
                "output_manifest": str(result.manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if validate_only:
        return result

    if task_resolver is None:
        # Delayed deliberately: validation mode must not import simulator/Ray-facing code.
        from capx.rl.capsule.server_factory import resolve_task_instances

        task_resolver = resolve_task_instances
    tasks = _validate_resolved_tasks(task_resolver(config), source_expectations)
    if artifact_file_sha256(config_file) != source_config_sha256:
        raise ConfigValidationError("source config changed during task resolution")
    if artifact_file_sha256(source_dataset_path) != source_dataset_sha256:
        raise ConfigValidationError("source dataset changed during task resolution")
    if runtime_dependency_hashes(config) != dependency_hashes:
        raise ConfigValidationError("runtime dependencies changed during task resolution")
    if artifact_file_sha256(gate7_audit.source_path) != gate7_audit.sha256:
        raise ConfigValidationError("Gate7 audit changed during task resolution")
    resolved_identities = {
        (task.task_id, task.environment_seed, task.initial_state_sha256) for task in tasks
    }
    audit_identities = {
        (
            str(identity["task_id"]),
            int(identity["environment_seed"]),
            str(identity["initial_state_sha256"]),
        )
        for identity in gate7_audit.typed_task_identities
    }
    if not audit_identities.issubset(resolved_identities):
        raise ConfigValidationError(
            "resolved tasks do not contain every Gate7 typed task identity"
        )
    return _publish_bundle(
        destination,
        tasks=tasks,
        config=config,
        source_config_path=config_file,
        source_config_sha256=source_config_sha256,
        source_dataset_path=source_dataset_path,
        source_dataset_sha256=source_dataset_sha256,
        dependency_hashes=dependency_hashes,
        gate7_audit=gate7_audit,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve and materialize immutable Capsule task initial-state hashes."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate7-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    add_validation_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    materialize(
        config_path=args.config,
        gate7_audit_path=args.gate7_audit,
        output_dir=args.output_dir,
        validate_only=validation_requested(args),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["MaterializationResult", "build_parser", "main", "materialize"]
