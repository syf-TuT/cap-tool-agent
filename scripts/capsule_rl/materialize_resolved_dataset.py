"""Materialize seed-resolved typed tasks; preview safely with --validate-only/--dry-run."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from capx.rl.capsule.actor_identity import ActorIdentityError, build_actor_identity
from capx.rl.capsule.provenance import project_path
from capx.rl.capsule.schema import TaskInstanceV1
from capx.rl.capsule.stable_io import (
    MutationWatch,
    PathMutationGuard,
    StableFileSnapshot,
    StablePathError,
    full_stat_identity,
    read_stable_regular_file,
)

from .common import (
    ConfigValidationError,
    GateArtifactError,
    add_validation_arguments,
    artifact_file_sha256,
    load_and_validate_server_config_bytes,
    runtime_dataset_path,
    validate_final_runtime_audit,
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
class _OwnedPublishedFile:
    name: str
    identity: tuple[int, int, int, int, int, int]
    expected_sha256: str


@dataclass(frozen=True)
class _OwnedBundlePublication:
    directory_identity: tuple[int, int]
    files: tuple[_OwnedPublishedFile, ...]


@dataclass(frozen=True)
class _Gate7Audit:
    source_path: Path
    raw_bytes: bytes
    sha256: str
    payload: Mapping[str, Any]
    run_id: str
    config_sha256: str
    dataset_sha256: str
    resolved_environment_sha256: str
    verl_resolved_config_sha256: str
    program_model_sha256: str
    actor_binding_sha256: str
    typed_task_identities: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class _SourceTaskExpectation:
    task: TaskInstanceV1
    initial_state_may_resolve: bool


@dataclass(frozen=True)
class _RuntimeDependencySnapshots:
    hashes: Mapping[str, str]
    resolved_verl_config: StableFileSnapshot
    environment_config: StableFileSnapshot | None


def _required_sha256(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ConfigValidationError(f"Gate7 audit {field_name} must be lowercase SHA-256")
    return value


def _build_actor_identity(
    config: Mapping[str, Any],
    *,
    resolved_config_snapshot: StableFileSnapshot | None = None,
) -> dict[str, Any]:
    try:
        return build_actor_identity(
            config,
            resolved_config_snapshot=resolved_config_snapshot,
        )
    except ActorIdentityError as error:
        raise ConfigValidationError(f"cannot bind Program actor identity: {error}") from error


def _stable_snapshot(path: str | Path, *, label: str) -> StableFileSnapshot:
    try:
        return read_stable_regular_file(path, label=label)
    except StablePathError as error:
        raise ConfigValidationError(str(error)) from error


def _snapshot_runtime_dependencies(
    config: Mapping[str, Any],
) -> _RuntimeDependencySnapshots:
    task = config.get("task")
    runtime = config.get("runtime")
    if not isinstance(task, Mapping):
        raise ConfigValidationError("task must be a mapping")
    if not isinstance(runtime, Mapping):
        raise ConfigValidationError("runtime must be a mapping")
    environment_snapshot: StableFileSnapshot | None = None
    environment_value = task.get("config_path")
    if isinstance(environment_value, str) and environment_value:
        try:
            environment_path = project_path(config, environment_value, "task.config_path")
        except ValueError as error:
            raise ConfigValidationError(
                f"cannot resolve task environment config: {error}"
            ) from error
        environment_snapshot = _stable_snapshot(
            environment_path,
            label="resolved environment config",
        )
    environment_sha256 = (
        environment_snapshot.sha256 if environment_snapshot is not None else None
    )
    serialized_environment = json.dumps(
        {
            "task": dict(task),
            "environment_config_sha256": environment_sha256,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    try:
        resolved_verl_path = project_path(
            config,
            runtime.get("verl_resolved_config_path"),
            "runtime.verl_resolved_config_path",
        )
    except ValueError as error:
        raise ConfigValidationError(
            f"cannot resolve runtime VeRL config: {error}"
        ) from error
    resolved_verl_snapshot = _stable_snapshot(
        resolved_verl_path,
        label="resolved VeRL config dependency",
    )
    return _RuntimeDependencySnapshots(
        hashes={
            "resolved_environment_sha256": hashlib.sha256(
                serialized_environment.encode("utf-8")
            ).hexdigest(),
            "verl_resolved_config_sha256": resolved_verl_snapshot.sha256,
        },
        resolved_verl_config=resolved_verl_snapshot,
        environment_config=environment_snapshot,
    )


def _load_gate7_audit(path: str | Path) -> _Gate7Audit:
    snapshot = _stable_snapshot(path, label="Gate7 audit")
    audit_path = snapshot.path
    raw_bytes = snapshot.raw_bytes
    try:
        payload = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfigValidationError(f"Gate7 audit is not valid UTF-8 JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise ConfigValidationError("Gate7 audit root must be a JSON mapping")
    try:
        validate_final_runtime_audit(payload)
    except GateArtifactError as error:
        raise ConfigValidationError(f"Gate7 audit contract is invalid: {error}") from error
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
        sha256=snapshot.sha256,
        payload=dict(payload),
        run_id=run_id,
        config_sha256=_required_sha256(payload, "config_sha256"),
        dataset_sha256=_required_sha256(payload, "dataset_sha256"),
        resolved_environment_sha256=_required_sha256(
            payload, "resolved_environment_sha256"
        ),
        verl_resolved_config_sha256=_required_sha256(
            payload, "verl_resolved_config_sha256"
        ),
        program_model_sha256=_required_sha256(payload, "program_model_sha256"),
        actor_binding_sha256=_required_sha256(payload, "actor_binding_sha256"),
        typed_task_identities=tuple(normalized_identities),
    )


def _validate_gate7_bindings(
    audit: _Gate7Audit,
    *,
    source_config_sha256: str,
    source_dataset_sha256: str,
    dependency_hashes: Mapping[str, str],
    actor_identity: Mapping[str, Any],
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
    if audit.program_model_sha256 != actor_identity["program_model_sha256"]:
        raise ConfigValidationError(
            "Gate7 audit Program model SHA does not match the source model bytes"
        )
    if audit.actor_binding_sha256 != actor_identity["actor_binding_sha256"]:
        raise ConfigValidationError(
            "Gate7 audit actor binding SHA does not match the source actor identity"
        )


def _gate7_evidence_paths(audit: _Gate7Audit) -> dict[str, Path]:
    root = audit.source_path.parent
    return {
        "candidate": root / "gate07_audit.candidate.json",
        "continuous_memory": root / "launcher_continuous_memory.json",
        "controller_attestation": root / "launcher_controller_attestation.json",
        "owned_cleanup": root / "launcher_owned_cleanup.json",
        "initial_audit": root / "launcher_initial_audit.json",
        "post_controller_memory": root / "launcher_memory_00_post-controller.json",
    }


def _recompute_final_runtime_audit(audit: _Gate7Audit) -> Mapping[str, Any]:
    """Re-run the producer verifier instead of trusting copied final JSON fields."""

    # Imported lazily to avoid making validate-only CLI import every gate implementation at
    # module import time.  ``analyze_artifacts`` imports ``common`` and therefore cannot be a
    # module-level dependency here without a circular import.
    from .analyze_artifacts import finalize_runtime_audit

    evidence = _gate7_evidence_paths(audit)
    return finalize_runtime_audit(
        audit.source_path.parent,
        candidate_artifact=evidence["candidate"],
        continuous_memory_artifact=evidence["continuous_memory"],
        controller_attestation_artifact=evidence["controller_attestation"],
        owned_cleanup_artifact=evidence["owned_cleanup"],
    )


@contextmanager
def _verified_gate7_evidence(audit: _Gate7Audit):
    """Pin Gate 7's producer evidence until the materialized copy is reverified."""

    from .analyze_artifacts import REQUIRED_GATE_FILES

    evidence = _gate7_evidence_paths(audit)
    watches = [MutationWatch(audit.source_path, "final Gate7 runtime audit")]
    watches.append(
        MutationWatch(
            audit.source_path.parent / "resolved" / "verl.yaml",
            "Gate7 resolved VeRL profile",
        )
    )
    watches.extend(
        MutationWatch(path, f"Gate7 {label.replace('_', ' ')} evidence")
        for label, path in evidence.items()
    )
    watches.extend(
        MutationWatch(audit.source_path.parent / filename, f"Gate7 {gate} artifact")
        for gate, filename in REQUIRED_GATE_FILES.items()
    )
    try:
        guard = PathMutationGuard.open(watches)
    except StablePathError as error:
        raise ConfigValidationError(f"cannot guard Gate7 producer evidence: {error}") from error
    try:
        try:
            recomputed = _recompute_final_runtime_audit(audit)
            validate_final_runtime_audit(recomputed)
        except GateArtifactError as error:
            raise ConfigValidationError(
                f"cannot independently reproduce final Gate7 runtime audit: {error}"
            ) from error
        if dict(recomputed) != dict(audit.payload):
            raise ConfigValidationError(
                "Gate7 audit fields do not equal the independently reproduced final audit"
            )
        yield guard
        guard.assert_unchanged(context="during seed-resolved bundle materialization")
    except StablePathError as error:
        raise ConfigValidationError(str(error)) from error
    finally:
        guard.close()


def _source_rows(
    dataset_snapshot: StableFileSnapshot,
) -> list[tuple[int, Mapping[str, Any]]]:
    dataset_path = dataset_snapshot.path
    suffix = dataset_path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        try:
            dataset_text = dataset_snapshot.raw_bytes.decode("utf-8")
        except UnicodeError as error:
            raise ConfigValidationError(
                f"runtime.dataset_path is not valid UTF-8: {error}"
            ) from error
        rows: list[tuple[int, Mapping[str, Any]]] = []
        for line_number, line in enumerate(
            dataset_text.splitlines(), start=1
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
                pd.read_parquet(io.BytesIO(dataset_snapshot.raw_bytes)).to_dict("records"),
                start=1,
            )
        ]
    raise ConfigValidationError(
        "runtime.dataset_path must be JSONL, JSON, or Parquet"
    )


def _source_task_expectations(
    config: Mapping[str, Any], dataset_snapshot: StableFileSnapshot
) -> tuple[_SourceTaskExpectation, ...]:
    task_config = config["task"]
    identities: set[tuple[str, int]] = set()
    expectations: list[_SourceTaskExpectation] = []
    for line_number, payload in _source_rows(dataset_snapshot):
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


def _capture_owned_published_file(
    path: Path,
    *,
    expected_identity: tuple[int, int],
    expected_sha256: str,
) -> _OwnedPublishedFile:
    snapshot = _stable_snapshot(path, label=f"published bundle file {path.name}")
    if snapshot.identity[:2] != expected_identity or snapshot.sha256 != expected_sha256:
        raise ConfigValidationError(
            f"published bundle file {path.name} changed before ownership was recorded"
        )
    return _OwnedPublishedFile(path.name, snapshot.identity, expected_sha256)


def _owned_directory_is_current(
    destination: Path, expected_identity: tuple[int, int]
) -> bool:
    try:
        current = destination.stat(follow_symlinks=False)
    except OSError:
        return False
    return (
        stat.S_ISDIR(current.st_mode)
        and not destination.is_symlink()
        and (current.st_dev, current.st_ino) == expected_identity
    )


def _remove_owned_destination(
    destination: Path, publication: _OwnedBundlePublication
) -> None:
    if not _owned_directory_is_current(destination, publication.directory_identity):
        return
    for owned_file in reversed(publication.files):
        path = destination / owned_file.name
        try:
            snapshot = read_stable_regular_file(
                path,
                label=f"owned published bundle file {owned_file.name}",
            )
        except StablePathError:
            continue
        if (
            snapshot.identity != owned_file.identity
            or snapshot.sha256 != owned_file.expected_sha256
            or not _owned_directory_is_current(
                destination, publication.directory_identity
            )
        ):
            continue
        try:
            named = path.stat(follow_symlinks=False)
        except OSError:
            continue
        if full_stat_identity(named) != owned_file.identity:
            continue
        path.unlink()
    if not _owned_directory_is_current(destination, publication.directory_identity):
        return
    try:
        if any(destination.iterdir()):
            return
    except OSError:
        return
    if not _owned_directory_is_current(destination, publication.directory_identity):
        return
    destination.rmdir()


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
) -> tuple[MaterializationResult, _OwnedBundlePublication]:
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
    owned_files: list[_OwnedPublishedFile] = []
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
            "program_model_sha256": gate7_audit.program_model_sha256,
            "actor_binding_sha256": gate7_audit.actor_binding_sha256,
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
        expected_file_sha256s = {
            dataset_name: output_dataset_sha256,
            config_name: output_config_sha256,
            audit_name: gate7_audit.sha256,
            manifest_name: artifact_file_sha256(staging / manifest_name),
        }
        staged_file_identities: dict[str, tuple[int, int]] = {}
        for filename, expected_sha256 in expected_file_sha256s.items():
            staged_snapshot = _stable_snapshot(
                staging / filename,
                label=f"staged bundle file {filename}",
            )
            if staged_snapshot.sha256 != expected_sha256:
                raise ConfigValidationError(
                    f"staged bundle file {filename} changed before publication"
                )
            staged_file_identities[filename] = staged_snapshot.identity[:2]
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
            owned_files.append(
                _capture_owned_published_file(
                    destination / filename,
                    expected_identity=staged_file_identities[filename],
                    expected_sha256=expected_file_sha256s[filename],
                )
            )
        _fsync_directory(destination)
        _fsync_directory(destination.parent)
        copied_audit = _load_gate7_audit(final_audit)
        if (
            copied_audit.sha256 != gate7_audit.sha256
            or dict(copied_audit.payload) != dict(gate7_audit.payload)
        ):
            raise ConfigValidationError(
                "materialized Gate7 audit does not equal the verified source bytes"
            )
        publication_complete = True
        publication = _OwnedBundlePublication(
            directory_identity=owned_destination_identity,
            files=tuple(owned_files),
        )
        return (
            MaterializationResult(len(tasks), final_dataset, final_config, final_manifest),
            publication,
        )
    except BaseException as error:
        primary_error = error
        if owned_destination_identity is not None:
            try:
                _remove_owned_destination(
                    destination,
                    _OwnedBundlePublication(
                        directory_identity=owned_destination_identity,
                        files=tuple(owned_files),
                    ),
                )
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


@contextmanager
def _staged_resolver_config(
    config: Mapping[str, Any],
    *,
    source_dataset: StableFileSnapshot,
    dependencies: _RuntimeDependencySnapshots,
):
    """Point the resolver at small immutable byte snapshots instead of mutable source paths."""

    staging = Path(tempfile.mkdtemp(prefix=".capsule-resolver-inputs-"))
    try:
        dataset_suffix = source_dataset.path.suffix.lower()
        staged_dataset = staging / f"dataset{dataset_suffix}"
        staged_verl = staging / "resolved_verl.yaml"
        _write_synced_bytes(staged_dataset, source_dataset.raw_bytes)
        _write_synced_bytes(staged_verl, dependencies.resolved_verl_config.raw_bytes)
        resolver_config = deepcopy(dict(config))
        resolver_config["runtime"] = dict(resolver_config["runtime"])
        resolver_config["runtime"]["dataset_path"] = str(staged_dataset)
        resolver_config["runtime"]["verl_resolved_config_path"] = str(staged_verl)
        if dependencies.environment_config is not None:
            environment_suffix = dependencies.environment_config.path.suffix.lower() or ".yaml"
            staged_environment = staging / f"environment{environment_suffix}"
            _write_synced_bytes(
                staged_environment,
                dependencies.environment_config.raw_bytes,
            )
            resolver_config["task"] = dict(resolver_config["task"])
            resolver_config["task"]["config_path"] = str(staged_environment)
        _fsync_directory(staging)
        yield resolver_config
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _materialize_verified_inputs(
    *,
    config_file: Path,
    destination: Path,
    config: Mapping[str, Any],
    source_config_sha256: str,
    source_dataset_path: Path,
    source_dataset_snapshot: StableFileSnapshot,
    source_dataset_sha256: str,
    dependency_snapshots: _RuntimeDependencySnapshots,
    dependency_hashes: Mapping[str, str],
    actor_identity: Mapping[str, Any],
    gate7_audit: _Gate7Audit,
    validate_only: bool,
    task_resolver: TaskResolver | None,
) -> tuple[MaterializationResult, _OwnedBundlePublication | None]:
    _validate_gate7_bindings(
        gate7_audit,
        source_config_sha256=source_config_sha256,
        source_dataset_sha256=source_dataset_sha256,
        dependency_hashes=dependency_hashes,
        actor_identity=actor_identity,
    )
    if destination.exists():
        raise FileExistsError(f"materialization output already exists: {destination}")
    nearest_parent = destination.parent
    while not nearest_parent.exists() and nearest_parent != nearest_parent.parent:
        nearest_parent = nearest_parent.parent
    if not nearest_parent.is_dir():
        raise ConfigValidationError(f"output directory has no existing parent: {destination}")

    source_expectations = _source_task_expectations(config, source_dataset_snapshot)
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
        return result, None

    if task_resolver is None:
        # Delayed deliberately: validation mode must not import simulator/Ray-facing code.
        from capx.rl.capsule.server_factory import resolve_task_instances

        task_resolver = resolve_task_instances
    with _staged_resolver_config(
        config,
        source_dataset=source_dataset_snapshot,
        dependencies=dependency_snapshots,
    ) as resolver_config:
        tasks = _validate_resolved_tasks(
            task_resolver(resolver_config),
            source_expectations,
        )
    if _stable_snapshot(config_file, label="source config").sha256 != source_config_sha256:
        raise ConfigValidationError("source config changed during task resolution")
    if _stable_snapshot(source_dataset_path, label="source dataset").sha256 != (
        source_dataset_sha256
    ):
        raise ConfigValidationError("source dataset changed during task resolution")
    dependencies_after = _snapshot_runtime_dependencies(config)
    if dependencies_after.hashes != dependency_hashes:
        raise ConfigValidationError("runtime dependencies changed during task resolution")
    if _build_actor_identity(
        config,
        resolved_config_snapshot=dependencies_after.resolved_verl_config,
    ) != actor_identity:
        raise ConfigValidationError("Program actor identity changed during task resolution")
    if (
        _stable_snapshot(gate7_audit.source_path, label="Gate7 audit").sha256
        != gate7_audit.sha256
    ):
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
    published = _publish_bundle(
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
    return published


def materialize(
    *,
    config_path: str | Path,
    gate7_audit_path: str | Path,
    output_dir: str | Path,
    validate_only: bool,
    task_resolver: TaskResolver | None = None,
) -> MaterializationResult:
    config_file = Path(os.path.abspath(Path(config_path).expanduser()))
    destination = Path(output_dir).expanduser().resolve()
    config_snapshot = _stable_snapshot(config_file, label="source config")
    config = load_and_validate_server_config_bytes(
        config_snapshot.raw_bytes,
        check_runtime_paths=True,
    )
    source_config_sha256 = config_snapshot.sha256
    source_dataset_path = runtime_dataset_path(config)
    source_dataset_snapshot = _stable_snapshot(
        source_dataset_path,
        label="source dataset",
    )
    source_dataset_sha256 = source_dataset_snapshot.sha256
    dependency_snapshots = _snapshot_runtime_dependencies(config)
    dependency_hashes = dependency_snapshots.hashes
    actor_identity = _build_actor_identity(
        config,
        resolved_config_snapshot=dependency_snapshots.resolved_verl_config,
    )
    gate7_audit = _load_gate7_audit(gate7_audit_path)
    published_ownership: _OwnedBundlePublication | None = None
    try:
        with _verified_gate7_evidence(gate7_audit):
            result, published_ownership = _materialize_verified_inputs(
                config_file=config_file,
                destination=destination,
                config=config,
                source_config_sha256=source_config_sha256,
                source_dataset_path=source_dataset_path,
                source_dataset_snapshot=source_dataset_snapshot,
                source_dataset_sha256=source_dataset_sha256,
                dependency_snapshots=dependency_snapshots,
                dependency_hashes=dependency_hashes,
                actor_identity=actor_identity,
                gate7_audit=gate7_audit,
                validate_only=validate_only,
                task_resolver=task_resolver,
            )
            return result
    except BaseException as error:
        if published_ownership is not None:
            try:
                _remove_owned_destination(destination, published_ownership)
            except BaseException as cleanup_error:
                _record_cleanup_error(error, cleanup_error)
        raise


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
