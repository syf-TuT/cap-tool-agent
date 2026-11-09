"""Pair an online GRPO run with the repair actions selected by a CC-GRPO run."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .group import GroupAssemblyResult
from .repair import BaseUnitSpan, RepairDraft
from .schema import RepairTraceV1, source_sha256

MANIFEST_NAME = "repair-trajectories.json"
TRAJECTORY_BATCH_SIZE = 19


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return source_sha256(encoded)


def _is_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def supervised_turns(trace: RepairTraceV1) -> tuple[Mapping[str, Any], ...]:
    """Select accepted action targets and verify their alignment with the trace."""
    data = trace.to_dict()
    turns = data.get("controller_turns", [])
    events = sorted(data["edits"] + data["audits"], key=lambda event: event["turn_index"])
    if not turns or len(turns) != len(events) or len(turns) > 12:
        raise ValueError(f"{trace.repair_trajectory_id}: missing or incomplete Controller turns")
    draft = RepairDraft(
        task_id=trace.task_id,
        environment_seed=trace.environment_seed,
        program_sample_id=trace.program_sample_id,
        repair_trajectory_id=trace.repair_trajectory_id,
        base_source=trace.base_source,
        base_units=tuple(
            BaseUnitSpan(unit.target.removeprefix("base:"), unit.start_offset, unit.end_offset,
                         unit.source)
            for unit in trace.base_units
        ),
    )
    supervised = []
    for index, (turn, event) in enumerate(zip(turns, events, strict=True), start=1):
        messages = turn.get("messages")
        if (
            turn.get("turn_index") != index
            or event["turn_index"] != index
            or turn.get("result") != event
            or not isinstance(messages, list)
            or len(messages) != 2
            or any(
                not isinstance(message, dict)
                or message.get("role") != role
                or not isinstance(message.get("content"), str)
                or not message["content"]
                for message, role in zip(messages, ("system", "user"), strict=True)
            )
            or not isinstance(turn.get("response"), str)
        ):
            raise ValueError(f"{trace.repair_trajectory_id}: invalid Controller turn {index}")
        if "edit_index" in event or (
            event.get("event_type") == "finish" and event.get("status") == "accepted"
        ):
            supervised.append(turn)
        submission = draft.submit_json(turn["response"])
        replayed = submission.edit if submission.committed else submission.audit
        # Parse-error wording varies by Python version; the action and status must still match.
        actual = replayed.to_dict() if replayed is not None else {}
        expected = event
        if event.get("event_type") == "parse_failure":
            actual.pop("message", None)
            expected = {key: value for key, value in event.items() if key != "message"}
        if actual != expected:
            raise ValueError(f"{trace.repair_trajectory_id}: reply does not replay at turn {index}")
    if not supervised:
        raise ValueError(f"{trace.repair_trajectory_id}: no accepted repair actions")
    return tuple(supervised)


def export_manifest(root: Path) -> dict[str, Any]:
    """Read a completed, single-round CC-GRPO run and materialize a portable snapshot."""
    from .server_factory import _schedule_training_tasks, load_task_instances
    from .trainer import validate_group_provenance

    import yaml

    protocol_bytes = (root / "protocol.json").read_bytes()
    result_bytes = (root / "training_result.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    result = json.loads(result_bytes)
    if (
        protocol.get("method", "cc-grpo") != "cc-grpo"
        or protocol.get("resume_from") is not None
        or result.get("status") not in {"completed", "completed_no_updates_all_constant"}
    ):
        raise ValueError("export requires a completed CC-GRPO run from the base model")
    if not protocol.get("record_repair_turns"):
        raise ValueError("source run lacks Controller turns; prepare with --record-repair-turns")
    for name in ("runtime.yaml", "verl.yaml", "dataset.seed_resolved.jsonl"):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != protocol["input_sha256"][name]:
            raise ValueError(f"source run input changed: {name}")
    config = yaml.safe_load((root / "runtime.yaml").read_text(encoding="utf-8"))
    if (
        config["actor_rollout_ref"]["actor"]["policy_loss"]["loss_mode"] != "capsule_critique"
        or not config["capsule"].get("record_repair_turns")
        or config["runtime"].get("base_model_sha256") != protocol.get("base_model_sha256")
    ):
        raise ValueError("source runtime does not match the recorded CC-GRPO protocol")
    worker_config = yaml.safe_load((root / "verl.yaml").read_text(encoding="utf-8"))
    if worker_config["trainer"]["total_epochs"] != 1:
        raise ValueError("export requires exactly one training round")
    # Load the verified dataset snapshot from this run, including when the run was copied.
    config["runtime"]["dataset_path"] = str((root / "dataset.seed_resolved.jsonl").resolve())
    tasks = _schedule_training_tasks(load_task_instances(config), 1)
    paths = sorted((root / "training/groups").glob("*.json"))
    artifact_hashes = result.get("group_artifact_sha256")
    if (
        len(paths) != len(tasks)
        or len(tasks) != result.get("completed_group_count")
        or len(tasks) != protocol.get("group_count")
        or [task.environment_seed for task in tasks] != protocol.get("new_training_seeds")
        or not isinstance(artifact_hashes, dict)
        or set(artifact_hashes) != {path.name for path in paths}
    ):
        raise ValueError("source run does not have one saved group per scheduled seed")
    groups = []
    for task, path in zip(tasks, paths, strict=True):
        raw = path.read_bytes()
        artifact_sha256 = hashlib.sha256(raw).hexdigest()
        if artifact_sha256 != artifact_hashes[path.name]:
            raise ValueError(f"source group artifact changed: {path.name}")
        assembly = GroupAssemblyResult.from_dict(json.loads(raw)["assembly"])
        validate_group_provenance(task, assembly, config["capsule"].get("repair_trigger", "all_failed"))
        selected = next((attempt for attempt in assembly.repair_attempts if attempt.selected), None)
        trace = selected.trace if selected is not None else None
        if selected is not None and trace is None:
            raise ValueError(f"selected attempt has no trace: {selected.repair_trajectory_id}")
        if trace is not None:
            supervised_turns(trace)
        serialized = trace.to_dict() if trace is not None else None
        groups.append({
            "group_uid": assembly.group.group_uid,
            "task_id": task.task_id,
            "seed": task.environment_seed,
            "prompt_sha256": source_sha256(task.prompt),
            "initial_state_sha256": task.initial_state_sha256,
            "source_artifact_sha256": artifact_sha256,
            "trajectory": serialized,
            "trajectory_sha256": _digest(serialized) if serialized is not None else None,
        })
    manifest = {
        "schema_version": 1,
        "source_method": "cc-grpo",
        "source_protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "source_result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "environment": config["task"]["environment"],
        "base_model_sha256": protocol.get("base_model_sha256"),
        "program_system_prompt_sha256": source_sha256(config["program_service"]["system_prompt"]),
        "groups": groups,
    }
    manifest["manifest_sha256"] = _digest(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    body = dict(manifest)
    if (
        body.pop("manifest_sha256", None) != _digest(body)
        or body.get("schema_version") != 1
        or body.get("source_method") != "cc-grpo"
        or any(not _is_hash(body.get(key)) for key in (
            "source_protocol_sha256", "source_result_sha256",
            "base_model_sha256", "program_system_prompt_sha256",
        ))
    ):
        raise ValueError("invalid CC-GRPO repair manifest or manifest hash")
    groups = body.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("repair manifest requires at least one source group")
    seeds, group_ids, trace_ids = set(), set(), set()
    for entry in groups:
        if not isinstance(entry, dict):
            raise ValueError("repair manifest group must be a JSON object")
        seed, group_uid = entry.get("seed"), entry.get("group_uid")
        if (
            type(seed) is not int or seed < 0 or seed in seeds
            or not isinstance(group_uid, str) or not group_uid or group_uid in group_ids
            or not isinstance(entry.get("task_id"), str) or not entry["task_id"]
            or any(not _is_hash(entry.get(key)) for key in (
                "prompt_sha256", "initial_state_sha256", "source_artifact_sha256"
            ))
        ):
            raise ValueError(f"invalid or duplicate repair manifest group for seed {seed}")
        seeds.add(seed)
        group_ids.add(group_uid)
        trajectory = entry.get("trajectory")
        if trajectory is None:
            if entry.get("trajectory_sha256") is not None:
                raise ValueError(f"unexpected trajectory hash for seed {seed}")
            continue
        if _digest(trajectory) != entry.get("trajectory_sha256"):
            raise ValueError(f"trajectory hash mismatch for seed {seed}")
        trace = RepairTraceV1.from_dict(trajectory)
        if (
            trace.task_id != entry["task_id"]
            or trace.environment_seed != seed
            or trace.repair_trajectory_id in trace_ids
            or not trace.repair_trajectory_id.startswith(f"{group_uid}:")
        ):
            raise ValueError(f"invalid or duplicate trajectory identity for seed {seed}")
        trace_ids.add(trace.repair_trajectory_id)
        supervised_turns(trace)


def read_manifest(path: Path, expected_file_sha256: str | None = None) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_file_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise ValueError("repair manifest snapshot changed")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("repair manifest must be a JSON object")
    validate_manifest(manifest)
    return manifest


def match_manifest(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *,
    environment: str, model_sha256: str, system_prompt: str,
) -> None:
    validate_manifest(manifest)
    if (
        manifest["environment"] != environment
        or manifest["base_model_sha256"] != model_sha256
        or manifest["program_system_prompt_sha256"] != source_sha256(system_prompt)
        or len(manifest["groups"]) != len(rows)
    ):
        raise ValueError("repair manifest does not match the training task, model, or schedule")
    for row, entry in zip(rows, manifest["groups"], strict=True):
        if (
            entry["seed"] != row["environment_seed"]
            or entry["task_id"] != row["task_id"]
            or entry["initial_state_sha256"] != row["initial_state_sha256"]
            or entry["prompt_sha256"] != source_sha256(row["prompt"])
        ):
            raise ValueError(f"repair manifest task identity mismatch for seed {row['environment_seed']}")
