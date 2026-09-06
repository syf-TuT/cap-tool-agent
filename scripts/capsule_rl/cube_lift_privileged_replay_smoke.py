"""Run the narrow privileged Cube Lift deterministic clean-replay smoke.

Static validation is import-safe and deliberately separate from the lazy Robosuite/process
boundary. This is readiness evidence only; it does not run Program sampling, Controller repair,
Ray, VeRL, or an optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import socket
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from capx.rl.capsule.schema import ProgramReplayResultV1, ReplayOutcome, TaskInstanceV1
from capx.rl.capsule.task_profiles import (
    ROBOSUITE_CUBE_LIFT_PRIVILEGED_HIGHLEVEL,
    CapsuleTaskProfile,
    CapsuleTaskProfileError,
    collect_environment_profile_errors,
    resolve_task_profile,
)
from scripts.capsule_rl import common


SMOKE_MODE = "cube_lift_privileged_replay_smoke_v1"
REQUIRED_PROFILE = "robosuite_cube_lift_privileged_highlevel"
CANONICAL_TASK_ID = "cube-lift-red-cube"
CANONICAL_PROMPT = (
    "\n"
    "You are controlling a Franka Emika robot with API described below.\n"
    "Goal: pick up the red cube and lift it.\n"
    "You may write python code comments for reasoning but ONLY write the executable Python code "
    "and do not write it in code fences.\n"
    "The functions (APIs) below are already imported to the environment. If you want to use "
    "numpy, you need to import it explicitly.\n"
    "\n"
    "APIs:\n"
    "\n"
    "get_object_pose(object_name: str, return_bbox_extent: bool = False) -> "
    "tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray | None]\n"
    "  Doc:\n"
    "    Get the pose of an object in the environment from a natural language description.\n"
    "    The quaternion from get_object_pose may be unreliable, so disregard it and use the "
    "grasp pose quaternion OR (0, 0, 1, 0) wxyz as the gripper down orientation if using this "
    "for placement position.\n"
    "    \n"
    "    Args:\n"
    "        object_name: The name of the object to get the pose of.\n"
    "    \n"
    "    Returns:\n"
    "        position: (3,) XYZ in meters.\n"
    "        quaternion_wxyz: (4,) WXYZ unit quaternion.\n"
    "        bbox_extent: (3,) object extent in meters of x, y, z axes respectively in the "
    "world frame (full side length, not half-length extent). If return_bbox_extent is False, "
    "returns None.\n"
    "\n"
    "sample_grasp_pose(object_name: str) -> tuple[numpy.ndarray, numpy.ndarray]\n"
    "  Doc:\n"
    "    Sample a grasp pose for an object in the environment from a natural language "
    "description.\n"
    "    Do use the grasp sample quaternion from sample_grasp_pose.\n"
    "    \n"
    "    Args:\n"
    "        object_name: The name of the object to sample a grasp pose for.\n"
    "    \n"
    "    Returns:\n"
    "        position: (3,) XYZ in meters.\n"
    "        quaternion_wxyz: (4,) WXYZ unit quaternion.\n"
    "\n"
    "goto_pose(position: numpy.ndarray, quaternion_wxyz: numpy.ndarray, z_approach: float = "
    "0.0) -> None\n"
    "  Doc:\n"
    "    Go to pose using Inverse Kinematics.\n"
    "    There is no need to call a second goto_pose with the same position and "
    "quaternion_wxyz after calling it with z_approach.\n"
    "    Args:\n"
    "        position: (3,) XYZ in meters.\n"
    "        quaternion_wxyz: (4,) WXYZ unit quaternion.\n"
    "        z_approach: (float) Z-axis distance offset for goto_pose insertion approach "
    "motion. Will first arrive at position + z_approach meters in Z-axis before moving to the "
    "requested pose. Useful for more precise grasp approaches. Default is 0.0.\n"
    "    Returns:\n"
    "        None\n"
    "\n"
    "open_gripper() -> None\n"
    "  Doc:\n"
    "    Open gripper fully.\n"
    "    \n"
    "    Args:\n"
    "        None\n"
    "\n"
    "close_gripper() -> None\n"
    "  Doc:\n"
    "    Close gripper fully.\n"
    "    \n"
    "    Args:\n"
    "        None"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FORMAL_GATE_NAME_RE = re.compile(
    r"(?:^|[^a-z0-9])gate(?:[-_]?\d+)?(?:[^a-z0-9]|$)", re.IGNORECASE
)


class CubeLiftSmokeError(RuntimeError):
    """Raised when smoke input or runtime evidence fails closed."""


@dataclass(frozen=True)
class SmokeInputs:
    config: Mapping[str, Any]
    config_path: Path
    config_sha256: str
    profile: CapsuleTaskProfile
    environment_payload: Mapping[str, Any]
    environment_path: Path
    environment_bytes: bytes
    environment_sha256: str
    source_row: Mapping[str, str]
    source_path: Path
    source_sha256: str
    pyroki_host: str
    pyroki_port: int


@dataclass(frozen=True)
class RuntimeComponents:
    """Lazy runtime constructors, injectable only at the simulator/process boundary."""

    environment_factory_type: Callable[..., Any]
    replay_backend_type: Callable[..., Any]
    replay_evaluator_type: Callable[..., Any]


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise CubeLiftSmokeError(f"cannot read {label} {path}: {error}") from error


def _decode_yaml(raw_bytes: bytes, label: str) -> Mapping[str, Any]:
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CubeLiftSmokeError(f"{label} must be UTF-8 YAML") from error
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise CubeLiftSmokeError(f"{label} is not valid YAML: {error}") from error
    if not isinstance(payload, Mapping):
        raise CubeLiftSmokeError(f"{label} root must be a mapping")
    return payload


def _configured_project_root(config: Mapping[str, Any]) -> Path:
    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping):
        raise CubeLiftSmokeError("runtime must be a mapping")
    configured_root = runtime.get("project_root")
    if configured_root is None:
        return common.repository_root()
    if not isinstance(configured_root, str) or not configured_root.strip():
        raise CubeLiftSmokeError("runtime.project_root must be a non-empty path when set")
    return Path(configured_root).expanduser().resolve()


def _resolve_environment_path(config: Mapping[str, Any]) -> Path:
    task = config.get("task")
    if not isinstance(task, Mapping):
        raise CubeLiftSmokeError("task must be a mapping")
    configured_path = task.get("config_path")
    if not isinstance(configured_path, str) or not configured_path.strip():
        raise CubeLiftSmokeError("task.config_path must be a non-empty path")
    path = Path(configured_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (_configured_project_root(config) / path).resolve()


def _load_source_row(raw_bytes: bytes, source_path: Path) -> Mapping[str, str]:
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CubeLiftSmokeError(f"source task must be UTF-8 JSONL: {source_path}") from error
    rows = [line for line in text.splitlines() if line.strip()]
    if len(rows) != 1:
        raise CubeLiftSmokeError(
            f"source task must contain exactly one JSONL mapping row; found {len(rows)}"
        )
    try:
        row = json.loads(rows[0])
    except json.JSONDecodeError as error:
        raise CubeLiftSmokeError(f"source task row is not valid JSON: {error}") from error
    if not isinstance(row, Mapping):
        raise CubeLiftSmokeError("source task row must be a mapping")
    if set(row) != {"task_id", "prompt"}:
        raise CubeLiftSmokeError(
            "source task row must contain exact keys ['prompt', 'task_id']"
        )
    if row.get("task_id") != CANONICAL_TASK_ID or row.get("prompt") != CANONICAL_PROMPT:
        raise CubeLiftSmokeError(
            "source task row must exactly match the canonical privileged Cube Lift task"
        )
    return {"task_id": CANONICAL_TASK_ID, "prompt": CANONICAL_PROMPT}


def load_smoke_inputs(config_path: str | Path, source_path: str | Path) -> SmokeInputs:
    """Load and statically validate every smoke input without runtime side effects."""

    resolved_config = Path(config_path).expanduser().resolve()
    resolved_source = Path(source_path).expanduser().resolve()
    config_bytes = _read_bytes(resolved_config, "Capsule config")
    raw_config = _decode_yaml(config_bytes, "Capsule config")
    raw_task = raw_config.get("task")
    if not isinstance(raw_task, Mapping) or raw_task.get("profile") != REQUIRED_PROFILE:
        raise CubeLiftSmokeError(
            "smoke requires the explicit privileged Cube Lift profile "
            f"{REQUIRED_PROFILE!r}"
        )
    try:
        config = common.load_and_validate_server_config_bytes(
            config_bytes, check_runtime_paths=False
        )
    except common.ConfigValidationError as error:
        raise CubeLiftSmokeError(f"invalid Capsule config: {error}") from error
    try:
        profile = resolve_task_profile(config)
    except CapsuleTaskProfileError as error:
        raise CubeLiftSmokeError(f"cannot resolve Cube Lift task profile: {error}") from error
    if profile != ROBOSUITE_CUBE_LIFT_PRIVILEGED_HIGHLEVEL:
        raise CubeLiftSmokeError(
            "smoke requires the explicit privileged Cube Lift profile "
            f"{REQUIRED_PROFILE!r}"
        )

    environment_path = _resolve_environment_path(config)
    environment_bytes = _read_bytes(environment_path, "environment config")
    environment_payload = _decode_yaml(environment_bytes, "environment config")
    environment_errors = collect_environment_profile_errors(environment_payload, profile)
    if environment_errors:
        raise CubeLiftSmokeError(
            "environment config does not match the selected Cube Lift profile: "
            + "; ".join(environment_errors)
        )

    servers = environment_payload["api_servers"]
    server = servers[0]
    pyroki_host = server["host"]
    pyroki_port = server["port"]
    if not isinstance(pyroki_host, str) or type(pyroki_port) is not int:
        raise CubeLiftSmokeError("validated PyRoKi host/port have invalid types")

    source_bytes = _read_bytes(resolved_source, "source task")
    source_row = _load_source_row(source_bytes, resolved_source)
    return SmokeInputs(
        config=config,
        config_path=resolved_config,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        profile=profile,
        environment_payload=environment_payload,
        environment_path=environment_path,
        environment_bytes=environment_bytes,
        environment_sha256=hashlib.sha256(environment_bytes).hexdigest(),
        source_row=source_row,
        source_path=resolved_source,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        pyroki_host=pyroki_host,
        pyroki_port=pyroki_port,
    )


def parse_seed_sequence(value: str) -> tuple[int, ...]:
    if not isinstance(value, str):
        raise CubeLiftSmokeError("seed sequence must be comma-separated integers")
    parts = value.split(",")
    if not parts or any(not part.strip() for part in parts):
        raise CubeLiftSmokeError("seed sequence must be comma-separated integers")
    try:
        return tuple(int(part.strip()) for part in parts)
    except ValueError as error:
        raise CubeLiftSmokeError("seed sequence must be comma-separated integers") from error


def validate_execution_contract(
    *,
    seed_sequence: Sequence[int],
    replay_seed: int,
    replays: int,
    timeout_s: float,
    output_path: str | Path | None,
) -> Path:
    """Validate the deliberately narrow runtime contract before any network access."""

    if tuple(seed_sequence) != (5, 6, 5):
        raise CubeLiftSmokeError("smoke requires the exact seed sequence 5,6,5")
    if isinstance(replay_seed, bool) or replay_seed != 5:
        raise CubeLiftSmokeError("smoke requires replay seed 5")
    if isinstance(replays, bool) or replays != 2:
        raise CubeLiftSmokeError("smoke requires exactly two clean replays")
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(float(timeout_s))
        or float(timeout_s) <= 0
    ):
        raise CubeLiftSmokeError("timeout must be finite and positive")
    if float(timeout_s) != 180.0:
        raise CubeLiftSmokeError("smoke timeout must be exactly 180 seconds")
    if output_path is None:
        raise CubeLiftSmokeError("--output is required outside --validate-only")
    output = Path(output_path).expanduser().resolve()
    if _FORMAL_GATE_NAME_RE.search(output.name):
        raise CubeLiftSmokeError(
            "smoke output must not use a formal Gate artifact name"
        )
    if output.exists():
        raise FileExistsError(f"smoke output already exists: {output}")
    return output


def validate_seed_hashes(
    seed_sequence: Sequence[int], hashes: Sequence[object]
) -> str:
    """Require lowercase SHA-256 evidence for deterministic 5,6,5 resets."""

    seeds = tuple(seed_sequence)
    values = tuple(hashes)
    valid_hashes = len(values) == 3 and all(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None
        for value in values
    )
    deterministic = (
        valid_hashes
        and seeds == (5, 6, 5)
        and values[0] == values[2]
        and values[0] != values[1]
    )
    if not deterministic:
        raise CubeLiftSmokeError(
            "seed reset hashes failed the lowercase SHA-256 5,6,5 contract; "
            f"seeds={list(seeds)!r}, hashes={list(values)!r}"
        )
    return values[0]  # type: ignore[return-value]


def build_task_instance(
    inputs: SmokeInputs, *, environment_seed: int, initial_hash: str
) -> TaskInstanceV1:
    """Bind the canonical source row to the selected profile and measured reset state."""

    return TaskInstanceV1(
        task_id=inputs.source_row["task_id"],
        environment_seed=environment_seed,
        prompt=inputs.source_row["prompt"],
        environment=inputs.profile.environment,
        api=inputs.profile.api,
        privilege=inputs.profile.privilege,
        initial_state_sha256=initial_hash,
        metadata={"task_profile": inputs.profile.name},
    )


def _check_pyroki_ready(host: str, port: int) -> None:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            pass
    except OSError as error:
        raise CubeLiftSmokeError(
            f"PyRoKi endpoint {host}:{port} is not ready"
        ) from error


def _load_runtime_components() -> RuntimeComponents:
    """Import Robosuite/process-facing classes only after PyRoKi readiness succeeds."""

    from capx.rl.capsule.evaluator import CleanReplayEvaluator, PersistentProcessReplayBackend
    from capx.rl.capsule.server_factory import YamlEnvironmentFactory

    return RuntimeComponents(
        environment_factory_type=YamlEnvironmentFactory,
        replay_backend_type=PersistentProcessReplayBackend,
        replay_evaluator_type=CleanReplayEvaluator,
    )


def _close(resource: object) -> None:
    close = getattr(resource, "close", None)
    if callable(close):
        close()


def _probe_environment(
    factory: Callable[[TaskInstanceV1 | None], Any],
    seed_sequence: tuple[int, ...],
) -> tuple[tuple[object, ...], str]:
    probe = factory(None)
    hashes: list[object] = []
    try:
        for seed in seed_sequence:
            _observation, info = probe.reset(
                seed=seed, options={"capsule_smoke": "seed"}
            )
            initial_hash = (
                info.get("initial_state_sha256") if isinstance(info, Mapping) else None
            )
            hashes.append(initial_hash)
        oracle_source = getattr(probe, "oracle_code", None)
        if not isinstance(oracle_source, str) or not oracle_source.strip():
            raise CubeLiftSmokeError(
                "configured Cube Lift environment does not expose non-empty oracle_code"
            )
        return tuple(hashes), oracle_source
    finally:
        _close(probe)


def _validate_worker_pid(pid: object) -> int:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise CubeLiftSmokeError("clean replay backend did not expose a valid worker PID")
    return pid


def validate_replay_result(
    result: ProgramReplayResultV1,
    *,
    task: TaskInstanceV1,
    expected_program_sample_id: str,
) -> None:
    """Validate one replay's complete typed outcome and reset-cleanliness evidence."""

    if not isinstance(result, ProgramReplayResultV1):
        raise CubeLiftSmokeError("clean replay evaluator returned an untyped result")
    if result.task_id != task.task_id or result.environment_seed != task.environment_seed:
        raise CubeLiftSmokeError("clean replay task identity drifted")
    if result.program_sample_id != expected_program_sample_id:
        raise CubeLiftSmokeError("clean replay program_sample_id drifted")
    if result.initial_state_sha256 != task.initial_state_sha256:
        raise CubeLiftSmokeError("clean replay initial state hash drifted from probed seed 5")
    if result.outcome is not ReplayOutcome.SUCCESS:
        raise CubeLiftSmokeError("clean replay outcome was not success")
    if isinstance(result.binary_reward, bool) or result.binary_reward != 1.0:
        raise CubeLiftSmokeError("clean replay binary reward was not 1.0")
    if result.task_completed is not True:
        raise CubeLiftSmokeError("clean replay task completion was not true")
    if result.attempts != 1:
        raise CubeLiftSmokeError("clean replay did not use exactly one attempt")

    history = result.diagnostics.get("evaluator_attempt_history")
    if not isinstance(history, (list, tuple)) or len(history) != 1:
        raise CubeLiftSmokeError("clean replay attempt history must contain exactly one event")
    event = history[0]
    if (
        not isinstance(event, Mapping)
        or event.get("attempt") != 1
        or event.get("outcome") != ReplayOutcome.SUCCESS.value
        or event.get("worker_replaced") is not False
        or event.get("retry_scheduled") is not False
    ):
        raise CubeLiftSmokeError(
            "clean replay attempt history must prove no replacement or retry"
        )

    reset_info = result.diagnostics.get("reset_info")
    evidence = (
        reset_info.get("capsule_reset_evidence")
        if isinstance(reset_info, Mapping)
        else None
    )
    if not isinstance(reset_info, Mapping) or (
        reset_info.get("initial_state_sha256") != task.initial_state_sha256
    ):
        raise CubeLiftSmokeError("clean replay reset initial state hash drifted")
    if not isinstance(evidence, Mapping):
        raise CubeLiftSmokeError("clean replay reset evidence is missing")
    reset_count = evidence.get("api_reset_count")
    confirmed_count = evidence.get("api_reset_confirmed_count")
    if (
        evidence.get("namespace_fresh") is not True
        or evidence.get("api_state_cleared") is not True
        or isinstance(reset_count, bool)
        or not isinstance(reset_count, int)
        or reset_count < 1
        or isinstance(confirmed_count, bool)
        or not isinstance(confirmed_count, int)
        or confirmed_count != reset_count
    ):
        raise CubeLiftSmokeError(
            "clean replay reset evidence did not prove fresh namespace and cleared API state"
        )


def _artifact_payload(
    inputs: SmokeInputs,
    *,
    seed_sequence: tuple[int, ...],
    hashes: tuple[object, ...],
    task: TaskInstanceV1,
    replay_seed: int,
    replays: int,
    timeout_s: float,
    worker_ids: Sequence[int],
    results: Sequence[ProgramReplayResultV1],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": SMOKE_MODE,
        "passed": True,
        "config_path": str(inputs.config_path),
        "config_sha256": inputs.config_sha256,
        "environment_path": str(inputs.environment_path),
        "environment_sha256": inputs.environment_sha256,
        "source_task_path": str(inputs.source_path),
        "source_task_sha256": inputs.source_sha256,
        "profile": {
            "name": inputs.profile.name,
            "environment": inputs.profile.environment,
            "api": inputs.profile.api,
            "privilege": inputs.profile.privilege,
        },
        "task": task.to_dict(),
        "seed_sequence": list(seed_sequence),
        "initial_state_sha256": list(hashes),
        "replay_seed": replay_seed,
        "replay_count": replays,
        "timeout_s": float(timeout_s),
        "worker_ids": list(worker_ids),
        "replays": [result.to_dict() for result in results],
        "render_enabled": False,
        "record_video": False,
        "program_actor_used": False,
        "controller_used": False,
        "ray_used": False,
        "verl_used": False,
        "optimizer_used": False,
    }


def execute_smoke(
    inputs: SmokeInputs,
    *,
    seed_sequence: Sequence[int],
    replay_seed: int,
    replays: int,
    timeout_s: float,
    output_path: str | Path | None,
    readiness_checker: Callable[[str, int], None] | None = None,
    runtime_loader: Callable[[], RuntimeComponents] | None = None,
) -> dict[str, Any]:
    """Run deterministic resets and two same-worker oracle clean replays."""

    seeds = tuple(seed_sequence)
    output = validate_execution_contract(
        seed_sequence=seeds,
        replay_seed=replay_seed,
        replays=replays,
        timeout_s=timeout_s,
        output_path=output_path,
    )
    readiness = _check_pyroki_ready if readiness_checker is None else readiness_checker
    try:
        readiness(inputs.pyroki_host, inputs.pyroki_port)
    except CubeLiftSmokeError:
        raise
    except OSError as error:
        raise CubeLiftSmokeError(
            f"PyRoKi endpoint {inputs.pyroki_host}:{inputs.pyroki_port} is not ready"
        ) from error

    load_runtime = _load_runtime_components if runtime_loader is None else runtime_loader
    components = load_runtime()
    factory = components.environment_factory_type(
        str(inputs.environment_path), config_bytes=inputs.environment_bytes
    )
    hashes, oracle_source = _probe_environment(factory, seeds)
    seed_five_hash = validate_seed_hashes(seeds, hashes)
    task = build_task_instance(
        inputs, environment_seed=replay_seed, initial_hash=seed_five_hash
    )
    program_sample_id = f"{task.task_id}:seed-{replay_seed}:oracle"

    backend = components.replay_backend_type(factory, start_method="spawn")
    evaluator = None
    results: list[ProgramReplayResultV1] = []
    worker_ids: list[int] = []
    try:
        evaluator = components.replay_evaluator_type(
            backend,
            timeout_s=float(timeout_s),
            max_failure_retries=0,
        )
        for _ in range(replays):
            result = evaluator.evaluate_program(
                task,
                oracle_source,
                replay_seed,
                program_sample_id=program_sample_id,
            )
            worker_ids.append(_validate_worker_pid(backend.worker_pid))
            validate_replay_result(
                result,
                task=task,
                expected_program_sample_id=program_sample_id,
            )
            results.append(result)
        if len(set(worker_ids)) != 1:
            raise CubeLiftSmokeError(
                f"clean replays did not use the same persistent worker: {worker_ids!r}"
            )
    finally:
        if evaluator is not None:
            _close(evaluator)
        else:
            _close(backend)

    payload = _artifact_payload(
        inputs,
        seed_sequence=seeds,
        hashes=hashes,
        task=task,
        replay_seed=replay_seed,
        replays=replays,
        timeout_s=float(timeout_s),
        worker_ids=worker_ids,
        results=results,
    )
    common.atomic_write_json(output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-task", type=Path, required=True)
    parser.add_argument("--seed-sequence", default="5,6,5")
    parser.add_argument("--replay-seed", type=int, default=5)
    parser.add_argument("--replays", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def _validation_summary(inputs: SmokeInputs) -> dict[str, Any]:
    return {
        "mode": SMOKE_MODE,
        "validation_only": True,
        "config_path": str(inputs.config_path),
        "config_sha256": inputs.config_sha256,
        "environment_path": str(inputs.environment_path),
        "environment_sha256": inputs.environment_sha256,
        "source_task_path": str(inputs.source_path),
        "source_task_sha256": inputs.source_sha256,
        "profile": inputs.profile.name,
        "pyroki_endpoint": f"{inputs.pyroki_host}:{inputs.pyroki_port}",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = load_smoke_inputs(args.config, args.source_task)
    if args.validate_only:
        print(json.dumps(_validation_summary(inputs), indent=2, sort_keys=True))
        return 0
    payload = execute_smoke(
        inputs,
        seed_sequence=parse_seed_sequence(args.seed_sequence),
        replay_seed=args.replay_seed,
        replays=args.replays,
        timeout_s=args.timeout_s,
        output_path=args.output,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
