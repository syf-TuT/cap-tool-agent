"""Concrete server runtime for Capsule-RL gates 2 through 6.

The command has a deliberately thin, dependency-injected dispatcher.  ``--validate-only``
loads paths and expands the requested gate without importing Robosuite, VeRL, Ray, a model
client, or an optimizer.  Removing that flag selects :class:`ConcreteGateRuntime`, whose heavy
imports are local to the gate method that needs them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from capx.rl.capsule.actor_identity import (
    ActorIdentityError,
    build_actor_identity,
    verify_actor_identity_payload,
)
from capx.rl.capsule.controller import FrozenControllerConfig
from capx.rl.capsule.schema import (
    ProgramReplayResultV1,
    ReplayOutcome,
    TaskInstanceV1,
)
from capx.rl.capsule.stable_io import (
    MutationWatch,
    PathMutationGuard,
    StablePathError,
    read_stable_regular_file,
)
from capx.rl.capsule.telemetry import summarize_replay_results

from .common import (
    CANONICAL_EXECUTION_MODE,
    GateArtifactError,
    artifact_file_sha256,
    atomic_write_json,
    direct_lora_adapter_evidence,
    gate_failure_artifact_path,
    load_and_validate_server_config,
    load_and_validate_server_config_bytes,
    load_json_artifact,
    runtime_dataset_path,
    runtime_dependency_hashes,
    verify_collector_gate_artifact,
    verify_guided_gate_artifact,
    verify_oracle_gate_artifact,
    verify_seed_gate_artifact,
    verify_trainer_gate_artifact,
    write_gate_failure_artifact,
)


class ServerAdapterError(RuntimeError):
    """A server gate cannot produce trustworthy evidence."""


def _controller_runtime_config(config: Mapping[str, Any]) -> FrozenControllerConfig:
    capsule = config["capsule"]
    controller = config["controller_service"]
    return FrozenControllerConfig(
        endpoint=str(controller["endpoint"]),
        model=str(controller["model"]),
        api_key_env=str(controller["api_key_env"]),
        frozen=True,
        max_turns=int(capsule["max_controller_turns"]),
        request_timeout_s=float(controller["request_timeout_s"]),
        max_output_tokens=int(controller["max_output_tokens"]),
        stream=controller["stream"],
        enable_thinking=controller["enable_thinking"],
        temperature=float(controller["temperature"]),
    )


def _derive_training_tensor_evidence(result: Any, group: Any) -> dict[str, Any]:
    """Derive Gate 6 mask/KL evidence from the exact batch consumed by ``update_actor``."""

    import torch

    expected_trace = ("old_logprob", "reference_logprob", "update")
    trace = getattr(result, "execution_trace", None)
    normalized_trace = tuple(trace) if isinstance(trace, (list, tuple)) else ()
    if normalized_trace != expected_trace:
        raise ServerAdapterError(
            "trainer execution trace must be exactly "
            "old_logprob -> reference_logprob -> update"
        )
    tensors = getattr(result.batch, "batch", result.batch)
    required = (
        "response_mask",
        "guided_token_mask",
        "rollout_is_weights",
        "old_log_probs",
        "ref_log_prob",
    )
    if any(name not in tensors for name in required):
        raise ServerAdapterError(
            "trainer update batch omitted response/guided/old/reference log-prob evidence"
        )
    response_mask, guided_mask, rollout_mask, old_log_probs, ref_log_prob = (
        tensors[name] for name in required
    )
    if any(
        not isinstance(value, torch.Tensor)
        for value in (response_mask, guided_mask, rollout_mask, old_log_probs, ref_log_prob)
    ):
        raise ServerAdapterError("trainer Gate 6 evidence fields must be torch tensors")
    expected_shape = response_mask.shape
    if (
        response_mask.ndim != 2
        or expected_shape[0] != 8
        or any(
            value.shape != expected_shape
            for value in (guided_mask, rollout_mask, old_log_probs, ref_log_prob)
        )
        or response_mask.dtype != torch.bool
        or guided_mask.dtype != torch.bool
        or rollout_mask.dtype != torch.bool
        or not old_log_probs.is_floating_point()
        or not ref_log_prob.is_floating_point()
        or torch.any(response_mask.sum(dim=-1) == 0).item()
    ):
        raise ServerAdapterError(
            "trainer mask and old/reference log-prob tensors must share shape (8, response_length)"
        )
    guided_rows = torch.tensor(
        [member.member_type == "critique_guided_revision" for member in group.members],
        dtype=torch.bool,
        device=response_mask.device,
    )
    expected_guided_mask = response_mask & guided_rows.unsqueeze(-1)
    if (
        int(guided_rows.sum().item()) != 1
        or not torch.equal(guided_mask, expected_guided_mask)
        or not torch.equal(rollout_mask, guided_mask)
    ):
        raise ServerAdapterError(
            "trainer guided/rollout mask does not derive from the single guided response row"
        )
    if not torch.all(torch.isfinite(old_log_probs)).item():
        raise ServerAdapterError("trainer old_log_probs contains non-finite values")
    if not torch.all(torch.isfinite(ref_log_prob)).item():
        raise ServerAdapterError("trainer ref_log_prob contains non-finite values")
    guided_token_count = int(guided_mask.sum().item())
    if guided_token_count < 1:
        raise ServerAdapterError("trainer guided response contains no guided tokens")
    return {
        "guided_token_mask_present": True,
        "guided_token_count": guided_token_count,
        "guided_token_mask_shape": list(expected_shape),
        "guided_row_indices": torch.nonzero(guided_rows, as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .tolist(),
        "guided_mask_response_only": True,
        "rollout_mask_matches_guided": True,
        "old_log_probs_finite": True,
        "reference_log_probs_finite": True,
        "reference_log_prob_shape": list(ref_log_prob.shape),
        "reference_log_prob_response_token_counts": response_mask.sum(dim=-1)
        .detach()
        .cpu()
        .tolist(),
        "training_call_trace": list(expected_trace),
        "reference_kl_enabled": True,
    }


def _read_host_mem_available_bytes(path: Path = Path("/proc/meminfo")) -> int:
    try:
        for line in path.read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB":
                    value = int(fields[1]) * 1024
                    if value > 0:
                        return value
    except (OSError, UnicodeError, ValueError) as error:
        raise ServerAdapterError(f"cannot read host MemAvailable: {error}") from error
    raise ServerAdapterError("/proc/meminfo omitted a positive MemAvailable value")


def _finite_numeric_metric_values(value: object) -> list[float] | None:
    if isinstance(value, (str, bytes, bool)):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return [numeric] if math.isfinite(numeric) else None
    if isinstance(value, Sequence):
        flattened: list[float] = []
        for item in value:
            item_values = _finite_numeric_metric_values(item)
            if item_values is None:
                return None
            flattened.extend(item_values)
        return flattened or None
    return None


class _HostMemoryMonitor:
    """Poll MemAvailable from before worker startup until after Ray shutdown."""

    def __init__(
        self,
        *,
        loader: Callable[[], int] = _read_host_mem_available_bytes,
        poll_interval_s: float = 0.25,
        monitor_scope: str = "before_worker_start_through_after_ray_shutdown",
    ) -> None:
        if poll_interval_s <= 0:
            raise ValueError("host memory poll interval must be positive")
        if not monitor_scope:
            raise ValueError("host memory monitor scope must be non-empty")
        self.loader = loader
        self.poll_interval_s = poll_interval_s
        self.monitor_scope = monitor_scope
        self._minimum: int | None = None
        self._sample_count = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def sample(self) -> int:
        value = self.loader()
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ServerAdapterError("host MemAvailable loader returned an invalid value")
        with self._lock:
            self._minimum = value if self._minimum is None else min(self._minimum, value)
            self._sample_count += 1
        return value

    def _poll(self) -> None:
        while not self._stop.wait(self.poll_interval_s):
            try:
                self.sample()
            except BaseException as error:
                self._error = error
                self._stop.set()
                return

    def __enter__(self) -> _HostMemoryMonitor:
        self.sample()
        self._thread = threading.Thread(
            target=self._poll,
            name="capsule-host-memory-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        del exc_type, traceback
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                error = ServerAdapterError("host memory monitor did not stop")
                if isinstance(exc, BaseException):
                    _record_cleanup_error(exc, error)
                else:
                    raise error
        if self._error is not None:
            if isinstance(exc, BaseException):
                _record_cleanup_error(exc, self._error)
            else:
                raise self._error
        return False

    def evidence(self) -> dict[str, Any]:
        with self._lock:
            if self._minimum is None or self._sample_count < 1:
                raise ServerAdapterError("host memory monitor recorded no samples")
            return {
                "sample_count": self._sample_count,
                "minimum_mem_available_bytes": self._minimum,
                "poll_interval_s": self.poll_interval_s,
                "monitor_scope": self.monitor_scope,
            }


class GateRuntime(Protocol):
    def seed(self, seeds: tuple[int, ...], *, run_id: str) -> Mapping[str, Any]: ...

    def oracle(self, seed: int, replay_count: int, *, run_id: str) -> Mapping[str, Any]: ...

    def collector(
        self,
        p0_count: int,
        trajectories_per_p0: int,
        max_turns: int,
        *,
        run_id: str,
    ) -> Mapping[str, Any]: ...

    def guided(
        self,
        group_size: int,
        base_count: int,
        guided_count: int,
        max_group_attempts: int,
        *,
        run_id: str,
    ) -> Mapping[str, Any]: ...

    def trainer(
        self,
        optimizer_steps: int,
        group_rewards: tuple[int, ...],
        guided_artifact: Path,
        *,
        run_id: str,
    ) -> Mapping[str, Any]: ...


_VERIFIERS: dict[str, Callable[[Mapping[str, Any]], None]] = {
    "seed": verify_seed_gate_artifact,
    "oracle_replay": verify_oracle_gate_artifact,
    "collector": verify_collector_gate_artifact,
    "guided": verify_guided_gate_artifact,
    "trainer": verify_trainer_gate_artifact,
}


@dataclass
class GateTransaction(Mapping[str, Any]):
    """Gate evidence whose external resources remain rollback-capable until publication."""

    evidence: Mapping[str, Any]
    commit_callback: Callable[[], None]
    rollback_callback: Callable[[], None]
    _finished: bool = False

    def __getitem__(self, key: str) -> Any:
        return self.evidence[key]

    def __iter__(self):
        return iter(self.evidence)

    def __len__(self) -> int:
        return len(self.evidence)

    def commit(self) -> None:
        if self._finished:
            raise RuntimeError("gate transaction is already finished")
        self.commit_callback()
        self._finished = True

    def rollback(self) -> None:
        if self._finished:
            raise RuntimeError("gate transaction is already finished")
        self.rollback_callback()
        self._finished = True


@dataclass(frozen=True)
class _PublishedArtifactIdentity:
    device: int
    inode: int
    content: bytes
    stream: BinaryIO

    def close(self) -> None:
        self.stream.close()


def _capture_published_artifact(path: Path) -> _PublishedArtifactIdentity:
    """Capture the exact regular file created by this gate before committing resources."""

    if path.is_symlink() or not path.is_file():
        raise ServerAdapterError("published gate artifact is not a regular file")
    stream = path.open("rb")
    try:
        opened = os.fstat(stream.fileno())
        content = stream.read()
        current = path.stat()
        opened_identity = (opened.st_dev, opened.st_ino, opened.st_size)
        current_identity = (current.st_dev, current.st_ino, current.st_size)
        if opened_identity != current_identity or len(content) != current.st_size:
            raise ServerAdapterError("published gate artifact changed while capturing ownership")
        return _PublishedArtifactIdentity(
            device=opened.st_dev,
            inode=opened.st_ino,
            content=content,
            stream=stream,
        )
    except BaseException:
        stream.close()
        raise


def _remove_owned_published_artifact(
    path: Path, identity: _PublishedArtifactIdentity
) -> bool:
    """Remove only the unchanged inode and bytes published by this process."""

    try:
        if path.is_symlink():
            return False
        before = path.stat()
    except FileNotFoundError:
        return True
    if (before.st_dev, before.st_ino) != (identity.device, identity.inode):
        return False
    try:
        content = path.read_bytes()
        after = path.stat()
    except FileNotFoundError:
        return True
    if (
        (after.st_dev, after.st_ino) != (identity.device, identity.inode)
        or content != identity.content
    ):
        return False
    path.unlink()
    return True


def _record_cleanup_error(primary_error: BaseException, cleanup_error: BaseException) -> None:
    """Retain cleanup evidence without replacing the exception that caused unwinding."""

    try:
        existing = getattr(primary_error, "cleanup_errors", ())
        errors = existing if isinstance(existing, tuple) else ()
        setattr(primary_error, "cleanup_errors", (*errors, cleanup_error))
        if not hasattr(primary_error, "cleanup_error"):
            setattr(primary_error, "cleanup_error", cleanup_error)
    except BaseException:
        pass


def _run_cleanup_preserving_primary(
    cleanup: Callable[[], None], primary_error: BaseException | None
) -> None:
    try:
        cleanup()
    except BaseException as cleanup_error:
        if primary_error is None:
            raise
        _record_cleanup_error(primary_error, cleanup_error)


def _project_root(config: Mapping[str, Any]) -> Path:
    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ServerAdapterError("runtime must be a mapping")
    configured = runtime.get("project_root")
    repository_root = Path(__file__).resolve().parents[2]
    if isinstance(configured, str) and configured:
        path = Path(configured).expanduser()
        return path.resolve() if path.is_absolute() else (repository_root / path).resolve()
    return repository_root


def _project_path(config: Mapping[str, Any], value: object, field_name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ServerAdapterError(f"{field_name} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (_project_root(config) / path).resolve()


def _git_sha(project_root: Path) -> str:
    git_environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            env=git_environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ServerAdapterError(f"cannot read project Git SHA: {error}") from error
    value = completed.stdout.strip().lower()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ServerAdapterError(f"project Git returned an invalid SHA: {value!r}")
    try:
        top_level = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            env=git_environment,
        )
        if Path(top_level.stdout.strip()).resolve() != project_root.resolve():
            raise ServerAdapterError(
                "runtime.project_root must be the project Git worktree top level"
            )
        status = subprocess.run(
            [
                "git",
                "-C",
                str(project_root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            env=git_environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ServerAdapterError(f"cannot inspect project Git worktree: {error}") from error
    if status.stdout.strip():
        raise ServerAdapterError(
            "project checkout has staged, unstaged, or untracked files; commit or restore them "
            "before executing a runtime gate"
        )
    return value


def _runtime_mutation_watches(
    config: Mapping[str, Any],
    *,
    config_file: Path,
    command: str,
    args: argparse.Namespace,
) -> tuple[MutationWatch, ...]:
    """Resolve every immutable input consumed by a concrete runtime gate."""

    runtime = config.get("runtime")
    task = config.get("task")
    if not isinstance(runtime, Mapping):
        raise ServerAdapterError("runtime must be a mapping")
    if not isinstance(task, Mapping):
        raise ServerAdapterError("task must be a mapping")
    watches = [
        MutationWatch(config_file, "server config"),
        MutationWatch(runtime_dataset_path(config), "runtime dataset"),
        MutationWatch(
            _project_path(config, task.get("config_path"), "task.config_path"),
            "resolved environment config",
        ),
        MutationWatch(
            _project_path(
                config,
                runtime.get("verl_resolved_config_path"),
                "runtime.verl_resolved_config_path",
            ),
            "resolved VeRL config",
        ),
        MutationWatch(
            _project_path(
                config,
                runtime.get("program_model_path"),
                "runtime.program_model_path",
            ),
            "Program model tree",
            recursive=True,
        ),
        MutationWatch(
            _project_path(
                config,
                runtime.get("verl_source_path"),
                "runtime.verl_source_path",
            ),
            "VeRL source tree",
            recursive=True,
        ),
    ]
    if command == "trainer":
        guided_artifact = args.guided_artifact.expanduser().resolve()
        watches.append(MutationWatch(guided_artifact, "guided Gate 5 artifact"))
    return tuple(watches)


def _assert_runtime_inputs_unchanged(guard: PathMutationGuard) -> None:
    try:
        guard.assert_unchanged(context="during runtime gate")
    except StablePathError as error:
        raise ServerAdapterError(str(error)) from error


def _gate_name(command: str) -> str:
    return "oracle_replay" if command == "oracle" else command


def _checkpoint_run_slug(run_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "-", run_id).strip("._-") or "run"
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:10]
    return f"{readable[:64]}-{digest}"


def _validate_gate_request(
    *,
    config_path: Path,
    artifact_path: Path,
    args: argparse.Namespace,
    expected_git_sha_loader: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Validate and expand one gate request without constructing runtime dependencies."""

    if not isinstance(args.run_id, str) or not args.run_id.strip():
        raise ServerAdapterError("run_id must be non-empty")
    if artifact_path.exists():
        raise FileExistsError(f"artifact already exists: {artifact_path}")

    command = args.command
    if command == "seed":
        if args.seeds != (5, 6, 5):
            raise ServerAdapterError("seed gate requires the exact 5,6,5 sequence")
        return {"seed_sequence": list(args.seeds)}
    if command == "oracle":
        if args.seed != 5 or args.replays != 2:
            raise ServerAdapterError("oracle gate requires seed=5 and exactly two replays")
        return {"environment_seed": args.seed, "replay_count": args.replays}
    if command == "collector":
        values = (args.p0_count, args.trajectories, args.max_turns)
        if values != (2, 2, 12):
            raise ServerAdapterError("collector gate requires 2 P0 x 2 trajectories x 12 turns")
        return {
            "p0_count": args.p0_count,
            "trajectories_per_p0": args.trajectories,
            "max_controller_turns": args.max_turns,
        }
    if command == "guided":
        if (args.group_size, args.base_count, args.guided_count) != (8, 7, 1):
            raise ServerAdapterError("guided gate requires one 7+1 group")
        if args.max_group_attempts < 1:
            raise ServerAdapterError("max_group_attempts must be positive")
        return {
            "group_size": args.group_size,
            "base_count": args.base_count,
            "guided_count": args.guided_count,
            "max_group_attempts": args.max_group_attempts,
        }
    if command == "trainer":
        required_rewards = (0, 0, 0, 0, 0, 0, 0, 1)
        if args.optimizer_steps != 1 or args.group_rewards != required_rewards:
            raise ServerAdapterError("trainer gate requires exactly one verified 7+1 update")
        guided_artifact = args.guided_artifact.expanduser().resolve()
        if not guided_artifact.is_file():
            raise FileNotFoundError(f"guided artifact does not exist: {guided_artifact}")
        dependency = load_json_artifact(guided_artifact)
        verify_guided_gate_artifact(dependency)
        identity_fields = ("run_id", "config_sha256")
        expected_identity = (
            args.run_id,
            artifact_file_sha256(config_path),
        )
        if expected_git_sha_loader is not None:
            identity_fields += ("git_sha",)
            expected_identity += (expected_git_sha_loader(),)
        actual_identity = tuple(dependency.get(field) for field in identity_fields)
        if actual_identity != expected_identity:
            raise ServerAdapterError(
                f"guided artifact {'/'.join(identity_fields)} does not match the trainer gate"
            )
        return {
            "optimizer_steps": args.optimizer_steps,
            "group_rewards": list(args.group_rewards),
            "guided_artifact": str(guided_artifact),
            "guided_artifact_sha256": artifact_file_sha256(guided_artifact),
        }
    raise AssertionError(f"unhandled gate command: {command}")


def _dispatch(runtime: GateRuntime, command: str, args: argparse.Namespace) -> Mapping[str, Any]:
    if command == "seed":
        return runtime.seed(args.seeds, run_id=args.run_id)
    if command == "oracle":
        return runtime.oracle(args.seed, args.replays, run_id=args.run_id)
    if command == "collector":
        return runtime.collector(
            args.p0_count,
            args.trajectories,
            args.max_turns,
            run_id=args.run_id,
        )
    if command == "guided":
        return runtime.guided(
            args.group_size,
            args.base_count,
            args.guided_count,
            args.max_group_attempts,
            run_id=args.run_id,
        )
    if command == "trainer":
        return runtime.trainer(
            args.optimizer_steps,
            args.group_rewards,
            args.guided_artifact.resolve(),
            run_id=args.run_id,
        )
    raise AssertionError(f"unhandled gate command: {command}")


def execute_gate(
    *,
    config_path: str | Path,
    artifact_path: str | Path,
    command: str,
    run_id: str,
    runtime: GateRuntime | None,
    args: argparse.Namespace,
    config_loader: Callable[..., dict[str, Any]] | None = None,
    git_sha_loader: Callable[[Path], str] = _git_sha,
    runtime_factory: Callable[[Mapping[str, Any]], GateRuntime] | None = None,
) -> dict[str, Any]:
    """Execute one injected gate, verify its typed evidence, then publish it atomically."""

    config_file = Path(config_path).expanduser().resolve()
    artifact_file = Path(artifact_path).expanduser().resolve()
    failure_file = gate_failure_artifact_path(artifact_file)
    if artifact_file.exists():
        raise FileExistsError(f"artifact already exists: {artifact_file}")
    if failure_file.exists():
        raise FileExistsError(f"failure artifact already exists: {failure_file}")

    gate = _gate_name(command)
    if (runtime is None) == (runtime_factory is None):
        raise ValueError("provide exactly one of runtime or runtime_factory")
    config_sha256: str | None = None
    git_sha: str | None = None
    dataset_sha256: str | None = None
    dependency_hashes: dict[str, str] | None = None
    actor_identity: dict[str, Any] | None = None
    transaction: GateTransaction | None = None
    published_artifact: _PublishedArtifactIdentity | None = None
    mutation_guard: PathMutationGuard | None = None
    previous_dont_write_bytecode = sys.dont_write_bytecode
    stage = "config_hash"
    try:
        try:
            config_snapshot = read_stable_regular_file(
                config_file, label="server config"
            )
        except StablePathError as error:
            raise ServerAdapterError(str(error)) from error
        config_bytes = config_snapshot.raw_bytes
        config_sha256 = config_snapshot.sha256
        stage = "config_load"
        resolved_config_loader = load_and_validate_server_config_bytes
        if config_loader is not None:
            resolved_config_loader = config_loader
        config = resolved_config_loader(config_bytes, check_runtime_paths=True)
        stage = "runtime_mutation_guard"
        sys.dont_write_bytecode = True
        try:
            mutation_guard = PathMutationGuard.open(
                _runtime_mutation_watches(
                    config,
                    config_file=config_file,
                    command=command,
                    args=args,
                )
            )
        except StablePathError as error:
            raise ServerAdapterError(str(error)) from error
        stage = "dataset_hash"
        dataset_path = runtime_dataset_path(config)
        dataset_sha256 = artifact_file_sha256(dataset_path)
        stage = "runtime_dependency_hash"
        dependency_hashes = runtime_dependency_hashes(config)
        stage = "actor_identity_hash"
        try:
            actor_identity = build_actor_identity(config)
        except ActorIdentityError as error:
            raise ServerAdapterError(f"cannot bind Program actor identity: {error}") from error
        request_values = vars(args).copy()
        request_values.update(command=command, run_id=run_id)
        request_args = argparse.Namespace(**request_values)
        stage = "request_validation"
        _validate_gate_request(
            config_path=config_file,
            artifact_path=artifact_file,
            args=request_args,
        )
        stage = "pre_git"
        project_root = _project_root(config)
        git_sha = git_sha_loader(project_root)
        if command == "trainer":
            stage = "dependency_validation"
            dependency = load_json_artifact(args.guided_artifact.resolve())
            expected_identity = (
                run_id,
                config_sha256,
                git_sha,
                dataset_sha256,
                dependency_hashes["resolved_environment_sha256"],
                dependency_hashes["verl_resolved_config_sha256"],
                actor_identity["program_model_sha256"],
                actor_identity["actor_binding_sha256"],
            )
            actual_identity = tuple(
                dependency.get(field)
                for field in (
                    "run_id",
                    "config_sha256",
                    "git_sha",
                    "dataset_sha256",
                    "resolved_environment_sha256",
                    "verl_resolved_config_sha256",
                    "program_model_sha256",
                    "actor_binding_sha256",
                )
            )
            if actual_identity != expected_identity:
                raise ServerAdapterError(
                    "guided artifact run_id/config_sha256/git_sha/dataset_sha256/"
                    "resolved_environment_sha256/verl_resolved_config_sha256/"
                    "program_model_sha256/actor_binding_sha256 does not "
                    "match the trainer gate"
                )
        if runtime_factory is not None:
            stage = "runtime_construction"
            runtime = runtime_factory(config)
        assert runtime is not None
        stage = "runtime_dispatch"
        execution = _dispatch(runtime, command, args)
        transaction = execution if isinstance(execution, GateTransaction) else None
        evidence = execution.evidence if transaction is not None else execution
        stage = "post_git"
        post_git_sha = git_sha_loader(_project_root(config))
        if post_git_sha != git_sha:
            raise ServerAdapterError("project Git SHA changed while the gate was executing")
        stage = "post_dataset"
        if artifact_file_sha256(dataset_path) != dataset_sha256:
            raise ServerAdapterError(
                "runtime.dataset_path bytes changed while the gate was executing"
            )
        stage = "post_runtime_dependencies"
        post_dependency_hashes = runtime_dependency_hashes(config)
        for field_name, expected_sha256 in dependency_hashes.items():
            if post_dependency_hashes[field_name] != expected_sha256:
                label = (
                    "resolved environment"
                    if field_name == "resolved_environment_sha256"
                    else "resolved VeRL config"
                )
                raise ServerAdapterError(
                    f"{label} bytes changed while the gate was executing"
                )
        stage = "post_actor_identity"
        try:
            post_actor_identity = build_actor_identity(config)
            verify_actor_identity_payload(post_actor_identity, actor_identity)
        except ActorIdentityError as error:
            raise ServerAdapterError(
                f"Program actor identity changed while the gate was executing: {error}"
            ) from error
        stage = "evidence_validation"
        if not isinstance(evidence, Mapping):
            raise ServerAdapterError("gate runtime must return a mapping")
        forbidden = {
            "schema_version",
            "gate",
            "passed",
            "execution_mode",
            "run_id",
            "config_sha256",
            "git_sha",
            "dataset_sha256",
            "resolved_environment_sha256",
            "verl_resolved_config_sha256",
            "program_model_sha256",
            "actor_binding_sha256",
        }.intersection(evidence)
        if forbidden:
            raise ServerAdapterError(
                "gate runtime cannot override envelope field(s): "
                + ", ".join(sorted(forbidden))
            )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "gate": gate,
            "passed": True,
            "execution_mode": CANONICAL_EXECUTION_MODE,
            "run_id": run_id,
            "config_sha256": config_sha256,
            "git_sha": git_sha,
            "dataset_sha256": dataset_sha256,
            **dependency_hashes,
            "program_model_sha256": actor_identity["program_model_sha256"],
            "actor_binding_sha256": actor_identity["actor_binding_sha256"],
            **dict(evidence),
        }
        stage = "artifact_verification"
        _VERIFIERS[gate](payload)
        stage = "post_config"
        if artifact_file_sha256(config_file) != config_sha256:
            raise ServerAdapterError("server config bytes changed while the gate was executing")
        stage = "runtime_mutation_guard"
        _assert_runtime_inputs_unchanged(mutation_guard)
        stage = "artifact_publish"
        atomic_write_json(artifact_file, payload)
        if transaction is not None:
            published_artifact = _capture_published_artifact(artifact_file)
            stage = "transaction_commit"
            transaction.commit()
            committed_artifact = published_artifact
            published_artifact = None
            try:
                committed_artifact.close()
            except BaseException:
                # A read-only ownership handle is no longer transactional after commit.
                pass
    except BaseException as error:
        if stage == "transaction_commit" and published_artifact is not None:
            try:
                removed = _remove_owned_published_artifact(
                    artifact_file, published_artifact
                )
                if not removed:
                    try:
                        setattr(error, "success_artifact_cleanup_refused", True)
                    except BaseException:
                        pass
            except BaseException as cleanup_error:
                _record_cleanup_error(error, cleanup_error)
        if published_artifact is not None:
            try:
                published_artifact.close()
            except BaseException as cleanup_error:
                _record_cleanup_error(error, cleanup_error)
        rollback_error: BaseException | None = None
        if transaction is not None:
            try:
                transaction.rollback()
            except BaseException as caught_rollback_error:
                rollback_error = caught_rollback_error
        try:
            write_gate_failure_artifact(
                artifact_file,
                gate=gate,
                run_id=run_id,
                config_sha256=config_sha256,
                git_sha=git_sha,
                dataset_sha256=dataset_sha256,
                error=error,
                stage=stage,
                rollback_error=rollback_error,
            )
        except BaseException as recording_error:
            try:
                setattr(error, "failure_artifact_recording_error", recording_error)
            except BaseException:
                pass
        raise
    finally:
        if mutation_guard is not None:
            mutation_guard.close()
        sys.dont_write_bytecode = previous_dont_write_bytecode
    return payload


@dataclass
class _CollectionSession:
    task: TaskInstanceV1
    workers: Any
    generator: Any
    clean_evaluator: Any
    evaluator: Any
    repair_collector: Any
    assembler: Any
    group_encoder: Any

    def close(self) -> None:
        _close_collection_resources(self.repair_collector, self.evaluator, self.workers)


def _close_collection_resources(*resources: Any | None) -> None:
    """Attempt every close operation and report the first cleanup failure afterwards."""

    first_error: BaseException | None = None
    for resource in resources:
        if resource is None:
            continue
        close = getattr(resource, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except BaseException as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


class ConcreteGateRuntime:
    """Heavy, server-only implementation of gates 2 through 6."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config

    def _environment_config_path(self) -> Path:
        task = self.config.get("task")
        if not isinstance(task, Mapping):
            raise ServerAdapterError("task must be a mapping")
        return _project_path(self.config, task.get("config_path"), "task.config_path")

    def _tasks(self) -> tuple[TaskInstanceV1, ...]:
        from capx.rl.capsule.server_factory import resolve_task_instances

        return resolve_task_instances(self.config)

    @staticmethod
    def _reset_evidence(result: ProgramReplayResultV1) -> tuple[bool, bool]:
        reset_info = result.diagnostics.get("reset_info")
        reset_evidence = (
            reset_info.get("capsule_reset_evidence")
            if isinstance(reset_info, Mapping)
            else None
        )
        if not isinstance(reset_evidence, Mapping):
            raise ServerAdapterError(
                "oracle replay omitted reset_info.capsule_reset_evidence"
            )
        namespace_fresh = reset_evidence.get("namespace_fresh")
        api_state_cleared = reset_evidence.get("api_state_cleared")
        reset_count = reset_evidence.get("api_reset_count")
        confirmed_count = reset_evidence.get("api_reset_confirmed_count")
        if (
            namespace_fresh is not True
            or api_state_cleared is not True
            or isinstance(reset_count, bool)
            or not isinstance(reset_count, int)
            or reset_count < 1
            or confirmed_count != reset_count
        ):
            raise ServerAdapterError(
                "oracle replay reset evidence did not prove namespace/API cleanup"
            )
        return namespace_fresh, api_state_cleared

    def _task_for_seed(self, seed: int) -> TaskInstanceV1:
        matches = [task for task in self._tasks() if task.environment_seed == seed]
        if len(matches) != 1:
            raise ServerAdapterError(
                f"expected exactly one seed-resolved task for environment seed {seed}; "
                f"found {len(matches)}"
            )
        return matches[0]

    def _open_collection_session(self, task: TaskInstanceV1) -> _CollectionSession:
        from capx.rl.capsule.controller import (
            ControllerRepairCollector,
            OpenAICompatibleControllerTransport,
        )
        from capx.rl.capsule.evaluator import (
            CandidateCleanReplayAdapter,
            CleanReplayEvaluator,
            PersistentProcessReplayBackend,
        )
        from capx.rl.capsule.group import CapsuleGroupAssembler
        from capx.rl.capsule.server_factory import (
            ActorBaseSampler,
            ActorRevisionGenerator,
            VeRLGroupEncoder,
            VeRLProgramGenerator,
            YamlEnvironmentFactory,
            start_verl_workers,
        )

        capsule = self.config["capsule"]
        program = self.config["program_service"]
        workers = start_verl_workers(self.config)
        evaluator = None
        repair_collector = None
        try:
            backend = PersistentProcessReplayBackend(
                YamlEnvironmentFactory(str(self._environment_config_path()))
            )
            evaluator = CleanReplayEvaluator(backend)
            clean_evaluator = CandidateCleanReplayAdapter(evaluator)
            system_prompt = str(
                program.get(
                    "system_prompt",
                    "Generate only one complete independently executable Python robot program.",
                )
            )
            generator = VeRLProgramGenerator(
                actor_rollout_wg=workers.actor_rollout_wg,
                tokenizer=workers.tokenizer,
                data_proto_factory=workers.data_proto_factory,
                prompt_token_limit=int(capsule["revision_input_max_tokens"]),
                response_token_limit=int(capsule["revision_response_max_tokens"]),
                system_prompt=system_prompt,
            )
            controller_config = _controller_runtime_config(self.config)
            repair_collector = ControllerRepairCollector(
                transport=OpenAICompatibleControllerTransport(controller_config),
                max_turns=controller_config.max_turns,
            )
            assembler = CapsuleGroupAssembler(
                base_sampler=ActorBaseSampler(generator),
                repair_collector=repair_collector,
                revision_generator=ActorRevisionGenerator(generator),
                clean_evaluator=clean_evaluator,
                revision_prompt_token_counter=generator.count_prompt_tokens,
                revision_response_token_counter=generator.count_raw_response_tokens,
                revision_input_token_limit=int(capsule["revision_input_max_tokens"]),
                revision_response_token_limit=int(capsule["revision_response_max_tokens"]),
            )
            group_encoder = VeRLGroupEncoder(
                tokenizer=workers.tokenizer,
                data_proto_factory=workers.data_proto_factory,
                prompt_token_limit=int(capsule["revision_input_max_tokens"]),
                response_token_limit=int(capsule["revision_response_max_tokens"]),
                system_prompt=system_prompt,
            )
            return _CollectionSession(
                task=task,
                workers=workers,
                generator=generator,
                clean_evaluator=clean_evaluator,
                evaluator=evaluator,
                repair_collector=repair_collector,
                assembler=assembler,
                group_encoder=group_encoder,
            )
        except BaseException:
            try:
                _close_collection_resources(repair_collector, evaluator, workers)
            except BaseException:
                pass
            raise

    def seed(self, seeds: tuple[int, ...], *, run_id: str) -> Mapping[str, Any]:
        del run_id
        from capx.rl.capsule.server_factory import YamlEnvironmentFactory

        if seeds != (5, 6, 5):
            raise ServerAdapterError("seed gate requires the exact 5,6,5 sequence")
        environment_factory = YamlEnvironmentFactory(
            str(self._environment_config_path())
        )
        environment = environment_factory(None)  # type: ignore[arg-type]
        hashes: list[str] = []
        try:
            for seed in seeds:
                _observation, info = environment.reset(
                    seed=seed, options={"capsule_gate": "seed"}
                )
                initial_hash = (
                    info.get("initial_state_sha256")
                    if isinstance(info, Mapping)
                    else None
                )
                if not isinstance(initial_hash, str):
                    raise ServerAdapterError(
                        f"environment reset for seed {seed} omitted initial_state_sha256"
                    )
                hashes.append(initial_hash)
        finally:
            close = getattr(environment, "close", None)
            if callable(close):
                close()
        return {"seeds": list(seeds), "initial_state_sha256": hashes}

    def oracle(self, seed: int, replay_count: int, *, run_id: str) -> Mapping[str, Any]:
        del run_id
        from capx.rl.capsule.evaluator import CleanReplayEvaluator, PersistentProcessReplayBackend
        from capx.rl.capsule.server_factory import YamlEnvironmentFactory

        if seed != 5 or replay_count != 2:
            raise ServerAdapterError("oracle gate requires seed=5 and exactly two replays")
        task = self._task_for_seed(seed)
        factory = YamlEnvironmentFactory(str(self._environment_config_path()))
        probe = factory(task)
        try:
            oracle_source = getattr(probe, "oracle_code", None)
            if not isinstance(oracle_source, str) or not oracle_source.strip():
                raise ServerAdapterError(
                    "configured environment does not expose oracle_code"
                )
        finally:
            close = getattr(probe, "close", None)
            if callable(close):
                close()

        backend = PersistentProcessReplayBackend(factory)
        evaluator = CleanReplayEvaluator(backend)
        records: list[dict[str, Any]] = []
        try:
            for _ in range(replay_count):
                result = evaluator.evaluate_program(
                    task,
                    oracle_source,
                    seed,
                    program_sample_id=f"{task.task_id}:seed-{seed}:oracle",
                )
                pid = backend.worker_pid
                if isinstance(pid, bool) or not isinstance(pid, int):
                    raise ServerAdapterError("clean replay backend did not expose a worker PID")
                namespace_fresh, api_state_cleared = self._reset_evidence(result)
                records.append(
                    {
                        "worker_id": f"pid:{pid}",
                        "reset_seed": seed,
                        "namespace_fresh": namespace_fresh,
                        "api_state_cleared": api_state_cleared,
                        "watchdog_active": evaluator.timeout_s > 0,
                        "result": result.to_dict(),
                    }
                )
        finally:
            evaluator.close()
        replay_results = tuple(
            ProgramReplayResultV1.from_dict(record["result"]) for record in records
        )
        return {
            "direct_replay": True,
            "controller_used": False,
            "replays": records,
            **summarize_replay_results(replay_results, require_attempt_history=True),
        }

    def collector(
        self,
        p0_count: int,
        trajectories_per_p0: int,
        max_turns: int,
        *,
        run_id: str,
    ) -> Mapping[str, Any]:
        from capx.rl.capsule.group import CapsuleGroupAssembler, ProgramCandidate

        if (p0_count, trajectories_per_p0, max_turns) != (2, 2, 12):
            raise ServerAdapterError("collector gate requires 2 P0 x 2 trajectories x 12 turns")
        task = self._task_for_seed(5)
        session = self._open_collection_session(task)
        primary_error: BaseException | None = None
        try:
            candidates: list[ProgramCandidate] | None = None
            results: list[ProgramReplayResultV1] | None = None
            selected_batch_index: int | None = None
            replay_events: list[dict[str, Any]] = []
            discarded_batches: list[dict[str, Any]] = []
            for batch_index in range(20):
                batch_candidates: list[ProgramCandidate] = []
                batch_results: list[ProgramReplayResultV1] = []
                for base_index in range(7):
                    candidate = session.generator.generate(
                        task.prompt,
                        f"{run_id}:collector-batch-{batch_index}:base-{base_index}",
                    )
                    result = session.clean_evaluator(task, candidate)
                    batch_candidates.append(candidate)
                    batch_results.append(result)
                unknown_outcomes = {ReplayOutcome.INFRA_ERROR, ReplayOutcome.EVALUATOR_ERROR}
                accepted = all(
                    result.outcome is not ReplayOutcome.SUCCESS
                    and result.outcome not in unknown_outcomes
                    and result.binary_reward == 0
                    for result in batch_results
                )
                for base_index, result in enumerate(batch_results):
                    replay_events.append(
                        {
                            "batch_index": batch_index,
                            "base_index": base_index,
                            "selected_batch": accepted,
                            "result": result.to_dict(),
                        }
                    )
                if accepted:
                    candidates, results = batch_candidates, batch_results
                    selected_batch_index = batch_index
                    break
                if any(result.outcome in unknown_outcomes for result in batch_results):
                    reason = "unknown_replay_reward"
                elif any(result.outcome is ReplayOutcome.SUCCESS for result in batch_results):
                    reason = "batch_contains_success"
                else:
                    reason = "batch_not_all_semantic_failure"
                discarded_batches.append(
                    {
                        "batch_index": batch_index,
                        "reason": reason,
                        "results": [result.to_dict() for result in batch_results],
                    }
                )
            if candidates is None or results is None:
                raise ServerAdapterError(
                    "collector could not obtain one all-failed seven-base batch"
                )
            assert selected_batch_index is not None
            selected_indices = CapsuleGroupAssembler._select_p0_indices(candidates, results)
            selected_candidates = [candidates[index] for index in selected_indices]
            selected_results = [results[index] for index in selected_indices]
            traces: list[dict[str, Any]] = []
            for p0_rank, (candidate, result) in enumerate(
                zip(selected_candidates, selected_results, strict=True)
            ):
                for trajectory_index in range(2):
                    trajectory_id = (
                        f"{run_id}:collector:p0-{p0_rank}:trajectory-{trajectory_index}"
                    )
                    trace = session.repair_collector(
                        task,
                        candidate,
                        result,
                        p0_rank,
                        trajectory_index,
                        trajectory_id,
                    )
                    traces.append(
                        {
                            "p0_rank": p0_rank,
                            "trajectory_index": trajectory_index,
                            "trace": trace.to_dict(),
                        }
                    )
            return {
                "controller_frozen": True,
                "intermediate_replay_count": 0,
                "p0_count": 2,
                "repair_trajectories_per_p0": 2,
                "base_results": [result.to_dict() for result in selected_results],
                "selected_batch_index": selected_batch_index,
                "selected_batch_results": [result.to_dict() for result in results],
                "discarded_batches": discarded_batches,
                "replay_events": replay_events,
                "repair_traces": traces,
                **summarize_replay_results(
                    tuple(
                        ProgramReplayResultV1.from_dict(event["result"])
                        for event in replay_events
                    ),
                    require_attempt_history=True,
                ),
            }
        except BaseException as error:
            primary_error = error
            raise
        finally:
            _run_cleanup_preserving_primary(session.close, primary_error)

    def guided(
        self,
        group_size: int,
        base_count: int,
        guided_count: int,
        max_group_attempts: int,
        *,
        run_id: str,
    ) -> Mapping[str, Any]:
        from capx.rl.capsule.group import GroupDiscarded

        if (group_size, base_count, guided_count) != (8, 7, 1):
            raise ServerAdapterError("guided gate requires one 7+1 group")
        if max_group_attempts < 1:
            raise ServerAdapterError("max_group_attempts must be positive")
        task = self._task_for_seed(5)
        session = self._open_collection_session(task)
        primary_error: BaseException | None = None
        try:
            assembly = None
            drain_history = getattr(session.clean_evaluator, "drain_history", None)
            if not callable(drain_history):
                raise ServerAdapterError(
                    "guided gate clean evaluator does not expose typed replay history"
                )
            if getattr(session.assembler, "clean_evaluator", None) is not session.clean_evaluator:
                raise ServerAdapterError(
                    "guided assembler and audit recorder do not share the clean evaluator"
                )
            drain_history()
            replay_events: list[dict[str, Any]] = []
            discarded_group_attempts: list[dict[str, Any]] = []
            selected_group_attempt_index: int | None = None
            for group_attempt_index in range(max_group_attempts):
                try:
                    candidate = session.assembler.assemble(task)
                except GroupDiscarded as error:
                    attempt_results = drain_history()
                    for result_index, result in enumerate(attempt_results):
                        replay_events.append(
                            {
                                "group_attempt_index": group_attempt_index,
                                "result_index": result_index,
                                "selected_group": False,
                                "result": result.to_dict(),
                            }
                        )
                    discarded_group_attempts.append(
                        {
                            "group_attempt_index": group_attempt_index,
                            "reason": error.reason,
                            "message": str(error),
                            "replay_results": [result.to_dict() for result in attempt_results],
                            "partial_repair_attempts": [
                                attempt.to_dict()
                                for attempt in error.partial_repair_attempts
                            ],
                            "assembly": None,
                        }
                    )
                    continue
                attempt_results = drain_history()
                selected = candidate.group.metadata.get("guided_member_selected") is True
                for result_index, result in enumerate(attempt_results):
                    replay_events.append(
                        {
                            "group_attempt_index": group_attempt_index,
                            "result_index": result_index,
                            "selected_group": selected,
                            "result": result.to_dict(),
                        }
                    )
                if selected:
                    assembly = candidate
                    selected_group_attempt_index = group_attempt_index
                    break
                discarded_group_attempts.append(
                    {
                        "group_attempt_index": group_attempt_index,
                        "reason": "no_guided_member",
                        "message": "assembled fallback group had no PT/P_hat double-success",
                        "replay_results": [result.to_dict() for result in attempt_results],
                        "partial_repair_attempts": [],
                        "assembly": candidate.to_dict(),
                    }
                )
            if assembly is None:
                raise ServerAdapterError(
                    f"no PT/P_hat double-success group after {max_group_attempts} attempts"
                )
            selected = next(attempt for attempt in assembly.repair_attempts if attempt.selected)
            if (
                selected.trace is None
                or selected.pt_result is None
                or selected.revision_result is None
            ):
                raise ServerAdapterError("selected guided repair is missing typed provenance")
            p0_result = next(
                result
                for result in assembly.base_results
                if result.program_sample_id == selected.p0_program_sample_id
            )
            return {
                "task_instance": task.to_dict(),
                "original_prompt": task.prompt,
                "training_input_contains_critique": False,
                "learning_group": assembly.group.to_dict(),
                "base_results": [result.to_dict() for result in assembly.base_results[:7]],
                "repair_attempts": [attempt.to_dict() for attempt in assembly.repair_attempts],
                "selected_group_attempt_index": selected_group_attempt_index,
                "discarded_group_attempts": discarded_group_attempts,
                "replay_events": replay_events,
                "selected_repair": {
                    "p0_rank": selected.p0_rank,
                    "trajectory_index": selected.trajectory_index,
                    "trace": selected.trace.to_dict(),
                    "p0_result": p0_result.to_dict(),
                    "pt_result": selected.pt_result.to_dict(),
                    "p_hat_result": selected.revision_result.to_dict(),
                },
                **summarize_replay_results(
                    tuple(
                        ProgramReplayResultV1.from_dict(event["result"])
                        for event in replay_events
                    ),
                    require_attempt_history=True,
                ),
            }
        except BaseException as error:
            primary_error = error
            raise
        finally:
            _run_cleanup_preserving_primary(session.close, primary_error)

    @staticmethod
    def _actor_metrics(actor_output: Any) -> dict[str, float]:
        candidates: list[object] = [actor_output]
        meta_info = getattr(actor_output, "meta_info", None)
        if isinstance(meta_info, Mapping):
            candidates.extend((meta_info, meta_info.get("metrics")))
        if isinstance(actor_output, Mapping):
            candidates.append(actor_output.get("metrics"))
        metrics: dict[str, float] = {}
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            for key, value in candidate.items():
                if not isinstance(key, str):
                    continue
                numeric_values = _finite_numeric_metric_values(value)
                if numeric_values:
                    metrics[key] = sum(numeric_values) / len(numeric_values)
        if not metrics:
            raise ServerAdapterError("actor update returned no finite numeric metrics")
        return metrics

    def trainer(
        self,
        optimizer_steps: int,
        group_rewards: tuple[int, ...],
        guided_artifact: Path,
        *,
        run_id: str,
    ) -> Mapping[str, Any]:
        from capx.rl.capsule.checkpoint import AtomicCheckpointClaim
        from capx.rl.capsule.group import GroupAssemblyResult
        from capx.rl.capsule.server_factory import VeRLGroupEncoder, start_verl_workers
        from capx.rl.capsule.trainer import CapsuleCritiqueRayTrainer, MemoryArtifactSink

        if optimizer_steps != 1 or group_rewards != (0, 0, 0, 0, 0, 0, 0, 1):
            raise ServerAdapterError("trainer gate requires exactly one verified 7+1 update")
        guided_payload = load_json_artifact(guided_artifact)
        verify_guided_gate_artifact(guided_payload)
        if guided_payload.get("run_id") != run_id:
            raise ServerAdapterError("trainer guided artifact must belong to the same run_id")
        assembly = GroupAssemblyResult.from_dict(
            {
                "group": guided_payload["learning_group"],
                "base_results": guided_payload["base_results"],
                "repair_attempts": guided_payload["repair_attempts"],
            }
        )
        group = assembly.group
        task_payload = guided_payload.get("task_instance")
        if not isinstance(task_payload, Mapping):
            raise ServerAdapterError("guided artifact omitted its typed task_instance")
        try:
            task = TaskInstanceV1.from_dict(task_payload)
        except (KeyError, TypeError, ValueError) as error:
            raise ServerAdapterError(f"guided task_instance is invalid: {error}") from error
        configured_task = self._task_for_seed(5)
        if task != configured_task:
            raise ServerAdapterError(
                "guided task_instance does not exactly match the configured seed-5 task"
            )
        if (
            group.task_id != task.task_id
            or group.environment_seed != task.environment_seed
            or group.initial_state_sha256 != task.initial_state_sha256
        ):
            raise ServerAdapterError("guided learning group does not match its typed task_instance")
        output_dir = _project_path(
            self.config,
            self.config["runtime"]["output_dir"],
            "runtime.output_dir",
        )
        checkpoint = (
            output_dir
            / "gate06"
            / _checkpoint_run_slug(run_id)
            / "global_step_1"
            / "actor"
        ).resolve()
        checkpoint_claim_root = checkpoint.parents[1]

        class _StaticAssembler:
            def assemble(self, requested_task: TaskInstanceV1) -> GroupAssemblyResult:
                if requested_task != task:
                    raise ServerAdapterError("trainer requested an unexpected task")
                return assembly

        checkpoint_claim = AtomicCheckpointClaim(
            checkpoint, claim_root=checkpoint_claim_root
        )
        checkpoint_claim.__enter__()
        try:
            memory_monitor = _HostMemoryMonitor()
            memory_monitor.__enter__()
            monitor_primary_error: BaseException | None = None
            try:
                workers = start_verl_workers(self.config)
                primary_error: BaseException | None = None
                try:
                    memory_monitor.sample()
                    verl_provenance_before = workers.verl_provenance()
                    lora_runtime_before = workers.lora_runtime_evidence()
                    reference_policy_mode = getattr(
                        workers, "reference_policy_mode", None
                    )
                    if reference_policy_mode != "actor_base_adapter_disabled":
                        raise ServerAdapterError(
                            "LoRA Gate 6 requires reference_policy_mode="
                            "actor_base_adapter_disabled"
                        )
                    capsule = self.config["capsule"]
                    program = self.config["program_service"]
                    system_prompt = str(
                        program.get(
                            "system_prompt",
                            "Generate only one complete independently executable Python robot "
                            "program.",
                        )
                    )
                    encoder = VeRLGroupEncoder(
                        tokenizer=workers.tokenizer,
                        data_proto_factory=workers.data_proto_factory,
                        prompt_token_limit=int(capsule["revision_input_max_tokens"]),
                        response_token_limit=int(capsule["revision_response_max_tokens"]),
                        system_prompt=system_prompt,
                    )
                    sink = MemoryArtifactSink()
                    trainer = CapsuleCritiqueRayTrainer(
                        assembler=_StaticAssembler(),
                        batch_encoder=encoder,
                        actor_rollout_wg=workers.actor_rollout_wg,
                        ref_policy_wg=workers.ref_policy_wg,
                        reference_policy_mode=reference_policy_mode,
                        artifact_sink=sink,
                        config=self.config,
                    )
                    optimizer_step_before = workers.optimizer_step()
                    result = trainer.run_step(task)
                    training_tensor_evidence = _derive_training_tensor_evidence(
                        result, group
                    )
                    memory_monitor.sample()
                    actor_update_rpcs = getattr(trainer, "actor_updates_completed", None)
                    if actor_update_rpcs != 1 or isinstance(actor_update_rpcs, bool):
                        raise ServerAdapterError(
                            "trainer did not complete exactly one actor update RPC"
                        )
                    optimizer_step_after = workers.optimizer_step()
                    optimizer_step_delta = optimizer_step_after - optimizer_step_before
                    if optimizer_step_delta != 1:
                        raise ServerAdapterError(
                            "actor optimizer step did not advance exactly once; "
                            f"before={optimizer_step_before}, after={optimizer_step_after}"
                        )
                    metrics = self._actor_metrics(result.actor_output)
                    grad_values = [
                        value
                        for key, value in metrics.items()
                        if "grad" in key.lower() and "norm" in key.lower()
                    ]
                    if not grad_values or not any(value > 0 for value in grad_values):
                        raise ServerAdapterError(
                            "actor metrics contain no positive finite gradient norm: "
                            f"{json.dumps(metrics, sort_keys=True)}"
                        )
                    gradient_norm = max(grad_values)
                    rollout_mode = getattr(workers, "rollout_mode", None)
                    ppo_epochs = getattr(workers, "ppo_epochs", None)
                    ppo_mini_batch_size = getattr(workers, "ppo_mini_batch_size", None)
                    reference_kl_coef = getattr(workers, "kl_loss_coef", None)
                    data_parallel_world_size = getattr(
                        workers, "data_parallel_world_size", None
                    )
                    sequence_parallel_size = getattr(
                        workers, "sequence_parallel_size", None
                    )
                    if (
                        rollout_mode != "sync"
                        or isinstance(ppo_epochs, bool)
                        or ppo_epochs != 1
                        or isinstance(ppo_mini_batch_size, bool)
                        or ppo_mini_batch_size != 8
                    ):
                        raise ServerAdapterError(
                            "trainer worker must use sync rollout, ppo_epochs=1, and "
                            "ppo_mini_batch_size=8"
                        )
                    if (
                        isinstance(reference_kl_coef, bool)
                        or not isinstance(reference_kl_coef, (int, float))
                        or not math.isfinite(float(reference_kl_coef))
                        or reference_kl_coef <= 0
                    ):
                        raise ServerAdapterError(
                            "trainer worker reference KL coefficient must be positive"
                        )
                    if (
                        isinstance(data_parallel_world_size, bool)
                        or not isinstance(data_parallel_world_size, int)
                        or data_parallel_world_size < 1
                        or 8 % data_parallel_world_size != 0
                    ):
                        raise ServerAdapterError(
                            "trainer FSDP data-parallel world size must be a positive divisor "
                            "of 8"
                        )
                    if sequence_parallel_size != 1 or isinstance(
                        sequence_parallel_size, bool
                    ):
                        raise ServerAdapterError(
                            "trainer must disable Ulysses sequence parallelism"
                        )
                    checkpoint_evidence = checkpoint_claim.publish(
                        lambda staging: workers.save_checkpoint(
                            staging, optimizer_step_after
                        ),
                        optimizer_step_before=optimizer_step_before,
                        optimizer_step_after=optimizer_step_after,
                    )
                    memory_monitor.sample()
                    lora_runtime_after = workers.lora_runtime_evidence()
                    stable_lora_fields = (
                        "lora_rank",
                        "lora_alpha",
                        "lora_target_modules",
                        "worker_count",
                        "worker_ranks",
                        "trainable_parameter_name_sha256s",
                        "total_parameter_count",
                        "trainable_parameter_count",
                        "non_lora_trainable_parameter_count",
                        "only_lora_trainable",
                        "lora_layer_count",
                        "lora_projection_suffixes",
                        "lora_tensor_count_per_worker",
                    )
                    if any(
                        lora_runtime_after.get(field) != lora_runtime_before.get(field)
                        for field in stable_lora_fields
                    ):
                        raise ServerAdapterError(
                            "actor LoRA trainability evidence changed during Gate 6"
                        )
                    verl_provenance_after = workers.verl_provenance()
                    if verl_provenance_after != verl_provenance_before:
                        raise ServerAdapterError(
                            "VeRL driver/worker provenance changed during the trainer gate"
                        )
                    gate_evidence: dict[str, Any] = {
                        "learning_group": group.to_dict(),
                        "actor_update_rpcs": actor_update_rpcs,
                        "optimizer_steps": optimizer_step_delta,
                        "optimizer_step_before": optimizer_step_before,
                        "optimizer_step_after": optimizer_step_after,
                        "gradient_norm": gradient_norm,
                        "group_rewards": list(group_rewards),
                        **training_tensor_evidence,
                        "rollout_is": False,
                        "norm_adv_by_std_in_grpo": False,
                        "loss_mode": "capsule_critique",
                        "capsule_gamma": 0.1,
                        "reference_kl_coef": float(reference_kl_coef),
                        "reference_policy_mode": reference_policy_mode,
                        "rollout_mode": rollout_mode,
                        "ppo_epochs": ppo_epochs,
                        "ppo_mini_batch_size": ppo_mini_batch_size,
                        "data_parallel_world_size": data_parallel_world_size,
                        "sequence_parallel_size": sequence_parallel_size,
                        "metrics": metrics,
                        "actor_update_skipped": result.skipped_actor_update,
                        "checkpoint": str(checkpoint_evidence.path),
                        "checkpoint_file_count": checkpoint_evidence.file_count,
                        "checkpoint_sha256": checkpoint_evidence.sha256,
                        "checkpoint_manifest": str(checkpoint_evidence.manifest_path),
                        "guided_artifact_sha256": artifact_file_sha256(guided_artifact),
                        "verl_provenance_before": verl_provenance_before,
                        "verl_provenance_after": verl_provenance_after,
                        "lora_runtime_before": lora_runtime_before,
                        "lora_runtime_after": lora_runtime_after,
                        "cuda_peak_reserved_bytes": max(
                            int(lora_runtime_before["cuda_peak_reserved_bytes"]),
                            int(lora_runtime_after["cuda_peak_reserved_bytes"]),
                        ),
                    }
                except BaseException as error:
                    primary_error = error
                    raise
                finally:
                    _run_cleanup_preserving_primary(workers.close, primary_error)
                ray_release = workers.ray_release_evidence()
                memory_monitor.sample()
            except BaseException as error:
                monitor_primary_error = error
                raise
            finally:
                memory_monitor.__exit__(
                    type(monitor_primary_error) if monitor_primary_error is not None else None,
                    monitor_primary_error,
                    monitor_primary_error.__traceback__
                    if monitor_primary_error is not None
                    else None,
                )
            host_memory = memory_monitor.evidence()
            if host_memory["minimum_mem_available_bytes"] < 12 * 1024**3:
                raise ServerAdapterError(
                    "host MemAvailable fell below the 12 GiB Gate 6 runtime limit"
                )
            adapter_evidence = direct_lora_adapter_evidence(checkpoint_evidence.path)
            gate_evidence.update(adapter_evidence)
            gate_evidence["ray_release"] = ray_release
            gate_evidence["host_memory"] = host_memory
            return GateTransaction(
                evidence=gate_evidence,
                commit_callback=checkpoint_claim.commit,
                rollback_callback=checkpoint_claim.abort,
            )
        except BaseException as error:
            _run_cleanup_preserving_primary(checkpoint_claim.abort, error)
            raise


def _comma_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must contain comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("value must contain at least one integer")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute one concrete Capsule-RL server gate.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser("seed")
    seed.add_argument("--seeds", type=_comma_ints, default=(5, 6, 5))

    oracle = subparsers.add_parser("oracle")
    oracle.add_argument("--seed", type=int, default=5)
    oracle.add_argument("--replays", type=int, default=2)

    collector = subparsers.add_parser("collector")
    collector.add_argument("--p0-count", type=int, default=2)
    collector.add_argument("--trajectories", type=int, default=2)
    collector.add_argument("--max-turns", type=int, default=12)

    guided = subparsers.add_parser("guided")
    guided.add_argument("--group-size", type=int, default=8)
    guided.add_argument("--base-count", type=int, default=7)
    guided.add_argument("--guided-count", type=int, default=1)
    guided.add_argument("--max-group-attempts", type=int, default=20)

    trainer = subparsers.add_parser("trainer")
    trainer.add_argument("--optimizer-steps", type=int, default=1)
    trainer.add_argument(
        "--group-rewards", type=_comma_ints, default=(0, 0, 0, 0, 0, 0, 0, 1)
    )
    trainer.add_argument("--guided-artifact", type=Path, required=True)
    return parser


def validate_cli_request(
    argv: Sequence[str] | None = None,
    *,
    git_sha_loader: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    """Validate one adapter CLI request without constructing a gate runtime."""

    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    artifact_path = args.artifact.expanduser().resolve()
    config = load_and_validate_server_config(config_path, check_runtime_paths=True)
    resolved_git_sha_loader = _git_sha if git_sha_loader is None else git_sha_loader
    request = _validate_gate_request(
        config_path=config_path,
        artifact_path=artifact_path,
        args=args,
        expected_git_sha_loader=lambda: resolved_git_sha_loader(_project_root(config)),
    )
    return {
        "mode": "VALIDATION ONLY",
        "gate": _gate_name(args.command),
        "run_id": args.run_id,
        "artifact": str(artifact_path),
        "config_sha256": artifact_file_sha256(config_path),
        "request": request,
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_factory: Callable[[Mapping[str, Any]], GateRuntime] = ConcreteGateRuntime,
    git_sha_loader: Callable[[Path], str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    artifact_path = args.artifact.expanduser().resolve()
    config = load_and_validate_server_config(config_path, check_runtime_paths=True)
    validate_only = bool(args.validate_only or args.dry_run)
    resolved_git_sha_loader = _git_sha if git_sha_loader is None else git_sha_loader
    request = _validate_gate_request(
        config_path=config_path,
        artifact_path=artifact_path,
        args=args,
        expected_git_sha_loader=(
            (lambda: resolved_git_sha_loader(_project_root(config)))
            if validate_only
            else None
        ),
    )
    plan = {
        "mode": "VALIDATION ONLY" if validate_only else "EXECUTE",
        "gate": _gate_name(args.command),
        "run_id": args.run_id,
        "artifact": str(artifact_path),
        "config_sha256": artifact_file_sha256(config_path),
        "request": request,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    if validate_only:
        return 0
    execute_gate(
        config_path=args.config,
        artifact_path=args.artifact,
        command=args.command,
        run_id=args.run_id,
        runtime=None,
        args=args,
        git_sha_loader=resolved_git_sha_loader,
        runtime_factory=runtime_factory,
    )
    print(f"{_gate_name(args.command)}: PASS ({args.artifact.resolve()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ConcreteGateRuntime",
    "GateRuntime",
    "GateTransaction",
    "ServerAdapterError",
    "build_parser",
    "execute_gate",
    "main",
    "validate_cli_request",
]
