"""Server-only factory for the project-owned Capsule trainer.

The entrypoint imports this module only on the executable path, never during
``main_ppo --validate-only``. Runtime side effects (Ray, VeRL workers, simulator workers, and
Controller clients) remain delayed until :meth:`fit`, while this module provides the concrete
repository-owned binding for the complete trainer dataflow.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from capx.envs.configs.instantiate import instantiate
from capx.envs.configs.loader import DictLoader

from .compat import (
    VeRLCompatibilityError,
    bind_pinned_verl_import,
    check_verl_compatibility,
    validate_capsule_config,
    verify_imported_verl_path,
)
from .checkpoint import AtomicCheckpointClaim
from .controller import (
    ControllerRepairCollector,
    FrozenControllerConfig,
    OpenAICompatibleControllerTransport,
)
from .evaluator import (
    CandidateCleanReplayAdapter,
    CleanReplayEvaluator,
    PersistentProcessReplayBackend,
)
from .group import (
    BASE_GROUP_SIZE,
    CandidateCollectionError,
    CapsuleGroupAssembler,
    ProgramCandidate,
    deterministic_group_uid,
)
from .schema import TaskInstanceV1
from .trainer import (
    AtomicJsonArtifactSink,
    CapsuleCritiqueRayTrainer,
)


class ServerFactoryError(RuntimeError):
    """The server runtime cannot be assembled safely from the resolved configuration."""


class UnresolvedTaskStateError(ServerFactoryError):
    """A task row has not yet been bound to a deterministic initial-state hash."""


class RuntimeSession(Protocol):
    def fit(self) -> Any: ...


class DataProtoFactory(Protocol):
    def __call__(self, tensors: dict[str, torch.Tensor]) -> Any: ...


@dataclass(frozen=True)
class YamlEnvironmentFactory:
    """Pickle-safe factory used by spawned clean-replay workers."""

    config_path: str

    def __call__(self, task: TaskInstanceV1) -> Any:
        del task
        config = DictLoader.load(self.config_path)
        if not isinstance(config, Mapping) or "env" not in config:
            raise ServerFactoryError(
                f"environment config must contain an env factory: {self.config_path}"
            )
        return instantiate(config["env"])


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ServerFactoryError(f"{name} must be a mapping")
    return value


def _write_discard_audit(destination: Path, records: Sequence[Any]) -> Path | None:
    """Persist complete immutable discard records before deciding whether training can continue."""

    if not records:
        return None
    if destination.exists():
        raise FileExistsError(f"discard audit already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    payload = {
        "schema_version": 1,
        "discarded_groups": [record.to_dict() for record in records],
    }
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, allow_nan=False, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _run_slug(run_id: str) -> str:
    readable = "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in run_id.strip()
    ).strip("._-")
    if not readable:
        raise ServerFactoryError("runtime.run_id has no safe path characters")
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:10]
    return f"{readable[:64]}-{digest}"


def _project_root(config: Mapping[str, Any]) -> Path:
    runtime = _mapping(config.get("runtime"), "runtime")
    configured = runtime.get("project_root")
    if isinstance(configured, str) and configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def _resolve_project_path(value: object, project_root: Path, field_name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ServerFactoryError(f"{field_name} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_task_instances(
    config: Mapping[str, Any],
    *,
    initial_state_resolver: Callable[[Mapping[str, Any]], str] | None = None,
) -> tuple[TaskInstanceV1, ...]:
    """Load immutable, seed-resolved task rows from JSONL or Parquet.

    The server seed-resolution step must populate ``initial_state_sha256``.  Training refuses
    to manufacture or guess it because the hash is part of every replay/group identity.
    """

    project_root = _project_root(config)
    runtime = _mapping(config.get("runtime"), "runtime")
    task_config = _mapping(config.get("task"), "task")
    dataset_path = _resolve_project_path(
        runtime.get("dataset_path"), project_root, "runtime.dataset_path"
    )
    if not dataset_path.is_file():
        raise ServerFactoryError(f"runtime.dataset_path does not exist: {dataset_path}")

    rows: list[Mapping[str, Any]]
    if dataset_path.suffix.lower() in {".jsonl", ".json"}:
        rows = []
        for line_number, line in enumerate(
            dataset_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise UnresolvedTaskStateError(
                    f"dataset line {line_number} is invalid JSON: {error}"
                ) from error
            rows.append(_mapping(row, f"dataset line {line_number}"))
    elif dataset_path.suffix.lower() == ".parquet":
        try:
            import pandas as pd
        except ModuleNotFoundError as error:
            raise ServerFactoryError("reading a Parquet task dataset requires pandas") from error
        rows = [dict(row) for row in pd.read_parquet(dataset_path).to_dict("records")]
    else:
        raise ServerFactoryError("runtime.dataset_path must be JSONL, JSON, or Parquet")
    if not rows:
        raise ServerFactoryError("runtime.dataset_path contains no task rows")

    tasks: list[TaskInstanceV1] = []
    seen: set[tuple[str, int]] = set()
    for index, row in enumerate(rows):
        data = dict(row)
        data.setdefault("schema_version", 1)
        data.setdefault("environment", task_config["environment"])
        data.setdefault("api", task_config["api"])
        data.setdefault("privilege", str(task_config["privilege"]))
        for field_name in ("environment", "api", "privilege"):
            expected = str(task_config[field_name])
            if data.get(field_name) != expected:
                raise ServerFactoryError(
                    f"dataset row {index} {field_name}={data.get(field_name)!r} does not "
                    f"match task config {expected!r}"
                )
        if initial_state_resolver is not None:
            data["initial_state_sha256"] = initial_state_resolver(data)
        elif not data.get("initial_state_sha256"):
            raise UnresolvedTaskStateError(
                f"dataset row {index} has no initial_state_sha256; run the server "
                "seed-resolution gate before training"
            )
        task = TaskInstanceV1.from_dict(data)
        if task.environment_seed < 0:
            raise ServerFactoryError(
                f"dataset row {index} environment_seed must be non-negative"
            )
        if "capsule_collection_id" in task.metadata:
            raise ServerFactoryError(
                "dataset metadata cannot set reserved field capsule_collection_id"
            )
        identity = (task.task_id, task.environment_seed)
        if identity in seen:
            raise ServerFactoryError(f"duplicate task/environment seed row: {identity}")
        seen.add(identity)
        tasks.append(task)
    return tuple(tasks)


def resolve_task_instances(
    config: Mapping[str, Any],
    *,
    environment_factory: Callable[[TaskInstanceV1 | None], Any] | None = None,
) -> tuple[TaskInstanceV1, ...]:
    """Resolve every initial hash by real server reset without trusting source claims."""

    project_root = _project_root(config)
    task_config = _mapping(config.get("task"), "task")
    if environment_factory is None:
        environment_path = _resolve_project_path(
            task_config.get("config_path"), project_root, "task.config_path"
        )
        environment_factory = YamlEnvironmentFactory(str(environment_path))
    environment = environment_factory(None)
    resolved_by_seed: dict[int, str] = {}
    try:
        def resolve_row(row: Mapping[str, Any]) -> str:
            seed = row.get("environment_seed")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ServerFactoryError("task environment_seed must be an integer")
            if seed not in resolved_by_seed:
                _observation, info = environment.reset(
                    seed=seed,
                    options={"capsule_task_state_resolution": True},
                )
                initial_hash = (
                    info.get("initial_state_sha256") if isinstance(info, Mapping) else None
                )
                if not isinstance(initial_hash, str):
                    raise ServerFactoryError(
                        f"environment reset for seed {seed} omitted initial_state_sha256"
                    )
                resolved_by_seed[seed] = initial_hash
            return resolved_by_seed[seed]

        return load_task_instances(config, initial_state_resolver=resolve_row)
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            close()


def _apply_chat_template(tokenizer: Any, prompt: str, system_prompt: str) -> list[int]:
    messages = (
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    )
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        token_ids = apply_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        token_ids = tokenizer.encode(
            f"{system_prompt}\n\n{prompt}", add_special_tokens=False
        )
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
        raise ServerFactoryError("tokenizer chat template must return token IDs")
    result = list(token_ids)
    if not result or any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise ServerFactoryError("tokenizer chat template returned invalid token IDs")
    return result


def _encode_raw_response(tokenizer: Any, source: str) -> list[int]:
    token_ids = tokenizer.encode(source, add_special_tokens=False)
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
        raise ServerFactoryError("tokenizer.encode must return raw response token IDs")
    result = list(token_ids)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in result
    ):
        raise ServerFactoryError("tokenizer.encode returned invalid raw response token IDs")
    return result


class VeRLProgramGenerator:
    """Generate complete Program candidates through the trainable VeRL rollout worker."""

    def __init__(
        self,
        *,
        actor_rollout_wg: Any,
        tokenizer: Any,
        data_proto_factory: Callable[..., Any],
        prompt_token_limit: int,
        response_token_limit: int,
        system_prompt: str,
    ) -> None:
        self.actor_rollout_wg = actor_rollout_wg
        self.tokenizer = tokenizer
        self.data_proto_factory = data_proto_factory
        self.prompt_token_limit = prompt_token_limit
        self.response_token_limit = response_token_limit
        self.system_prompt = system_prompt

    def count_prompt_tokens(self, text: str) -> int:
        return len(_apply_chat_template(self.tokenizer, text, self.system_prompt))

    def count_raw_response_tokens(self, text: str) -> int:
        return len(_encode_raw_response(self.tokenizer, text))

    def count_tokens(self, text: str) -> int:
        """Compatibility alias for callers that only count chat-templated prompts."""

        return self.count_prompt_tokens(text)

    def _decode(self, output: Any) -> tuple[str, str, bool]:
        batch = getattr(output, "batch", output)
        responses = batch["responses"]
        if not isinstance(responses, torch.Tensor) or responses.ndim != 2 or responses.shape[0] < 1:
            raise CandidateCollectionError("VeRL rollout returned invalid responses")
        response_ids = responses[0].detach().cpu().tolist()
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        active_length = len(response_ids)
        if "attention_mask" in batch:
            attention_mask = batch["attention_mask"]
            if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
                active_length = int(attention_mask[0, -len(response_ids) :].sum().item())
        if active_length > self.response_token_limit:
            raise CandidateCollectionError(
                f"Program response exceeds {self.response_token_limit} tokens; it was not truncated"
            )
        active = response_ids[:active_length]
        if not isinstance(eos_token_id, int) or eos_token_id not in active:
            raise CandidateCollectionError(
                "VeRL rollout response did not terminate with EOS; action identity is untrusted"
            )
        eos_index = active.index(eos_token_id)
        if eos_index != len(active) - 1:
            raise CandidateCollectionError(
                "VeRL rollout returned active tokens after EOS; action identity is untrusted"
            )
        raw_response_ids = active[:eos_index]
        source = self.tokenizer.decode(raw_response_ids, skip_special_tokens=True)
        if not isinstance(source, str) or not source:
            raise CandidateCollectionError("VeRL rollout returned an empty Program response")
        retokenized = _encode_raw_response(self.tokenizer, source)
        if retokenized != raw_response_ids:
            raise CandidateCollectionError(
                "VeRL rollout response failed the decode/retokenize token round-trip"
            )
        return source, "stop", False

    def generate(self, prompt: str, program_sample_id: str) -> ProgramCandidate:
        prompt_ids = _apply_chat_template(self.tokenizer, prompt, self.system_prompt)
        if len(prompt_ids) > self.prompt_token_limit:
            raise CandidateCollectionError(
                f"Program prompt exceeds {self.prompt_token_limit} tokens; it was not truncated"
            )
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        position_ids = torch.arange(len(prompt_ids), dtype=torch.long).unsqueeze(0)
        request = self.data_proto_factory(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            non_tensors={"raw_prompt_ids": np.array([prompt_ids], dtype=object)},
            meta_info={"do_sample": True, "validate": False},
        )
        output = self.actor_rollout_wg.generate_sequences(request)
        source, finish_reason, truncated = self._decode(output)
        return ProgramCandidate(
            program_sample_id=program_sample_id,
            source=source,
            finish_reason=finish_reason,
            truncated=truncated,
        )


class VeRLGroupEncoder:
    """Encode training pairs with the exact chat template used for Program rollout."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        data_proto_factory: Callable[..., Any],
        prompt_token_limit: int,
        response_token_limit: int,
        system_prompt: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.data_proto_factory = data_proto_factory
        self.prompt_token_limit = prompt_token_limit
        self.response_token_limit = response_token_limit
        self.system_prompt = system_prompt
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        for name, token_id in (
            ("pad_token_id", self.pad_token_id),
            ("eos_token_id", self.eos_token_id),
        ):
            if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                raise ServerFactoryError(f"tokenizer.{name} must be a non-negative integer")

    def encode(self, prompts: tuple[str, ...], responses: tuple[str, ...]) -> Any:
        if not prompts or len(prompts) != len(responses):
            raise ServerFactoryError("training prompts/responses must have equal non-zero size")
        prompt_rows: list[list[int]] = []
        response_rows: list[list[int]] = []
        prompt_lengths: list[int] = []
        response_lengths: list[int] = []
        for row_index, (prompt, response) in enumerate(
            zip(prompts, responses, strict=True)
        ):
            prompt_ids = _apply_chat_template(self.tokenizer, prompt, self.system_prompt)
            response_ids = _encode_raw_response(self.tokenizer, response)
            if not response_ids:
                raise ServerFactoryError(f"training response {row_index} has invalid token IDs")
            response_ids.append(self.eos_token_id)
            if len(prompt_ids) > self.prompt_token_limit:
                raise ServerFactoryError(
                    f"training prompt {row_index} exceeds {self.prompt_token_limit} tokens; "
                    "it was not truncated"
                )
            if len(response_ids) > self.response_token_limit:
                raise ServerFactoryError(
                    f"training response {row_index} exceeds {self.response_token_limit} tokens; "
                    "it was not truncated"
                )
            prompt_rows.append(
                [self.pad_token_id] * (self.prompt_token_limit - len(prompt_ids)) + prompt_ids
            )
            response_rows.append(
                response_ids
                + [self.pad_token_id] * (self.response_token_limit - len(response_ids))
            )
            prompt_lengths.append(len(prompt_ids))
            response_lengths.append(len(response_ids))

        prompt_tensor = torch.tensor(prompt_rows, dtype=torch.long)
        response_tensor = torch.tensor(response_rows, dtype=torch.long)
        prompt_mask = torch.zeros_like(prompt_tensor, dtype=torch.bool)
        response_mask = torch.zeros_like(response_tensor, dtype=torch.bool)
        for row_index, (prompt_length, response_length) in enumerate(
            zip(prompt_lengths, response_lengths, strict=True)
        ):
            prompt_mask[row_index, self.prompt_token_limit - prompt_length :] = True
            response_mask[row_index, :response_length] = True
        attention_mask = torch.cat((prompt_mask, response_mask), dim=-1)
        input_ids = torch.cat((prompt_tensor, response_tensor), dim=-1)
        position_ids = torch.clamp(attention_mask.long().cumsum(dim=-1) - 1, min=0)
        return self.data_proto_factory(
            tensors={
                "prompts": prompt_tensor,
                "responses": response_tensor,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "response_mask": response_mask,
            }
        )


class ActorBaseSampler:
    def __init__(self, generator: VeRLProgramGenerator) -> None:
        self.generator = generator

    def __call__(self, task: TaskInstanceV1, base_index: int) -> ProgramCandidate:
        sample_id = f"{deterministic_group_uid(task)}:base-{base_index}"
        return self.generator.generate(task.prompt, sample_id)


class ActorRevisionGenerator:
    def __init__(self, generator: VeRLProgramGenerator) -> None:
        self.generator = generator

    def __call__(
        self,
        task: TaskInstanceV1,
        p0: ProgramCandidate,
        trace: Any,
        revision_prompt: Any,
        p0_rank: int,
        trajectory_index: int,
    ) -> ProgramCandidate:
        del task, p0, p0_rank, trajectory_index
        return self.generator.generate(
            revision_prompt.text,
            f"{trace.repair_trajectory_id}:p-hat",
        )


@dataclass
class VeRLWorkerSession:
    actor_rollout_wg: Any
    ref_policy_wg: Any
    tokenizer: Any
    data_proto_factory: Callable[..., Any]
    ray_module: Any
    owns_ray: bool
    total_epochs: int = 1
    total_training_steps: int = 1
    rollout_mode: str = "sync"
    ppo_epochs: int = 1
    ppo_mini_batch_size: int = 8
    kl_loss_coef: float = 0.001
    data_parallel_world_size: int = 1
    sequence_parallel_size: int = 1
    verl_source_path: Path | None = None
    verl_pinned_sha: str | None = None
    initial_verl_provenance: dict[str, Any] | None = None
    final_verl_provenance: dict[str, Any] | None = None

    def save_checkpoint(self, path: Path, step: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.actor_rollout_wg.save_checkpoint(str(path), None, step, 2)

    def optimizer_step(self) -> int:
        probe = getattr(self.actor_rollout_wg, "get_capsule_optimizer_step", None)
        if not callable(probe):
            raise ServerFactoryError(
                "pinned VeRL actor worker does not expose Capsule optimizer-step evidence"
            )
        raw = probe()
        values: list[int] = []

        def collect(value: Any) -> None:
            if isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)
                return
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ServerFactoryError(
                    f"actor optimizer-step probe returned an invalid value: {value!r}"
                )
            values.append(value)

        collect(raw)
        if not values or len(set(values)) != 1:
            raise ServerFactoryError(
                f"actor optimizer-step evidence must agree across ranks; got {values!r}"
            )
        return values[0]

    def verl_provenance(self) -> dict[str, Any]:
        """Require the driver and every actor rank to see one clean pinned VeRL tree."""

        if self.verl_source_path is None or self.verl_pinned_sha is None:
            raise ServerFactoryError("VeRL session has no pinned source provenance contract")
        expected_root = self.verl_source_path.expanduser().resolve()
        expected_package = (expected_root / "verl").resolve()
        try:
            report = check_verl_compatibility(expected_root, self.verl_pinned_sha)
        except VeRLCompatibilityError as error:
            raise ServerFactoryError(f"driver VeRL provenance failed: {error}") from error
        probe = getattr(self.actor_rollout_wg, "get_capsule_verl_provenance", None)
        if not callable(probe):
            raise ServerFactoryError(
                "pinned VeRL actor worker does not expose provenance evidence"
            )
        records: list[Mapping[str, Any]] = []

        def collect(value: Any) -> None:
            if isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)
                return
            if not isinstance(value, Mapping):
                raise ServerFactoryError(
                    f"actor VeRL provenance returned an invalid value: {value!r}"
                )
            records.append(value)

        collect(probe())
        if len(records) != self.data_parallel_world_size:
            raise ServerFactoryError(
                "actor VeRL provenance rank count does not match data-parallel world size; "
                f"got {len(records)}, expected {self.data_parallel_world_size}"
            )
        normalized: list[dict[str, Any]] = []
        ranks: set[int] = set()
        for record in records:
            rank = record.get("rank")
            source_path = record.get("source_path")
            module_path = record.get("module_path")
            expected_sha = record.get("expected_sha")
            actual_sha = record.get("actual_sha")
            clean = record.get("clean")
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
                raise ServerFactoryError(f"actor VeRL provenance has invalid rank: {rank!r}")
            if rank in ranks:
                raise ServerFactoryError(f"actor VeRL provenance repeats rank {rank}")
            ranks.add(rank)
            try:
                worker_root = Path(str(source_path)).expanduser().resolve()
                worker_module = Path(str(module_path)).expanduser().resolve()
            except (OSError, TypeError, ValueError) as error:
                raise ServerFactoryError(
                    f"actor VeRL provenance has invalid paths: {record!r}"
                ) from error
            if (
                worker_root != expected_root
                or not worker_module.is_relative_to(expected_package)
                or expected_sha != self.verl_pinned_sha
                or actual_sha != self.verl_pinned_sha
                or clean is not True
            ):
                raise ServerFactoryError(
                    f"actor VeRL provenance does not match the pinned checkout: {record!r}"
                )
            normalized.append(
                {
                    "rank": rank,
                    "module_path": str(worker_module),
                    "actual_sha": actual_sha,
                    "clean": True,
                }
            )
        normalized.sort(key=lambda item: item["rank"])
        if report.actual_sha != self.verl_pinned_sha:
            raise ServerFactoryError("driver VeRL provenance does not match the pinned SHA")
        return {
            "source_path": str(expected_root),
            "expected_sha": self.verl_pinned_sha,
            "actual_sha": report.actual_sha,
            "clean": True,
            "worker_count": len(normalized),
            "worker_ranks": [record["rank"] for record in normalized],
            "worker_module_paths": [record["module_path"] for record in normalized],
        }

    def close(self) -> None:
        if not self.owns_ray:
            raise ServerFactoryError("Capsule VeRL sessions must own their isolated Ray runtime")
        try:
            self.final_verl_provenance = self.verl_provenance()
        finally:
            self.ray_module.shutdown()


def _verify_imported_verl_path(verl_path: Path) -> None:
    """Fail if any loaded VeRL module comes from outside the pinned checkout."""

    try:
        verify_imported_verl_path(verl_path)
    except VeRLCompatibilityError as error:
        raise ServerFactoryError(str(error)) from error


def _bind_pinned_verl_import(verl_path: Path) -> None:
    """Import VeRL from exactly the checkout whose SHA/API surface was validated."""

    try:
        bind_pinned_verl_import(verl_path)
    except VeRLCompatibilityError as error:
        raise ServerFactoryError(str(error)) from error


def _pinned_ray_runtime_env(
    runtime_env: object, verl_path: Path, expected_sha: str
) -> dict[str, Any]:
    """Copy a Ray runtime env and make the validated VeRL checkout explicit to workers."""

    if not isinstance(runtime_env, Mapping):
        raise ServerFactoryError("VeRL Ray runtime_env must resolve to a mapping")
    resolved = dict(runtime_env)
    configured_env_vars = resolved.get("env_vars", {})
    if not isinstance(configured_env_vars, Mapping):
        raise ServerFactoryError("VeRL Ray runtime_env.env_vars must be a mapping")
    env_vars = {str(key): str(value) for key, value in configured_env_vars.items()}
    existing_pythonpath = env_vars.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    if not isinstance(existing_pythonpath, str):
        raise ServerFactoryError("Ray PYTHONPATH must be text")
    pinned = str(verl_path.expanduser().resolve())
    path_entries = [entry for entry in existing_pythonpath.split(os.pathsep) if entry]
    path_entries = [entry for entry in path_entries if Path(entry).resolve() != Path(pinned)]
    env_vars["PYTHONPATH"] = os.pathsep.join((pinned, *path_entries))
    env_vars["CAPX_PINNED_VERL_SOURCE_PATH"] = pinned
    env_vars["CAPX_PINNED_VERL_SHA"] = expected_sha
    resolved["env_vars"] = env_vars
    return resolved


def _dataset_row_count(config: Mapping[str, Any]) -> int:
    """Count scheduled task rows without resolving simulator state."""

    project_root = _project_root(config)
    runtime = _mapping(config.get("runtime"), "runtime")
    dataset_path = _resolve_project_path(
        runtime.get("dataset_path"), project_root, "runtime.dataset_path"
    )
    suffix = dataset_path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        count = sum(
            1
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    elif suffix == ".parquet":
        try:
            import pandas as pd
        except ModuleNotFoundError as error:
            raise ServerFactoryError("reading a Parquet task dataset requires pandas") from error
        count = len(pd.read_parquet(dataset_path))
    else:
        raise ServerFactoryError("runtime.dataset_path must be JSONL, JSON, or Parquet")
    if count < 1:
        raise ServerFactoryError("runtime.dataset_path contains no task rows")
    return count


def _configure_verl_training_schedule(
    verl_config: Any, dataset_row_count: int
) -> tuple[int, int]:
    """Bound optimizer scheduling to the number of immutable task groups."""

    if (
        isinstance(dataset_row_count, bool)
        or not isinstance(dataset_row_count, int)
        or dataset_row_count < 1
    ):
        raise ServerFactoryError("dataset_row_count must be a positive integer")
    total_epochs = verl_config.trainer.total_epochs
    if (
        isinstance(total_epochs, bool)
        or not isinstance(total_epochs, int)
        or total_epochs < 1
    ):
        raise ServerFactoryError("VeRL trainer.total_epochs must be a positive integer")
    total_training_steps = dataset_row_count * total_epochs
    from omegaconf import open_dict

    with open_dict(verl_config):
        verl_config.trainer.total_training_steps = total_training_steps
        verl_config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
    return total_epochs, total_training_steps


def _capsule_data_parallel_world_size(verl_config: Any) -> int:
    """Validate that one global 7+1 mini-batch splits evenly across FSDP ranks."""

    n_gpus_per_node = verl_config.trainer.n_gpus_per_node
    nnodes = verl_config.trainer.nnodes
    for field_name, value in (
        ("trainer.n_gpus_per_node", n_gpus_per_node),
        ("trainer.nnodes", nnodes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ServerFactoryError(f"VeRL {field_name} must be a positive integer")
    world_size = n_gpus_per_node * nnodes
    if BASE_GROUP_SIZE % world_size != 0:
        raise ServerFactoryError(
            "Capsule group_size=8 must be divisible by the FSDP data-parallel world size; "
            f"got {world_size}"
        )
    return world_size


def _load_resolved_verl_config(config: Mapping[str, Any], project_root: Path) -> Any:
    runtime = _mapping(config.get("runtime"), "runtime")
    capsule = _mapping(config.get("capsule"), "capsule")
    capsule_actor = _mapping(
        _mapping(config.get("actor_rollout_ref"), "actor_rollout_ref").get("actor"),
        "actor_rollout_ref.actor",
    )
    resolved_path = _resolve_project_path(
        runtime.get("verl_resolved_config_path"),
        project_root,
        "runtime.verl_resolved_config_path",
    )
    if not resolved_path.is_file():
        raise ServerFactoryError(f"resolved VeRL config does not exist: {resolved_path}")
    from omegaconf import OmegaConf, open_dict

    verl_config = OmegaConf.load(resolved_path)
    with open_dict(verl_config):
        verl_config.actor_rollout_ref.model.path = str(
            _resolve_project_path(
                runtime.get("program_model_path"), project_root, "runtime.program_model_path"
            )
        )
        verl_config.actor_rollout_ref.model.external_lib = (
            "capx.rl.capsule.verl_external"
        )
        # Capsule owns the eight-member group.  Each low-level rollout request therefore emits
        # one candidate, even though the externally validated learning-group size remains eight.
        verl_config.actor_rollout_ref.rollout.n = 1
        verl_config.actor_rollout_ref.rollout.calculate_log_probs = False
        verl_config.actor_rollout_ref.rollout.mode = "sync"
        verl_config.actor_rollout_ref.rollout.prompt_length = int(
            capsule["revision_input_max_tokens"]
        )
        verl_config.actor_rollout_ref.rollout.response_length = int(
            capsule["revision_response_max_tokens"]
        )
        verl_config.actor_rollout_ref.actor.use_kl_loss = True
        verl_config.actor_rollout_ref.actor.kl_loss_coef = float(capsule_actor["kl_loss_coef"])
        verl_config.actor_rollout_ref.actor.ppo_epochs = 1
        verl_config.actor_rollout_ref.actor.ppo_mini_batch_size = 8
        verl_config.actor_rollout_ref.actor.ulysses_sequence_parallel_size = 1
        verl_config.actor_rollout_ref.actor.policy_loss.loss_mode = "capsule_critique"
        verl_config.actor_rollout_ref.actor.policy_loss.capsule_gamma = 0.1
        verl_config.algorithm.norm_adv_by_std_in_grpo = False
        verl_config.algorithm.rollout_is = False
        verl_config.algorithm.rollout_is_threshold = None
        verl_config.algorithm.use_kl_in_reward = False
        verl_config.reward_model.enable = False
        verl_config.reward_model.launch_reward_fn_async = False
        verl_config.trainer.default_local_dir = str(
            _resolve_project_path(runtime.get("output_dir"), project_root, "runtime.output_dir")
            / "checkpoints"
        )
    return verl_config


def start_verl_workers(config: Mapping[str, Any]) -> VeRLWorkerSession:
    """Start only the pinned actor/rollout and reference workers needed by Capsule."""

    project_root = _project_root(config)
    runtime = _mapping(config.get("runtime"), "runtime")
    verl_path = _resolve_project_path(
        runtime.get("verl_source_path"), project_root, "runtime.verl_source_path"
    )
    expected_verl_sha = str(runtime["verl_pinned_sha"])
    check_verl_compatibility(verl_path, expected_verl_sha)
    verl_config = _load_resolved_verl_config(config, project_root)
    total_epochs, total_training_steps = _configure_verl_training_schedule(
        verl_config, _dataset_row_count(config)
    )
    data_parallel_world_size = _capsule_data_parallel_world_size(verl_config)
    _bind_pinned_verl_import(verl_path)

    import ray
    from omegaconf import OmegaConf
    from verl import DataProto
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
    from verl.single_controller.ray.base import create_colocated_worker_cls
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from verl.utils import hf_tokenizer
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    _verify_imported_verl_path(verl_path)
    strategy = str(verl_config.actor_rollout_ref.actor.strategy)
    if strategy not in {"fsdp", "fsdp2"}:
        raise ServerFactoryError("Cube Stack Capsule MVP supports only VeRL FSDP/FSDP2 workers")
    if ray.is_initialized():
        raise ServerFactoryError(
            "Capsule requires an isolated Ray runtime so actor/ref workers can be cleaned up"
        )
    owns_ray = True
    default_runtime_env = get_ppo_ray_runtime_env()
    ray_init = OmegaConf.to_container(
        verl_config.ray_kwargs.get("ray_init", {}), resolve=True
    )
    if not isinstance(ray_init, dict):
        raise ServerFactoryError("VeRL ray_kwargs.ray_init must resolve to a mapping")
    configured_runtime_env = ray_init.pop("runtime_env", {})
    merged_runtime_env = OmegaConf.to_container(
        OmegaConf.merge(default_runtime_env, configured_runtime_env), resolve=True
    )
    ray_init["runtime_env"] = _pinned_ray_runtime_env(
        merged_runtime_env, verl_path, expected_verl_sha
    )
    try:
        ray.init(**ray_init)
    except BaseException:
        if ray.is_initialized():
            ray.shutdown()
        raise

    try:
        class CapsuleActorRolloutRefWorker(ActorRolloutRefWorker):
            @register(dispatch_mode=Dispatch.ONE_TO_ALL)
            def get_capsule_verl_provenance(self) -> dict[str, Any]:
                import verl

                from capx.rl.capsule.compat import (
                    check_verl_compatibility,
                    verify_imported_verl_path,
                )

                source = os.environ.get("CAPX_PINNED_VERL_SOURCE_PATH")
                expected_sha = os.environ.get("CAPX_PINNED_VERL_SHA")
                if not source or not expected_sha:
                    raise RuntimeError("actor worker omitted pinned VeRL provenance markers")
                report = check_verl_compatibility(source, expected_sha)
                verify_imported_verl_path(source)
                module_file = getattr(verl, "__file__", None)
                if not isinstance(module_file, str) or not module_file:
                    raise RuntimeError("actor worker cannot resolve verl.__file__")
                return {
                    "rank": int(self.rank),
                    "source_path": str(Path(source).expanduser().resolve()),
                    "module_path": str(Path(module_file).resolve()),
                    "expected_sha": expected_sha,
                    "actual_sha": report.actual_sha,
                    "clean": True,
                }

            @register(dispatch_mode=Dispatch.ONE_TO_ALL)
            def get_capsule_optimizer_step(self) -> int:
                optimizer = getattr(self, "actor_optimizer", None)
                if optimizer is None:
                    raise RuntimeError("actor worker has no optimizer")
                steps: list[int] = []
                for state in optimizer.state.values():
                    if "step" not in state:
                        continue
                    raw_step = state["step"]
                    if isinstance(raw_step, torch.Tensor):
                        if raw_step.numel() != 1:
                            raise RuntimeError("optimizer state contains a non-scalar step")
                        raw_step = raw_step.item()
                    if (
                        isinstance(raw_step, bool)
                        or not isinstance(raw_step, (int, float))
                        or int(raw_step) != raw_step
                        or raw_step < 0
                    ):
                        raise RuntimeError(
                            f"optimizer state contains an invalid step: {raw_step!r}"
                        )
                    steps.append(int(raw_step))
                if not steps:
                    return 0
                if len(set(steps)) != 1:
                    raise RuntimeError(
                        f"optimizer parameter steps disagree on rank {self.rank}: {steps!r}"
                    )
                return steps[0]

        remote_worker = ray.remote(CapsuleActorRolloutRefWorker)
        global_pool = "capsule_global_pool"
        mapping = {Role.ActorRollout: global_pool, Role.RefPolicy: global_pool}
        resource_manager = ResourcePoolManager(
            resource_pool_spec={
                global_pool: [verl_config.trainer.n_gpus_per_node]
                * verl_config.trainer.nnodes
            },
            mapping=mapping,
        )
        resource_manager.create_resource_pool()
        resource_pool = resource_manager.get_resource_pool(Role.ActorRollout)
        class_dict = {
            "actor_rollout": RayClassWithInitArgs(
                cls=remote_worker,
                config=verl_config.actor_rollout_ref,
                role="actor_rollout",
            ),
            "ref": RayClassWithInitArgs(
                remote_worker,
                config=verl_config.actor_rollout_ref,
                role="ref",
            ),
        }
        worker_cls = create_colocated_worker_cls(class_dict=class_dict)
        worker_group = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_cls,
            device_name=verl_config.trainer.device,
        )
        groups = worker_group.spawn(prefix_set=class_dict.keys())
        ref_policy_wg = groups["ref"]
        ref_policy_wg.init_model()
        actor_rollout_wg = groups["actor_rollout"]
        actor_rollout_wg.init_model()
        tokenizer = hf_tokenizer(
            verl_config.actor_rollout_ref.model.path,
            trust_remote_code=bool(verl_config.data.get("trust_remote_code", False)),
        )
        session = VeRLWorkerSession(
            actor_rollout_wg=actor_rollout_wg,
            ref_policy_wg=ref_policy_wg,
            tokenizer=tokenizer,
            data_proto_factory=DataProto.from_dict,
            ray_module=ray,
            owns_ray=owns_ray,
            total_epochs=total_epochs,
            total_training_steps=total_training_steps,
            rollout_mode=str(verl_config.actor_rollout_ref.rollout.mode),
            ppo_epochs=int(verl_config.actor_rollout_ref.actor.ppo_epochs),
            ppo_mini_batch_size=int(
                verl_config.actor_rollout_ref.actor.ppo_mini_batch_size
            ),
            kl_loss_coef=float(verl_config.actor_rollout_ref.actor.kl_loss_coef),
            data_parallel_world_size=data_parallel_world_size,
            sequence_parallel_size=int(
                verl_config.actor_rollout_ref.actor.ulysses_sequence_parallel_size
            ),
            verl_source_path=verl_path,
            verl_pinned_sha=expected_verl_sha,
        )
        session.initial_verl_provenance = session.verl_provenance()
        return session
    except BaseException:
        ray.shutdown()
        raise


def _close_runtime_resources(
    repair_collector: ControllerRepairCollector | None,
    evaluator: CleanReplayEvaluator | None,
    workers: VeRLWorkerSession,
) -> None:
    """Attempt every teardown even when an earlier resource reports an error."""

    try:
        if repair_collector is not None:
            repair_collector.close()
    finally:
        try:
            if evaluator is not None:
                evaluator.close()
        finally:
            workers.close()


def _schedule_training_tasks(
    tasks: tuple[TaskInstanceV1, ...], total_epochs: int
) -> tuple[TaskInstanceV1, ...]:
    """Give every repeated task/epoch a unique collection identity."""

    if isinstance(total_epochs, bool) or not isinstance(total_epochs, int) or total_epochs < 1:
        raise ServerFactoryError("total_epochs must be a positive integer")
    scheduled: list[TaskInstanceV1] = []
    for epoch in range(total_epochs):
        for task_index, task in enumerate(tasks):
            metadata = dict(task.metadata)
            metadata["capsule_collection_id"] = (
                f"epoch-{epoch:08d}:task-{task_index:08d}"
            )
            scheduled.append(replace(task, metadata=metadata))
    return tuple(scheduled)


class CapsuleServerRuntime:
    """Concrete server session binding all Capsule components for training."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        worker_starter: Callable[[Mapping[str, Any]], VeRLWorkerSession] = start_verl_workers,
        task_loader: Callable[
            [Mapping[str, Any]], tuple[TaskInstanceV1, ...]
        ] = load_task_instances,
    ) -> None:
        self.config = config
        self.worker_starter = worker_starter
        self.task_loader = task_loader

    def fit(self) -> dict[str, Any]:
        project_root = _project_root(self.config)
        runtime = _mapping(self.config.get("runtime"), "runtime")
        task_config = _mapping(self.config.get("task"), "task")
        capsule = _mapping(self.config.get("capsule"), "capsule")
        controller_service = _mapping(
            self.config.get("controller_service"), "controller_service"
        )
        program_service = _mapping(self.config.get("program_service"), "program_service")
        tasks = self.task_loader(self.config)
        configured_run_id = runtime.get("run_id", "formal")
        if not isinstance(configured_run_id, str) or not configured_run_id.strip():
            raise ServerFactoryError("runtime.run_id must be non-empty when configured")
        safe_run_id = _run_slug(configured_run_id)
        output_dir = _resolve_project_path(
            runtime.get("output_dir"), project_root, "runtime.output_dir"
        )
        discard_audit_path = output_dir / "discarded_groups" / f"{safe_run_id}.json"
        if discard_audit_path.exists():
            raise FileExistsError(f"discard audit already exists: {discard_audit_path}")
        workers = self.worker_starter(self.config)
        evaluator: CleanReplayEvaluator | None = None
        repair_collector: ControllerRepairCollector | None = None
        fit_error: BaseException | None = None
        try:
            scheduled_tasks = _schedule_training_tasks(
                tasks, getattr(workers, "total_epochs", 1)
            )
            environment_config_path = _resolve_project_path(
                task_config.get("config_path"), project_root, "task.config_path"
            )
            environment_factory = YamlEnvironmentFactory(str(environment_config_path))
            replay_backend = PersistentProcessReplayBackend(environment_factory)
            evaluator = CleanReplayEvaluator(replay_backend)
            clean_evaluator = CandidateCleanReplayAdapter(evaluator)

            system_prompt = str(
                program_service.get(
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
            controller_config = FrozenControllerConfig(
                endpoint=str(controller_service["endpoint"]),
                model=str(controller_service["model"]),
                api_key_env=str(controller_service["api_key_env"]),
                frozen=True,
                max_turns=int(capsule["max_controller_turns"]),
                request_timeout_s=float(controller_service["request_timeout_s"]),
                temperature=float(controller_service["temperature"]),
            )
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
            batch_encoder = VeRLGroupEncoder(
                tokenizer=workers.tokenizer,
                data_proto_factory=workers.data_proto_factory,
                prompt_token_limit=int(capsule["revision_input_max_tokens"]),
                response_token_limit=int(capsule["revision_response_max_tokens"]),
                system_prompt=system_prompt,
            )
            trainer = CapsuleCritiqueRayTrainer(
                assembler=assembler,
                batch_encoder=batch_encoder,
                actor_rollout_wg=workers.actor_rollout_wg,
                ref_policy_wg=workers.ref_policy_wg,
                artifact_sink=AtomicJsonArtifactSink(output_dir / "groups"),
                config=self.config,
            )
            claim_root = output_dir / "checkpoints" / safe_run_id
            checkpoint = claim_root / "final" / "actor"
            with AtomicCheckpointClaim(checkpoint, claim_root=claim_root) as checkpoint_claim:
                verl_provenance_before = workers.verl_provenance()
                optimizer_step_before = workers.optimizer_step()
                results = trainer.fit(scheduled_tasks)
                actor_updates = sum(not result.skipped_actor_update for result in results)
                discarded_groups = tuple(getattr(trainer, "discarded_groups", ()))
                discard_audit = _write_discard_audit(
                    discard_audit_path, discarded_groups
                )
                if not results and discarded_groups:
                    raise ServerFactoryError(
                        "all scheduled groups were discarded; no checkpoint was written; "
                        f"inspect {discard_audit}"
                    )
                optimizer_step_after = workers.optimizer_step()
                optimizer_step_delta = optimizer_step_after - optimizer_step_before
                if optimizer_step_delta != actor_updates:
                    raise ServerFactoryError(
                        "actor optimizer-step evidence does not match successful actor updates; "
                        f"updates={actor_updates}, before={optimizer_step_before}, "
                        f"after={optimizer_step_after}"
                    )
                checkpoint_evidence = checkpoint_claim.publish(
                    lambda staging: workers.save_checkpoint(
                        staging, optimizer_step_after
                    ),
                    optimizer_step_before=optimizer_step_before,
                    optimizer_step_after=optimizer_step_after,
                )
                verl_provenance_after = workers.verl_provenance()
                if verl_provenance_after != verl_provenance_before:
                    raise ServerFactoryError(
                        "VeRL driver/worker provenance changed during Capsule training"
                    )
            run_status = (
                "completed_no_updates_all_constant"
                if results and actor_updates == 0 and not discarded_groups
                else "completed"
            )
            return {
                "status": run_status,
                "task_count": len(tasks),
                "scheduled_group_count": len(scheduled_tasks),
                "step_count": len(results),
                "actor_updates": actor_updates,
                "skipped_actor_updates": sum(result.skipped_actor_update for result in results),
                "discarded_groups": len(discarded_groups),
                "discard_reasons": [record.reason for record in discarded_groups],
                "discarded_group_records": [record.to_dict() for record in discarded_groups],
                "discard_audit": None if discard_audit is None else str(discard_audit),
                "optimizer_step_before": optimizer_step_before,
                "optimizer_step_after": optimizer_step_after,
                "optimizer_step_delta": optimizer_step_delta,
                "checkpoint": str(checkpoint_evidence.path),
                "checkpoint_file_count": checkpoint_evidence.file_count,
                "checkpoint_sha256": checkpoint_evidence.sha256,
                "checkpoint_manifest": str(checkpoint_evidence.manifest_path),
                "verl_provenance_before": verl_provenance_before,
                "verl_provenance_after": verl_provenance_after,
            }
        except BaseException as error:
            fit_error = error
            raise
        finally:
            try:
                _close_runtime_resources(repair_collector, evaluator, workers)
            except BaseException:
                if fit_error is None:
                    raise


def create_trainer(config: Mapping[str, Any]) -> CapsuleServerRuntime:
    """Entrypoint named by the repository YAML template."""

    validate_capsule_config(config)
    return CapsuleServerRuntime(config)


__all__ = [
    "ActorBaseSampler",
    "ActorRevisionGenerator",
    "CapsuleServerRuntime",
    "ServerFactoryError",
    "UnresolvedTaskStateError",
    "VeRLProgramGenerator",
    "VeRLGroupEncoder",
    "VeRLWorkerSession",
    "YamlEnvironmentFactory",
    "create_trainer",
    "load_task_instances",
    "resolve_task_instances",
    "start_verl_workers",
]
