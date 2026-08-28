"""Server-only factory for the project-owned Capsule trainer.

The entrypoint imports this module only on the executable path, never during
``main_ppo --validate-only``. Runtime side effects (Ray, VeRL workers, simulator workers, and
Controller clients) remain delayed until :meth:`fit`, while this module provides the concrete
repository-owned binding for the complete trainer dataflow.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

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
from .lora_contract import (
    QWEN25_ALL_LINEAR_PROJECTIONS,
    QWEN25_ALL_LINEAR_TENSOR_COUNT,
    QWEN25_CODER_7B_LAYER_COUNT,
    validate_qwen_all_linear_coverage,
)
from .schema import TaskInstanceV1
from .trainer import (
    AtomicJsonArtifactSink,
    CapsuleCritiqueRayTrainer,
    GroupAttemptBudgetExhausted,
)


class ServerFactoryError(RuntimeError):
    """The server runtime cannot be assembled safely from the resolved configuration."""


class UnresolvedTaskStateError(ServerFactoryError):
    """A task row has not yet been bound to a deterministic initial-state hash."""


class RuntimeSession(Protocol):
    def fit(self) -> Any: ...


class DataProtoFactory(Protocol):
    def __call__(self, tensors: dict[str, torch.Tensor]) -> Any: ...


def _host_mem_available_bytes(path: Path = Path("/proc/meminfo")) -> int:
    try:
        for line in path.read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB":
                    value = int(fields[1]) * 1024
                    if value > 0:
                        return value
    except (OSError, UnicodeError, ValueError) as error:
        raise RuntimeError(f"cannot read host MemAvailable evidence: {error}") from error
    raise RuntimeError("/proc/meminfo omitted a positive MemAvailable value")


def _actor_lora_runtime_record(
    model: Any,
    *,
    rank: int,
    cuda_peak_reserved_loader: Callable[[], int],
    host_mem_available_loader: Callable[[], int],
) -> dict[str, Any]:
    """Collect one rank's trainability and memory evidence without retaining tensors."""

    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise RuntimeError("actor worker has no named model parameters")
    total = 0
    trainable = 0
    trainable_names: list[str] = []
    non_lora_trainable: list[str] = []
    for name, parameter in named_parameters():
        if not isinstance(name, str) or not name:
            raise RuntimeError("actor model returned an invalid parameter name")
        count = int(parameter.numel())
        if count < 0:
            raise RuntimeError(f"actor parameter has a negative size: {name}")
        total += count
        if not bool(parameter.requires_grad):
            continue
        trainable += count
        trainable_names.append(name)
        lowered = name.lower()
        if ".lora_a." not in lowered and ".lora_b." not in lowered:
            non_lora_trainable.append(name)
    if non_lora_trainable:
        preview = ", ".join(non_lora_trainable[:3])
        raise RuntimeError(f"actor contains non-LoRA trainable parameters: {preview}")
    if total <= 0 or trainable <= 0 or not trainable_names:
        raise RuntimeError("actor LoRA runtime evidence found no trainable LoRA parameters")
    try:
        coverage = validate_qwen_all_linear_coverage(trainable_names)
    except ValueError as error:
        raise RuntimeError(f"actor Qwen all-linear coverage validation failed: {error}") from error
    names_digest = hashlib.sha256()
    for name in sorted(trainable_names):
        encoded = name.encode("utf-8")
        names_digest.update(len(encoded).to_bytes(8, "big"))
        names_digest.update(encoded)
    cuda_peak = cuda_peak_reserved_loader()
    host_available = host_mem_available_loader()
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (cuda_peak, host_available)
    ):
        raise RuntimeError("actor memory probe returned invalid byte counts")
    return {
        "rank": rank,
        "total_parameter_count": total,
        "trainable_parameter_count": trainable,
        "trainable_tensor_count": len(trainable_names),
        "lora_layer_count": coverage.layer_count,
        "lora_projection_suffixes": list(coverage.projection_suffixes),
        "non_lora_trainable_parameter_count": 0,
        "only_lora_trainable": True,
        "trainable_parameter_names_sha256": names_digest.hexdigest(),
        "cuda_peak_reserved_bytes": cuda_peak,
        "host_mem_available_bytes": host_available,
    }


@dataclass(frozen=True)
class YamlEnvironmentFactory:
    """Pickle-safe factory used by spawned clean-replay workers."""

    config_path: str
    config_bytes: bytes | None = None

    def _load_config(self) -> Any:
        if self.config_bytes is None:
            return DictLoader.load(self.config_path)
        if not isinstance(self.config_bytes, bytes):
            raise ServerFactoryError("environment config snapshot must be bytes")
        try:
            config_text = self.config_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ServerFactoryError(
                f"environment config must be a UTF-8 YAML snapshot: {self.config_path}"
            ) from error
        try:
            loaded = OmegaConf.create(yaml.unsafe_load(config_text))
            return OmegaConf.to_container(loaded, resolve=True)
        except Exception as error:
            raise ServerFactoryError(
                f"cannot parse environment config snapshot {self.config_path}: {error}"
            ) from error

    def __call__(self, task: TaskInstanceV1) -> Any:
        del task
        config = self._load_config()
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
    reference_policy_mode: str = "standalone"
    lora_rank: int = 0
    lora_alpha: int = 0
    lora_target_modules: tuple[str, ...] = ()
    verl_source_path: Path | None = None
    verl_pinned_sha: str | None = None
    initial_verl_provenance: dict[str, Any] | None = None
    final_verl_provenance: dict[str, Any] | None = None
    _worker_close_calls: int = 0
    _ray_shutdown_calls: int = 0
    _ray_shutdown_complete: bool = False
    _close_error: BaseException | None = None

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

    def lora_runtime_evidence(self) -> dict[str, Any]:
        """Require every actor rank to prove that only LoRA tensors are trainable."""

        if (
            self.lora_rank != 16
            or self.lora_alpha != 32
            or self.lora_target_modules != ("all-linear",)
        ):
            raise ServerFactoryError(
                "Capsule LoRA runtime evidence requires rank=16, alpha=32, all-linear"
            )
        probe = getattr(self.actor_rollout_wg, "get_capsule_lora_runtime_evidence", None)
        if not callable(probe):
            raise ServerFactoryError(
                "pinned VeRL actor worker does not expose LoRA runtime evidence"
            )
        records: list[Mapping[str, Any]] = []

        def collect(value: Any) -> None:
            if isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)
                return
            if not isinstance(value, Mapping):
                raise ServerFactoryError(
                    f"actor LoRA runtime probe returned an invalid value: {value!r}"
                )
            records.append(value)

        collect(probe())
        if len(records) != self.data_parallel_world_size:
            raise ServerFactoryError(
                "actor LoRA runtime rank count does not match data parallelism"
            )
        normalized: list[dict[str, Any]] = []
        ranks: set[int] = set()
        for record in records:
            rank = record.get("rank")
            total = record.get("total_parameter_count")
            trainable = record.get("trainable_parameter_count")
            trainable_tensors = record.get("trainable_tensor_count")
            lora_layer_count = record.get("lora_layer_count")
            lora_projection_suffixes = record.get("lora_projection_suffixes")
            non_lora = record.get("non_lora_trainable_parameter_count")
            names_sha256 = record.get("trainable_parameter_names_sha256")
            cuda_peak = record.get("cuda_peak_reserved_bytes")
            host_available = record.get("host_mem_available_bytes")
            integer_values = (rank, total, trainable, trainable_tensors, non_lora, cuda_peak)
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in integer_values
            ):
                raise ServerFactoryError("actor LoRA runtime evidence contains non-integer counts")
            if (
                rank < 0
                or rank in ranks
                or total <= 0
                or trainable <= 0
                or trainable > total
                or trainable_tensors <= 0
                or trainable_tensors != QWEN25_ALL_LINEAR_TENSOR_COUNT
                or lora_layer_count != QWEN25_CODER_7B_LAYER_COUNT
                or lora_projection_suffixes != list(QWEN25_ALL_LINEAR_PROJECTIONS)
                or non_lora != 0
                or record.get("only_lora_trainable") is not True
            ):
                raise ServerFactoryError(
                    "actor LoRA runtime evidence found a non-LoRA trainable parameter "
                    "or invalid parameter counts"
                )
            if (
                not isinstance(names_sha256, str)
                or len(names_sha256) != 64
                or any(character not in "0123456789abcdef" for character in names_sha256)
            ):
                raise ServerFactoryError("actor LoRA trainable-name SHA-256 is invalid")
            if cuda_peak < 0 or cuda_peak > 70 * 1024**3:
                raise ServerFactoryError(
                    "actor CUDA peak reserved memory exceeds the 70 GiB Gate 6 limit"
                )
            if (
                isinstance(host_available, bool)
                or not isinstance(host_available, int)
                or host_available <= 0
            ):
                raise ServerFactoryError("actor host MemAvailable evidence is invalid")
            ranks.add(rank)
            normalized.append(dict(record))
        normalized.sort(key=lambda item: int(item["rank"]))
        return {
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_target_modules": ["all-linear"],
            "worker_count": len(normalized),
            "worker_ranks": [int(record["rank"]) for record in normalized],
            "trainable_parameter_name_sha256s": [
                str(record["trainable_parameter_names_sha256"])
                for record in normalized
            ],
            "total_parameter_count": sum(
                int(record["total_parameter_count"]) for record in normalized
            ),
            "trainable_parameter_count": sum(
                int(record["trainable_parameter_count"]) for record in normalized
            ),
            "non_lora_trainable_parameter_count": 0,
            "only_lora_trainable": True,
            "lora_layer_count": QWEN25_CODER_7B_LAYER_COUNT,
            "lora_projection_suffixes": list(QWEN25_ALL_LINEAR_PROJECTIONS),
            "lora_tensor_count_per_worker": QWEN25_ALL_LINEAR_TENSOR_COUNT,
            "cuda_peak_reserved_bytes": max(
                int(record["cuda_peak_reserved_bytes"]) for record in normalized
            ),
            "host_mem_available_min_bytes": min(
                int(record["host_mem_available_bytes"]) for record in normalized
            ),
            "workers": normalized,
        }

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
        if self._ray_shutdown_complete:
            if self._close_error is not None:
                raise self._close_error
            return
        self._worker_close_calls += 1
        failure = self._close_error
        try:
            if self.final_verl_provenance is None:
                self.final_verl_provenance = self.verl_provenance()
        except BaseException as error:
            failure = failure or error
        try:
            self._ray_shutdown_calls += 1
            self.ray_module.shutdown()
        except BaseException as error:
            failure = failure or error
        is_initialized = getattr(self.ray_module, "is_initialized", None)
        try:
            self._ray_shutdown_complete = not callable(is_initialized) or not is_initialized()
        except BaseException as error:
            failure = failure or error
        if not self._ray_shutdown_complete:
            failure = failure or ServerFactoryError(
                "Ray remained initialized after Capsule worker shutdown"
            )
        self._close_error = failure
        if failure is not None:
            raise failure

    def ray_release_evidence(self) -> dict[str, Any]:
        if (
            self._worker_close_calls != 1
            or self._ray_shutdown_calls != 1
            or not self._ray_shutdown_complete
        ):
            raise ServerFactoryError("Capsule worker/Ray release is incomplete")
        return {
            "worker_close_calls": self._worker_close_calls,
            "ray_shutdown_calls": self._ray_shutdown_calls,
            "ray_shutdown_complete": True,
        }


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


def _bind_project_import_root(project_root: Path) -> None:
    """Keep the repository package root ahead of VeRL for spawned main-module imports."""

    root = project_root.expanduser().resolve()
    sys.path[:] = [
        entry for entry in sys.path if Path(entry or ".").resolve() != root
    ]
    sys.path.insert(0, str(root))


def _pinned_ray_runtime_env(
    runtime_env: object,
    project_root: Path,
    verl_path: Path,
    expected_sha: str,
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
    project = str(project_root.expanduser().resolve())
    pinned = str(verl_path.expanduser().resolve())
    explicit_paths = {Path(project), Path(pinned)}
    path_entries = [entry for entry in existing_pythonpath.split(os.pathsep) if entry]
    path_entries = [
        entry for entry in path_entries if Path(entry).resolve() not in explicit_paths
    ]
    env_vars["PYTHONPATH"] = os.pathsep.join((project, pinned, *path_entries))
    env_vars["CAPX_PINNED_VERL_SOURCE_PATH"] = pinned
    env_vars["CAPX_PINNED_VERL_SHA"] = expected_sha
    env_vars["PYTHONDONTWRITEBYTECODE"] = "1"
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


@dataclass(frozen=True)
class VeRLWorkerTopology:
    """Resolved worker layout and the LoRA identity bound to that layout."""

    worker_roles: tuple[str, ...]
    reference_policy_mode: str
    lora_rank: int
    lora_alpha: int
    lora_target_modules: tuple[str, ...]

    @property
    def shares_actor_reference(self) -> bool:
        return self.lora_rank > 0


def _config_value(node: Any, name: str, default: Any) -> Any:
    if isinstance(node, Mapping):
        return node.get(name, default)
    getter = getattr(node, "get", None)
    if callable(getter):
        return getter(name, default)
    return getattr(node, name, default)


def _resolve_verl_worker_topology(verl_config: Any) -> VeRLWorkerTopology:
    """Validate the supported VeRL layout before Ray is initialized."""

    actor_rollout_ref = verl_config.actor_rollout_ref
    model = actor_rollout_ref.model
    actor = actor_rollout_ref.actor
    rollout = actor_rollout_ref.rollout
    strategy = str(_config_value(actor, "strategy", ""))
    if strategy not in {"fsdp", "fsdp2"}:
        raise ServerFactoryError("Cube Stack Capsule MVP supports only VeRL FSDP/FSDP2 workers")

    lora_rank = _config_value(model, "lora_rank", 0)
    if isinstance(lora_rank, bool) or not isinstance(lora_rank, int) or lora_rank < 0:
        raise ServerFactoryError(
            "VeRL actor_rollout_ref.model.lora_rank must be a non-negative integer"
        )
    if lora_rank == 0:
        return VeRLWorkerTopology(
            worker_roles=("actor_rollout", "ref"),
            reference_policy_mode="standalone",
            lora_rank=0,
            lora_alpha=0,
            lora_target_modules=(),
        )

    lora_alpha = _config_value(model, "lora_alpha", None)
    if isinstance(lora_alpha, bool) or not isinstance(lora_alpha, int) or lora_alpha < 1:
        raise ServerFactoryError(
            "VeRL actor_rollout_ref.model.lora_alpha must be a positive integer when "
            "LoRA is enabled"
        )
    raw_targets = _config_value(model, "target_modules", None)
    if isinstance(raw_targets, str):
        target_modules = (raw_targets.strip(),) if raw_targets.strip() else ()
    elif isinstance(raw_targets, Sequence):
        target_modules = tuple(
            str(target).strip()
            for target in raw_targets
            if isinstance(target, str) and target.strip()
        )
    else:
        target_modules = ()
    if not target_modules:
        raise ServerFactoryError(
            "VeRL actor_rollout_ref.model.target_modules must be non-empty when LoRA is enabled"
        )
    if target_modules != ("all-linear",):
        raise ServerFactoryError("Capsule LoRA target_modules must resolve to all-linear")
    if str(_config_value(rollout, "name", "")).lower() != "vllm":
        raise ServerFactoryError("Capsule LoRA requires the vLLM rollout backend")
    if str(_config_value(rollout, "load_format", "")).lower() != "safetensors":
        raise ServerFactoryError("Capsule LoRA requires vLLM load_format=safetensors")

    world_size = verl_config.trainer.n_gpus_per_node * verl_config.trainer.nnodes
    if world_size == 1:
        parallel_sizes = {
            "tensor_model_parallel_size": _config_value(
                rollout, "tensor_model_parallel_size", 1
            ),
            "data_parallel_size": _config_value(rollout, "data_parallel_size", 1),
            "pipeline_model_parallel_size": _config_value(
                rollout, "pipeline_model_parallel_size", 1
            ),
        }
        invalid = {
            name: value
            for name, value in parallel_sizes.items()
            if isinstance(value, bool) or not isinstance(value, int) or value != 1
        }
        if invalid:
            raise ServerFactoryError(
                "single-GPU Capsule LoRA requires tensor/data/pipeline parallel sizes of 1; "
                f"got {invalid!r}"
            )
    return VeRLWorkerTopology(
        worker_roles=("actor_rollout",),
        reference_policy_mode="actor_base_adapter_disabled",
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_target_modules=target_modules,
    )


def _verify_lora_reference_source_contract(verl_path: Path) -> None:
    """Prove the pinned FSDP actor maps ``is_lora`` to PEFT ``disable_adapter``."""

    worker_path = verl_path / "verl" / "workers" / "fsdp_workers.py"
    try:
        worker_tree = ast.parse(
            worker_path.read_text(encoding="utf-8"), filename=str(worker_path)
        )
    except (OSError, SyntaxError, UnicodeError) as error:
        raise ServerFactoryError(
            f"cannot inspect VeRL LoRA reference implementation at {worker_path}: {error}"
        ) from error

    def assigned_name(node: ast.Assign) -> str | None:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return None
        return node.targets[0].id

    def references_name(node: ast.AST, names: set[str]) -> bool:
        return any(
            isinstance(child, ast.Name) and child.id in names
            for child in ast.walk(node)
        )

    def calls_disable_adapter(node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "disable_adapter"
            for child in ast.walk(node)
        )

    for node in ast.walk(worker_tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != "compute_log_prob":
            continue
        marker_names: set[str] = set()
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign):
                continue
            name = assigned_name(child)
            value = child.value
            if (
                name is not None
                and isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "pop"
                and value.args
                and isinstance(value.args[0], ast.Constant)
                and value.args[0].value == "is_lora"
            ):
                marker_names.add(name)
        adapter_context_names: set[str] = set()
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign):
                continue
            name = assigned_name(child)
            value = child.value
            if (
                name is not None
                and isinstance(value, ast.IfExp)
                and references_name(value.test, marker_names)
                and calls_disable_adapter(value.body)
                and not calls_disable_adapter(value.orelse)
            ):
                adapter_context_names.add(name)
        enters_adapter_context = any(
            isinstance(child, (ast.With, ast.AsyncWith))
            and any(
                references_name(item.context_expr, adapter_context_names)
                for item in child.items
            )
            for child in ast.walk(node)
        )
        if marker_names and adapter_context_names and enters_adapter_context:
            return
    raise ServerFactoryError(
        "pinned VeRL compute_log_prob must map is_lora to disable_adapter for the "
        "frozen-base LoRA reference policy"
    )


def _initialize_verl_worker_groups(
    groups: Mapping[str, Any], topology: VeRLWorkerTopology
) -> tuple[Any, Any]:
    """Initialize every physical worker exactly once and bind its logical roles."""

    missing = set(topology.worker_roles).difference(groups)
    if missing:
        raise ServerFactoryError(f"VeRL worker spawn omitted roles: {sorted(missing)!r}")
    actor_rollout_wg = groups["actor_rollout"]
    if topology.shares_actor_reference:
        actor_rollout_wg.init_model()
        return actor_rollout_wg, actor_rollout_wg
    ref_policy_wg = groups["ref"]
    ref_policy_wg.init_model()
    actor_rollout_wg.init_model()
    return actor_rollout_wg, ref_policy_wg


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
    topology = _resolve_verl_worker_topology(verl_config)
    if topology.shares_actor_reference:
        _verify_lora_reference_source_contract(verl_path)
    _bind_pinned_verl_import(verl_path)
    _bind_project_import_root(project_root)

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
        merged_runtime_env, project_root, verl_path, expected_verl_sha
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

            @register(dispatch_mode=Dispatch.ONE_TO_ALL)
            def get_capsule_lora_runtime_evidence(self) -> dict[str, Any]:
                from capx.rl.capsule.server_factory import (
                    _actor_lora_runtime_record,
                    _host_mem_available_bytes,
                )

                actor_model = getattr(self, "actor_module_fsdp", None)
                if actor_model is None:
                    raise RuntimeError("actor worker has no FSDP actor module")
                return _actor_lora_runtime_record(
                    actor_model,
                    rank=int(self.rank),
                    cuda_peak_reserved_loader=lambda: int(
                        torch.cuda.max_memory_reserved()
                    ),
                    host_mem_available_loader=_host_mem_available_bytes,
                )

        remote_worker = ray.remote(CapsuleActorRolloutRefWorker)
        global_pool = "capsule_global_pool"
        mapping = {Role.ActorRollout: global_pool}
        if not topology.shares_actor_reference:
            mapping[Role.RefPolicy] = global_pool
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
            )
        }
        if not topology.shares_actor_reference:
            class_dict["ref"] = RayClassWithInitArgs(
                remote_worker,
                config=verl_config.actor_rollout_ref,
                role="ref",
            )
        worker_cls = create_colocated_worker_cls(class_dict=class_dict)
        worker_group = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_cls,
            device_name=verl_config.trainer.device,
        )
        groups = worker_group.spawn(prefix_set=class_dict.keys())
        actor_rollout_wg, ref_policy_wg = _initialize_verl_worker_groups(
            groups, topology
        )
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
            reference_policy_mode=topology.reference_policy_mode,
            lora_rank=topology.lora_rank,
            lora_alpha=topology.lora_alpha,
            lora_target_modules=topology.lora_target_modules,
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
                max_output_tokens=int(controller_service.get("max_output_tokens", 512)),
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
                reference_policy_mode=getattr(
                    workers, "reference_policy_mode", "standalone"
                ),
                artifact_sink=AtomicJsonArtifactSink(output_dir / "groups"),
                config=self.config,
                max_group_attempts=capsule["max_group_attempts"],
            )
            claim_root = output_dir / "checkpoints" / safe_run_id
            checkpoint = claim_root / "final" / "actor"
            with AtomicCheckpointClaim(checkpoint, claim_root=claim_root) as checkpoint_claim:
                verl_provenance_before = workers.verl_provenance()
                optimizer_step_before = workers.optimizer_step()
                try:
                    results = trainer.fit(scheduled_tasks)
                except GroupAttemptBudgetExhausted as error:
                    discarded_group_attempts = tuple(
                        getattr(trainer, "discarded_groups", ())
                    )
                    discard_audit = _write_discard_audit(
                        discard_audit_path, discarded_group_attempts
                    )
                    raise ServerFactoryError(
                        f"{error}; no checkpoint was written; inspect {discard_audit}"
                    ) from error
                actor_updates = sum(not result.skipped_actor_update for result in results)
                discarded_group_attempts = tuple(
                    getattr(trainer, "discarded_groups", ())
                )
                discard_audit = _write_discard_audit(
                    discard_audit_path, discarded_group_attempts
                )
                if len(results) != len(scheduled_tasks):
                    raise ServerFactoryError(
                        f"training completed {len(results)} of {len(scheduled_tasks)} scheduled "
                        "groups; no checkpoint was written"
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
                if results and actor_updates == 0
                else "completed"
            )
            return {
                "status": run_status,
                "task_count": len(tasks),
                "scheduled_group_count": len(scheduled_tasks),
                "step_count": len(results),
                "completed_group_count": len(results),
                "actor_updates": actor_updates,
                "skipped_actor_updates": sum(result.skipped_actor_update for result in results),
                "discarded_groups": len(discarded_group_attempts),
                "discarded_group_attempts": len(discarded_group_attempts),
                "discard_reasons": [record.reason for record in discarded_group_attempts],
                "discarded_group_records": [
                    record.to_dict() for record in discarded_group_attempts
                ],
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
