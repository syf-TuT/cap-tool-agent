"""Project-owned one-step Capsule-Critique trainer orchestration.

The module intentionally depends only on the public shape of VeRL worker groups.  It accepts
plain mappings in unit tests and DataProto-like objects on the server, without importing VeRL.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .group import (
    BASE_GROUP_SIZE,
    GroupAssemblyResult,
    GroupDiscarded,
    RepairAttempt,
    deterministic_group_uid,
    token_levenshtein_distance,
)
from .telemetry import summarize_replay_results
from .schema import ProgramReplayResultV1, ReplayOutcome, TaskInstanceV1

_CONFIG_MISSING = object()
GUIDED_TOKEN_MASK_FIELD = "guided_token_mask"
VERL_MASK_SLOT = "rollout_is_weights"
VERL_DISABLE_ADAPTER_META_FIELD = "is_lora"
REFERENCE_POLICY_MODES = frozenset({"standalone", "actor_base_adapter_disabled"})


class GroupAssembler(Protocol):
    def assemble(self, task: TaskInstanceV1) -> GroupAssemblyResult: ...


class BatchEncoder(Protocol):
    def encode(self, prompts: tuple[str, ...], responses: tuple[str, ...]) -> Any: ...


class TextTokenizer(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


class ArtifactSink(Protocol):
    def write(self, artifact: TrainingStepArtifact) -> None: ...


@dataclass(frozen=True)
class TrainingStepArtifact:
    assembly: GroupAssemblyResult
    guided_token_mask: tuple[tuple[bool, ...], ...]
    sequence_rewards: tuple[float, ...]
    sequence_advantages: tuple[float, ...]
    skipped_actor_update: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "assembly": self.assembly.to_dict(),
            GUIDED_TOKEN_MASK_FIELD: [list(row) for row in self.guided_token_mask],
            "sequence_rewards": list(self.sequence_rewards),
            "sequence_advantages": list(self.sequence_advantages),
            "skipped_actor_update": self.skipped_actor_update,
        }


@dataclass(frozen=True)
class TrainingStepResult:
    artifact: TrainingStepArtifact
    batch: Any
    actor_output: Any | None
    events: tuple[str, ...]
    execution_trace: tuple[str, ...]
    skipped_actor_update: bool


@dataclass(frozen=True)
class DiscardedGroupRecord:
    """Auditable identity and reason for a seed-local group rejected during collection."""

    task_index: int
    task_id: str
    environment_seed: int
    initial_state_sha256: str
    reason: str
    message: str
    group_attempt_index: int = 0
    replay_results: tuple[ProgramReplayResultV1, ...] = ()
    partial_repair_attempts: tuple[RepairAttempt, ...] = ()
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_index": self.task_index,
            "task_id": self.task_id,
            "environment_seed": self.environment_seed,
            "initial_state_sha256": self.initial_state_sha256,
            "group_attempt_index": self.group_attempt_index,
            "reason": self.reason,
            "message": self.message,
            "replay_results": [result.to_dict() for result in self.replay_results],
            "partial_repair_attempts": [
                attempt.to_dict() for attempt in self.partial_repair_attempts
            ],
            **summarize_replay_results(self.replay_results),
        }


class GroupAttemptBudgetExhausted(RuntimeError):
    """A scheduled task could not produce one valid group within its attempt budget."""

    def __init__(self, task: TaskInstanceV1, max_group_attempts: int) -> None:
        super().__init__(
            f"group collection for {task.task_id} seed {task.environment_seed} exhausted "
            f"after {max_group_attempts} attempts"
        )
        self.task = task
        self.max_group_attempts = max_group_attempts


class MemoryArtifactSink:
    def __init__(self) -> None:
        self.artifacts: list[TrainingStepArtifact] = []

    def write(self, artifact: TrainingStepArtifact) -> None:
        self.artifacts.append(artifact)


class TokenBudgetExceeded(ValueError):
    """Raised instead of silently truncating an injected prompt or program."""

    def __init__(self, field: str, observed_tokens: int, token_limit: int) -> None:
        super().__init__(
            f"{field} uses {observed_tokens} tokens, exceeding the {token_limit}-token limit; "
            "the value was not truncated"
        )
        self.field = field
        self.observed_tokens = observed_tokens
        self.token_limit = token_limit


class TokenizerGroupEncoder:
    """Encode complete prompt/program pairs using VeRL's fixed padding convention.

    Prompts are left padded and responses are right padded.  Limits include the response EOS
    token and are hard rejections, which keeps injected source byte-complete.
    """

    def __init__(
        self,
        tokenizer: TextTokenizer,
        *,
        prompt_token_limit: int = 8192,
        response_token_limit: int = 2048,
        batch_factory: Callable[[dict[str, torch.Tensor]], Any] | None = None,
    ) -> None:
        if prompt_token_limit < 1 or response_token_limit < 1:
            raise ValueError("token limits must be positive")
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        for name, token_id in (("pad_token_id", pad_token_id), ("eos_token_id", eos_token_id)):
            if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                raise ValueError(f"tokenizer.{name} must be a non-negative integer")
        self.tokenizer = tokenizer
        self.prompt_token_limit = prompt_token_limit
        self.response_token_limit = response_token_limit
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.batch_factory = batch_factory

    def _tokenize(self, text: str, *, field: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError(f"{field} must be text")
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not isinstance(token_ids, list) or any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in token_ids
        ):
            raise TypeError("tokenizer.encode must return a list of non-negative integer token ids")
        return token_ids

    def encode(
        self,
        prompts: tuple[str, ...],
        responses: tuple[str, ...],
    ) -> Any:
        import torch

        if not prompts or len(prompts) != len(responses):
            raise ValueError("prompts and responses must have the same non-zero batch size")

        prompt_rows: list[list[int]] = []
        response_rows: list[list[int]] = []
        prompt_lengths: list[int] = []
        response_lengths: list[int] = []
        for row, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
            prompt_ids = self._tokenize(prompt, field=f"prompt[{row}]")
            if len(prompt_ids) > self.prompt_token_limit:
                raise TokenBudgetExceeded("prompt", len(prompt_ids), self.prompt_token_limit)
            response_ids = self._tokenize(response, field=f"response[{row}]")
            if not response_ids:
                raise ValueError(f"empty response at row {row}")
            response_ids = [*response_ids, self.eos_token_id]
            if len(response_ids) > self.response_token_limit:
                raise TokenBudgetExceeded("response", len(response_ids), self.response_token_limit)

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
        for row, (prompt_length, response_length) in enumerate(
            zip(prompt_lengths, response_lengths, strict=True)
        ):
            prompt_mask[row, self.prompt_token_limit - prompt_length :] = True
            response_mask[row, :response_length] = True
        attention_mask = torch.cat((prompt_mask, response_mask), dim=-1)
        input_ids = torch.cat((prompt_tensor, response_tensor), dim=-1)
        position_ids = torch.clamp(attention_mask.long().cumsum(dim=-1) - 1, min=0)
        tensors = {
            "prompts": prompt_tensor,
            "responses": response_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
        }
        if self.batch_factory is None:
            return tensors
        return self.batch_factory(tensors)


def _fsync_directory(path: Path) -> None:
    """Best-effort directory sync after publishing an artifact.

    POSIX filesystems need this for the new hard-link directory entry to survive a power loss.
    Windows and some network filesystems do not permit opening directories, so unsupported
    platforms safely retain the file-level fsync and atomic no-replace link semantics.
    """

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            return
    finally:
        os.close(descriptor)


class AtomicJsonArtifactSink:
    """Write immutable JSON artifacts with a deterministic sink-local sequence."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self._next_sequence: int | None = None
        self._sequence_lock = threading.Lock()

    def _claim_sequence(self) -> int:
        with self._sequence_lock:
            if self._next_sequence is None:
                highest_sequence = -1
                for path in self.output_dir.glob("*.json"):
                    match = re.match(r"^(\d+)-.*\.json$", path.name)
                    if match is not None:
                        highest_sequence = max(highest_sequence, int(match.group(1)))
                self._next_sequence = highest_sequence + 1
            sequence = self._next_sequence
            self._next_sequence += 1
            return sequence

    def write(self, artifact: TrainingStepArtifact) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        uid = artifact.assembly.group.group_uid
        safe_uid = re.sub(r"[^A-Za-z0-9_.-]", "_", uid)
        temporary = self.output_dir / f".{safe_uid}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(
                    artifact.to_dict(),
                    stream,
                    allow_nan=False,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            while True:
                sequence = self._claim_sequence()
                destination = self.output_dir / f"{sequence:08d}-{safe_uid}.json"
                try:
                    os.link(temporary, destination)
                    break
                except FileExistsError:
                    # Another sink may have claimed this sequence after our directory scan.
                    continue
            _fsync_directory(self.output_dir)
        finally:
            temporary.unlink(missing_ok=True)


def _tensor_batch(batch: Any) -> Any:
    tensors = getattr(batch, "batch", batch)
    if not hasattr(tensors, "__getitem__") or not hasattr(tensors, "__setitem__"):
        raise TypeError("encoded batch must be a mutable mapping or DataProto-like object")
    return tensors


def _merge_batch(batch: Any, addition: Any) -> Any:
    if addition is None:
        return batch
    union = getattr(batch, "union", None)
    if callable(union):
        merged = union(addition)
        return batch if merged is None else merged
    tensors = _tensor_batch(batch)
    source = getattr(addition, "batch", addition)
    if not isinstance(source, Mapping):
        raise TypeError("worker output must be a mapping or DataProto-like object")
    tensors.update(source)
    return batch


def _nested_config_value(config: Any, path: tuple[str, ...], default: Any = None) -> Any:
    current = config
    for key in path:
        if isinstance(current, Mapping):
            value = current.get(key, _CONFIG_MISSING)
        else:
            value = getattr(current, key, _CONFIG_MISSING)
            if value is _CONFIG_MISSING:
                getter = getattr(current, "get", None)
                if callable(getter):
                    try:
                        value = getter(key, _CONFIG_MISSING)
                    except TypeError:
                        value = _CONFIG_MISSING
        if value is _CONFIG_MISSING:
            return default
        current = value
    return current


def _mask_tuple(mask: torch.Tensor) -> tuple[tuple[bool, ...], ...]:
    return tuple(tuple(bool(value) for value in row) for row in mask.detach().cpu().tolist())


class CapsuleCritiqueRayTrainer:
    """Drive one verified group through VeRL-style actor workers.

    Collection remains inside the injected assembler.  This boundary ensures P0 and repair
    traces are exploration-only: the encoder receives only eight copies of the original prompt
    and the eight final program responses.
    """

    def __init__(
        self,
        *,
        assembler: GroupAssembler,
        batch_encoder: BatchEncoder,
        actor_rollout_wg: Any,
        artifact_sink: ArtifactSink,
        config: Any,
        ref_policy_wg: Any | None = None,
        reference_policy_mode: str = "standalone",
        event_log: list[str] | None = None,
        max_group_attempts: int = 3,
    ) -> None:
        from .policy_loss import validate_capsule_training_config

        validate_capsule_training_config(config)
        reference_kl_enabled = _nested_config_value(
            config,
            ("actor_rollout_ref", "actor", "use_kl_loss"),
            False,
        )
        if reference_kl_enabled is True and ref_policy_wg is None:
            raise ValueError("ref_policy_wg is required when actor reference KL is enabled")
        if reference_policy_mode not in REFERENCE_POLICY_MODES:
            raise ValueError(
                "reference_policy_mode must be standalone or actor_base_adapter_disabled"
            )
        if (
            isinstance(max_group_attempts, bool)
            or not isinstance(max_group_attempts, int)
            or max_group_attempts < 1
        ):
            raise ValueError("max_group_attempts must be a positive integer")
        if (
            reference_policy_mode == "actor_base_adapter_disabled"
            and ref_policy_wg is not actor_rollout_wg
        ):
            raise ValueError(
                "actor_base_adapter_disabled reference mode requires the same actor worker"
            )
        self.assembler = assembler
        self.batch_encoder = batch_encoder
        self.actor_rollout_wg = actor_rollout_wg
        self.ref_policy_wg = ref_policy_wg
        self.reference_policy_mode = reference_policy_mode
        self.artifact_sink = artifact_sink
        self.config = config
        self.events = event_log if event_log is not None else []
        self.max_group_attempts = max_group_attempts
        self._discarded_groups: list[DiscardedGroupRecord] = []
        self._actor_updates_completed = 0

    @property
    def discarded_groups(self) -> tuple[DiscardedGroupRecord, ...]:
        return tuple(self._discarded_groups)

    @property
    def discarded_count(self) -> int:
        return len(self._discarded_groups)

    @property
    def actor_updates_completed(self) -> int:
        return self._actor_updates_completed

    @property
    def discard_reasons(self) -> tuple[str, ...]:
        return tuple(record.reason for record in self._discarded_groups)

    @staticmethod
    def _validate_group(task: TaskInstanceV1, assembly: GroupAssemblyResult) -> None:
        group = assembly.group
        if len(group.members) != BASE_GROUP_SIZE:
            raise ValueError("Capsule trainer requires exactly eight final members")
        if (
            group.task_id != task.task_id
            or group.environment_seed != task.environment_seed
            or group.initial_state_sha256 != task.initial_state_sha256
        ):
            raise ValueError("learning group identity does not match the task instance")
        if group.group_uid != deterministic_group_uid(task):
            raise ValueError("learning group UID does not match the complete task instance")
        for member in group.members:
            if member.prompt != task.prompt:
                raise ValueError("every final member must train under the original task prompt")
        rewards = tuple(member.reward for member in group.members)
        expected_skip = len(set(rewards)) == 1
        if group.skip_actor_update is not expected_skip:
            raise ValueError(
                "skip_actor_update must equal the constant-reward status computed from members"
            )
        guided_indices = tuple(
            index
            for index, member in enumerate(group.members)
            if member.member_type == "critique_guided_revision"
        )
        if len(guided_indices) > 1:
            raise ValueError("a 7+1 group may contain at most one guided member")
        if guided_indices:
            if guided_indices != (BASE_GROUP_SIZE - 1,):
                raise ValueError("the guided member must be the eighth and final member")
            guided = group.members[-1]
            if guided.reward != 1.0:
                raise ValueError("the guided member must have binary reward 1")
            if not guided.repair_trajectory_id:
                raise ValueError("the guided member must retain its repair trajectory id")
            if any(member.reward != 0.0 for member in group.members[:-1]):
                raise ValueError("a guided 7+1 group requires seven failed base members")

        expected_base_count = BASE_GROUP_SIZE - len(guided_indices)
        if len(assembly.base_results) != expected_base_count:
            raise ValueError(
                f"base_results must contain exactly {expected_base_count} typed clean replays"
            )
        for index, (member, replay) in enumerate(
            zip(group.members[:expected_base_count], assembly.base_results, strict=True)
        ):
            if member.member_type != "base":
                raise ValueError(f"base_results[{index}] does not correspond to a base member")
            if not isinstance(replay, ProgramReplayResultV1):
                raise TypeError(f"base_results[{index}] must be ProgramReplayResultV1")
            actual_identity = (
                replay.task_id,
                replay.environment_seed,
                replay.program_sample_id,
                replay.source,
                replay.initial_state_sha256,
            )
            expected_identity = (
                task.task_id,
                task.environment_seed,
                member.program_sample_id,
                member.response,
                task.initial_state_sha256,
            )
            if actual_identity != expected_identity:
                raise ValueError(f"base_results[{index}] identity does not match its member")
            if replay.binary_reward is None or float(replay.binary_reward) != member.reward:
                raise ValueError(f"base_results[{index}] reward does not match its member")
            if (replay.outcome is ReplayOutcome.SUCCESS) != (member.reward == 1.0):
                raise ValueError(f"base_results[{index}] outcome does not match its member")

        repair_triggered = all(member.reward == 0.0 for member in group.members[:7])
        expected_attempt_count = 4 if repair_triggered else 0
        if len(assembly.repair_attempts) != expected_attempt_count:
            raise ValueError(
                f"repair_attempts must contain exactly {expected_attempt_count} attempts"
            )
        base_by_sample_id = {
            replay.program_sample_id: replay for replay in assembly.base_results[:7]
        }
        p0_ids_by_rank: dict[int, str] = {}
        selected_attempts: list[RepairAttempt] = []
        allowed_statuses = {
            "rejected",
            "pt_failed",
            "revision_failed",
            "guided_success",
        }
        for expected_index, attempt in enumerate(assembly.repair_attempts):
            if not isinstance(attempt, RepairAttempt):
                raise TypeError(f"repair_attempts[{expected_index}] must be RepairAttempt")
            expected_rank, expected_trajectory = divmod(expected_index, 2)
            expected_trajectory_id = (
                f"{group.group_uid}:p0-{expected_rank}:trajectory-{expected_trajectory}"
            )
            if (
                attempt.p0_rank != expected_rank
                or attempt.trajectory_index != expected_trajectory
                or attempt.repair_trajectory_id != expected_trajectory_id
            ):
                raise ValueError(
                    f"repair_attempts[{expected_index}] rank/index/trajectory identity mismatch"
                )
            if attempt.status not in allowed_statuses:
                raise ValueError(f"repair_attempts[{expected_index}] has an invalid status")
            p0_replay = base_by_sample_id.get(attempt.p0_program_sample_id)
            if p0_replay is None or p0_replay.outcome is ReplayOutcome.SUCCESS:
                raise ValueError(
                    f"repair_attempts[{expected_index}] does not reference a failed base replay"
                )
            previous_p0_id = p0_ids_by_rank.setdefault(
                attempt.p0_rank, attempt.p0_program_sample_id
            )
            if previous_p0_id != attempt.p0_program_sample_id:
                raise ValueError("both trajectories for one P0 rank must share the same P0")

            trace = attempt.trace
            trace_is_rejected_audit = (
                attempt.status == "rejected"
                and attempt.rejection_reason == "trace_mismatch"
            )
            if trace is not None and not trace_is_rejected_audit:
                trace_identity = (
                    trace.task_id,
                    trace.environment_seed,
                    trace.program_sample_id,
                    trace.repair_trajectory_id,
                    trace.base_source,
                )
                expected_trace_identity = (
                    task.task_id,
                    task.environment_seed,
                    p0_replay.program_sample_id,
                    attempt.repair_trajectory_id,
                    p0_replay.source,
                )
                if trace_identity != expected_trace_identity:
                    raise ValueError(
                        f"repair_attempts[{expected_index}] trace identity mismatch"
                    )
                if trace.reconstruct() != trace.final_source:
                    raise ValueError(
                        f"repair_attempts[{expected_index}] trace does not reconstruct PT"
                    )

            pt_replay = attempt.pt_result
            if pt_replay is not None:
                if trace is None:
                    raise ValueError(
                        f"repair_attempts[{expected_index}] PT replay requires a repair trace"
                    )
                pt_identity = (
                    pt_replay.task_id,
                    pt_replay.environment_seed,
                    pt_replay.program_sample_id,
                    pt_replay.source,
                    pt_replay.initial_state_sha256,
                )
                expected_pt_identity = (
                    task.task_id,
                    task.environment_seed,
                    f"{attempt.repair_trajectory_id}:pt",
                    trace.final_source,
                    task.initial_state_sha256,
                )
                if pt_identity != expected_pt_identity or pt_replay.binary_reward is None:
                    raise ValueError(
                        f"repair_attempts[{expected_index}] PT replay identity/reward mismatch"
                    )

            revision_replay = attempt.revision_result
            if revision_replay is not None:
                if (
                    attempt.revision_program_sample_id is None
                    or attempt.revision_source is None
                ):
                    raise ValueError(
                        f"repair_attempts[{expected_index}] revision replay lacks generated source"
                    )
                revision_identity = (
                    revision_replay.task_id,
                    revision_replay.environment_seed,
                    revision_replay.program_sample_id,
                    revision_replay.source,
                    revision_replay.initial_state_sha256,
                )
                expected_revision_identity = (
                    task.task_id,
                    task.environment_seed,
                    attempt.revision_program_sample_id,
                    attempt.revision_source,
                    task.initial_state_sha256,
                )
                if (
                    revision_identity != expected_revision_identity
                    or revision_replay.binary_reward is None
                ):
                    raise ValueError(
                        f"repair_attempts[{expected_index}] revision replay "
                        "identity/reward mismatch"
                    )

            if attempt.status == "pt_failed" and (
                pt_replay is None or pt_replay.outcome is ReplayOutcome.SUCCESS
            ):
                raise ValueError("pt_failed attempt must retain a failed PT replay")
            if attempt.status == "revision_failed" and (
                pt_replay is None
                or pt_replay.outcome is not ReplayOutcome.SUCCESS
                or revision_replay is None
                or revision_replay.outcome is ReplayOutcome.SUCCESS
            ):
                raise ValueError(
                    "revision_failed attempt must retain successful PT and failed revision replays"
                )
            if attempt.status == "guided_success" and (
                trace is None
                or pt_replay is None
                or pt_replay.outcome is not ReplayOutcome.SUCCESS
                or revision_replay is None
                or revision_replay.outcome is not ReplayOutcome.SUCCESS
            ):
                raise ValueError(
                    "guided_success attempt requires trace plus successful PT and revision replays"
                )
            if (
                revision_replay is not None
                and revision_replay.outcome is ReplayOutcome.SUCCESS
                and attempt.status != "guided_success"
            ):
                raise ValueError(
                    "a successful revision replay must be recorded as guided_success"
                )
            if attempt.selected:
                if attempt.status != "guided_success":
                    raise ValueError("only a guided_success attempt may be selected")
                selected_attempts.append(attempt)

        if repair_triggered:
            def reward_key(index: int) -> tuple[float, int]:
                raw_reward = assembly.base_results[index].raw_reward
                return (
                    float("-inf") if raw_reward is None else float(raw_reward),
                    -index,
                )

            first_p0_index = max(range(7), key=reward_key)
            second_p0_index = max(
                (index for index in range(7) if index != first_p0_index),
                key=lambda index: (
                    token_levenshtein_distance(
                        group.members[first_p0_index].response,
                        group.members[index].response,
                    ),
                    -index,
                ),
            )
            expected_p0_ids = (
                group.members[first_p0_index].program_sample_id,
                group.members[second_p0_index].program_sample_id,
            )
            actual_p0_ids = (p0_ids_by_rank.get(0), p0_ids_by_rank.get(1))
            if actual_p0_ids != expected_p0_ids:
                raise ValueError("repair_attempts do not match deterministic P0 selection")
        if guided_indices:
            if len(selected_attempts) != 1:
                raise ValueError("guided 7+1 group requires exactly one selected repair attempt")
            selected = selected_attempts[0]
            first_success_index = next(
                index
                for index, attempt in enumerate(assembly.repair_attempts)
                if attempt.status == "guided_success"
            )
            selected_index = next(
                index
                for index, attempt in enumerate(assembly.repair_attempts)
                if attempt.selected
            )
            if selected_index != first_success_index:
                raise ValueError(
                    "selected repair attempt must be the first successful revision "
                    "in fixed rank/index order"
                )
            guided = group.members[-1]
            if (
                selected.repair_trajectory_id != guided.repair_trajectory_id
                or selected.revision_program_sample_id != guided.program_sample_id
                or selected.revision_source != guided.response
            ):
                raise ValueError("selected revision replay does not match the guided member")
        elif selected_attempts or any(
            attempt.status == "guided_success" for attempt in assembly.repair_attempts
        ):
            raise ValueError("base-only group cannot retain a successful guided repair")

    @staticmethod
    def _validate_encoded_tensors(tensors: Any) -> None:
        import torch

        required = (
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
            "response_mask",
        )
        for name in required:
            if name not in tensors:
                raise KeyError(f"batch encoder must provide {name}")
            if not isinstance(tensors[name], torch.Tensor):
                raise TypeError(f"batch encoder field {name} must be a torch.Tensor")

        input_ids = tensors["input_ids"]
        attention_mask = tensors["attention_mask"]
        position_ids = tensors["position_ids"]
        responses = tensors["responses"]
        response_mask = tensors["response_mask"]
        if input_ids.ndim != 2 or input_ids.shape[0] != BASE_GROUP_SIZE:
            raise ValueError("input_ids must have shape (8, sequence_length)")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask shape must match input_ids")
        if position_ids.shape != input_ids.shape:
            raise ValueError("position_ids shape must match input_ids")
        if responses.ndim != 2 or responses.shape[0] != BASE_GROUP_SIZE:
            raise ValueError("responses must have shape (8, response_length)")
        if response_mask.shape != responses.shape:
            raise ValueError("responses and response_mask must have identical shapes")
        if responses.shape[1] > input_ids.shape[1]:
            raise ValueError("response length cannot exceed the complete input length")
        if response_mask.dtype != torch.bool:
            raise TypeError("batch encoder must provide a boolean response_mask tensor")

    def _inject_group(self, task: TaskInstanceV1, assembly: GroupAssemblyResult) -> Any:
        import numpy as np
        import torch

        from .policy_loss import map_guided_token_mask_to_verl_slot

        group = assembly.group
        prompts = (task.prompt,) * BASE_GROUP_SIZE
        responses = tuple(member.response for member in group.members)
        batch = self.batch_encoder.encode(prompts, responses)
        tensors = _tensor_batch(batch)
        self._validate_encoded_tensors(tensors)
        response_mask = tensors["response_mask"]
        if response_mask.ndim != 2 or response_mask.shape[0] != BASE_GROUP_SIZE:
            raise ValueError("response_mask must have shape (8, response_length)")
        if torch.any(response_mask.sum(dim=-1) == 0).item():
            raise ValueError("every final program must contain at least one response token")

        rewards = torch.tensor(
            [member.reward for member in group.members],
            dtype=torch.float32,
            device=response_mask.device,
        )
        token_scores = torch.zeros(
            response_mask.shape,
            dtype=torch.float32,
            device=response_mask.device,
        )
        for row, reward in enumerate(rewards):
            final_token = int(torch.nonzero(response_mask[row], as_tuple=False)[-1].item())
            token_scores[row, final_token] = reward

        guided_rows = torch.tensor(
            [member.member_type == "critique_guided_revision" for member in group.members],
            dtype=torch.bool,
            device=response_mask.device,
        )
        guided_mask = response_mask & guided_rows.unsqueeze(-1)
        tensors["token_level_scores"] = token_scores
        tensors["token_level_rewards"] = token_scores.clone()
        tensors[GUIDED_TOKEN_MASK_FIELD] = guided_mask
        if VERL_MASK_SLOT in tensors:
            raise ValueError(
                f"{VERL_MASK_SLOT} is already populated; standard rollout importance "
                "sampling must be disabled"
            )
        mapped = map_guided_token_mask_to_verl_slot(
            {
                "response_mask": response_mask,
                GUIDED_TOKEN_MASK_FIELD: guided_mask,
            }
        )
        tensors[VERL_MASK_SLOT] = mapped[VERL_MASK_SLOT]
        uid = (group.group_uid,) * BASE_GROUP_SIZE
        if hasattr(batch, "non_tensor_batch"):
            if not isinstance(batch.non_tensor_batch, dict):
                raise TypeError("DataProto-like non_tensor_batch must be a mutable dictionary")
            batch.non_tensor_batch["uid"] = np.full(
                (BASE_GROUP_SIZE,), group.group_uid, dtype=object
            )
            if not isinstance(batch.meta_info, dict):
                raise TypeError("DataProto-like meta_info must be a mutable dictionary")
            batch.meta_info["global_token_num"] = tensors["attention_mask"].sum(
                dim=-1
            ).tolist()
        else:
            tensors["uid"] = uid
        self.events.append("inject")
        return batch

    def _add_advantages(self, batch: Any, assembly: GroupAssemblyResult) -> tuple[float, ...]:
        import torch

        tensors = _tensor_batch(batch)
        response_mask = tensors["response_mask"]
        rewards = torch.tensor(
            [member.reward for member in assembly.group.members],
            dtype=torch.float32,
            device=response_mask.device,
        )
        sequence_advantages = rewards - rewards.mean()
        advantages = sequence_advantages.unsqueeze(-1) * response_mask.to(torch.float32)
        tensors["advantages"] = advantages
        tensors["returns"] = advantages.clone()
        self.events.append("advantage")
        return tuple(float(value) for value in sequence_advantages.detach().cpu().tolist())

    def _compute_actor_base_reference_log_prob(self, batch: Any) -> Any:
        """Ask the LoRA actor for frozen-base log-probs via VeRL's adapter-disable flag."""

        meta_info = getattr(batch, "meta_info", None)
        if not isinstance(meta_info, dict):
            raise TypeError(
                "actor_base_adapter_disabled reference requires mutable DataProto meta_info"
            )
        if VERL_DISABLE_ADAPTER_META_FIELD in meta_info:
            raise ValueError(
                f"batch meta_info already contains {VERL_DISABLE_ADAPTER_META_FIELD!r}"
            )
        meta_info[VERL_DISABLE_ADAPTER_META_FIELD] = True
        try:
            output = self.ref_policy_wg.compute_log_prob(batch)
        finally:
            meta_info.pop(VERL_DISABLE_ADAPTER_META_FIELD, None)
        tensors = getattr(output, "batch", output)
        if not hasattr(tensors, "__contains__") or "old_log_probs" not in tensors:
            raise TypeError(
                "LoRA actor reference output must contain VeRL old_log_probs"
            )
        if "entropys" in tensors:
            del tensors["entropys"]
        if "ref_log_prob" in tensors:
            raise ValueError("LoRA actor reference output unexpectedly contains ref_log_prob")
        tensors["ref_log_prob"] = tensors.pop("old_log_probs")
        return output

    def run_step(self, task: TaskInstanceV1) -> TrainingStepResult:
        execution_trace: list[str] = []
        assembly = self.assembler.assemble(task)
        validate_group_provenance(task, assembly)
        batch = self._inject_group(task, assembly)

        if not assembly.group.skip_actor_update:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_tensors = _tensor_batch(old_log_prob)
            if "entropys" in old_log_prob_tensors:
                del old_log_prob_tensors["entropys"]
            batch = _merge_batch(batch, old_log_prob)
            execution_trace.append("old_logprob")
            if self.ref_policy_wg is not None:
                if self.reference_policy_mode == "actor_base_adapter_disabled":
                    ref_log_prob = self._compute_actor_base_reference_log_prob(batch)
                else:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                batch = _merge_batch(batch, ref_log_prob)
                execution_trace.append("reference_logprob")

        sequence_advantages = self._add_advantages(batch, assembly)
        tensors = _tensor_batch(batch)
        artifact = TrainingStepArtifact(
            assembly=assembly,
            guided_token_mask=_mask_tuple(tensors[GUIDED_TOKEN_MASK_FIELD]),
            sequence_rewards=tuple(member.reward for member in assembly.group.members),
            sequence_advantages=sequence_advantages,
            skipped_actor_update=assembly.group.skip_actor_update,
        )

        actor_output = None
        if not assembly.group.skip_actor_update:
            actor_output = self.actor_rollout_wg.update_actor(batch)
            self._actor_updates_completed += 1
            execution_trace.append("update")
        self.artifact_sink.write(artifact)
        return TrainingStepResult(
            artifact=artifact,
            batch=batch,
            actor_output=actor_output,
            events=tuple(self.events),
            execution_trace=tuple(execution_trace),
            skipped_actor_update=assembly.group.skip_actor_update,
        )

    def fit(self, tasks: Iterable[TaskInstanceV1]) -> tuple[TrainingStepResult, ...]:
        results: list[TrainingStepResult] = []
        clean_evaluator = getattr(self.assembler, "clean_evaluator", None)
        drain_history = getattr(clean_evaluator, "drain_history", None)
        for task_index, task in enumerate(tasks):
            for group_attempt_index in range(self.max_group_attempts):
                if callable(drain_history):
                    drain_history()
                try:
                    result = self.run_step(task)
                except GroupDiscarded as error:
                    replay_results = drain_history() if callable(drain_history) else ()
                    self._discarded_groups.append(
                        DiscardedGroupRecord(
                            task_index=task_index,
                            task_id=task.task_id,
                            environment_seed=task.environment_seed,
                            initial_state_sha256=task.initial_state_sha256,
                            reason=error.reason,
                            message=str(error),
                            group_attempt_index=group_attempt_index,
                            replay_results=replay_results,
                            partial_repair_attempts=tuple(error.partial_repair_attempts),
                        )
                    )
                    self.events.append(f"discard:{error.reason}")
                    if group_attempt_index + 1 == self.max_group_attempts:
                        raise GroupAttemptBudgetExhausted(
                            task, self.max_group_attempts
                        ) from error
                    continue
                if callable(drain_history):
                    drain_history()
                results.append(result)
                break
        return tuple(results)


# Bind the pure provenance validator once so artifact verification remains independent of any
# runtime trainer factory or test double replacing ``CapsuleCritiqueRayTrainer``.
validate_group_provenance = CapsuleCritiqueRayTrainer._validate_group


__all__ = [
    "AtomicJsonArtifactSink",
    "CapsuleCritiqueRayTrainer",
    "DiscardedGroupRecord",
    "GroupAttemptBudgetExhausted",
    "MemoryArtifactSink",
    "TokenBudgetExceeded",
    "TokenizerGroupEncoder",
    "TrainingStepArtifact",
    "TrainingStepResult",
    "validate_group_provenance",
]
