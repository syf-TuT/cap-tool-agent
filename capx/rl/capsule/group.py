"""Deterministic 7+1 learning-group assembly for Capsule-Critique-GRPO.

The assembler is deliberately framework-neutral.  Sampling, repair, revision generation, and
clean replay are injected callbacks so the collection policy can be tested without importing a
simulator, model server, or VeRL.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Protocol

from .repair import RepairInvariantError
from .revision import (
    RevisionPrompt,
    RevisionRejection,
    TokenCounter,
    build_revision_prompt,
    validate_complete_program,
)
from .schema import (
    LearningGroupV1,
    LearningMemberV1,
    ProgramReplayResultV1,
    RepairTraceV1,
    ReplayOutcome,
    TaskInstanceV1,
    source_sha256,
)

BASE_GROUP_SIZE = 8
REPAIR_TRIGGER_BASE_COUNT = 7
TRAJECTORIES_PER_P0 = 2
_UNKNOWN_REPLAY_OUTCOMES = {ReplayOutcome.INFRA_ERROR, ReplayOutcome.EVALUATOR_ERROR}
_REGEX_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class ProgramCandidate:
    """One immutable actor response before clean replay."""

    program_sample_id: str
    source: str
    finish_reason: str | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.program_sample_id, str) or not self.program_sample_id:
            raise ValueError("program_sample_id must not be empty")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("program source must not be empty")
        if self.finish_reason is not None and not isinstance(self.finish_reason, str):
            raise TypeError("finish_reason must be a string or null")
        if not isinstance(self.truncated, bool):
            raise TypeError("truncated must be a boolean")


class BaseSampler(Protocol):
    def __call__(self, task: TaskInstanceV1, base_index: int) -> ProgramCandidate: ...


class RepairCollector(Protocol):
    def __call__(
        self,
        task: TaskInstanceV1,
        p0: ProgramCandidate,
        p0_result: ProgramReplayResultV1,
        p0_rank: int,
        trajectory_index: int,
        repair_trajectory_id: str,
    ) -> RepairTraceV1: ...


class RevisionGenerator(Protocol):
    def __call__(
        self,
        task: TaskInstanceV1,
        p0: ProgramCandidate,
        trace: RepairTraceV1,
        revision_prompt: RevisionPrompt,
        p0_rank: int,
        trajectory_index: int,
    ) -> ProgramCandidate: ...


class CleanEvaluator(Protocol):
    def __call__(
        self, task: TaskInstanceV1, candidate: ProgramCandidate
    ) -> ProgramReplayResultV1: ...


class GroupDiscarded(RuntimeError):
    """Raised when any member makes the whole seed-local group untrustworthy."""

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        partial_repair_attempts: tuple[RepairAttempt, ...] = (),
    ) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason
        self.partial_repair_attempts = partial_repair_attempts


class CollectionInfrastructureError(RuntimeError):
    """A repair or revision service failure which invalidates the seed group."""


class CandidateCollectionError(ValueError):
    """A single sampled candidate is invalid while the collection runtime remains healthy."""


@dataclass(frozen=True)
class RepairAttempt:
    """Persistable provenance for one fixed P0-rank/trajectory-index attempt."""

    p0_rank: int
    trajectory_index: int
    p0_program_sample_id: str
    repair_trajectory_id: str
    status: str
    trace: RepairTraceV1 | None = None
    pt_result: ProgramReplayResultV1 | None = None
    revision_program_sample_id: str | None = None
    revision_source: str | None = None
    revision_finish_reason: str | None = None
    revision_truncated: bool | None = None
    revision_result: ProgramReplayResultV1 | None = None
    rejection_reason: str | None = None
    rejection_message: str | None = None
    selected: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "p0_rank": self.p0_rank,
            "trajectory_index": self.trajectory_index,
            "p0_program_sample_id": self.p0_program_sample_id,
            "repair_trajectory_id": self.repair_trajectory_id,
            "status": self.status,
            "trace": None if self.trace is None else self.trace.to_dict(),
            "pt_result": None if self.pt_result is None else self.pt_result.to_dict(),
            "revision_program_sample_id": self.revision_program_sample_id,
            "revision_source": self.revision_source,
            "revision_finish_reason": self.revision_finish_reason,
            "revision_truncated": self.revision_truncated,
            "revision_result": (
                None if self.revision_result is None else self.revision_result.to_dict()
            ),
            "rejection_reason": self.rejection_reason,
            "rejection_message": self.rejection_message,
            "selected": self.selected,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RepairAttempt:
        data = dict(payload)
        trace = data.get("trace")
        pt_result = data.get("pt_result")
        revision_result = data.get("revision_result")
        return cls(
            p0_rank=_required_int(data.get("p0_rank"), "p0_rank"),
            trajectory_index=_required_int(
                data.get("trajectory_index"), "trajectory_index"
            ),
            p0_program_sample_id=_required_string(
                data.get("p0_program_sample_id"), "p0_program_sample_id"
            ),
            repair_trajectory_id=_required_string(
                data.get("repair_trajectory_id"), "repair_trajectory_id"
            ),
            status=_required_string(data.get("status"), "status"),
            trace=(
                None
                if trace is None
                else RepairTraceV1.from_dict(_require_mapping(trace, "repair trace"))
            ),
            pt_result=(
                None
                if pt_result is None
                else ProgramReplayResultV1.from_dict(
                    _require_mapping(pt_result, "PT replay result")
                )
            ),
            revision_program_sample_id=_optional_string(
                data.get("revision_program_sample_id"), "revision_program_sample_id"
            ),
            revision_source=_optional_string(data.get("revision_source"), "revision_source"),
            revision_finish_reason=_optional_string(
                data.get("revision_finish_reason"), "revision_finish_reason"
            ),
            revision_truncated=_optional_bool(
                data.get("revision_truncated"), "revision_truncated"
            ),
            revision_result=(
                None
                if revision_result is None
                else ProgramReplayResultV1.from_dict(
                    _require_mapping(revision_result, "revision replay result")
                )
            ),
            rejection_reason=_optional_string(
                data.get("rejection_reason"), "rejection_reason"
            ),
            rejection_message=_optional_string(
                data.get("rejection_message"), "rejection_message"
            ),
            selected=_required_bool(data.get("selected", False), "selected"),
        )


@dataclass(frozen=True)
class GroupAssemblyResult:
    group: LearningGroupV1
    base_results: tuple[ProgramReplayResultV1, ...]
    repair_attempts: tuple[RepairAttempt, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "group": self.group.to_dict(),
            "base_results": [result.to_dict() for result in self.base_results],
            "repair_attempts": [attempt.to_dict() for attempt in self.repair_attempts],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> GroupAssemblyResult:
        data = dict(payload)
        group = LearningGroupV1.from_dict(_require_mapping(data.get("group"), "group"))
        base_results = _require_sequence(data.get("base_results"), "base_results")
        repair_attempts = _require_sequence(data.get("repair_attempts"), "repair_attempts")
        return cls(
            group=group,
            base_results=tuple(
                ProgramReplayResultV1.from_dict(_require_mapping(item, "base replay result"))
                for item in base_results
            ),
            repair_attempts=tuple(
                RepairAttempt.from_dict(_require_mapping(item, "repair attempt"))
                for item in repair_attempts
            ),
        )


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _require_sequence(value: object, field_name: str) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list or tuple")
    return tuple(value)


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or null")
    return value


def _required_string(value: object, field_name: str) -> str:
    selected = _optional_string(value, field_name)
    if not selected:
        raise ValueError(f"{field_name} must be non-empty")
    return selected


def _required_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _required_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean")
    return value


def _optional_bool(value: object, field_name: str) -> bool | None:
    if value is None:
        return None
    return _required_bool(value, field_name)


def regex_tokens(source: str) -> tuple[str, ...]:
    """Tokenize source deterministically without depending on an actor tokenizer."""

    if not isinstance(source, str):
        raise TypeError("source must be a string")
    return tuple(_REGEX_TOKEN_PATTERN.findall(source))


def token_levenshtein_distance(left: str, right: str) -> int:
    """Return Levenshtein edit distance over regex tokens using linear memory."""

    left_tokens = regex_tokens(left)
    right_tokens = regex_tokens(right)
    if len(left_tokens) > len(right_tokens):
        left_tokens, right_tokens = right_tokens, left_tokens
    previous = list(range(len(left_tokens) + 1))
    for right_index, right_token in enumerate(right_tokens, start=1):
        current = [right_index]
        for left_index, left_token in enumerate(left_tokens, start=1):
            insertion = current[-1] + 1
            deletion = previous[left_index] + 1
            substitution = previous[left_index - 1] + (left_token != right_token)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def deterministic_group_uid(task: TaskInstanceV1) -> str:
    """Bind a group UID to the complete immutable typed task, including collection metadata."""

    payload = json.dumps(
        {"namespace": "capsule-group-v1", "task": task.to_dict()},
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"capsule-v1-{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _default_token_counter(text: str) -> int:
    return len(text.split())


class CapsuleGroupAssembler:
    """Assemble exactly eight seed-local members under the Capsule-Critique policy."""

    def __init__(
        self,
        *,
        base_sampler: BaseSampler,
        repair_collector: RepairCollector,
        revision_generator: RevisionGenerator,
        clean_evaluator: CleanEvaluator,
        token_counter: TokenCounter | None = None,
        revision_prompt_token_counter: TokenCounter | None = None,
        revision_response_token_counter: TokenCounter | None = None,
        revision_input_token_limit: int = 8192,
        revision_response_token_limit: int = 2048,
    ) -> None:
        if revision_input_token_limit < 1 or revision_response_token_limit < 1:
            raise ValueError("revision token limits must be positive")
        if token_counter is not None and (
            revision_prompt_token_counter is not None
            or revision_response_token_counter is not None
        ):
            raise ValueError(
                "legacy token_counter cannot be combined with distinct revision token counters"
            )
        self.base_sampler = base_sampler
        self.repair_collector = repair_collector
        self.revision_generator = revision_generator
        self.clean_evaluator = clean_evaluator
        if token_counter is not None:
            revision_prompt_token_counter = token_counter
            revision_response_token_counter = token_counter
        self.revision_prompt_token_counter = (
            revision_prompt_token_counter or _default_token_counter
        )
        self.revision_response_token_counter = (
            revision_response_token_counter or _default_token_counter
        )
        self.revision_input_token_limit = revision_input_token_limit
        self.revision_response_token_limit = revision_response_token_limit

    def _count_revision_response_tokens_with_eos(self, source: str) -> int:
        raw_count = self.revision_response_token_counter(source)
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
            raise TypeError(
                "revision_response_token_counter must return a non-negative integer"
            )
        return raw_count + 1

    @staticmethod
    def _discard(reason: str, message: str) -> None:
        raise GroupDiscarded(reason, message)

    def _evaluate(
        self,
        task: TaskInstanceV1,
        candidate: ProgramCandidate,
        *,
        context: str,
    ) -> ProgramReplayResultV1:
        result = self.clean_evaluator(task, candidate)
        if not isinstance(result, ProgramReplayResultV1):
            self._discard("invalid_replay_result", f"{context}: evaluator returned wrong type")
        expected_fields = {
            "task_id": (result.task_id, task.task_id),
            "environment_seed": (result.environment_seed, task.environment_seed),
            "program_sample_id": (result.program_sample_id, candidate.program_sample_id),
            "source": (result.source, candidate.source),
            "source_sha256": (result.source_sha256, source_sha256(candidate.source)),
            "initial_state_sha256": (
                result.initial_state_sha256,
                task.initial_state_sha256,
            ),
        }
        for field_name, (actual, expected) in expected_fields.items():
            if actual != expected:
                self._discard(
                    "replay_identity_mismatch",
                    f"{context}: {field_name} does not match the requested candidate",
                )
        if result.outcome in _UNKNOWN_REPLAY_OUTCOMES or result.binary_reward is None:
            self._discard(
                "unknown_replay_reward",
                f"{context}: unknown clean replay reward ({result.outcome.value})",
            )
        return result

    def _sample_and_evaluate_base(
        self,
        task: TaskInstanceV1,
        base_index: int,
        seen_sample_ids: set[str],
    ) -> tuple[ProgramCandidate, ProgramReplayResultV1]:
        try:
            candidate = self.base_sampler(task, base_index)
        except CandidateCollectionError as error:
            self._discard(
                "base_sampling_error",
                f"base {base_index}: {type(error).__name__}: {error}",
            )
        if not isinstance(candidate, ProgramCandidate):
            self._discard("invalid_base_candidate", f"base {base_index}: wrong candidate type")
        if candidate.program_sample_id in seen_sample_ids:
            self._discard(
                "duplicate_program_sample_id",
                f"base {base_index}: duplicate program_sample_id {candidate.program_sample_id!r}",
            )
        seen_sample_ids.add(candidate.program_sample_id)
        result = self._evaluate(task, candidate, context=f"base {base_index}")
        return candidate, result

    @staticmethod
    def _base_member(
        task: TaskInstanceV1,
        candidate: ProgramCandidate,
        result: ProgramReplayResultV1,
        base_index: int,
    ) -> LearningMemberV1:
        assert result.binary_reward is not None
        return LearningMemberV1(
            member_type="base",
            program_sample_id=candidate.program_sample_id,
            prompt=task.prompt,
            response=candidate.source,
            reward=float(result.binary_reward),
            metadata={
                "base_index": base_index,
                "replay_outcome": result.outcome.value,
                "raw_reward": result.raw_reward,
            },
        )

    @staticmethod
    def _select_p0_indices(
        candidates: list[ProgramCandidate],
        results: list[ProgramReplayResultV1],
    ) -> tuple[int, int]:
        def reward_key(index: int) -> tuple[float, int]:
            raw_reward = results[index].raw_reward
            return (float("-inf") if raw_reward is None else float(raw_reward), -index)

        first = max(range(REPAIR_TRIGGER_BASE_COUNT), key=reward_key)
        alternatives = [index for index in range(REPAIR_TRIGGER_BASE_COUNT) if index != first]
        second = max(
            alternatives,
            key=lambda index: (
                token_levenshtein_distance(candidates[first].source, candidates[index].source),
                -index,
            ),
        )
        return first, second

    @staticmethod
    def _trace_rejection(
        *,
        p0_rank: int,
        trajectory_index: int,
        p0: ProgramCandidate,
        repair_trajectory_id: str,
        reason: str,
        message: str,
        trace: RepairTraceV1 | None = None,
        pt_result: ProgramReplayResultV1 | None = None,
        revision_program_sample_id: str | None = None,
        revision_source: str | None = None,
        revision_finish_reason: str | None = None,
        revision_truncated: bool | None = None,
    ) -> RepairAttempt:
        return RepairAttempt(
            p0_rank=p0_rank,
            trajectory_index=trajectory_index,
            p0_program_sample_id=p0.program_sample_id,
            repair_trajectory_id=repair_trajectory_id,
            status="rejected",
            trace=trace,
            pt_result=pt_result,
            revision_program_sample_id=revision_program_sample_id,
            revision_source=revision_source,
            revision_finish_reason=revision_finish_reason,
            revision_truncated=revision_truncated,
            rejection_reason=reason,
            rejection_message=message,
        )

    @staticmethod
    def _validate_trace(
        task: TaskInstanceV1,
        p0: ProgramCandidate,
        trace: RepairTraceV1,
        expected_trajectory_id: str,
    ) -> None:
        if not isinstance(trace, RepairTraceV1):
            raise TypeError("repair collector did not return RepairTraceV1")
        expected_identity = (
            task.task_id,
            task.environment_seed,
            p0.program_sample_id,
            expected_trajectory_id,
            p0.source,
            source_sha256(p0.source),
        )
        actual_identity = (
            trace.task_id,
            trace.environment_seed,
            trace.program_sample_id,
            trace.repair_trajectory_id,
            trace.base_source,
            trace.base_source_sha256,
        )
        if actual_identity != expected_identity:
            raise ValueError("repair trace task/seed/sample/trajectory/P0 identity mismatch")
        reconstructed = trace.reconstruct()
        if reconstructed != trace.final_source:
            raise RepairInvariantError("reconstructed PT does not equal trace.final_source")

    def _run_repair_attempt(
        self,
        task: TaskInstanceV1,
        p0: ProgramCandidate,
        p0_result: ProgramReplayResultV1,
        p0_rank: int,
        trajectory_index: int,
        group_uid: str,
        seen_sample_ids: set[str],
    ) -> tuple[RepairAttempt, ProgramCandidate | None]:
        trajectory_id = f"{group_uid}:p0-{p0_rank}:trajectory-{trajectory_index}"
        try:
            trace = self.repair_collector(
                task,
                p0,
                p0_result,
                p0_rank,
                trajectory_index,
                trajectory_id,
            )
        except CollectionInfrastructureError as error:
            raise GroupDiscarded(
                "repair_infrastructure_error",
                str(error),
                partial_repair_attempts=(
                    self._trace_rejection(
                        p0_rank=p0_rank,
                        trajectory_index=trajectory_index,
                        p0=p0,
                        repair_trajectory_id=trajectory_id,
                        reason="repair_infrastructure_error",
                        message=str(error),
                    ),
                ),
            ) from error
        except CandidateCollectionError as error:
            return (
                self._trace_rejection(
                    p0_rank=p0_rank,
                    trajectory_index=trajectory_index,
                    p0=p0,
                    repair_trajectory_id=trajectory_id,
                    reason="collector_error",
                    message=f"{type(error).__name__}: {error}",
                ),
                None,
            )
        try:
            self._validate_trace(task, p0, trace, trajectory_id)
        except (TypeError, ValueError) as error:
            return (
                self._trace_rejection(
                    p0_rank=p0_rank,
                    trajectory_index=trajectory_index,
                    p0=p0,
                    repair_trajectory_id=trajectory_id,
                    reason="trace_mismatch",
                    message=str(error),
                    trace=trace if isinstance(trace, RepairTraceV1) else None,
                ),
                None,
            )

        pt_candidate = ProgramCandidate(
            program_sample_id=f"{trajectory_id}:pt",
            source=trace.final_source,
        )
        if pt_candidate.program_sample_id in seen_sample_ids:
            self._discard(
                "duplicate_program_sample_id",
                f"duplicate PT sample ID {pt_candidate.program_sample_id!r}",
            )
        seen_sample_ids.add(pt_candidate.program_sample_id)
        try:
            pt_result = self._evaluate(task, pt_candidate, context=f"PT {trajectory_id}")
        except GroupDiscarded as error:
            error.partial_repair_attempts = (
                self._trace_rejection(
                    p0_rank=p0_rank,
                    trajectory_index=trajectory_index,
                    p0=p0,
                    repair_trajectory_id=trajectory_id,
                    reason=error.reason,
                    message=str(error),
                    trace=trace,
                ),
            )
            raise
        if pt_result.outcome is not ReplayOutcome.SUCCESS:
            return (
                RepairAttempt(
                    p0_rank=p0_rank,
                    trajectory_index=trajectory_index,
                    p0_program_sample_id=p0.program_sample_id,
                    repair_trajectory_id=trajectory_id,
                    status="pt_failed",
                    trace=trace,
                    pt_result=pt_result,
                ),
                None,
            )

        try:
            revision_prompt = build_revision_prompt(
                task,
                trace,
                token_counter=self.revision_prompt_token_counter,
                input_token_limit=self.revision_input_token_limit,
                response_token_limit=self.revision_response_token_limit,
            )
        except RevisionRejection as error:
            return (
                self._trace_rejection(
                    p0_rank=p0_rank,
                    trajectory_index=trajectory_index,
                    p0=p0,
                    repair_trajectory_id=trajectory_id,
                    reason=error.reason.value,
                    message=str(error),
                    trace=trace,
                    pt_result=pt_result,
                ),
                None,
            )

        try:
            revision = self.revision_generator(
                task,
                p0,
                trace,
                revision_prompt,
                p0_rank,
                trajectory_index,
            )
        except CollectionInfrastructureError as error:
            raise GroupDiscarded(
                "revision_infrastructure_error",
                str(error),
                partial_repair_attempts=(
                    self._trace_rejection(
                        p0_rank=p0_rank,
                        trajectory_index=trajectory_index,
                        p0=p0,
                        repair_trajectory_id=trajectory_id,
                        reason="revision_infrastructure_error",
                        message=str(error),
                        trace=trace,
                        pt_result=pt_result,
                    ),
                ),
            ) from error
        except CandidateCollectionError as error:
            return (
                self._trace_rejection(
                    p0_rank=p0_rank,
                    trajectory_index=trajectory_index,
                    p0=p0,
                    repair_trajectory_id=trajectory_id,
                    reason="revision_generator_error",
                    message=f"{type(error).__name__}: {error}",
                    trace=trace,
                    pt_result=pt_result,
                ),
                None,
            )
        if not isinstance(revision, ProgramCandidate):
            return (
                self._trace_rejection(
                    p0_rank=p0_rank,
                    trajectory_index=trajectory_index,
                    p0=p0,
                    repair_trajectory_id=trajectory_id,
                    reason="invalid_revision_candidate",
                    message="revision generator did not return ProgramCandidate",
                    trace=trace,
                    pt_result=pt_result,
                ),
                None,
            )
        if revision.program_sample_id in seen_sample_ids:
            return (
                self._trace_rejection(
                    p0_rank=p0_rank,
                    trajectory_index=trajectory_index,
                    p0=p0,
                    repair_trajectory_id=trajectory_id,
                    reason="duplicate_program_sample_id",
                    message=f"duplicate revision sample ID {revision.program_sample_id!r}",
                    trace=trace,
                    pt_result=pt_result,
                    revision_program_sample_id=revision.program_sample_id,
                    revision_source=revision.source,
                    revision_finish_reason=revision.finish_reason,
                    revision_truncated=revision.truncated,
                ),
                None,
            )
        try:
            validate_complete_program(
                revision.source,
                token_counter=self._count_revision_response_tokens_with_eos,
                response_token_limit=self.revision_response_token_limit,
                finish_reason=revision.finish_reason,
                truncated=revision.truncated,
            )
        except RevisionRejection as error:
            return (
                self._trace_rejection(
                    p0_rank=p0_rank,
                    trajectory_index=trajectory_index,
                    p0=p0,
                    repair_trajectory_id=trajectory_id,
                    reason=error.reason.value,
                    message=str(error),
                    trace=trace,
                    pt_result=pt_result,
                    revision_program_sample_id=revision.program_sample_id,
                    revision_source=revision.source,
                    revision_finish_reason=revision.finish_reason,
                    revision_truncated=revision.truncated,
                ),
                None,
            )

        seen_sample_ids.add(revision.program_sample_id)
        try:
            revision_result = self._evaluate(
                task,
                revision,
                context=f"P_hat {trajectory_id}",
            )
        except GroupDiscarded as error:
            error.partial_repair_attempts = (
                self._trace_rejection(
                    p0_rank=p0_rank,
                    trajectory_index=trajectory_index,
                    p0=p0,
                    repair_trajectory_id=trajectory_id,
                    reason=error.reason,
                    message=str(error),
                    trace=trace,
                    pt_result=pt_result,
                    revision_program_sample_id=revision.program_sample_id,
                    revision_source=revision.source,
                    revision_finish_reason=revision.finish_reason,
                    revision_truncated=revision.truncated,
                ),
            )
            raise
        succeeded = revision_result.outcome is ReplayOutcome.SUCCESS
        return (
            RepairAttempt(
                p0_rank=p0_rank,
                trajectory_index=trajectory_index,
                p0_program_sample_id=p0.program_sample_id,
                repair_trajectory_id=trajectory_id,
                status="guided_success" if succeeded else "revision_failed",
                trace=trace,
                pt_result=pt_result,
                revision_program_sample_id=revision.program_sample_id,
                revision_source=revision.source,
                revision_finish_reason=revision.finish_reason,
                revision_truncated=revision.truncated,
                revision_result=revision_result,
            ),
            revision if succeeded else None,
        )

    def assemble(self, task: TaskInstanceV1) -> GroupAssemblyResult:
        """Build one complete, auditable group for exactly one environment seed."""

        if not isinstance(task, TaskInstanceV1):
            raise TypeError("task must be TaskInstanceV1")
        group_uid = deterministic_group_uid(task)
        seen_sample_ids: set[str] = set()
        candidates: list[ProgramCandidate] = []
        base_results: list[ProgramReplayResultV1] = []
        for base_index in range(REPAIR_TRIGGER_BASE_COUNT):
            candidate, result = self._sample_and_evaluate_base(
                task, base_index, seen_sample_ids
            )
            candidates.append(candidate)
            base_results.append(result)

        repair_attempts: list[RepairAttempt] = []
        guided_candidate: ProgramCandidate | None = None
        selected_attempt_index: int | None = None
        any_base_success = any(
            result.outcome is ReplayOutcome.SUCCESS for result in base_results
        )
        if not any_base_success:
            p0_indices = self._select_p0_indices(candidates, base_results)
            successful_revisions: list[tuple[int, ProgramCandidate]] = []
            for p0_rank, p0_index in enumerate(p0_indices):
                p0 = candidates[p0_index]
                p0_result = base_results[p0_index]
                for trajectory_index in range(TRAJECTORIES_PER_P0):
                    try:
                        attempt, successful_revision = self._run_repair_attempt(
                            task,
                            p0,
                            p0_result,
                            p0_rank,
                            trajectory_index,
                            group_uid,
                            seen_sample_ids,
                        )
                    except GroupDiscarded as error:
                        error.partial_repair_attempts = (
                            *repair_attempts,
                            *error.partial_repair_attempts,
                        )
                        raise
                    repair_attempts.append(attempt)
                    if successful_revision is not None:
                        successful_revisions.append(
                            (len(repair_attempts) - 1, successful_revision)
                        )
            if successful_revisions:
                selected_attempt_index, guided_candidate = successful_revisions[0]
                repair_attempts[selected_attempt_index] = replace(
                    repair_attempts[selected_attempt_index], selected=True
                )

        if any_base_success or guided_candidate is None:
            candidate, result = self._sample_and_evaluate_base(
                task, REPAIR_TRIGGER_BASE_COUNT, seen_sample_ids
            )
            candidates.append(candidate)
            base_results.append(result)

        members = [
            self._base_member(task, candidate, result, index)
            for index, (candidate, result) in enumerate(
                zip(candidates, base_results, strict=True)
            )
        ]
        if guided_candidate is not None:
            assert selected_attempt_index is not None
            selected_attempt = repair_attempts[selected_attempt_index]
            assert selected_attempt.revision_result is not None
            members.append(
                LearningMemberV1(
                    member_type="critique_guided_revision",
                    program_sample_id=guided_candidate.program_sample_id,
                    repair_trajectory_id=selected_attempt.repair_trajectory_id,
                    prompt=task.prompt,
                    response=guided_candidate.source,
                    reward=1.0,
                    metadata={
                        "p0_program_sample_id": selected_attempt.p0_program_sample_id,
                        "p0_rank": selected_attempt.p0_rank,
                        "trajectory_index": selected_attempt.trajectory_index,
                        "pt_source_sha256": selected_attempt.trace.final_source_sha256
                        if selected_attempt.trace is not None
                        else None,
                        "revision_replay_outcome": (
                            selected_attempt.revision_result.outcome.value
                        ),
                    },
                )
            )

        if len(members) != BASE_GROUP_SIZE:
            self._discard(
                "invalid_group_size",
                f"assembled {len(members)} members instead of {BASE_GROUP_SIZE}",
            )
        rewards = {member.reward for member in members}
        group = LearningGroupV1(
            task_id=task.task_id,
            environment_seed=task.environment_seed,
            group_uid=group_uid,
            initial_state_sha256=task.initial_state_sha256,
            members=tuple(members),
            skip_actor_update=len(rewards) == 1,
            metadata={
                "repair_triggered": not any_base_success,
                "guided_member_selected": guided_candidate is not None,
                "base_member_count": len(base_results),
            },
        )
        return GroupAssemblyResult(
            group=group,
            base_results=tuple(base_results),
            repair_attempts=tuple(repair_attempts),
        )


__all__ = [
    "BASE_GROUP_SIZE",
    "CandidateCollectionError",
    "CapsuleGroupAssembler",
    "CollectionInfrastructureError",
    "GroupAssemblyResult",
    "GroupDiscarded",
    "ProgramCandidate",
    "RepairAttempt",
    "deterministic_group_uid",
    "regex_tokens",
    "token_levenshtein_distance",
]
