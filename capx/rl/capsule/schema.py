"""Versioned JSON contracts for Capsule-Critique-GRPO artifacts.

The contracts deliberately use only the Python standard library.  They are written to disk by
collectors and read later by trainers, so schema-version mismatches are rejected rather than
silently coerced.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any, ClassVar

SCHEMA_VERSION = 1
_SHA256_LENGTH = 64
_STABLE_TARGET_PATTERNS = {
    "base": re.compile(r"^base:[A-Za-z0-9_.-]+$"),
    "recovery": re.compile(r"^recovery:[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$"),
}


class FrozenMapping(Mapping[str, Any]):
    """Small pickle-safe immutable mapping for nested JSON artifact fields."""

    __slots__ = ("_items",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._items = tuple(values.items())

    def __getitem__(self, key: str) -> Any:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and dict(self.items()) == dict(other.items())

    def __repr__(self) -> str:
        return f"FrozenMapping({dict(self.items())!r})"

    def __reduce__(self) -> tuple[object, tuple[dict[str, Any]]]:
        return FrozenMapping, (dict(self.items()),)


def _freeze_json(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} JSON object keys must be strings")
            frozen[key] = _freeze_json(item, f"{name}.{key}")
        return FrozenMapping(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{name}[]") for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} numbers must be finite")
        return value
    raise TypeError(f"{name} must contain only JSON-compatible values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def source_sha256(source: str) -> str:
    """Hash source exactly as emitted, without newline or whitespace normalization."""

    if not isinstance(source, str):
        raise TypeError("source must be a string")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _validate_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    invalid_character = any(character not in "0123456789abcdef" for character in value)
    if len(value) != _SHA256_LENGTH or invalid_character:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


def _require_schema_version(value: object) -> None:
    if type(value) is not int or value != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {value!r}; expected {SCHEMA_VERSION}. "
            "Legacy artifacts are not migrated automatically."
        )


def _require_nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _validate_stable_target(target: object, origin: str) -> None:
    pattern = _STABLE_TARGET_PATTERNS.get(origin)
    if pattern is None or not isinstance(target, str) or pattern.fullmatch(target) is None:
        raise ValueError(f"target does not match the stable {origin} target grammar")


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an array")
    return value


def _json_dumps(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except ValueError as error:
        raise ValueError("JSON payload contains a non-finite number") from error


def _json_loads(payload: str) -> Mapping[str, Any]:
    return _mapping(json.loads(payload), "JSON payload")


class ReplayOutcome(str, Enum):
    SUCCESS = "success"
    TASK_FAILURE = "task_failure"
    PROGRAM_ERROR = "program_error"
    PROGRAM_TIMEOUT = "program_timeout"
    STEP_BUDGET_EXHAUSTED = "step_budget_exhausted"
    INFRA_ERROR = "infra_error"
    EVALUATOR_ERROR = "evaluator_error"


@dataclass(frozen=True)
class TaskInstanceV1:
    task_id: str
    environment_seed: int
    prompt: str
    environment: str
    api: str
    privilege: str
    initial_state_sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_nonempty_string(self.task_id, "task_id")
        if isinstance(self.environment_seed, bool) or not isinstance(self.environment_seed, int):
            raise TypeError("environment_seed must be an integer")
        for name in ("prompt", "environment", "api", "privilege"):
            _require_nonempty_string(getattr(self, name), name)
        _validate_sha256(self.initial_state_sha256, "initial_state_sha256")
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "environment_seed": self.environment_seed,
            "prompt": self.prompt,
            "environment": self.environment,
            "api": self.api,
            "privilege": self.privilege,
            "initial_state_sha256": self.initial_state_sha256,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskInstanceV1:
        data = dict(_mapping(payload, "task instance"))
        _require_schema_version(data.get("schema_version"))
        return cls(
            task_id=data["task_id"],
            environment_seed=data["environment_seed"],
            prompt=data["prompt"],
            environment=data["environment"],
            api=data["api"],
            privilege=data["privilege"],
            initial_state_sha256=data["initial_state_sha256"],
            metadata=_mapping(data.get("metadata", {}), "metadata"),
            schema_version=data["schema_version"],
        )

    def to_json(self) -> str:
        return _json_dumps(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> TaskInstanceV1:
        return cls.from_dict(_json_loads(payload))


@dataclass(frozen=True)
class ProgramReplayResultV1:
    task_id: str
    environment_seed: int
    program_sample_id: str
    source: str
    initial_state_sha256: str
    outcome: ReplayOutcome
    raw_reward: float | None
    binary_reward: float | None
    task_completed: bool
    terminated: bool = False
    truncated: bool = False
    sandbox_rc: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    attempts: int = 1
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    source_sha256: str = ""
    schema_version: int = SCHEMA_VERSION

    _UNKNOWN_REWARD_OUTCOMES: ClassVar[frozenset[ReplayOutcome]] = frozenset(
        {ReplayOutcome.INFRA_ERROR, ReplayOutcome.EVALUATOR_ERROR}
    )

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        if isinstance(self.outcome, str):
            object.__setattr__(self, "outcome", ReplayOutcome(self.outcome))
        elif not isinstance(self.outcome, ReplayOutcome):
            raise TypeError("outcome must be a ReplayOutcome or its string value")
        _require_nonempty_string(self.task_id, "task_id")
        _require_nonempty_string(self.program_sample_id, "program_sample_id")
        if not isinstance(self.source, str):
            raise TypeError("source must be a string")
        for field_name, value in (
            ("task_completed", self.task_completed),
            ("terminated", self.terminated),
            ("truncated", self.truncated),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{field_name} must be a boolean")
        if isinstance(self.environment_seed, bool) or not isinstance(self.environment_seed, int):
            raise TypeError("environment_seed must be an integer")
        _validate_sha256(self.initial_state_sha256, "initial_state_sha256")
        computed_hash = source_sha256(self.source)
        if self.source_sha256 and self.source_sha256 != computed_hash:
            raise ValueError("source_sha256 does not match source")
        object.__setattr__(self, "source_sha256", computed_hash)
        for field_name, value in (
            ("raw_reward", self.raw_reward),
            ("binary_reward", self.binary_reward),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{field_name} must be a finite real number or null")
        if self.outcome in self._UNKNOWN_REWARD_OUTCOMES:
            if self.raw_reward is not None:
                raise ValueError("infra/evaluator raw_reward must be null")
            if self.binary_reward is not None:
                raise ValueError("infra/evaluator binary_reward must be null")
        elif self.binary_reward not in (0, 0.0, 1, 1.0):
            raise ValueError("binary_reward must be 0 or 1 for semantic outcomes")
        if self.outcome is ReplayOutcome.SUCCESS:
            if self.binary_reward != 1 or not self.task_completed:
                raise ValueError("success requires binary_reward=1 and task_completed=true")
            if self.raw_reward is None or self.raw_reward < 1.0:
                raise ValueError("success requires raw_reward >= 1.0")
            if self.truncated:
                raise ValueError("success cannot be truncated")
            if self.error_type is not None:
                raise ValueError("success cannot carry a typed fatal error")
        elif self.outcome not in self._UNKNOWN_REWARD_OUTCOMES and self.binary_reward != 0:
            raise ValueError("non-success semantic outcomes require binary_reward=0")
        if self.outcome is not ReplayOutcome.SUCCESS and self.task_completed:
            raise ValueError("non-success outcomes require task_completed=false")
        if (
            isinstance(self.attempts, bool)
            or not isinstance(self.attempts, int)
            or self.attempts < 1
        ):
            raise ValueError("attempts must be a positive integer")
        if self.sandbox_rc is not None and (
            isinstance(self.sandbox_rc, bool) or not isinstance(self.sandbox_rc, int)
        ):
            raise TypeError("sandbox_rc must be an integer or null")
        for field_name, value in (
            ("error_type", self.error_type),
            ("error_message", self.error_message),
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or null")
        if self.outcome is ReplayOutcome.TASK_FAILURE:
            if self.raw_reward is None or self.truncated or self.error_type is not None:
                raise ValueError(
                    "TASK_FAILURE requires numeric raw_reward, truncated=false, and no error_type"
                )
        elif self.outcome is ReplayOutcome.PROGRAM_ERROR:
            if self.raw_reward is None:
                raise ValueError("PROGRAM_ERROR requires numeric raw_reward")
            if not self.error_type:
                raise ValueError("PROGRAM_ERROR requires non-empty error_type")
        elif self.outcome is ReplayOutcome.STEP_BUDGET_EXHAUSTED:
            if self.raw_reward is None:
                raise ValueError("STEP_BUDGET_EXHAUSTED requires numeric raw_reward")
            if not self.truncated:
                raise ValueError("STEP_BUDGET_EXHAUSTED requires truncated=true")
            if self.error_type is not None:
                raise ValueError("STEP_BUDGET_EXHAUSTED cannot carry error_type")
        elif self.outcome is ReplayOutcome.PROGRAM_TIMEOUT:
            if self.raw_reward is not None or self.truncated or not self.error_type:
                raise ValueError(
                    "PROGRAM_TIMEOUT requires raw_reward=null, truncated=false, and error_type"
                )
        object.__setattr__(
            self,
            "diagnostics",
            _freeze_json(self.diagnostics, "diagnostics"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "environment_seed": self.environment_seed,
            "program_sample_id": self.program_sample_id,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "initial_state_sha256": self.initial_state_sha256,
            "outcome": self.outcome.value,
            "raw_reward": self.raw_reward,
            "binary_reward": self.binary_reward,
            "task_completed": self.task_completed,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "sandbox_rc": self.sandbox_rc,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "attempts": self.attempts,
            "diagnostics": _thaw_json(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProgramReplayResultV1:
        data = dict(_mapping(payload, "program replay result"))
        _require_schema_version(data.get("schema_version"))
        return cls(
            task_id=data["task_id"],
            environment_seed=data["environment_seed"],
            program_sample_id=data["program_sample_id"],
            source=data["source"],
            source_sha256=data["source_sha256"],
            initial_state_sha256=data["initial_state_sha256"],
            outcome=ReplayOutcome(data["outcome"]),
            raw_reward=data.get("raw_reward"),
            binary_reward=data.get("binary_reward"),
            task_completed=data["task_completed"],
            terminated=data.get("terminated", False),
            truncated=data.get("truncated", False),
            sandbox_rc=data.get("sandbox_rc"),
            error_type=data.get("error_type"),
            error_message=data.get("error_message"),
            attempts=data.get("attempts", 1),
            diagnostics=_mapping(data.get("diagnostics", {}), "diagnostics"),
            schema_version=data["schema_version"],
        )

    def to_json(self) -> str:
        return _json_dumps(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> ProgramReplayResultV1:
        return cls.from_dict(_json_loads(payload))


@dataclass(frozen=True)
class SourceUnitV1:
    target: str
    start_offset: int
    end_offset: int
    source: str
    origin: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        if self.origin not in {"base", "recovery"}:
            raise ValueError("origin must be 'base' or 'recovery'")
        _validate_stable_target(self.target, self.origin)
        for field_name, value in (
            ("start_offset", self.start_offset),
            ("end_offset", self.end_offset),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
        if not isinstance(self.source, str):
            raise TypeError("source must be a string")
        if self.start_offset < 0 or self.end_offset < self.start_offset:
            raise ValueError("source unit offsets are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target": self.target,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "source": self.source,
            "origin": self.origin,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SourceUnitV1:
        data = dict(_mapping(payload, "source unit"))
        _require_schema_version(data.get("schema_version"))
        return cls(
            target=data["target"],
            start_offset=data["start_offset"],
            end_offset=data["end_offset"],
            source=data["source"],
            origin=data["origin"],
            schema_version=data["schema_version"],
        )


@dataclass(frozen=True)
class CommittedEditV1:
    edit_index: int
    turn_index: int
    action: str
    target: str
    origin: str
    input_revision: int
    output_revision: int
    input_sha256: str
    output_sha256: str
    rationale: str
    before_source: str
    after_source: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        if self.action not in {"append", "replace"}:
            raise ValueError("committed edit action must be append or replace")
        if self.origin not in {"base", "recovery"}:
            raise ValueError("committed edit origin must be base or recovery")
        _validate_stable_target(self.target, self.origin)
        for field_name, value in (
            ("edit_index", self.edit_index),
            ("turn_index", self.turn_index),
            ("input_revision", self.input_revision),
            ("output_revision", self.output_revision),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
        for field_name, value in (
            ("rationale", self.rationale),
            ("before_source", self.before_source),
            ("after_source", self.after_source),
        ):
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
        if self.edit_index < 0 or self.input_revision < 0 or self.turn_index < 1:
            raise ValueError("edit indices and revisions must be non-negative")
        if self.output_revision != self.input_revision + 1:
            raise ValueError("each committed edit must advance exactly one revision")
        _validate_sha256(self.input_sha256, "input_sha256")
        _validate_sha256(self.output_sha256, "output_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "edit_index": self.edit_index,
            "turn_index": self.turn_index,
            "action": self.action,
            "target": self.target,
            "origin": self.origin,
            "input_revision": self.input_revision,
            "output_revision": self.output_revision,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "rationale": self.rationale,
            "before_source": self.before_source,
            "after_source": self.after_source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CommittedEditV1:
        data = dict(_mapping(payload, "committed edit"))
        _require_schema_version(data.get("schema_version"))
        return cls(
            edit_index=data["edit_index"],
            turn_index=data["turn_index"],
            action=data["action"],
            target=data["target"],
            origin=data["origin"],
            input_revision=data["input_revision"],
            output_revision=data["output_revision"],
            input_sha256=data["input_sha256"],
            output_sha256=data["output_sha256"],
            rationale=data.get("rationale", ""),
            before_source=data["before_source"],
            after_source=data["after_source"],
            schema_version=data["schema_version"],
        )


@dataclass(frozen=True)
class RepairAuditV1:
    task_id: str
    environment_seed: int
    program_sample_id: str
    repair_trajectory_id: str
    turn_index: int
    event_type: str
    status: str
    message: str = ""
    action: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        for field_name in ("task_id", "program_sample_id", "repair_trajectory_id"):
            _require_nonempty_string(getattr(self, field_name), field_name)
        if isinstance(self.environment_seed, bool) or not isinstance(self.environment_seed, int):
            raise TypeError("environment_seed must be an integer")
        if isinstance(self.turn_index, bool) or not isinstance(self.turn_index, int):
            raise TypeError("turn_index must be an integer")
        if self.turn_index < 1:
            raise ValueError("turn_index must be positive")
        for field_name in ("event_type", "status"):
            _require_nonempty_string(getattr(self, field_name), field_name)
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
        object.__setattr__(self, "action", _freeze_json(self.action, "action"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "environment_seed": self.environment_seed,
            "program_sample_id": self.program_sample_id,
            "repair_trajectory_id": self.repair_trajectory_id,
            "turn_index": self.turn_index,
            "event_type": self.event_type,
            "status": self.status,
            "message": self.message,
            "action": _thaw_json(self.action),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RepairAuditV1:
        data = dict(_mapping(payload, "repair audit"))
        _require_schema_version(data.get("schema_version"))
        return cls(
            task_id=data["task_id"],
            environment_seed=data["environment_seed"],
            program_sample_id=data["program_sample_id"],
            repair_trajectory_id=data["repair_trajectory_id"],
            turn_index=data["turn_index"],
            event_type=data["event_type"],
            status=data["status"],
            message=data.get("message", ""),
            action=_mapping(data.get("action", {}), "action"),
            schema_version=data["schema_version"],
        )

    def to_json(self) -> str:
        return _json_dumps(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> RepairAuditV1:
        return cls.from_dict(_json_loads(payload))


@dataclass(frozen=True)
class RepairTraceV1:
    task_id: str
    environment_seed: int
    program_sample_id: str
    repair_trajectory_id: str
    base_source: str
    base_units: tuple[SourceUnitV1, ...]
    final_source: str
    edits: tuple[CommittedEditV1, ...] = ()
    audits: tuple[RepairAuditV1, ...] = ()
    base_source_sha256: str = ""
    final_source_sha256: str = ""
    event_sequence_sha256: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        if not self.task_id or not self.program_sample_id or not self.repair_trajectory_id:
            raise ValueError("task, program sample, and repair trajectory IDs must not be empty")
        if isinstance(self.environment_seed, bool) or not isinstance(self.environment_seed, int):
            raise TypeError("environment_seed must be an integer")
        computed_base_hash = source_sha256(self.base_source)
        computed_final_hash = source_sha256(self.final_source)
        if self.base_source_sha256 and self.base_source_sha256 != computed_base_hash:
            raise ValueError("base_source_sha256 does not match base_source")
        if self.final_source_sha256 and self.final_source_sha256 != computed_final_hash:
            raise ValueError("final_source_sha256 does not match final_source")
        object.__setattr__(self, "base_source_sha256", computed_base_hash)
        object.__setattr__(self, "final_source_sha256", computed_final_hash)
        object.__setattr__(self, "base_units", tuple(self.base_units))
        object.__setattr__(self, "edits", tuple(self.edits))
        object.__setattr__(self, "audits", tuple(self.audits))
        for field_name, records in (("edits", self.edits), ("audits", self.audits)):
            indices = [record.turn_index for record in records]
            if any(current >= following for current, following in zip(indices, indices[1:])):
                raise ValueError(
                    f"repair trace {field_name} turn_index values must be strictly increasing"
                )
        turn_indices = [edit.turn_index for edit in self.edits]
        turn_indices.extend(audit.turn_index for audit in self.audits)
        if len(turn_indices) > 12:
            raise ValueError("repair trace exceeds the 12 controller-turn limit")
        if sorted(turn_indices) != list(range(1, len(turn_indices) + 1)):
            raise ValueError("repair trace controller turn indices must be unique and contiguous")
        for audit in self.audits:
            context = (
                audit.task_id,
                audit.environment_seed,
                audit.program_sample_id,
                audit.repair_trajectory_id,
            )
            expected = (
                self.task_id,
                self.environment_seed,
                self.program_sample_id,
                self.repair_trajectory_id,
            )
            if context != expected:
                raise ValueError("repair audit IDs do not match repair trace IDs")
        event_records = [
            {"kind": "edit", "turn_index": edit.turn_index}
            for edit in self.edits
        ]
        event_records.extend(
            {"kind": "audit", "turn_index": audit.turn_index}
            for audit in self.audits
        )
        event_records.sort(key=lambda event: int(event["turn_index"]))
        event_bytes = json.dumps(
            event_records,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        computed_event_hash = hashlib.sha256(event_bytes).hexdigest()
        if self.event_sequence_sha256 and self.event_sequence_sha256 != computed_event_hash:
            raise ValueError("event_sequence_sha256 does not match repair turn chronology")
        object.__setattr__(self, "event_sequence_sha256", computed_event_hash)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "environment_seed": self.environment_seed,
            "program_sample_id": self.program_sample_id,
            "repair_trajectory_id": self.repair_trajectory_id,
            "base_source": self.base_source,
            "base_source_sha256": self.base_source_sha256,
            "base_units": [unit.to_dict() for unit in self.base_units],
            "edits": [edit.to_dict() for edit in self.edits],
            "audits": [audit.to_dict() for audit in self.audits],
            "final_source": self.final_source,
            "final_source_sha256": self.final_source_sha256,
            "event_sequence_sha256": self.event_sequence_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RepairTraceV1:
        data = dict(_mapping(payload, "repair trace"))
        _require_schema_version(data.get("schema_version"))
        return cls(
            task_id=data["task_id"],
            environment_seed=data["environment_seed"],
            program_sample_id=data["program_sample_id"],
            repair_trajectory_id=data["repair_trajectory_id"],
            base_source=data["base_source"],
            base_source_sha256=data["base_source_sha256"],
            base_units=tuple(
                SourceUnitV1.from_dict(_mapping(item, "base unit"))
                for item in _sequence(data["base_units"], "base_units")
            ),
            edits=tuple(
                CommittedEditV1.from_dict(_mapping(item, "edit"))
                for item in _sequence(data.get("edits", []), "edits")
            ),
            audits=tuple(
                RepairAuditV1.from_dict(_mapping(item, "audit"))
                for item in _sequence(data.get("audits", []), "audits")
            ),
            final_source=data["final_source"],
            final_source_sha256=data["final_source_sha256"],
            event_sequence_sha256=data["event_sequence_sha256"],
            schema_version=data["schema_version"],
        )

    def to_json(self) -> str:
        return _json_dumps(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> RepairTraceV1:
        return cls.from_dict(_json_loads(payload))

    def reconstruct(self) -> str:
        """Rebuild every version and verify the immutable edit chain."""

        from .repair import reconstruct_trace

        return reconstruct_trace(self)


@dataclass(frozen=True)
class LearningMemberV1:
    member_type: str
    program_sample_id: str
    prompt: str
    response: str
    reward: float
    repair_trajectory_id: str | None = None
    response_sha256: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        if self.member_type not in {"base", "critique_guided_revision"}:
            raise ValueError("member_type must be base or critique_guided_revision")
        for field_name in ("program_sample_id", "prompt", "response"):
            _require_nonempty_string(getattr(self, field_name), field_name)
        if (
            isinstance(self.reward, bool)
            or not isinstance(self.reward, Real)
            or not math.isfinite(float(self.reward))
            or self.reward not in (0, 0.0, 1, 1.0)
        ):
            raise ValueError("learning member reward must be binary")
        if self.repair_trajectory_id is not None:
            _require_nonempty_string(self.repair_trajectory_id, "repair_trajectory_id")
        computed_hash = source_sha256(self.response)
        if self.response_sha256 and self.response_sha256 != computed_hash:
            raise ValueError("response_sha256 does not match response")
        object.__setattr__(self, "response_sha256", computed_hash)
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "member_type": self.member_type,
            "program_sample_id": self.program_sample_id,
            "repair_trajectory_id": self.repair_trajectory_id,
            "prompt": self.prompt,
            "response": self.response,
            "response_sha256": self.response_sha256,
            "reward": self.reward,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LearningMemberV1:
        data = dict(_mapping(payload, "learning member"))
        _require_schema_version(data.get("schema_version"))
        return cls(
            member_type=data["member_type"],
            program_sample_id=data["program_sample_id"],
            repair_trajectory_id=data.get("repair_trajectory_id"),
            prompt=data["prompt"],
            response=data["response"],
            response_sha256=data["response_sha256"],
            reward=data["reward"],
            metadata=_mapping(data.get("metadata", {}), "metadata"),
            schema_version=data["schema_version"],
        )


@dataclass(frozen=True)
class LearningGroupV1:
    task_id: str
    environment_seed: int
    group_uid: str
    members: tuple[LearningMemberV1, ...]
    initial_state_sha256: str | None = None
    skip_actor_update: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_nonempty_string(self.task_id, "task_id")
        _require_nonempty_string(self.group_uid, "group_uid")
        if isinstance(self.environment_seed, bool) or not isinstance(self.environment_seed, int):
            raise TypeError("environment_seed must be an integer")
        object.__setattr__(self, "members", tuple(self.members))
        if not self.members:
            raise ValueError("learning group must contain at least one member")
        if any(not isinstance(member, LearningMemberV1) for member in self.members):
            raise TypeError("learning group members must be LearningMemberV1 values")
        program_sample_ids = [member.program_sample_id for member in self.members]
        if len(set(program_sample_ids)) != len(program_sample_ids):
            raise ValueError("learning group program_sample_id values must be unique")
        if not isinstance(self.skip_actor_update, bool):
            raise TypeError("skip_actor_update must be a boolean")
        if self.initial_state_sha256 is None:
            raise ValueError("initial_state_sha256 is required for a learning group")
        _validate_sha256(self.initial_state_sha256, "initial_state_sha256")
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "environment_seed": self.environment_seed,
            "group_uid": self.group_uid,
            "initial_state_sha256": self.initial_state_sha256,
            "members": [member.to_dict() for member in self.members],
            "skip_actor_update": self.skip_actor_update,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LearningGroupV1:
        data = dict(_mapping(payload, "learning group"))
        _require_schema_version(data.get("schema_version"))
        return cls(
            task_id=data["task_id"],
            environment_seed=data["environment_seed"],
            group_uid=data["group_uid"],
            initial_state_sha256=data.get("initial_state_sha256"),
            members=tuple(
                LearningMemberV1.from_dict(_mapping(item, "learning member"))
                for item in _sequence(data["members"], "members")
            ),
            skip_actor_update=data.get("skip_actor_update", False),
            metadata=_mapping(data.get("metadata", {}), "metadata"),
            schema_version=data["schema_version"],
        )

    def to_json(self) -> str:
        return _json_dumps(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> LearningGroupV1:
        return cls.from_dict(_json_loads(payload))


__all__ = [
    "CommittedEditV1",
    "LearningGroupV1",
    "LearningMemberV1",
    "ProgramReplayResultV1",
    "ReplayOutcome",
    "RepairAuditV1",
    "RepairTraceV1",
    "SCHEMA_VERSION",
    "SourceUnitV1",
    "TaskInstanceV1",
    "source_sha256",
]
