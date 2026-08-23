"""Isolated, side-effect-free repair drafts with immutable version lineage."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .schema import (
    CommittedEditV1,
    RepairAuditV1,
    RepairTraceV1,
    SourceUnitV1,
    source_sha256,
)

_UNIT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class RepairInvariantError(ValueError):
    """Raised when immutable repair lineage cannot be trusted."""


@dataclass(frozen=True)
class BaseUnitSpan:
    """A stable editable unit within immutable P0 byte offsets."""

    unit_id: str
    start_offset: int
    end_offset: int
    expected_source: str | None = None


@dataclass(frozen=True)
class RepairSubmissionResult:
    committed: bool
    edit: CommittedEditV1 | None = None
    audit: RepairAuditV1 | None = None


class RepairDraft:
    """Build a candidate program without executing it in the simulator.

    All controller submissions consume one turn. Only valid append/replace submissions advance
    the canonical revision chain; inspections, malformed requests, and invalid targets remain in
    the audit stream.
    """

    def __init__(
        self,
        *,
        task_id: str,
        environment_seed: int,
        program_sample_id: str,
        repair_trajectory_id: str,
        base_source: str,
        base_units: Sequence[BaseUnitSpan],
        max_turns: int = 12,
    ) -> None:
        if not task_id or not program_sample_id or not repair_trajectory_id:
            raise RepairInvariantError("task, sample, and trajectory IDs must not be empty")
        if isinstance(environment_seed, bool) or not isinstance(environment_seed, int):
            raise RepairInvariantError("environment_seed must be an integer")
        if max_turns < 1:
            raise RepairInvariantError("max_turns must be positive")
        if max_turns > 12:
            raise RepairInvariantError("max_turns must be at most 12")
        if not isinstance(base_source, str):
            raise RepairInvariantError("base_source must be a string")

        self.task_id = task_id
        self.environment_seed = environment_seed
        self.program_sample_id = program_sample_id
        self.repair_trajectory_id = repair_trajectory_id
        self.base_source = base_source
        self.max_turns = max_turns

        self._base_spans = self._validate_base_units(base_source, base_units)
        self._current_units = {
            unit.target: unit.source for unit in self._base_spans
        }
        self._recovery_order: list[str] = []
        self._edits: list[CommittedEditV1] = []
        self._audits: list[RepairAuditV1] = []
        self._turn_count = 0
        self._revision = 0
        self._finished = False
        self._current_source = base_source

    @staticmethod
    def _validate_unit_id(unit_id: object, name: str) -> str:
        if not isinstance(unit_id, str) or not _UNIT_ID_PATTERN.fullmatch(unit_id):
            raise RepairInvariantError(
                f"{name} must match {_UNIT_ID_PATTERN.pattern} and must not contain ':'"
            )
        return unit_id

    @classmethod
    def _validate_base_units(
        cls, base_source: str, base_units: Sequence[BaseUnitSpan]
    ) -> tuple[SourceUnitV1, ...]:
        if not base_units:
            raise RepairInvariantError("at least one editable base unit is required")
        ordered = sorted(base_units, key=lambda item: (item.start_offset, item.end_offset))
        result: list[SourceUnitV1] = []
        previous_end = 0
        seen_ids: set[str] = set()
        for span in ordered:
            unit_id = cls._validate_unit_id(span.unit_id, "base unit_id")
            if unit_id in seen_ids:
                raise RepairInvariantError(f"duplicate base unit_id: {unit_id}")
            if span.start_offset < previous_end:
                raise RepairInvariantError("base unit spans must not overlap")
            if span.start_offset < 0 or span.end_offset < span.start_offset:
                raise RepairInvariantError("base unit span offsets are invalid")
            if span.end_offset > len(base_source):
                raise RepairInvariantError("base unit span exceeds P0 source")
            actual_source = base_source[span.start_offset : span.end_offset]
            if span.expected_source is not None and span.expected_source != actual_source:
                raise RepairInvariantError(
                    f"base unit span source mismatch for base:{unit_id}"
                )
            result.append(
                SourceUnitV1(
                    target=f"base:{unit_id}",
                    start_offset=span.start_offset,
                    end_offset=span.end_offset,
                    source=actual_source,
                    origin="base",
                )
            )
            previous_end = span.end_offset
            seen_ids.add(unit_id)
        return tuple(result)

    @property
    def current_source(self) -> str:
        return self._current_source

    @property
    def current_revision(self) -> int:
        return self._revision

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def edits(self) -> tuple[CommittedEditV1, ...]:
        return tuple(self._edits)

    @property
    def audits(self) -> tuple[RepairAuditV1, ...]:
        return tuple(self._audits)

    @property
    def editable_units(self) -> dict[str, str]:
        """Return a detached snapshot of the current stable-target source map."""

        return dict(self._current_units)

    def _next_turn(self) -> int:
        if self._turn_count >= self.max_turns:
            raise RepairInvariantError(
                f"repair controller turn limit exhausted ({self.max_turns})"
            )
        self._turn_count += 1
        return self._turn_count

    def submit_json(self, payload: str) -> RepairSubmissionResult:
        """Submit one JSON controller response, recording decoding failures as audit only."""

        turn_index = self._next_turn()
        try:
            action = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as error:
            return self._audit(
                turn_index,
                "parse_failure",
                "rejected",
                f"controller response is not valid JSON: {error}",
                {"raw_response": payload if isinstance(payload, str) else repr(payload)},
            )
        return self._submit_at_turn(action, turn_index)

    def submit(self, action: Mapping[str, Any]) -> RepairSubmissionResult:
        """Submit one already-decoded controller action."""

        return self._submit_at_turn(action, self._next_turn())

    def _submit_at_turn(self, action: object, turn_index: int) -> RepairSubmissionResult:
        if not isinstance(action, Mapping):
            return self._audit(
                turn_index,
                "parse_failure",
                "rejected",
                "controller action must be a JSON object",
                {"decoded_type": type(action).__name__},
            )
        action_copy = dict(action)
        action_name = action_copy.get("action")
        if not isinstance(action_name, str):
            return self._audit(
                turn_index,
                "parse_failure",
                "rejected",
                "controller action must contain a string 'action' field",
                action_copy,
            )
        if self._finished:
            return self._audit(
                turn_index,
                "invalid",
                "rejected",
                "repair trajectory is already finished",
                action_copy,
            )
        try:
            if action_name == "inspect":
                return self._audit(
                    turn_index,
                    "inspect",
                    "observed",
                    self._optional_text(action_copy, "message", "inspection requested"),
                    action_copy,
                )
            if action_name == "finish":
                rationale = self._optional_text(
                    action_copy, "rationale", "repair finished"
                )
                self._finished = True
                return self._audit(
                    turn_index,
                    "finish",
                    "accepted",
                    rationale,
                    action_copy,
                )
            if action_name == "append":
                return self._append(action_copy, turn_index)
            if action_name == "replace":
                return self._replace(action_copy, turn_index)
            raise RepairInvariantError(f"unsupported repair action: {action_name}")
        except (KeyError, TypeError, RepairInvariantError, ValueError) as error:
            return self._audit(
                turn_index,
                "invalid",
                "rejected",
                str(error),
                action_copy,
            )

    @staticmethod
    def _optional_text(
        action: Mapping[str, Any], field_name: str, default: str
    ) -> str:
        if field_name not in action:
            return default
        value = action[field_name]
        if not isinstance(value, str):
            raise RepairInvariantError(f"{field_name} must be a string")
        return value

    def _append(
        self, action: Mapping[str, Any], turn_index: int
    ) -> RepairSubmissionResult:
        generation_id = self._validate_unit_id(action.get("generation_id"), "generation_id")
        unit_id = self._validate_unit_id(action.get("unit_id"), "recovery unit_id")
        source = action.get("source")
        if not isinstance(source, str) or not source.strip():
            raise RepairInvariantError("append source must be a non-empty string")
        target = f"recovery:{generation_id}:{unit_id}"
        if target in self._current_units:
            raise RepairInvariantError(f"recovery target already exists: {target}")
        rationale = self._optional_text(action, "rationale", "")
        return self._commit(
            action="append",
            target=target,
            origin="recovery",
            before_source="",
            after_source=source,
            rationale=rationale,
            turn_index=turn_index,
        )

    def _replace(
        self, action: Mapping[str, Any], turn_index: int
    ) -> RepairSubmissionResult:
        target = action.get("target")
        if not isinstance(target, str) or target not in self._current_units:
            raise RepairInvariantError(f"unknown stable repair target: {target}")
        source = action.get("source")
        if not isinstance(source, str):
            raise RepairInvariantError("replace source must be a string")
        origin = "base" if target.startswith("base:") else "recovery"
        return self._commit(
            action="replace",
            target=target,
            origin=origin,
            before_source=self._current_units[target],
            after_source=source,
            rationale=self._optional_text(action, "rationale", ""),
            turn_index=turn_index,
        )

    def _commit(
        self,
        *,
        action: str,
        target: str,
        origin: str,
        before_source: str,
        after_source: str,
        rationale: str,
        turn_index: int,
    ) -> RepairSubmissionResult:
        input_source = self._current_source
        input_revision = self._revision
        next_recovery_order = list(self._recovery_order)
        next_units = dict(self._current_units)
        if action == "append":
            next_recovery_order.append(target)
        next_units[target] = after_source
        output_source = self._render_source(
            current_units=next_units,
            recovery_order=next_recovery_order,
        )
        edit = CommittedEditV1(
            edit_index=len(self._edits),
            turn_index=turn_index,
            action=action,
            target=target,
            origin=origin,
            input_revision=input_revision,
            output_revision=input_revision + 1,
            input_sha256=source_sha256(input_source),
            output_sha256=source_sha256(output_source),
            rationale=rationale,
            before_source=before_source,
            after_source=after_source,
        )
        self._current_units = next_units
        self._recovery_order = next_recovery_order
        self._edits.append(edit)
        self._revision += 1
        self._current_source = output_source
        return RepairSubmissionResult(committed=True, edit=edit)

    def _render_source(
        self,
        *,
        current_units: Mapping[str, str] | None = None,
        recovery_order: Sequence[str] | None = None,
    ) -> str:
        units = self._current_units if current_units is None else current_units
        ordered_recovery = self._recovery_order if recovery_order is None else recovery_order
        chunks: list[str] = []
        cursor = 0
        for unit in self._base_spans:
            chunks.append(self.base_source[cursor : unit.start_offset])
            chunks.append(units[unit.target])
            cursor = unit.end_offset
        chunks.append(self.base_source[cursor:])
        base = "".join(chunks)
        if not ordered_recovery:
            return base
        recovery = "\n\n".join(units[target] for target in ordered_recovery)
        if not base:
            return recovery
        if base.endswith("\n\n"):
            return base + recovery
        if base.endswith("\n"):
            return base + "\n" + recovery
        return base + "\n\n" + recovery

    def _audit(
        self,
        turn_index: int,
        event_type: str,
        status: str,
        message: str,
        action: Mapping[str, Any],
    ) -> RepairSubmissionResult:
        audit = RepairAuditV1(
            task_id=self.task_id,
            environment_seed=self.environment_seed,
            program_sample_id=self.program_sample_id,
            repair_trajectory_id=self.repair_trajectory_id,
            turn_index=turn_index,
            event_type=event_type,
            status=status,
            message=message,
            action=dict(action),
        )
        self._audits.append(audit)
        return RepairSubmissionResult(committed=False, audit=audit)

    def to_trace(self) -> RepairTraceV1:
        return RepairTraceV1(
            task_id=self.task_id,
            environment_seed=self.environment_seed,
            program_sample_id=self.program_sample_id,
            repair_trajectory_id=self.repair_trajectory_id,
            base_source=self.base_source,
            base_units=self._base_spans,
            edits=tuple(self._edits),
            audits=tuple(self._audits),
            final_source=self._current_source,
        )


def reconstruct_trace(trace: RepairTraceV1) -> str:
    """Replay committed edits and fail on any revision, source, or hash discrepancy."""

    base_spans = tuple(
        BaseUnitSpan(
            unit_id=unit.target.removeprefix("base:"),
            start_offset=unit.start_offset,
            end_offset=unit.end_offset,
            expected_source=unit.source,
        )
        for unit in trace.base_units
    )
    draft = RepairDraft(
        task_id=trace.task_id,
        environment_seed=trace.environment_seed,
        program_sample_id=trace.program_sample_id,
        repair_trajectory_id=trace.repair_trajectory_id,
        base_source=trace.base_source,
        base_units=base_spans,
        max_turns=max(1, len(trace.edits)),
    )
    if source_sha256(trace.base_source) != trace.base_source_sha256:
        raise RepairInvariantError("trace base hash does not match immutable P0")

    for expected_index, stored_edit in enumerate(trace.edits):
        if stored_edit.edit_index != expected_index:
            raise RepairInvariantError("edit index is not contiguous")
        if stored_edit.input_revision != draft.current_revision:
            raise RepairInvariantError("edit input revision does not match reconstructed revision")
        if stored_edit.input_sha256 != source_sha256(draft.current_source):
            raise RepairInvariantError("edit input hash does not match reconstructed source")
        if stored_edit.action == "append":
            target_parts = stored_edit.target.split(":")
            if len(target_parts) != 3 or target_parts[0] != "recovery":
                raise RepairInvariantError("append target is not a stable recovery target")
            result = draft.submit(
                {
                    "action": "append",
                    "generation_id": target_parts[1],
                    "unit_id": target_parts[2],
                    "source": stored_edit.after_source,
                    "rationale": stored_edit.rationale,
                }
            )
        else:
            result = draft.submit(
                {
                    "action": "replace",
                    "target": stored_edit.target,
                    "source": stored_edit.after_source,
                    "rationale": stored_edit.rationale,
                }
            )
        if not result.committed or result.edit is None:
            message = result.audit.message if result.audit else "unknown reconstruction failure"
            raise RepairInvariantError(f"stored edit could not be applied: {message}")
        replayed = result.edit
        if replayed.before_source != stored_edit.before_source:
            raise RepairInvariantError("edit before source does not match target lineage")
        if replayed.origin != stored_edit.origin:
            raise RepairInvariantError("edit origin does not match stable target")
        if replayed.output_revision != stored_edit.output_revision:
            raise RepairInvariantError("edit output revision is not contiguous")
        if replayed.output_sha256 != stored_edit.output_sha256:
            raise RepairInvariantError("edit output hash does not match reconstructed source")

    if draft.current_source != trace.final_source:
        raise RepairInvariantError("reconstructed final source does not match trace PT")
    if source_sha256(draft.current_source) != trace.final_source_sha256:
        raise RepairInvariantError("reconstructed final hash does not match trace PT")
    return draft.current_source


__all__ = [
    "BaseUnitSpan",
    "RepairDraft",
    "RepairInvariantError",
    "RepairSubmissionResult",
    "reconstruct_trace",
]
