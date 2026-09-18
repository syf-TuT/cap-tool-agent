from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

RuntimeActionName = Literal[
    "run_region",
    "run_group",
    "inspect_variables",
    "patch_region",
    "patch_group",
    "append_recovery",
    "resume_from_region",
    "finish",
]
RuntimeStatus = Literal["success", "failed", "warning", "invalid", "skipped"]

SUPPORTED_ACTIONS: set[str] = {
    "run_region",
    "run_group",
    "inspect_variables",
    "patch_region",
    "patch_group",
    "append_recovery",
    "resume_from_region",
    "finish",
}


@dataclass
class CodeRegion:
    region_id: str
    start_line: int
    end_line: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "source_span": {"start_line": self.start_line, "end_line": self.end_line},
            "source": self.source,
        }


@dataclass
class CodeRegionGroup:
    group_id: str
    start_line: int
    end_line: int
    source: str
    region_ids: list[str] = field(default_factory=list)
    primitive_calls: list[str] = field(default_factory=list)
    defined_names: list[str] = field(default_factory=list)
    used_names: list[str] = field(default_factory=list)
    has_robot_side_effect: bool = False

    @property
    def region_id(self) -> str:
        return self.group_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "source_span": {"start_line": self.start_line, "end_line": self.end_line},
            "source": self.source,
            "region_ids": list(self.region_ids),
            "primitive_calls": list(self.primitive_calls),
            "defined_names": list(self.defined_names),
            "used_names": list(self.used_names),
            "has_robot_side_effect": self.has_robot_side_effect,
        }


@dataclass
class RuntimeAction:
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    thought: str = ""
    step_id: int | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> RuntimeAction:
        if "action" not in data:
            raise ValueError("Runtime action must include 'action'")
        action = str(data["action"])
        if action not in SUPPORTED_ACTIONS:
            raise ValueError(f"Unsupported runtime action: {action}")
        args = data.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError("Runtime action 'args' must be an object")
        return cls(
            action=action,
            args=args,
            thought=str(data.get("thought", "")),
            step_id=data.get("step_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeEvent:
    action: str
    status: RuntimeStatus
    region_id: str | None = None
    message: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    duration_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PostActionObservation:
    step_id: int
    action: str
    unit_id: str | None
    unit_key: str | None
    event_status: RuntimeStatus
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    reward_before: float | None
    reward_after: float | None
    task_completed: bool
    new_trace_events: list[dict[str, Any]]
    trace_revision: int
    source_revision: int
    terminal_progress_unverified: bool = False
    safety_failure: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeFeedback:
    step_id: int
    status: RuntimeStatus
    region_id: str | None = None
    message: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    patch_scope: str | None = None
    repair_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
