from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

RuntimeActionName = Literal[
    "run_region",
    "inspect_trace",
    "inspect_variables",
    "patch_region",
    "rollback_to_checkpoint",
    "resume_from_region",
    "finish",
]
RuntimeStatus = Literal["success", "failed", "warning", "invalid", "skipped"]

SUPPORTED_ACTIONS: set[str] = {
    "run_region",
    "inspect_trace",
    "inspect_variables",
    "patch_region",
    "rollback_to_checkpoint",
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
