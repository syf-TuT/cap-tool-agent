from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ToolStatus = Literal["success", "failed", "warning", "invalid"]


@dataclass
class ToolSpec:
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)

    def to_prompt_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    thought: str = ""
    step_id: int | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ToolCall:
        if "tool" not in data:
            raise ValueError("Tool call must include 'tool'")
        args = data.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError("Tool call 'args' must be an object")
        return cls(
            tool=str(data["tool"]),
            args=args,
            thought=str(data.get("thought", "")),
            step_id=data.get("step_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    tool: str
    status: ToolStatus
    output_ref: str | None = None
    output_summary: Any = None
    message: str = ""
    failure_type: str | None = None
    stdout: str = ""
    stderr: str = ""
    duration_s: float | None = None
    exception_type: str | None = None

    @classmethod
    def failed(
        cls,
        *,
        tool: str,
        failure_type: str,
        message: str,
        exception_type: str | None = None,
        stderr: str = "",
    ) -> ToolResult:
        return cls(
            tool=tool,
            status="failed",
            failure_type=failure_type,
            message=message,
            exception_type=exception_type,
            stderr=stderr,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepFeedback:
    step_id: int
    tool: str
    status: ToolStatus
    failure_stage: str | None = None
    failure_type: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    repair_hints: list[str] = field(default_factory=list)
    recommended_next_tools: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
