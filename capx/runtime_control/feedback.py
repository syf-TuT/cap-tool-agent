from __future__ import annotations

from typing import Any

from capx.runtime_control.schema import (
    CodeRegion,
    RuntimeAction,
    RuntimeEvent,
    RuntimeFeedback,
    RuntimeStatus,
)


def build_runtime_feedback(
    *,
    step_id: int,
    action: RuntimeAction,
    event: RuntimeEvent,
    region: CodeRegion | None,
    trace_events: list[dict[str, Any]],
    before_state: dict[str, Any],
    after_state: dict[str, Any],
) -> RuntimeFeedback:
    evidence = dict(event.evidence)
    evidence.update(
        {
            "primitive_calls": [
                trace_event.get("name") for trace_event in trace_events if "name" in trace_event
            ],
            "trace_events": list(trace_events),
            "reward_before": before_state.get("reward"),
            "reward_after": after_state.get("reward"),
            "task_completed_before": before_state.get("task_completed"),
            "task_completed_after": after_state.get("task_completed"),
        }
    )
    if region is not None:
        evidence["source_span"] = {
            "start_line": region.start_line,
            "end_line": region.end_line,
        }

    status = _feedback_status(event, region, before_state, after_state)
    region_id = event.region_id or (region.region_id if region is not None else None)

    return RuntimeFeedback(
        step_id=step_id,
        status=status,
        region_id=region_id,
        message=_feedback_message(status, event, region),
        evidence=evidence,
        patch_scope=region_id if status in {"failed", "invalid", "warning"} else None,
        repair_hints=_repair_hints(status, event, action),
    )


def _feedback_status(
    event: RuntimeEvent,
    region: CodeRegion | None,
    before_state: dict[str, Any],
    after_state: dict[str, Any],
) -> RuntimeStatus:
    if event.status in {"failed", "invalid", "warning", "skipped"}:
        return event.status
    if region is not None and not _made_task_progress(before_state, after_state):
        return "warning"
    return "success"


def _made_task_progress(before_state: dict[str, Any], after_state: dict[str, Any]) -> bool:
    if after_state.get("task_completed") is True:
        return True
    before_reward = _numeric_reward(before_state.get("reward"))
    after_reward = _numeric_reward(after_state.get("reward"))
    return after_reward is not None and before_reward is not None and after_reward > before_reward


def _numeric_reward(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _feedback_message(
    status: RuntimeStatus,
    event: RuntimeEvent,
    region: CodeRegion | None,
) -> str:
    if event.message:
        detail = f": {event.message}"
    else:
        detail = ""
    if region is None:
        return f"{event.action} completed with status {status}{detail}."
    if status == "warning":
        return (
            f"{region.region_id} executed at source lines {region.start_line}-{region.end_line}, "
            "but no local task progress was observed."
        )
    return (
        f"{event.action} for {region.region_id} at source lines "
        f"{region.start_line}-{region.end_line} completed with status {status}{detail}."
    )


def _repair_hints(
    status: RuntimeStatus,
    event: RuntimeEvent,
    action: RuntimeAction,
) -> list[str]:
    if status == "failed":
        return ["Patch only the failed region unless the trace shows an upstream state error."]
    if status == "invalid":
        return [f"Check the {action.action} arguments and available region ids."]
    if status == "warning" and event.action == "run_region":
        return ["Inspect primitive call trace and patch this region if it should advance the task."]
    return []
