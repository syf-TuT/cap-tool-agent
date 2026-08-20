from __future__ import annotations

import json
from typing import Any

from capx.runtime_control.schema import CodeRegion, CodeRegionGroup, RuntimeAction


def parse_runtime_action_response(content: str) -> RuntimeAction:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        language = lines[0].strip("`").strip().lower()
        if language and language != "json":
            raise ValueError("Runtime action response must be a JSON object")
        if len(lines) < 3 or not lines[-1].strip().startswith("```"):
            raise ValueError("Runtime action response must be valid fenced JSON")
        text = "\n".join(lines[1:-1]).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        data = _extract_json_object(text)
        if data is None:
            raise ValueError("Runtime action response must be valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("Runtime action response JSON must be an object")
    return RuntimeAction.from_mapping(data)


def build_capsule_prompt(
    *,
    task: str,
    regions: list[CodeRegion],
    groups: list[CodeRegionGroup] | None = None,
    history: list[dict[str, Any]],
    trace_summary: dict[str, Any],
    side_effect_ledger: dict[str, Any] | None = None,
    recovery_observation_functions: set[str] | None = None,
    compact_context: bool = False,
    history_max_entries: int = 8,
    trace_max_events: int = 8,
    source_preview_chars: int = 240,
    focused_source_max_units: int = 0,
    prompt_char_budget: int | None = None,
) -> list[dict[str, Any]]:
    recovery_functions = sorted(
        {"get_observation"}
        if recovery_observation_functions is None
        else recovery_observation_functions
    )
    if recovery_functions:
        recovery_calls = ", ".join(f"{name}()" for name in recovery_functions)
        recovery_guidance = (
            "For recovery after robot side effects, prefer appending new recovery code with "
            "append_recovery; the appended code must call at least one fresh-state function "
            f"({recovery_calls}) so it starts from the current physical state."
        )
        recovery_example_line = (
            '{"action": "append_recovery", "args": {"source": '
            f'"state = {recovery_functions[0]}()\\n# recover from the current physical state"}}\n'
        )
        recovery_rule = (
            "For append_recovery, args.source must be executable Python code that includes "
            f"at least one of {recovery_calls} and continues from the current physical state."
        )
    else:
        recovery_guidance = (
            "append_recovery is unavailable because the active API does not declare a "
            "fresh-state observation function."
        )
        recovery_example_line = ""
        recovery_rule = recovery_guidance
    allowed_actions = [
        "run_group",
        "run_region",
        "inspect_trace",
        "inspect_variables",
        "patch_group",
        "patch_region",
    ]
    if recovery_functions:
        allowed_actions.append("append_recovery")
    allowed_actions.extend(["resume_from_region", "finish"])
    allowed_actions_text = ", ".join(allowed_actions)
    normalized_side_effect_ledger = _normalize_side_effect_ledger(side_effect_ledger)
    run_group_example_id = _example_group_id(
        groups or [], normalized_side_effect_ledger
    )
    run_region_example_id = _example_region_id(
        regions, normalized_side_effect_ledger
    )

    prompt_text = _build_capsule_prompt_text(
        task=task,
        regions=regions,
        groups=groups,
        history=history,
        trace_summary=trace_summary,
        side_effect_ledger=normalized_side_effect_ledger,
        compact_context=compact_context,
        history_max_entries=history_max_entries,
        trace_max_events=trace_max_events,
        source_preview_chars=source_preview_chars,
        focused_source_max_units=focused_source_max_units,
        recovery_guidance=recovery_guidance,
        allowed_actions_text=allowed_actions_text,
        recovery_example_line=recovery_example_line,
        recovery_rule=recovery_rule,
        run_group_example_id=run_group_example_id,
        run_region_example_id=run_region_example_id,
    )
    if compact_context and _prompt_text_over_budget(prompt_text, prompt_char_budget):
        prompt_text = _build_capsule_prompt_text(
            task=task,
            regions=regions,
            groups=groups,
            history=history,
            trace_summary=trace_summary,
            side_effect_ledger=normalized_side_effect_ledger,
            compact_context=True,
            history_max_entries=min(history_max_entries, 2),
            trace_max_events=min(trace_max_events, 2),
            source_preview_chars=min(source_preview_chars, 80),
            focused_source_max_units=0,
            recovery_guidance=recovery_guidance,
            allowed_actions_text=allowed_actions_text,
            recovery_example_line=recovery_example_line,
            recovery_rule=recovery_rule,
            run_group_example_id=run_group_example_id,
            run_region_example_id=run_region_example_id,
        )
    return _capsule_prompt_messages(prompt_text)


def _capsule_prompt_messages(prompt_text: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "You control execution of generated Python code regions."},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
    ]


def _build_capsule_prompt_text(
    *,
    task: str,
    regions: list[CodeRegion],
    groups: list[CodeRegionGroup] | None,
    history: list[dict[str, Any]],
    trace_summary: dict[str, Any],
    side_effect_ledger: dict[str, Any],
    compact_context: bool,
    history_max_entries: int,
    trace_max_events: int,
    source_preview_chars: int,
    focused_source_max_units: int,
    recovery_guidance: str,
    allowed_actions_text: str,
    recovery_example_line: str,
    recovery_rule: str,
    run_group_example_id: str,
    run_region_example_id: str,
) -> str:
    if compact_context:
        region_data = [
            _compact_region_for_prompt(region, source_preview_chars=source_preview_chars)
            for region in regions
        ]
        group_data = [
            _compact_group_for_prompt(group, source_preview_chars=source_preview_chars)
            for group in groups or []
        ]
        history_data = _summarize_history_for_prompt(
            history, max_entries=history_max_entries
        )
        trace_data = _bound_trace_summary(trace_summary, max_events=trace_max_events)
        focused_source_data = _focused_failed_units_for_prompt(
            history=history,
            regions=regions,
            groups=groups,
            max_units=focused_source_max_units,
        )
        region_heading = "Compact generated code regions"
        group_heading = "Compact effect-bounded execution units (preferred run_group targets)"
        history_heading = "Recent runtime history summary"
        trace_heading = "Recent primitive call trace summary"
    else:
        region_data = [region.to_dict() for region in regions]
        group_data = [group.to_dict() for group in groups or []]
        history_data = history[-8:]
        trace_data = trace_summary
        focused_source_data = []
        region_heading = "Generated code regions"
        group_heading = "Effect-bounded execution units (preferred run_group targets)"
        history_heading = "Recent runtime history"
        trace_heading = "Primitive call trace summary"

    region_data = _annotate_execution_state_for_prompt(
        region_data,
        id_key="region_id",
        executed_ids=set(side_effect_ledger["executed_side_effect_regions"]),
    )
    group_data = _annotate_execution_state_for_prompt(
        group_data,
        id_key="group_id",
        executed_ids=set(side_effect_ledger["executed_side_effect_groups"]),
    )
    group_text = ""
    if group_data:
        group_text = (
            f"{group_heading}:\n"
            f"{json.dumps(group_data, indent=2, default=str)}\n\n"
        )
    focused_source_text = ""
    if focused_source_data:
        focused_source_text = (
            "Focused source for recent failed or invalid units:\n"
            f"{json.dumps(focused_source_data, indent=2, default=str)}\n\n"
        )
    prompt_text = (
        "Task:\n"
        f"{task}\n\n"
        f"{group_text}"
        f"{region_heading}:\n"
        f"{json.dumps(region_data, indent=2, default=str)}\n\n"
        f"{history_heading}:\n"
        f"{json.dumps(history_data, indent=2, default=str)}\n\n"
        f"{trace_heading}:\n"
        f"{json.dumps(trace_data, indent=2, default=str)}\n\n"
        "Side-effect execution ledger:\n"
        f"{json.dumps(side_effect_ledger, indent=2, default=str)}\n\n"
        f"{focused_source_text}"
        "Choose exactly one runtime-control action. These actions control source-code "
        "execution, inspection, and local source patches. They do "
        "not directly perform robot manipulation.\n\n"
        "Rollback is unavailable. Previously executed robot-side-effect code may have "
        "changed the current physical state, so repairs must continue from that state. "
        "Use a fresh observation and patch or resume code as current-state recovery; do "
        "not assume earlier robot actions can be undone or replayed from their original "
        "preconditions. "
        f"{recovery_guidance}\n\n"
        "Execution constraints:\n"
        "- Do not choose run_group, run_region, patch_group, or patch_region for units "
        "marked execution_state=executed_side_effect or listed in the side-effect "
        "execution ledger; rollback is unavailable and those actions will be invalid.\n"
        "- If additional robot-side-effect motion is needed after an executed side-effect "
        "unit, use append_recovery with a fresh-state function and continue from the "
        "current physical state.\n\n"
        f"Allowed actions: {allowed_actions_text}.\n\n"
        "Prefer run_group over run_region when effect-bounded execution units are "
        "available. A group is an effect-bounded source unit that may include setup "
        "plus one robot side effect. Use patch_group for local repairs unless a "
        "single atomic region is clearly self-contained.\n\n"
        "Respond with exactly one JSON object. Examples:\n"
        f'{{"action": "run_group", "args": {{"group_id": "{run_group_example_id}"}}}}\n'
        f'{{"action": "run_region", "args": {{"region_id": "{run_region_example_id}"}}}}\n'
        '{"action": "inspect_variables", "args": {"names": ["variable_name"]}}\n'
        f'{{"action": "patch_group", "args": {{"group_id": "{run_group_example_id}", '
        f'"source": "replacement Python source for the complete {run_group_example_id} '
        'source span"}}\n'
        f'{{"action": "patch_region", "args": {{"region_id": "{run_region_example_id}", '
        f'"source": "replacement Python source for only {run_region_example_id}"}}\n'
        f"{recovery_example_line}"
        "For inspect_variables, args.names must be a non-empty list of Python variable "
        "names to inspect. Do not pass region_id to inspect_variables.\n"
        "For patch_group, args.source must be the complete replacement Python source "
        "for only the requested source group.\n"
        "For patch_region, args.source must be the complete replacement Python source "
        "for only the requested source region. Do not use new_source or patch for "
        "patch_region replacement text.\n"
        f"{recovery_rule} "
        "Do not ask "
        "for robot primitives as tools."
    )
    return prompt_text


def _source_preview(source: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    normalized = " ".join(source.strip().split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 4)].rstrip() + " ..."


def _compact_region_for_prompt(
    region: CodeRegion, *, source_preview_chars: int
) -> dict[str, Any]:
    return {
        "region_id": region.region_id,
        "source_span": {"start_line": region.start_line, "end_line": region.end_line},
        "source_preview": _source_preview(region.source, max_chars=source_preview_chars),
    }


def _compact_group_for_prompt(
    group: CodeRegionGroup, *, source_preview_chars: int
) -> dict[str, Any]:
    return {
        "group_id": group.group_id,
        "source_span": {"start_line": group.start_line, "end_line": group.end_line},
        "source_preview": _source_preview(group.source, max_chars=source_preview_chars),
        "region_ids": list(group.region_ids),
        "primitive_calls": list(group.primitive_calls),
        "defined_names": list(group.defined_names),
        "used_names": list(group.used_names),
        "has_robot_side_effect": group.has_robot_side_effect,
    }


def _normalize_side_effect_ledger(
    side_effect_ledger: dict[str, Any] | None,
) -> dict[str, list[str]]:
    ledger = side_effect_ledger if isinstance(side_effect_ledger, dict) else {}
    return {
        "executed_side_effect_groups": _string_list_for_prompt(
            ledger.get("executed_side_effect_groups")
        ),
        "executed_side_effect_regions": _string_list_for_prompt(
            ledger.get("executed_side_effect_regions")
        ),
    }


def _string_list_for_prompt(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return sorted(str(item) for item in value)


def _example_group_id(
    groups: list[CodeRegionGroup],
    side_effect_ledger: dict[str, list[str]],
) -> str:
    executed_groups = set(side_effect_ledger["executed_side_effect_groups"])
    for group in groups:
        if group.group_id not in executed_groups:
            return group.group_id
    return "new_unexecuted_group_id"


def _example_region_id(
    regions: list[CodeRegion],
    side_effect_ledger: dict[str, list[str]],
) -> str:
    executed_regions = set(side_effect_ledger["executed_side_effect_regions"])
    for region in regions:
        if region.region_id not in executed_regions:
            return region.region_id
    return "new_unexecuted_region_id"


def _annotate_execution_state_for_prompt(
    units: list[dict[str, Any]],
    *,
    id_key: str,
    executed_ids: set[str],
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for unit in units:
        unit_data = dict(unit)
        unit_id = unit_data.get(id_key)
        if isinstance(unit_id, str) and unit_id in executed_ids:
            unit_data.update(
                {
                    "unit_id": unit_id,
                    "execution_state": "executed_side_effect",
                    "run_allowed": False,
                    "patch_allowed": False,
                    "recovery_required": "append_recovery",
                }
            )
        annotated.append(unit_data)
    return annotated


def _action_unit_id(
    action: dict[str, Any], event: dict[str, Any], feedback: dict[str, Any]
) -> str | None:
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    return (
        args.get("group_id")
        or args.get("region_id")
        or event.get("region_id")
        or feedback.get("region_id")
    )


def _history_state_value(entry: dict[str, Any], state_key: str, value_key: str) -> Any:
    state = entry.get(state_key)
    if isinstance(state, dict):
        return state.get(value_key)
    return None


def _primitive_calls_from_history(entry: dict[str, Any]) -> list[str]:
    feedback = entry.get("feedback")
    if isinstance(feedback, dict):
        evidence = feedback.get("evidence")
        if isinstance(evidence, dict) and isinstance(evidence.get("primitive_calls"), list):
            return [str(name) for name in evidence["primitive_calls"]]
    trace_events = entry.get("trace_events")
    if isinstance(trace_events, list):
        return [
            str(event["name"])
            for event in trace_events
            if isinstance(event, dict) and "name" in event
        ]
    return []


def _summarize_history_for_prompt(
    history: list[dict[str, Any]], *, max_entries: int
) -> list[dict[str, Any]]:
    bounded_count = max(0, int(max_entries))
    bounded = history[-bounded_count:] if bounded_count else []
    summaries: list[dict[str, Any]] = []
    for entry in bounded:
        action = entry.get("action") if isinstance(entry.get("action"), dict) else {}
        event = entry.get("event") if isinstance(entry.get("event"), dict) else {}
        feedback = entry.get("feedback") if isinstance(entry.get("feedback"), dict) else {}
        evidence = event.get("evidence") if isinstance(event.get("evidence"), dict) else {}
        event_status = event.get("status")
        feedback_status = feedback.get("status")
        summary = {
            "step_id": entry.get("step_id"),
            "action": action.get("action") or event.get("action"),
            "unit_id": _action_unit_id(action, event, feedback),
            "status": feedback_status or event_status,
            "event_status": event_status,
            "feedback_status": feedback_status,
            "message": feedback.get("message") or event.get("message"),
            "exception_type": evidence.get("exception_type"),
            "reward_before": _history_state_value(entry, "state_before", "reward"),
            "reward_after": _history_state_value(entry, "state_after", "reward"),
            "task_completed_before": _history_state_value(
                entry, "state_before", "task_completed"
            ),
            "task_completed_after": _history_state_value(
                entry, "state_after", "task_completed"
            ),
            "primitive_calls": _primitive_calls_from_history(entry),
        }
        summaries.append(
            {key: value for key, value in summary.items() if value not in (None, [], "")}
        )
    return summaries


def _focused_failed_units_for_prompt(
    *,
    history: list[dict[str, Any]],
    regions: list[CodeRegion],
    groups: list[CodeRegionGroup] | None,
    max_units: int,
) -> list[dict[str, Any]]:
    if max_units <= 0:
        return []
    region_by_id = {region.region_id: region for region in regions}
    group_by_id = {group.group_id: group for group in groups or []}
    focused: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in reversed(history):
        event = entry.get("event") if isinstance(entry.get("event"), dict) else {}
        feedback = entry.get("feedback") if isinstance(entry.get("feedback"), dict) else {}
        status = feedback.get("status") or event.get("status")
        if status not in {"failed", "invalid"}:
            break
        action = entry.get("action") if isinstance(entry.get("action"), dict) else {}
        unit_id = _action_unit_id(action, event, feedback)
        if not isinstance(unit_id, str) or unit_id in seen:
            continue
        unit = group_by_id.get(unit_id) or region_by_id.get(unit_id)
        if unit is None:
            continue
        focused.append(unit.to_dict())
        seen.add(unit_id)
        if len(focused) >= max_units:
            break
    return list(reversed(focused))


def _prompt_text_over_budget(prompt_text: str, prompt_char_budget: int | None) -> bool:
    return (
        prompt_char_budget is not None
        and prompt_char_budget > 0
        and len(json.dumps(_capsule_prompt_messages(prompt_text), default=str))
        > prompt_char_budget
    )


def _bound_trace_summary(trace_summary: dict[str, Any], *, max_events: int) -> dict[str, Any]:
    bounded = dict(trace_summary)
    bounded_count = max(0, int(max_events))
    events = bounded.get("events")
    if isinstance(events, list):
        bounded["events"] = events[-bounded_count:] if bounded_count else []
    recent_events = bounded.get("recent_events")
    if isinstance(recent_events, list):
        bounded["recent_events"] = recent_events[-bounded_count:] if bounded_count else []
    failed_events = bounded.get("failed_events")
    if isinstance(failed_events, list):
        bounded["failed_events"] = failed_events[-bounded_count:] if bounded_count else []
    return bounded


def _extract_json_object(text: str) -> Any | None:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None
