from __future__ import annotations

import json
import re
from heapq import nsmallest
from itertools import islice
from typing import Any

from capx.runtime_control.schema import CodeRegion, CodeRegionGroup, RuntimeAction

_CONTRACT_SAFETY_MAX_CHARS = 12000
_CONTRACT_SAFETY_FALLBACK_MAX_CHARS = 6000
_CONTRACT_SAFETY_MIN_CHARS = 256
_CONTRACT_TEXT_MAX_CHARS = 640
_CONTRACT_ID_MAX_CHARS = 160
_CONTRACT_LIST_MAX_ITEMS = 6
_COMPACT_UNIT_MAX_ITEMS_BY_PHASE = (64, 16, 6, 3)
_COMPACT_UNIT_MAX_CHARS_BY_PHASE = (8000, 3200, 1400, 700)
_COMPACT_UNIT_LIST_MAX_ITEMS = 4
_COMPACT_UNIT_ID_MAX_CHARS = 120
_COMPACT_UNIT_TEXT_MAX_CHARS = 80
_COMPACT_LEDGER_MAX_ITEMS_BY_PHASE = (32, 12, 4, 2)
_COMPACT_LEDGER_ITEM_MAX_CHARS_BY_PHASE = (120, 100, 80, 64)
_COMPACT_LEDGER_MAX_CHARS_BY_PHASE = (4000, 1600, 800, 512)
_HISTORY_INSPECT_MAX_VARIABLES = 8
_HISTORY_INSPECT_MAX_NAME_CHARS = 80
_HISTORY_INSPECT_MAX_TEXT_CHARS = 200
_HISTORY_INSPECT_MAX_LIST_ITEMS = 32
_HISTORY_INSPECT_MAX_MAPPING_ITEMS = 8
_HISTORY_INSPECT_MAX_DEPTH = 3
_HISTORY_INSPECT_VALUE_KEYS = ("type", "shape", "value", "repr")
_HISTORY_INSPECT_MAX_NODES = 96
_HISTORY_INSPECT_MAX_SERIALIZED_CHARS = 1400
_HISTORY_INSPECT_FALLBACK_SERIALIZED_CHARS = 512
_HISTORY_INSPECT_MINIMAL_SERIALIZED_CHARS = 256
_HISTORY_SCALAR_MAX_CHARS = 120
_HISTORY_MESSAGE_MAX_CHARS = 240
_HISTORY_PRIMITIVE_MAX_ITEMS = 8
_HISTORY_PRIMITIVE_NAME_MAX_CHARS = 80
_MINIMAL_FALLBACK_MIN_PROMPT_CHARS = 4096
_DATA_URL_TEXT = re.compile(
    r"(?i)(?<![A-Za-z0-9_])data:(?:[-A-Za-z0-9.+]+/[-A-Za-z0-9.+]+)?"
    r"(?:;[-A-Za-z0-9.+]+(?:=[-A-Za-z0-9.+]*)?)*;base64,"
)
_STANDALONE_BASE64_PAYLOAD = re.compile(r"(?i)(?<![A-Za-z0-9_])base64,(?=\s*[A-Za-z0-9+/_=-]{8})")
_STRICT_CAPSULE_SOURCE_CONSTRAINTS = (
    "Strict Python subset for every generated or patched source:\n"
    "- Use no imports, classes, lambdas, try, while, async, dynamic or reflective "
    "calls, callable aliases, or attribute calls. Call only direct public API "
    "functions, safe builtins, and proven-pure helpers; use only bounded for loops."
)


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
    contract_violations: list[dict[str, Any]] | None = None,
    side_effect_ledger: dict[str, Any] | None = None,
    recovery_observation_functions: set[str] | None = None,
    strict_subset: bool = False,
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
        "inspect_variables",
        "patch_group",
        "patch_region",
    ]
    if recovery_functions:
        allowed_actions.append("append_recovery")
    allowed_actions.extend(["resume_from_region", "finish"])
    allowed_actions_text = ", ".join(allowed_actions)
    normalized_side_effect_ledger = _normalize_side_effect_ledger(side_effect_ledger)
    run_group_example_id = _example_group_id(groups or [], normalized_side_effect_ledger)
    run_region_example_id = _example_region_id(regions, normalized_side_effect_ledger)
    strict_source_constraints = f"{_STRICT_CAPSULE_SOURCE_CONSTRAINTS}\n" if strict_subset else ""

    prompt_text = _build_capsule_prompt_text(
        task=task,
        regions=regions,
        groups=groups,
        history=history,
        trace_summary=trace_summary,
        contract_safety_context=_compact_contract_violations(contract_violations),
        strict_source_constraints=strict_source_constraints,
        side_effect_ledger=normalized_side_effect_ledger,
        compact_context=compact_context,
        history_max_entries=history_max_entries,
        trace_max_events=trace_max_events,
        source_preview_chars=source_preview_chars,
        focused_source_max_units=focused_source_max_units,
        history_inspect_max_chars=_HISTORY_INSPECT_MAX_SERIALIZED_CHARS,
        compact_phase=0,
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
            contract_safety_context=_compact_contract_violations(
                contract_violations,
                max_chars=_contract_safety_budget(
                    prompt_char_budget,
                    divisor=1,
                    max_chars=_CONTRACT_SAFETY_FALLBACK_MAX_CHARS,
                ),
            ),
            strict_source_constraints=strict_source_constraints,
            side_effect_ledger=normalized_side_effect_ledger,
            compact_context=True,
            history_max_entries=min(history_max_entries, 2),
            trace_max_events=min(trace_max_events, 2),
            source_preview_chars=min(source_preview_chars, 80),
            focused_source_max_units=0,
            history_inspect_max_chars=_HISTORY_INSPECT_FALLBACK_SERIALIZED_CHARS,
            compact_phase=1,
            recovery_guidance=recovery_guidance,
            allowed_actions_text=allowed_actions_text,
            recovery_example_line=recovery_example_line,
            recovery_rule=recovery_rule,
            run_group_example_id=run_group_example_id,
            run_region_example_id=run_region_example_id,
        )
    if (
        compact_context
        and prompt_char_budget is not None
        and prompt_char_budget >= _MINIMAL_FALLBACK_MIN_PROMPT_CHARS
        and _prompt_text_over_budget(prompt_text, prompt_char_budget)
    ):
        prompt_text = _build_capsule_prompt_text(
            task=task,
            regions=regions,
            groups=groups,
            history=history,
            trace_summary=trace_summary,
            contract_safety_context=_compact_contract_violations(
                contract_violations,
                max_chars=_contract_safety_budget(
                    prompt_char_budget,
                    divisor=3,
                    max_chars=_CONTRACT_SAFETY_MAX_CHARS,
                ),
            ),
            strict_source_constraints=strict_source_constraints,
            side_effect_ledger=normalized_side_effect_ledger,
            compact_context=True,
            history_max_entries=min(history_max_entries, 1),
            trace_max_events=0,
            source_preview_chars=0,
            focused_source_max_units=0,
            history_inspect_max_chars=_HISTORY_INSPECT_MINIMAL_SERIALIZED_CHARS,
            compact_phase=2,
            recovery_guidance=recovery_guidance,
            allowed_actions_text=allowed_actions_text,
            recovery_example_line=recovery_example_line,
            recovery_rule=recovery_rule,
            run_group_example_id=run_group_example_id,
            run_region_example_id=run_region_example_id,
        )
    if (
        compact_context
        and prompt_char_budget is not None
        and prompt_char_budget >= _MINIMAL_FALLBACK_MIN_PROMPT_CHARS
        and _prompt_text_over_budget(prompt_text, prompt_char_budget)
    ):
        prompt_text = _build_capsule_prompt_text(
            task=task,
            regions=regions,
            groups=groups,
            history=history,
            trace_summary=trace_summary,
            contract_safety_context=_compact_contract_violations(
                contract_violations,
                max_chars=_CONTRACT_SAFETY_MIN_CHARS,
            ),
            strict_source_constraints=strict_source_constraints,
            side_effect_ledger=normalized_side_effect_ledger,
            compact_context=True,
            history_max_entries=min(history_max_entries, 1),
            trace_max_events=0,
            source_preview_chars=0,
            focused_source_max_units=0,
            history_inspect_max_chars=0,
            compact_phase=3,
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
    contract_safety_context: dict[str, Any] | None,
    strict_source_constraints: str,
    side_effect_ledger: dict[str, Any],
    compact_context: bool,
    history_max_entries: int,
    trace_max_events: int,
    source_preview_chars: int,
    focused_source_max_units: int,
    history_inspect_max_chars: int,
    compact_phase: int,
    recovery_guidance: str,
    allowed_actions_text: str,
    recovery_example_line: str,
    recovery_rule: str,
    run_group_example_id: str,
    run_region_example_id: str,
) -> str:
    if compact_context:
        region_units = [
            _compact_region_for_prompt(region, source_preview_chars=source_preview_chars)
            for region in regions
        ]
        group_units = [
            _compact_group_for_prompt(group, source_preview_chars=source_preview_chars)
            for group in groups or []
        ]
        history_data = _summarize_history_for_prompt(
            history,
            max_entries=history_max_entries,
            inspect_max_chars=history_inspect_max_chars,
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
        region_units = [region.to_dict() for region in regions]
        group_units = [group.to_dict() for group in groups or []]
        history_data = history[-8:]
        trace_data = trace_summary
        focused_source_data = []
        region_heading = "Generated code regions"
        group_heading = "Effect-bounded execution units (preferred run_group targets)"
        history_heading = "Recent runtime history"
        trace_heading = "Primitive call trace summary"

    region_units = _annotate_execution_state_for_prompt(
        region_units,
        id_key="region_id",
        executed_ids=set(side_effect_ledger["executed_side_effect_regions"]),
    )
    group_units = _annotate_execution_state_for_prompt(
        group_units,
        id_key="group_id",
        executed_ids=set(side_effect_ledger["executed_side_effect_groups"]),
    )
    ledger_data = side_effect_ledger
    if compact_context:
        phase_index = min(compact_phase, len(_COMPACT_UNIT_MAX_ITEMS_BY_PHASE) - 1)
        unit_max_items = _COMPACT_UNIT_MAX_ITEMS_BY_PHASE[phase_index]
        unit_max_chars = _COMPACT_UNIT_MAX_CHARS_BY_PHASE[phase_index]
        region_data = _compact_unit_envelope(
            region_units,
            id_key="region_id",
            history=history,
            example_unit_id=run_region_example_id,
            executed_ids=set(side_effect_ledger["executed_side_effect_regions"]),
            max_items=unit_max_items,
            max_chars=unit_max_chars,
        )
        group_data = _compact_unit_envelope(
            group_units,
            id_key="group_id",
            history=history,
            example_unit_id=run_group_example_id,
            executed_ids=set(side_effect_ledger["executed_side_effect_groups"]),
            max_items=unit_max_items,
            max_chars=unit_max_chars,
        )
        ledger_data = _compact_side_effect_ledger(
            side_effect_ledger,
            max_items=_COMPACT_LEDGER_MAX_ITEMS_BY_PHASE[phase_index],
            item_max_chars=_COMPACT_LEDGER_ITEM_MAX_CHARS_BY_PHASE[phase_index],
            max_chars=_COMPACT_LEDGER_MAX_CHARS_BY_PHASE[phase_index],
        )
    else:
        region_data = region_units
        group_data = group_units
    group_text = ""
    if group_data:
        group_text = f"{group_heading}:\n{json.dumps(group_data, indent=2, default=str)}\n\n"
    focused_source_text = ""
    if focused_source_data:
        focused_source_text = (
            "Focused source for recent failed or invalid units:\n"
            f"{json.dumps(focused_source_data, indent=2, default=str)}\n\n"
        )
    contract_violation_text = ""
    if contract_safety_context:
        contract_violation_text = (
            "Capsule-ready program contract violations:\n"
            f"{json.dumps(contract_safety_context, default=str)}\n"
            "Patch these violations before running any robot effects. Do not execute "
            "a robot-side-effect region or group until the program contract is repaired.\n\n"
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
        f"{json.dumps(ledger_data, indent=2, default=str)}\n\n"
        f"{contract_violation_text}"
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
        f"{strict_source_constraints}"
        "- Do not use callable introspection (__closure__, __self__, __globals__, "
        "__wrapped__, or related private attributes), globals()/eval()/exec(), "
        "vars()/dir(), inspect, gc, or dynamic API lookup. Use documented public "
        "API function names directly.\n"
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


def _compact_contract_violations(
    violations: list[dict[str, Any]] | None,
    *,
    max_chars: int = _CONTRACT_SAFETY_MAX_CHARS,
) -> dict[str, Any] | None:
    if not violations:
        return None

    total_count = len(violations)
    compact_items: list[dict[str, Any]] = []
    for violation in violations:
        item = _compact_contract_violation(violation)
        candidate_items = [*compact_items, item]
        candidate = {
            "total_count": total_count,
            "violations": candidate_items,
            "omitted_count": total_count - len(candidate_items),
        }
        if len(json.dumps(candidate, default=str)) > max_chars:
            if not compact_items:
                compact_items = [
                    _minimal_contract_violation(
                        violation,
                        total_count=total_count,
                        max_chars=max_chars,
                    )
                ]
            break
        compact_items = candidate_items

    return {
        "total_count": total_count,
        "violations": compact_items,
        "omitted_count": total_count - len(compact_items),
    }


def _minimal_contract_violation(
    violation: dict[str, Any],
    *,
    total_count: int,
    max_chars: int,
) -> dict[str, str]:
    raw_code = str(violation.get("code") or "<unknown>")

    def context_for_code(code: str) -> dict[str, Any]:
        return {
            "total_count": total_count,
            "violations": [{"code": code}],
            "omitted_count": total_count - 1,
        }

    low = 1
    high = min(len(raw_code), _CONTRACT_ID_MAX_CHARS)
    best_code = raw_code[:1]
    while low <= high:
        midpoint = (low + high) // 2
        candidate_code = raw_code[:midpoint]
        if len(json.dumps(context_for_code(candidate_code), default=str)) <= max_chars:
            best_code = candidate_code
            low = midpoint + 1
        else:
            high = midpoint - 1
    return {"code": best_code}


def _contract_safety_budget(
    prompt_char_budget: int | None,
    *,
    divisor: int,
    max_chars: int,
) -> int:
    if prompt_char_budget is None or prompt_char_budget < _MINIMAL_FALLBACK_MIN_PROMPT_CHARS:
        return max_chars
    return min(
        max_chars,
        max(_CONTRACT_SAFETY_MIN_CHARS, prompt_char_budget // divisor),
    )


def _compact_contract_violation(violation: dict[str, Any]) -> dict[str, Any]:
    source_span = violation.get("source_span")
    if not isinstance(source_span, dict):
        source_span = {}
    return {
        "code": _truncate_contract_text(violation.get("code"), max_chars=_CONTRACT_ID_MAX_CHARS),
        "message": _truncate_contract_text(
            violation.get("message"), max_chars=_CONTRACT_TEXT_MAX_CHARS
        ),
        "source_span": {
            "start_line": _compact_contract_scalar(source_span.get("start_line")),
            "end_line": _compact_contract_scalar(source_span.get("end_line")),
        },
        "region_ids": _compact_contract_list(violation.get("region_ids")),
        "group_ids": _compact_contract_list(violation.get("group_ids")),
        "side_effect_calls": _compact_contract_list(violation.get("side_effect_calls")),
        "helper_name": _truncate_contract_text(
            violation.get("helper_name"), max_chars=_CONTRACT_ID_MAX_CHARS
        ),
    }


def _compact_contract_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [
        _truncate_contract_text(item, max_chars=_CONTRACT_ID_MAX_CHARS)
        for item in value[:_CONTRACT_LIST_MAX_ITEMS]
    ]


def _compact_contract_scalar(value: Any) -> int | float | bool | str | None:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return _truncate_contract_text(value, max_chars=_CONTRACT_ID_MAX_CHARS)


def _truncate_contract_text(value: Any, *, max_chars: int) -> str:
    text = str(value) if value is not None else ""
    if len(text) <= max_chars:
        return text
    suffix = "...<truncated>"
    return f"{text[: max_chars - len(suffix)]}{suffix}"


def _source_preview(source: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    normalized = " ".join(source.strip().split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 4)].rstrip() + " ..."


def _compact_region_for_prompt(region: CodeRegion, *, source_preview_chars: int) -> dict[str, Any]:
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
        "region_ids": _bound_compact_unit_string_list(group.region_ids),
        "primitive_calls": _bound_compact_unit_string_list(group.primitive_calls),
        "defined_names": _bound_compact_unit_string_list(group.defined_names),
        "used_names": _bound_compact_unit_string_list(group.used_names),
        "has_robot_side_effect": group.has_robot_side_effect,
    }


def _bound_compact_unit_string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [
        _truncate_contract_text(item, max_chars=_COMPACT_UNIT_TEXT_MAX_CHARS)
        for item in value[:_COMPACT_UNIT_LIST_MAX_ITEMS]
    ]


def _compact_unit_envelope(
    units: list[dict[str, Any]],
    *,
    id_key: str,
    history: list[dict[str, Any]],
    example_unit_id: str,
    executed_ids: set[str],
    max_items: int,
    max_chars: int,
) -> dict[str, Any] | None:
    if not units:
        return None
    total_count = len(units)
    selected: list[dict[str, Any]] = []
    for unit in _prioritized_compact_units(
        units,
        id_key=id_key,
        history=history,
        example_unit_id=example_unit_id,
        executed_ids=executed_ids,
    ):
        if len(selected) >= max_items:
            break
        bounded_unit = _bound_compact_unit(unit, id_key=id_key)
        candidate_units = [*selected, bounded_unit]
        candidate = _compact_unit_envelope_data(total_count, candidate_units)
        if len(json.dumps(candidate, indent=2, default=str)) > max_chars:
            minimal_unit = _minimal_compact_unit(unit, id_key=id_key)
            candidate_units = [*selected, minimal_unit]
            candidate = _compact_unit_envelope_data(total_count, candidate_units)
            if len(json.dumps(candidate, indent=2, default=str)) > max_chars:
                continue
        selected = candidate_units
    return _compact_unit_envelope_data(total_count, selected)


def _compact_unit_envelope_data(
    total_count: int,
    units: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "total_count": total_count,
        "units": units,
        "omitted_count": total_count - len(units),
    }


def _prioritized_compact_units(
    units: list[dict[str, Any]],
    *,
    id_key: str,
    history: list[dict[str, Any]],
    example_unit_id: str,
    executed_ids: set[str],
) -> list[dict[str, Any]]:
    unit_by_id: dict[str, dict[str, Any]] = {}
    for unit in units:
        unit_id = unit.get(id_key)
        if isinstance(unit_id, str) and unit_id not in unit_by_id:
            unit_by_id[unit_id] = unit

    priority_ids: list[str] = []
    for entry in reversed(history):
        event = entry.get("event") if isinstance(entry.get("event"), dict) else {}
        feedback = entry.get("feedback") if isinstance(entry.get("feedback"), dict) else {}
        if (feedback.get("status") or event.get("status")) not in {"failed", "invalid"}:
            continue
        action = entry.get("action") if isinstance(entry.get("action"), dict) else {}
        unit_id = _action_unit_id(action, event, feedback)
        if isinstance(unit_id, str):
            priority_ids.append(unit_id)
    priority_ids.append(example_unit_id)
    priority_ids.extend(sorted(executed_ids & unit_by_id.keys()))

    source_order = sorted(
        units,
        key=lambda unit: _compact_unit_source_key(unit, id_key=id_key),
    )
    priority_ids.extend(unit[id_key] for unit in source_order if isinstance(unit.get(id_key), str))

    prioritized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for unit_id in priority_ids:
        unit = unit_by_id.get(unit_id)
        if unit is None or unit_id in seen:
            continue
        prioritized.append(unit)
        seen.add(unit_id)
    return prioritized


def _compact_unit_source_key(unit: dict[str, Any], *, id_key: str) -> tuple[int, int, str]:
    source_span = unit.get("source_span")
    if not isinstance(source_span, dict):
        source_span = {}
    start_line = source_span.get("start_line")
    end_line = source_span.get("end_line")
    return (
        start_line if isinstance(start_line, int) else 2**63 - 1,
        end_line if isinstance(end_line, int) else 2**63 - 1,
        str(unit.get(id_key, "")),
    )


def _bound_compact_unit(unit: dict[str, Any], *, id_key: str) -> dict[str, Any]:
    bounded = {
        id_key: _truncate_contract_text(
            unit.get(id_key),
            max_chars=_COMPACT_UNIT_ID_MAX_CHARS,
        ),
        "source_span": unit.get("source_span"),
    }
    source_preview = unit.get("source_preview")
    if isinstance(source_preview, str):
        bounded["source_preview"] = _truncate_contract_text(
            source_preview,
            max_chars=_COMPACT_UNIT_TEXT_MAX_CHARS,
        )
    for key in ("region_ids", "primitive_calls", "defined_names", "used_names"):
        if key in unit:
            bounded[key] = _bound_compact_unit_string_list(unit[key])
    if "has_robot_side_effect" in unit:
        bounded["has_robot_side_effect"] = unit["has_robot_side_effect"] is True
    for key in ("unit_id", "execution_state", "recovery_required"):
        value = unit.get(key)
        if isinstance(value, str):
            bounded[key] = _truncate_contract_text(
                value,
                max_chars=_COMPACT_UNIT_ID_MAX_CHARS,
            )
    for key in ("run_allowed", "patch_allowed"):
        if isinstance(unit.get(key), bool):
            bounded[key] = unit[key]
    return bounded


def _minimal_compact_unit(unit: dict[str, Any], *, id_key: str) -> dict[str, Any]:
    minimal = {
        id_key: _truncate_contract_text(
            unit.get(id_key),
            max_chars=_COMPACT_UNIT_ID_MAX_CHARS,
        )
    }
    for key in ("unit_id", "execution_state", "recovery_required"):
        value = unit.get(key)
        if isinstance(value, str):
            minimal[key] = _truncate_contract_text(
                value,
                max_chars=_COMPACT_UNIT_ID_MAX_CHARS,
            )
    for key in ("run_allowed", "patch_allowed"):
        if isinstance(unit.get(key), bool):
            minimal[key] = unit[key]
    return minimal


def _compact_side_effect_ledger(
    side_effect_ledger: dict[str, list[str]],
    *,
    max_items: int,
    item_max_chars: int,
    max_chars: int,
) -> dict[str, Any]:
    item_limit = max_items
    char_limit = item_max_chars
    while True:
        compact = _side_effect_ledger_with_limits(
            side_effect_ledger,
            max_items=item_limit,
            item_max_chars=char_limit,
        )
        if len(json.dumps(compact, indent=2, default=str)) <= max_chars:
            return compact
        if item_limit > 1:
            item_limit -= 1
            continue
        if char_limit > 8:
            char_limit = max(8, char_limit // 2)
            continue
        return compact


def _side_effect_ledger_with_limits(
    side_effect_ledger: dict[str, list[str]],
    *,
    max_items: int,
    item_max_chars: int,
) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("executed_side_effect_groups", "executed_side_effect_regions"):
        values = side_effect_ledger[key]
        selected = [
            _truncate_contract_text(value, max_chars=item_max_chars) for value in values[:max_items]
        ]
        compact[key] = selected
        omitted_count = len(values) - len(selected)
        if omitted_count:
            compact[f"{key}_total_count"] = len(values)
            compact[f"{key}_omitted_count"] = omitted_count
    return compact


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
            return _bound_history_text_list(
                evidence["primitive_calls"],
                max_items=_HISTORY_PRIMITIVE_MAX_ITEMS,
                max_chars=_HISTORY_PRIMITIVE_NAME_MAX_CHARS,
            )
    trace_events = entry.get("trace_events")
    if isinstance(trace_events, list):
        return _bound_history_text_list(
            [event.get("name") for event in trace_events if isinstance(event, dict)],
            max_items=_HISTORY_PRIMITIVE_MAX_ITEMS,
            max_chars=_HISTORY_PRIMITIVE_NAME_MAX_CHARS,
        )
    return []


def _truncate_history_text(value: str, *, max_chars: int) -> str:
    binary_matches = [
        match
        for match in (_DATA_URL_TEXT.search(value), _STANDALONE_BASE64_PAYLOAD.search(value))
        if match is not None
    ]
    if binary_matches:
        offset = min(match.start() for match in binary_matches)
        value = f"{value[:offset]}<redacted binary data>"
    if len(value) <= max_chars:
        return value
    suffix = "...<truncated>"
    return value[: max(0, max_chars - len(suffix))] + suffix


def _bound_history_scalar(value: Any, *, max_chars: int = _HISTORY_SCALAR_MAX_CHARS) -> Any:
    if value is None or isinstance(value, (bool, float)):
        return value
    if isinstance(value, int):
        return value if value.bit_length() <= 256 else "<large integer>"
    if isinstance(value, str):
        return _truncate_history_text(value, max_chars=max_chars)
    return None


def _bound_history_text_list(
    values: Any,
    *,
    max_items: int,
    max_chars: int,
) -> list[str]:
    if not isinstance(values, list):
        return []
    return [
        _truncate_history_text(value, max_chars=max_chars)
        for value in islice(values, max_items)
        if isinstance(value, str)
    ]


def _bound_history_runtime_value(
    value: Any,
    *,
    depth: int,
    node_budget: dict[str, int],
) -> Any:
    if node_budget["remaining"] <= 0:
        return "<summary limit>"
    node_budget["remaining"] -= 1
    if value is None or isinstance(value, (bool, float)):
        return value
    if isinstance(value, int):
        return value if value.bit_length() <= 256 else "<large integer>"
    if isinstance(value, str):
        return _truncate_history_text(value, max_chars=_HISTORY_INSPECT_MAX_TEXT_CHARS)
    if depth >= _HISTORY_INSPECT_MAX_DEPTH:
        return "<max depth>"
    if isinstance(value, (list, tuple)):
        return [
            _bound_history_runtime_value(
                item,
                depth=depth + 1,
                node_budget=node_budget,
            )
            for item in islice(value, _HISTORY_INSPECT_MAX_LIST_ITEMS)
            if node_budget["remaining"] > 0
        ]
    if isinstance(value, dict):
        bounded: dict[str, Any] = {}
        string_items = ((key, item) for key, item in value.items() if isinstance(key, str))
        for key, item in nsmallest(
            _HISTORY_INSPECT_MAX_MAPPING_ITEMS,
            string_items,
            key=lambda pair: pair[0],
        ):
            if node_budget["remaining"] <= 0:
                break
            bounded_key = _truncate_history_text(key, max_chars=_HISTORY_INSPECT_MAX_NAME_CHARS)
            bounded[bounded_key] = _bound_history_runtime_value(
                item,
                depth=depth + 1,
                node_budget=node_budget,
            )
        return bounded
    return f"<{type(value).__name__}>"


def _bound_inspected_variables(
    evidence: Any,
    *,
    max_serialized_chars: int,
) -> dict[str, Any]:
    if not isinstance(evidence, dict) or max_serialized_chars <= 0:
        return {}
    inspected: dict[str, Any] = {}
    node_budget = {"remaining": _HISTORY_INSPECT_MAX_NODES}
    string_items = ((name, value) for name, value in evidence.items() if isinstance(name, str))
    for name, raw_summary in nsmallest(
        _HISTORY_INSPECT_MAX_VARIABLES,
        string_items,
        key=lambda pair: pair[0],
    ):
        if not isinstance(raw_summary, dict):
            continue
        remaining_before = node_budget["remaining"]
        summary = {
            key: _bound_history_runtime_value(
                raw_summary[key],
                depth=0,
                node_budget=node_budget,
            )
            for key in _HISTORY_INSPECT_VALUE_KEYS
            if key in raw_summary
        }
        if not summary:
            continue
        bounded_name = _truncate_history_text(name, max_chars=_HISTORY_INSPECT_MAX_NAME_CHARS)
        candidate = {**inspected, bounded_name: summary}
        if len(json.dumps(candidate, default=str, separators=(",", ":"))) > max_serialized_chars:
            node_budget["remaining"] = remaining_before
            continue
        inspected = candidate
        if node_budget["remaining"] <= 0:
            break
    return inspected


def _summarize_history_for_prompt(
    history: list[dict[str, Any]],
    *,
    max_entries: int,
    inspect_max_chars: int = _HISTORY_INSPECT_MAX_SERIALIZED_CHARS,
) -> list[dict[str, Any]]:
    bounded_count = max(0, int(max_entries))
    bounded = history[-bounded_count:] if bounded_count else []
    summaries: list[dict[str, Any]] = []
    for entry in bounded:
        action = entry.get("action") if isinstance(entry.get("action"), dict) else {}
        event = entry.get("event") if isinstance(entry.get("event"), dict) else {}
        feedback = entry.get("feedback") if isinstance(entry.get("feedback"), dict) else {}
        evidence = event.get("evidence") if isinstance(event.get("evidence"), dict) else {}
        feedback_evidence = (
            feedback.get("evidence") if isinstance(feedback.get("evidence"), dict) else {}
        )
        event_status = event.get("status")
        feedback_status = feedback.get("status")
        action_name = action.get("action") or event.get("action")
        summary = {
            "step_id": _bound_history_scalar(entry.get("step_id")),
            "action": _bound_history_scalar(action_name),
            "unit_id": _bound_history_scalar(_action_unit_id(action, event, feedback)),
            "status": _bound_history_scalar(feedback_status or event_status),
            "event_status": _bound_history_scalar(event_status),
            "feedback_status": _bound_history_scalar(feedback_status),
            "message": _bound_history_scalar(
                feedback.get("message") or event.get("message"),
                max_chars=_HISTORY_MESSAGE_MAX_CHARS,
            ),
            "exception_type": _bound_history_scalar(evidence.get("exception_type")),
            "reward_before": _bound_history_scalar(
                _history_state_value(entry, "state_before", "reward")
            ),
            "reward_after": _bound_history_scalar(
                _history_state_value(entry, "state_after", "reward")
            ),
            "task_completed_before": _bound_history_scalar(
                _history_state_value(entry, "state_before", "task_completed")
            ),
            "task_completed_after": _bound_history_scalar(
                _history_state_value(entry, "state_after", "task_completed")
            ),
            "primitive_calls": _primitive_calls_from_history(entry),
        }
        if feedback_evidence.get("terminal_progress_unverified") is True:
            summary["terminal_progress_unverified"] = True
        progress_mode = feedback_evidence.get("progress_mode")
        if progress_mode in {"dense", "sparse_terminal"}:
            summary["progress_mode"] = progress_mode
        if action_name == "inspect_variables":
            inspected_variables = _bound_inspected_variables(
                evidence,
                max_serialized_chars=inspect_max_chars,
            )
            if inspected_variables:
                summary["inspected_variables"] = inspected_variables
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
        and len(json.dumps(_capsule_prompt_messages(prompt_text), default=str)) > prompt_char_budget
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
