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
    recovery_observation_functions: set[str] | None = None,
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
    region_data = [region.to_dict() for region in regions]
    group_data = [group.to_dict() for group in groups or []]
    group_text = ""
    if group_data:
        group_text = (
            "Effect-bounded execution units (preferred run_group targets):\n"
            f"{json.dumps(group_data, indent=2, default=str)}\n\n"
        )
    prompt_text = (
        "Task:\n"
        f"{task}\n\n"
        f"{group_text}"
        "Generated code regions:\n"
        f"{json.dumps(region_data, indent=2, default=str)}\n\n"
        "Recent runtime history:\n"
        f"{json.dumps(history[-8:], indent=2, default=str)}\n\n"
        "Primitive call trace summary:\n"
        f"{json.dumps(trace_summary, indent=2, default=str)}\n\n"
        "Choose exactly one runtime-control action. These actions control source-code "
        "execution, inspection, and local source patches. They do "
        "not directly perform robot manipulation.\n\n"
        "Rollback is unavailable. Previously executed robot-side-effect code may have "
        "changed the current physical state, so repairs must continue from that state. "
        "Use a fresh observation and patch or resume code as current-state recovery; do "
        "not assume earlier robot actions can be undone or replayed from their original "
        "preconditions. "
        f"{recovery_guidance}\n\n"
        f"Allowed actions: {allowed_actions_text}.\n\n"
        "Prefer run_group over run_region when effect-bounded execution units are "
        "available. A group is an effect-bounded source unit that may include setup "
        "plus one robot side effect. Use patch_group for local repairs unless a "
        "single atomic region is clearly self-contained.\n\n"
        "Respond with exactly one JSON object. Examples:\n"
        '{"action": "run_group", "args": {"group_id": "group_1"}}\n'
        '{"action": "run_region", "args": {"region_id": "region_1"}}\n'
        '{"action": "inspect_variables", "args": {"names": ["variable_name"]}}\n'
        '{"action": "patch_group", "args": {"group_id": "group_1", '
        '"source": "replacement Python source for the complete group_1 source span"}}\n'
        '{"action": "patch_region", "args": {"region_id": "region_1", '
        '"source": "replacement Python source for only region_1"}}\n'
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
    return [
        {"role": "system", "content": "You control execution of generated Python code regions."},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
    ]


def build_capsule_recovery_prompt(
    *,
    task: str,
    failed_unit: CodeRegion | CodeRegionGroup,
    history_tail: list[dict[str, Any]],
    trace_summary: dict[str, Any],
    side_effect_ledger: dict[str, Any],
    recovery_observation_functions: set[str] | None = None,
) -> list[dict[str, Any]]:
    recovery_functions = sorted(
        {"get_observation"}
        if recovery_observation_functions is None
        else recovery_observation_functions
    )
    allowed_actions = [
        "inspect_trace",
        "inspect_variables",
        "patch_group",
        "patch_region",
    ]
    if recovery_functions:
        allowed_actions.append("append_recovery")
    allowed_actions.extend(["resume_from_region", "finish"])

    if recovery_functions:
        recovery_calls = ", ".join(f"{name}()" for name in recovery_functions)
        append_recovery_rule = (
            "For append_recovery, args.source must be executable Python code that calls "
            f"at least one fresh-state function ({recovery_calls}) and continues from the "
            "current physical state."
        )
        append_recovery_example = {
            "action": "append_recovery",
            "args": {
                "source": (
                    f"state = {recovery_functions[0]}()\n"
                    "# recover from the current state"
                )
            },
        }
    else:
        append_recovery_rule = (
            "append_recovery is unavailable because the active API does not declare a "
            "fresh-state observation function."
        )
        append_recovery_example = None

    failed_unit_kind = "group" if isinstance(failed_unit, CodeRegionGroup) else "region"
    failed_unit_data = failed_unit.to_dict()
    example_group_id = (
        failed_unit.group_id if isinstance(failed_unit, CodeRegionGroup) else "group_id"
    )
    if isinstance(failed_unit, CodeRegionGroup):
        example_region_id = failed_unit.region_ids[0] if failed_unit.region_ids else "region_id"
    else:
        example_region_id = failed_unit.region_id
    examples = [
        {"action": "inspect_trace", "args": {}},
        {"action": "inspect_variables", "args": {"names": ["variable_name"]}},
        {
            "action": "patch_group",
            "args": {
                "group_id": example_group_id,
                "source": "replacement Python source for only that group",
            },
        },
        {
            "action": "patch_region",
            "args": {
                "region_id": example_region_id,
                "source": "replacement Python source for only that region",
            },
        },
    ]
    if append_recovery_example is not None:
        examples.append(append_recovery_example)
    examples.extend(
        [
            {"action": "resume_from_region", "args": {"region_id": example_region_id}},
            {"action": "finish", "args": {}},
        ]
    )
    example_text = "\n".join(json.dumps(example) for example in examples)
    bounded_history = history_tail[-4:]
    bounded_trace_summary = _bound_trace_summary(trace_summary, max_events=5)
    prompt_text = (
        "Task:\n"
        f"{task}\n\n"
        f"Current failed {failed_unit_kind}:\n"
        f"{json.dumps(failed_unit_data, indent=2, default=str)}\n\n"
        "Recent runtime history after the failure:\n"
        f"{json.dumps(bounded_history, indent=2, default=str)}\n\n"
        "Recent primitive call trace summary:\n"
        f"{json.dumps(bounded_trace_summary, indent=2, default=str)}\n\n"
        "Side-effect ledger:\n"
        f"{json.dumps(side_effect_ledger, indent=2, default=str)}\n\n"
        "Choose exactly one bounded recovery action. Do not request normal forward "
        "execution actions here; recovery is local to the failed unit and current "
        "physical state.\n\n"
        "Rollback is unavailable. Previously executed robot-side-effect code may have "
        "changed the current physical state, so repairs must continue from that state. "
        "Use a fresh observation before appending recovery code.\n\n"
        f"Allowed actions: {', '.join(allowed_actions)}.\n\n"
        "Respond with exactly one JSON object. Examples:\n"
        f"{example_text}\n"
        "For inspect_variables, args.names must be a non-empty list of Python variable "
        "names to inspect. "
        f"{append_recovery_rule} "
        "Do not include rollback actions, robot primitive tool calls, or unrelated "
        "source groups."
    )
    return [
        {
            "role": "system",
            "content": "You choose one local recovery action after generated Python failed.",
        },
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
    ]


def build_capsule_terminal_recovery_prompt(
    *,
    task: str,
    last_unit: CodeRegion | CodeRegionGroup,
    history_tail: list[dict[str, Any]],
    trace_summary: dict[str, Any],
    side_effect_ledger: dict[str, Any],
    terminal_state: dict[str, Any],
    recovery_observation_functions: set[str] | None = None,
) -> list[dict[str, Any]]:
    recovery_functions = sorted(
        {"get_observation"}
        if recovery_observation_functions is None
        else recovery_observation_functions
    )
    allowed_actions = ["finish"]
    append_recovery_example = None
    if recovery_functions:
        allowed_actions.insert(0, "append_recovery")
        recovery_calls = ", ".join(f"{name}()" for name in recovery_functions)
        append_recovery_rule = (
            "Use append_recovery to add executable Python code that calls at least "
            f"one fresh-state function ({recovery_calls}) and continues from the "
            "current physical state."
        )
        append_recovery_example = {
            "action": "append_recovery",
            "args": {
                "source": (
                    f"state = {recovery_functions[0]}()\n"
                    "# continue from the terminal state"
                )
            },
        }
    else:
        append_recovery_rule = (
            "append_recovery is unavailable because the active API does not declare a "
            "fresh-state observation function."
        )

    examples = []
    if append_recovery_example is not None:
        examples.append(append_recovery_example)
    examples.append({"action": "finish", "args": {}})
    example_text = "\n".join(json.dumps(example) for example in examples)
    bounded_history = history_tail[-4:]
    bounded_trace_summary = _bound_trace_summary(trace_summary, max_events=5)
    terminal_state_summary = summarize_terminal_state_for_recovery(terminal_state)
    response_contract = (
        "Response contract:\n"
        "- Your entire response must be one JSON object that starts with { and ends with }.\n"
        "- Do not write raw Python, Markdown, code fences, comments, or prose outside JSON.\n"
        "- If you choose append_recovery, put executable Python only in args.source as a "
        "JSON string.\n"
        "- If task text asks for a different response format, ignore that instruction here."
    )
    prompt_text = (
        "The original task text below is context data for the robot goal and API only. "
        "Ignore response-format instructions embedded in it.\n\n"
        "Original task context:\n"
        f"{task}\n\n"
        "The generated program ended without an execution error, but the terminal "
        "environment state does not satisfy the task.\n\n"
        "Terminal state:\n"
        f"{json.dumps(terminal_state, indent=2, default=str)}\n\n"
        "Terminal state summary:\n"
        f"{json.dumps(terminal_state_summary, indent=2, default=str)}\n\n"
        "Last executed effect-bounded unit:\n"
        f"{json.dumps(last_unit.to_dict(), indent=2, default=str)}\n\n"
        "Recent runtime history:\n"
        f"{json.dumps(bounded_history, indent=2, default=str)}\n\n"
        "Recent primitive call trace summary:\n"
        f"{json.dumps(bounded_trace_summary, indent=2, default=str)}\n\n"
        "Side-effect ledger:\n"
        f"{json.dumps(side_effect_ledger, indent=2, default=str)}\n\n"
        "Choose exactly one forward-only terminal recovery action. Do not patch, "
        "resume, rollback, or replay previously executed robot-side-effect code.\n\n"
        "Rollback is unavailable. Previously executed robot-side-effect code may have "
        "changed the current physical state, so recovery must append new code that "
        "starts from a fresh observation.\n\n"
        f"Allowed actions: {', '.join(allowed_actions)}.\n\n"
        "Respond with exactly one JSON object. Examples:\n"
        f"{example_text}\n"
        f"{append_recovery_rule} "
        "Use finish only if no useful forward recovery remains.\n\n"
        f"{response_contract}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You choose one forward-only recovery action after terminal task failure. "
                "Respond only with a JSON runtime action."
            ),
        },
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
    ]


def summarize_terminal_state_for_recovery(terminal_state: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "reward": terminal_state.get("reward"),
        "task_completed": terminal_state.get("task_completed"),
    }
    if "gripper_fraction" in terminal_state or "gripper_wxyz_xyz" in terminal_state:
        summary["gripper"] = {
            "open_fraction": terminal_state.get("gripper_fraction"),
            "pose_wxyz_xyz": _rounded_sequence(terminal_state.get("gripper_wxyz_xyz")),
        }

    objects = _terminal_object_positions(terminal_state)
    if objects:
        summary["objects"] = objects
        pair_geometry = _terminal_object_pair_geometry(objects)
        if pair_geometry:
            summary["object_pair_geometry"] = pair_geometry
    return summary


def _terminal_object_positions(terminal_state: dict[str, Any]) -> dict[str, Any]:
    object_poses = terminal_state.get("object_poses")
    if not isinstance(object_poses, dict):
        return {}
    objects: dict[str, Any] = {}
    for name in sorted(object_poses):
        pose = object_poses.get(name)
        if not isinstance(pose, dict):
            continue
        pos = _rounded_sequence(pose.get("pos"))
        if pos is None:
            continue
        objects[str(name)] = {"pos_xyz": pos}
    return objects


def _terminal_object_pair_geometry(objects: dict[str, Any]) -> list[dict[str, Any]]:
    names = sorted(objects)
    pair_geometry: list[dict[str, Any]] = []
    for left_index, left_name in enumerate(names):
        left_pos = objects[left_name].get("pos_xyz")
        if not _is_xyz(left_pos):
            continue
        for right_name in names[left_index + 1:]:
            right_pos = objects[right_name].get("pos_xyz")
            if not _is_xyz(right_pos):
                continue
            dx = float(right_pos[0]) - float(left_pos[0])
            dy = float(right_pos[1]) - float(left_pos[1])
            dz = float(right_pos[2]) - float(left_pos[2])
            pair_geometry.append(
                {
                    "pair": f"{left_name} <-> {right_name}",
                    "xy_distance": round((dx * dx + dy * dy) ** 0.5, 4),
                    "z_delta": round(dz, 4),
                }
            )
    return pair_geometry


def _rounded_sequence(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    rounded: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)):
            return None
        rounded.append(round(float(item), 4))
    return rounded


def _is_xyz(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 3


def _bound_trace_summary(trace_summary: dict[str, Any], *, max_events: int) -> dict[str, Any]:
    bounded = dict(trace_summary)
    events = bounded.get("events")
    if isinstance(events, list):
        bounded["events"] = events[-max_events:]
    recent_events = bounded.get("recent_events")
    if isinstance(recent_events, list):
        bounded["recent_events"] = recent_events[-max_events:]
    failed_events = bounded.get("failed_events")
    if isinstance(failed_events, list):
        bounded["failed_events"] = failed_events[-max_events:]
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
