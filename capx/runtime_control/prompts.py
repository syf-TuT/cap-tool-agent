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
) -> list[dict[str, Any]]:
    region_data = [region.to_dict() for region in regions]
    group_data = [group.to_dict() for group in groups or []]
    group_text = ""
    if group_data:
        group_text = (
            "Generated code groups (preferred execution units):\n"
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
        "preconditions. For recovery after robot side effects, prefer appending new "
        "recovery code with append_recovery; the appended code must call "
        "get_observation() so it starts from the current physical state.\n\n"
        "Allowed actions: run_group, run_region, inspect_trace, inspect_variables, "
        "patch_group, patch_region, append_recovery, resume_from_region, finish.\n\n"
        "Prefer run_group over run_region when code groups are available. A group is a "
        "semantic source chunk that may include setup plus one robot side effect. Use "
        "patch_group for local repairs unless a single atomic region is clearly "
        "self-contained.\n\n"
        "Respond with exactly one JSON object. Examples:\n"
        '{"action": "run_group", "args": {"group_id": "group_1"}}\n'
        '{"action": "run_region", "args": {"region_id": "region_1"}}\n'
        '{"action": "inspect_variables", "args": {"names": ["variable_name"]}}\n'
        '{"action": "patch_group", "args": {"group_id": "group_1", '
        '"source": "replacement Python source for the complete group_1 source span"}}\n'
        '{"action": "patch_region", "args": {"region_id": "region_1", '
        '"source": "replacement Python source for only region_1"}}\n'
        '{"action": "append_recovery", "args": {"source": '
        '"obs = get_observation()\\n# recover from the current physical state"}}\n'
        "For inspect_variables, args.names must be a non-empty list of Python variable "
        "names to inspect. Do not pass region_id to inspect_variables.\n"
        "For patch_group, args.source must be the complete replacement Python source "
        "for only the requested source group.\n"
        "For patch_region, args.source must be the complete replacement Python source "
        "for only the requested source region. Do not use new_source or patch for "
        "patch_region replacement text.\n"
        "For append_recovery, args.source must be executable Python code that includes "
        "get_observation() and continues from the current physical state. Do not ask "
        "for robot primitives as tools."
    )
    return [
        {"role": "system", "content": "You control execution of generated Python code regions."},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
    ]


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
