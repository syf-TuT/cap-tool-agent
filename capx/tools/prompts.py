from __future__ import annotations

import json
from typing import Any

from capx.tools.schema import ToolCall, ToolSpec


def parse_tool_call_response(content: str) -> ToolCall:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        language = lines[0].strip("`").strip().lower()
        if language and language != "json":
            raise ValueError("Tool call response must be a JSON object, not Python code")
        if len(lines) < 3 or not lines[-1].strip().startswith("```"):
            raise ValueError("Tool call response must be valid fenced JSON")
        text = "\n".join(lines[1:-1]).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Tool call response must be valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("Tool call response JSON must be an object")
    return ToolCall.from_mapping(data)


def build_tool_planner_prompt(
    *,
    task: str,
    tool_specs: list[ToolSpec],
    state_summary: dict[str, Any],
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tools = [spec.to_prompt_dict() for spec in tool_specs]
    prompt_text = (
        "Task:\n"
        f"{task}\n\n"
        "Available tools:\n"
        f"{json.dumps(tools, indent=2, default=str)}\n\n"
        "Current state summary:\n"
        f"{json.dumps(state_summary, indent=2, default=str)}\n\n"
        "To pass a prior tool output into another tool, wrap its output_ref exactly, "
        'for example {"state_ref": "solve_ik.0"}. '
        "Use output_ref values exactly as shown; do not renumber them, prefix them with $, "
        "or put them in plain strings.\n\n"
        "Recent tool history:\n"
        f"{json.dumps(history[-8:], indent=2, default=str)}\n\n"
        "Respond with exactly one JSON object of this form:\n"
        '{"thought": "brief reason", "tool": "tool_name", "args": {}}\n'
        'Use {"tool": "finish", "args": {}} only when the task is complete.\n'
        "Do not write Python code."
    )
    return [
        {"role": "system", "content": "You select one robot tool call at a time."},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
    ]
