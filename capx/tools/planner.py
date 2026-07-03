from __future__ import annotations

from collections.abc import Callable
from typing import Any

from capx.tools.prompts import parse_tool_call_response
from capx.tools.schema import ToolCall


class ScriptedToolPlanner:
    def __init__(self, script: list[dict[str, Any]]) -> None:
        self._script = list(script)
        self._idx = 0

    def next_call(self, *args: Any, **kwargs: Any) -> ToolCall:
        if self._idx >= len(self._script):
            return ToolCall(tool="finish", args={})
        data = self._script[self._idx]
        self._idx += 1
        return ToolCall.from_mapping(data)


class LlmToolPlanner:
    def __init__(
        self,
        *,
        query_model: Callable[[Any, list[dict[str, Any]]], dict[str, Any]],
        args: Any,
    ) -> None:
        self._query_model = query_model
        self._args = args

    def next_call(self, *, prompt: list[dict[str, Any]]) -> ToolCall:
        content = self._query_model(self._args, prompt)
        return parse_tool_call_response(content["content"])
