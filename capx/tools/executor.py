from __future__ import annotations

import contextlib
import io
import time
from collections.abc import Mapping
from typing import Any

from capx.tools.registry import ToolRegistry
from capx.tools.schema import ToolCall, ToolResult
from capx.tools.state import ToolState


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, state: ToolState) -> None:
        self.registry = registry
        self.state = state

    def run(self, tool_call: ToolCall) -> ToolResult:
        try:
            fn = self.registry.get(tool_call.tool)
        except KeyError as exc:
            return ToolResult(
                tool=tool_call.tool,
                status="invalid",
                failure_type="unknown_tool",
                message=str(exc),
            )

        stdout = io.StringIO()
        stderr = io.StringIO()
        start = time.perf_counter()
        try:
            args = self.state.resolve_refs(tool_call.args)
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                output = fn(**args)
        except BaseException as exc:
            return ToolResult.failed(
                tool=tool_call.tool,
                failure_type="exception",
                message=str(exc),
                exception_type=type(exc).__name__,
                stderr=stderr.getvalue(),
            )

        output_ref, output_summary = self._store_or_summarize(tool_call.tool, output)
        return ToolResult(
            tool=tool_call.tool,
            status="success",
            output_ref=output_ref,
            output_summary=output_summary,
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
            duration_s=time.perf_counter() - start,
        )

    def _store_or_summarize(self, tool: str, output: Any) -> tuple[str | None, Any]:
        if hasattr(output, "shape") or isinstance(output, Mapping):
            ref = self.state.put(tool, output)
            return ref, self.state.summary()[ref]
        return None, output
