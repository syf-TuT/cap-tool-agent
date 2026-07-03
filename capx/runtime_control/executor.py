from __future__ import annotations

import contextlib
import io
import time
import traceback
from typing import Any

from capx.runtime_control.schema import CodeRegion, RuntimeEvent
from capx.runtime_control.trace import RuntimeTrace


class CapsuleExecutor:
    def __init__(
        self,
        *,
        base_globals: dict[str, Any],
        trace: RuntimeTrace | None = None,
    ) -> None:
        self.globals = dict(base_globals)
        self.trace = trace

    def run_region(self, region: CodeRegion) -> RuntimeEvent:
        stdout = io.StringIO()
        stderr = io.StringIO()
        start = time.perf_counter()
        try:
            code = compile(region.source, f"<{region.region_id}>", "exec")
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(code, self.globals, self.globals)
        except BaseException as exc:
            traceback.print_exc(file=stderr)
            return RuntimeEvent(
                action="run_region",
                status="failed",
                region_id=region.region_id,
                message=str(exc),
                evidence={
                    "exception_type": type(exc).__name__,
                    "source_span": {
                        "start_line": region.start_line,
                        "end_line": region.end_line,
                    },
                },
                stdout=stdout.getvalue(),
                stderr=stderr.getvalue(),
                duration_s=time.perf_counter() - start,
            )

        return RuntimeEvent(
            action="run_region",
            status="success",
            region_id=region.region_id,
            evidence={
                "source_span": {
                    "start_line": region.start_line,
                    "end_line": region.end_line,
                },
            },
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
            duration_s=time.perf_counter() - start,
        )
