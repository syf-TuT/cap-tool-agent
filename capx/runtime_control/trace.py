from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any

import numpy as np


class RuntimeTrace:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def log(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def mark(self) -> int:
        return len(self.events)

    def events_since(self, index: int) -> list[dict[str, Any]]:
        return list(self.events[index:])

    def summary(self) -> dict[str, Any]:
        return {"events": list(self.events)}


def wrap_function_for_trace(name: str, fn: Callable[..., Any], trace: RuntimeTrace) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        event: dict[str, Any] = {
            "name": name,
            "args": [_summarize_value(arg) for arg in args],
            "kwargs": {key: _summarize_value(value) for key, value in kwargs.items()},
        }
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            event.update(
                {
                    "status": "failed",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "duration_s": time.perf_counter() - start,
                }
            )
            trace.log(event)
            raise

        event.update(
            {
                "status": "success",
                "result": _summarize_value(result),
                "duration_s": time.perf_counter() - start,
            }
        )
        trace.log(event)
        return result

    return wrapped


def _summarize_value(value: Any) -> dict[str, Any]:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    summary: dict[str, Any] = {"type": type(value).__name__}
    if shape is not None:
        summary["shape"] = list(shape)
    if dtype is not None:
        summary["dtype"] = str(dtype)
    if isinstance(value, np.ndarray) and value.size <= 32 and value.dtype.kind in "biuf":
        summary["value"] = value.tolist()
    if shape is None:
        summary["repr"] = repr(value)[:200]
    return summary
