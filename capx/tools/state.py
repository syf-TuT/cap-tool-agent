from __future__ import annotations

from collections import defaultdict
from typing import Any


class ToolState:
    def __init__(self) -> None:
        self._values: dict[str, Any] = {}
        self._summaries: dict[str, Any] = {}
        self._counts: defaultdict[str, int] = defaultdict(int)

    def put(self, namespace: str, value: Any, *, summary: Any = None) -> str:
        idx = self._counts[namespace]
        self._counts[namespace] += 1
        ref = f"{namespace}.{idx}"
        self._values[ref] = value
        self._summaries[ref] = summary if summary is not None else self._default_summary(value)
        return ref

    def get(self, ref: str) -> Any:
        if ref not in self._values:
            raise KeyError(f"Unknown state ref: {ref}")
        return self._values[ref]

    def summary(self) -> dict[str, Any]:
        return dict(self._summaries)

    def resolve_refs(self, value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {"state_ref"}:
                return self.get(str(value["state_ref"]))
            return {k: self.resolve_refs(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.resolve_refs(v) for v in value]
        return value

    def _default_summary(self, value: Any) -> Any:
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is not None:
            return {"type": type(value).__name__, "shape": list(shape), "dtype": str(dtype)}
        return {"type": type(value).__name__, "repr": repr(value)[:200]}
