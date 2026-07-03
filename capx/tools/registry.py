from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

from capx.tools.schema import ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._functions: dict[str, Callable[..., Any]] = {}

    def register(self, spec: ToolSpec, fn: Callable[..., Any]) -> None:
        self._specs[spec.name] = spec
        self._functions[spec.name] = fn

    def spec(self, name: str) -> ToolSpec:
        if name not in self._specs:
            raise KeyError(f"Unknown tool: {name}")
        return self._specs[name]

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._functions:
            raise KeyError(f"Unknown tool: {name}")
        return self._functions[name]

    def prompt_specs(self) -> list[dict[str, Any]]:
        return [spec.to_prompt_dict() for spec in self.specs()]


def build_registry_from_apis(
    apis: Mapping[str, Any],
    metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    overlays = metadata or {}
    for api in apis.values():
        for name, fn in api.functions().items():
            spec = ToolSpec(
                name=name,
                description=inspect.getdoc(fn) or "",
                input_schema=_schema_from_signature(fn),
            )
            for key, value in overlays.get(name, {}).items():
                setattr(spec, key, value)
            registry.register(spec, fn)
    return registry


def _schema_from_signature(fn: Callable[..., Any]) -> dict[str, dict[str, Any]]:
    schema: dict[str, dict[str, Any]] = {}
    for name, param in inspect.signature(fn).parameters.items():
        if name == "self":
            continue
        entry: dict[str, Any] = {"required": param.default is inspect.Parameter.empty}
        if param.annotation is not inspect.Parameter.empty:
            entry["type"] = getattr(param.annotation, "__name__", str(param.annotation))
        if param.default is not inspect.Parameter.empty:
            entry["default"] = param.default
        schema[name] = entry
    return schema
