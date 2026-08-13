from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from capx.runtime_control.schema import CodeRegion, CodeRegionGroup


@dataclass(frozen=True)
class ProgramContractViolation:
    code: str
    message: str
    start_line: int
    end_line: int
    region_ids: tuple[str, ...] = ()
    group_ids: tuple[str, ...] = ()
    side_effect_calls: tuple[str, ...] = ()
    helper_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "source_span": {
                "start_line": self.start_line,
                "end_line": self.end_line,
            },
            "region_ids": list(self.region_ids),
            "group_ids": list(self.group_ids),
            "side_effect_calls": list(self.side_effect_calls),
            "helper_name": self.helper_name,
        }


@dataclass(frozen=True)
class _CallOccurrence:
    name: str
    is_direct_name: bool
    line: int


def analyze_capsule_program_contract(
    source: str,
    regions: list[CodeRegion],
    groups: list[CodeRegionGroup],
    *,
    side_effect_calls: set[str],
) -> list[ProgramContractViolation]:
    """Validate that robot effects remain explicit, bounded execution steps."""
    module = ast.parse(source)
    helper_nodes = {
        node.name: node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    helper_calls = {
        name: _collect_calls(node.body)
        for name, node in helper_nodes.items()
    }

    violations: list[ProgramContractViolation] = []
    for helper_name, node in helper_nodes.items():
        effects = _expand_helper_effects(
            helper_name,
            helper_calls,
            side_effect_calls,
            visiting=set(),
        )
        if not effects:
            continue
        start_line, end_line = _node_span(node)
        violations.append(
            _build_violation(
                code="effectful_helper",
                message=f"Helper '{helper_name}' can execute a robot side effect",
                start_line=start_line,
                end_line=end_line,
                regions=regions,
                groups=groups,
                side_effects=effects,
                helper_name=helper_name,
            )
        )

    for node in _collect_control_flow_nodes(module):
        effects = _resolve_effects(
            _collect_calls([node]),
            helper_calls,
            side_effect_calls,
        )
        if not effects:
            continue
        start_line, end_line = _node_span(node)
        violations.append(
            _build_violation(
                code="effectful_control_flow",
                message=(
                    f"{type(node).__name__} can execute robot side effects; "
                    "effects must be explicit top-level steps"
                ),
                start_line=start_line,
                end_line=end_line,
                regions=regions,
                groups=groups,
                side_effects=effects,
            )
        )

    top_level_calls = _collect_top_level_calls(module)
    for group in groups:
        group_calls = [
            call
            for call in top_level_calls
            if group.start_line <= call.line <= group.end_line
        ]
        effects = _resolve_effects(group_calls, helper_calls, side_effect_calls)
        if len(effects) <= 1:
            continue
        violations.append(
            _build_violation(
                code="multiple_effects_in_group",
                message=(
                    f"Execution group '{group.group_id}' contains multiple robot "
                    "side-effect calls"
                ),
                start_line=group.start_line,
                end_line=group.end_line,
                regions=regions,
                groups=groups,
                side_effects=effects,
            )
        )

    return sorted(
        set(violations),
        key=lambda violation: (
            violation.start_line,
            violation.end_line,
            violation.code,
            violation.helper_name or "",
        ),
    )


def _collect_calls(statements: list[ast.stmt]) -> list[_CallOccurrence]:
    visitor = _ExecutableCallVisitor()
    for statement in statements:
        visitor.visit(statement)
    return visitor.calls


def _collect_top_level_calls(module: ast.Module) -> list[_CallOccurrence]:
    return _collect_calls(module.body)


def _collect_control_flow_nodes(module: ast.Module) -> list[ast.AST]:
    visitor = _ControlFlowVisitor()
    for statement in module.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for helper_statement in statement.body:
                visitor.visit(helper_statement)
        else:
            visitor.visit(statement)
    return visitor.nodes


def _resolve_effects(
    calls: list[_CallOccurrence],
    helper_calls: dict[str, list[_CallOccurrence]],
    side_effect_calls: set[str],
) -> tuple[str, ...]:
    effects: list[str] = []
    for call in calls:
        if call.name in side_effect_calls:
            effects.append(call.name)
        elif call.is_direct_name and call.name in helper_calls:
            effects.extend(
                _expand_helper_effects(
                    call.name,
                    helper_calls,
                    side_effect_calls,
                    visiting=set(),
                )
            )
    return tuple(effects)


def _expand_helper_effects(
    helper_name: str,
    helper_calls: dict[str, list[_CallOccurrence]],
    side_effect_calls: set[str],
    *,
    visiting: set[str],
) -> tuple[str, ...]:
    if helper_name in visiting:
        return ()

    effects: list[str] = []
    next_visiting = {*visiting, helper_name}
    for call in helper_calls.get(helper_name, []):
        if call.name in side_effect_calls:
            effects.append(call.name)
        elif call.is_direct_name and call.name in helper_calls:
            effects.extend(
                _expand_helper_effects(
                    call.name,
                    helper_calls,
                    side_effect_calls,
                    visiting=next_visiting,
                )
            )
    return tuple(effects)


def _build_violation(
    *,
    code: str,
    message: str,
    start_line: int,
    end_line: int,
    regions: list[CodeRegion],
    groups: list[CodeRegionGroup],
    side_effects: tuple[str, ...],
    helper_name: str | None = None,
) -> ProgramContractViolation:
    return ProgramContractViolation(
        code=code,
        message=message,
        start_line=start_line,
        end_line=end_line,
        region_ids=tuple(
            region.region_id
            for region in regions
            if _spans_overlap(start_line, end_line, region.start_line, region.end_line)
        ),
        group_ids=tuple(
            group.group_id
            for group in groups
            if _spans_overlap(start_line, end_line, group.start_line, group.end_line)
        ),
        side_effect_calls=side_effects,
        helper_name=helper_name,
    )


def _spans_overlap(
    first_start: int,
    first_end: int,
    second_start: int,
    second_end: int,
) -> bool:
    return first_start <= second_end and second_start <= first_end


def _node_span(node: ast.AST) -> tuple[int, int]:
    start_line = int(getattr(node, "lineno", 1))
    decorators = getattr(node, "decorator_list", ())
    if decorators:
        start_line = min(
            start_line,
            *(int(decorator.lineno) for decorator in decorators),
        )
    end_line = int(getattr(node, "end_lineno", start_line) or start_line)
    return start_line, end_line


def _callable_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


class _ExecutableCallVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: list[_CallOccurrence] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Call(self, node: ast.Call) -> None:
        name = _callable_name(node.func)
        if name is not None:
            self.calls.append(
                _CallOccurrence(
                    name=name,
                    is_direct_name=isinstance(node.func, ast.Name),
                    line=node.lineno,
                )
            )
        self.generic_visit(node)


_CONTROL_FLOW_TYPES: tuple[type[ast.AST], ...] = (
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
)
if hasattr(ast, "TryStar"):
    _CONTROL_FLOW_TYPES += (ast.TryStar,)


class _ControlFlowVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.nodes: list[ast.AST] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, _CONTROL_FLOW_TYPES):
            self.nodes.append(node)
        super().generic_visit(node)
