from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass

from capx.runtime_control.schema import CodeRegion


ROBOT_SIDE_EFFECT_CALLS = {
    "move_to_joints",
    "move_to_pose",
    "move_to_position",
    "move_to",
    "close_gripper",
    "open_gripper",
    "control_gripper",
    "set_gripper",
    "execute_trajectory",
}


@dataclass(frozen=True)
class RegionAnalysis:
    region_id: str
    primitive_calls: list[str]
    defined_names: list[str]
    used_names: list[str]
    has_robot_side_effect: bool
    has_structural_effect: bool
    defined_functions: list[str]
    top_level_call_names: list[str]
    lexical_side_effect_calls: list[str]
    defined_function_side_effect_calls: dict[str, list[str]]
    defined_function_call_names: dict[str, list[str]]


def segment_python_code(source: str) -> list[CodeRegion]:
    """Split Python source into deterministic top-level execution regions."""
    module = ast.parse(source)
    lines = source.splitlines(keepends=True)
    regions: list[CodeRegion] = []
    region_start_lines: list[int] = []

    for node in module.body:
        start_line = _effective_start_line(node)
        if start_line is None:
            continue
        if region_start_lines and start_line == region_start_lines[-1]:
            continue
        region_start_lines.append(start_line)

    for region_index, start_line in enumerate(region_start_lines):
        next_start_line = (
            region_start_lines[region_index + 1]
            if region_index + 1 < len(region_start_lines)
            else None
        )
        slice_start_line = 1 if region_index == 0 else start_line
        end_line = next_start_line - 1 if next_start_line is not None else len(lines)
        region_source = "".join(lines[slice_start_line - 1 : end_line])
        regions.append(
            CodeRegion(
                region_id=f"region_{region_index + 1}",
                start_line=slice_start_line,
                end_line=end_line,
                source=region_source,
            )
        )

    return regions


def _effective_start_line(node: ast.AST) -> int | None:
    start_line = getattr(node, "lineno", None)
    decorator_list = getattr(node, "decorator_list", None)
    if not decorator_list:
        return start_line
    decorator_lines = [
        decorator.lineno for decorator in decorator_list if getattr(decorator, "lineno", None)
    ]
    if not decorator_lines:
        return start_line
    if start_line is None:
        return min(decorator_lines)
    return min(start_line, *decorator_lines)


def analyze_python_regions(
    source: str,
    regions: list[CodeRegion],
    *,
    side_effect_calls: set[str],
) -> list[RegionAnalysis]:
    """Return structural analysis facts for source regions.

    ``source`` is accepted for the public structural-facts API and for the
    later normalizer handoff; analysis remains anchored to each region source.
    """
    return [_analyze_region(region, side_effect_calls) for region in regions]


def _analyze_region(region: CodeRegion, side_effect_calls: set[str]) -> RegionAnalysis:
    try:
        module = ast.parse(region.source)
    except SyntaxError:
        return RegionAnalysis(
            region_id=region.region_id,
            primitive_calls=[],
            defined_names=[],
            used_names=[],
            has_robot_side_effect=False,
            has_structural_effect=False,
            defined_functions=[],
            top_level_call_names=[],
            lexical_side_effect_calls=[],
            defined_function_side_effect_calls={},
            defined_function_call_names={},
        )

    primitive_calls, executable_direct_call_names = _executable_call_facts(module.body)
    lexical_side_effect_calls = _ordered_unique(
        name for name in primitive_calls if name in side_effect_calls
    )
    defined_function_side_effect_calls, defined_function_call_names = (
        _defined_function_effect_facts(module, side_effect_calls)
    )
    defined_names = _ordered_unique(_defined_names(module))
    defined_name_set = set(defined_names)
    used_names = _ordered_unique(
        name for name in _used_names(module) if name not in defined_name_set
    )
    return RegionAnalysis(
        region_id=region.region_id,
        primitive_calls=primitive_calls,
        defined_names=defined_names,
        used_names=used_names,
        has_robot_side_effect=bool(lexical_side_effect_calls),
        has_structural_effect=_is_effect_region(module),
        defined_functions=_defined_functions(module),
        top_level_call_names=executable_direct_call_names,
        lexical_side_effect_calls=lexical_side_effect_calls,
        defined_function_side_effect_calls=defined_function_side_effect_calls,
        defined_function_call_names=defined_function_call_names,
    )


def _is_effect_region(module: ast.Module) -> bool:
    """A region is an effect if any top-level statement is a bare call whose
    return value is discarded (e.g. ``move_to_joints(...)`` or ``publish(x)``).

    Boundary detection is structural and domain-agnostic: it does not consult
    any robot-primitive vocabulary.
    """
    return any(
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        for node in module.body
    )


def _callable_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _defined_functions(module: ast.Module) -> list[str]:
    return [
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _defined_function_effect_facts(
    module: ast.Module,
    side_effect_calls: set[str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    side_effects: dict[str, list[str]] = {}
    helper_calls: dict[str, list[str]] = {}

    for node in module.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        function_side_effects, function_helper_calls = _function_effect_facts(
            node,
            side_effect_calls,
        )
        side_effects[node.name] = function_side_effects
        helper_calls[node.name] = function_helper_calls

    return side_effects, helper_calls


def _function_effect_facts(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    side_effect_calls: set[str],
) -> tuple[list[str], list[str]]:
    call_names, direct_call_names = _executable_call_facts(node.body)
    local_function_facts = _local_function_effect_facts(node.body, side_effect_calls)
    local_function_names = set(local_function_facts)
    function_side_effects = [
        name for name in call_names if name in side_effect_calls
    ]
    external_helper_calls: list[str] = []

    for call_name in direct_call_names:
        if call_name in side_effect_calls:
            continue
        if call_name in local_function_names:
            function_side_effects.extend(
                _resolve_local_function_side_effects(
                    call_name,
                    local_function_facts,
                    visiting=set(),
                )
            )
            continue
        external_helper_calls.append(call_name)

    return _ordered_unique(function_side_effects), _ordered_unique(external_helper_calls)


def _local_function_effect_facts(
    statements: list[ast.stmt],
    side_effect_calls: set[str],
) -> dict[str, tuple[list[str], list[str]]]:
    return {
        name: _raw_function_effect_facts(function, side_effect_calls)
        for name, function in _local_function_defs(statements).items()
    }


def _raw_function_effect_facts(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    side_effect_calls: set[str],
) -> tuple[list[str], list[str]]:
    call_names, direct_call_names = _executable_call_facts(node.body)
    return (
        _ordered_unique(name for name in call_names if name in side_effect_calls),
        _ordered_unique(name for name in direct_call_names if name not in side_effect_calls),
    )


def _resolve_local_function_side_effects(
    name: str,
    local_function_facts: dict[str, tuple[list[str], list[str]]],
    *,
    visiting: set[str],
) -> list[str]:
    if name in visiting:
        return []
    direct_side_effects, direct_calls = local_function_facts.get(name, ([], []))
    side_effects = list(direct_side_effects)
    next_visiting = {*visiting, name}

    for call_name in direct_calls:
        if call_name in local_function_facts:
            side_effects.extend(
                _resolve_local_function_side_effects(
                    call_name,
                    local_function_facts,
                    visiting=next_visiting,
                )
            )

    return _ordered_unique(side_effects)


def _local_function_defs(
    statements: list[ast.stmt],
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    visitor = _LocalFunctionDefVisitor()
    for statement in statements:
        visitor.visit(statement)
    return visitor.functions


def _executable_call_facts(statements: list[ast.stmt]) -> tuple[list[str], list[str]]:
    visitor = _ExecutableCallVisitor()
    for statement in statements:
        visitor.visit(statement)
    return _ordered_unique(visitor.call_names), _ordered_unique(visitor.direct_call_names)


class _LocalFunctionDefVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions[node.name] = node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.functions[node.name] = node

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return


class _ExecutableCallVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.call_names: list[str] = []
        self.direct_call_names: list[str] = []

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
        if name:
            self.call_names.append(name)
        if isinstance(node.func, ast.Name):
            self.direct_call_names.append(node.func.id)

        if isinstance(node.func, ast.Lambda):
            self.visit(node.func.body)
        else:
            self.visit(node.func)

        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)


def _defined_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = child.targets if isinstance(child, ast.Assign) else [child.target]
            for target in targets:
                names.extend(_target_names(target))
        elif isinstance(child, ast.NamedExpr):
            names.extend(_target_names(child.target))
        elif isinstance(child, ast.For):
            names.extend(_target_names(child.target))
        elif isinstance(child, ast.With):
            for item in child.items:
                if item.optional_vars is not None:
                    names.extend(_target_names(item.optional_vars))
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(child.name)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                names.append(alias.asname or alias.name.split(".")[0])
    return names


def _target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        names: list[str] = []
        for element in node.elts:
            names.extend(_target_names(element))
        return names
    if isinstance(node, ast.Starred):
        return _target_names(node.value)
    return []


def _used_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            names.append(child.id)
    return names


def _ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
