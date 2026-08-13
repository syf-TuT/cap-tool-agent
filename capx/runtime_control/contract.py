from __future__ import annotations

import ast
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
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
class ProgramContractAnalysis:
    violations: tuple[ProgramContractViolation, ...]
    effectful_region_ids: tuple[str, ...]
    effectful_group_ids: tuple[str, ...]


_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
_EFFECT_SUMMARY_LIMIT = 2


@dataclass(frozen=True)
class _CallOccurrence:
    name: str
    is_direct_name: bool
    line: int
    column: int


@dataclass
class _Scope:
    scope_id: int
    parent_scope_id: int | None
    bindings: dict[str, list[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class _Definition:
    definition_id: int
    node: _FunctionNode
    defining_scope_id: int
    body_scope_id: int
    is_top_level: bool


@dataclass(frozen=True)
class _DefinitionGraph:
    module_scope_id: int
    scopes: dict[int, _Scope]
    definitions: dict[int, _Definition]
    top_level_definition_ids: tuple[int, ...]


@dataclass(frozen=True)
class _ResolvedCall:
    line: int
    column: int
    effect_name: str | None = None
    target_definition_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class _DefinitionAnalysis:
    calls: tuple[_ResolvedCall, ...]


def analyze_capsule_program_contract(
    source: str,
    regions: list[CodeRegion],
    groups: list[CodeRegionGroup],
    *,
    side_effect_calls: set[str],
) -> list[ProgramContractViolation]:
    """Validate that robot effects remain explicit, bounded execution steps."""
    return list(
        analyze_capsule_program_contract_details(
            source,
            regions,
            groups,
            side_effect_calls=side_effect_calls,
        ).violations
    )


def analyze_capsule_program_contract_details(
    source: str,
    regions: list[CodeRegion],
    groups: list[CodeRegionGroup],
    *,
    side_effect_calls: set[str],
) -> ProgramContractAnalysis:
    """Return violations and units whose execution can reach robot effects."""
    module = ast.parse(source)
    graph = _DefinitionGraphBuilder().build(module)
    definition_analyses = _analyze_definitions(
        graph,
        side_effect_calls=side_effect_calls,
    )
    definition_summaries = _compute_definition_effect_summaries(
        definition_analyses
    )
    module_calls = _resolve_calls(
        _collect_calls(module.body),
        graph,
        scope_id=graph.module_scope_id,
        side_effect_calls=side_effect_calls,
    )

    violations = _helper_violations(
        graph,
        definition_summaries,
        regions=regions,
        groups=groups,
    )
    violations.extend(
        _control_flow_violations(
            module,
            graph,
            definition_analyses,
            definition_summaries,
            regions=regions,
            groups=groups,
            side_effect_calls=side_effect_calls,
        )
    )
    violations.extend(
        _group_violations(
            module_calls,
            definition_summaries,
            regions=regions,
            groups=groups,
        )
    )
    sorted_violations = tuple(
        sorted(
            set(violations),
            key=lambda violation: (
                violation.start_line,
                violation.end_line,
                violation.code,
                violation.helper_name or "",
            ),
        )
    )
    effectful_region_ids, effectful_group_ids = _effectful_executable_unit_ids(
        module_calls,
        definition_summaries,
        regions=regions,
        groups=groups,
    )
    return ProgramContractAnalysis(
        violations=sorted_violations,
        effectful_region_ids=effectful_region_ids,
        effectful_group_ids=effectful_group_ids,
    )


class _DefinitionGraphBuilder:
    def __init__(self) -> None:
        self._next_scope_id = 0
        self._next_definition_id = 0
        self._scopes: dict[int, _Scope] = {}
        self._definitions: dict[int, _Definition] = {}

    def build(self, module: ast.Module) -> _DefinitionGraph:
        module_scope_id = self._new_scope(parent_scope_id=None)
        top_level_nodes = _collect_scope_function_definitions(module.body)
        top_level_definition_ids = self._register_definitions(
            top_level_nodes,
            defining_scope_id=module_scope_id,
            is_top_level=True,
        )
        return _DefinitionGraph(
            module_scope_id=module_scope_id,
            scopes=self._scopes,
            definitions=self._definitions,
            top_level_definition_ids=tuple(top_level_definition_ids),
        )

    def _new_scope(self, *, parent_scope_id: int | None) -> int:
        scope_id = self._next_scope_id
        self._next_scope_id += 1
        self._scopes[scope_id] = _Scope(
            scope_id=scope_id,
            parent_scope_id=parent_scope_id,
        )
        return scope_id

    def _register_definitions(
        self,
        nodes: list[_FunctionNode],
        *,
        defining_scope_id: int,
        is_top_level: bool,
    ) -> list[int]:
        definition_ids: list[int] = []
        for node in nodes:
            definition_id = self._next_definition_id
            self._next_definition_id += 1
            body_scope_id = self._new_scope(parent_scope_id=defining_scope_id)
            self._definitions[definition_id] = _Definition(
                definition_id=definition_id,
                node=node,
                defining_scope_id=defining_scope_id,
                body_scope_id=body_scope_id,
                is_top_level=is_top_level,
            )
            self._scopes[defining_scope_id].bindings.setdefault(
                node.name, []
            ).append(definition_id)
            definition_ids.append(definition_id)

        for definition_id in definition_ids:
            definition = self._definitions[definition_id]
            local_nodes = _collect_scope_function_definitions(
                definition.node.body
            )
            self._register_definitions(
                local_nodes,
                defining_scope_id=definition.body_scope_id,
                is_top_level=False,
            )
        return definition_ids


def _collect_scope_function_definitions(
    statements: list[ast.stmt],
) -> list[_FunctionNode]:
    visitor = _ScopeFunctionDefinitionVisitor()
    for statement in statements:
        visitor.visit(statement)
    return visitor.functions


class _ScopeFunctionDefinitionVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.functions: list[_FunctionNode] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.functions.append(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _analyze_definitions(
    graph: _DefinitionGraph,
    *,
    side_effect_calls: set[str],
) -> dict[int, _DefinitionAnalysis]:
    return {
        definition_id: _DefinitionAnalysis(
            calls=tuple(
                _resolve_calls(
                    _collect_calls(definition.node.body),
                    graph,
                    scope_id=definition.body_scope_id,
                    side_effect_calls=side_effect_calls,
                )
            ),
        )
        for definition_id, definition in graph.definitions.items()
    }


def _resolve_calls(
    calls: list[_CallOccurrence],
    graph: _DefinitionGraph,
    *,
    scope_id: int,
    side_effect_calls: set[str],
) -> list[_ResolvedCall]:
    resolved: list[_ResolvedCall] = []
    for call in calls:
        if call.name in side_effect_calls:
            resolved.append(
                _ResolvedCall(
                    line=call.line,
                    column=call.column,
                    effect_name=call.name,
                )
            )
            continue
        if not call.is_direct_name:
            continue
        targets = _resolve_binding(graph, scope_id=scope_id, name=call.name)
        if targets:
            resolved.append(
                _ResolvedCall(
                    line=call.line,
                    column=call.column,
                    target_definition_ids=targets,
                )
            )
    return resolved


def _resolve_binding(
    graph: _DefinitionGraph,
    *,
    scope_id: int,
    name: str,
) -> tuple[int, ...]:
    current_scope_id: int | None = scope_id
    while current_scope_id is not None:
        scope = graph.scopes[current_scope_id]
        if name in scope.bindings:
            return tuple(scope.bindings[name])
        current_scope_id = scope.parent_scope_id
    return ()


def _compute_definition_effect_summaries(
    analyses: dict[int, _DefinitionAnalysis],
) -> dict[int, tuple[str, ...]]:
    """Return memoized cap-two summaries for every definition, including pure ones."""
    adjacency = {
        definition_id: _ordered_unique(
            target_id
            for call in analysis.calls
            for target_id in call.target_definition_ids
        )
        for definition_id, analysis in analyses.items()
    }
    components = _strongly_connected_components(adjacency)
    component_by_definition = {
        definition_id: component_id
        for component_id, definition_ids in enumerate(components)
        for definition_id in definition_ids
    }
    memo: dict[int, tuple[str, ...]] = {}

    def summarize_component(component_id: int) -> tuple[str, ...]:
        if component_id in memo:
            return memo[component_id]

        terms: list[tuple[int, int, int, int, str | None, int | None]] = []
        for definition_id in components[component_id]:
            for call_index, call in enumerate(analyses[definition_id].calls):
                if call.effect_name is not None:
                    terms.append(
                        (
                            call.line,
                            call.column,
                            definition_id,
                            call_index,
                            call.effect_name,
                            None,
                        )
                    )
                for target_index, target_id in enumerate(
                    call.target_definition_ids
                ):
                    target_component_id = component_by_definition[target_id]
                    if target_component_id == component_id:
                        continue
                    terms.append(
                        (
                            call.line,
                            call.column,
                            definition_id,
                            call_index + target_index,
                            None,
                            target_component_id,
                        )
                    )

        effects: list[str] = []
        for _, _, _, _, effect_name, target_component_id in sorted(terms):
            if effect_name is not None:
                effects.append(effect_name)
            elif target_component_id is not None:
                _extend_effects(
                    effects,
                    summarize_component(target_component_id),
                    limit=_EFFECT_SUMMARY_LIMIT,
                )
            if len(effects) >= _EFFECT_SUMMARY_LIMIT:
                break

        memo[component_id] = tuple(effects[:_EFFECT_SUMMARY_LIMIT])
        return memo[component_id]

    return {
        definition_id: summarize_component(component_by_definition[definition_id])
        for definition_id in analyses
    }


def _strongly_connected_components(
    adjacency: dict[int, list[int]],
) -> list[tuple[int, ...]]:
    next_index = 0
    indices: dict[int, int] = {}
    low_links: dict[int, int] = {}
    stack: list[int] = []
    on_stack: set[int] = set()
    components: list[tuple[int, ...]] = []

    def connect(definition_id: int) -> None:
        nonlocal next_index
        indices[definition_id] = next_index
        low_links[definition_id] = next_index
        next_index += 1
        stack.append(definition_id)
        on_stack.add(definition_id)

        for target_id in adjacency[definition_id]:
            if target_id not in indices:
                connect(target_id)
                low_links[definition_id] = min(
                    low_links[definition_id], low_links[target_id]
                )
            elif target_id in on_stack:
                low_links[definition_id] = min(
                    low_links[definition_id], indices[target_id]
                )

        if low_links[definition_id] != indices[definition_id]:
            return
        component: list[int] = []
        while stack:
            member_id = stack.pop()
            on_stack.remove(member_id)
            component.append(member_id)
            if member_id == definition_id:
                break
        components.append(tuple(sorted(component)))

    for definition_id in sorted(adjacency):
        if definition_id not in indices:
            connect(definition_id)
    return components


def _helper_violations(
    graph: _DefinitionGraph,
    summaries: dict[int, tuple[str, ...]],
    *,
    regions: list[CodeRegion],
    groups: list[CodeRegionGroup],
) -> list[ProgramContractViolation]:
    violations: list[ProgramContractViolation] = []
    for definition_id in graph.top_level_definition_ids:
        effects = summaries[definition_id]
        if not effects:
            continue
        node = graph.definitions[definition_id].node
        start_line, end_line = _node_span(node)
        violations.append(
            _build_violation(
                code="effectful_helper",
                message=f"Helper '{node.name}' can execute a robot side effect",
                start_line=start_line,
                end_line=end_line,
                regions=regions,
                groups=groups,
                side_effects=effects,
                helper_name=node.name,
            )
        )
    return violations


def _control_flow_violations(
    module: ast.Module,
    graph: _DefinitionGraph,
    analyses: dict[int, _DefinitionAnalysis],
    summaries: dict[int, tuple[str, ...]],
    *,
    regions: list[CodeRegion],
    groups: list[CodeRegionGroup],
    side_effect_calls: set[str],
) -> list[ProgramContractViolation]:
    cases: list[tuple[ast.AST, int]] = [
        (node, graph.module_scope_id)
        for node in _collect_control_flow_nodes(module.body)
    ]
    for definition_id in _reachable_definition_ids(
        graph.top_level_definition_ids,
        analyses,
    ):
        definition = graph.definitions[definition_id]
        cases.extend(
            (node, definition.body_scope_id)
            for node in _collect_control_flow_nodes(definition.node.body)
        )

    violations: list[ProgramContractViolation] = []
    for node, scope_id in cases:
        calls = _resolve_calls(
            _collect_calls([node]),
            graph,
            scope_id=scope_id,
            side_effect_calls=side_effect_calls,
        )
        effects = _materialize_effects(
            calls,
            summaries,
            limit=_EFFECT_SUMMARY_LIMIT,
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
    return violations


def _reachable_definition_ids(
    roots: tuple[int, ...],
    analyses: dict[int, _DefinitionAnalysis],
) -> list[int]:
    reachable = set(roots)
    pending = deque(roots)
    while pending:
        definition_id = pending.popleft()
        for call in analyses[definition_id].calls:
            for target_id in call.target_definition_ids:
                if target_id in reachable:
                    continue
                reachable.add(target_id)
                pending.append(target_id)
    return sorted(reachable)


def _group_violations(
    module_calls: list[_ResolvedCall],
    summaries: dict[int, tuple[str, ...]],
    *,
    regions: list[CodeRegion],
    groups: list[CodeRegionGroup],
) -> list[ProgramContractViolation]:
    violations: list[ProgramContractViolation] = []
    for group in groups:
        group_calls = [
            call
            for call in module_calls
            if group.start_line <= call.line <= group.end_line
        ]
        effects = _materialize_effects(group_calls, summaries)
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
    return violations


def _effectful_executable_unit_ids(
    module_calls: list[_ResolvedCall],
    summaries: dict[int, tuple[str, ...]],
    *,
    regions: list[CodeRegion],
    groups: list[CodeRegionGroup],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    def unit_has_effect(start_line: int, end_line: int) -> bool:
        unit_calls = [
            call for call in module_calls if start_line <= call.line <= end_line
        ]
        return bool(_materialize_effects(unit_calls, summaries, limit=1))

    return (
        tuple(
            region.region_id
            for region in regions
            if unit_has_effect(region.start_line, region.end_line)
        ),
        tuple(
            group.group_id
            for group in groups
            if unit_has_effect(group.start_line, group.end_line)
        ),
    )


def _materialize_effects(
    calls: list[_ResolvedCall] | tuple[_ResolvedCall, ...],
    summaries: dict[int, tuple[str, ...]],
    *,
    limit: int | None = None,
) -> tuple[str, ...]:
    effects: list[str] = []
    for call in calls:
        if call.effect_name is not None:
            effects.append(call.effect_name)
        else:
            for target_id in call.target_definition_ids:
                _extend_effects(effects, summaries[target_id], limit=limit)
                if limit is not None and len(effects) >= limit:
                    break
        if limit is not None and len(effects) >= limit:
            break
    if limit is not None:
        return tuple(effects[:limit])
    return tuple(effects)


def _extend_effects(
    destination: list[str],
    values: tuple[str, ...],
    *,
    limit: int | None,
) -> None:
    if limit is None:
        destination.extend(values)
        return
    remaining = limit - len(destination)
    if remaining > 0:
        destination.extend(values[:remaining])


def _collect_calls(statements: list[ast.stmt]) -> list[_CallOccurrence]:
    visitor = _ExecutableCallVisitor()
    for statement in statements:
        visitor.visit(statement)
    return visitor.calls


class _ExecutableCallVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: list[_CallOccurrence] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_definition_time_expressions(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_definition_time_expressions(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in sorted(
            [*node.decorator_list, *node.bases, *(item.value for item in node.keywords)],
            key=_source_position,
        ):
            self.visit(expression)
        for statement in node.body:
            self.visit(statement)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for expression in sorted(
            [*node.args.defaults, *(item for item in node.args.kw_defaults if item)],
            key=_source_position,
        ):
            self.visit(expression)

    def visit_Call(self, node: ast.Call) -> None:
        self.visit(node.func)
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            self.visit(keyword.value)

        name = _callable_name(node.func)
        if name is not None:
            self.calls.append(
                _CallOccurrence(
                    name=name,
                    is_direct_name=isinstance(node.func, ast.Name),
                    line=node.lineno,
                    column=node.col_offset,
                )
            )

    def _visit_definition_time_expressions(self, node: _FunctionNode) -> None:
        expressions: list[ast.expr] = [
            *node.decorator_list,
            *node.args.defaults,
            *(item for item in node.args.kw_defaults if item is not None),
        ]
        arguments = [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]
        if node.args.vararg is not None:
            arguments.append(node.args.vararg)
        if node.args.kwarg is not None:
            arguments.append(node.args.kwarg)
        expressions.extend(
            argument.annotation
            for argument in arguments
            if argument.annotation is not None
        )
        if node.returns is not None:
            expressions.append(node.returns)

        for expression in sorted(expressions, key=_source_position):
            self.visit(expression)


def _source_position(node: ast.AST) -> tuple[int, int]:
    return int(getattr(node, "lineno", 0)), int(getattr(node, "col_offset", 0))


def _collect_control_flow_nodes(statements: list[ast.stmt]) -> list[ast.AST]:
    visitor = _ControlFlowVisitor()
    for statement in statements:
        visitor.visit(statement)
    return visitor.nodes


_CONTROL_FLOW_TYPES: tuple[type[ast.AST], ...] = (
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    # Generator bodies are lazy, but the safety contract conservatively rejects
    # effectful generator expressions before a later consumer can iterate them.
    ast.GeneratorExp,
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
        for statement in node.body:
            self.visit(statement)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, _CONTROL_FLOW_TYPES):
            self.nodes.append(node)
        super().generic_visit(node)


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


def _ordered_unique(values: Iterable[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
