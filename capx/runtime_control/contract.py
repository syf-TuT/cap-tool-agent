from __future__ import annotations

import ast
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
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
_CallableNode = _FunctionNode | ast.Lambda
_EFFECT_SUMMARY_LIMIT = 2
_DYNAMIC_EFFECT_MARKER = "dynamic_effect_call"

STRICT_CAPSULE_SAFE_BUILTINS: frozenset[str] = frozenset(
    {
        "RuntimeError",
        "ValueError",
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "print",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }
)
STRICT_CAPSULE_MAX_HELPERS = 256
STRICT_CAPSULE_MAX_STATIC_ITERATIONS = 10_000
STRICT_CAPSULE_MAX_ITERATIONS = STRICT_CAPSULE_MAX_STATIC_ITERATIONS

_STRICT_CAPSULE_FORBIDDEN_CALLABLES = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "dir",
        "eval",
        "exec",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "object",
        "open",
        "setattr",
        "super",
        "type",
        "vars",
    }
)
_STRICT_CAPSULE_SENSITIVE_NAMES = frozenset(
    {
        "APIS",
        "__builtins__",
        "builtins",
        "ctypes",
        "env",
        "frame",
        "gc",
        "importlib",
        "inspect",
        "marshal",
        "os",
        "pathlib",
        "pickle",
        "sim",
        "simulator",
        "socket",
        "subprocess",
        "sys",
        "types",
    }
)
_STRICT_CAPSULE_EXCEPTION_CLASSES = frozenset({"RuntimeError", "ValueError"})
_STRICT_CAPSULE_KEY_CALLBACK_CALLS = frozenset({"max", "min", "sorted"})
_STRICT_CAPSULE_ITERABLE_CONSUMERS = frozenset(
    {"all", "any", "dict", "list", "max", "min", "set", "sorted", "sum", "tuple"}
)
_STRICT_CAPSULE_BOUNDED_ITERABLE_WRAPPERS = frozenset(
    {"dict", "enumerate", "list", "reversed", "set", "tuple"}
)

_STRICT_CAPSULE_ALLOWED_NODES = (
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.NamedExpr,
    ast.If,
    ast.For,
    ast.FunctionDef,
    ast.Return,
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.Raise,
    ast.Constant,
    ast.Name,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.Starred,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.keyword,
    ast.JoinedStr,
    ast.FormattedValue,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.arguments,
    ast.arg,
)
_STRICT_CAPSULE_ALLOWED_STATEMENTS = (
    ast.Expr,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.If,
    ast.For,
    ast.FunctionDef,
    ast.Return,
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.Raise,
)
_STRICT_CAPSULE_ANNOTATION_NAMES = frozenset(
    {"bool", "dict", "float", "int", "list", "set", "str", "tuple"}
)
_STRICT_CAPSULE_ALLOWED_NODE_BASES = (
    ast.operator,
    ast.unaryop,
    ast.boolop,
    ast.cmpop,
    ast.expr_context,
)


@dataclass(frozen=True)
class _CallOccurrence:
    name: str
    is_direct_name: bool
    line: int
    end_line: int
    column: int
    dynamic_code: str | None = None
    class_method_name: str | None = None
    class_method_span: tuple[int, int] | None = None
    lambda_span: tuple[int, int] | None = None
    uncertain_assignment: bool = False


@dataclass(frozen=True)
class _AliasBinding:
    line: int
    column: int
    target_name: str | None = None
    target_definition_id: int | None = None
    is_dynamic: bool = False


@dataclass
class _Scope:
    scope_id: int
    parent_scope_id: int | None
    bindings: dict[str, list[int]] = field(default_factory=dict)
    aliases: dict[str, list[_AliasBinding]] = field(default_factory=dict)
    dynamic_callable_names: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _Definition:
    definition_id: int
    node: _CallableNode
    name: str
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
    end_line: int
    column: int
    effect_name: str | None = None
    target_definition_ids: tuple[int, ...] = ()
    via_alias: bool = False
    dynamic_code: str | None = None
    attribute_name: str | None = None
    class_method_name: str | None = None
    class_method_span: tuple[int, int] | None = None
    lambda_span: tuple[int, int] | None = None


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
    definition_summaries = _compute_definition_effect_summaries(definition_analyses)
    module_calls = _resolve_calls(
        _collect_calls(module.body),
        graph,
        scope_id=graph.module_scope_id,
        side_effect_calls=side_effect_calls,
    )
    module_calls = _mark_effectful_class_attribute_calls(
        module_calls,
        definition_summaries,
    )

    violations = _helper_violations(
        graph,
        definition_summaries,
        regions=regions,
        groups=groups,
    )
    violations.extend(
        _call_contract_violations(
            graph,
            definition_analyses,
            definition_summaries,
            module_calls=module_calls,
            regions=regions,
            groups=groups,
        )
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


def analyze_capsule_strict_subset(
    source: str,
    regions: list[CodeRegion],
    groups: list[CodeRegionGroup],
    *,
    public_api_calls: set[str],
    side_effect_calls: set[str],
    safe_builtin_calls: set[str] | None = None,
) -> list[ProgramContractViolation]:
    """Validate the fail-closed Python subset used by non-privileged Capsules."""
    module = ast.parse(source)
    direct_helpers = tuple(
        statement for statement in module.body if isinstance(statement, ast.FunctionDef)
    )
    preflight_violations = _preflight_capsule_strict_module(
        module,
        regions=regions,
        groups=groups,
    )
    if preflight_violations:
        return preflight_violations
    safe_calls = (
        set(STRICT_CAPSULE_SAFE_BUILTINS) if safe_builtin_calls is None else set(safe_builtin_calls)
    )
    public_calls = set(public_api_calls) | set(side_effect_calls)
    helper_names, pure_helper_names, helper_issues = _prove_strict_helper_purity(
        direct_helpers,
        public_api_calls=public_calls,
        side_effect_calls=set(side_effect_calls),
        safe_builtin_calls=safe_calls,
    )
    visitor = _StrictCapsuleSubsetVisitor(
        regions=regions,
        groups=groups,
        public_api_calls=public_calls,
        side_effect_calls=set(side_effect_calls),
        safe_builtin_calls=safe_calls,
        helper_names=helper_names,
        pure_helper_names=pure_helper_names,
        direct_helper_node_ids={id(helper) for helper in direct_helpers},
    )
    visitor.reject_misplaced_helpers(module)
    for helper, reason in helper_issues:
        visitor.add_helper_violation(helper, reason)
    visitor.visit(module)
    visitor.violations.extend(
        _strict_static_compute_violations(
            module,
            direct_helpers,
            pure_helper_names=pure_helper_names,
            regions=regions,
            groups=groups,
        )
    )
    return sorted(
        set(visitor.violations),
        key=lambda violation: (
            violation.start_line,
            violation.end_line,
            violation.code,
            violation.message,
            violation.helper_name or "",
        ),
    )


def preflight_capsule_strict_source(source: str) -> list[ProgramContractViolation]:
    """Reject structurally excessive strict Capsule source before segmentation."""
    module = ast.parse(source)
    return _preflight_capsule_strict_module(
        module,
        regions=[],
        groups=[],
    )


def _preflight_capsule_strict_module(
    module: ast.Module,
    *,
    regions: list[CodeRegion],
    groups: list[CodeRegionGroup],
) -> list[ProgramContractViolation]:
    all_helpers = sorted(
        (
            node
            for node in ast.walk(module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        key=lambda node: (
            int(getattr(node, "lineno", 1)),
            int(getattr(node, "col_offset", 0)),
        ),
    )
    if len(all_helpers) <= STRICT_CAPSULE_MAX_HELPERS:
        return []
    start_line, _ = _node_span(all_helpers[0])
    _, end_line = _node_span(all_helpers[-1])
    return [
        _build_violation(
            code="strict_subset_violation",
            message=(
                "Strict Capsule subset violation: helper limit exceeded "
                f"({len(all_helpers)} > {STRICT_CAPSULE_MAX_HELPERS})"
            ),
            start_line=start_line,
            end_line=end_line,
            regions=regions,
            groups=groups,
            side_effects=(),
        )
    ]


def _prove_strict_helper_purity(
    direct_helpers: tuple[ast.FunctionDef, ...],
    *,
    public_api_calls: set[str],
    side_effect_calls: set[str],
    safe_builtin_calls: set[str],
) -> tuple[set[str], set[str], list[tuple[ast.FunctionDef, str]]]:
    helpers_by_name: dict[str, list[ast.FunctionDef]] = {}
    for helper in direct_helpers:
        helpers_by_name.setdefault(helper.name, []).append(helper)
    helper_names = set(helpers_by_name)

    dependencies: dict[str, set[str]] = {helper_name: set() for helper_name in helper_names}
    unsafe_reasons: dict[str, str] = {}
    for helper_name, definitions in helpers_by_name.items():
        if len(definitions) > 1:
            unsafe_reasons[helper_name] = f"helper name '{helper_name}' is defined more than once"
            continue

        helper = definitions[0]
        calls = _collect_strict_helper_calls(helper.body)
        for call in calls:
            if not isinstance(call.func, ast.Name):
                unsafe_reasons.setdefault(
                    helper_name,
                    f"helper '{helper_name}' contains a non-direct call and is not provably pure",
                )
                continue
            called_name = call.func.id
            if called_name in side_effect_calls:
                unsafe_reasons.setdefault(
                    helper_name, f"helper '{helper_name}' calls robot side effect '{called_name}'"
                )
                continue
            if called_name in helper_names:
                dependencies[helper_name].add(called_name)
                continue
            if called_name in public_api_calls or called_name in safe_builtin_calls:
                continue
            unsafe_reasons.setdefault(
                helper_name,
                f"helper '{helper_name}' calls unknown callable '{called_name}' "
                "and is not provably pure",
            )

    recursive_helpers = _strict_recursive_helper_names(dependencies)
    for helper_name in recursive_helpers:
        unsafe_reasons.setdefault(
            helper_name,
            f"recursive helper '{helper_name}' is unsupported",
        )

    changed = True
    while changed:
        changed = False
        for helper_name, helper_dependencies in dependencies.items():
            if helper_name in unsafe_reasons:
                continue
            unsafe_dependency = next(
                (
                    dependency
                    for dependency in sorted(helper_dependencies)
                    if dependency in unsafe_reasons
                ),
                None,
            )
            if unsafe_dependency is None:
                continue
            unsafe_reasons[helper_name] = (
                f"helper '{helper_name}' depends on unsafe helper '{unsafe_dependency}'"
            )
            changed = True

    pure_helper_names = helper_names - set(unsafe_reasons)
    issues = [
        (helper, unsafe_reasons[helper_name])
        for helper_name, helpers in helpers_by_name.items()
        if helper_name in unsafe_reasons
        for helper in helpers
    ]
    for helper_name in sorted(recursive_helpers):
        recursive_reason = f"recursive helper '{helper_name}' is unsupported"
        if unsafe_reasons[helper_name] == recursive_reason:
            continue
        issues.extend((helper, recursive_reason) for helper in helpers_by_name[helper_name])
    return helper_names, pure_helper_names, issues


def _strict_recursive_helper_names(
    dependencies: dict[str, set[str]],
) -> set[str]:
    ordered_names = sorted(dependencies)
    index_by_name = {helper_name: index for index, helper_name in enumerate(ordered_names)}
    adjacency = {
        index_by_name[helper_name]: [
            index_by_name[dependency] for dependency in sorted(helper_dependencies)
        ]
        for helper_name, helper_dependencies in dependencies.items()
    }
    recursive: set[str] = set()
    for component in _strongly_connected_components(adjacency):
        component_names = {ordered_names[index] for index in component}
        if len(component_names) > 1:
            recursive.update(component_names)
            continue
        helper_name = next(iter(component_names))
        if helper_name in dependencies[helper_name]:
            recursive.add(helper_name)
    return recursive


def _strict_static_compute_violations(
    module: ast.Module,
    direct_helpers: tuple[ast.FunctionDef, ...],
    *,
    pure_helper_names: set[str],
    regions: list[CodeRegion],
    groups: list[CodeRegionGroup],
) -> list[ProgramContractViolation]:
    helpers_by_name = {
        helper.name: helper for helper in direct_helpers if helper.name in pure_helper_names
    }
    estimator = _StrictStaticCostEstimator(helpers_by_name)
    violations: list[ProgramContractViolation] = []
    for helper in direct_helpers:
        if helper.name not in pure_helper_names:
            continue
        cost = estimator.helper_cost(helper.name)
        if cost <= STRICT_CAPSULE_MAX_STATIC_ITERATIONS:
            continue
        start_line, end_line = _node_span(helper)
        violations.append(
            _build_violation(
                code="strict_subset_violation",
                message=(
                    "Strict Capsule subset violation: helper "
                    f"'{helper.name}' exceeds static compute budget "
                    f"{STRICT_CAPSULE_MAX_STATIC_ITERATIONS}"
                ),
                start_line=start_line,
                end_line=end_line,
                regions=regions,
                groups=groups,
                side_effects=(),
                helper_name=helper.name,
            )
        )

    module_cost = estimator.block_cost(module.body, include_definitions=False)
    if module_cost > STRICT_CAPSULE_MAX_STATIC_ITERATIONS and module.body:
        start_line, _ = _node_span(module.body[0])
        _, end_line = _node_span(module.body[-1])
        violations.append(
            _build_violation(
                code="strict_subset_violation",
                message=(
                    "Strict Capsule subset violation: module exceeds static "
                    f"compute budget {STRICT_CAPSULE_MAX_STATIC_ITERATIONS}"
                ),
                start_line=start_line,
                end_line=end_line,
                regions=regions,
                groups=groups,
                side_effects=(),
            )
        )
    return violations


class _StrictStaticCostEstimator:
    """Conservatively estimate bounded work, saturating at budget + one."""

    def __init__(self, helpers_by_name: dict[str, ast.FunctionDef]) -> None:
        self.helpers_by_name = helpers_by_name
        self._helper_costs: dict[str, int] = {}
        self._active_helpers: set[str] = set()

    def helper_cost(self, helper_name: str) -> int:
        if helper_name in self._helper_costs:
            return self._helper_costs[helper_name]
        if helper_name in self._active_helpers:
            return STRICT_CAPSULE_MAX_STATIC_ITERATIONS + 1
        helper = self.helpers_by_name.get(helper_name)
        if helper is None:
            return 1
        self._active_helpers.add(helper_name)
        cost = self.block_cost(helper.body)
        self._active_helpers.remove(helper_name)
        self._helper_costs[helper_name] = cost
        return cost

    def block_cost(
        self,
        statements: list[ast.stmt],
        *,
        include_definitions: bool = True,
    ) -> int:
        cost = 0
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                statement_cost = 0 if not include_definitions else 1
            else:
                statement_cost = self.statement_cost(statement)
            cost = _add_static_cost(cost, statement_cost)
        return cost

    def statement_cost(self, node: ast.stmt) -> int:
        if isinstance(node, ast.For):
            bound = _static_iteration_bound(node.iter)
            if bound is None:
                return STRICT_CAPSULE_MAX_STATIC_ITERATIONS + 1
            iterable_cost = self.expression_cost(node.iter)
            body_cost = max(1, self.block_cost(node.body))
            per_iteration_cost = _add_static_cost(
                self.assignment_target_cost(node.target),
                body_cost,
            )
            loop_cost = _multiply_static_cost(bound, per_iteration_cost)
            cost = _add_static_cost(iterable_cost, loop_cost)
            return _add_static_cost(cost, self.block_cost(node.orelse))
        if isinstance(node, (ast.While, ast.AsyncFor)):
            return STRICT_CAPSULE_MAX_STATIC_ITERATIONS + 1
        if isinstance(node, ast.If):
            branch_cost = max(
                self.block_cost(node.body),
                self.block_cost(node.orelse),
            )
            return _add_static_cost(self.expression_cost(node.test), branch_cost)
        if isinstance(node, ast.Assign):
            target_cost = 0
            for target in node.targets:
                target_cost = _add_static_cost(
                    target_cost,
                    self.assignment_target_cost(target),
                )
            return max(
                1,
                _add_static_cost(target_cost, self.expression_cost(node.value)),
            )
        if isinstance(node, ast.AnnAssign):
            return max(
                1,
                _add_static_cost(
                    self.assignment_target_cost(node.target),
                    self.expression_cost(node.value),
                ),
            )
        if isinstance(node, ast.AugAssign):
            return max(
                1,
                _add_static_cost(
                    self.assignment_target_cost(node.target),
                    self.expression_cost(node.value),
                ),
            )
        if isinstance(node, ast.Expr):
            return max(1, self.expression_cost(node.value))
        if isinstance(node, ast.Return):
            return max(
                1,
                self.expression_cost(node.value) if node.value is not None else 0,
            )
        if isinstance(node, ast.Raise):
            return max(
                1,
                self.expression_cost(node.exc) if node.exc is not None else 0,
            )
        if isinstance(node, (ast.Pass, ast.Break, ast.Continue)):
            return 1
        return STRICT_CAPSULE_MAX_STATIC_ITERATIONS + 1

    def expression_cost(self, node: ast.expr | None) -> int:
        if node is None or isinstance(node, (ast.Constant, ast.Name)):
            return 0
        if isinstance(node, ast.Call):
            argument_cost = self._call_argument_cost(node)
            if isinstance(node.func, ast.Name):
                callable_name = node.func.id
                if callable_name in self.helpers_by_name:
                    return _add_static_cost(
                        argument_cost,
                        self.helper_cost(callable_name),
                    )
                if _strict_call_consumes_iterable(node, callable_name):
                    bound = _static_iteration_bound(node.args[0])
                    if bound is None:
                        return STRICT_CAPSULE_MAX_STATIC_ITERATIONS + 1
                    return _add_static_cost(argument_cost, bound)
                if callable_name == "range":
                    return argument_cost
            return _add_static_cost(argument_cost, 1)
        if isinstance(
            node,
            (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp),
        ):
            return self._comprehension_cost(node)
        if isinstance(node, ast.NamedExpr):
            if not isinstance(node.target, ast.Name):
                return STRICT_CAPSULE_MAX_STATIC_ITERATIONS + 1
            return self.expression_cost(node.value)
        cost = 0
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                cost = _add_static_cost(cost, self.expression_cost(child))
        return cost

    def assignment_target_cost(self, node: ast.expr) -> int:
        if isinstance(node, ast.Name):
            return 0
        if isinstance(node, (ast.Tuple, ast.List)):
            cost = 0
            for element in node.elts:
                cost = _add_static_cost(
                    cost,
                    self.assignment_target_cost(element),
                )
            return cost
        if isinstance(node, ast.Starred):
            return self.assignment_target_cost(node.value)
        if isinstance(node, ast.Subscript):
            return max(
                1,
                _add_static_cost(
                    self.expression_cost(node.value),
                    self.expression_cost(node.slice),
                ),
            )
        if isinstance(node, ast.Attribute):
            return max(1, self.expression_cost(node.value))
        return STRICT_CAPSULE_MAX_STATIC_ITERATIONS + 1

    def _call_argument_cost(self, node: ast.Call) -> int:
        cost = 0
        for argument in node.args:
            cost = _add_static_cost(cost, self.expression_cost(argument))
        for keyword in node.keywords:
            cost = _add_static_cost(cost, self.expression_cost(keyword.value))
        return cost

    def _comprehension_cost(
        self,
        node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp,
    ) -> int:
        product = 1
        iterable_construction_cost = 0
        target_assignment_cost = 0
        condition_cost = 0
        for generator in node.generators:
            iterable_construction_cost = _add_static_cost(
                iterable_construction_cost,
                _multiply_static_cost(
                    product,
                    self.expression_cost(generator.iter),
                ),
            )
            bound = _static_iteration_bound(generator.iter)
            if bound is None:
                return STRICT_CAPSULE_MAX_STATIC_ITERATIONS + 1
            product = _multiply_static_cost(product, bound)
            target_assignment_cost = _add_static_cost(
                target_assignment_cost,
                _multiply_static_cost(
                    product,
                    self.assignment_target_cost(generator.target),
                ),
            )
            for condition in generator.ifs:
                condition_cost = _add_static_cost(
                    condition_cost,
                    self.expression_cost(condition),
                )
        if isinstance(node, ast.DictComp):
            result_cost = _add_static_cost(
                self.expression_cost(node.key),
                self.expression_cost(node.value),
            )
        else:
            result_cost = self.expression_cost(node.elt)
        per_iteration = max(1, _add_static_cost(condition_cost, result_cost))
        result_iterations_cost = _multiply_static_cost(product, per_iteration)
        cost = _add_static_cost(
            iterable_construction_cost,
            target_assignment_cost,
        )
        return _add_static_cost(
            cost,
            result_iterations_cost,
        )


def _add_static_cost(first: int, second: int) -> int:
    return min(
        STRICT_CAPSULE_MAX_STATIC_ITERATIONS + 1,
        first + second,
    )


def _multiply_static_cost(first: int, second: int) -> int:
    if first == 0 or second == 0:
        return 0
    limit = STRICT_CAPSULE_MAX_STATIC_ITERATIONS + 1
    if first > STRICT_CAPSULE_MAX_STATIC_ITERATIONS // second:
        return limit
    return min(limit, first * second)


def _collect_strict_helper_calls(statements: list[ast.stmt]) -> list[ast.Call]:
    visitor = _StrictHelperCallVisitor()
    for statement in statements:
        visitor.visit(statement)
    return sorted(
        visitor.calls,
        key=lambda call: (
            int(getattr(call, "lineno", 1)),
            int(getattr(call, "col_offset", 0)),
        ),
    )


class _StrictHelperCallVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: list[ast.Call] = []

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


class _StrictCapsuleSubsetVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        regions: list[CodeRegion],
        groups: list[CodeRegionGroup],
        public_api_calls: set[str],
        side_effect_calls: set[str],
        safe_builtin_calls: set[str],
        helper_names: set[str],
        pure_helper_names: set[str],
        direct_helper_node_ids: set[int],
    ) -> None:
        self.regions = regions
        self.groups = groups
        self.public_api_calls = public_api_calls
        self.side_effect_calls = side_effect_calls
        self.safe_builtin_calls = safe_builtin_calls
        self.helper_names = helper_names
        self.pure_helper_names = pure_helper_names
        self.direct_helper_node_ids = direct_helper_node_ids
        self.allowed_calls = public_api_calls | safe_builtin_calls | pure_helper_names
        self.protected_callable_names = (
            self.allowed_calls | helper_names | _STRICT_CAPSULE_FORBIDDEN_CALLABLES
        )
        self.violations: list[ProgramContractViolation] = []
        self._helper_stack: list[str] = []
        self._iteration_products: list[int] = [1]
        self._loop_depth = 0

    def _emit(
        self,
        node: ast.AST,
        reason: str,
        *,
        helper_name: str | None = None,
    ) -> None:
        start_line, end_line = _node_span(node)
        side_effects = tuple(
            dict.fromkeys(
                child.id
                for child in ast.walk(node)
                if isinstance(child, ast.Name) and child.id in self.side_effect_calls
            )
        )
        self.violations.append(
            _build_violation(
                code="strict_subset_violation",
                message=f"Strict Capsule subset violation: {reason}",
                start_line=start_line,
                end_line=end_line,
                regions=self.regions,
                groups=self.groups,
                side_effects=side_effects,
                helper_name=(
                    helper_name or (self._helper_stack[-1] if self._helper_stack else None)
                ),
            )
        )

    def reject_misplaced_helpers(self, module: ast.Module) -> None:
        for node in ast.walk(module):
            if isinstance(node, ast.FunctionDef) and (id(node) not in self.direct_helper_node_ids):
                self._emit(
                    node,
                    "nested or conditional helper definitions must be direct module statements",
                    helper_name=node.name,
                )

    def add_helper_violation(self, helper: ast.FunctionDef, reason: str) -> None:
        self._emit(helper, reason, helper_name=helper.name)

    def _identifier_violation(self, name: str) -> str | None:
        if name.startswith("_"):
            return f"private name '{name}' is not available"
        if name in _STRICT_CAPSULE_SENSITIVE_NAMES:
            return f"sensitive runtime name '{name}' is not available"
        return None

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, ast.stmt) and not isinstance(
            node,
            _STRICT_CAPSULE_ALLOWED_STATEMENTS,
        ):
            self._emit(node, f"syntax '{type(node).__name__}' is not allowed")
            return
        if not isinstance(
            node,
            _STRICT_CAPSULE_ALLOWED_NODES + _STRICT_CAPSULE_ALLOWED_NODE_BASES,
        ):
            self._emit(node, f"syntax '{type(node).__name__}' is not allowed")
            return
        super().generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if id(node) not in self.direct_helper_node_ids:
            self._emit(
                node,
                "nested or conditional helper definitions must be direct module statements",
                helper_name=node.name,
            )
            return

        identifier_reason = self._identifier_violation(node.name)
        if identifier_reason is not None:
            self._emit(node, identifier_reason)
        elif node.name in (
            self.public_api_calls | self.safe_builtin_calls | _STRICT_CAPSULE_FORBIDDEN_CALLABLES
        ):
            self._emit(node, f"callable '{node.name}' cannot be redefined")

        if node.decorator_list:
            self._emit(node, "function decorators are not allowed")
        if _function_has_annotations(node):
            self._emit(node, "function annotations are not allowed")
        if any(
            not _is_literal_expression(default)
            for default in (*node.args.defaults, *node.args.kw_defaults)
            if default is not None
        ):
            self._emit(node, "function defaults must be literal values")

        for argument in _function_arguments(node.args):
            reason = self._identifier_violation(argument.arg)
            if reason is not None:
                self._emit(argument, reason)
            elif argument.arg in self.protected_callable_names:
                self._emit(
                    argument,
                    f"parameter cannot rebind callable '{argument.arg}'",
                )

        self._helper_stack.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self._helper_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._emit(node, "syntax 'AsyncFunctionDef' is not allowed")
        for statement in node.body:
            self.visit(statement)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._emit(node, "syntax 'ClassDef' is not allowed")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._emit(node, "syntax 'Lambda' is not allowed")

    def visit_Name(self, node: ast.Name) -> None:
        reason = self._identifier_violation(node.id)
        if reason is not None:
            self._emit(node, reason)
            return
        if isinstance(node.ctx, ast.Store) and node.id in self.protected_callable_names:
            self._emit(node, f"assignment cannot rebind callable '{node.id}'")
        elif isinstance(node.ctx, ast.Load) and node.id in self.protected_callable_names:
            self._emit(
                node,
                f"callable '{node.id}' may only be used as a direct call",
            )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        reason = self._identifier_violation(node.attr)
        if reason is not None:
            self._emit(node, reason)
            return
        if not isinstance(node.ctx, ast.Load):
            self._emit(node, "data attributes are read-only")
            return
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            self._emit(
                node,
                "calls must target a direct public API, safe builtin, or top-level helper name",
            )
        else:
            name = node.func.id
            reason = self._identifier_violation(name)
            if reason is not None:
                self._emit(node, reason)
            elif name in _STRICT_CAPSULE_FORBIDDEN_CALLABLES:
                self._emit(node, f"call to forbidden callable '{name}' is not allowed")
            elif name in self.helper_names and name not in self.pure_helper_names:
                self._emit(node, f"call to unsafe helper '{name}' is not allowed")
            elif name not in self.allowed_calls:
                self._emit(node, f"call to unknown callable '{name}' is not allowed")

            if name in _STRICT_CAPSULE_KEY_CALLBACK_CALLS:
                for keyword in node.keywords:
                    if keyword.arg == "key" and not _is_none_literal(keyword.value):
                        self._emit(
                            keyword,
                            f"callback argument 'key' is not allowed for '{name}'",
                        )
            if _strict_call_consumes_iterable(node, name):
                bound = _static_iteration_bound(node.args[0])
                if bound is None or bound > STRICT_CAPSULE_MAX_STATIC_ITERATIONS:
                    self._emit(
                        node.args[0],
                        f"iterable consumer '{name}' requires a statically bounded "
                        "input of at most "
                        f"{STRICT_CAPSULE_MAX_STATIC_ITERATIONS} items",
                    )

        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            self.visit(keyword)

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg is None:
            self._emit(node, "dynamic keyword arguments are not allowed")
        else:
            reason = self._identifier_violation(node.arg)
            if reason is not None:
                self._emit(node, reason)
        self.visit(node.value)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self._visit_exception_type(node.type)
        if node.name is not None:
            reason = self._identifier_violation(node.name)
            if reason is not None:
                self._emit(node, reason)
            elif node.name in self.protected_callable_names:
                self._emit(node, f"exception binding cannot rebind callable '{node.name}'")
        for statement in node.body:
            self.visit(statement)

    def visit_Raise(self, node: ast.Raise) -> None:
        is_safe_exception_call = (
            isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id in _STRICT_CAPSULE_EXCEPTION_CLASSES
        )
        if not is_safe_exception_call or node.cause is not None:
            self._emit(
                node,
                "raise must directly construct ValueError or RuntimeError without 'from'",
            )
        if node.exc is not None:
            self.visit(node.exc)
        if node.cause is not None:
            self.visit(node.cause)

    def visit_Try(self, node: ast.Try) -> None:
        self._emit(node, "syntax 'Try' is not allowed")

    def visit_Assert(self, node: ast.Assert) -> None:
        self._emit(node, "syntax 'Assert' is not allowed")

    def visit_While(self, node: ast.While) -> None:
        self._emit(
            node,
            "bounded control flow does not allow while loops",
        )
        self.visit(node.test)
        for statement in (*node.body, *node.orelse):
            self.visit(statement)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.target)
        self.visit(node.iter)
        product = self._bounded_iteration_product(node.iter)
        if product is not None:
            self._iteration_products.append(product)
        self._loop_depth += 1
        for statement in node.body:
            self.visit(statement)
        self._loop_depth -= 1
        if product is not None:
            self._iteration_products.pop()
        for statement in node.orelse:
            self.visit(statement)

    def visit_Break(self, node: ast.Break) -> None:
        if self._loop_depth == 0:
            self._emit(node, "break is not allowed outside a loop")

    def visit_Continue(self, node: ast.Continue) -> None:
        if self._loop_depth == 0:
            self._emit(node, "continue is not allowed outside a loop")

    def visit_Return(self, node: ast.Return) -> None:
        if not self._helper_stack:
            self._emit(node, "return is not allowed outside a helper")
        if node.value is not None:
            self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.target)
        if not (
            isinstance(node.annotation, ast.Name)
            and node.annotation.id in _STRICT_CAPSULE_ANNOTATION_NAMES
        ):
            self._emit(node.annotation, "annotation type is not allowed")
        if node.value is not None:
            self.visit(node.value)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._emit(
            node,
            "bounded control flow does not allow asynchronous for loops",
        )
        self.visit(node.target)
        self.visit(node.iter)
        for statement in (*node.body, *node.orelse):
            self.visit(statement)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension_expression(node.generators, (node.elt,))

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension_expression(node.generators, (node.elt,))

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension_expression(
            node.generators,
            (node.key, node.value),
        )

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension_expression(node.generators, (node.elt,))

    def _visit_comprehension_expression(
        self,
        generators: list[ast.comprehension],
        result_expressions: tuple[ast.expr, ...],
    ) -> None:
        pushed_products = 0
        for generator in generators:
            self.visit(generator.target)
            self.visit(generator.iter)
            if generator.is_async:
                self._emit(
                    generator.target,
                    "asynchronous comprehensions are not allowed",
                )
                product = None
            else:
                product = self._bounded_iteration_product(generator.iter)

            if product is not None:
                self._iteration_products.append(product)
                pushed_products += 1
            for condition in generator.ifs:
                self.visit(condition)

        for expression in result_expressions:
            self.visit(expression)
        for _ in range(pushed_products):
            self._iteration_products.pop()

    def _bounded_iteration_product(self, iterable: ast.expr) -> int | None:
        bound = _static_iteration_bound(iterable)
        if bound is None or bound > STRICT_CAPSULE_MAX_ITERATIONS:
            self._emit(
                iterable,
                "bounded control flow requires a statically bounded iterable "
                f"of at most {STRICT_CAPSULE_MAX_ITERATIONS} items",
            )
            return None
        product = self._iteration_products[-1] * bound
        if product > STRICT_CAPSULE_MAX_ITERATIONS:
            self._emit(
                iterable,
                f"nested iteration budget exceeds {STRICT_CAPSULE_MAX_ITERATIONS} items",
            )
            return None
        return product

    def _visit_exception_type(self, node: ast.expr) -> None:
        if isinstance(node, ast.Name) and node.id in _STRICT_CAPSULE_EXCEPTION_CLASSES:
            return
        if isinstance(node, ast.Tuple):
            for element in node.elts:
                self._visit_exception_type(element)
            return
        self.visit(node)


def _function_arguments(arguments: ast.arguments) -> tuple[ast.arg, ...]:
    return (
        *arguments.posonlyargs,
        *arguments.args,
        *((arguments.vararg,) if arguments.vararg is not None else ()),
        *arguments.kwonlyargs,
        *((arguments.kwarg,) if arguments.kwarg is not None else ()),
    )


def _function_has_annotations(node: ast.FunctionDef) -> bool:
    return node.returns is not None or any(
        argument.annotation is not None for argument in _function_arguments(node.args)
    )


def _is_literal_expression(node: ast.expr) -> bool:
    try:
        ast.literal_eval(node)
    except (ValueError, TypeError):
        return False
    return True


def _is_none_literal(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _strict_call_consumes_iterable(node: ast.Call, name: str) -> bool:
    if name not in _STRICT_CAPSULE_ITERABLE_CONSUMERS:
        return False
    if name in {"min", "max"}:
        return len(node.args) == 1
    if name == "sum":
        return len(node.args) in {1, 2}
    return len(node.args) == 1


def _static_iteration_bound(node: ast.expr) -> int | None:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        if any(isinstance(element, ast.Starred) for element in node.elts):
            return None
        return len(node.elts)
    if isinstance(node, ast.Dict):
        if any(key is None for key in node.keys):
            return None
        return len(node.keys)
    if isinstance(
        node,
        (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
    ):
        product = 1
        for generator in node.generators:
            if generator.is_async:
                return None
            bound = _static_iteration_bound(generator.iter)
            if bound is None:
                return None
            product = _multiply_static_cost(product, bound)
        return product
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    if node.keywords:
        return None

    callable_name = node.func.id
    if callable_name == "range":
        return _static_range_bound(node.args)
    if callable_name in _STRICT_CAPSULE_BOUNDED_ITERABLE_WRAPPERS:
        if len(node.args) != 1:
            return None
        return _static_iteration_bound(node.args[0])
    if callable_name == "zip":
        bounds = [_static_iteration_bound(argument) for argument in node.args]
        if any(bound is None for bound in bounds):
            return None
        return min(bounds, default=0)
    return None


def _static_range_bound(arguments: list[ast.expr]) -> int | None:
    if not 1 <= len(arguments) <= 3:
        return None
    values: list[int] = []
    for argument in arguments:
        value = _static_integer_literal(argument)
        if value is None:
            return None
        values.append(value)

    if len(values) == 1:
        start, stop, step = 0, values[0], 1
    elif len(values) == 2:
        start, stop, step = values[0], values[1], 1
    else:
        start, stop, step = values
    if step == 0:
        return None
    if step > 0:
        return max(0, (stop - start + step - 1) // step)
    positive_step = -step
    return max(0, (start - stop + positive_step - 1) // positive_step)


def _static_integer_literal(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) is int
    ):
        return node.operand.value if isinstance(node.op, ast.UAdd) else -node.operand.value
    return None


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
        self._register_aliases(
            module.body,
            scope_id=module_scope_id,
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
                name=node.name,
                defining_scope_id=defining_scope_id,
                body_scope_id=body_scope_id,
                is_top_level=is_top_level,
            )
            self._scopes[body_scope_id].dynamic_callable_names.update(_argument_names(node.args))
            self._scopes[defining_scope_id].bindings.setdefault(node.name, []).append(definition_id)
            definition_ids.append(definition_id)

        for definition_id in definition_ids:
            definition = self._definitions[definition_id]
            local_nodes = _collect_scope_function_definitions(definition.node.body)
            self._register_definitions(
                local_nodes,
                defining_scope_id=definition.body_scope_id,
                is_top_level=False,
            )
            self._register_aliases(
                definition.node.body,
                scope_id=definition.body_scope_id,
                is_top_level=False,
            )
        return definition_ids

    def _register_aliases(
        self,
        statements: list[ast.stmt],
        *,
        scope_id: int,
        is_top_level: bool,
    ) -> None:
        for name, value, line, column in _collect_scope_alias_assignments(statements):
            target_definition_id: int | None = None
            target_name: str | None = None
            if isinstance(value, ast.Name):
                target_name = value.id
            elif isinstance(value, ast.Lambda):
                target_definition_id = self._register_lambda(
                    value,
                    name=name,
                    defining_scope_id=scope_id,
                    is_top_level=is_top_level,
                )
            is_dynamic = not isinstance(value, (ast.Name, ast.Lambda))
            self._scopes[scope_id].aliases.setdefault(name, []).append(
                _AliasBinding(
                    line=line,
                    column=column,
                    target_name=target_name,
                    target_definition_id=target_definition_id,
                    is_dynamic=is_dynamic,
                )
            )

    def _register_lambda(
        self,
        node: ast.Lambda,
        *,
        name: str,
        defining_scope_id: int,
        is_top_level: bool,
    ) -> int:
        definition_id = self._next_definition_id
        self._next_definition_id += 1
        body_scope_id = self._new_scope(parent_scope_id=defining_scope_id)
        self._definitions[definition_id] = _Definition(
            definition_id=definition_id,
            node=node,
            name=name,
            defining_scope_id=defining_scope_id,
            body_scope_id=body_scope_id,
            is_top_level=is_top_level,
        )
        self._scopes[body_scope_id].dynamic_callable_names.update(_argument_names(node.args))
        return definition_id


def _collect_scope_function_definitions(
    statements: list[ast.stmt],
) -> list[_FunctionNode]:
    visitor = _ScopeFunctionDefinitionVisitor()
    for statement in statements:
        visitor.visit(statement)
    return visitor.functions


def _argument_names(arguments: ast.arguments) -> set[str]:
    names = {
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    }
    if arguments.vararg is not None:
        names.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.add(arguments.kwarg.arg)
    return names


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


def _collect_scope_alias_assignments(
    statements: list[ast.stmt],
) -> list[tuple[str, ast.expr | None, int, int]]:
    visitor = _ScopeAliasAssignmentVisitor()
    for statement in statements:
        visitor.visit(statement)
    return sorted(visitor.assignments, key=lambda item: (item[2], item[3], item[0]))


class _ScopeAliasAssignmentVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.assignments: list[tuple[str, ast.expr | None, int, int]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self.assignments.extend(_assignment_alias_bindings(target, node.value))
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            self.assignments.append((node.target.id, node.value, node.lineno, node.col_offset))

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            self.assignments.append((node.target.id, None, node.lineno, node.col_offset))

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        if isinstance(node.target, ast.Name):
            self.assignments.append((node.target.id, node.value, node.lineno, node.col_offset))
        self.visit(node.value)


def _assignment_alias_bindings(
    target: ast.expr,
    value: ast.expr,
) -> list[tuple[str, ast.expr | None, int, int]]:
    if isinstance(target, ast.Name):
        return [(target.id, value, target.lineno, target.col_offset)]
    if isinstance(target, ast.Starred):
        return [
            (name, None, line, column)
            for name, _, line, column in _assignment_alias_bindings(
                target.value,
                value,
            )
        ]
    if isinstance(target, (ast.Tuple, ast.List)):
        if (
            isinstance(value, (ast.Tuple, ast.List))
            and len(target.elts) == len(value.elts)
            and not any(isinstance(item, ast.Starred) for item in target.elts)
        ):
            return [
                binding
                for nested_target, nested_value in zip(
                    target.elts,
                    value.elts,
                    strict=True,
                )
                for binding in _assignment_alias_bindings(
                    nested_target,
                    nested_value,
                )
            ]
        return [
            (name, None, line, column)
            for item in target.elts
            for name, _, line, column in _assignment_alias_bindings(item, value)
        ]
    return []


def _analyze_definitions(
    graph: _DefinitionGraph,
    *,
    side_effect_calls: set[str],
) -> dict[int, _DefinitionAnalysis]:
    return {
        definition_id: _DefinitionAnalysis(
            calls=tuple(
                _resolve_calls(
                    _collect_calls(_definition_body_nodes(definition.node)),
                    graph,
                    scope_id=definition.body_scope_id,
                    side_effect_calls=side_effect_calls,
                )
            ),
        )
        for definition_id, definition in graph.definitions.items()
    }


def _definition_body_nodes(node: _CallableNode) -> list[ast.stmt | ast.expr]:
    if isinstance(node, ast.Lambda):
        return [node.body]
    return list(node.body)


def _resolve_calls(
    calls: list[_CallOccurrence],
    graph: _DefinitionGraph,
    *,
    scope_id: int,
    side_effect_calls: set[str],
) -> list[_ResolvedCall]:
    resolved: list[_ResolvedCall] = []
    for call in calls:
        if call.dynamic_code is not None:
            resolved.append(
                _ResolvedCall(
                    line=call.line,
                    end_line=call.end_line,
                    column=call.column,
                    effect_name=_DYNAMIC_EFFECT_MARKER,
                    dynamic_code=call.dynamic_code,
                    class_method_name=call.class_method_name,
                    class_method_span=call.class_method_span,
                    lambda_span=call.lambda_span,
                )
            )
            continue
        if not call.is_direct_name:
            resolved.append(
                _ResolvedCall(
                    line=call.line,
                    end_line=call.end_line,
                    column=call.column,
                    attribute_name=call.name,
                    class_method_name=call.class_method_name,
                    class_method_span=call.class_method_span,
                    lambda_span=call.lambda_span,
                )
            )
            continue
        effect_name, targets, via_alias, dynamic_code = _resolve_callable_binding(
            graph,
            scope_id=scope_id,
            name=call.name,
            line=call.line,
            side_effect_calls=side_effect_calls,
        )
        if dynamic_code is None and call.uncertain_assignment:
            dynamic_code = "dynamic_effect_call"
        if effect_name is not None or targets:
            resolved.append(
                _ResolvedCall(
                    line=call.line,
                    end_line=call.end_line,
                    column=call.column,
                    effect_name=effect_name,
                    target_definition_ids=targets,
                    via_alias=via_alias,
                    dynamic_code=dynamic_code,
                    class_method_name=call.class_method_name,
                    class_method_span=call.class_method_span,
                    lambda_span=call.lambda_span,
                )
            )
    return resolved


def _resolve_callable_binding(
    graph: _DefinitionGraph,
    *,
    scope_id: int,
    name: str,
    line: int,
    side_effect_calls: set[str],
    visited: frozenset[tuple[int, str, int]] = frozenset(),
) -> tuple[str | None, tuple[int, ...], bool, str | None]:
    binding = _resolve_alias_binding(
        graph,
        scope_id=scope_id,
        name=name,
        line=line,
    )
    if binding is not None:
        marker = (scope_id, name, binding.line)
        if marker in visited:
            return _DYNAMIC_EFFECT_MARKER, (), True, "dynamic_effect_call"
        if binding.is_dynamic:
            return _DYNAMIC_EFFECT_MARKER, (), True, "dynamic_effect_call"
        if binding.target_definition_id is not None:
            return None, (binding.target_definition_id,), True, None
        if binding.target_name is None:
            return None, (), True, None
        effect_name, targets, _, dynamic_code = _resolve_callable_binding(
            graph,
            scope_id=scope_id,
            name=binding.target_name,
            line=binding.line,
            side_effect_calls=side_effect_calls,
            visited=visited | {marker},
        )
        return effect_name, targets, True, dynamic_code

    if _is_dynamic_callable_name(graph, scope_id=scope_id, name=name):
        return _DYNAMIC_EFFECT_MARKER, (), False, "dynamic_effect_call"
    if name in _FORBIDDEN_RUNTIME_CALLS:
        return _DYNAMIC_EFFECT_MARKER, (), False, "forbidden_runtime_access"
    if name in side_effect_calls:
        return name, (), False, None
    return (
        None,
        _resolve_binding(graph, scope_id=scope_id, name=name),
        False,
        None,
    )


def _is_dynamic_callable_name(
    graph: _DefinitionGraph,
    *,
    scope_id: int,
    name: str,
) -> bool:
    current_scope_id: int | None = scope_id
    while current_scope_id is not None:
        scope = graph.scopes[current_scope_id]
        if name in scope.dynamic_callable_names:
            return True
        current_scope_id = scope.parent_scope_id
    return False


def _resolve_alias_binding(
    graph: _DefinitionGraph,
    *,
    scope_id: int,
    name: str,
    line: int,
) -> _AliasBinding | None:
    current_scope_id: int | None = scope_id
    maximum_line = line
    while current_scope_id is not None:
        scope = graph.scopes[current_scope_id]
        applicable = [
            binding for binding in scope.aliases.get(name, ()) if binding.line <= maximum_line
        ]
        if applicable:
            return max(applicable, key=lambda item: (item.line, item.column))
        current_scope_id = scope.parent_scope_id
        maximum_line = 2**31 - 1
    return None


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
            target_id for call in analysis.calls for target_id in call.target_definition_ids
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
                for target_index, target_id in enumerate(call.target_definition_ids):
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
                low_links[definition_id] = min(low_links[definition_id], low_links[target_id])
            elif target_id in on_stack:
                low_links[definition_id] = min(low_links[definition_id], indices[target_id])

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
        definition = graph.definitions[definition_id]
        node = definition.node
        start_line, end_line = _node_span(node)
        violations.append(
            _build_violation(
                code="effectful_helper",
                message=(f"Helper '{definition.name}' can execute a robot side effect"),
                start_line=start_line,
                end_line=end_line,
                regions=regions,
                groups=groups,
                side_effects=effects,
                helper_name=definition.name,
            )
        )
    return violations


def _mark_effectful_class_attribute_calls(
    calls: list[_ResolvedCall],
    summaries: dict[int, tuple[str, ...]],
) -> list[_ResolvedCall]:
    effectful_method_names = {
        call.class_method_name.rsplit(".", 1)[-1]
        for call in calls
        if call.class_method_name is not None and _materialize_effects([call], summaries, limit=1)
    }
    return [
        replace(
            call,
            effect_name=_DYNAMIC_EFFECT_MARKER,
            dynamic_code="dynamic_effect_call",
        )
        if call.class_method_name is None and call.attribute_name in effectful_method_names
        else call
        for call in calls
    ]


def _call_contract_violations(
    graph: _DefinitionGraph,
    analyses: dict[int, _DefinitionAnalysis],
    summaries: dict[int, tuple[str, ...]],
    *,
    module_calls: list[_ResolvedCall],
    regions: list[CodeRegion],
    groups: list[CodeRegionGroup],
) -> list[ProgramContractViolation]:
    root_ids = set(graph.top_level_definition_ids)
    root_ids.update(target_id for call in module_calls for target_id in call.target_definition_ids)
    reachable_ids = _reachable_definition_ids(tuple(sorted(root_ids)), analyses)
    calls = list(module_calls)
    for definition_id in reachable_ids:
        calls.extend(analyses[definition_id].calls)

    violations: list[ProgramContractViolation] = []
    class_calls: dict[tuple[str, tuple[int, int]], list[_ResolvedCall]] = {}
    lambda_calls: dict[tuple[int, int], list[_ResolvedCall]] = {}
    for call in calls:
        if call.class_method_name is not None and call.class_method_span is not None:
            class_calls.setdefault((call.class_method_name, call.class_method_span), []).append(
                call
            )
        if call.lambda_span is not None:
            lambda_calls.setdefault(call.lambda_span, []).append(call)

    for (helper_name, span), helper_calls in sorted(class_calls.items()):
        effects = _materialize_effects(
            helper_calls,
            summaries,
            limit=_EFFECT_SUMMARY_LIMIT,
        )
        if effects:
            violations.append(
                _build_violation(
                    code="effectful_helper",
                    message=(
                        f"Class method '{helper_name}' can execute a robot side "
                        "effect; class definitions are not Capsule-safe"
                    ),
                    start_line=span[0],
                    end_line=span[1],
                    regions=regions,
                    groups=groups,
                    side_effects=effects,
                    helper_name=helper_name,
                )
            )

    for span, contextual_calls in sorted(lambda_calls.items()):
        effects = _materialize_effects(
            contextual_calls,
            summaries,
            limit=_EFFECT_SUMMARY_LIMIT,
        )
        if effects:
            violations.append(
                _build_violation(
                    code="dynamic_effect_call",
                    message=(
                        "An invoked or passed lambda can execute robot side "
                        "effects; use explicit top-level public API calls"
                    ),
                    start_line=span[0],
                    end_line=span[1],
                    regions=regions,
                    groups=groups,
                    side_effects=effects,
                )
            )

    for call in calls:
        if call.class_method_name is not None or call.lambda_span is not None:
            continue
        effects = _materialize_effects([call], summaries, limit=_EFFECT_SUMMARY_LIMIT)
        if call.dynamic_code is not None:
            message = (
                "Callable/private runtime introspection is forbidden in Capsule "
                "programs; use only the documented public API functions directly"
                if call.dynamic_code == "forbidden_runtime_access"
                else ("Dynamic runtime access can bypass Capsule tracing and side-effect guards")
            )
            violations.append(
                _build_violation(
                    code=call.dynamic_code,
                    message=message,
                    start_line=call.line,
                    end_line=call.end_line,
                    regions=regions,
                    groups=groups,
                    side_effects=effects,
                )
            )
        elif call.via_alias and effects:
            violations.append(
                _build_violation(
                    code="aliased_effect_call",
                    message=(
                        "Callable alias can execute a robot side effect; invoke "
                        "the public API function directly"
                    ),
                    start_line=call.line,
                    end_line=call.end_line,
                    regions=regions,
                    groups=groups,
                    side_effects=effects,
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
        (node, graph.module_scope_id) for node in _collect_control_flow_nodes(module.body)
    ]
    for definition_id in _reachable_definition_ids(
        graph.top_level_definition_ids,
        analyses,
    ):
        definition = graph.definitions[definition_id]
        cases.extend(
            (node, definition.body_scope_id)
            for node in _collect_control_flow_nodes(_definition_body_nodes(definition.node))
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
            call for call in module_calls if group.start_line <= call.line <= group.end_line
        ]
        effects = _materialize_effects(group_calls, summaries)
        if len(effects) <= 1:
            continue
        violations.append(
            _build_violation(
                code="multiple_effects_in_group",
                message=(
                    f"Execution group '{group.group_id}' contains multiple robot side-effect calls"
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
        unit_calls = [call for call in module_calls if start_line <= call.line <= end_line]
        return bool(_materialize_effects(unit_calls, summaries, limit=1))

    return (
        tuple(
            region.region_id
            for region in regions
            if unit_has_effect(region.start_line, region.end_line)
        ),
        tuple(
            group.group_id for group in groups if unit_has_effect(group.start_line, group.end_line)
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


def _collect_calls(statements: Iterable[ast.AST]) -> list[_CallOccurrence]:
    visitor = _ExecutableCallVisitor()
    for statement in statements:
        visitor.visit(statement)
    return visitor.calls


class _ExecutableCallVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: list[_CallOccurrence] = []
        self._dynamic_positions: set[tuple[int, int, str]] = set()
        self._class_local_callable_names: list[set[str]] = []
        self._class_names: list[str] = []
        self._active_class_methods: list[tuple[str, tuple[int, int]]] = []
        self._active_lambda_spans: list[tuple[int, int]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_definition_time_expressions(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_definition_time_expressions(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        local_names = {name for name, _, _, _ in _collect_scope_alias_assignments(node.body)}
        local_names.update(
            definition.name for definition in _collect_scope_function_definitions(node.body)
        )
        self._class_local_callable_names.append(local_names)
        self._class_names.append(node.name)
        for expression in sorted(
            [*node.decorator_list, *node.bases, *(item.value for item in node.keywords)],
            key=_source_position,
        ):
            self.visit(expression)
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._visit_definition_time_expressions(statement)
                method_name = ".".join([*self._class_names, statement.name])
                self._active_class_methods.append((method_name, _node_span(statement)))
                for method_statement in statement.body:
                    self.visit(method_statement)
                self._active_class_methods.pop()
            else:
                self.visit(statement)
        self._class_names.pop()
        self._class_local_callable_names.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for expression in sorted(
            [*node.args.defaults, *(item for item in node.args.kw_defaults if item)],
            key=_source_position,
        ):
            self.visit(expression)

    def visit_Call(self, node: ast.Call) -> None:
        for argument in node.args:
            self._visit_call_argument(argument)
        for keyword in node.keywords:
            self._visit_call_argument(keyword.value)

        if isinstance(node.func, ast.Lambda):
            self._visit_effectful_lambda_context(node.func)
            return

        if (
            isinstance(node.func, ast.Name)
            and self._class_local_callable_names
            and node.func.id in self._class_local_callable_names[-1]
        ):
            self._record_dynamic(node, "dynamic_effect_call")
            return

        dynamic_code = _dynamic_call_code(node.func)
        if dynamic_code is not None:
            self._record_dynamic(node, dynamic_code)
            self.visit(node.func)
            return

        self.visit(node.func)

        name = _callable_name(node.func)
        if name is not None:
            self.calls.append(
                _CallOccurrence(
                    name=name,
                    is_direct_name=_is_direct_callable_reference(node.func),
                    line=node.lineno,
                    end_line=int(node.end_lineno or node.lineno),
                    column=node.col_offset,
                    class_method_name=self._current_class_method_name(),
                    class_method_span=self._current_class_method_span(),
                    lambda_span=self._current_lambda_span(),
                )
            )

    def visit_Assign(self, node: ast.Assign) -> None:
        if any(not _assignment_target_is_precise(target, node.value) for target in node.targets):
            for candidate in ast.walk(node.value):
                if not isinstance(candidate, ast.Name) or not isinstance(candidate.ctx, ast.Load):
                    continue
                self.calls.append(
                    _CallOccurrence(
                        name=candidate.id,
                        is_direct_name=True,
                        line=node.lineno,
                        end_line=int(node.end_lineno or node.lineno),
                        column=node.col_offset,
                        class_method_name=self._current_class_method_name(),
                        class_method_span=self._current_class_method_span(),
                        lambda_span=self._current_lambda_span(),
                        uncertain_assignment=True,
                    )
                )
        self.visit(node.value)

    def visit_Import(self, node: ast.Import) -> None:
        if any(alias.name.split(".", 1)[0] in _INTROSPECTION_MODULES for alias in node.names):
            self._record_dynamic(node, "forbidden_runtime_access")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if (node.module or "").split(".", 1)[0] in _INTROSPECTION_MODULES:
            self._record_dynamic(node, "forbidden_runtime_access")

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id in _INTROSPECTION_MODULES:
            self._record_dynamic(node, "forbidden_runtime_access")

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if _is_dynamic_runtime_subscript(node):
            self._record_dynamic(node, "dynamic_effect_call")
            self.visit(node.slice)
            return
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _SENSITIVE_RUNTIME_ATTRIBUTES:
            self._record_dynamic(node, "forbidden_runtime_access")
            return
        self.generic_visit(node)

    def _visit_call_argument(self, node: ast.expr) -> None:
        if isinstance(node, ast.Lambda):
            self._visit_effectful_lambda_context(node)
        else:
            self.visit(node)

    def _visit_effectful_lambda_context(self, node: ast.Lambda) -> None:
        self.visit_Lambda(node)
        self._active_lambda_spans.append(_node_span(node))
        self.visit(node.body)
        self._active_lambda_spans.pop()

    def _record_dynamic(self, node: ast.AST, code: str) -> None:
        line = int(getattr(node, "lineno", 1))
        column = int(getattr(node, "col_offset", 0))
        key = (line, column, code)
        if key in self._dynamic_positions:
            return
        self._dynamic_positions.add(key)
        self.calls.append(
            _CallOccurrence(
                name=_DYNAMIC_EFFECT_MARKER,
                is_direct_name=False,
                line=line,
                end_line=int(getattr(node, "end_lineno", line) or line),
                column=column,
                dynamic_code=code,
                class_method_name=self._current_class_method_name(),
                class_method_span=self._current_class_method_span(),
                lambda_span=self._current_lambda_span(),
            )
        )

    def _current_class_method_name(self) -> str | None:
        return self._active_class_methods[-1][0] if self._active_class_methods else None

    def _current_class_method_span(self) -> tuple[int, int] | None:
        return self._active_class_methods[-1][1] if self._active_class_methods else None

    def _current_lambda_span(self) -> tuple[int, int] | None:
        return self._active_lambda_spans[-1] if self._active_lambda_spans else None

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
            argument.annotation for argument in arguments if argument.annotation is not None
        )
        if node.returns is not None:
            expressions.append(node.returns)

        for expression in sorted(expressions, key=_source_position):
            self.visit(expression)


def _assignment_target_is_precise(target: ast.expr, value: ast.expr) -> bool:
    if isinstance(target, ast.Name):
        return True
    if isinstance(target, ast.Starred):
        return False
    if isinstance(target, (ast.Tuple, ast.List)):
        return (
            isinstance(value, (ast.Tuple, ast.List))
            and len(target.elts) == len(value.elts)
            and all(
                _assignment_target_is_precise(nested_target, nested_value)
                for nested_target, nested_value in zip(
                    target.elts,
                    value.elts,
                    strict=True,
                )
            )
        )
    return False


_FORBIDDEN_RUNTIME_CALLS = {
    "globals",
    "locals",
    "eval",
    "exec",
    "getattr",
    "vars",
    "dir",
    "__import__",
}
_INTROSPECTION_MODULES = {"inspect", "gc", "builtins", "__builtins__"}
_SENSITIVE_RUNTIME_ATTRIBUTES = {
    "__wrapped__",
    "__closure__",
    "cell_contents",
    "__globals__",
    "__self__",
    "__func__",
    "__code__",
    "__dict__",
    "__getattribute__",
}


def _dynamic_call_code(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name) and node.id in _FORBIDDEN_RUNTIME_CALLS:
        return "forbidden_runtime_access"
    if isinstance(node, ast.Attribute):
        if node.attr in _SENSITIVE_RUNTIME_ATTRIBUTES:
            return "forbidden_runtime_access"
        if _root_name(node) in _INTROSPECTION_MODULES:
            return "forbidden_runtime_access"
        if _contains_dynamic_runtime_root(node.value):
            return "dynamic_effect_call"
    if isinstance(node, ast.Subscript):
        return "dynamic_effect_call"
    if isinstance(node, ast.Call):
        inner_name = _callable_name(node.func)
        if inner_name in _FORBIDDEN_RUNTIME_CALLS:
            return "dynamic_effect_call"
    return None


def _root_name(node: ast.AST) -> str | None:
    current = node
    while isinstance(current, (ast.Attribute, ast.Subscript)):
        current = current.value
    return current.id if isinstance(current, ast.Name) else None


def _is_dynamic_runtime_subscript(node: ast.Subscript) -> bool:
    return _contains_dynamic_runtime_root(node.value)


def _contains_dynamic_runtime_root(node: ast.AST) -> bool:
    current = node
    while isinstance(current, (ast.Attribute, ast.Subscript)):
        current = current.value
    if isinstance(current, ast.Name):
        return current.id == "APIS"
    return (
        isinstance(current, ast.Call)
        and isinstance(current.func, ast.Name)
        and current.func.id in {"globals", "locals", "getattr"}
    )


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
    if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
        return node.target.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_direct_callable_reference(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) or (
        isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name)
    )


def _ordered_unique(values: Iterable[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
