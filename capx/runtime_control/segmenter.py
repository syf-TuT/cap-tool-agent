from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass

from capx.runtime_control.schema import CodeRegion, CodeRegionGroup


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
class _RegionAnalysis:
    primitive_calls: list[str]
    defined_names: list[str]
    used_names: list[str]
    has_robot_side_effect: bool
    is_effect: bool


def segment_python_code(source: str) -> list[CodeRegion]:
    """Split Python source into deterministic top-level execution regions."""
    module = ast.parse(source)
    lines = source.splitlines()
    regions: list[CodeRegion] = []

    for idx, node in enumerate(module.body, start=1):
        start_line = getattr(node, "lineno", None)
        end_line = getattr(node, "end_lineno", None)
        if start_line is None or end_line is None:
            continue
        region_source = "\n".join(lines[start_line - 1 : end_line])
        regions.append(
            CodeRegion(
                region_id=f"region_{idx}",
                start_line=start_line,
                end_line=end_line,
                source=region_source,
            )
        )

    return regions


def segment_python_code_groups(
    source: str,
    regions: list[CodeRegion] | None = None,
    *,
    max_regions_per_group: int = 20,
    side_effect_calls: set[str] | None = None,
) -> list[CodeRegionGroup]:
    """Merge atomic regions into deterministic sense->act execution groups.

    A group spans a sense/compute prologue plus one or more consecutive effect
    statements (bare calls whose return value is discarded). When code returns
    to sensing/computing after an effect, a new group begins. Boundaries are
    structural and domain-agnostic; the effect-primitive vocabulary is used only
    to mark ``has_robot_side_effect``, never to decide boundaries.

    ``side_effect_calls`` is the set of primitive names the environment declares
    as rollback-relevant robot side effects. Defaults to ``ROBOT_SIDE_EFFECT_CALLS``
    when the caller does not inject one.

    ``max_regions_per_group`` is a loose safety fallback that caps pathological
    boundary-free code, not the primary segmentation signal.
    """
    if regions is None:
        regions = segment_python_code(source)
    if not regions:
        return []

    if side_effect_calls is None:
        side_effect_calls = ROBOT_SIDE_EFFECT_CALLS

    groups: list[CodeRegionGroup] = []
    current: list[tuple[CodeRegion, _RegionAnalysis]] = []
    current_has_effect = False

    for region in regions:
        analysis = _analyze_region(region, side_effect_calls)
        returns_to_sense = current_has_effect and not analysis.is_effect
        if current and (returns_to_sense or len(current) >= max_regions_per_group):
            groups.append(_build_group(len(groups) + 1, current))
            current = []
            current_has_effect = False
        current.append((region, analysis))
        current_has_effect = current_has_effect or analysis.is_effect

    if current:
        groups.append(_build_group(len(groups) + 1, current))

    return groups


def _analyze_region(region: CodeRegion, side_effect_calls: set[str]) -> _RegionAnalysis:
    try:
        module = ast.parse(region.source)
    except SyntaxError:
        return _RegionAnalysis([], [], [], False, False)

    primitive_calls = _ordered_unique(_call_names(module))
    defined_names = _ordered_unique(_defined_names(module))
    defined_name_set = set(defined_names)
    used_names = _ordered_unique(
        name for name in _used_names(module) if name not in defined_name_set
    )
    return _RegionAnalysis(
        primitive_calls=primitive_calls,
        defined_names=defined_names,
        used_names=used_names,
        has_robot_side_effect=any(name in side_effect_calls for name in primitive_calls),
        is_effect=_is_effect_region(module),
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


def _build_group(
    group_index: int,
    items: list[tuple[CodeRegion, _RegionAnalysis]],
) -> CodeRegionGroup:
    regions = [region for region, _ in items]
    analyses = [analysis for _, analysis in items]
    return CodeRegionGroup(
        group_id=f"group_{group_index}",
        start_line=regions[0].start_line,
        end_line=regions[-1].end_line,
        source="\n".join(region.source for region in regions),
        region_ids=[region.region_id for region in regions],
        primitive_calls=_ordered_unique(
            call for analysis in analyses for call in analysis.primitive_calls
        ),
        defined_names=_ordered_unique(
            name for analysis in analyses for name in analysis.defined_names
        ),
        used_names=_ordered_unique(name for analysis in analyses for name in analysis.used_names),
        has_robot_side_effect=any(analysis.has_robot_side_effect for analysis in analyses),
    )


def _call_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _callable_name(child.func)
            if name:
                names.append(name)
    return names


def _callable_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


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
