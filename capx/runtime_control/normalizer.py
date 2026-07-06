from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from capx.runtime_control.schema import CodeRegion, CodeRegionGroup
from capx.runtime_control.segmenter import (
    ROBOT_SIDE_EFFECT_CALLS,
    RegionAnalysis,
    analyze_python_regions,
    segment_python_code,
)


@dataclass(frozen=True)
class GroupingPolicy:
    max_regions_per_group: int = 20
    min_groups: int = 3
    max_groups: int = 8


@dataclass(frozen=True)
class _NormalizedRegionAnalysis:
    region_id: str
    primitive_calls: list[str]
    defined_names: list[str]
    used_names: list[str]
    has_robot_side_effect: bool
    has_structural_effect: bool


def segment_python_code_groups(
    source: str,
    regions: list[CodeRegion] | None = None,
    *,
    max_regions_per_group: int = 20,
    side_effect_calls: set[str] | None = None,
) -> list[CodeRegionGroup]:
    """Normalize structural regions into sense->act execution groups."""
    if regions is None:
        regions = segment_python_code(source)
    if side_effect_calls is None:
        side_effect_calls = ROBOT_SIDE_EFFECT_CALLS

    analyses = analyze_python_regions(source, regions, side_effect_calls=side_effect_calls)
    return normalize_python_code_groups(
        source,
        regions,
        analyses,
        policy=GroupingPolicy(max_regions_per_group=max_regions_per_group),
    )


def normalize_python_code_groups(
    source: str,
    regions: list[CodeRegion],
    analyses: list[RegionAnalysis],
    *,
    policy: GroupingPolicy,
) -> list[CodeRegionGroup]:
    """Apply group-boundary policy to already-segmented code regions.

    This layer only builds grouping metadata. It never rewrites executable
    source; each group source is the byte-for-byte concatenation of its member
    region sources.
    """
    _validate_normalizer_inputs(source, regions, analyses)
    if not regions:
        return []

    normalized_analyses = _normalize_effect_metadata(analyses)
    groups: list[CodeRegionGroup] = []
    current: list[tuple[CodeRegion, _NormalizedRegionAnalysis]] = []
    current_has_effect = False

    for region, analysis in zip(regions, normalized_analyses):
        returns_to_sense = current_has_effect and not analysis.has_structural_effect
        if current and (
            returns_to_sense or len(current) >= policy.max_regions_per_group
        ):
            groups.append(_build_group(len(groups) + 1, current))
            current = []
            current_has_effect = False

        current.append((region, analysis))
        current_has_effect = current_has_effect or analysis.has_structural_effect

    if current:
        groups.append(_build_group(len(groups) + 1, current))

    return groups


def _validate_normalizer_inputs(
    source: str,
    regions: list[CodeRegion],
    analyses: list[RegionAnalysis],
) -> None:
    if regions and source != "".join(region.source for region in regions):
        raise ValueError("Normalizer source must match concatenated region source")

    if len(regions) != len(analyses):
        raise ValueError("Normalizer regions and analyses must have the same number of items")

    for region, analysis in zip(regions, analyses):
        if region.region_id != analysis.region_id:
            raise ValueError(
                "Normalizer analysis region_id must match the paired region "
                f"({region.region_id} != {analysis.region_id})"
            )


def _normalize_effect_metadata(
    analyses: list[RegionAnalysis],
) -> list[_NormalizedRegionAnalysis]:
    helper_side_effect_calls = _helper_side_effect_calls(analyses)

    return [
        _normalize_region_effect_metadata(analysis, helper_side_effect_calls)
        for analysis in analyses
    ]


def _helper_side_effect_calls(
    analyses: list[RegionAnalysis],
) -> dict[str, list[str]]:
    direct_side_effects: dict[str, list[str]] = {}
    helper_calls: dict[str, list[str]] = {}
    for analysis in analyses:
        for helper_name in analysis.defined_functions:
            direct_side_effects[helper_name] = list(
                analysis.defined_function_side_effect_calls.get(helper_name, [])
            )
            helper_calls[helper_name] = list(
                analysis.defined_function_call_names.get(helper_name, [])
            )

    return {
        helper_name: _resolve_helper_side_effect_calls(
            helper_name,
            direct_side_effects,
            helper_calls,
            visiting=set(),
        )
        for helper_name in direct_side_effects
    }


def _resolve_helper_side_effect_calls(
    helper_name: str,
    direct_side_effects: dict[str, list[str]],
    helper_calls: dict[str, list[str]],
    *,
    visiting: set[str],
) -> list[str]:
    if helper_name in visiting:
        return []

    side_effects = list(direct_side_effects.get(helper_name, []))
    next_visiting = {*visiting, helper_name}
    for called_helper_name in helper_calls.get(helper_name, []):
        if called_helper_name not in direct_side_effects:
            continue
        side_effects.extend(
            _resolve_helper_side_effect_calls(
                called_helper_name,
                direct_side_effects,
                helper_calls,
                visiting=next_visiting,
            )
        )

    return _ordered_unique(side_effects)


def _normalize_region_effect_metadata(
    analysis: RegionAnalysis,
    helper_side_effect_calls: dict[str, list[str]],
) -> _NormalizedRegionAnalysis:
    inherited_side_effect_calls = _ordered_unique(
        call
        for top_level_call_name in analysis.top_level_call_names
        for call in helper_side_effect_calls.get(top_level_call_name, [])
    )
    is_helper_definition = bool(analysis.defined_functions)
    local_side_effect_calls = (
        [] if is_helper_definition else list(analysis.lexical_side_effect_calls)
    )
    normalized_side_effect_calls = _ordered_unique(
        [*local_side_effect_calls, *inherited_side_effect_calls]
    )
    primitive_calls = list(analysis.primitive_calls)
    if is_helper_definition:
        lexical_side_effect_call_set = set(analysis.lexical_side_effect_calls)
        primitive_calls = [
            call for call in primitive_calls if call not in lexical_side_effect_call_set
        ]

    return _NormalizedRegionAnalysis(
        region_id=analysis.region_id,
        primitive_calls=_ordered_unique([*primitive_calls, *inherited_side_effect_calls]),
        defined_names=analysis.defined_names,
        used_names=analysis.used_names,
        has_robot_side_effect=bool(normalized_side_effect_calls),
        has_structural_effect=(
            analysis.has_structural_effect or bool(normalized_side_effect_calls)
        ),
    )


def _build_group(
    group_index: int,
    items: list[tuple[CodeRegion, _NormalizedRegionAnalysis]],
) -> CodeRegionGroup:
    regions = [region for region, _ in items]
    analyses = [analysis for _, analysis in items]
    return CodeRegionGroup(
        group_id=f"group_{group_index}",
        start_line=regions[0].start_line,
        end_line=regions[-1].end_line,
        source="".join(region.source for region in regions),
        region_ids=[region.region_id for region in regions],
        primitive_calls=_ordered_unique(
            call for analysis in analyses for call in analysis.primitive_calls
        ),
        defined_names=_ordered_unique(
            name for analysis in analyses for name in analysis.defined_names
        ),
        used_names=_ordered_unique(
            name for analysis in analyses for name in analysis.used_names
        ),
        has_robot_side_effect=any(analysis.has_robot_side_effect for analysis in analyses),
    )


def _ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
