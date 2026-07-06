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
    if not regions:
        return []

    groups: list[CodeRegionGroup] = []
    current: list[tuple[CodeRegion, RegionAnalysis]] = []
    current_has_effect = False

    for region, analysis in zip(regions, analyses):
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


def _build_group(
    group_index: int,
    items: list[tuple[CodeRegion, RegionAnalysis]],
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
