from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from capx.runtime_control.schema import CodeRegion, CodeRegionGroup


class LineageAmbiguityError(ValueError):
    pass


@dataclass(frozen=True)
class SourceRevision:
    revision: int
    source_sha256: str
    edit_kind: Literal["initial", "patch_region", "patch_group", "append_recovery"]
    parent_revision: int | None
    old_line_count: int


@dataclass
class UnitLineage:
    next_region_key: int = 1
    next_group_key: int = 1
    region_key_by_id: dict[str, str] = field(default_factory=dict)
    group_key_by_id: dict[str, str] = field(default_factory=dict)
    executed_region_keys: set[str] = field(default_factory=set)
    executed_group_keys: set[str] = field(default_factory=set)

    @classmethod
    def create(
        cls,
        regions: list[CodeRegion],
        groups: list[CodeRegionGroup],
    ) -> UnitLineage:
        lineage = cls()
        for region in regions:
            lineage.region_key_by_id[region.region_id] = lineage.allocate_region_key()
        for group in groups:
            lineage.group_key_by_id[group.group_id] = lineage.allocate_group_key()
        return lineage

    def allocate_region_key(self) -> str:
        key = f"region_key_{self.next_region_key:06d}"
        self.next_region_key += 1
        return key

    def allocate_group_key(self) -> str:
        key = f"group_key_{self.next_group_key:06d}"
        self.next_group_key += 1
        return key


def _unit_id(unit: CodeRegion | CodeRegionGroup, unit_kind: str) -> str:
    if unit_kind == "region":
        return unit.region_id
    return unit.group_id


def _validate_unique_current_ids(
    units: list[CodeRegion] | list[CodeRegionGroup],
    unit_kind: str,
) -> None:
    seen: set[str] = set()
    for unit in units:
        unit_id = _unit_id(unit, unit_kind)
        if unit_id in seen:
            raise LineageAmbiguityError(f"duplicate current {unit_kind} id: {unit_id}")
        seen.add(unit_id)


def _expected_span(
    *,
    unit: CodeRegion | CodeRegionGroup,
    edit_kind: str,
    old_line_count: int,
    edit_start_line: int,
    edit_end_line: int,
    line_delta: int,
) -> tuple[int, int] | None:
    if edit_kind == "append_recovery":
        if unit.end_line <= old_line_count:
            return unit.start_line, unit.end_line
        return None
    if unit.end_line < edit_start_line:
        return unit.start_line, unit.end_line
    if unit.start_line > edit_end_line:
        return unit.start_line + line_delta, unit.end_line + line_delta
    return None


def _reconcile_units(
    *,
    unit_kind: str,
    edit_kind: str,
    old_line_count: int,
    edit_start_line: int,
    edit_end_line: int,
    line_delta: int,
    previous_units: list[CodeRegion] | list[CodeRegionGroup],
    current_units: list[CodeRegion] | list[CodeRegionGroup],
    previous_key_by_id: dict[str, str],
    executed_keys: set[str],
    current_key_by_id: dict[str, str],
    allocate_key: Callable[[], str],
) -> None:
    _validate_unique_current_ids(current_units, unit_kind)

    missing_executed_keys = executed_keys - set(previous_key_by_id.values())
    if missing_executed_keys:
        missing = ", ".join(sorted(missing_executed_keys))
        raise LineageAmbiguityError(
            f"executed {unit_kind} key is missing from previous lineage map: {missing}"
        )
    for executed_key in executed_keys:
        previous_matches = [
            previous_unit
            for previous_unit in previous_units
            if previous_key_by_id.get(_unit_id(previous_unit, unit_kind)) == executed_key
        ]
        if len(previous_matches) != 1:
            raise LineageAmbiguityError(
                f"executed {unit_kind} key does not identify exactly one previous unit: "
                f"{executed_key}"
            )

    mapped_executed_keys: set[str] = set()
    assigned_current_ids: set[str] = set()
    for previous_unit in previous_units:
        previous_id = _unit_id(previous_unit, unit_kind)
        stable_key = previous_key_by_id.get(previous_id)
        if stable_key is None:
            continue
        expected_span = _expected_span(
            unit=previous_unit,
            edit_kind=edit_kind,
            old_line_count=old_line_count,
            edit_start_line=edit_start_line,
            edit_end_line=edit_end_line,
            line_delta=line_delta,
        )
        matches: list[CodeRegion | CodeRegionGroup] = []
        if expected_span is not None:
            expected_start, expected_end = expected_span
            matches = [
                current_unit
                for current_unit in current_units
                if current_unit.start_line == expected_start
                and current_unit.end_line == expected_end
                and current_unit.source == previous_unit.source
            ]
        if len(matches) > 1:
            raise LineageAmbiguityError(
                f"multiple current {unit_kind} units match previous unit {previous_id}"
            )
        if not matches:
            if stable_key in executed_keys:
                raise LineageAmbiguityError(
                    f"executed {unit_kind} key could not be mapped: {stable_key}"
                )
            continue

        current_id = _unit_id(matches[0], unit_kind)
        if current_id in assigned_current_ids:
            raise LineageAmbiguityError(
                f"current {unit_kind} unit matches multiple previous units: {current_id}"
            )
        current_key_by_id[current_id] = stable_key
        assigned_current_ids.add(current_id)
        if stable_key in executed_keys:
            mapped_executed_keys.add(stable_key)

    unmapped_executed_keys = executed_keys - mapped_executed_keys
    if unmapped_executed_keys:
        unmapped = ", ".join(sorted(unmapped_executed_keys))
        raise LineageAmbiguityError(
            f"executed {unit_kind} key could not be uniquely mapped: {unmapped}"
        )

    for current_unit in current_units:
        current_id = _unit_id(current_unit, unit_kind)
        if current_id not in current_key_by_id:
            current_key_by_id[current_id] = allocate_key()


def reconcile_lineage(
    *,
    edit_kind: str,
    previous_source: str,
    current_source: str,
    previous_regions: list[CodeRegion],
    current_regions: list[CodeRegion],
    previous_groups: list[CodeRegionGroup],
    current_groups: list[CodeRegionGroup],
    previous_lineage: UnitLineage,
    edit_start_line: int,
    edit_end_line: int,
    line_delta: int,
) -> UnitLineage:
    if edit_kind not in {"append_recovery", "patch_region", "patch_group"}:
        raise ValueError(f"Unsupported lineage edit kind: {edit_kind}")

    old_lines = previous_source.splitlines()
    old_line_count = len(old_lines)
    if edit_kind == "append_recovery":
        current_lines = current_source.splitlines()
        if current_lines[:old_line_count] != old_lines:
            raise LineageAmbiguityError("append changed the previous source prefix")
        if any(
            group.start_line <= old_line_count < group.end_line for group in current_groups
        ):
            raise LineageAmbiguityError(
                "current group crosses the append boundary between old and new source"
            )

    reconciled = UnitLineage(
        next_region_key=previous_lineage.next_region_key,
        next_group_key=previous_lineage.next_group_key,
        executed_region_keys=set(previous_lineage.executed_region_keys),
        executed_group_keys=set(previous_lineage.executed_group_keys),
    )
    _reconcile_units(
        unit_kind="region",
        edit_kind=edit_kind,
        old_line_count=old_line_count,
        edit_start_line=edit_start_line,
        edit_end_line=edit_end_line,
        line_delta=line_delta,
        previous_units=previous_regions,
        current_units=current_regions,
        previous_key_by_id=previous_lineage.region_key_by_id,
        executed_keys=previous_lineage.executed_region_keys,
        current_key_by_id=reconciled.region_key_by_id,
        allocate_key=reconciled.allocate_region_key,
    )
    _reconcile_units(
        unit_kind="group",
        edit_kind=edit_kind,
        old_line_count=old_line_count,
        edit_start_line=edit_start_line,
        edit_end_line=edit_end_line,
        line_delta=line_delta,
        previous_units=previous_groups,
        current_units=current_groups,
        previous_key_by_id=previous_lineage.group_key_by_id,
        executed_keys=previous_lineage.executed_group_keys,
        current_key_by_id=reconciled.group_key_by_id,
        allocate_key=reconciled.allocate_group_key,
    )
    return reconciled
