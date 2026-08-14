from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Literal

from capx.runtime_control.schema import CodeRegion, CodeRegionGroup


class LineageAmbiguityError(ValueError):
    pass


def _first_duplicate(values: list[str]) -> str | None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


@dataclass(frozen=True)
class SourceRevision:
    revision: int
    source_sha256: str
    edit_kind: Literal["initial", "patch_region", "patch_group", "append_recovery"]
    parent_revision: int | None
    old_line_count: int


@dataclass
class RecoveryGeneration:
    generation_id: str
    source_revision: int
    start_line: int
    end_line: int
    observation_functions: tuple[str, ...]
    authorized_group_keys: set[str] = field(default_factory=set)
    executed_group_keys: set[str] = field(default_factory=set)
    append_trace_revision: int = 0


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
        duplicate_region_id = _first_duplicate([region.region_id for region in regions])
        if duplicate_region_id is not None:
            raise LineageAmbiguityError(f"duplicate region id: {duplicate_region_id}")
        duplicate_group_id = _first_duplicate([group.group_id for group in groups])
        if duplicate_group_id is not None:
            raise LineageAmbiguityError(f"duplicate group id: {duplicate_group_id}")

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


def _stable_key_suffix(stable_key: str, unit_kind: str) -> int:
    match = re.fullmatch(rf"{unit_kind}_key_([0-9]+)", stable_key)
    if match is None:
        raise LineageAmbiguityError(
            f"{unit_kind} stable key format is invalid: {stable_key}"
        )
    suffix = int(match.group(1))
    if suffix <= 0 or stable_key != f"{unit_kind}_key_{suffix:06d}":
        raise LineageAmbiguityError(
            f"{unit_kind} stable key format is invalid: {stable_key}"
        )
    return suffix


def _validate_lineage_units(
    *,
    phase: str,
    unit_kind: str,
    units: list[CodeRegion] | list[CodeRegionGroup],
    key_by_id: dict[str, str],
    executed_keys: set[str],
    next_key: int,
) -> None:
    unit_ids = [_unit_id(unit, unit_kind) for unit in units]
    duplicate_id = _first_duplicate(unit_ids)
    if duplicate_id is not None:
        raise LineageAmbiguityError(
            f"duplicate {phase} {unit_kind} id: {duplicate_id}"
        )

    if set(key_by_id) != set(unit_ids):
        raise LineageAmbiguityError(
            f"{phase} {unit_kind} map ids do not exactly match {unit_kind} units"
        )

    stable_keys = list(key_by_id.values())
    duplicate_stable_key = _first_duplicate(stable_keys)
    if duplicate_stable_key is not None:
        if duplicate_stable_key in executed_keys:
            raise LineageAmbiguityError(
                f"executed {unit_kind} key identifies multiple previous units: "
                f"{duplicate_stable_key}"
            )
        raise LineageAmbiguityError(
            f"{phase} {unit_kind} stable keys must be unique: {duplicate_stable_key}"
        )

    stable_key_set = set(stable_keys)
    missing_executed_keys = executed_keys - stable_key_set
    if missing_executed_keys:
        missing = ", ".join(sorted(missing_executed_keys))
        raise LineageAmbiguityError(
            f"executed {unit_kind} key is missing from {phase} lineage map: {missing}"
        )

    suffixes = [_stable_key_suffix(stable_key, unit_kind) for stable_key in stable_keys]
    if isinstance(next_key, bool) or not isinstance(next_key, int) or next_key <= 0:
        raise LineageAmbiguityError(f"next {unit_kind} key counter must be a positive integer")
    maximum_suffix = max(suffixes, default=0)
    if next_key <= maximum_suffix:
        raise LineageAmbiguityError(
            f"next {unit_kind} key counter must exceed existing stable keys"
        )


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
    current_lines = current_source.splitlines()
    old_line_count = len(old_lines)
    actual_line_delta = len(current_lines) - old_line_count
    if line_delta != actual_line_delta:
        raise LineageAmbiguityError(
            f"line_delta {line_delta} does not match source line delta {actual_line_delta}"
        )
    if edit_kind == "append_recovery":
        if not current_source.startswith(previous_source):
            raise LineageAmbiguityError("append changed the previous source prefix")
        if (
            edit_start_line != old_line_count + 1
            or edit_end_line != old_line_count
            or actual_line_delta <= 0
        ):
            raise LineageAmbiguityError(
                "append edit must start after the old source, use an empty old span, "
                "and add at least one line"
            )
        suffix = current_source[len(previous_source) :]
        if (
            previous_source
            and not previous_source.endswith(("\n", "\r"))
            and not suffix.startswith(("\n", "\r\n"))
        ):
            raise LineageAmbiguityError("append must start on a new line")
        if any(
            group.start_line <= old_line_count < group.end_line for group in current_groups
        ):
            raise LineageAmbiguityError(
                "current group crosses the append boundary between old and new source"
            )
    elif not (1 <= edit_start_line <= edit_end_line <= old_line_count):
        raise LineageAmbiguityError(
            "patch edit span must be a valid closed interval in the previous source"
        )

    _validate_lineage_units(
        phase="previous",
        unit_kind="region",
        units=previous_regions,
        key_by_id=previous_lineage.region_key_by_id,
        executed_keys=previous_lineage.executed_region_keys,
        next_key=previous_lineage.next_region_key,
    )
    _validate_lineage_units(
        phase="previous",
        unit_kind="group",
        units=previous_groups,
        key_by_id=previous_lineage.group_key_by_id,
        executed_keys=previous_lineage.executed_group_keys,
        next_key=previous_lineage.next_group_key,
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
    _validate_lineage_units(
        phase="output",
        unit_kind="region",
        units=current_regions,
        key_by_id=reconciled.region_key_by_id,
        executed_keys=reconciled.executed_region_keys,
        next_key=reconciled.next_region_key,
    )
    _validate_lineage_units(
        phase="output",
        unit_kind="group",
        units=current_groups,
        key_by_id=reconciled.group_key_by_id,
        executed_keys=reconciled.executed_group_keys,
        next_key=reconciled.next_group_key,
    )
    return reconciled
