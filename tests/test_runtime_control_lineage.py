from dataclasses import FrozenInstanceError

import pytest

import capx.runtime_control as runtime_control
import capx.runtime_control.lineage as lineage_module
from capx.runtime_control import LineageAmbiguityError, SourceRevision, UnitLineage
from capx.runtime_control.schema import CodeRegion, CodeRegionGroup


def _region(region_id: str, source: str) -> CodeRegion:
    return CodeRegion(region_id=region_id, start_line=1, end_line=1, source=source)


def _group(group_id: str, source: str) -> CodeRegionGroup:
    return CodeRegionGroup(group_id=group_id, start_line=1, end_line=1, source=source)


def _region_at(region_id: str, start_line: int, end_line: int, source: str) -> CodeRegion:
    return CodeRegion(
        region_id=region_id,
        start_line=start_line,
        end_line=end_line,
        source=source,
    )


def _group_at(group_id: str, start_line: int, end_line: int, source: str) -> CodeRegionGroup:
    return CodeRegionGroup(
        group_id=group_id,
        start_line=start_line,
        end_line=end_line,
        source=source,
    )


def _lineage_snapshot(lineage: UnitLineage) -> tuple[object, ...]:
    return (
        lineage.next_region_key,
        lineage.next_group_key,
        dict(lineage.region_key_by_id),
        dict(lineage.group_key_by_id),
        set(lineage.executed_region_keys),
        set(lineage.executed_group_keys),
    )


def _reconcile(**kwargs: object) -> UnitLineage:
    return lineage_module.reconcile_lineage(**kwargs)


def test_create_assigns_stable_keys_in_input_order() -> None:
    lineage = UnitLineage.create(
        [_region("region_a", "a = 1"), _region("region_b", "b = 2")],
        [_group("group_a", "a = 1"), _group("group_b", "b = 2")],
    )

    assert lineage.region_key_by_id == {
        "region_a": "region_key_000001",
        "region_b": "region_key_000002",
    }
    assert lineage.group_key_by_id == {
        "group_a": "group_key_000001",
        "group_b": "group_key_000002",
    }


def test_create_assigns_distinct_keys_to_regions_with_identical_source() -> None:
    lineage = UnitLineage.create(
        [_region("region_a", "shared = True"), _region("region_b", "shared = True")],
        [],
    )

    assert lineage.region_key_by_id["region_a"] != lineage.region_key_by_id["region_b"]


def test_create_rejects_duplicate_temporary_ids() -> None:
    with pytest.raises(LineageAmbiguityError, match="duplicate region id"):
        UnitLineage.create(
            [_region("duplicate", "a = 1"), _region("duplicate", "b = 2")],
            [],
        )
    with pytest.raises(LineageAmbiguityError, match="duplicate group id"):
        UnitLineage.create(
            [],
            [_group("duplicate", "a = 1"), _group("duplicate", "b = 2")],
        )


def test_key_allocators_continue_monotonically_with_zero_padding() -> None:
    lineage = UnitLineage.create(
        [_region("region_a", "a = 1"), _region("region_b", "b = 2")],
        [_group("group_a", "a = 1")],
    )

    assert lineage.allocate_region_key() == "region_key_000003"
    assert lineage.allocate_region_key() == "region_key_000004"
    assert lineage.allocate_group_key() == "group_key_000002"
    assert lineage.allocate_group_key() == "group_key_000003"


def test_executed_key_sets_are_empty_and_not_shared_between_instances() -> None:
    first = UnitLineage.create([], [])
    second = UnitLineage.create([], [])

    assert first.executed_region_keys == set()
    assert first.executed_group_keys == set()
    first.executed_region_keys.add("region_key_000001")
    first.executed_group_keys.add("group_key_000001")

    assert second.executed_region_keys == set()
    assert second.executed_group_keys == set()


def test_source_revision_has_required_fields_and_is_frozen() -> None:
    revision = SourceRevision(
        revision=3,
        source_sha256="abc123",
        edit_kind="patch_group",
        parent_revision=2,
        old_line_count=41,
    )

    assert revision.revision == 3
    assert revision.source_sha256 == "abc123"
    assert revision.edit_kind == "patch_group"
    assert revision.parent_revision == 2
    assert revision.old_line_count == 41
    with pytest.raises(FrozenInstanceError):
        revision.revision = 4


def test_lineage_ambiguity_error_is_a_value_error() -> None:
    assert issubclass(LineageAmbiguityError, ValueError)


def test_reconcile_lineage_is_exported_from_runtime_control_package() -> None:
    assert runtime_control.reconcile_lineage is lineage_module.reconcile_lineage


def test_append_preserves_prefix_keys_and_allocates_fresh_keys_for_repeated_source() -> None:
    previous_source = "x = 1\nmove()"
    current_source = "x = 1\nmove()\nx = 1\nmove()"
    previous_regions = [
        _region_at("old_region_1", 1, 1, "x = 1"),
        _region_at("old_region_2", 2, 2, "move()"),
    ]
    previous_groups = [_group_at("old_group", 1, 2, previous_source)]
    current_regions = [
        _region_at("current_region_1", 1, 1, "x = 1"),
        _region_at("current_region_2", 2, 2, "move()"),
        _region_at("new_region_1", 3, 3, "x = 1"),
        _region_at("new_region_2", 4, 4, "move()"),
    ]
    current_groups = [
        _group_at("current_old_group", 1, 2, previous_source),
        _group_at("new_repeated_group", 3, 4, previous_source),
    ]
    previous_lineage = UnitLineage.create(previous_regions, previous_groups)
    previous_lineage.executed_region_keys.add(
        previous_lineage.region_key_by_id["old_region_1"]
    )
    previous_lineage.executed_group_keys.add(previous_lineage.group_key_by_id["old_group"])

    reconciled = _reconcile(
        edit_kind="append_recovery",
        previous_source=previous_source,
        current_source=current_source,
        previous_regions=previous_regions,
        current_regions=current_regions,
        previous_groups=previous_groups,
        current_groups=current_groups,
        previous_lineage=previous_lineage,
        edit_start_line=3,
        edit_end_line=2,
        line_delta=2,
    )

    assert reconciled.region_key_by_id["current_region_1"] == "region_key_000001"
    assert reconciled.region_key_by_id["current_region_2"] == "region_key_000002"
    assert reconciled.group_key_by_id["current_old_group"] == "group_key_000001"
    assert reconciled.region_key_by_id["new_region_1"] == "region_key_000003"
    assert reconciled.region_key_by_id["new_region_2"] == "region_key_000004"
    assert reconciled.group_key_by_id["new_repeated_group"] == "group_key_000002"
    assert not (
        {
            reconciled.region_key_by_id["new_region_1"],
            reconciled.region_key_by_id["new_region_2"],
        }
        & reconciled.executed_region_keys
    )
    assert (
        reconciled.group_key_by_id["new_repeated_group"]
        not in reconciled.executed_group_keys
    )
    assert reconciled.executed_region_keys == previous_lineage.executed_region_keys
    assert reconciled.executed_group_keys == previous_lineage.executed_group_keys
    assert reconciled.executed_region_keys is not previous_lineage.executed_region_keys
    assert reconciled.executed_group_keys is not previous_lineage.executed_group_keys
    assert len(set(reconciled.region_key_by_id.values())) == len(
        reconciled.region_key_by_id
    )
    assert len(set(reconciled.group_key_by_id.values())) == len(
        reconciled.group_key_by_id
    )
    assert reconciled.executed_region_keys <= set(reconciled.region_key_by_id.values())
    assert reconciled.executed_group_keys <= set(reconciled.group_key_by_id.values())
    assert reconciled.next_region_key > 4
    assert reconciled.next_group_key > 2


def test_append_rejects_group_that_crosses_old_and_new_source_boundary() -> None:
    previous_source = "first()\nsecond()"
    current_source = f"{previous_source}\nthird()"
    previous_lineage = UnitLineage.create([], [])

    with pytest.raises(LineageAmbiguityError, match="boundary"):
        _reconcile(
            edit_kind="append_recovery",
            previous_source=previous_source,
            current_source=current_source,
            previous_regions=[],
            current_regions=[],
            previous_groups=[],
            current_groups=[_group_at("crossing", 2, 3, "second()\nthird()")],
            previous_lineage=previous_lineage,
            edit_start_line=3,
            edit_end_line=2,
            line_delta=1,
        )


def test_patch_preserves_exact_before_and_shifted_after_units_but_not_intersections() -> None:
    previous_source = "a = 1\nb = 2\nc = 3\nd = 4\ne = 5"
    current_source = "a = 1\nb = 2\nc = 3\ninserted = True\nd = 4\ne = 5"
    previous_regions = [
        _region_at("before", 1, 1, "a = 1"),
        _region_at("intersecting", 3, 3, "c = 3"),
        _region_at("after", 4, 4, "d = 4"),
    ]
    previous_groups = [
        _group_at("before_group", 1, 2, "a = 1\nb = 2"),
        _group_at("intersecting_group", 3, 3, "c = 3"),
        _group_at("after_group", 4, 5, "d = 4\ne = 5"),
    ]
    current_regions = [
        _region_at("current_before", 1, 1, "a = 1"),
        _region_at("current_intersecting", 3, 3, "c = 3"),
        _region_at("current_after", 5, 5, "d = 4"),
    ]
    current_groups = [
        _group_at("current_before_group", 1, 2, "a = 1\nb = 2"),
        _group_at("current_intersecting_group", 3, 3, "c = 3"),
        _group_at("current_after_group", 5, 6, "d = 4\ne = 5"),
    ]
    previous_lineage = UnitLineage.create(previous_regions, previous_groups)

    reconciled = _reconcile(
        edit_kind="patch_region",
        previous_source=previous_source,
        current_source=current_source,
        previous_regions=previous_regions,
        current_regions=current_regions,
        previous_groups=previous_groups,
        current_groups=current_groups,
        previous_lineage=previous_lineage,
        edit_start_line=3,
        edit_end_line=3,
        line_delta=1,
    )

    assert reconciled.region_key_by_id["current_before"] == "region_key_000001"
    assert reconciled.region_key_by_id["current_after"] == "region_key_000003"
    assert reconciled.region_key_by_id["current_intersecting"] == "region_key_000004"
    assert reconciled.group_key_by_id["current_before_group"] == "group_key_000001"
    assert reconciled.group_key_by_id["current_after_group"] == "group_key_000003"
    assert reconciled.group_key_by_id["current_intersecting_group"] == "group_key_000004"


def test_reconcile_rejects_multiple_exact_candidates_without_mutating_previous_lineage() -> None:
    previous_regions = [_region_at("old", 1, 1, "same = True")]
    previous_lineage = UnitLineage.create(previous_regions, [])
    previous_lineage.executed_region_keys.add(previous_lineage.region_key_by_id["old"])
    before = _lineage_snapshot(previous_lineage)

    with pytest.raises(LineageAmbiguityError, match="multiple"):
        _reconcile(
            edit_kind="patch_region",
            previous_source="same = True\npatch = 1",
            current_source="same = True\npatch = 2",
            previous_regions=previous_regions,
            current_regions=[
                _region_at("candidate_a", 1, 1, "same = True"),
                _region_at("candidate_b", 1, 1, "same = True"),
            ],
            previous_groups=[],
            current_groups=[],
            previous_lineage=previous_lineage,
            edit_start_line=2,
            edit_end_line=2,
            line_delta=0,
        )

    assert _lineage_snapshot(previous_lineage) == before


def test_reconcile_rejects_executed_key_missing_from_previous_map() -> None:
    previous_lineage = UnitLineage.create([], [])
    previous_lineage.executed_region_keys.add("region_key_999999")

    with pytest.raises(LineageAmbiguityError, match="executed region key"):
        _reconcile(
            edit_kind="append_recovery",
            previous_source="x = 1",
            current_source="x = 1\ny = 2",
            previous_regions=[],
            current_regions=[],
            previous_groups=[],
            current_groups=[],
            previous_lineage=previous_lineage,
            edit_start_line=2,
            edit_end_line=1,
            line_delta=1,
        )


def test_reconcile_rejects_executed_old_unit_that_cannot_be_mapped() -> None:
    previous_groups = [_group_at("old_group", 1, 1, "acted()")]
    previous_lineage = UnitLineage.create([], previous_groups)
    previous_lineage.executed_group_keys.add(previous_lineage.group_key_by_id["old_group"])

    with pytest.raises(LineageAmbiguityError, match="executed group key"):
        _reconcile(
            edit_kind="patch_group",
            previous_source="acted()\nold = 1",
            current_source="changed()\nold = 1",
            previous_regions=[],
            current_regions=[],
            previous_groups=previous_groups,
            current_groups=[_group_at("changed_group", 1, 1, "changed()")],
            previous_lineage=previous_lineage,
            edit_start_line=2,
            edit_end_line=2,
            line_delta=0,
        )


def test_reconcile_rejects_executed_key_with_multiple_previous_units() -> None:
    previous_regions = [
        _region_at("old_a", 1, 1, "a = 1"),
        _region_at("old_b", 2, 2, "b = 2"),
    ]
    previous_lineage = UnitLineage.create(previous_regions, [])
    executed_key = previous_lineage.region_key_by_id["old_a"]
    previous_lineage.region_key_by_id["old_b"] = executed_key
    previous_lineage.executed_region_keys.add(executed_key)

    with pytest.raises(LineageAmbiguityError, match="executed region key"):
        _reconcile(
            edit_kind="append_recovery",
            previous_source="a = 1\nb = 2",
            current_source="a = 1\nb = 2\nc = 3",
            previous_regions=previous_regions,
            current_regions=[
                _region_at("current_a", 1, 1, "a = 1"),
                _region_at("current_b", 2, 2, "b = 2"),
            ],
            previous_groups=[],
            current_groups=[],
            previous_lineage=previous_lineage,
            edit_start_line=3,
            edit_end_line=2,
            line_delta=1,
        )


def test_append_rejects_changed_old_prefix_and_does_not_mutate_lineage() -> None:
    previous_regions = [_region_at("old", 1, 1, "x = 1")]
    previous_lineage = UnitLineage.create(previous_regions, [])
    before = _lineage_snapshot(previous_lineage)

    with pytest.raises(LineageAmbiguityError, match="prefix"):
        _reconcile(
            edit_kind="append_recovery",
            previous_source="x = 1",
            current_source="x = 2\ny = 3",
            previous_regions=previous_regions,
            current_regions=[
                _region_at("changed", 1, 1, "x = 2"),
                _region_at("new", 2, 2, "y = 3"),
            ],
            previous_groups=[],
            current_groups=[],
            previous_lineage=previous_lineage,
            edit_start_line=2,
            edit_end_line=1,
            line_delta=1,
        )

    assert _lineage_snapshot(previous_lineage) == before


def test_patch_does_not_fallback_to_same_source_at_wrong_span() -> None:
    previous_regions = [_region_at("old_after", 3, 3, "repeated = True")]
    previous_lineage = UnitLineage.create(previous_regions, [])

    reconciled = _reconcile(
        edit_kind="patch_region",
        previous_source="edit = 1\nother = 2\nrepeated = True",
        current_source="edit = 0\nother = 2\nblank = None\nrepeated = True",
        previous_regions=previous_regions,
        current_regions=[_region_at("wrong_span", 5, 5, "repeated = True")],
        previous_groups=[],
        current_groups=[],
        previous_lineage=previous_lineage,
        edit_start_line=1,
        edit_end_line=1,
        line_delta=1,
    )

    assert reconciled.region_key_by_id["wrong_span"] == "region_key_000002"


def test_reconcile_rejects_duplicate_current_temporary_ids() -> None:
    previous_lineage = UnitLineage.create([], [])

    with pytest.raises(LineageAmbiguityError, match="duplicate current region id"):
        _reconcile(
            edit_kind="append_recovery",
            previous_source="x = 1",
            current_source="x = 1\ny = 2",
            previous_regions=[],
            current_regions=[
                _region_at("duplicate", 1, 1, "x = 1"),
                _region_at("duplicate", 2, 2, "y = 2"),
            ],
            previous_groups=[],
            current_groups=[],
            previous_lineage=previous_lineage,
            edit_start_line=2,
            edit_end_line=1,
            line_delta=1,
        )


def test_reconcile_rejects_duplicate_previous_temporary_ids_without_mutation() -> None:
    previous_regions = [
        _region_at("duplicate", 1, 1, "x = 1"),
        _region_at("duplicate", 1, 1, "x = 1"),
    ]
    previous_lineage = UnitLineage(
        next_region_key=2,
        region_key_by_id={"duplicate": "region_key_000001"},
    )
    before = _lineage_snapshot(previous_lineage)

    with pytest.raises(LineageAmbiguityError, match="duplicate previous region id"):
        _reconcile(
            edit_kind="append_recovery",
            previous_source="x = 1",
            current_source="x = 1\ny = 2",
            previous_regions=previous_regions,
            current_regions=[_region_at("current", 1, 1, "x = 1")],
            previous_groups=[],
            current_groups=[],
            previous_lineage=previous_lineage,
            edit_start_line=2,
            edit_end_line=1,
            line_delta=1,
        )

    assert _lineage_snapshot(previous_lineage) == before


@pytest.mark.parametrize("map_shape", ["missing", "extra"])
def test_reconcile_rejects_previous_map_ids_that_do_not_match_units(map_shape: str) -> None:
    previous_regions = [_region_at("old", 1, 1, "x = 1")]
    previous_lineage = UnitLineage.create(previous_regions, [])
    if map_shape == "missing":
        previous_lineage.region_key_by_id.clear()
    else:
        previous_lineage.region_key_by_id["ghost"] = "region_key_000002"
        previous_lineage.next_region_key = 3
    before = _lineage_snapshot(previous_lineage)

    with pytest.raises(LineageAmbiguityError, match="previous region map ids"):
        _reconcile(
            edit_kind="append_recovery",
            previous_source="x = 1",
            current_source="x = 1\ny = 2",
            previous_regions=previous_regions,
            current_regions=[_region_at("current", 1, 1, "x = 1")],
            previous_groups=[],
            current_groups=[],
            previous_lineage=previous_lineage,
            edit_start_line=2,
            edit_end_line=1,
            line_delta=1,
        )

    assert _lineage_snapshot(previous_lineage) == before


def test_reconcile_rejects_duplicate_previous_stable_keys_without_mutation() -> None:
    previous_regions = [
        _region_at("old_a", 1, 1, "a = 1"),
        _region_at("old_b", 2, 2, "b = 2"),
    ]
    previous_lineage = UnitLineage.create(previous_regions, [])
    previous_lineage.region_key_by_id["old_b"] = previous_lineage.region_key_by_id["old_a"]
    before = _lineage_snapshot(previous_lineage)

    with pytest.raises(LineageAmbiguityError, match="previous region stable keys"):
        _reconcile(
            edit_kind="append_recovery",
            previous_source="a = 1\nb = 2",
            current_source="a = 1\nb = 2\nc = 3",
            previous_regions=previous_regions,
            current_regions=[
                _region_at("current_a", 1, 1, "a = 1"),
                _region_at("current_b", 2, 2, "b = 2"),
            ],
            previous_groups=[],
            current_groups=[],
            previous_lineage=previous_lineage,
            edit_start_line=3,
            edit_end_line=2,
            line_delta=1,
        )

    assert _lineage_snapshot(previous_lineage) == before


def test_reconcile_rejects_invalid_stable_key_format() -> None:
    previous_regions = [_region_at("old", 1, 1, "x = 1")]
    previous_lineage = UnitLineage(
        next_region_key=2,
        region_key_by_id={"old": "region-1"},
    )

    with pytest.raises(LineageAmbiguityError, match="region stable key format"):
        _reconcile(
            edit_kind="append_recovery",
            previous_source="x = 1",
            current_source="x = 1\ny = 2",
            previous_regions=previous_regions,
            current_regions=[_region_at("current", 1, 1, "x = 1")],
            previous_groups=[],
            current_groups=[],
            previous_lineage=previous_lineage,
            edit_start_line=2,
            edit_end_line=1,
            line_delta=1,
        )


@pytest.mark.parametrize(
    ("existing_key", "next_region_key"),
    [("region_key_000001", 1), ("region_key_000005", 2)],
)
def test_reconcile_rejects_counter_that_can_reuse_an_existing_key(
    existing_key: str,
    next_region_key: int,
) -> None:
    previous_regions = [_region_at("old", 1, 1, "x = 1")]
    previous_lineage = UnitLineage.create(previous_regions, [])
    previous_lineage.region_key_by_id["old"] = existing_key
    previous_lineage.next_region_key = next_region_key
    before = _lineage_snapshot(previous_lineage)

    with pytest.raises(LineageAmbiguityError, match="next region key"):
        _reconcile(
            edit_kind="append_recovery",
            previous_source="x = 1",
            current_source="x = 1\ny = 2",
            previous_regions=previous_regions,
            current_regions=[_region_at("current", 1, 1, "x = 1")],
            previous_groups=[],
            current_groups=[],
            previous_lineage=previous_lineage,
            edit_start_line=2,
            edit_end_line=1,
            line_delta=1,
        )

    assert _lineage_snapshot(previous_lineage) == before


def test_append_rejects_line_ending_rewrite_of_previous_source() -> None:
    with pytest.raises(LineageAmbiguityError, match="prefix"):
        _reconcile(
            edit_kind="append_recovery",
            previous_source="x = 1\r\n",
            current_source="x = 1\ny = 2",
            previous_regions=[],
            current_regions=[],
            previous_groups=[],
            current_groups=[],
            previous_lineage=UnitLineage.create([], []),
            edit_start_line=2,
            edit_end_line=1,
            line_delta=1,
        )


def test_reconcile_rejects_line_delta_that_disagrees_with_sources() -> None:
    with pytest.raises(LineageAmbiguityError, match="line_delta"):
        _reconcile(
            edit_kind="patch_region",
            previous_source="a = 1\nb = 2",
            current_source="a = 2\nb = 2",
            previous_regions=[],
            current_regions=[],
            previous_groups=[],
            current_groups=[],
            previous_lineage=UnitLineage.create([], []),
            edit_start_line=1,
            edit_end_line=1,
            line_delta=1,
        )


@pytest.mark.parametrize(
    ("current_source", "edit_start_line", "edit_end_line", "line_delta"),
    [
        ("x = 1\ny = 2", 1, 1, 1),
        ("x = 1", 2, 1, 0),
    ],
)
def test_append_rejects_invalid_edit_shape(
    current_source: str,
    edit_start_line: int,
    edit_end_line: int,
    line_delta: int,
) -> None:
    with pytest.raises(LineageAmbiguityError, match="append edit"):
        _reconcile(
            edit_kind="append_recovery",
            previous_source="x = 1",
            current_source=current_source,
            previous_regions=[],
            current_regions=[],
            previous_groups=[],
            current_groups=[],
            previous_lineage=UnitLineage.create([], []),
            edit_start_line=edit_start_line,
            edit_end_line=edit_end_line,
            line_delta=line_delta,
        )


@pytest.mark.parametrize(
    ("edit_start_line", "edit_end_line"),
    [(0, 1), (2, 1), (1, 3)],
)
def test_patch_rejects_span_outside_previous_source(
    edit_start_line: int,
    edit_end_line: int,
) -> None:
    with pytest.raises(LineageAmbiguityError, match="patch edit span"):
        _reconcile(
            edit_kind="patch_group",
            previous_source="a = 1\nb = 2",
            current_source="a = 2\nb = 2",
            previous_regions=[],
            current_regions=[],
            previous_groups=[],
            current_groups=[],
            previous_lineage=UnitLineage.create([], []),
            edit_start_line=edit_start_line,
            edit_end_line=edit_end_line,
            line_delta=0,
        )


def test_patch_allows_negative_line_delta_for_deletion() -> None:
    previous_regions = [
        _region_at("before", 1, 1, "a = 1"),
        _region_at("deleted", 2, 2, "b = 2"),
        _region_at("after", 3, 3, "c = 3"),
    ]
    previous_lineage = UnitLineage.create(previous_regions, [])

    reconciled = _reconcile(
        edit_kind="patch_region",
        previous_source="a = 1\nb = 2\nc = 3",
        current_source="a = 1\nc = 3",
        previous_regions=previous_regions,
        current_regions=[
            _region_at("current_before", 1, 1, "a = 1"),
            _region_at("current_after", 2, 2, "c = 3"),
        ],
        previous_groups=[],
        current_groups=[],
        previous_lineage=previous_lineage,
        edit_start_line=2,
        edit_end_line=2,
        line_delta=-1,
    )

    assert reconciled.region_key_by_id == {
        "current_before": "region_key_000001",
        "current_after": "region_key_000003",
    }


def test_patch_allows_zero_line_delta_replacement() -> None:
    previous_regions = [
        _region_at("before", 1, 1, "a = 1"),
        _region_at("replaced", 2, 2, "b = 2"),
        _region_at("after", 3, 3, "c = 3"),
    ]
    previous_lineage = UnitLineage.create(previous_regions, [])

    reconciled = _reconcile(
        edit_kind="patch_region",
        previous_source="a = 1\nb = 2\nc = 3",
        current_source="a = 1\nb = 9\nc = 3",
        previous_regions=previous_regions,
        current_regions=[
            _region_at("current_before", 1, 1, "a = 1"),
            _region_at("replacement", 2, 2, "b = 9"),
            _region_at("current_after", 3, 3, "c = 3"),
        ],
        previous_groups=[],
        current_groups=[],
        previous_lineage=previous_lineage,
        edit_start_line=2,
        edit_end_line=2,
        line_delta=0,
    )

    assert reconciled.region_key_by_id["current_before"] == "region_key_000001"
    assert reconciled.region_key_by_id["current_after"] == "region_key_000003"
    assert reconciled.region_key_by_id["replacement"] == "region_key_000004"


def test_reconcile_rejects_unsupported_edit_kind() -> None:
    with pytest.raises(ValueError, match="Unsupported lineage edit kind"):
        _reconcile(
            edit_kind="rewrite_everything",
            previous_source="",
            current_source="",
            previous_regions=[],
            current_regions=[],
            previous_groups=[],
            current_groups=[],
            previous_lineage=UnitLineage.create([], []),
            edit_start_line=1,
            edit_end_line=1,
            line_delta=0,
        )
