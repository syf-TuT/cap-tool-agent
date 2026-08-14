from dataclasses import FrozenInstanceError

import pytest

from capx.runtime_control import LineageAmbiguityError, SourceRevision, UnitLineage
from capx.runtime_control.schema import CodeRegion, CodeRegionGroup


def _region(region_id: str, source: str) -> CodeRegion:
    return CodeRegion(region_id=region_id, start_line=1, end_line=1, source=source)


def _group(group_id: str, source: str) -> CodeRegionGroup:
    return CodeRegionGroup(group_id=group_id, start_line=1, end_line=1, source=source)


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
