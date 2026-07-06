from capx.runtime_control.normalizer import segment_python_code_groups
from capx.runtime_control.segmenter import segment_python_code


def _source_with_spacing():
    return "\n".join(
        [
            "x = 1",
            "",
            "# first motion",
            "move_to(x)",
            "",
            "# compute follow-up target",
            "y = x + 1",
            "",
            "# second motion",
            "move_to(y)",
            "",
        ]
    )


def _assert_group_sources_match_member_regions(groups, regions):
    region_by_id = {region.region_id: region for region in regions}

    for group in groups:
        assert group.source == "".join(
            region_by_id[region_id].source for region_id in group.region_ids
        )


def test_groups_preserve_original_source_bytes():
    source = _source_with_spacing()

    regions = segment_python_code(source)
    groups = segment_python_code_groups(
        source,
        regions,
        side_effect_calls={"move_to"},
    )

    _assert_group_sources_match_member_regions(groups, regions)
    assert "".join(group.source for group in groups) == source


def test_groups_partition_regions_without_gaps_or_reordering():
    source = _source_with_spacing()

    regions = segment_python_code(source)
    groups = segment_python_code_groups(
        source,
        regions,
        side_effect_calls={"move_to"},
    )

    grouped_region_ids = [
        region_id
        for group in groups
        for region_id in group.region_ids
    ]

    assert grouped_region_ids == [region.region_id for region in regions]
    assert len(grouped_region_ids) == len(set(grouped_region_ids))
