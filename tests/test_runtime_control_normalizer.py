from capx.runtime_control.normalizer import segment_python_code_groups
from capx.runtime_control.segmenter import segment_python_code


def test_groups_preserve_original_source_bytes():
    source = "\n".join(
        [
            "x = 1",
            "move_to(x)",
            "y = x + 1",
            "move_to(y)",
        ]
    )

    regions = segment_python_code(source)
    groups = segment_python_code_groups(
        source,
        regions,
        side_effect_calls={"move_to"},
    )

    assert "".join(group.source for group in groups) == source


def test_groups_partition_regions_without_gaps_or_reordering():
    source = "\n".join(
        [
            "x = 1",
            "move_to(x)",
            "y = x + 1",
            "move_to(y)",
        ]
    )
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
