from capx.runtime_control.segmenter import (
    analyze_python_regions,
    segment_python_code,
)


def test_segmenter_regions_partition_source_bytes():
    source = "x = 1\nmove_to(x)\ny = x + 1"

    regions = segment_python_code(source)

    assert "".join(region.source for region in regions) == source


def test_segmenter_coalesces_same_line_top_level_statements():
    source = "x = 1; y = 2\n"

    regions = segment_python_code(source)

    assert len(regions) == 1
    assert regions[0].source == source
    assert "".join(region.source for region in regions) == source
    assert all(region.start_line <= region.end_line for region in regions)


def test_segmenter_includes_decorator_lines_in_decorated_regions():
    source = (
        "x = 1\n"
        "@decorate\n"
        "def fn():\n"
        "    return x\n"
        "y = 2\n"
        "@class_decorator\n"
        "class Thing:\n"
        "    pass\n"
    )

    regions = segment_python_code(source)

    assert regions[0].source == "x = 1\n"
    assert regions[1].source.startswith("@decorate\ndef fn():\n")
    assert regions[2].source == "y = 2\n"
    assert regions[3].source.startswith("@class_decorator\nclass Thing:\n")


def test_segmenter_preserves_comments_blank_lines_and_trailing_blank_lines():
    source = (
        "# leading comment\n"
        "x = 1\n"
        "\n"
        "# between regions\n"
        "move_to(x)\n"
        "\n"
        "\n"
    )

    regions = segment_python_code(source)

    assert "".join(region.source for region in regions) == source


def test_segmenter_exposes_region_analysis_facts():
    source = "x = 1\nmove_to(x)\n"
    regions = segment_python_code(source)

    analyses = analyze_python_regions(
        source,
        regions,
        side_effect_calls={"move_to"},
    )

    assert [analysis.region_id for analysis in analyses] == ["region_1", "region_2"]
    assert analyses[0].defined_names == ["x"]
    assert analyses[1].primitive_calls == ["move_to"]
    assert analyses[1].has_robot_side_effect is True


def test_segmenter_splits_top_level_statements_into_regions():
    source = "import numpy as np\nx = 1\ny = x + 2\nprint(y)\n"

    regions = segment_python_code(source)

    assert [region.region_id for region in regions] == [
        "region_1",
        "region_2",
        "region_3",
        "region_4",
    ]
    assert regions[0].source == "import numpy as np\n"
    assert regions[-1].start_line == 4


def test_segmenter_keeps_compound_statement_together():
    source = "if True:\n    x = 1\n    y = 2\nprint(x)\n"

    regions = segment_python_code(source)

    assert len(regions) == 2
    assert "y = 2" in regions[0].source
