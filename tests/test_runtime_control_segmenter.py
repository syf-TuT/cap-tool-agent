from capx.runtime_control.segmenter import segment_python_code


def test_segmenter_groups_top_level_statements():
    source = "import numpy as np\nx = 1\ny = x + 2\nprint(y)\n"

    regions = segment_python_code(source)

    assert [region.region_id for region in regions] == [
        "region_1",
        "region_2",
        "region_3",
        "region_4",
    ]
    assert regions[0].source == "import numpy as np"
    assert regions[-1].start_line == 4


def test_segmenter_keeps_compound_statement_together():
    source = "if True:\n    x = 1\n    y = 2\nprint(x)\n"

    regions = segment_python_code(source)

    assert len(regions) == 2
    assert "y = 2" in regions[0].source
