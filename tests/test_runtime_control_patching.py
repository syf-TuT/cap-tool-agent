from capx.runtime_control.patching import replace_region_source
from capx.runtime_control.segmenter import segment_python_code


def test_replace_region_source_only_changes_target_region():
    source = "x = 1\ny = x + 2\nprint(y)\n"
    regions = segment_python_code(source)

    patched = replace_region_source(source, regions[1], "y = x + 3")

    assert "y = x + 3" in patched
    assert "print(y)" in patched
