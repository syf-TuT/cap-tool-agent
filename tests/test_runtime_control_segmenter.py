from capx.runtime_control.segmenter import segment_python_code, segment_python_code_groups


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


def test_segmenter_groups_setup_with_first_robot_side_effect():
    source = "\n".join(
        [
            "obs = get_observation()",
            "red_mask = segment_sam3_text_prompt(obs, 'red cube')",
            "grasp = plan_grasp(red_mask)",
            "pre_joints = solve_ik(grasp)",
            "move_to_joints(pre_joints)",
            "close_gripper()",
            "lift_joints = solve_ik(grasp)",
            "move_to_joints(lift_joints)",
        ]
    )

    groups = segment_python_code_groups(source)

    assert [group.group_id for group in groups] == ["group_1", "group_2", "group_3"]
    assert groups[0].region_ids == [
        "region_1",
        "region_2",
        "region_3",
        "region_4",
        "region_5",
    ]
    assert "move_to_joints(pre_joints)" in groups[0].source
    assert groups[0].has_robot_side_effect is True
    assert groups[0].primitive_calls == [
        "get_observation",
        "segment_sam3_text_prompt",
        "plan_grasp",
        "solve_ik",
        "move_to_joints",
    ]
    assert groups[1].source == "close_gripper()"
    assert groups[2].region_ids == ["region_7", "region_8"]


def test_segmenter_keeps_non_side_effect_setup_in_one_group():
    groups = segment_python_code_groups("x = 1\ny = x + 2\nz = y + 3\n")

    assert len(groups) == 1
    assert groups[0].source == "x = 1\ny = x + 2\nz = y + 3"
    assert groups[0].defined_names == ["x", "y", "z"]
    assert groups[0].used_names == ["x", "y"]
    assert groups[0].has_robot_side_effect is False
