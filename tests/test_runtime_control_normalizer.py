import pytest

from capx.runtime_control.normalizer import (
    GroupingPolicy,
    normalize_python_code_groups,
    segment_python_code_groups,
)
from capx.runtime_control.segmenter import analyze_python_regions, segment_python_code


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


def test_normalizer_rejects_analysis_length_mismatch():
    source = "x = 1\nmove_to(x)\n"
    regions = segment_python_code(source)
    analyses = analyze_python_regions(source, regions, side_effect_calls={"move_to"})

    with pytest.raises(ValueError, match="same number"):
        normalize_python_code_groups(
            source,
            regions,
            analyses[:-1],
            policy=GroupingPolicy(),
        )


def test_normalizer_rejects_analysis_region_id_mismatch():
    source = "x = 1\nmove_to(x)\n"
    regions = segment_python_code(source)
    analyses = analyze_python_regions(source, regions, side_effect_calls={"move_to"})

    with pytest.raises(ValueError, match="region_id"):
        normalize_python_code_groups(
            source,
            regions,
            list(reversed(analyses)),
            policy=GroupingPolicy(),
        )


def test_normalizer_rejects_source_that_does_not_match_regions():
    source = "x = 1\nmove_to(x)\n"
    regions = segment_python_code(source)
    analyses = analyze_python_regions(source, regions, side_effect_calls={"move_to"})

    with pytest.raises(ValueError, match="source"):
        normalize_python_code_groups(
            "stale = True\n",
            regions,
            analyses,
            policy=GroupingPolicy(),
        )


def test_normalizer_merges_consecutive_effects_into_sense_act_block():
    """A group spans sense/compute plus one or more consecutive effect calls."""
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

    assert [group.group_id for group in groups] == ["group_1", "group_2"]
    # sense/compute + BOTH consecutive effects stay in one block
    assert groups[0].region_ids == [
        "region_1",
        "region_2",
        "region_3",
        "region_4",
        "region_5",
        "region_6",
    ]
    assert "move_to_joints(pre_joints)" in groups[0].source
    assert "close_gripper()" in groups[0].source
    assert groups[0].has_robot_side_effect is True
    # return-to-sense (solve_ik) after an effect opens a new block
    assert groups[1].region_ids == ["region_7", "region_8"]


def test_normalizer_keeps_non_side_effect_setup_in_one_group():
    groups = segment_python_code_groups("x = 1\ny = x + 2\nz = y + 3\n")

    assert len(groups) == 1
    assert groups[0].source == "x = 1\ny = x + 2\nz = y + 3\n"
    assert groups[0].defined_names == ["x", "y", "z"]
    assert groups[0].used_names == ["x", "y"]
    assert groups[0].has_robot_side_effect is False


def test_normalizer_splits_cube_stack_program_into_task_phases():
    """Realistic CaP program collapses to ~4 sense->act phases, not per-move regions."""
    source = "\n".join(
        [
            "import numpy as np",
            "red_pose = get_object_pose('red cube', return_bbox_extent=True)",
            "red_pos, red_quat, red_extent = red_pose",
            "green_pose = get_object_pose('green cube', return_bbox_extent=True)",
            "green_pos, green_quat, green_extent = green_pose",
            "grasp_pos, grasp_quat = sample_grasp_pose('red cube')",
            "orient = grasp_quat",
            "open_gripper()",
            "goto_pose(grasp_pos, orient, z_approach=0.1)",
            "close_gripper()",
            "lift_z = grasp_pos[2] + 0.3",
            "lift_pos = np.array([grasp_pos[0], grasp_pos[1], lift_z])",
            "goto_pose(lift_pos, orient, z_approach=0.0)",
            "place_z = green_pos[2] + green_extent[2] / 2 + red_extent[2] / 2",
            "place_pos = np.array([green_pos[0], green_pos[1], place_z])",
            "goto_pose(place_pos, orient, z_approach=0.1)",
            "open_gripper()",
            "retreat_pos = np.array([place_pos[0], place_pos[1], place_pos[2] + 0.1])",
            "goto_pose(retreat_pos, orient, z_approach=0.0)",
        ]
    )

    groups = segment_python_code_groups(source)

    assert len(groups) == 4
    assert groups[0].region_ids == [f"region_{i}" for i in range(1, 11)]
    assert groups[1].region_ids == ["region_11", "region_12", "region_13"]
    assert groups[2].region_ids == ["region_14", "region_15", "region_16", "region_17"]
    assert groups[3].region_ids == ["region_18", "region_19"]


def test_normalizer_is_domain_agnostic_without_robot_primitives():
    """sense->act boundaries work on any code; no robot vocabulary required."""
    source = "\n".join(
        [
            "data = load_rows()",
            "cleaned = normalize(data)",
            "save(cleaned)",
            "report = summarize(cleaned)",
            "publish(report)",
        ]
    )

    groups = segment_python_code_groups(source)

    assert [group.region_ids for group in groups] == [
        ["region_1", "region_2", "region_3"],
        ["region_4", "region_5"],
    ]
    assert all(group.has_robot_side_effect is False for group in groups)


def test_normalizer_cap_fallback_allows_long_setup_block():
    """Default fallback cap is loose (20), so long compute setup stays one block."""
    source = "\n".join(f"v{i} = {i}" for i in range(8))

    groups = segment_python_code_groups(source)

    assert len(groups) == 1


def test_normalizer_marks_injected_primitive_as_robot_side_effect():
    """The environment declares its own effect primitives; the normalizer marks
    side effects from that injected set rather than a hardcoded vocabulary.
    Regression: goto_pose was the most common effect primitive in real programs
    but absent from the hardcoded list, so no-rollback protection failed for it."""
    groups = segment_python_code_groups(
        "p = get_pose('a')\ngoto_pose(p)\n",
        side_effect_calls={"goto_pose"},
    )

    assert groups[0].has_robot_side_effect is True


def test_normalizer_ignores_effect_primitive_outside_injected_set():
    """A bare-call effect that is not in the injected primitive set is a
    structural boundary but not a rollback-protected robot side effect."""
    groups = segment_python_code_groups(
        "p = observe()\npick_object(p)\n",
        side_effect_calls={"goto_pose"},
    )

    assert groups[0].has_robot_side_effect is False


def test_normalizer_marks_arbitrary_injected_primitive_name():
    """No hardcoded list is consulted: any name the environment declares works."""
    groups = segment_python_code_groups(
        "p = observe()\npick_object(p)\n",
        side_effect_calls={"pick_object"},
    )

    assert groups[0].has_robot_side_effect is True
