import pytest

from capx.runtime_control.feedback import build_runtime_feedback
from capx.runtime_control.schema import CodeRegion, CodeRegionGroup, RuntimeAction, RuntimeEvent


def _effect_group() -> CodeRegionGroup:
    return CodeRegionGroup(
        "group_1",
        1,
        2,
        "joints = solve_ik(target)\nmove_to_joints(joints)",
        region_ids=["region_1", "region_2"],
        primitive_calls=["solve_ik", "move_to_joints"],
        defined_names=["joints"],
        used_names=["solve_ik", "target", "move_to_joints"],
        has_robot_side_effect=True,
    )


def test_feedback_binds_failed_region_to_source_span():
    region = CodeRegion("region_2", 5, 7, "raise RuntimeError('bad')")
    action = RuntimeAction("run_region", {"region_id": "region_2"})
    event = RuntimeEvent(
        action="run_region",
        status="failed",
        region_id="region_2",
        message="bad",
        evidence={"exception_type": "RuntimeError"},
    )

    feedback = build_runtime_feedback(
        step_id=1,
        action=action,
        event=event,
        region=region,
        trace_events=[{"name": "goto_pose", "status": "failed"}],
        before_state={"reward": 0.0, "task_completed": False},
        after_state={"reward": 0.0, "task_completed": False},
    )

    assert feedback.status == "failed"
    assert feedback.region_id == "region_2"
    assert feedback.evidence["source_span"]["start_line"] == 5
    assert feedback.patch_scope == "region_2"


def test_feedback_warns_when_region_has_no_task_progress():
    region = CodeRegion("region_1", 1, 1, "x = 1")
    feedback = build_runtime_feedback(
        step_id=1,
        action=RuntimeAction("run_region", {"region_id": "region_1"}),
        event=RuntimeEvent(action="run_region", status="success", region_id="region_1"),
        region=region,
        trace_events=[],
        before_state={"reward": 0.0, "task_completed": False},
        after_state={"reward": 0.0, "task_completed": False},
    )

    assert feedback.status == "warning"


def test_feedback_allows_non_side_effect_group_without_reward_progress():
    group = CodeRegionGroup(
        "group_1",
        1,
        3,
        "obs = get_observation()\nmask = segment_sam3_text_prompt(obs, 'cube')",
        region_ids=["region_1", "region_2"],
        primitive_calls=["get_observation", "segment_sam3_text_prompt"],
        defined_names=["obs", "mask"],
        used_names=["get_observation", "segment_sam3_text_prompt"],
        has_robot_side_effect=False,
    )

    feedback = build_runtime_feedback(
        step_id=1,
        action=RuntimeAction("run_group", {"group_id": "group_1"}),
        event=RuntimeEvent(action="run_group", status="success", region_id="group_1"),
        region=group,
        trace_events=[],
        before_state={"reward": 0.0, "task_completed": False},
        after_state={"reward": 0.0, "task_completed": False},
    )

    assert feedback.status == "success"
    assert feedback.patch_scope is None


def test_feedback_warns_for_side_effect_group_without_reward_progress():
    group = _effect_group()

    feedback = build_runtime_feedback(
        step_id=1,
        action=RuntimeAction("run_group", {"group_id": "group_1"}),
        event=RuntimeEvent(action="run_group", status="success", region_id="group_1"),
        region=group,
        trace_events=[],
        before_state={"reward": 0.0, "task_completed": False},
        after_state={"reward": 0.0, "task_completed": False},
    )

    assert feedback.status == "warning"
    assert feedback.patch_scope == "group_1"
    assert any("No rollback is available" in hint for hint in feedback.repair_hints)
    assert any("fresh observation" in hint for hint in feedback.repair_hints)
    assert any("append_recovery" in hint for hint in feedback.repair_hints)


def test_feedback_sparse_terminal_keeps_successful_effect_without_terminal_progress():
    feedback = build_runtime_feedback(
        step_id=1,
        action=RuntimeAction("run_group", {"group_id": "group_1"}),
        event=RuntimeEvent(action="run_group", status="success", region_id="group_1"),
        region=_effect_group(),
        trace_events=[{"name": "goto_pose", "status": "success"}],
        before_state={"reward": 0.0, "task_completed": False},
        after_state={"reward": 0.0, "task_completed": False},
        progress_mode="sparse_terminal",
    )

    assert feedback.status == "success"
    assert feedback.evidence["progress_mode"] == "sparse_terminal"
    assert feedback.evidence["terminal_progress_unverified"] is True
    assert "no local task progress" not in feedback.message


def test_feedback_sparse_terminal_rejects_unknown_progress_mode():
    with pytest.raises(ValueError, match="progress_mode"):
        build_runtime_feedback(
            step_id=1,
            action=RuntimeAction("run_group", {"group_id": "group_1"}),
            event=RuntimeEvent(action="run_group", status="success", region_id="group_1"),
            region=_effect_group(),
            trace_events=[],
            before_state={"reward": 0.0, "task_completed": False},
            after_state={"reward": 0.0, "task_completed": False},
            progress_mode="unknown",
        )


def test_feedback_sparse_terminal_dense_default_still_warns_without_progress():
    feedback = build_runtime_feedback(
        step_id=1,
        action=RuntimeAction("run_group", {"group_id": "group_1"}),
        event=RuntimeEvent(action="run_group", status="success", region_id="group_1"),
        region=_effect_group(),
        trace_events=[{"name": "goto_pose", "status": "success"}],
        before_state={"reward": 0.0, "task_completed": False},
        after_state={"reward": 0.0, "task_completed": False},
    )

    assert feedback.status == "warning"
    assert "no local task progress" in feedback.message


def test_feedback_keeps_source_patch_successful_without_reward_progress():
    group = CodeRegionGroup(
        "group_1",
        1,
        2,
        "joints = solve_ik(target)\nmove_to_joints(joints)",
        region_ids=["region_1", "region_2"],
        primitive_calls=["solve_ik", "move_to_joints"],
        defined_names=["joints"],
        used_names=["solve_ik", "target", "move_to_joints"],
        has_robot_side_effect=True,
    )

    feedback = build_runtime_feedback(
        step_id=1,
        action=RuntimeAction("patch_group", {"group_id": "group_1", "source": "x = 1"}),
        event=RuntimeEvent(action="patch_group", status="success", region_id="group_1"),
        region=group,
        trace_events=[],
        before_state={"reward": 0.0, "task_completed": False},
        after_state={"reward": 0.0, "task_completed": False},
    )

    assert feedback.status == "success"
    assert feedback.patch_scope is None
    assert "executed" not in feedback.message
    assert "no local task progress" not in feedback.message


def test_feedback_name_error_hint_mentions_missing_variable():
    group = CodeRegionGroup(
        "group_1",
        10,
        12,
        "move_to_joints(pre_joints)",
        region_ids=["region_10"],
        primitive_calls=["move_to_joints"],
        used_names=["move_to_joints", "pre_joints"],
        has_robot_side_effect=True,
    )

    feedback = build_runtime_feedback(
        step_id=1,
        action=RuntimeAction("run_group", {"group_id": "group_1"}),
        event=RuntimeEvent(
            action="run_group",
            status="failed",
            region_id="group_1",
            message="name 'pre_joints' is not defined",
            evidence={"exception_type": "NameError"},
        ),
        region=group,
        trace_events=[],
        before_state={"reward": 0.0, "task_completed": False},
        after_state={"reward": 0.0, "task_completed": False},
    )

    assert feedback.patch_scope == "group_1"
    assert any("pre_joints" in hint for hint in feedback.repair_hints)
