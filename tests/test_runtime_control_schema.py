import pytest

from capx.runtime_control import PostActionObservation
from capx.runtime_control.schema import (
    CodeRegion,
    CodeRegionGroup,
    RuntimeAction,
    RuntimeEvent,
)


def test_code_region_exports_source_span():
    region = CodeRegion(region_id="region_1", start_line=2, end_line=4, source="x = 1")

    assert region.to_dict()["region_id"] == "region_1"
    assert region.to_dict()["source_span"] == {"start_line": 2, "end_line": 4}


def test_code_region_group_exports_execution_metadata():
    group = CodeRegionGroup(
        group_id="group_1",
        start_line=2,
        end_line=5,
        source="x = 1\nmove_to_joints(x)",
        region_ids=["region_2", "region_3"],
        primitive_calls=["move_to_joints"],
        defined_names=["x"],
        used_names=["move_to_joints"],
        has_robot_side_effect=True,
    )

    data = group.to_dict()

    assert data["group_id"] == "group_1"
    assert data["source_span"] == {"start_line": 2, "end_line": 5}
    assert data["region_ids"] == ["region_2", "region_3"]
    assert data["primitive_calls"] == ["move_to_joints"]
    assert data["defined_names"] == ["x"]
    assert data["used_names"] == ["move_to_joints"]
    assert data["has_robot_side_effect"] is True


def test_runtime_action_validates_args_mapping():
    action = RuntimeAction.from_mapping(
        {"action": "run_region", "args": {"region_id": "region_1"}}
    )

    assert action.action == "run_region"
    assert action.args["region_id"] == "region_1"


def test_runtime_action_accepts_group_actions():
    run_action = RuntimeAction.from_mapping(
        {"action": "run_group", "args": {"group_id": "group_1"}}
    )
    patch_action = RuntimeAction.from_mapping(
        {"action": "patch_group", "args": {"group_id": "group_1", "source": "x = 2"}}
    )

    assert run_action.action == "run_group"
    assert patch_action.action == "patch_group"


def test_runtime_action_accepts_append_recovery():
    action = RuntimeAction.from_mapping(
        {"action": "append_recovery", "args": {"source": "obs = get_observation()"}}
    )

    assert action.action == "append_recovery"
    assert action.args["source"] == "obs = get_observation()"


def test_runtime_action_rejects_rollback_action():
    with pytest.raises(ValueError, match="Unsupported runtime action"):
        RuntimeAction.from_mapping({"action": "rollback_to_checkpoint", "args": {}})


def test_runtime_action_rejects_removed_trace_inspection_action():
    with pytest.raises(ValueError, match="Unsupported runtime action"):
        RuntimeAction.from_mapping({"action": "inspect_" "trace", "args": {}})


def test_runtime_event_is_jsonable():
    event = RuntimeEvent(
        action="run_region",
        status="failed",
        region_id="region_2",
        message="boom",
        evidence={"exception_type": "ValueError"},
    )

    assert event.to_dict()["status"] == "failed"


def test_post_action_observation_exports_stable_audit_fields():
    observation = PostActionObservation(
        step_id=3,
        action="run_group",
        unit_id="group_2",
        unit_key="group_key_000002",
        event_status="success",
        state_before={"reward": 0.0},
        state_after={"reward": 0.0},
        reward_before=0.0,
        reward_after=0.0,
        task_completed=False,
        new_trace_events=[{"name": "move_to", "status": "success"}],
        trace_revision=7,
        source_revision=2,
        terminal_progress_unverified=True,
    )

    assert observation.to_dict() == {
        "step_id": 3,
        "action": "run_group",
        "unit_id": "group_2",
        "unit_key": "group_key_000002",
        "event_status": "success",
        "state_before": {"reward": 0.0},
        "state_after": {"reward": 0.0},
        "reward_before": 0.0,
        "reward_after": 0.0,
        "task_completed": False,
        "new_trace_events": [{"name": "move_to", "status": "success"}],
        "trace_revision": 7,
        "source_revision": 2,
        "terminal_progress_unverified": True,
        "safety_failure": None,
    }
