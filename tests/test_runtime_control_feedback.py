from capx.runtime_control.feedback import build_runtime_feedback
from capx.runtime_control.schema import CodeRegion, RuntimeAction, RuntimeEvent


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
