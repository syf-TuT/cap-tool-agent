from capx.tools.schema import StepFeedback, ToolCall, ToolResult, ToolSpec


def test_tool_call_parses_mapping():
    call = ToolCall.from_mapping({"tool": "solve_ik", "args": {"x": 1}, "thought": "try ik"})

    assert call.tool == "solve_ik"
    assert call.args == {"x": 1}
    assert call.thought == "try ik"


def test_tool_spec_exports_prompt_dict():
    spec = ToolSpec(
        name="solve_ik",
        description="Solve IK",
        input_schema={"position": "array[3]"},
        tags=["planning"],
        preconditions=["target_pose_available"],
        postconditions=["joint_solution_valid"],
        failure_modes=["unreachable_pose"],
    )

    prompt_dict = spec.to_prompt_dict()

    assert prompt_dict["name"] == "solve_ik"
    assert prompt_dict["input_schema"] == {"position": "array[3]"}
    assert prompt_dict["failure_modes"] == ["unreachable_pose"]


def test_tool_result_and_feedback_are_jsonable():
    result = ToolResult.failed(
        tool="solve_ik",
        failure_type="exception",
        message="bad pose",
        exception_type="ValueError",
    )
    feedback = StepFeedback(
        step_id=3,
        tool="solve_ik",
        status="failed",
        failure_stage="planning",
        failure_type="exception",
        evidence={"message": result.message},
        repair_hints=["choose another target"],
        recommended_next_tools=["solve_ik"],
    )

    assert result.to_dict()["status"] == "failed"
    assert feedback.to_dict()["repair_hints"] == ["choose another target"]
