from capx.tools.prompts import build_tool_planner_prompt, parse_tool_call_response
from capx.tools.schema import ToolCall, ToolSpec


def test_parse_tool_call_response_accepts_json_object():
    call = parse_tool_call_response('{"thought": "look", "tool": "get_observation", "args": {}}')

    assert isinstance(call, ToolCall)
    assert call.tool == "get_observation"


def test_parse_tool_call_response_rejects_python_code():
    try:
        parse_tool_call_response("```python\nmove_to_joints(joints)\n```")
    except ValueError as exc:
        assert "JSON" in str(exc)
    else:
        raise AssertionError("Python code should be rejected")


def test_prompt_contains_tools_and_last_feedback():
    prompt = build_tool_planner_prompt(
        task="stack red cube on green cube",
        tool_specs=[ToolSpec(name="get_observation", description="Capture obs")],
        state_summary={"reward": 0.0},
        history=[{"feedback": {"status": "failed", "failure_type": "low_confidence_mask"}}],
    )

    text = prompt[-1]["content"][0]["text"]

    assert "Do not write Python code" in text
    assert "get_observation" in text
    assert "low_confidence_mask" in text


def test_prompt_explains_how_to_reference_prior_outputs():
    prompt = build_tool_planner_prompt(
        task="move using prior IK",
        tool_specs=[
            ToolSpec(name="solve_ik", description="Solve IK"),
            ToolSpec(name="move_to_joints", description="Move"),
        ],
        state_summary={"solve_ik.0": {"type": "ndarray", "shape": [7]}},
        history=[],
    )

    text = prompt[-1]["content"][0]["text"]

    assert '{"state_ref": "solve_ik.0"}' in text
    assert "Use output_ref values exactly as shown" in text


def test_prompt_explains_nested_observation_refs():
    prompt = build_tool_planner_prompt(
        task="segment red cube",
        tool_specs=[ToolSpec(name="segment_sam3_text_prompt", description="Segment")],
        state_summary={
            "get_observation.0": {
                "type": "dict",
                "nested_refs": {
                    "robot0_robotview.images.rgb": {
                        "ref": "get_observation.0.robot0_robotview.images.rgb"
                    }
                },
            }
        },
        history=[],
    )

    text = prompt[-1]["content"][0]["text"]

    assert '{"state_ref": "get_observation.0.robot0_robotview.images.rgb"}' in text
    assert "nested_refs" in text
