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
