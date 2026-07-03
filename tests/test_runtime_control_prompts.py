from capx.runtime_control.prompts import build_capsule_prompt, parse_runtime_action_response
from capx.runtime_control.schema import CodeRegion


def test_parse_runtime_action_response():
    action = parse_runtime_action_response(
        '{"action": "run_region", "args": {"region_id": "region_1"}}'
    )

    assert action.action == "run_region"


def test_capsule_prompt_excludes_robot_tool_list():
    prompt = build_capsule_prompt(
        task="stack cubes",
        regions=[CodeRegion(region_id="region_1", start_line=1, end_line=1, source="x = 1")],
        history=[],
        trace_summary={},
    )
    text = str(prompt)

    assert "run_region" in text
    assert "solve_ik" not in text
    assert "move_to_joints" not in text
