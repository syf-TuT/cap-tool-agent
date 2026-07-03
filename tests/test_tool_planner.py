from capx.tools.planner import LlmToolPlanner, ScriptedToolPlanner
from capx.tools.schema import ToolCall


def test_scripted_tool_planner_returns_calls_in_order():
    planner = ScriptedToolPlanner([
        {"tool": "get_observation", "args": {}},
        {"tool": "finish", "args": {}},
    ])

    assert planner.next_call([], {}) == ToolCall(tool="get_observation", args={})
    assert planner.next_call([], {}).tool == "finish"


def test_llm_tool_planner_parses_query_result():
    calls = []

    def fake_query(args, prompt):
        calls.append(prompt)
        return {"content": '{"tool": "finish", "args": {}}', "reasoning": ""}

    planner = LlmToolPlanner(query_model=fake_query, args=object())

    call = planner.next_call(prompt=[{"role": "user", "content": [{"type": "text", "text": "x"}]}])

    assert call.tool == "finish"
    assert calls
