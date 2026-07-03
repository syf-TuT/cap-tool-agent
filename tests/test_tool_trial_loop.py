import json
from types import SimpleNamespace

import capx.envs.trial as trial_module
from capx.envs.trial import _run_tool_trial
from capx.tools.schema import ToolResult, ToolSpec


class FakeToolEnv:
    oracle_code = ""

    def __init__(self):
        self.calls = []
        self.completed = False

    def reset(self, *, seed=None, options=None):
        return {
            "full_prompt": [
                {"role": "system", "content": "x"},
                {"role": "user", "content": [{"type": "text", "text": "task"}]},
            ]
        }, {}

    def tool_specs(self):
        return [ToolSpec(name="finish", description="Finish")]

    def tool_state_summary(self):
        return {}

    def snapshot_state(self):
        return {"reward": 1.0 if self.completed else 0.0, "task_completed": self.completed}

    def call_tool(self, tool_call):
        self.calls.append(tool_call.tool)
        if tool_call.tool == "finish":
            self.completed = True
        return ToolResult(tool=tool_call.tool, status="success", output_summary=None)


def test_tool_trial_loop_finishes_with_scripted_planner(tmp_path):
    env = FakeToolEnv()
    args = SimpleNamespace(model="test")
    config = {
        "output_dir": str(tmp_path),
        "max_tool_steps": 3,
        "record_video": False,
        "use_img_differencing": False,
        "use_video_differencing": False,
    }

    summary = _run_tool_trial(
        env=env,
        trial=1,
        args=args,
        config=config,
        scripted_tool_calls=[{"tool": "finish", "args": {}}],
    )

    assert summary.success is True
    assert summary.task_completed is True
    assert env.calls == ["finish"]


def test_tool_trial_loop_records_invalid_llm_tool_response_and_continues(tmp_path, monkeypatch):
    env = FakeToolEnv()
    responses = [
        {"content": "not json", "reasoning": ""},
        {"content": '{"tool": "finish", "args": {}}', "reasoning": ""},
    ]

    def fake_query(args, prompt):
        return responses.pop(0)

    monkeypatch.setattr(trial_module, "_query_model", fake_query)

    summary = _run_tool_trial(
        env=env,
        trial=1,
        args=SimpleNamespace(model="test"),
        config={
            "output_dir": str(tmp_path),
            "max_tool_steps": 2,
            "record_video": False,
            "use_img_differencing": False,
            "use_video_differencing": False,
        },
    )

    assert summary.success is True
    assert env.calls == ["finish"]
    assert "First failure step: 1" in summary.log

    trace = json.loads((tmp_path / "tool_trace_trial_01.json").read_text())
    assert trace[0]["call"]["tool"] == "__invalid_tool_call__"
    assert trace[0]["result"]["failure_type"] == "invalid_tool_call_response"
