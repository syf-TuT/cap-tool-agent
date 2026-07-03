import json
from types import SimpleNamespace

from capx.envs.trial import _run_tool_trial
from capx.tools.schema import ToolResult, ToolSpec


class FakeToolEnv:
    def __init__(self):
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
        if tool_call.tool == "finish":
            self.completed = True
        return ToolResult(tool=tool_call.tool, status="success", output_summary=None)


def test_tool_trial_writes_trace_artifact(tmp_path):
    summary = _run_tool_trial(
        env=FakeToolEnv(),
        trial=1,
        args=SimpleNamespace(model="test"),
        config={
            "output_dir": str(tmp_path),
            "max_tool_steps": 3,
            "record_video": False,
            "use_img_differencing": False,
            "use_video_differencing": False,
        },
        scripted_tool_calls=[{"tool": "finish", "args": {}}],
    )

    trace_path = tmp_path / "tool_trace_trial_01.json"

    assert summary.success is True
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text())
    assert trace[0]["call"]["tool"] == "finish"
    assert trace[0]["feedback"]["status"] == "success"
    assert "Feedback latency" in summary.log
