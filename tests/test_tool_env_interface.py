from gymnasium import Env

from capx.envs.tasks.base import CodeExecEnvConfig, CodeExecutionEnvBase
from capx.tools.schema import ToolCall


class FakeLowLevelEnv(Env):
    def reset(self, *, seed=None, options=None):
        return {"value": 1}, {}

    def get_observation(self):
        return {"value": 1}

    def compute_reward(self):
        return 0.0

    def task_completed(self):
        return False


class FakeApi:
    def __init__(self, env):
        self.env = env

    def functions(self):
        return {"add": self.add}

    def combined_doc(self):
        return "add(x, y)"

    def add(self, x: int, y: int = 1) -> int:
        """Add values."""
        return x + y


def test_code_env_exposes_tool_specs_and_call(monkeypatch):
    monkeypatch.setattr(
        "capx.envs.tasks.base.get_api",
        lambda name: (lambda env: FakeApi(env)),
    )
    env = CodeExecutionEnvBase(
        CodeExecEnvConfig(low_level=FakeLowLevelEnv(), apis=["FakeApi"], prompt="test")
    )

    specs = env.tool_specs()
    result = env.call_tool(ToolCall(tool="add", args={"x": 2, "y": 3}))

    assert [spec.name for spec in specs] == ["add"]
    assert result.status == "success"
    assert result.output_summary == 5


def test_code_env_snapshot_includes_reward_and_task_completed(monkeypatch):
    monkeypatch.setattr(
        "capx.envs.tasks.base.get_api",
        lambda name: (lambda env: FakeApi(env)),
    )
    env = CodeExecutionEnvBase(
        CodeExecEnvConfig(low_level=FakeLowLevelEnv(), apis=["FakeApi"], prompt="test")
    )

    snapshot = env.snapshot_state()

    assert snapshot["reward"] == 0.0
    assert snapshot["task_completed"] is False
