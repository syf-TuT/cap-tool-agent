import numpy as np
import pytest

from capx.envs.tasks.base import CodeExecutionEnvBase


class _CompletionEnv:
    viser_debug = False
    _sim_step_count = 0
    max_steps = 100

    def __init__(self, task_completed):
        self._task_completed = task_completed

    def task_completed(self):
        return self._task_completed


@pytest.mark.parametrize(
    ("raw_task_completed", "expected"),
    [
        (np.bool_(False), False),
        (np.bool_(True), True),
        (None, None),
    ],
)
def test_step_normalizes_task_completion(raw_task_completed, expected):
    env = object.__new__(CodeExecutionEnvBase)
    env._step_count = 0
    env._task_prompt = "test"
    env.low_level_env = _CompletionEnv(raw_task_completed)
    env._exec_user_code = lambda action: {
        "ok": True,
        "stdout": "",
        "stderr": "",
        "result": None,
    }
    env._get_observation = lambda: {}
    env.compute_reward = lambda: 0.0

    _, _, _, _, info = env.step("pass")

    assert info["task_completed"] is expected
