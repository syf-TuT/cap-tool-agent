from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.integrations.franka.control import FrankaControlApi
from capx.runtime_control.trace import RuntimeTrace


class FakeApi:
    def functions(self):
        return {"get_pose": self.get_pose}

    def get_pose(self, name):
        return {"name": name}


def test_capsule_globals_can_bind_traced_api_functions():
    env = object.__new__(CodeExecutionEnvBase)
    env.low_level_env = object()
    env._apis = {"fake": FakeApi()}

    trace = RuntimeTrace()
    globals_dict = env._build_capsule_globals(trace=trace)

    assert globals_dict["get_pose"]("cube") == {"name": "cube"}
    assert trace.events[0]["name"] == "get_pose"


def test_cube_lift_api_keeps_get_observation_recovery_contract():
    api = FrankaControlApi.__new__(FrankaControlApi)
    api.real = False

    assert api.recovery_observation_functions() == {"get_observation"}
