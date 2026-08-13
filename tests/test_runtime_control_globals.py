from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.integrations.franka.control import FrankaControlApi
from capx.integrations.franka.handover import FrankaHandoverApi
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


def test_public_capsule_globals_exclude_internal_handles_and_opaque_trace_callable():
    env = object.__new__(CodeExecutionEnvBase)
    env.low_level_env = object()
    env._apis = {"fake": FakeApi()}

    trace = RuntimeTrace()
    globals_dict = env._build_capsule_globals(
        trace=trace,
        include_internal_handles=False,
    )

    assert set(globals_dict) == {
        "__name__",
        "INPUTS",
        "RESULT",
        "get_pose",
    }
    assert not hasattr(globals_dict["get_pose"], "__wrapped__")
    assert globals_dict["get_pose"]("cube") == {"name": "cube"}
    assert trace.events[0]["name"] == "get_pose"


def test_legacy_capsule_globals_keep_internal_handles_by_default():
    env = object.__new__(CodeExecutionEnvBase)
    env.low_level_env = object()
    env._apis = {"fake": FakeApi()}

    globals_dict = env._build_capsule_globals()

    assert globals_dict["env"] is env.low_level_env
    assert globals_dict["APIS"] is env._apis


def test_cube_lift_api_keeps_get_observation_recovery_contract():
    api = FrankaControlApi.__new__(FrankaControlApi)
    api.real = False

    assert api.recovery_observation_functions() == {"get_observation"}


def test_handover_api_exposes_environment_observation_for_recovery():
    observation = {"agentview": {"images": {"rgb": "frame"}}}

    class FakeEnv:
        def get_observation(self):
            return observation

    api = FrankaHandoverApi.__new__(FrankaHandoverApi)
    api._env = FakeEnv()

    assert api.functions()["get_observation"]() is observation
    assert api.recovery_observation_functions() == {"get_observation"}
