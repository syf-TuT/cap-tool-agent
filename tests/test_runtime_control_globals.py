from types import MappingProxyType, SimpleNamespace

import pytest

from capx.envs.trial import _build_capsule_execution_globals
from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.integrations.franka.control import FrankaControlApi
from capx.integrations.franka.handover import FrankaHandoverApi
from capx.runtime_control.contract import STRICT_CAPSULE_SAFE_BUILTINS
from capx.runtime_control.executor import CapsuleExecutor
from capx.runtime_control.schema import CodeRegion
from capx.runtime_control.trace import RuntimeTrace


FORBIDDEN_CAPSULE_BUILTINS = {
    "__import__",
    "breakpoint",
    "bytearray",
    "compile",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "object",
    "open",
    "setattr",
    "super",
    "type",
    "vars",
}


class FakeApi:
    def functions(self):
        return {"get_pose": self.get_pose}

    def get_pose(self, name):
        return {"name": name}


class FailingApi:
    def functions(self):
        return {"fail_primitive": self.fail_primitive}

    def fail_primitive(self):
        raise RuntimeError("primitive failed")


class NamedApi:
    def __init__(self, functions):
        self._functions = functions

    def functions(self):
        return self._functions


def _capsule_env(*, privileged, api=None):
    env = object.__new__(CodeExecutionEnvBase)
    env.cfg = SimpleNamespace(privileged=privileged)
    env.low_level_env = object()
    env._apis = {"fake": api or FakeApi()}
    return env


def _region(source):
    return CodeRegion(
        region_id="region_1",
        source=source,
        start_line=1,
        end_line=max(1, source.count("\n")),
    )


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
        "__builtins__",
        "INPUTS",
        "RESULT",
        "get_pose",
    }
    assert isinstance(globals_dict["__builtins__"], MappingProxyType)
    assert set(globals_dict["__builtins__"]) == STRICT_CAPSULE_SAFE_BUILTINS
    assert not FORBIDDEN_CAPSULE_BUILTINS & set(globals_dict["__builtins__"])
    assert not hasattr(globals_dict["get_pose"], "__wrapped__")
    assert globals_dict["get_pose"]("cube") == {"name": "cube"}
    assert trace.events[0]["name"] == "get_pose"


@pytest.mark.parametrize("operation", ["set", "delete", "clear", "update"])
def test_public_capsule_builtins_mapping_is_immutable(operation):
    env = _capsule_env(privileged=False)
    safe_builtins = env._build_capsule_globals(
        include_internal_handles=False,
    )["__builtins__"]

    with pytest.raises((AttributeError, TypeError)):
        if operation == "set":
            safe_builtins["open"] = open
        elif operation == "delete":
            del safe_builtins["len"]
        elif operation == "clear":
            safe_builtins.clear()
        else:
            safe_builtins.update({"open": open})

    assert set(safe_builtins) == STRICT_CAPSULE_SAFE_BUILTINS


def test_public_capsule_builtins_reject_base_dict_mutation():
    env = _capsule_env(privileged=False)
    safe_builtins = env._build_capsule_globals(
        include_internal_handles=False,
    )["__builtins__"]

    with pytest.raises(TypeError):
        dict.__setitem__(safe_builtins, "open", open)


def test_restricted_executor_supports_declared_safe_builtins():
    env = _capsule_env(privileged=False)
    executor = CapsuleExecutor(
        base_globals=env._build_capsule_globals(include_internal_handles=False)
    )
    source = (
        "values = list(range(4))\n"
        "pairs = list(enumerate(zip(values, reversed(values))))\n"
        "stats = (sum(values), min(values), max(values), sorted(values), len(values))\n"
        "types = (bool(1), int(2.5), float(2), str(3), tuple(values), set(values), dict())\n"
        "numbers = (abs(-3), round(1.25, 1), all([True]), any([False, True]))\n"
        "try:\n"
        "    raise ValueError('expected')\n"
        "except ValueError:\n"
        "    print('caught')\n"
        "try:\n"
        "    raise RuntimeError('expected')\n"
        "except RuntimeError:\n"
        "    RESULT = (pairs, stats, types, numbers)\n"
    )

    event = executor.run_region(_region(source))

    assert event.status == "success"
    assert event.stdout == "caught\n"
    assert executor.globals["RESULT"][1] == (6, 0, 3, [0, 1, 2, 3], 4)


@pytest.mark.parametrize(
    ("source", "forbidden_name"),
    [
        ("RESULT = __import__('os')\n", "__import__"),
        ("RESULT = eval('40 + 2')\n", "eval"),
        ("open(INPUTS['path'], 'w')\n", "open"),
    ],
)
def test_restricted_executor_cannot_resolve_forbidden_builtins(
    tmp_path,
    source,
    forbidden_name,
):
    env = _capsule_env(privileged=False)
    base_globals = env._build_capsule_globals(include_internal_handles=False)
    target = tmp_path / "forbidden-write.txt"
    base_globals["INPUTS"] = {"path": str(target)}
    executor = CapsuleExecutor(base_globals=base_globals)

    event = executor.run_region(_region(source))

    assert event.status == "failed"
    assert event.evidence["exception_type"] == "NameError"
    assert forbidden_name in event.message
    assert not target.exists()


def test_restricted_executor_cannot_mutate_builtins_mapping():
    env = _capsule_env(privileged=False)
    executor = CapsuleExecutor(
        base_globals=env._build_capsule_globals(include_internal_handles=False)
    )

    event = executor.run_region(_region("__builtins__['open'] = print\n"))

    assert event.status == "failed"
    assert event.evidence["exception_type"] == "TypeError"
    assert "open" not in executor.globals["__builtins__"]


def test_public_capsule_failed_api_call_remains_opaque_and_traced():
    env = _capsule_env(privileged=False, api=FailingApi())
    trace = RuntimeTrace()
    globals_dict = env._build_capsule_globals(
        trace=trace,
        include_internal_handles=False,
    )
    executor = CapsuleExecutor(base_globals=globals_dict, trace=trace)

    assert not hasattr(globals_dict["fail_primitive"], "__wrapped__")
    event = executor.run_region(_region("fail_primitive()\n"))

    assert event.status == "failed"
    assert event.evidence["trace_events"][0]["name"] == "fail_primitive"
    assert event.evidence["trace_events"][0]["status"] == "failed"


@pytest.mark.parametrize(
    "reserved_name",
    sorted(
        STRICT_CAPSULE_SAFE_BUILTINS
        | {"__builtins__", "__name__", "APIS", "INPUTS", "RESULT", "env"}
    ),
)
def test_public_capsule_globals_reject_reserved_api_names(reserved_name):
    api = NamedApi({reserved_name: lambda: None})
    env = _capsule_env(privileged=False, api=api)

    with pytest.raises(ValueError) as exc_info:
        env._build_capsule_globals(include_internal_handles=False)

    assert "reserved" in str(exc_info.value).lower()
    assert reserved_name in str(exc_info.value)


@pytest.mark.parametrize("private_name", ["_private_api", "__dunder_api__"])
def test_public_capsule_globals_reject_private_api_names(private_name):
    api = NamedApi({private_name: lambda: None})
    env = _capsule_env(privileged=False, api=api)

    with pytest.raises(ValueError) as exc_info:
        env._build_capsule_globals(include_internal_handles=False)

    assert "private" in str(exc_info.value).lower()
    assert private_name in str(exc_info.value)


def test_public_capsule_globals_reject_duplicate_names_across_apis():
    env = _capsule_env(privileged=False)
    env._apis = {
        "first": NamedApi({"observe": lambda: "first"}),
        "second": NamedApi({"observe": lambda: "second"}),
    }

    with pytest.raises(ValueError) as exc_info:
        env._build_capsule_globals(include_internal_handles=False)

    assert "duplicate" in str(exc_info.value).lower()
    assert "observe" in str(exc_info.value)


def test_legacy_capsule_globals_preserve_api_name_overwrite_behavior():
    replacement = lambda: "legacy-env"
    env = _capsule_env(
        privileged=True,
        api=NamedApi({"env": replacement}),
    )

    globals_dict = env._build_capsule_globals(include_internal_handles=True)

    assert globals_dict["env"] is replacement


def test_legacy_capsule_globals_keep_internal_handles_by_default():
    env = object.__new__(CodeExecutionEnvBase)
    env.low_level_env = object()
    env._apis = {"fake": FakeApi()}

    globals_dict = env._build_capsule_globals()

    assert globals_dict["env"] is env.low_level_env
    assert globals_dict["APIS"] is env._apis
    assert "__builtins__" not in globals_dict


@pytest.mark.parametrize(
    ("privileged", "validate_contract", "restricted"),
    [
        (False, False, True),
        (False, True, True),
        (True, True, True),
        (True, False, False),
    ],
)
def test_capsule_execution_globals_follow_runtime_safety_policy(
    privileged,
    validate_contract,
    restricted,
):
    env = _capsule_env(privileged=privileged)
    trace = RuntimeTrace()

    globals_dict = _build_capsule_execution_globals(
        env,
        trace,
        {"capsule_validate_program_contract": validate_contract},
    )

    if restricted:
        assert set(globals_dict) == {
            "__name__",
            "__builtins__",
            "INPUTS",
            "RESULT",
            "get_pose",
        }
        assert set(globals_dict["__builtins__"]) == STRICT_CAPSULE_SAFE_BUILTINS
        assert "env" not in globals_dict
        assert "APIS" not in globals_dict
    else:
        assert globals_dict["env"] is env.low_level_env
        assert globals_dict["APIS"] is env._apis
        assert "__builtins__" not in globals_dict
        executor = CapsuleExecutor(base_globals=globals_dict, trace=trace)
        event = executor.run_region(
            _region("RESULT = __import__('math').sqrt(9)\n")
        )
        assert event.status == "success"
        assert executor.globals["RESULT"] == 3.0


def test_public_policy_ignores_subclass_capsule_global_injection():
    class UnsafeOverride(CodeExecutionEnvBase):
        def _build_capsule_globals(self, trace=None, *, include_internal_handles=True):
            globals_dict = super()._build_capsule_globals(
                trace=trace,
                include_internal_handles=include_internal_handles,
            )
            globals_dict["raw_env"] = self.low_level_env
            return globals_dict

    env = object.__new__(UnsafeOverride)
    env.cfg = SimpleNamespace(privileged=False)
    env.low_level_env = object()
    env._apis = {"fake": FakeApi()}

    globals_dict = _build_capsule_execution_globals(
        env,
        RuntimeTrace(),
        {"capsule_validate_program_contract": False},
    )

    assert "raw_env" not in globals_dict
    assert "env" not in globals_dict
    assert "APIS" not in globals_dict


def test_traditional_execution_globals_keep_legacy_internal_handles():
    env = _capsule_env(privileged=False)

    env._init_exec_globals()

    assert env._exec_globals["env"] is env.low_level_env
    assert env._exec_globals["APIS"] is env._apis
    assert "__builtins__" not in env._exec_globals


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
