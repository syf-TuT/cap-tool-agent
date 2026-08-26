from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from capx.envs.robosuite_seed import reseed_robosuite_owner
from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.integrations.base_api import ApiBase
from capx.integrations.franka.control_privileged import FrankaControlPrivilegedApi


class _FakeApi:
    def __init__(self, events: list[tuple[Any, ...]]) -> None:
        self.events = events
        self.episode_value = "stale"

    def functions(self) -> dict[str, Any]:
        return {}

    def reset_episode(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> bool:
        self.events.append(("api_reset", seed, options))
        self.episode_value = None
        return True


class _UnconfirmedApi(_FakeApi):
    def reset_episode(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> bool:
        super().reset_episode(seed=seed, options=options)
        return False


class _StatelessApi(ApiBase):
    def functions(self) -> dict[str, Any]:
        return {}


class _FakeLowLevelEnv:
    def __init__(self, events: list[tuple[Any, ...]]) -> None:
        self.events = events

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.events.append(("env_reset", seed, options))
        return {"reset_obs": seed}, {"initial_state_sha256": "a" * 64}

    def get_observation(self) -> dict[str, Any]:
        self.events.append(("get_observation",))
        return {"fresh_obs": True}


class _Sampler:
    def __init__(self, children: Any = None) -> None:
        self.rng = object()
        self.samplers = [] if children is None else children


def _source_tree(relative_path: str) -> ast.Module:
    repository = Path(__file__).resolve().parents[1]
    return ast.parse((repository / relative_path).read_text(encoding="utf-8"))


def _call_keywords(tree: ast.AST, *, attribute: str) -> list[set[str]]:
    matches: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == attribute:
            matches.append({keyword.arg for keyword in node.keywords if keyword.arg is not None})
    return matches


def test_api_base_reset_episode_is_a_default_noop() -> None:
    api = _StatelessApi(None)  # type: ignore[arg-type]
    assert api.reset_episode(seed=5, options={"mode": "clean"}) is True


def test_privileged_franka_reset_episode_clears_ik_warm_start_without_init() -> None:
    api = object.__new__(FrankaControlPrivilegedApi)
    api.cfg = object()

    api.reset_episode(seed=5, options={"mode": "clean"})

    assert api.cfg is None


def test_code_execution_reset_clears_episode_state_before_low_level_reset() -> None:
    events: list[tuple[Any, ...]] = []
    env = object.__new__(CodeExecutionEnvBase)
    api = _FakeApi(events)
    env._apis = {"fake": api}
    env.low_level_env = _FakeLowLevelEnv(events)
    env._task_prompt = "stack"
    env._full_prompt = []
    env._step_count = 9
    old_globals = {"leaked": object(), "INPUTS": {"old": True}}
    env._exec_globals = old_globals

    original_init = env._init_exec_globals

    def recording_init() -> None:
        events.append(("namespace_reset",))
        original_init()

    env._init_exec_globals = recording_init  # type: ignore[method-assign]

    obs, info = env.reset(seed=11, options={"mode": "clean"})

    assert events[:3] == [
        ("api_reset", 11, {"mode": "clean"}),
        ("namespace_reset",),
        ("env_reset", 11, {"mode": "clean"}),
    ]
    assert env._step_count == 0
    assert env._exec_globals is not old_globals
    assert "leaked" not in env._exec_globals
    assert env._exec_globals["INPUTS"] is obs
    assert obs == {"reset_obs": 11, "fresh_obs": True, "full_prompt": []}
    assert info["initial_state_sha256"] == "a" * 64
    assert info["capsule_reset_evidence"] == {
        "namespace_fresh": True,
        "api_state_cleared": True,
        "api_reset_count": 1,
        "api_reset_confirmed_count": 1,
    }


def test_code_execution_reset_does_not_invent_api_reset_confirmation() -> None:
    events: list[tuple[Any, ...]] = []
    env = object.__new__(CodeExecutionEnvBase)
    env._apis = {"unconfirmed": _UnconfirmedApi(events)}
    env.low_level_env = _FakeLowLevelEnv(events)
    env._task_prompt = "stack"
    env._full_prompt = []
    env._step_count = 0
    env._exec_globals = {}

    _obs, info = env.reset(seed=11)

    assert info["capsule_reset_evidence"] == {
        "namespace_fresh": True,
        "api_state_cleared": False,
        "api_reset_count": 1,
        "api_reset_confirmed_count": 0,
    }


def test_exec_user_code_exposes_typed_error_in_step_info() -> None:
    events: list[tuple[Any, ...]] = []
    env = object.__new__(CodeExecutionEnvBase)
    env._apis = {}
    env.low_level_env = _FakeLowLevelEnv(events)
    env._task_prompt = "stack"
    env._full_prompt = []
    env._step_count = 0
    env._init_exec_globals()
    env.compute_reward = lambda: 0.0  # type: ignore[method-assign]
    env.low_level_env.task_completed = lambda: False  # type: ignore[attr-defined]

    _obs, _reward, _terminated, _truncated, info = env.step(
        "raise ValueError('bad program')"
    )

    assert info["sandbox_rc"] == 1
    assert info["error_type"] == "ValueError"
    assert info["error_message"] == "bad program"


def test_reseed_helper_synchronizes_env_and_nested_placement_samplers() -> None:
    shared = _Sampler()
    nested = _Sampler([shared])
    mapping = _Sampler({"nested": nested, "shared": shared})
    robosuite_env = type("FakeRobosuite", (), {})()
    robosuite_env.seed = None
    robosuite_env.rng = object()
    robosuite_env.placement_initializer = mapping
    owner = type("Owner", (), {"robosuite_env": robosuite_env})()

    reseed_robosuite_owner(owner, 17)

    assert robosuite_env.seed == 17
    assert owner._rng is robosuite_env.rng
    assert mapping.rng is owner._rng
    assert nested.rng is owner._rng
    assert shared.rng is owner._rng


def test_cube_stack_constructors_receive_seed_in_all_three_branches() -> None:
    tree = _source_tree("capx/envs/simulators/robosuite_cubes.py")
    stack_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "Stack":
            stack_calls.append(node)

    assert len(stack_calls) == 3
    assert all("seed" in {keyword.arg for keyword in call.keywords} for call in stack_calls)


def test_cube_lift_constructors_receive_seed_in_all_three_branches() -> None:
    tree = _source_tree("capx/envs/simulators/robosuite_cube_lift.py")
    lift_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "Lift":
            lift_calls.append(node)

    assert len(lift_calls) == 3
    assert all(
        any(
            keyword.arg == "seed"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "seed"
            for keyword in call.keywords
        )
        for call in lift_calls
    )


def test_cube_reset_reseeds_before_calling_robosuite_reset() -> None:
    tree = _source_tree("capx/envs/simulators/robosuite_cubes.py")
    reset_method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "reset"
    )
    calls = [node for node in ast.walk(reset_method) if isinstance(node, ast.Call)]
    reseed_position = next(
        call.lineno
        for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "_reseed_robosuite"
    )
    reset_position = next(
        call.lineno
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "reset"
        and isinstance(call.func.value, ast.Attribute)
        and call.func.value.attr == "robosuite_env"
    )

    assert reseed_position < reset_position


def test_cube_lift_reset_reseeds_before_calling_robosuite_reset() -> None:
    tree = _source_tree("capx/envs/simulators/robosuite_cube_lift.py")
    reset_method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "reset"
    )
    calls = [node for node in ast.walk(reset_method) if isinstance(node, ast.Call)]
    reseed_positions = [
        call.lineno
        for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "_reseed_robosuite"
    ]
    reset_position = next(
        call.lineno
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "reset"
        and isinstance(call.func.value, ast.Attribute)
        and call.func.value.attr == "robosuite_env"
    )

    assert reseed_positions, "Cube Lift reset must reseed the Robosuite owner"
    assert reseed_positions[0] < reset_position
