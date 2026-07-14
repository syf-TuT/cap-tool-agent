from __future__ import annotations

from typing import Any

import pytest

from capx.envs.simulators import robosuite_cube_lift
from capx.envs.simulators import robosuite_cubes
from capx.envs.simulators import robosuite_cubes_restack
from capx.envs.simulators import robosuite_handover
from capx.envs.simulators import robosuite_nut_assembly
from capx.envs.simulators import robosuite_spill_wipe
from capx.envs.simulators import robosuite_two_arm_lift
from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv


class ConstructorCaptured(Exception):
    pass


def test_robosuite_base_retains_privilege_mode() -> None:
    assert RobosuiteBaseEnv(privileged=False).privileged is False
    assert RobosuiteBaseEnv(privileged=True).privileged is True


ConstructorCase = tuple[type[Any], Any, str]


CONSTRUCTOR_CASES: tuple[ConstructorCase, ...] = (
    (
        robosuite_cubes.FrankaRobosuiteCubesLowLevel,
        robosuite_cubes.suite.environments.manipulation.stack,
        "Stack",
    ),
    (
        robosuite_cube_lift.FrankaRobosuiteCubeLiftLowLevel,
        robosuite_cube_lift.suite.environments.manipulation.lift,
        "Lift",
    ),
    (
        robosuite_cubes_restack.FrankaRobosuiteCubesRestackLowLevel,
        robosuite_cubes_restack.suite.environments.manipulation.stack,
        "Stack",
    ),
    (
        robosuite_handover.RobosuiteHandoverEnv,
        robosuite_handover.suite.environments.manipulation.two_arm_handover,
        "TwoArmHandover",
    ),
    (
        robosuite_spill_wipe.FrankaRobosuiteSpillWipeLowLevel,
        robosuite_spill_wipe.suite.environments.manipulation.wipe,
        "Wipe",
    ),
    (
        robosuite_nut_assembly.FrankaRobosuiteNutAssembly,
        robosuite_nut_assembly.suite.environments.manipulation.nut_assembly,
        "NutAssemblySquare",
    ),
)


@pytest.mark.parametrize(("wrapper_cls", "constructor_owner", "constructor_name"), CONSTRUCTOR_CASES)
@pytest.mark.parametrize("privileged", [False, True])
def test_wrapper_selects_object_observations_at_robosuite_source(
    monkeypatch: pytest.MonkeyPatch,
    wrapper_cls: type[Any],
    constructor_owner: Any,
    constructor_name: str,
    privileged: bool,
) -> None:
    captured_kwargs: dict[str, Any] = {}

    def capture_constructor(**kwargs: Any) -> None:
        captured_kwargs.update(kwargs)
        raise ConstructorCaptured

    monkeypatch.setattr(constructor_owner, constructor_name, capture_constructor)
    module = __import__(wrapper_cls.__module__, fromlist=["load_composite_controller_config"])
    monkeypatch.setattr(module, "load_composite_controller_config", lambda **_: {})

    with pytest.raises(ConstructorCaptured):
        wrapper_cls(privileged=privileged)

    assert captured_kwargs["use_object_obs"] is privileged


@pytest.mark.parametrize("privileged", [False, True])
def test_two_arm_lift_selects_object_observations_at_robosuite_source(
    monkeypatch: pytest.MonkeyPatch,
    privileged: bool,
) -> None:
    captured_kwargs: dict[str, Any] = {}

    def capture_make(*args: Any, **kwargs: Any) -> None:
        captured_kwargs.update(kwargs)
        raise ConstructorCaptured

    monkeypatch.setattr(robosuite_two_arm_lift.suite, "make", capture_make)
    monkeypatch.setattr(
        robosuite_two_arm_lift, "load_composite_controller_config", lambda **_: {}
    )

    with pytest.raises(ConstructorCaptured):
        robosuite_two_arm_lift.RobosuiteTwoArmLiftEnv(privileged=privileged)

    assert captured_kwargs["use_object_obs"] is privileged
