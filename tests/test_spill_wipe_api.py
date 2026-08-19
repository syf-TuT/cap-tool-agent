import inspect

import pytest

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv
from capx.envs.simulators.robosuite_spill_wipe import FrankaRobosuiteSpillWipeLowLevel
from capx.integrations.franka.spill_wipe import FrankaControlSpillWipeApi
from capx.integrations.franka.spill_wipe_privileged import FrankaControlSpillWipePrivilegedApi


@pytest.mark.parametrize(
    "api_cls",
    (FrankaControlSpillWipeApi, FrankaControlSpillWipePrivilegedApi),
)
def test_spill_wipe_apis_declare_object_pose_as_recovery_observation_function(api_cls):
    api = object.__new__(api_cls)

    assert api.recovery_observation_functions() == {"get_object_pose"}


def test_spill_wipe_motion_and_video_capture_defaults_match_other_robosuite_tasks():
    base_max_steps = inspect.signature(RobosuiteBaseEnv.move_to_joints_blocking).parameters[
        "max_steps"
    ].default
    spill_max_steps = inspect.signature(
        FrankaRobosuiteSpillWipeLowLevel.move_to_joints_blocking
    ).parameters["max_steps"].default

    assert spill_max_steps >= base_max_steps
    assert FrankaRobosuiteSpillWipeLowLevel._SUBSAMPLE_RATE <= RobosuiteBaseEnv._SUBSAMPLE_RATE
