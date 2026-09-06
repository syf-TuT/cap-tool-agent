from capx.integrations.franka.control import FrankaControlApi
from capx.integrations.franka.control_privileged import FrankaControlPrivilegedApi
from capx.integrations.franka.control_reduced import FrankaControlApiReduced
from capx.integrations.franka.handover import FrankaHandoverApi
from capx.integrations.franka.handover_privileged import FrankaHandoverPrivilegedApi
from capx.integrations.franka.libero import FrankaLiberoApi
from capx.integrations.franka.libero_privileged import FrankaLiberoPrivilegedApi
from capx.integrations.franka.libero_reduced import FrankaLiberoApiReduced
from capx.integrations.franka.nut_assembly_privileged import (
    FrankaControlNutAssemblyPrivilegedApi,
)
from capx.integrations.franka.nut_assembly_visual import FrankaControlNutAssemblyVisualApi
from capx.integrations.franka.spill_wipe import FrankaControlSpillWipeApi
from capx.integrations.franka.spill_wipe_privileged import FrankaControlSpillWipePrivilegedApi
from capx.integrations.franka.two_arm_lift import FrankaTwoArmLiftApi
from capx.integrations.franka.two_arm_lift_privileged import FrankaTwoArmLiftPrivilegedApi
from capx.runtime_control.side_effects import collect_side_effect_calls


SINGLE_ARM_CONTROL_SIDE_EFFECTS = {
    "goto_pose",
    "home_pose",
    "open_gripper",
    "close_gripper",
}

SINGLE_ARM_REDUCED_SIDE_EFFECTS = {
    "move_to_joints",
    "open_gripper",
    "close_gripper",
}

BIMANUAL_REDUCED_SIDE_EFFECTS = {
    "move_to_joints_both",
    "move_to_joints_arm0",
    "move_to_joints_arm1",
    "open_gripper_arm0",
    "close_gripper_arm0",
    "open_gripper_arm1",
    "close_gripper_arm1",
}

SINGLE_ARM_HOME_SIDE_EFFECTS = {
    "goto_pose",
    "goto_home_joint_position",
    "open_gripper",
    "close_gripper",
}

BIMANUAL_GOTO_POSE_SIDE_EFFECTS = {
    "goto_pose_arm0",
    "goto_pose_arm1",
    "goto_pose_both",
    "open_gripper_arm0",
    "close_gripper_arm0",
    "open_gripper_arm1",
    "close_gripper_arm1",
}

HANDOVER_SIDE_EFFECTS = {
    "goto_pose_arm0",
    "goto_pose_arm1",
    "open_gripper_arm0",
    "close_gripper_arm0",
    "open_gripper_arm1",
    "close_gripper_arm1",
}


class _FakeApi:
    def __init__(self, names):
        self._names = names

    def side_effect_functions(self):
        return set(self._names)


class _SensingOnlyApi:
    """An API that declares no side effects (all sensing/compute)."""

    def side_effect_functions(self):
        return set()


class _LegacyApi:
    """An older API that predates the declaration method."""


class _ObservationEnv:
    def __init__(self):
        self.observation = {"robot": "current"}

    def get_observation(self):
        return self.observation


class _MotionEnv:
    def __init__(self):
        self.moved_joints = []

    def move_to_joints_blocking(self, joints):
        self.moved_joints.append(joints)


def test_collect_returns_empty_for_no_apis():
    assert collect_side_effect_calls([]) == set()


def test_collect_gathers_declared_side_effect_names():
    apis = [_FakeApi({"goto_pose", "close_gripper"})]

    assert collect_side_effect_calls(apis) == {"goto_pose", "close_gripper"}


def test_collect_merges_across_multiple_apis():
    apis = [_FakeApi({"goto_pose"}), _FakeApi({"open_gripper", "goto_home_joint_position"})]

    assert collect_side_effect_calls(apis) == {
        "goto_pose",
        "open_gripper",
        "goto_home_joint_position",
    }


def test_collect_skips_apis_without_declaration_method():
    apis = [_FakeApi({"goto_pose"}), _LegacyApi(), _SensingOnlyApi()]

    assert collect_side_effect_calls(apis) == {"goto_pose"}


def test_franka_control_api_declares_pose_and_gripper_side_effects():
    api = object.__new__(FrankaControlApi)

    assert SINGLE_ARM_CONTROL_SIDE_EFFECTS <= api.side_effect_functions()


def test_franka_control_api_exposes_current_observation():
    env = _ObservationEnv()
    api = object.__new__(FrankaControlApi)
    api._env = env
    api.real = False

    functions = api.functions()

    assert "get_observation" in functions
    assert functions["get_observation"]() is env.observation


def test_franka_privileged_api_declares_pose_and_gripper_side_effects():
    api = object.__new__(FrankaControlPrivilegedApi)

    assert SINGLE_ARM_CONTROL_SIDE_EFFECTS <= api.side_effect_functions()


def test_franka_privileged_api_exposes_home_pose_when_declared_as_side_effect():
    env = _MotionEnv()
    api = object.__new__(FrankaControlPrivilegedApi)
    api._env = env

    functions = api.functions()

    assert "home_pose" in functions
    functions["home_pose"]()
    assert len(env.moved_joints) == 1


def test_franka_reduced_api_declares_joint_and_gripper_side_effects():
    api = object.__new__(FrankaControlApiReduced)

    assert (
        SINGLE_ARM_REDUCED_SIDE_EFFECTS | BIMANUAL_REDUCED_SIDE_EFFECTS
    ) <= api.side_effect_functions()


def test_franka_libero_api_declares_pose_home_and_gripper_side_effects():
    api = object.__new__(FrankaLiberoApi)

    assert SINGLE_ARM_HOME_SIDE_EFFECTS <= api.side_effect_functions()


def test_franka_libero_reduced_api_declares_pose_joint_home_and_gripper_side_effects():
    api = object.__new__(FrankaLiberoApiReduced)

    assert (
        SINGLE_ARM_HOME_SIDE_EFFECTS | {"move_to_joints"}
    ) <= api.side_effect_functions()


def test_franka_libero_privileged_api_declares_pose_and_gripper_side_effects():
    api = object.__new__(FrankaLiberoPrivilegedApi)

    assert {
        "goto_pose",
        "goto_pose_interactive_cartesian",
        "open_gripper",
        "close_gripper",
    } <= api.side_effect_functions()


def test_franka_nut_assembly_apis_declare_pose_home_and_gripper_side_effects():
    for api_cls in (FrankaControlNutAssemblyVisualApi, FrankaControlNutAssemblyPrivilegedApi):
        api = object.__new__(api_cls)

        assert SINGLE_ARM_HOME_SIDE_EFFECTS <= api.side_effect_functions()


def test_franka_spill_wipe_apis_declare_pose_side_effects():
    for api_cls in (FrankaControlSpillWipeApi, FrankaControlSpillWipePrivilegedApi):
        api = object.__new__(api_cls)

        assert {"goto_pose"} <= api.side_effect_functions()


def test_franka_handover_apis_declare_bimanual_side_effects():
    for api_cls in (FrankaHandoverApi, FrankaHandoverPrivilegedApi):
        api = object.__new__(api_cls)

        assert HANDOVER_SIDE_EFFECTS <= api.side_effect_functions()


def test_franka_two_arm_lift_apis_declare_bimanual_side_effects():
    for api_cls in (FrankaTwoArmLiftApi, FrankaTwoArmLiftPrivilegedApi):
        api = object.__new__(api_cls)

        assert BIMANUAL_GOTO_POSE_SIDE_EFFECTS <= api.side_effect_functions()
