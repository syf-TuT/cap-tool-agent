from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from capx.envs.simulators import robosuite_cube_lift
from capx.envs.simulators import robosuite_cubes
from capx.envs.simulators import robosuite_cubes_restack
from capx.envs.simulators import robosuite_handover
from capx.envs.simulators import robosuite_nut_assembly
from capx.envs.simulators import robosuite_spill_wipe
from capx.envs.simulators import robosuite_two_arm_lift
from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv
from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.integrations.franka import nut_assembly_visual
from capx.integrations.franka.control import FrankaControlApi
from capx.integrations.franka.handover import FrankaHandoverApi
from capx.integrations.franka.nut_assembly_visual import FrankaControlNutAssemblyVisualApi


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


@pytest.mark.parametrize(
    ("wrapper_cls", "constructor_owner", "constructor_name"), CONSTRUCTOR_CASES
)
@pytest.mark.parametrize("privileged", [False, True])
@pytest.mark.parametrize("enable_render", [False, True])
def test_wrapper_selects_object_observations_at_robosuite_source(
    monkeypatch: pytest.MonkeyPatch,
    wrapper_cls: type[Any],
    constructor_owner: Any,
    constructor_name: str,
    privileged: bool,
    enable_render: bool,
) -> None:
    captured_kwargs: dict[str, Any] = {}

    def capture_constructor(**kwargs: Any) -> None:
        captured_kwargs.update(kwargs)
        raise ConstructorCaptured

    monkeypatch.setattr(constructor_owner, constructor_name, capture_constructor)
    module = __import__(wrapper_cls.__module__, fromlist=["load_composite_controller_config"])
    monkeypatch.setattr(module, "load_composite_controller_config", lambda **_: {})

    with pytest.raises(ConstructorCaptured):
        wrapper_cls(privileged=privileged, enable_render=enable_render)

    assert captured_kwargs["use_object_obs"] is privileged


@pytest.mark.parametrize("privileged", [False, True])
@pytest.mark.parametrize("enable_render", [False, True])
def test_two_arm_lift_selects_object_observations_at_robosuite_source(
    monkeypatch: pytest.MonkeyPatch,
    privileged: bool,
    enable_render: bool,
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
        robosuite_two_arm_lift.RobosuiteTwoArmLiftEnv(
            privileged=privileged, enable_render=enable_render
        )

    assert captured_kwargs["use_object_obs"] is privileged


class FakeRobosuiteEnv:
    def __init__(self, observation: dict[str, Any], *, with_sim: bool = False) -> None:
        self.observation = observation
        if with_sim:
            self.sim = FakeSim()

    def _get_observations(self, *, force_update: bool = False) -> dict[str, Any]:
        return self.observation.copy()


class FakeModel:
    def __init__(self) -> None:
        self.cam_pos = np.zeros((1, 3), dtype=np.float64)
        self.cam_quat = np.zeros((1, 4), dtype=np.float64)
        self.cam_fovy = np.array([45.0], dtype=np.float64)

    def camera_name2id(self, _: str) -> int:
        return 0


class FakeData:
    def __init__(self) -> None:
        self.xquat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (4, 1))
        self.xpos = np.zeros((4, 3), dtype=np.float64)

    def get_camera_xmat(self, _: str) -> np.ndarray:
        return np.eye(3)

    def get_camera_xpos(self, _: str) -> np.ndarray:
        return np.zeros(3)


class FakeSim:
    def __init__(self) -> None:
        self.model = FakeModel()
        self.data = FakeData()

    def forward(self) -> None:
        pass


CUBE_OBSERVATION_CASES = (
    (robosuite_cubes.FrankaRobosuiteCubesLowLevel, "_cube_pose_dict"),
    (robosuite_cube_lift.FrankaRobosuiteCubeLiftLowLevel, "_cube_pose_dict"),
    (robosuite_cubes_restack.FrankaRobosuiteCubesRestackLowLevel, "_cube_pose_dict"),
)


@pytest.mark.parametrize(("wrapper_cls", "pose_method_name"), CUBE_OBSERVATION_CASES)
def test_non_privileged_cube_observation_does_not_require_object_state(
    wrapper_cls: type[Any],
    pose_method_name: str,
) -> None:
    env = wrapper_cls.__new__(wrapper_cls)
    env.privileged = False
    env.robosuite_env = FakeRobosuiteEnv({"camera": "frame", "robot0_joint_pos": "joints"})
    env._process_camera_observations = lambda _: None
    env._compute_gripper_obs = lambda _: None
    setattr(
        env,
        pose_method_name,
        lambda _: pytest.fail("non-privileged observation requested cube ground truth"),
    )

    observation = env.get_observation()

    assert observation["camera"] == "frame"
    assert observation["robot0_joint_pos"] == "joints"
    assert "cube_poses" not in observation


@pytest.mark.parametrize(("wrapper_cls", "pose_method_name"), CUBE_OBSERVATION_CASES)
def test_privileged_cube_observation_keeps_derived_poses(
    wrapper_cls: type[Any],
    pose_method_name: str,
) -> None:
    env = wrapper_cls.__new__(wrapper_cls)
    env.privileged = True
    env.robosuite_env = FakeRobosuiteEnv({})
    env._process_camera_observations = lambda _: None
    env._compute_gripper_obs = lambda _: None
    setattr(
        env,
        pose_method_name,
        lambda _: {
            "primary": np.zeros(7, dtype=np.float32),
            "secondary": np.ones(7, dtype=np.float32),
        },
    )

    observation = env.get_observation()

    assert "cube_poses" in observation
    assert "primary" in observation["cube_poses"]


def _make_two_arm_wrapper(wrapper_cls: type[Any], *, privileged: bool) -> Any:
    env = wrapper_cls.__new__(wrapper_cls)
    env.privileged = privileged
    env.render_camera_names = ["agentview"]
    env.segmentation_level = "instance"
    env._render_width = 8
    env._render_height = 8
    env.base_link_wxyz_xyz_0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    env.base_link_wxyz_xyz_1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    env.base_link_idx_0 = 0
    env.gripper_link_idx_0 = 1
    env.gripper_link_idx_1 = 2
    env.gripper_metric_length = 0.04
    env.robosuite_env = FakeRobosuiteEnv(
        {
            "robot0_gripper_qpos": np.array([0.0]),
            "robot1_gripper_qpos": np.array([0.0]),
        },
        with_sim=True,
    )
    return env


@pytest.mark.parametrize(
    ("wrapper_cls", "pose_method_name", "derived_key"),
    (
        (robosuite_handover.RobosuiteHandoverEnv, "_hammer_pose_dict", "hammer_poses"),
        (robosuite_two_arm_lift.RobosuiteTwoArmLiftEnv, "_pot_pose_dict", "pot_poses"),
    ),
)
def test_non_privileged_two_arm_observation_does_not_require_object_state(
    wrapper_cls: type[Any],
    pose_method_name: str,
    derived_key: str,
) -> None:
    env = _make_two_arm_wrapper(wrapper_cls, privileged=False)
    setattr(
        env,
        pose_method_name,
        lambda _: pytest.fail("non-privileged observation requested object ground truth"),
    )

    observation = env.get_observation()

    assert derived_key not in observation
    assert "robot0_cartesian_pos" in observation


@pytest.mark.parametrize(
    ("wrapper_cls", "pose_method_name", "derived_key", "pose_names"),
    (
        (
            robosuite_handover.RobosuiteHandoverEnv,
            "_hammer_pose_dict",
            "hammer_poses",
            ("hammer", "handle"),
        ),
        (
            robosuite_two_arm_lift.RobosuiteTwoArmLiftEnv,
            "_pot_pose_dict",
            "pot_poses",
            ("pot", "handle0", "handle1"),
        ),
    ),
)
def test_privileged_two_arm_observation_keeps_derived_poses(
    wrapper_cls: type[Any],
    pose_method_name: str,
    derived_key: str,
    pose_names: tuple[str, ...],
) -> None:
    env = _make_two_arm_wrapper(wrapper_cls, privileged=True)
    setattr(env, pose_method_name, lambda _: {name: np.zeros(7) for name in pose_names})

    observation = env.get_observation()

    assert set(observation[derived_key]) == set(pose_names)


def test_non_privileged_nut_observation_does_not_require_object_state() -> None:
    env = robosuite_nut_assembly.FrankaRobosuiteNutAssembly.__new__(
        robosuite_nut_assembly.FrankaRobosuiteNutAssembly
    )
    env.privileged = False
    env.render_camera_names = []
    env.robosuite_env = FakeRobosuiteEnv({"robot0_joint_pos": "joints"})
    env._compute_gripper_obs = lambda _: None
    env._get_nut_pose = lambda _: pytest.fail(
        "non-privileged observation requested nut ground truth"
    )

    observation = env.get_observation()

    assert observation["robot0_joint_pos"] == "joints"
    assert "nut_poses" not in observation


def test_privileged_nut_observation_keeps_derived_poses() -> None:
    env = robosuite_nut_assembly.FrankaRobosuiteNutAssembly.__new__(
        robosuite_nut_assembly.FrankaRobosuiteNutAssembly
    )
    env.privileged = True
    env.render_camera_names = []
    env.robosuite_env = FakeRobosuiteEnv({})
    env._compute_gripper_obs = lambda _: None
    env._get_nut_pose = lambda _: {"square_nut": np.zeros(7)}

    observation = env.get_observation()

    assert "square_nut" in observation["nut_poses"]


def test_spill_wipe_observation_works_without_object_state() -> None:
    env = robosuite_spill_wipe.FrankaRobosuiteSpillWipeLowLevel.__new__(
        robosuite_spill_wipe.FrankaRobosuiteSpillWipeLowLevel
    )
    env.privileged = False
    env.base_link_wxyz_xyz = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    env.gripper_link_wxyz_xyz = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    env.robosuite_env = FakeRobosuiteEnv({"robot0_joint_pos": np.zeros(7)})
    env._process_camera_observations = lambda _: None

    observation = env.get_observation()

    assert "robot_joint_pos" in observation
    assert "robot_cartesian_pos" in observation


class FakeRestackEvaluationEnv:
    def __init__(self, *, cube_heights: tuple[float, float]) -> None:
        base_height = 0.8
        body_xpos = np.zeros((2, 3), dtype=np.float64)
        body_xpos[:, 2] = base_height + np.asarray(cube_heights)
        xquat = np.array([[1.0, 0.0, 0.0, 0.0]])
        xpos = np.array([[0.0, 0.0, base_height]])
        self.sim = SimpleNamespace(
            data=SimpleNamespace(body_xpos=body_xpos, xquat=xquat, xpos=xpos)
        )
        self.cubeA_body_id = 0
        self.cubeB_body_id = 1

    def reward(self, action: Any = None) -> float:
        return 0.5

    def _check_success(self) -> bool:
        return True

    def _get_observations(self) -> dict[str, Any]:
        pytest.fail("evaluation requested public Robosuite object observations")


@pytest.mark.parametrize(
    ("cube_heights", "expected_reward", "expected_completed"),
    (
        ((0.03, 0.08), 0.5, True),
        ((0.05, 0.06), 0.0, False),
    ),
)
def test_restack_evaluation_uses_internal_simulator_state(
    cube_heights: tuple[float, float],
    expected_reward: float,
    expected_completed: bool,
) -> None:
    env = robosuite_cubes_restack.FrankaRobosuiteCubesRestackLowLevel.__new__(
        robosuite_cubes_restack.FrankaRobosuiteCubesRestackLowLevel
    )
    env.privileged = False
    env.base_link_idx = 0
    env.robosuite_env = FakeRestackEvaluationEnv(cube_heights=cube_heights)

    assert env.compute_reward() == expected_reward
    assert env.task_completed() is expected_completed


def test_non_privileged_nut_viser_tolerates_missing_ground_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = robosuite_nut_assembly.FrankaRobosuiteNutAssembly.__new__(
        robosuite_nut_assembly.FrankaRobosuiteNutAssembly
    )
    env.viser_debug = True
    env.gripper_metric_length = 0.04
    env.get_observation = lambda: {
        "robot_cartesian_pos": np.zeros(8),
        "robot_joint_pos": np.zeros(8),
    }
    env._viser_init_check = lambda: None
    env.urdf_vis = SimpleNamespace(update_cfg=lambda _: None)
    env.cube_center = None
    env.cube_rot = None
    env.cube_points = None
    env.cube_color = None
    env.grasp_frame_position = None
    env.grasp_frame_orientation = None
    monkeypatch.setattr(robosuite_nut_assembly, "obs_get_rgb", lambda _: {})

    env._update_viser_server()


def test_visual_nut_pose_uses_non_privileged_observation_for_square_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = {
        "robot0_robotview": {
            "images": {
                "rgb": np.zeros((1, 1, 3), dtype=np.uint8),
                "depth": np.ones((1, 1, 1), dtype=np.float64),
            },
            "intrinsics": np.eye(3),
            "pose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        }
    }
    low_level_env = SimpleNamespace(
        get_observation=lambda: observation,
        viser_debug=False,
        viser_server=None,
    )
    api = FrankaControlNutAssemblyVisualApi.__new__(FrankaControlNutAssemblyVisualApi)
    api._env = low_level_env
    api.camera_name = "robot0_robotview"
    api._segment_object_from_language = lambda *_: (
        np.ones((1, 1), dtype=bool),
        (0, 0),
        [1.0],
    )
    fake_obb = SimpleNamespace(center=np.zeros(3), R=np.eye(3))
    fake_point_cloud = SimpleNamespace(
        points=None,
        get_oriented_bounding_box=lambda: fake_obb,
    )
    monkeypatch.setattr(
        nut_assembly_visual.o3d.geometry, "PointCloud", lambda: fake_point_cloud
    )
    monkeypatch.setattr(
        nut_assembly_visual.o3d.utility, "Vector3dVector", lambda points: points
    )

    _, quaternion_wxyz = api.get_object_pose("square block")

    np.testing.assert_allclose(quaternion_wxyz, np.array([0.0, 1.0, 0.0, 0.0]))


class SafeLowLevelEnv:
    def __init__(self) -> None:
        self.internal_state = {"cube_poses": {"primary": "ground-truth"}}

    def get_observation(self) -> dict[str, Any]:
        return {"camera": "frame", "robot_joint_pos": "joints"}

    def reset(self, **_: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.get_observation(), {}

    def compute_reward(self) -> float:
        return 0.0

    def task_completed(self) -> bool:
        return False


def _make_code_execution_env(low_level_env: SafeLowLevelEnv) -> CodeExecutionEnvBase:
    env = CodeExecutionEnvBase.__new__(CodeExecutionEnvBase)
    env.low_level_env = low_level_env
    env._apis = {}
    env._full_prompt = []
    env._task_prompt = "task"
    env._step_count = 0
    env._init_exec_globals()
    return env


def test_high_level_observation_paths_do_not_reintroduce_internal_ground_truth() -> None:
    low_level_env = SafeLowLevelEnv()
    env = _make_code_execution_env(low_level_env)

    reset_observation, _ = env.reset()
    step_observation, *_ = env.step("RESULT = obs")

    assert "cube_poses" not in reset_observation
    assert "cube_poses" not in env._exec_globals["INPUTS"]
    assert "cube_poses" not in step_observation
    assert "cube_poses" not in env._exec_globals["obs"]


@pytest.mark.parametrize("api_cls", [FrankaControlApi, FrankaHandoverApi])
def test_non_privileged_api_get_observation_delegates_to_safe_low_level_contract(
    api_cls: type[Any],
) -> None:
    low_level_env = SafeLowLevelEnv()
    api = api_cls.__new__(api_cls)
    api._env = low_level_env

    observation = api.get_observation()

    assert observation == {"camera": "frame", "robot_joint_pos": "joints"}
    assert "cube_poses" not in observation
