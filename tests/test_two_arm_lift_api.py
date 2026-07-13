import numpy as np
import pytest

from capx.integrations.franka import two_arm_lift
from capx.envs.tasks.franka.two_arm_lift import PROMPT

FrankaTwoArmLiftApi = two_arm_lift.FrankaTwoArmLiftApi


def test_two_arm_api_declares_task_specific_recovery_observation_functions():
    api = FrankaTwoArmLiftApi.__new__(FrankaTwoArmLiftApi)

    assert api.recovery_observation_functions() == {
        "get_handle0_pos",
        "get_handle1_pos",
        "get_handle0_grasp_pose",
        "get_handle1_grasp_pose",
        "get_arm0_gripper_pose",
        "get_arm1_gripper_pose",
    }


def test_two_arm_api_exposes_grasp_pose_without_changing_position_getters():
    api = FrankaTwoArmLiftApi.__new__(FrankaTwoArmLiftApi)
    functions = api.functions()

    assert "get_handle0_pos" in functions
    assert "get_handle1_pos" in functions
    assert "get_handle0_grasp_pose" in functions
    assert "get_handle1_grasp_pose" in functions


def test_two_arm_grasp_pose_getter_requests_contact_graspnet_planning():
    api = FrankaTwoArmLiftApi.__new__(FrankaTwoArmLiftApi)
    calls = []
    expected_position = np.array([0.1, 0.2, 0.3])
    expected_quaternion = np.array([1.0, 0.0, 0.0, 0.0])

    def fake_get_pose(object_name, index=0, *, plan_grasp=False):
        calls.append((object_name, index, plan_grasp))
        return expected_position, expected_quaternion, np.zeros(3)

    api._get_handle_pose_with_graspnet = fake_get_pose

    position, quaternion = api.get_handle0_grasp_pose()

    assert calls == [("The green square frame", 0, True)]
    assert position is expected_position
    assert quaternion is expected_quaternion


def test_two_arm_task_prompt_documents_grasp_pose_getters():
    assert "get_handle0_grasp_pose()" in PROMPT
    assert "get_handle1_grasp_pose()" in PROMPT


def test_best_grasp_pose_world_selects_highest_score_and_applies_approach_offset():
    samples = np.stack([np.eye(4), np.eye(4)])
    samples[0, :3, 3] = [9.0, 9.0, 9.0]
    samples[1, :3, 3] = [1.0, 2.0, 3.0]

    position, quaternion = two_arm_lift._best_grasp_pose_world(
        samples,
        np.array([0.1, 0.9]),
        camera_pose=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(position, [1.0, 2.0, 3.12])
    np.testing.assert_allclose(quaternion, [1.0, 0.0, 0.0, 0.0])


def test_best_grasp_pose_world_rejects_empty_candidates():
    with pytest.raises(ValueError, match="no grasp candidates"):
        two_arm_lift._best_grasp_pose_world(
            np.empty((0, 4, 4)),
            np.empty((0,)),
            camera_pose=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        )


def test_best_grasp_pose_world_applies_camera_rotation_and_translation():
    sample = np.eye(4)[None, ...]
    sample[0, :3, 3] = [1.0, 0.0, 0.0]
    half_sqrt = np.sqrt(0.5)

    position, quaternion = two_arm_lift._best_grasp_pose_world(
        sample,
        np.array([1.0]),
        camera_pose=np.array([10.0, 0.0, 0.0, half_sqrt, 0.0, 0.0, half_sqrt]),
    )

    np.testing.assert_allclose(position, [10.0, 1.0, 0.12], atol=1e-7)
    np.testing.assert_allclose(quaternion, [half_sqrt, 0.0, 0.0, half_sqrt], atol=1e-7)
