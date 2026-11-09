from capx.envs.tasks.base import CodeExecutionEnvBase

PROMPT = """
You are controlling a Franka Emika robot with FrankaControlApi.
Goal: pick up the red cube, place it on top of the green cube, then open the gripper.

Use non-privileged camera observations, SAM3 segmentation, Contact-GraspNet grasp
planning, Pyroki IK/planning, joint motion, and gripper commands. Do not use
privileged object-state shortcuts.

Write only executable Python code. Import numpy/scipy explicitly if needed. The
runtime-control Capsule llm_step loop may execute, inspect, and repair semantic
source groups after the initial program is generated.
"""
ORACLE_CODE = """
import numpy as np

_, _, green_ext = get_object_pose("green cube", return_bbox_extent=True)
_, _, red_ext = get_object_pose("red cube", return_bbox_extent=True)

# Sample a grasp pose for the red cube and pick it up
pick_pos, pick_quat = sample_grasp_pose("red cube")
goto_pose(pick_pos, pick_quat, z_approach=0.1)
close_gripper()
# Lift the red cube after grasping
post_pick_pos = pick_pos.copy()
post_pick_pos[2] += 0.2
goto_pose(post_pick_pos, pick_quat)

# Compute placement pose on top of the green cube
green_pos, _, _ = get_object_pose("green cube", return_bbox_extent=False)

place_pos = green_pos.copy()
place_pos[2] = green_pos[2] + green_ext[2]/2 + red_ext[2]/2
# Use down orientation for placement
place_quat = np.array([0.0, 0.0, 1.0, 0.0])

# Approach and place the red cube on the green cube
goto_pose(place_pos, pick_quat, z_approach=0.1)
open_gripper()

# Retract after placing
post_place_pos = place_pos.copy()
post_place_pos[2] += 0.1
goto_pose(post_place_pos, place_quat)
"""


# ---------------------------- High-level Env -----------------------------
class FrankaPickPlaceCodeEnv(CodeExecutionEnvBase):
    """High-level code environment for Franka pick-and-place using SimpleExecutor."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE


__all__ = [
    "FrankaPickPlaceCodeEnv",
]
