FRANKA_TOOL_METADATA = {
    "get_observation": {
        "tags": ["perception"],
        "postconditions": ["observation_available"],
        "failure_modes": ["camera_unavailable"],
    },
    "segment_sam3_text_prompt": {
        "tags": ["perception"],
        "postconditions": ["non_empty_mask"],
        "failure_modes": ["object_not_found", "low_confidence_mask"],
    },
    "segment_sam3_point_prompt": {
        "tags": ["perception"],
        "postconditions": ["non_empty_mask"],
        "failure_modes": ["point_outside_object", "low_confidence_mask"],
    },
    "point_prompt_molmo": {
        "tags": ["perception"],
        "postconditions": ["point_prompt_available"],
        "failure_modes": ["point_not_found", "ambiguous_object"],
    },
    "mask_to_world_points": {
        "tags": ["perception", "geometry"],
        "preconditions": ["mask_available", "depth_available"],
        "postconditions": ["world_points_available"],
        "failure_modes": ["empty_mask", "invalid_depth"],
    },
    "get_oriented_bounding_box_from_3d_points": {
        "tags": ["geometry"],
        "preconditions": ["world_points_available"],
        "postconditions": ["object_bbox_available"],
        "failure_modes": ["insufficient_points"],
    },
    "plan_grasp": {
        "tags": ["planning"],
        "preconditions": ["segmentation_available", "depth_available"],
        "postconditions": ["grasp_candidates_available"],
        "failure_modes": ["no_grasp_candidates", "grasp_model_error"],
    },
    "select_top_down_grasp": {
        "tags": ["planning"],
        "preconditions": ["grasp_candidates_available"],
        "postconditions": ["selected_grasp_available"],
        "failure_modes": ["no_vertical_grasp", "low_grasp_score"],
    },
    "solve_ik": {
        "tags": ["planning"],
        "preconditions": ["target_pose_available"],
        "postconditions": ["joint_solution_valid"],
        "failure_modes": ["unreachable_pose", "invalid_pose", "ik_nonconvergence"],
    },
    "move_to_joints": {
        "tags": ["execution"],
        "preconditions": ["joint_solution_valid"],
        "postconditions": ["robot_reached_target"],
        "failure_modes": ["motion_timeout", "collision", "controller_error"],
    },
    "open_gripper": {
        "tags": ["execution"],
        "postconditions": ["gripper_open"],
        "failure_modes": ["controller_error"],
    },
    "close_gripper": {
        "tags": ["execution"],
        "postconditions": ["gripper_closed"],
        "failure_modes": ["controller_error", "object_slipped"],
    },
}
