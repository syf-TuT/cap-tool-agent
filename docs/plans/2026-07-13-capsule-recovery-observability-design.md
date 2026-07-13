# Capsule Recovery Observability Design

## Goal

Make runtime-control Capsule recovery usable for APIs that refresh state through task-specific
sensing functions, expose compact numeric state to the controller, and improve Two Arm Lift grasp
feedback without changing rollback behavior or unrelated task defaults.

## Scope and compatibility

- Keep rollback behavior unchanged.
- Preserve the existing `get_observation()` requirement for APIs such as `FrankaControlApi` that
  expose it, including Cube Lift.
- Let each API declare which side-effect-free functions constitute a fresh state read. A recovery
  must call at least one declared function.
- Preserve existing Two Arm Lift position getter signatures. Add separate grasp-pose getters so
  existing prompts, oracle code, and generated programs continue to run.
- Keep the global reward-drop guard defaults unchanged. Override the guard only in the standard
  non-privileged Two Arm Lift configuration.
- Keep existing recovery metric fields for compatibility and add an unambiguous effectiveness
  field based only on reward improvement.

## Recovery observation contract

`ApiBase` will expose a `recovery_observation_functions()` capability. Its default will return
`get_observation` when that function is public, so current Cube Lift behavior is unchanged.
`FrankaTwoArmLiftApi` will override the capability with its fresh handle and gripper pose getters.

The Capsule prompt and `append_recovery` validator will receive the union of capabilities from the
active APIs. Appended recovery code must call at least one of these functions. If an environment
declares none, recovery is rejected with a clear diagnostic instead of requiring an undefined
global.

## Compact numeric observability

Runtime trace and variable inspection will include values for bounded numeric arrays. Arrays with
at most 32 elements will be serialized to lists; larger arrays such as RGB, depth, masks, and point
clouds will retain shape and dtype summaries only. This gives the controller positions and
quaternions without injecting large observations into prompts.

## Two Arm Lift grasp poses

The existing handle perception pipeline will continue returning only bounding-box centers for
`get_handle0_pos()` and `get_handle1_pos()`. New `get_handle0_grasp_pose()` and
`get_handle1_grasp_pose()` functions will run Contact-GraspNet on the selected handle mask and
return the best world-frame position and quaternion. Empty or malformed candidate sets will raise
explicit errors.

## Reward guard and metrics

The standard non-privileged Two Arm Lift YAML will set a task-scale-appropriate reward guard
activation threshold. Global defaults remain untouched, so Cube Lift and other task configurations
retain their previous behavior.

`recovery_execution_improved` remains available for existing consumers. A new
`recovery_execution_effective` field will mean strictly that an attempted recovery increased the
reward. The existing reward- and trace-specific fields remain unchanged.

## Testing

- Verify Cube-style APIs still require `get_observation()`.
- Verify Two Arm-style APIs accept declared fresh sensing calls and reject blind recovery.
- Verify prompts document the active recovery functions.
- Verify small arrays expose values and large arrays do not.
- Verify Two Arm API capabilities and new public grasp-pose functions without changing position
  getter contracts.
- Verify the Two Arm configuration overrides only the reward guard and does not enable rollback.
- Run focused runtime-control and integration API tests, then the broader relevant test files.
