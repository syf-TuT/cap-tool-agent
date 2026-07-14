# Robosuite Non-Privileged Observation Design

## Problem

CaP-X currently uses the same low-level `get_observation()` path for privileged and
non-privileged Robosuite experiments. Several task wrappers create Robosuite with object
observables enabled and then add derived ground-truth task poses such as `cube_poses`,
`hammer_poses`, `pot_poses`, and `nut_poses` to the returned dictionary. Non-privileged APIs,
the code-execution globals, and Capsule recovery can therefore observe simulator ground truth
instead of relying on RGB-D perception.

This affects the local Robosuite wrappers for Cube Stack, Cube Lift, Cube Restack, Handover,
Two-arm Lift, Nut Assembly, and Spill Wipe.

## Required Semantics

Observation privilege must be selected at the Robosuite source, not implemented by deleting
fields after a full observation has been produced.

- A privileged environment creates Robosuite with `use_object_obs=True` and may expose derived
  task-pose fields.
- A non-privileged environment creates Robosuite with `use_object_obs=False` and receives only
  the available camera, depth, and robot proprioceptive observations.
- The public `get_observation()` function remains available because Capsule recovery relies on
  it, but it follows the privilege mode of the environment.
- Visual APIs such as `get_object_pose()` continue to estimate poses from the non-privileged
  RGB-D observation.
- Reward calculation, success checks, and optional debug visualization may read MuJoCo state
  through explicit private helpers, but must not expose those values through public observations,
  `obs`, or `INPUTS`.

## Architecture

`RobosuiteBaseEnv` stores the requested `privileged` mode. Every task wrapper passes
`use_object_obs=privileged` when constructing its Robosuite environment.

Each task-level `get_observation()` first obtains the native Robosuite observation. Common
camera and robot-state processing runs in both modes. Ground-truth pose conversion and derived
pose dictionaries run only in privileged mode:

```python
obs = self.robosuite_env._get_observations(...)

if self.privileged:
    obs["task_poses"] = self._task_pose_dict(obs)

self._process_camera_observations(obs)
self._compute_gripper_obs(obs)
return obs
```

No blacklist or post-hoc field deletion is used. In non-privileged mode, task object keys do not
exist; callers that try to access them receive the normal `KeyError` rather than a zero pose or a
silently substituted perception estimate.

Cube Restack currently uses object observation fields during custom reward and completion
checks. That logic will move to an explicitly internal simulator-state helper so task evaluation
continues to work when public object observables are disabled. Viser ground-truth overlays will
likewise use private simulator-state access or tolerate the absence of derived poses.

## Public Data Flow

```text
privileged=True
  -> use_object_obs=True
  -> camera + robot state + object state
  -> derived ground-truth task poses
  -> privileged APIs

privileged=False
  -> use_object_obs=False
  -> camera + depth + robot state
  -> no ground-truth task-pose conversion
  -> visual APIs estimate object poses from RGB-D
```

The same semantics apply to observations returned by `reset()` and `step()`, the persistent
execution globals `obs` and `INPUTS`, direct API calls to `get_observation()`, and Capsule recovery.

## Compatibility and Error Handling

- Function names and Capsule recovery-observation contracts remain unchanged.
- Privileged task APIs retain their existing ground-truth pose inputs.
- Non-privileged visual APIs retain their camera calibration and robot-state inputs.
- Non-privileged code that depended on leaked object-state keys intentionally breaks with a
  missing-key error; this is the security boundary being restored.
- Reward and completion behavior must remain identical across privilege modes.
- This change does not modify quaternion conversion. The separate XYZW/WXYZ issue should be
  fixed and tested independently.

## Test Strategy

Development follows test-driven development:

1. Add failing tests showing that `RobosuiteBaseEnv` retains the privilege mode.
2. Add constructor tests for all affected wrappers proving that non-privileged construction uses
   `use_object_obs=False` and privileged construction uses `use_object_obs=True`.
3. Feed object-free fake Robosuite observations into non-privileged task wrappers and prove that
   `get_observation()` succeeds, preserves camera and robot state, and does not synthesize task
   pose dictionaries.
4. Prove that privileged observations retain their current task pose dictionaries.
5. Test Cube Restack reward and completion through internal simulator state without public object
   observables.
6. Run high-level observation and Capsule recovery tests to detect bypasses through `reset()`,
   `step()`, `obs`, `INPUTS`, or API delegation.
7. Run focused pytest and Robosuite smoke tests in the prepared WSL environment, including both a
   non-privileged Cube Stack run and the existing privileged oracle regression.

