# Robosuite Non-Privileged Observation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make every non-privileged Robosuite experiment obtain observations from Robosuite with object observables disabled while preserving privileged ground-truth observations and task evaluation.

**Architecture:** Store the privilege mode in `RobosuiteBaseEnv`, pass it to every Robosuite task constructor as `use_object_obs=privileged`, and guard task-specific pose synthesis behind `self.privileged`. Keep reward, success, and debug-only ground-truth access on explicit private simulator-state paths so those values never enter public non-privileged observations.

**Tech Stack:** Python 3.12, Gymnasium, Robosuite/MuJoCo, pytest, uv, WSL2 Ubuntu 22.04.

---

### Task 1: Lock the source-level privilege contract

**Files:**
- Create: `tests/test_robosuite_observation_privilege.py`
- Modify: `capx/envs/simulators/robosuite_base.py:41-75`
- Modify: `capx/envs/simulators/robosuite_cubes.py:51-100`
- Modify: `capx/envs/simulators/robosuite_cube_lift.py:45-94`
- Modify: `capx/envs/simulators/robosuite_cubes_restack.py:269-326`
- Modify: `capx/envs/simulators/robosuite_handover.py:58-114`
- Modify: `capx/envs/simulators/robosuite_two_arm_lift.py:55-108`
- Modify: `capx/envs/simulators/robosuite_spill_wipe.py:46-96`
- Modify: `capx/envs/simulators/robosuite_nut_assembly.py:58-108`

**Step 1: Write the failing base-state test**

Add a test which constructs `RobosuiteBaseEnv(privileged=False)` and
`RobosuiteBaseEnv(privileged=True)` and asserts that each instance exposes the matching
`privileged` value.

```python
def test_robosuite_base_retains_privilege_mode():
    assert RobosuiteBaseEnv(privileged=False).privileged is False
    assert RobosuiteBaseEnv(privileged=True).privileged is True
```

**Step 2: Sync the new test to WSL and verify RED**

Run from elevated PowerShell:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/cp /mnt/f/code/cap-x/tests/test_robosuite_observation_privilege.py tests/test_robosuite_observation_privilege.py
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin MUJOCO_GL=egl /home/capx/.local/bin/uv run --no-sync pytest tests/test_robosuite_observation_privilege.py::test_robosuite_base_retains_privilege_mode -q
```

Expected: FAIL because `RobosuiteBaseEnv` does not retain `privileged`.

**Step 3: Implement the minimal base change**

In `RobosuiteBaseEnv.__init__`, add:

```python
self.privileged = privileged
```

Sync the file to WSL and rerun the focused test. Expected: PASS.

**Step 4: Write failing constructor tests**

Parametrize all seven wrapper classes. Monkeypatch the relevant Robosuite task constructor with a
callable which records `use_object_obs` and raises a local sentinel exception immediately. For each
wrapper, assert:

```python
@pytest.mark.parametrize("privileged", [False, True])
def test_wrapper_selects_object_observations_at_robosuite_source(..., privileged):
    with pytest.raises(ConstructorCaptured):
        wrapper_cls(privileged=privileged)
    assert captured_kwargs["use_object_obs"] is privileged
```

Cover these constructor targets:

- `stack.Stack` for Cube Stack and Cube Restack
- the Lift task used by Cube Lift
- the TwoArmHandover task used by Handover
- `TwoArmLift` for Two-arm Lift
- `Wipe` for Spill Wipe
- `NutAssemblySquare` for Nut Assembly

**Step 5: Verify constructor tests are RED**

Run the new test module in WSL. Expected: at least the non-privileged cases fail because the
wrappers currently omit `use_object_obs=False`; Handover and Two-arm Lift currently force `True`.

**Step 6: Implement source-level selection**

In every Robosuite constructor call, pass:

```python
use_object_obs=privileged
```

Do this in privileged/non-rendering, privileged/rendering, and non-privileged branches so every
construction path is explicit.

**Step 7: Verify GREEN**

Sync the modified simulator files to WSL and rerun
`tests/test_robosuite_observation_privilege.py`. Expected: all source-selection tests PASS.

**Step 8: Commit**

```bash
git add tests/test_robosuite_observation_privilege.py capx/envs/simulators/robosuite_base.py capx/envs/simulators/robosuite_cubes.py capx/envs/simulators/robosuite_cube_lift.py capx/envs/simulators/robosuite_cubes_restack.py capx/envs/simulators/robosuite_handover.py capx/envs/simulators/robosuite_two_arm_lift.py capx/envs/simulators/robosuite_spill_wipe.py capx/envs/simulators/robosuite_nut_assembly.py
git commit -m "Select Robosuite object observations by privilege"
```

### Task 2: Make Cube observations work without object observables

**Files:**
- Modify: `tests/test_robosuite_observation_privilege.py`
- Modify: `capx/envs/simulators/robosuite_cubes.py:199-219`
- Modify: `capx/envs/simulators/robosuite_cube_lift.py:173-190`
- Modify: `capx/envs/simulators/robosuite_cubes_restack.py:476-495`

**Step 1: Write failing non-privileged observation tests**

Build each wrapper with `__new__`, set `privileged=False`, and attach a fake `robosuite_env` whose
`_get_observations()` returns camera/robot fields but no cube fields. Stub only camera/gripper
post-processing when their geometry is irrelevant to this behavior.

Assert that `get_observation()`:

- does not raise;
- preserves the fake camera/robot fields;
- does not add `cube_poses`.

**Step 2: Verify RED**

Run the three focused tests in WSL. Expected: FAIL with missing `cubeA_pos` / `cubeB_pos` or another
object-key lookup.

**Step 3: Implement the minimal conditional pose synthesis**

For each Cube wrapper:

```python
robosuite_obs = self.robosuite_env._get_observations()
if self.privileged:
    pose_dict = self._cube_pose_dict(robosuite_obs)
    robosuite_obs["cube_poses"] = ...
```

Run common camera and gripper processing after the conditional.

**Step 4: Add privileged regression tests**

Use a fake object observation with valid cube positions/quaternions and assert the existing
`cube_poses` structure is still produced when `privileged=True`.

Do not change quaternion ordering in this task.

**Step 5: Verify GREEN and commit**

Run the test module, then:

```bash
git add tests/test_robosuite_observation_privilege.py capx/envs/simulators/robosuite_cubes.py capx/envs/simulators/robosuite_cube_lift.py capx/envs/simulators/robosuite_cubes_restack.py
git commit -m "Separate Cube observations by privilege"
```

### Task 3: Make the remaining task observations privilege-aware

**Files:**
- Modify: `tests/test_robosuite_observation_privilege.py`
- Modify: `capx/envs/simulators/robosuite_handover.py:470-590`
- Modify: `capx/envs/simulators/robosuite_two_arm_lift.py:461-575`
- Modify: `capx/envs/simulators/robosuite_nut_assembly.py:335-405`
- Inspect: `capx/envs/simulators/robosuite_spill_wipe.py:167-202`

**Step 1: Write failing non-privileged tests**

For Handover, Two-arm Lift, and Nut Assembly, feed object-free native Robosuite observations to
`get_observation()` with `privileged=False`. Assert that camera and robot processing succeeds and
these keys are absent:

```python
assert "hammer_poses" not in observation
assert "pot_poses" not in observation
assert "nut_poses" not in observation
```

Add a Spill Wipe regression proving its object-free observation path remains usable.

**Step 2: Verify RED**

Run only the new tests. Expected: the first three fail because their pose helpers are currently
called unconditionally.

**Step 3: Guard task-pose synthesis**

Only call `_hammer_pose_dict`, `_pot_pose_dict`, or `_get_nut_pose` and add the derived dictionary
when `self.privileged` is true. Keep camera, robot joint, gripper, and Cartesian pose construction
outside the guard.

**Step 4: Add privileged regression tests**

Provide valid fake object state and assert all existing derived task-pose dictionaries remain
available in privileged mode.

**Step 5: Verify GREEN and commit**

```bash
git add tests/test_robosuite_observation_privilege.py capx/envs/simulators/robosuite_handover.py capx/envs/simulators/robosuite_two_arm_lift.py capx/envs/simulators/robosuite_nut_assembly.py capx/envs/simulators/robosuite_spill_wipe.py
git commit -m "Separate Robosuite task observations by privilege"
```

### Task 4: Decouple internal evaluation and Viser from public observations

**Files:**
- Modify: `tests/test_robosuite_observation_privilege.py`
- Modify: `capx/envs/simulators/robosuite_cubes_restack.py:401-495`
- Modify as needed: `capx/envs/simulators/robosuite_cube_lift.py:195-220`
- Modify as needed: `capx/envs/simulators/robosuite_handover.py:647-790`
- Modify as needed: `capx/envs/simulators/robosuite_two_arm_lift.py:647-740`
- Modify as needed: `capx/envs/simulators/robosuite_nut_assembly.py:407-530`

**Step 1: Write failing Cube Restack evaluation tests**

Set up Cube Restack in non-privileged mode with a fake public observation lacking cube fields and
an internal simulator state containing the cube body positions. Assert `compute_reward()` and
`task_completed()` return their expected values without requiring public object observations.

**Step 2: Verify RED**

Expected: FAIL because the custom paths call `_get_observations()` and `_cube_pose_dict()`.

**Step 3: Add a private simulator-state helper**

Introduce a narrowly scoped private helper that reads the two cube body transforms directly from
MuJoCo for evaluation. Use it only from `compute_reward()`, `task_completed()`, and optional debug
visualization. Do not merge its result into `get_observation()`.

**Step 4: Make debug paths tolerate non-privileged observations**

Where Viser currently assumes a derived pose dictionary exists, either obtain debug-only state
through an explicit private helper or skip that ground-truth overlay. Camera and robot frames must
continue updating.

**Step 5: Verify GREEN and commit**

```bash
git add tests/test_robosuite_observation_privilege.py capx/envs/simulators/robosuite_cubes_restack.py capx/envs/simulators/robosuite_cube_lift.py capx/envs/simulators/robosuite_handover.py capx/envs/simulators/robosuite_two_arm_lift.py capx/envs/simulators/robosuite_nut_assembly.py
git commit -m "Keep Robosuite evaluation state internal"
```

### Task 5: Prove there is no high-level observation bypass

**Files:**
- Modify: `tests/test_robosuite_observation_privilege.py`
- Modify if needed: `tests/test_runtime_control_globals.py`
- Modify if needed: `capx/envs/tasks/base.py:154-206,242-285`
- Modify if needed: `capx/integrations/franka/control.py:89-108`
- Modify if needed: `capx/integrations/franka/handover.py:64-82`

**Step 1: Write high-level propagation tests**

Use a fake non-privileged low-level environment whose `get_observation()` returns a sentinel safe
observation. Prove the same safe contract is used by:

- `reset()` and `step()`;
- execution globals `obs` and `INPUTS`;
- `FrankaControlApi.get_observation()`;
- `FrankaHandoverApi.get_observation()`;
- Capsule recovery-observation discovery.

Also include a forbidden sentinel ground-truth key in a separate internal-state fixture and prove
it never appears in those public paths.

**Step 2: Verify the tests**

These tests should pass without production changes if all delegation already respects the repaired
low-level boundary. If one fails, make the smallest delegation fix and first confirm that the test
failed for the bypass being corrected.

**Step 3: Run runtime-control regressions**

Run in WSL:

```text
pytest tests/test_robosuite_observation_privilege.py tests/test_runtime_control_globals.py tests/test_runtime_control_trial_loop.py tests/test_runtime_control_prompts.py -q
```

Expected: PASS.

**Step 4: Commit**

Commit only if this task changed files:

```bash
git add tests/test_robosuite_observation_privilege.py tests/test_runtime_control_globals.py capx/envs/tasks/base.py capx/integrations/franka/control.py capx/integrations/franka/handover.py
git commit -m "Verify non-privileged observation propagation"
```

### Task 6: Focused simulator verification

**Files:**
- Modify only if a verified regression requires a fix: relevant files above

**Step 1: Sync all modified files to WSL**

Copy the changed Windows files from `/mnt/f/code/cap-x` into `/home/capx/code/cap-x`, preserving
their relative paths. Do not run experiments from the Windows checkout.

**Step 2: Run the full focused test set**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin MUJOCO_GL=egl /home/capx/.local/bin/uv run --no-sync pytest tests/test_robosuite_observation_privilege.py tests/test_runtime_control_globals.py tests/test_runtime_control_trial_loop.py tests/test_runtime_control_prompts.py -q
```

Expected: PASS with no warnings introduced by this change.

**Step 3: Run a non-privileged Cube Stack observation smoke test**

Instantiate the non-privileged Cube Stack environment, reset it, and assert camera/robot fields are
present while raw cube fields and `cube_poses` are absent. Use a focused pytest integration test or
a temporary non-committed WSL script.

Expected: reset succeeds and no object-state key is exposed.

**Step 4: Run the privileged oracle regression**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin MUJOCO_GL=egl /home/capx/.local/bin/uv run --no-sync capx/envs/launch.py --config-path env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml --use-oracle-code True --total-trials 1 --num-workers 1 --record-video False
```

Expected: reward `1.0` and `Task Completed: True`.

**Step 5: Run lint on touched files**

```text
ruff check tests/test_robosuite_observation_privilege.py capx/envs/simulators/robosuite_base.py capx/envs/simulators/robosuite_cubes.py capx/envs/simulators/robosuite_cube_lift.py capx/envs/simulators/robosuite_cubes_restack.py capx/envs/simulators/robosuite_handover.py capx/envs/simulators/robosuite_two_arm_lift.py capx/envs/simulators/robosuite_spill_wipe.py capx/envs/simulators/robosuite_nut_assembly.py
```

Expected: PASS.

**Step 6: Review the final diff**

Confirm that:

- every non-privileged constructor uses `use_object_obs=False`;
- no blacklist-based filtering was introduced;
- no public non-privileged path synthesizes task ground truth;
- privileged observations and task completion still work;
- quaternion conversion was not changed as part of this fix.

