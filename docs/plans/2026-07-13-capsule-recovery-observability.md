# Capsule Recovery Observability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve Capsule recovery state validation and observability for Two Arm Lift while preserving Cube Lift behavior and leaving rollback unchanged.

**Architecture:** APIs declare their valid fresh-state recovery functions through a small capability method. Runtime control uses that declaration in both prompts and AST validation, while trace serialization exposes only bounded numeric arrays. Two Arm Lift gains separate Contact-GraspNet pose getters and a task-local reward guard override.

**Tech Stack:** Python 3.12, pytest, Ruff, YAML, NumPy, AST-based runtime action validation.

---

### Task 1: Recovery observation capabilities

**Files:**
- Modify: `capx/integrations/base_api.py`
- Modify: `capx/integrations/franka/two_arm_lift.py`
- Modify: `capx/envs/trial.py`
- Modify: `capx/runtime_control/prompts.py`
- Test: `tests/test_runtime_control_trial_loop.py`
- Test: `tests/test_runtime_control_prompts.py`

**Steps:**

1. Add failing tests proving a Cube-style API still requires `get_observation`, a Two Arm-style API accepts a declared handle getter, blind recovery is rejected, and the prompt names the active sensing functions.
2. Run the focused tests and confirm failures are caused by the missing capability behavior.
3. Add `ApiBase.recovery_observation_functions()`, collect capabilities in the trial loop, pass them to the prompt, and validate the recovery AST against them.
4. Override the capability in `FrankaTwoArmLiftApi` with fresh handle and gripper getters.
5. Run the focused tests until green.

### Task 2: Bounded numeric trace values

**Files:**
- Modify: `capx/runtime_control/trace.py`
- Modify: `capx/envs/trial.py`
- Test: `tests/test_runtime_control_trace.py`
- Test: `tests/test_runtime_control_trial_loop.py`

**Steps:**

1. Add failing tests showing arrays with at most 32 elements include numeric values and larger arrays omit values.
2. Run the tests and confirm the small-array assertions fail under the current shape-only summaries.
3. Implement the same bounded serialization rule in trace events and variable inspection.
4. Run focused tests until green.

### Task 3: Two Arm Contact-GraspNet pose getters

**Files:**
- Modify: `capx/integrations/franka/two_arm_lift.py`
- Test: `tests/test_two_arm_lift_api.py`

**Steps:**

1. Add failing tests for the new public functions and for pure best-candidate world-frame pose selection, including empty candidate handling.
2. Run the tests and confirm failure because the functions and helper do not exist.
3. Add `get_handle0_grasp_pose()` and `get_handle1_grasp_pose()` while preserving existing position getter behavior. Run Contact-GraspNet only for the new functions.
4. Run focused tests until green.

### Task 4: Task-local reward guard and precise metric

**Files:**
- Modify: `env_configs/two_arm_lift/franka_robosuite_two_arm_lift.yaml`
- Modify: `capx/envs/trial.py`
- Test: `tests/test_runtime_control_trial_loop.py`
- Test: `tests/test_two_arm_lift_config.py`

**Steps:**

1. Add failing tests for `recovery_execution_effective` and the Two Arm-only reward guard override, including absence of rollback settings.
2. Run the tests and confirm expected failures.
3. Add the strict reward-based metric alias and YAML override without changing global defaults.
4. Run focused tests until green.

### Task 5: Regression verification

**Files:**
- Verify all files above.

**Steps:**

1. Run the full runtime-control test set and Two Arm API/config tests.
2. Run Ruff on modified Python files.
3. Inspect `git diff --check` and the final diff for unintended rollback or Cube Lift configuration changes.
4. Report exact commands, results, and any remaining remote simulator validation requirement.
