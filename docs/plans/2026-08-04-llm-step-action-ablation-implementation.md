# Cube-Stack Four-Group Ablation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore the generic cube-stack prompt and provide controlled A1-A4 configurations
covering one-shot generation and every patch/append-recovery combination.

**Architecture:** Keep the existing runtime permission implementation unchanged. Use the
ordinary code-agent path for the true one-shot A1 baseline and Capsule `llm_step` for A2-A4;
keep all shared YAML fields byte-for-byte equivalent apart from the documented group switches
and output paths. Lock the generic prompt and configuration matrix with regression tests.

**Tech Stack:** Python 3.10-3.12, pytest, YAML, Ruff, WSL2 Ubuntu runtime.

---

### Task 1: Add failing prompt and configuration regressions

**Files:**
- Modify: `tests/test_runtime_control_config.py`

**Step 1: Write the failing prompt test**

Import `PROMPT` from `capx.envs.tasks.franka.franka_pick_place` and assert exact equality with
the original generic prompt:

```python
expected = """
You are controlling a Franka Emika robot with FrankaControlApi.
Goal: pick up the red cube, place it on top of the green cube, then open the gripper.

Use non-privileged camera observations, SAM3 segmentation, Contact-GraspNet grasp
planning, Pyroki IK/planning, joint motion, and gripper commands. Do not use
privileged object-state shortcuts.

Write only executable Python code. Import numpy/scipy explicitly if needed. The
runtime-control Capsule llm_step loop may execute, inspect, and repair semantic
source groups after the initial program is generated.
"""
assert PROMPT == expected
```

Also assert the prompt does not contain the old answer-bearing phrases `half-height`,
`reuse the grasp quaternion`, `at least +0.2m`, or `stacking height formula`.

**Step 2: Write the failing A1-A4 matrix test**

Extend the current A2/A3 YAML test to load:

- `env_configs/cube_stack/franka_robosuite_cube_stack_a1_one_shot.yaml`;
- `env_configs/cube_stack/franka_robosuite_cube_stack_capsule_llm_step_a2_no_patch.yaml`;
- `env_configs/cube_stack/franka_robosuite_cube_stack_capsule_llm_step_a3_no_append_recovery.yaml`;
- `env_configs/cube_stack/franka_robosuite_cube_stack_capsule_llm_step_a4_full.yaml`.

Assert the exact matrix:

```python
assert (a1["agent_mode"], a1["capsule_llm_step_allow_patch"],
        a1["capsule_llm_step_allow_append_recovery"]) == ("code", False, False)
assert (a2["agent_mode"], a2["capsule_llm_step_allow_patch"],
        a2["capsule_llm_step_allow_append_recovery"]) == ("capsule", False, True)
assert (a3["agent_mode"], a3["capsule_llm_step_allow_patch"],
        a3["capsule_llm_step_allow_append_recovery"]) == ("capsule", True, False)
assert (a4["agent_mode"], a4["capsule_llm_step_allow_patch"],
        a4["capsule_llm_step_allow_append_recovery"]) == ("capsule", True, True)
```

Normalize A2-A4 by removing only the two permission keys and `output_dir`, then assert equality.
Normalize all four by additionally removing `agent_mode`, then assert equality. Keep the shared
non-VDM, API, recording, trial-count, and worker-count assertions.

**Step 3: Sync only the changed test into WSL**

Copy `tests/test_runtime_control_config.py` from the Windows checkout to
`/home/capx/code/cap-x/tests/test_runtime_control_config.py`.

**Step 4: Run the test to verify RED**

Run in `/home/capx/code/cap-x`:

```bash
uv run --no-sync pytest tests/test_runtime_control_config.py -q
```

Expected: FAIL because the prompt still leaks geometric instructions and A1/A4 do not exist.

### Task 2: Restore the generic task prompt

**Files:**
- Modify: `capx/envs/tasks/franka/franka_pick_place.py`
- Test: `tests/test_runtime_control_config.py`

**Step 1: Replace only the class-level task prompt**

Set `PROMPT` to the exact generic text asserted in Task 1. Do not modify `ORACLE_CODE`, runtime
control, simulator state, or API implementations.

**Step 2: Sync the prompt module into WSL**

Copy `capx/envs/tasks/franka/franka_pick_place.py` to the matching WSL project path.

**Step 3: Run the prompt test**

Run the prompt regression alone. Expected: PASS while the full configuration test file remains
RED because A1/A4 are still absent.

### Task 3: Add the A1 and A4 configurations

**Files:**
- Create: `env_configs/cube_stack/franka_robosuite_cube_stack_a1_one_shot.yaml`
- Create: `env_configs/cube_stack/franka_robosuite_cube_stack_capsule_llm_step_a4_full.yaml`
- Review: `env_configs/cube_stack/franka_robosuite_cube_stack_capsule_llm_step_a2_no_patch.yaml`
- Review: `env_configs/cube_stack/franka_robosuite_cube_stack_capsule_llm_step_a3_no_append_recovery.yaml`
- Test: `tests/test_runtime_control_config.py`

**Step 1: Create A1 from the shared A2/A3 structure**

Use `agent_mode: code`, retain `capsule_control_mode: llm_step` and all shared Capsule fields for
matrix comparability, and set both permissions to `false`. Do not add `multi_turn_prompt`; this
keeps code-agent execution to one initial model generation. Give A1 a unique output directory.

**Step 2: Create A4 from the same structure**

Use `agent_mode: capsule`, `capsule_control_mode: llm_step`, and set both permissions to `true`.
Give A4 a unique output directory.

**Step 3: Check A2 and A3 for exact shared-field identity**

Change A2/A3 only if required to make every non-group field identical to A1/A4.

**Step 4: Sync all four YAML files into WSL**

Copy the YAML files to `/home/capx/code/cap-x/env_configs/cube_stack/`.

**Step 5: Run the focused configuration tests**

```bash
uv run --no-sync pytest tests/test_runtime_control_config.py -q
```

Expected: PASS.

### Task 4: Verify and commit

**Files:**
- Review all changed prompt, YAML, test, and plan files.

**Step 1: Run focused tests in WSL**

```bash
uv run --no-sync pytest tests/test_runtime_control_config.py -q
```

Expected: all tests pass.

**Step 2: Run lint in WSL**

```bash
uv run --no-sync ruff check \
  capx/envs/tasks/franka/franka_pick_place.py \
  tests/test_runtime_control_config.py
```

Expected: Ruff reports no errors.

**Step 3: Inspect the final diff**

Run `git diff --check`, inspect the complete diff, and confirm no runtime implementation or
initial-state code changed.

**Step 4: Commit the implementation**

Stage only the prompt module, A1-A4 configs, regression test, and implementation plan, then
commit with an imperative subject describing the controlled four-group ablation.
