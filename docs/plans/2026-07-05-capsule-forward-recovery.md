# Capsule Forward Recovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enforce no-rollback Capsule recovery by adding append-only current-state recovery and blocking patch/rerun of already executed robot-side-effect source units.

**Architecture:** Extend runtime action schema with `append_recovery`. Track executed side-effect regions and groups in the Capsule trial loop. Reject patching or rerunning historical side-effect units, and append recovery source only when it starts from a fresh `get_observation()` call.

**Tech Stack:** Python 3.10-3.12, AST parsing, dataclasses, pytest, CaP-X runtime-control modules.

---

### Task 1: Add Append-Recovery Schema

**Files:**
- Modify: `capx/runtime_control/schema.py`
- Test: `tests/test_runtime_control_schema.py`

**Step 1: Write the failing test**

Add:

```python
def test_runtime_action_accepts_append_recovery():
    action = RuntimeAction.from_mapping(
        {"action": "append_recovery", "args": {"source": "obs = get_observation()"}}
    )

    assert action.action == "append_recovery"
    assert action.args["source"] == "obs = get_observation()"
```

**Step 2: Run the test to verify failure**

Run in WSL:

```bash
uv run --no-sync pytest tests/test_runtime_control_schema.py::test_runtime_action_accepts_append_recovery -q
```

Expected: fails with unsupported runtime action.

**Step 3: Implement the minimal schema change**

Add `append_recovery` to `RuntimeActionName` and `SUPPORTED_ACTIONS`.

**Step 4: Run the test to verify pass**

Run the same pytest command. Expected: PASS.

### Task 2: Document Append Recovery in Prompts

**Files:**
- Modify: `capx/runtime_control/prompts.py`
- Test: `tests/test_runtime_control_prompts.py`

**Step 1: Write the failing prompt test**

Assert prompt text includes:

- `append_recovery`
- an example JSON object with `args.source`
- `get_observation()`
- current physical state recovery language

**Step 2: Run the test to verify failure**

Run:

```bash
uv run --no-sync pytest tests/test_runtime_control_prompts.py::test_capsule_prompt_documents_append_recovery -q
```

Expected: fails because prompt does not document the action.

**Step 3: Update prompt text**

Add `append_recovery` to allowed actions and examples. State that appended recovery code must
take a fresh observation and continue from current physical state.

**Step 4: Run the prompt tests**

Run:

```bash
uv run --no-sync pytest tests/test_runtime_control_prompts.py -q
```

Expected: PASS.

### Task 3: Validate and Apply Append Recovery

**Files:**
- Modify: `capx/envs/trial.py`
- Test: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write failing append tests**

Add trial-loop tests that:

- scripted `append_recovery` appends source, re-segments, and lets a later `run_group`
  execute the new recovery group;
- `append_recovery` without `get_observation()` returns invalid and fails the trial.

**Step 2: Run tests to verify failure**

Run:

```bash
uv run --no-sync pytest tests/test_runtime_control_trial_loop.py::test_capsule_trial_appends_recovery_and_regroups tests/test_runtime_control_trial_loop.py::test_append_recovery_requires_fresh_observation -q
```

Expected: fails because action is unsupported.

**Step 3: Implement append handling**

In `_execute_runtime_action()`:

- validate `args.source`;
- parse it with `ast.parse`;
- require a `get_observation` call;
- append it to `source` with blank-line separation;
- return updated source in event evidence.

In `_run_capsule_trial()`, treat successful `append_recovery` like a successful patch for
source replacement and regrouping.

**Step 4: Run append tests**

Run the same focused tests. Expected: PASS.

### Task 4: Track Executed Side-Effect Units

**Files:**
- Modify: `capx/envs/trial.py`
- Test: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write failing guard tests**

Add tests that:

- running a side-effect group twice returns invalid on the second run;
- patching a side-effect group after it has executed returns invalid and recommends
  `append_recovery`;
- non-side-effect groups remain rerunnable or patchable.

**Step 2: Run tests to verify failure**

Run:

```bash
uv run --no-sync pytest tests/test_runtime_control_trial_loop.py::test_capsule_trial_rejects_rerun_of_executed_side_effect_group tests/test_runtime_control_trial_loop.py::test_capsule_trial_rejects_patch_of_executed_side_effect_group -q
```

Expected: fails because the trial loop does not track executed side effects.

**Step 3: Implement guard**

In `_run_capsule_trial()`:

- maintain `executed_side_effect_regions: set[str]`;
- maintain `executed_side_effect_groups: set[str]`;
- before dispatch, reject run/patch actions targeting already executed side-effect units;
- after successful run of a side-effect group or region, record its id;
- include clear invalid event messages.

**Step 4: Run guard tests**

Run the same focused tests. Expected: PASS.

### Task 5: Update Feedback Hints

**Files:**
- Modify: `capx/runtime_control/feedback.py`
- Test: `tests/test_runtime_control_feedback.py`

**Step 1: Write/update tests**

Assert side-effect warnings mention:

- no rollback;
- fresh observation;
- `append_recovery`.

**Step 2: Run tests**

```bash
uv run --no-sync pytest tests/test_runtime_control_feedback.py -q
```

Expected before implementation: fails if `append_recovery` is missing.

**Step 3: Implement feedback wording**

Update warning hints to recommend append recovery for current-state repair.

**Step 4: Re-run tests**

Expected: PASS.

### Task 6: Focused Verification and Commit

**Files:**
- All modified runtime-control, trial-loop, config, docs, and tests.

**Step 1: Run focused test suite**

Run in WSL:

```bash
uv run --no-sync pytest tests/test_runtime_control_schema.py tests/test_runtime_control_prompts.py tests/test_runtime_control_feedback.py tests/test_runtime_control_config.py tests/test_runtime_control_trial_loop.py -q
```

Expected: PASS.

**Step 2: Inspect diff**

Run:

```bash
git diff --stat
git diff --check
```

Expected: no whitespace errors; diff only includes intended docs, runtime-control, trial-loop,
config, and tests.

**Step 3: Commit**

Stage only intended files, excluding `.codex-results/`, then commit:

```bash
git add docs/plans/2026-07-05-capsule-forward-recovery-design.md docs/plans/2026-07-05-capsule-forward-recovery.md capx/runtime_control/schema.py capx/runtime_control/prompts.py capx/runtime_control/feedback.py capx/envs/trial.py capx/utils/launch_utils.py tests/test_runtime_control_schema.py tests/test_runtime_control_prompts.py tests/test_runtime_control_feedback.py tests/test_runtime_control_config.py tests/test_runtime_control_trial_loop.py env_configs/benchmarks/strict_l1/cube_stack_capsule.yaml env_configs/benchmarks/strict_l1/nut_assembly_capsule.yaml env_configs/benchmarks/lowlevel_primitives/cube_stack_capsule.yaml env_configs/benchmarks/lowlevel_primitives/nut_assembly_capsule.yaml env_configs/cube_stack/franka_robosuite_cube_stack_capsule_vdm.yaml env_configs/nut_assembly/franka_robosuite_nut_assembly_privileged_capsule_deepseek_v4_flash.yaml
git commit -m "Enforce no-rollback capsule recovery"
```
