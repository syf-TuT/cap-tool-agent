# Capsule Semantic Groups Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add semantic region groups to Capsule runtime-control so each Capsule action executes or patches a meaningful source chunk rather than a single Python statement.

**Architecture:** Keep atomic `CodeRegion` parsing and add `CodeRegionGroup` as the model-facing execution unit. Extend schema, prompt, executor dispatch, patching, feedback, and tests while preserving `run_region` compatibility.

**Tech Stack:** Python 3.12, dataclasses, AST analysis, pytest, CaP-X runtime-control modules.

---

### Task 1: Add Region Group Schema

**Files:**
- Modify: `capx/runtime_control/schema.py`
- Test: `tests/test_runtime_control_schema.py`

**Step 1: Write failing schema tests**

Add tests for:

- `CodeRegionGroup.to_dict()` includes `group_id`, source span, source, `region_ids`, `primitive_calls`, and `has_robot_side_effect`.
- `RuntimeAction.from_mapping()` accepts `run_group`.
- `RuntimeAction.from_mapping()` accepts `patch_group`.

**Step 2: Run tests to verify failure**

Run:

```bash
uv run --no-sync pytest tests/test_runtime_control_schema.py -q
```

Expected: fails because `CodeRegionGroup`, `run_group`, and `patch_group` do not exist.

**Step 3: Implement schema**

Add `CodeRegionGroup` dataclass with:

- `group_id: str`
- `start_line: int`
- `end_line: int`
- `source: str`
- `region_ids: list[str]`
- `primitive_calls: list[str]`
- `has_robot_side_effect: bool`

Add `run_group` and `patch_group` to `RuntimeActionName` and `SUPPORTED_ACTIONS`.

**Step 4: Re-run tests**

Run the same schema tests and expect PASS.

### Task 2: Build Deterministic Semantic Grouping

**Files:**
- Modify: `capx/runtime_control/segmenter.py`
- Test: `tests/test_runtime_control_segmenter.py`

**Step 1: Write failing segmenter tests**

Create tests that:

- preserve current `segment_python_code()` behavior;
- group adjacent setup statements before a robot side-effect call;
- mark groups containing `move_to_joints`, `close_gripper`, or `open_gripper` as side-effect groups;
- keep generated group ids stable after regrouping.

**Step 2: Run tests to verify failure**

Run:

```bash
uv run --no-sync pytest tests/test_runtime_control_segmenter.py -q
```

Expected: fails because grouping does not exist.

**Step 3: Implement grouping**

Add:

- `ROBOT_SIDE_EFFECT_CALLS`
- `PHASE_BOUNDARY_CALLS`
- `analyze_region_source()`
- `segment_python_code_groups(source, regions=None, max_regions_per_group=6)`

Use AST to extract function calls and names. Merge adjacent regions until a side-effect
boundary or max size boundary is reached.

**Step 4: Re-run tests**

Run the segmenter tests and expect PASS.

### Task 3: Add Group Execution and Patching

**Files:**
- Modify: `capx/envs/trial.py`
- Modify: `capx/runtime_control/patching.py` if needed
- Test: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write failing runtime tests**

Add tests that:

- scripted `run_group` executes all statements in a group;
- `patch_group` replaces the full group source span;
- patching a group triggers re-segmentation and regrouping;
- `summary.num_code_blocks` records atomic regions and trace records group id.

**Step 2: Run tests to verify failure**

Run:

```bash
uv run --no-sync pytest tests/test_runtime_control_trial_loop.py -q
```

Expected: fails because `_execute_runtime_action()` does not support groups.

**Step 3: Implement group dispatch**

In `_run_capsule_trial()`:

- build `groups = segment_python_code_groups(source, regions)`;
- build `group_by_id`;
- pass groups to the prompt;
- dispatch `run_group` through `executor.run_region(group_like_object)`;
- dispatch `patch_group` through `replace_region_source()` against the group span;
- after successful patch, rebuild regions and groups.

**Step 4: Re-run tests**

Run the trial-loop tests and expect PASS.

### Task 4: Update Prompt to Prefer Groups

**Files:**
- Modify: `capx/runtime_control/prompts.py`
- Test: `tests/test_runtime_control_prompts.py`

**Step 1: Write failing prompt tests**

Assert the Capsule prompt:

- includes generated code groups;
- documents `run_group`;
- documents `patch_group`;
- says groups are preferred for benchmark execution;
- still omits robot primitives as direct tools.

**Step 2: Run tests to verify failure**

Run:

```bash
uv run --no-sync pytest tests/test_runtime_control_prompts.py -q
```

Expected: fails because prompt only documents regions.

**Step 3: Implement prompt changes**

Change `build_capsule_prompt()` to accept `groups`. Keep `regions` optional or include atomic
regions after groups for repair detail. Update examples to use `run_group` first.

**Step 4: Re-run tests**

Run prompt tests and expect PASS.

### Task 5: Reduce Feedback Noise

**Files:**
- Modify: `capx/runtime_control/feedback.py`
- Test: `tests/test_runtime_control_feedback.py`

**Step 1: Write failing feedback tests**

Add tests that:

- pure computation/perception success without reward gain is `success`;
- robot side-effect success without reward gain is `warning`;
- `NameError` failures include a missing-variable repair hint;
- failed group feedback points patch scope to the group id.

**Step 2: Run tests to verify failure**

Run:

```bash
uv run --no-sync pytest tests/test_runtime_control_feedback.py -q
```

Expected: fails because all no-progress regions warn.

**Step 3: Implement feedback changes**

Extend `build_runtime_feedback()` to accept a region or group. Use `has_robot_side_effect` to
decide whether no-progress success is warning-worthy. Extract missing variable names from
`NameError` messages.

**Step 4: Re-run tests**

Run feedback tests and expect PASS.

### Task 6: Add Config Switch and Benchmark Defaults

**Files:**
- Modify: `capx/utils/launch_utils.py`
- Modify: `env_configs/benchmarks/strict_l1/cube_stack_capsule.yaml`
- Modify: `env_configs/benchmarks/lowlevel_primitives/cube_stack_capsule.yaml`
- Test: `tests/test_runtime_control_config.py`

**Step 1: Write failing config tests**

Assert `_load_config()` reads:

- `capsule_execution_granularity`;
- `capsule_max_regions_per_group`;

**Step 2: Run tests to verify failure**

Run:

```bash
uv run --no-sync pytest tests/test_runtime_control_config.py -q
```

Expected: fails because config keys are not loaded.

**Step 3: Implement config defaults**

Add defaults:

- `capsule_execution_granularity: semantic_group`
- `capsule_max_regions_per_group: 20`

Set cube-stack Capsule benchmark `max_capsule_steps: 20`.

**Step 4: Re-run tests**

Run config tests and expect PASS.

### Task 7: Focused Verification

**Files:**
- No code edits unless tests expose issues.

**Step 1: Run local focused tests in WSL**

Run through WSL per `AGENTS.md`:

```bash
uv run --no-sync pytest \
  tests/test_runtime_control_schema.py \
  tests/test_runtime_control_segmenter.py \
  tests/test_runtime_control_prompts.py \
  tests/test_runtime_control_feedback.py \
  tests/test_runtime_control_trial_loop.py \
  tests/test_runtime_control_config.py \
  -q
```

Expected: PASS.

**Step 2: Sync to remote**

Copy modified files to `/root/autodl-tmp/cap-x` on SeeTaCloud.

**Step 3: Run remote focused tests**

Run the same focused pytest command remotely.

Expected: PASS.

**Step 4: Run remote seed-3 smoke**

Run cube-stack Capsule strict L1 with:

- `seed=3`
- `max_capsule_steps=20`
- semantic groups
- streaming enabled
- reasoning disabled
- video enabled

Expected: trace shows `run_group` actions and reaches more task phases in fewer actions.

**Step 5: Run seeds 1-5 pilot**

Run Capsule seeds 1-5 and compare with existing baseline pilot outputs.

Expected: report success rate, reward, group count, atomic region count, patch count, time, and video paths.
