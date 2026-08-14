# Capsule Single-Effect Group Normalization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `llm_step` receive semantic groups that do not combine distinct robot side-effect regions, so valid robot execution can begin instead of spending every step patching contract-invalid groups.

**Architecture:** Keep the LLM-step controller, action budget, prompts, and YAML unchanged. Teach the metadata-only normalizer to split before a second robot-effect region and make group-count reduction refuse effectful-to-effectful merges; when safety and `max_groups` conflict, safety wins and the maximum is soft.

**Tech Stack:** Python 3.12, AST-derived CaP-X runtime metadata, pytest, Ruff, WSL2 runtime verification, SeeTaCloud LIBERO smoke experiment.

---

### Task 1: Lock the single-effect behavior with failing tests

**Files:**
- Modify: `tests/test_runtime_control_normalizer.py:123-326`
- Modify: `tests/test_runtime_control_contract.py:1137-1151`

**Step 1: Replace the consecutive-effects expectation**

Rename `test_normalizer_merges_consecutive_effects_into_sense_act_block` to
`test_normalizer_splits_consecutive_robot_effects` and assert these ordered partitions:

```python
assert [group.region_ids for group in groups] == [
    ["region_1", "region_2", "region_3", "region_4", "region_5"],
    ["region_6"],
    ["region_7", "region_8"],
]
assert all(group.has_robot_side_effect for group in groups)
```

This proves setup stays with the first action, the immediately following `close_gripper()` gets
its own group, and later setup stays with the following motion.

**Step 2: Update the realistic pick/place regression**

Keep the existing source and assert seven ordered groups:

```python
assert [group.region_ids for group in groups] == [
    [f"region_{i}" for i in range(1, 9)],
    ["region_9"],
    ["region_10"],
    ["region_11", "region_12", "region_13"],
    ["region_14", "region_15", "region_16"],
    ["region_17"],
    ["region_18", "region_19"],
]
```

**Step 3: Make the max-group policy explicitly soft for effect groups**

Rename `test_normalizer_reduces_many_single_effect_groups_within_policy_band` to
`test_normalizer_keeps_effect_groups_separate_above_policy_maximum` and assert that all ten
effect phases remain separate and source partitioning is lossless:

```python
assert len(groups) == 10
assert all(group.has_robot_side_effect for group in groups)
```

Rename `test_normalizer_finds_safe_non_greedy_merge_path` to
`test_normalizer_does_not_merge_effectful_groups_under_group_pressure` and assert the four
two-region groups remain unchanged even with `max_groups=2`.

**Step 4: Turn the contract contradiction into an integration regression**

Rename `test_counts_repeated_side_effect_occurrences_within_a_group` to
`test_normalized_consecutive_side_effects_do_not_violate_group_contract` and assert:

```python
assert not [
    violation
    for violation in _analyze(source)
    if violation.code == "multiple_effects_in_group"
]
```

The contract's existing control-flow and explicitly constructed multi-effect-group tests remain
responsible for proving that a genuinely indivisible multi-effect region is still rejected.

**Step 5: Synchronize only the modified test files into WSL**

First verify the runnable copy has no changes to the target files:

```bash
git diff --quiet -- tests/test_runtime_control_normalizer.py tests/test_runtime_control_contract.py
```

Then copy the two worktree files from `/mnt/f/code/cap-x/.worktrees/capsule-single-effect-groups/`
to `/home/capx/code/cap-x/tests/`.

**Step 6: Run the tests and verify RED**

Run:

```bash
uv run --no-sync pytest \
  tests/test_runtime_control_normalizer.py::test_normalizer_splits_consecutive_robot_effects \
  tests/test_runtime_control_normalizer.py::test_normalizer_keeps_effect_groups_separate_above_policy_maximum \
  tests/test_runtime_control_normalizer.py::test_normalizer_does_not_merge_effectful_groups_under_group_pressure \
  tests/test_runtime_control_contract.py::test_normalized_consecutive_side_effects_do_not_violate_group_contract -q
```

Expected: assertion failures showing consecutive effects remain merged and group-count bounding
still combines effectful groups. No import, fixture, or collection error is acceptable.

### Task 2: Implement the minimal normalizer correction

**Files:**
- Modify: `capx/runtime_control/normalizer.py:71-91`
- Modify: `capx/runtime_control/normalizer.py:329-354`

**Step 1: Track robot effects separately from structural effects**

Add `current_has_robot_effect = False`. Before appending each region, split when the current group
already has a robot effect and the incoming region also has one:

```python
starts_second_robot_effect = (
    current_has_robot_effect and analysis.has_robot_side_effect
)
if current and (
    starts_second_robot_effect
    or returns_to_sense
    or len(current) >= policy.max_regions_per_group
):
    ...
```

Reset the flag when flushing and update it after appending the region. Keep
`current_has_effect` unchanged for the existing return-to-sense behavior.

**Step 2: Forbid effectful-to-effectful bounding merges**

Change `_can_merge_adjacent_groups` so the safety condition is:

```python
if left.has_robot_side_effect and right.has_robot_side_effect:
    return False
```

Keep the existing ordering and dependency checks. This also permits a non-effect group to merge
with one neighboring effect group when dependency-safe, without ever producing two effects.

**Step 3: Synchronize `normalizer.py` into WSL and verify GREEN**

Run the four RED tests again. Expected: `4 passed`.

**Step 4: Run focused regression suites**

Run:

```bash
uv run --no-sync pytest \
  tests/test_runtime_control_normalizer.py \
  tests/test_runtime_control_contract.py \
  tests/test_runtime_control_trial_loop.py \
  tests/test_runtime_control_config.py -q
```

Expected: all tests pass. Then run Ruff on the two changed Python files:

```bash
uv run --no-sync ruff check \
  capx/runtime_control/normalizer.py \
  tests/test_runtime_control_normalizer.py \
  tests/test_runtime_control_contract.py
```

Expected: no diagnostics.

**Step 5: Commit the implementation**

```bash
git add capx/runtime_control/normalizer.py \
  tests/test_runtime_control_normalizer.py \
  tests/test_runtime_control_contract.py \
  docs/plans/2026-08-14-capsule-single-effect-group-normalization-design.md \
  docs/plans/2026-08-14-capsule-single-effect-group-normalization-implementation.md
git commit -m "Fix capsule single-effect group normalization"
```

### Task 3: Integrate and verify the target branch

**Files:**
- No additional source files.

**Step 1: Fast-forward the requested feature branch**

From `F:/code/cap-x`, fast-forward `feature/libero-capsule-llm-step` to the verified worktree
branch. Do not stage or modify the unrelated Molmo cache files in the main checkout.

**Step 2: Verify the resulting diff and history**

Run:

```bash
git diff 0679222e --check
git log --oneline 0679222e..HEAD
```

Expected: only the design/plan, normalizer, and two test files are present; no whitespace errors.

### Task 4: Run the remote LIBERO llm_step smoke experiment

**Files:**
- Verify only: `env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml`

**Step 1: Use the SeeTaCloud experiment skill and synchronize the verified commit**

Follow `run-seetacloud-capx-experiment`. Confirm the remote checkout is on
`feature/libero-capsule-llm-step` at the new commit and reuse the existing Hugging Face/Molmo
caches; do not download the model again.

**Step 2: Verify the required runtime configuration**

Confirm:

```yaml
capsule_control_mode: llm_step
use_visual_feedback: false
use_img_differencing: false
use_video_differencing: false
```

Confirm PackyAPI text completion works through `https://cf.api.fan`, reasoning is disabled, and
the existing `allenai/Molmo2-8B` perception service is healthy on port 8122. Molmo must remain
internal to `FrankaLiberoApi`.

**Step 3: Run exactly one trial**

```bash
source .venv-libero/bin/activate
MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml \
  --total-trials 1 --num-workers 1
```

**Step 4: Inspect the runtime trace before judging task success**

Acceptance for the minimal loop is at least one actual execution action (`run_group` or recorded
robot side-effect trace), with no `multiple_effects_in_group` rejection caused by normalization.
If the trace still contains only successful patches and no execution, stop and design the
separate `llm_step` no-progress/budget guard; do not change control modes.

**Step 5: Report the complete outcome**

Report reward, task completed, sandbox return code, relevant stderr/error log, action sequence,
and the absolute remote output directory.
