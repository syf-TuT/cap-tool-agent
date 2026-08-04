# LLM-Step Action Ablation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add enforceable patch and append-recovery permission switches for Capsule `llm_step`
and ship non-VDM cube-stack A2/A3 experiment configurations.

**Architecture:** Resolve two backward-compatible booleans in config loading and at the start
of the `llm_step` loop. Pass them into prompt construction so forbidden actions are not
advertised, and apply a loop-local policy guard before the existing safety guards so model and
scripted actions cannot bypass the experiment condition. Keep the global runtime schema and
`auto_forward` behavior unchanged.

**Tech Stack:** Python 3.10-3.12, pytest, YAML, Ruff, WSL2 Ubuntu runtime.

---

### Task 1: Load the permission fields

**Files:**
- Modify: `tests/test_runtime_control_config.py`
- Modify: `capx/utils/launch_utils.py`

**Step 1: Write the failing tests**

Extend the default config assertion with:

```python
assert config["capsule_llm_step_allow_patch"] is True
assert config["capsule_llm_step_allow_append_recovery"] is True
```

Add explicit YAML values to `test_load_config_reads_compact_llm_step_prompt_fields` and assert
that both are loaded unchanged.

**Step 2: Verify RED in WSL**

Sync the edited test to `/home/capx/code/cap-x` and run:

```bash
uv run --no-sync pytest tests/test_runtime_control_config.py -q
```

Expected: FAIL because the merged config lacks the two keys.

**Step 3: Implement minimal config loading**

Add to `merged_config`:

```python
"capsule_llm_step_allow_patch": configs_dict.get(
    "capsule_llm_step_allow_patch", True
),
"capsule_llm_step_allow_append_recovery": configs_dict.get(
    "capsule_llm_step_allow_append_recovery", True
),
```

**Step 4: Verify GREEN**

Sync `capx/utils/launch_utils.py` and rerun the focused config test. Expected: PASS.

### Task 2: Filter the Capsule prompt

**Files:**
- Modify: `tests/test_runtime_control_prompts.py`
- Modify: `capx/runtime_control/prompts.py`

**Step 1: Write failing prompt tests**

Add one test with `allow_patch=False` asserting that `patch_group` and `patch_region` do not
appear anywhere in the prompt while `append_recovery` remains advertised. Add another with
`allow_append_recovery=False` asserting the inverse. Include an API recovery function in the
second test so omission is caused by the switch rather than unavailable API support.

**Step 2: Verify RED**

Run:

```bash
uv run --no-sync pytest \
  tests/test_runtime_control_prompts.py::test_capsule_prompt_omits_patch_actions_when_disabled \
  tests/test_runtime_control_prompts.py::test_capsule_prompt_omits_append_recovery_when_disabled \
  -q
```

Expected: FAIL because `build_capsule_prompt` does not accept the switches.

**Step 3: Implement prompt filtering**

Add keyword arguments defaulting to `True`. Build allowed actions, recovery guidance, examples,
constraints, preferences, and schema rules conditionally. Pass the resolved strings through both
the normal and compact-budget fallback prompt builders. Do not change defaults.

**Step 4: Verify GREEN and prompt regression coverage**

Run all of `tests/test_runtime_control_prompts.py`. Expected: PASS.

### Task 3: Enforce permissions in `llm_step`

**Files:**
- Modify: `tests/test_runtime_control_trial_loop.py`
- Modify: `capx/envs/trial.py`

**Step 1: Write failing runtime tests**

Add A2 tests showing:

- a scripted `patch_group` produces an `invalid` event and leaves saved source unchanged;
- a valid `append_recovery` succeeds when patch is disabled.

Add A3 tests showing:

- a scripted `append_recovery` produces an `invalid` event and leaves saved source unchanged;
- a valid `patch_group` succeeds when append recovery is disabled.

Assert trace messages name the disabled configuration field. Assert step metrics contain both
resolved permission booleans.

**Step 2: Verify RED**

Run the four new focused tests. Expected: FAIL because forbidden actions still execute and metric
fields do not exist.

**Step 3: Implement the minimal policy guard**

Resolve both fields with `_coerce_config_bool`. Pass them into `build_capsule_prompt`. Add a
small helper returning an `invalid` `RuntimeEvent` for disabled `patch_group`/`patch_region` or
`append_recovery`, and invoke it before `_no_rollback_guard_event`. Record both booleans on every
step metric.

**Step 4: Verify GREEN and loop regressions**

Run the new tests, then all of `tests/test_runtime_control_trial_loop.py`. Expected: PASS.

### Task 4: Add runnable non-VDM A2/A3 configurations

**Files:**
- Modify: `tests/test_runtime_control_config.py`
- Create: `env_configs/cube_stack/franka_robosuite_cube_stack_capsule_llm_step_a2_no_patch.yaml`
- Create: `env_configs/cube_stack/franka_robosuite_cube_stack_capsule_llm_step_a3_no_append_recovery.yaml`

**Step 1: Write a failing configuration-matrix test**

Load both YAML files and assert:

```python
assert a2["agent_mode"] == a3["agent_mode"] == "capsule"
assert a2["capsule_control_mode"] == a3["capsule_control_mode"] == "llm_step"
assert (a2["capsule_llm_step_allow_patch"], a2["capsule_llm_step_allow_append_recovery"]) == (False, True)
assert (a3["capsule_llm_step_allow_patch"], a3["capsule_llm_step_allow_append_recovery"]) == (True, False)
assert not a2.get("use_img_differencing", False)
assert not a3.get("use_img_differencing", False)
```

Also compare the environment, API servers, trial count, worker count, and all shared Capsule
settings; only permissions and output paths may differ.

**Step 2: Verify RED**

Run the new test. Expected: FAIL because both files are absent.

**Step 3: Add both YAML files**

Copy the non-VDM environment and server topology from
`env_configs/cube_stack/franka_robosuite_cube_stack.yaml`. Add Capsule `llm_step` settings, the
A2/A3 permission matrix, explicit `use_img_differencing: false`, and distinct output paths.

**Step 4: Verify GREEN**

Run `tests/test_runtime_control_config.py`. Expected: PASS.

### Task 5: Refactor and verify

**Files:**
- Review all changed implementation, tests, YAML, and plan files.

**Step 1: Simplify without changing behavior**

Remove duplicated prompt fragments where a small conditional helper improves readability. Keep
the runtime guard local to `llm_step` and avoid adding global schema state.

**Step 2: Run focused verification in WSL**

```bash
uv run --no-sync pytest \
  tests/test_runtime_control_config.py \
  tests/test_runtime_control_prompts.py \
  tests/test_runtime_control_trial_loop.py -q
uv run --no-sync ruff check \
  capx/utils/launch_utils.py \
  capx/runtime_control/prompts.py \
  capx/envs/trial.py \
  tests/test_runtime_control_config.py \
  tests/test_runtime_control_prompts.py \
  tests/test_runtime_control_trial_loop.py
```

Expected: all tests pass and Ruff reports no errors.

**Step 3: Inspect the final diff and status**

Confirm defaults remain permissive, `auto_forward` is untouched, the two YAML files are non-VDM,
and no unrelated files changed.

