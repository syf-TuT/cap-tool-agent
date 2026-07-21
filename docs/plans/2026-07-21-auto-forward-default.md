# Auto-Forward Default Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `auto_forward` the default Capsule control mode without removing explicit `llm_step` support.

**Architecture:** Update the launch-time merged configuration and the runtime dispatch fallback so both entry paths agree. Protect each fallback with a focused regression assertion and retain existing explicit-mode coverage.

**Tech Stack:** Python 3.10+, pytest, uv, WSL2 Ubuntu runtime

---

### Task 1: Specify the new defaults

**Files:**
- Modify: `tests/test_runtime_control_config.py`
- Modify: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write the failing configuration assertion**

Add this assertion to `test_load_config_reads_capsule_fields`:

```python
assert config["capsule_control_mode"] == "auto_forward"
```

**Step 2: Write the failing runtime dispatch test**

Add a test that replaces both loop functions, calls `_run_capsule_trial` with an
empty configuration, and asserts that the auto-forward sentinel is returned
without invoking the `llm_step` loop.

**Step 3: Run tests to verify they fail**

Run in the prepared WSL checkout:

```bash
uv run --no-sync pytest \
  tests/test_runtime_control_config.py::test_load_config_reads_capsule_fields \
  tests/test_runtime_control_trial_loop.py::test_capsule_trial_defaults_to_auto_forward -q
```

Expected: both tests fail because the current fallback is `llm_step`.

### Task 2: Change both fallback values

**Files:**
- Modify: `capx/utils/launch_utils.py`
- Modify: `capx/envs/trial.py`

**Step 1: Implement the minimal change**

Replace only the two omitted-value fallbacks:

```python
configs_dict.get("capsule_control_mode", "auto_forward")
config.get("capsule_control_mode", "auto_forward")
```

**Step 2: Run the focused tests**

Run the two tests from Task 1. Expected: PASS.

**Step 3: Run regression coverage**

```bash
uv run --no-sync pytest \
  tests/test_runtime_control_config.py \
  tests/test_runtime_control_trial_loop.py -q
```

Expected: all tests pass, including explicit `llm_step` coverage.

**Step 4: Commit and push**

```bash
git add capx/envs/trial.py capx/utils/launch_utils.py \
  tests/test_runtime_control_config.py tests/test_runtime_control_trial_loop.py
git commit -m "Default Capsule control to auto-forward"
git push origin feature/Micro-step-CaP-Agent0
```

