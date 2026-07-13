# Capsule Initial Syntax Recovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let runtime-control Capsule repair syntactically invalid initial Python and retry syntactically invalid source patches without terminating the trial.

**Architecture:** Reuse `patch_group` by exposing invalid initial source as a temporary whole-source group with a step-zero syntax diagnostic. Validate the complete patched program before committing either group or region replacements, then use the existing regrouping path after a valid patch.

**Tech Stack:** Python 3.12, `ast`, pytest, CaP-X runtime-control schemas and trial loop.

---

### Task 1: Specify initial syntax recovery

**Files:**
- Modify: `tests/test_runtime_control_trial_loop.py`
- Test: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write the failing test**

Add a test that calls `_run_capsule_trial` with `initial_code="value = (\n"`,
then scripts `patch_group(group_1)` with a valid complete program, runs the
regrouped group, and finishes. Assert that the trial succeeds, the first real
action is the patch, and the trace contains the initial syntax diagnostic.

**Step 2: Run the focused test to verify it fails**

Run from the prepared WSL project:

```bash
uv run --no-sync pytest tests/test_runtime_control_trial_loop.py::test_capsule_repairs_invalid_initial_source_with_patch_group -q
```

Expected: FAIL because `segment_python_code` raises before the Capsule loop.

**Step 3: Implement whole-source fallback units**

In `capx/envs/trial.py`, add small helpers that:

```python
def _whole_source_fallback_units(source: str) -> tuple[list[CodeRegion], list[CodeRegionGroup]]:
    line_count = max(1, len(source.splitlines()))
    region = CodeRegion("region_1", 1, line_count, source)
    group = CodeRegionGroup("group_1", 1, line_count, source, region_ids=["region_1"])
    return [region], [group]
```

Catch only `SyntaxError` during initial segmentation, use these units, and seed
history with a step-zero diagnostic containing `lineno`, `offset`, and `text`.

**Step 4: Run the focused test to verify it passes**

Run the same focused pytest command. Expected: PASS.

### Task 2: Reject invalid source patches without aborting

**Files:**
- Modify: `tests/test_runtime_control_trial_loop.py`
- Modify: `capx/envs/trial.py`
- Test: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write the failing test**

Add a test starting from invalid initial source. Script one invalid
`patch_group` replacement followed by a valid replacement, execution, and
finish. Assert the invalid action is recorded as `invalid`, the original source
is retained for the retry, and the later valid patch executes.

**Step 2: Run the focused test to verify it fails**

```bash
uv run --no-sync pytest tests/test_runtime_control_trial_loop.py::test_capsule_retries_after_syntax_error_in_group_patch -q
```

Expected: FAIL because regrouping the first invalid candidate raises
`SyntaxError` out of the trial loop.

**Step 3: Validate complete candidate source**

In `_execute_runtime_action`, parse the complete candidate after
`replace_region_source` for both `patch_region` and `patch_group`. Return an
`invalid` `RuntimeEvent` with structured syntax evidence when parsing fails;
only put `source` into event evidence for a valid candidate.

**Step 4: Run both focused tests**

```bash
uv run --no-sync pytest \
  tests/test_runtime_control_trial_loop.py::test_capsule_repairs_invalid_initial_source_with_patch_group \
  tests/test_runtime_control_trial_loop.py::test_capsule_retries_after_syntax_error_in_group_patch -q
```

Expected: 2 passed.

### Task 3: Regression verification

**Files:**
- Verify: `capx/envs/trial.py`
- Verify: `tests/test_runtime_control_trial_loop.py`

**Step 1: Run the runtime-control trial-loop suite**

```bash
uv run --no-sync pytest tests/test_runtime_control_trial_loop.py -q
```

Expected: all tests pass.

**Step 2: Run focused lint checks**

```bash
uv run --no-sync ruff check capx/envs/trial.py tests/test_runtime_control_trial_loop.py
```

Expected: no lint errors.

**Step 3: Inspect the final diff**

Confirm that changes are limited to syntax recovery, patch validation, tests,
and these plan documents.

