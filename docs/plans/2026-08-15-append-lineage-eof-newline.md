# Append Recovery EOF Newline Lineage Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow append recovery to preserve executed EOF lineage when segmentation adds only a trailing newline.

**Architecture:** Keep the existing span-and-source reconciliation algorithm. Add one narrowly
scoped source-equivalence helper for final pre-append units, and exercise it through the public
`reconcile_lineage` path with executed region and group keys.

**Tech Stack:** Python 3.12, dataclasses, pytest, Ruff, WSL2 Ubuntu runtime.

---

### Task 1: Add the failing lineage regression

**Files:**
- Test: `tests/test_runtime_control_lineage.py`

**Step 1: Write the failing test**

Add a test with old source `goto_home_joint_position()` and appended source ending the old unit
as `goto_home_joint_position()\n`. Mark both the final region and group stable keys as executed,
then assert reconciliation preserves those keys.

**Step 2: Run the test to verify it fails**

Run in WSL:

```bash
uv run --no-sync pytest tests/test_runtime_control_lineage.py::test_append_maps_executed_eof_units_when_only_trailing_newline_changes -q
```

Expected: fail with `executed region key could not be mapped`.

### Task 2: Implement the narrow source equivalence

**Files:**
- Modify: `capx/runtime_control/lineage.py`
- Test: `tests/test_runtime_control_lineage.py`

**Step 1: Implement the minimal helper**

Accept exact equality first. Only for `append_recovery` and a previous unit ending at
`old_line_count`, compare sources after stripping terminal CR/LF characters.

**Step 2: Add the safety assertion**

Verify that a substantive change to the final executed unit remains rejected even if the
candidate ends with a newline.

**Step 3: Run the focused tests**

```bash
uv run --no-sync pytest tests/test_runtime_control_lineage.py -q
```

Expected: all lineage tests pass.

### Task 3: Verify and commit

**Files:**
- Modify: `capx/runtime_control/lineage.py`
- Modify: `tests/test_runtime_control_lineage.py`

**Step 1: Run related runtime-control tests**

```bash
uv run --no-sync pytest tests/test_runtime_control_lineage.py tests/test_runtime_control_trial_loop.py -q
uv run --no-sync ruff check capx/runtime_control/lineage.py tests/test_runtime_control_lineage.py
```

Expected: all tests and Ruff checks pass.

**Step 2: Inspect the final diff**

Confirm no unrelated or cache files are staged.

**Step 3: Commit**

```bash
git add capx/runtime_control/lineage.py tests/test_runtime_control_lineage.py docs/plans/2026-08-15-append-lineage-eof-newline-design.md docs/plans/2026-08-15-append-lineage-eof-newline.md
git commit -m "Fix append lineage matching at EOF"
```
