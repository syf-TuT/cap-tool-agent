# Capsule Quarantined Repair Draft Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow an initially invalid Capsule program to be repaired one semantic group at a
time without ever executing a partially repaired source revision.

**Architecture:** Keep AST-valid invalid source in an explicit repair-pending state. Source
edits remain transactional, but a candidate with fewer normalized diagnostics may commit as
a quarantined draft; execution stays globally blocked until the complete analysis is valid.
Syntax-invalid or strict-preflight-invalid source continues to use the whole-source fallback.

**Tech Stack:** Python 3.12, dataclasses, `collections.Counter`, pytest, existing Capsule
runtime-control analysis and lineage APIs.

---

### Task 1: Define deterministic partial-repair progress

**Files:**
- Modify: `capx/envs/trial.py:1170-1378`
- Test: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write failing unit tests**

Add focused tests for a helper that compares current and candidate diagnostics. Cover:

- removing one of two normalized violations is progress;
- line and group-id renumbering do not create a new diagnostic identity;
- replacing an old violation with a different violation is not progress;
- an unchanged violation multiset is not progress;
- a fully valid candidate is progress from an invalid source.

Construct `ProgramContractViolation` values directly so the tests isolate comparison logic.

**Step 2: Run the focused tests and verify RED**

Run in the WSL runtime snapshot:

```bash
/home/capx/code/cap-x/.venv/bin/python -m pytest \
  tests/test_runtime_control_trial_loop.py -k partial_repair_progress -q
```

Expected: collection or assertion failure because the helper does not exist.

**Step 3: Implement the minimal comparison helper**

In `capx/envs/trial.py`:

- normalize whitespace in the diagnostic message;
- fingerprint `code`, normalized message, `helper_name`, and `side_effect_calls`;
- compare fingerprints as `Counter` multisets;
- return true only when the candidate multiset is a proper sub-multiset of the current one.

Do not compare line spans, region ids, or group ids.

**Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

**Step 5: Commit**

```bash
git add capx/envs/trial.py tests/test_runtime_control_trial_loop.py
git commit -m "Define Capsule partial repair progress"
```

### Task 2: Commit improving drafts and block their execution

**Files:**
- Modify: `capx/envs/trial.py:1206-1378,1718-2154,2933-3027`
- Test: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write the failing end-to-end regression test**

Create an AST-valid initial program with strict or contract violations in two distinct
semantic groups. Script these actions:

1. patch the first group;
2. attempt to run the remaining effectful group;
3. patch the second group;
4. run a now-valid group.

Assert that:

- the first patch succeeds and increments the source revision;
- the run attempt is rejected with `repair_pending` safety evidence and performs no API call;
- the second patch succeeds, clears repair mode, and increments the revision again;
- the final execution succeeds;
- the first patch's metrics still report one or more remaining violations.

Add separate regression cases proving that a non-improving draft patch and a valid-to-invalid
patch remain atomic rejections.

**Step 2: Run the new tests and verify RED**

```bash
/home/capx/code/cap-x/.venv/bin/python -m pytest \
  tests/test_runtime_control_trial_loop.py \
  -k 'quarantined_repair_draft or non_improving_repair or valid_source_rejects_invalid_patch' \
  -q
```

Expected: the first partial patch is rejected by the current global candidate validation.

**Step 3: Implement repair-draft admission**

- Pass the current analysis diagnostics and lineage execution state into
  `_prepare_capsule_source_edit()`.
- Continue rejecting syntax errors and all structural/lineage/recovery failures.
- For a currently valid source, continue rejecting any candidate strict or contract
  violation.
- For a currently invalid source, permit a remaining-invalid `patch_group` or
  `patch_region` only when no side-effect key has executed and the diagnostic comparison
  reports strict progress.
- Reject `append_recovery` while current source diagnostics remain unresolved.
- Atomically commit accepted draft source, analysis, groups, maps, lineage, and revision
  through the existing `_PreparedSourceEdit` path.

**Step 4: Implement the repair-pending execution guard**

Add a guard before the existing strict and contract guards. While any current source
diagnostic remains, reject `run_group`, `run_region`, and `resume_from_region` with:

```python
evidence={
    # Preserve the existing strict/program-contract safety classification.
    "safety_failure": "program_contract_violation",
    "repair_pending": True,
    "remaining_violation_count": ...,
}
```

Syntax-only repair state uses `repair_pending` as the safety classification. Strict-subset
violations continue through the existing strict-subset guard for compatibility.

Patch actions remain available. An accepted partial patch event reports
`repair_pending=True` and its remaining count without marking the successful edit itself as
a safety failure.

**Step 5: Run the focused tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass and fake API calls occur only
after the final valid repair.

**Step 6: Commit**

```bash
git add capx/envs/trial.py tests/test_runtime_control_trial_loop.py
git commit -m "Quarantine partial Capsule repairs"
```

### Task 3: Expose repair state and run regressions

**Files:**
- Modify: `capx/runtime_control/prompts.py:82-335,341-510`
- Modify: `capx/envs/trial.py:1506-1521,2255-2320`
- Test: `tests/test_runtime_control_prompts.py`
- Test: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write failing prompt and metric tests**

Verify that a prompt with unresolved diagnostics includes explicit quarantined-repair
guidance: continue with a source patch and do not run any code. Verify step metrics contain
`repair_pending` and `remaining_violation_count`, and that both clear after the final repair.

**Step 2: Run the focused tests and verify RED**

```bash
/home/capx/code/cap-x/.venv/bin/python -m pytest \
  tests/test_runtime_control_prompts.py \
  tests/test_runtime_control_trial_loop.py \
  -k 'repair_pending or quarantined_repair_draft' -q
```

Expected: assertions fail because prompt and metrics do not expose repair state.

**Step 3: Implement prompt and metric fields**

- Derive prompt repair state from the non-empty contract-violation context.
- State that all execution actions are quarantined until every listed violation is repaired.
- Add `repair_pending` and `remaining_violation_count` to each step metric from the current
  post-action analysis.
- Keep existing contract and strict-subset metric fields unchanged for compatibility.

**Step 4: Run focused and regression tests**

```bash
/home/capx/code/cap-x/.venv/bin/python -m pytest \
  tests/test_runtime_control_contract.py \
  tests/test_runtime_control_prompts.py \
  tests/test_runtime_control_trial_loop.py -q
```

Expected: all tests pass with no new warnings.

**Step 5: Run lint on changed Python files**

```bash
/home/capx/code/cap-x/.venv/bin/python -m ruff check \
  capx/envs/trial.py capx/runtime_control/prompts.py \
  tests/test_runtime_control_trial_loop.py tests/test_runtime_control_prompts.py
```

Expected: exit code 0.

**Step 6: Commit**

```bash
git add capx/envs/trial.py capx/runtime_control/prompts.py \
  tests/test_runtime_control_trial_loop.py tests/test_runtime_control_prompts.py
git commit -m "Expose Capsule repair draft state"
```
