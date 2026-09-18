# Capsule Gate Failure Evidence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Persist immutable, independent failure evidence and child logs for Gates 2--6 without weakening Gate 6 checkpoint rollback.

**Architecture:** Add a small shared failure-envelope writer in `scripts/capsule_rl/common.py`. Wrap adapter execution in explicit stages, and make the outer wrapper promote staged child failure evidence or synthesize wrapper-level evidence while publishing captured logs exclusively.

**Tech Stack:** Python 3.10+, pathlib, subprocess, JSON, pytest, WSL2 `uv run --no-sync`.

---

### Task 1: Adapter failure evidence

**Files:**
- Modify: `tests/test_capsule_server_adapter.py`
- Modify: `scripts/capsule_rl/server_adapter.py`
- Modify: `scripts/capsule_rl/common.py`

**Step 1: Write failing tests**

Add focused tests for runtime-dispatch, post-Git, verifier, rollback, and immutable collision
failures. Assert the success path is absent and `<artifact>.failure.json` has the required
envelope and stage.

**Step 2: Verify RED**

Run in WSL:

```bash
uv run --no-sync pytest tests/test_capsule_server_adapter.py -q
```

Expected: the new assertions fail because no failure artifact exists.

**Step 3: Implement the minimum adapter flow**

Add shared helpers for failure paths/envelopes and stage `execute_gate` from validation through
publication. Preserve the original exception, append rollback failure details, write the failure
file exclusively, and re-raise.

**Step 4: Verify GREEN**

Run the same focused test command and expect all tests to pass.

### Task 2: Wrapper logs and staged failure promotion

**Files:**
- Modify: `tests/test_capsule_scripts.py`
- Modify: `scripts/capsule_rl/common.py`

**Step 1: Write failing tests**

Add tests for captured stdout/stderr, nonzero custom runner failures, staged child failure
promotion, verifier failures, direct Gate 6 failure reuse, and collision refusal.

**Step 2: Verify RED**

Run in WSL:

```bash
uv run --no-sync pytest tests/test_capsule_scripts.py -q
```

Expected: tests fail because subprocess output is not captured or persisted and staged failure
evidence is not promoted.

**Step 3: Implement wrapper behavior**

Run children with captured text output, publish immutable companion logs, promote child staging
failure evidence before cleanup, and synthesize wrapper failures when no valid child failure
exists.

**Step 4: Verify GREEN**

Run the focused tests again and expect all tests to pass.

### Task 3: Operator documentation and regression verification

**Files:**
- Modify: `docs/capsule_rl.md`

**Step 1: Document recovery semantics**

Explain success/failure/log path separation, immutable collision behavior, rollback metadata,
and which files operators must retain.

**Step 2: Run pure regression tests**

```bash
uv run --no-sync pytest tests/test_capsule_server_adapter.py tests/test_capsule_scripts.py tests/test_capsule_checkpoint.py -q
```

Expected: all selected pure tests pass without starting external services or simulators.

**Step 3: Inspect repository hygiene**

Run `git diff --check` and inspect the scoped diff for accidental changes to evaluator, schema,
analyzer, or server factory code.
