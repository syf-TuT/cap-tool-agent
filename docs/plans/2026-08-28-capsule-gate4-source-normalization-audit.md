# Capsule Gate4 Source-Normalization Audit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align Gate4's fenced-P0 audit with canonical clean replay while retaining raw-source provenance and explicit protocol-edit lineage.

**Architecture:** Reuse `normalize_program_source` at the Gate4 verifier boundary to recompute the expected executed-source hash from the immutable raw P0. Keep the existing collector outcome checks and protocol-unit/edit checks, replacing only the obsolete fence-caused SyntaxError assertion with strict normalization diagnostics.

**Tech Stack:** Python 3.12, pytest, Capsule typed replay schemas, SHA-256 provenance, SeeTaCloud A800 runtime.

---

### Task 1: Add the Gate4 normalization-audit regression

**Files:**
- Modify: `tests/test_capsule_scripts.py:3789-3914`

**Step 1: Write the failing test fixture**

Change the fenced P0 fixture from a fence-caused `SyntaxError` to a normalized
`TASK_FAILURE`. Persist `raw_source_sha256`, `executed_source_sha256`, and
`source_normalized=true` in diagnostics.

Add focused tests that mutate the fixture to remove `source_normalized` or forge
`executed_source_sha256`, expecting a `GateArtifactError` that names the invalid normalization
evidence.

**Step 2: Run the focused tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_capsule_scripts.py::test_collector_verifier_accepts_explicit_fence_and_suffix_deletions \
  tests/test_capsule_scripts.py::test_collector_verifier_rejects_missing_fenced_source_normalization \
  tests/test_capsule_scripts.py::test_collector_verifier_rejects_mismatched_fenced_executed_source_hash \
  -q
```

Expected: the acceptance test fails with the stale message `fenced Actor P0 must preserve its
SyntaxError`, while the new rejection expectations are not yet implemented.

### Task 2: Implement strict source-normalization verification

**Files:**
- Modify: `scripts/capsule_rl/common.py:22-45,1622-1661`
- Test: `tests/test_capsule_scripts.py`

**Step 1: Import the canonical normalizer and source hash helper**

Import `source_sha256` from `capx.rl.capsule.schema` and `normalize_program_source` from
`capx.utils.program_source`.

**Step 2: Replace the obsolete SyntaxError assertion**

For fenced P0 traces, require:

```python
diagnostics = p0_result.diagnostics
expected_executed_source = normalize_program_source(trace.base_source)
if diagnostics.get("source_normalized") is not True:
    raise GateArtifactError("fenced Actor P0 must prove source normalization")
if diagnostics.get("raw_source_sha256") != source_sha256(trace.base_source):
    raise GateArtifactError("fenced Actor P0 raw source hash is invalid")
if diagnostics.get("executed_source_sha256") != source_sha256(expected_executed_source):
    raise GateArtifactError("fenced Actor P0 executed source hash is invalid")
```

Leave exact protocol-unit discovery, explicit deletion count, no-op rejection, and edit ordering
unchanged.

**Step 3: Run focused tests to verify GREEN**

Run the three focused tests from Task 1 plus
`test_collector_verifier_rejects_whole_program_cleanup_of_fenced_p0`.

Expected: all four pass.

### Task 3: Update the canonical documentation and run regressions

**Files:**
- Modify: `docs/capsule_rl.md:356-374`

**Step 1: Document the new execution/provenance split**

State that clean replay executes canonical source, retains the raw fenced response, and Gate4
requires both hashes plus explicit raw-protocol edit lineage. Remove the claim that the outer
fence itself must cause a SyntaxError.

**Step 2: Run focused and broad local tests**

Run in WSL:

```bash
.venv/bin/python -m pytest tests/test_capsule_scripts.py -q
.venv/bin/python -m pytest \
  tests/test_capsule_config.py \
  tests/test_capsule_evaluator.py \
  tests/test_capsule_server_factory.py \
  tests/test_capsule_trainer.py \
  tests/test_program_source.py \
  tests/test_capsule_scripts.py -q
```

Expected: zero failures.

**Step 3: Commit the implementation**

```bash
git add scripts/capsule_rl/common.py tests/test_capsule_scripts.py docs/capsule_rl.md
git commit -m "Fix Gate4 source normalization audit"
```

### Task 4: Verify remotely and resume training

**Files:**
- Sync the three implementation files to `/root/autodl-tmp/cap-x`

**Step 1: Run the focused remote tests**

Run the same focused Gate4 tests with the remote `.venv` and require zero failures.

**Step 2: Commit the exact remote snapshot**

Commit only the synchronized files so formal runtime provenance sees a clean checkout and a new
full Git SHA.

**Step 3: Generate a new immutable 20-seed preparation directory**

Use seeds 5--24 and the existing cube-lift source task. Do not overwrite the failed attempt.

**Step 4: Rerun canonical Gate1--7**

Use the same owned-service workflow, final BF16 FSDP/vLLM 0.45 profile selected by the OOM ladder,
external non-streaming/non-thinking `qwen3.7-plus`, and command-based monitoring.

**Step 5: Materialize the Gate7-bound bundle and run formal training**

Require exactly twenty completed groups, preserve all retry/discard audit, verify one optimizer
step per scheduled seed, and verify a valid final LoRA checkpoint before reporting the result.
