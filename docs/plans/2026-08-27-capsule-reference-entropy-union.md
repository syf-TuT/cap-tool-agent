# Capsule Reference Entropy Union Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent multi-step LoRA training from failing when adapter-enabled and adapter-disabled log-probability calls return different diagnostic entropy tensors.

**Architecture:** Keep VeRL's strict `DataProto.union` behavior and normalize worker outputs at the Capsule trainer boundary. Drop the optional diagnostic `entropys` field from both old-policy and adapter-disabled reference outputs before merging, while preserving the required probability tensors and metadata lifecycle.

**Tech Stack:** Python 3.12, PyTorch, VeRL v0.6.1 `DataProto`, pytest, remote A800 Capsule workflow.

---

### Task 1: Add a strict-union regression test

**Files:**
- Modify: `tests/test_capsule_trainer.py`

**Step 1: Write the failing test**

Add a DataProto-like fixture whose `union` method rejects unequal duplicate tensor keys,
then make the LoRA actor return different `entropys` values for enabled and disabled
adapter calls:

```python
class _StrictDataProtoLike(_DataProtoLike):
    def union(self, other):
        source = getattr(other, "batch", other)
        for key, value in source.items():
            if key in self.batch:
                assert self.batch[key].equal(value), (
                    f"{key} in tensor_dict1 and tensor_dict2 are not the same object"
                )
            else:
                self.batch[key] = value
        return self


def test_lora_reference_drops_entropy_diagnostics_before_strict_union() -> None:
    # Actor returns unequal entropys for LoRA-enabled and adapter-disabled calls.
    # run_step must retain only old_log_probs/ref_log_prob in the merged batch.
```

Assert that the updated batch contains the two required probability fields, omits
`entropys`, and clears the adapter-disable metadata.

**Step 2: Run the focused test to verify RED**

Run on the Linux runtime copy:

```bash
.venv/bin/python -m pytest -q \
  tests/test_capsule_trainer.py::test_lora_reference_drops_entropy_diagnostics_before_strict_union
```

Expected: FAIL at strict union with `entropys in tensor_dict1 and tensor_dict2 are not the same object`.

### Task 2: Normalize diagnostic worker fields

**Files:**
- Modify: `capx/rl/capsule/trainer.py:847-889`
- Test: `tests/test_capsule_trainer.py`

**Step 1: Implement the minimal fix**

Before merging the first actor output, delete optional `entropys`. In
`_compute_actor_base_reference_log_prob`, delete optional `entropys` before renaming
`old_log_probs` to `ref_log_prob`:

```python
tensors = _tensor_batch(output)
if "entropys" in tensors:
    del tensors["entropys"]
```

Do not relax `_merge_batch`, rename diagnostic fields, or change loss behavior.

**Step 2: Run the focused test to verify GREEN**

Run the same focused pytest command.

Expected: PASS.

**Step 3: Run the complete trainer and Capsule suites**

```bash
.venv/bin/python -m pytest -q tests/test_capsule_trainer.py
.venv/bin/python -m pytest -q tests/test_capsule_*.py
```

Expected: all tests pass with only the two known Pyroki deprecation warnings.

**Step 4: Commit the regression fix**

```bash
git add capx/rl/capsule/trainer.py tests/test_capsule_trainer.py
git commit -m "Fix Capsule reference entropy merge"
```

### Task 3: Re-establish canonical runtime provenance

**Files:**
- Create remotely: new ignored prepare, Gate, bundle, log, and output artifacts

**Step 1: Preserve failed-run evidence and reclaim the old Gate checkpoint**

Archive the five successful group JSON files and `main_ppo_train.log`. Preserve the
validated LoRA adapter and Gate JSONs, verify their hashes, then remove only the exact old
full FSDP Gate checkpoint required to restore the launcher's 80 GiB free-space floor.

**Step 2: Synchronize the fixed commit to the remote branch**

Apply the exact local commit to `/root/autodl-tmp/cap-x`, run the focused and full Capsule
test suites there, and confirm the remote worktree is clean.

**Step 3: Run a new 20-seed canonical Gate 1--7 workflow**

Reuse seeds 5--24 and the proven BF16 FSDP/vLLM 0.45 final profile. Require non-streaming,
thinking-disabled `qwen3.7-plus`, a nonzero Gate 6 gradient, adapter reload success, and a
new Gate 7 audit bound to the fixed Git SHA.

**Step 4: Materialize and validate a new immutable training bundle**

Run `materialize_resolved_dataset` in validation and write modes, then run:

```bash
.venv/bin/python -m capx.rl.capsule.main_ppo \
  --config <new-seed-resolved-config> \
  --validate-only
```

Expected: 20 records, compatible pinned VeRL SHA, clean project, runtime not started.

### Task 4: Restart and monitor the 20-seed training

**Files:**
- Create remotely: new `main_ppo_train.log`, group artifacts, discard audit, and final checkpoint

**Step 1: Start owned Program identity and Pyroki services**

Use PID files and readiness probes for ports 8101 and 8116. Keep credentials in environment
variables only.

**Step 2: Start formal `main_ppo` in the existing SSH foreground session**

Keep `CAPX_FORCE_STREAMING_CHAT_COMPLETIONS=0` and tee output to the new bundle log.

**Step 3: Monitor with commands every 30 seconds**

Use a foreground shell loop that reports successful group-file count, process liveness,
GPU memory/utilization, log growth, and disk space. Do not create an automation.

**Step 4: Verify completion**

Require process exit code zero, 20 scheduled groups accounted for, optimizer-step delta equal
to successful actor updates, a valid final checkpoint and manifest, no owned processes, idle
GPU, clean Git state, and no credential material in tracked files or run artifacts.
