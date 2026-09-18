# Capsule Recovery Transaction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent a new recovery append or unrelated group action until the latest Capsule recovery generation has observed fresh state and completed all of its pending side-effect groups.

**Architecture:** Replace the trace-revision recovery gate with an explicit transaction-completion predicate over the latest `RecoveryGeneration`. Derive pending, runnable, and patchable recovery group IDs from stable lineage, use them to narrow prompt choices, and enforce the same boundaries with runtime guards before execution or source editing.

**Tech Stack:** Python 3.10-3.12, dataclasses, existing Capsule lineage and runtime-control types, pytest, Ruff, WSL2 Ubuntu with `uv --no-sync`.

---

### Task 1: Reproduce the incomplete-recovery append bug

**Files:**
- Modify: `tests/test_runtime_control_trial_loop.py:6739`
- Test: `tests/test_runtime_control_trial_loop.py`

**Step 1: Replace the permissive regression with the required behavior**

Replace `test_physical_trace_after_append_allows_later_append` with a test that
appends a multi-group recovery, executes only its observation group, and then
tries another append:

```python
def test_pending_recovery_groups_block_later_append_after_new_trace(tmp_path):
    first_recovery = 'obs = get_observation()\nmove_to("recover")'
    trial_module._run_capsule_loop(
        FakeRewardDropCapsuleEnv(),
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 3,
            "capsule_max_regions_per_group": 1,
        },
        initial_code="x = 1\n",
        scripted_actions=[
            {"action": "append_recovery", "args": {"source": first_recovery}},
            {"action": "run_group", "args": {"group_id": "group_2"}},
            {
                "action": "append_recovery",
                "args": {
                    "source": 'obs2 = get_observation()\nmove_to("again")'
                },
            },
        ],
    )

    rows = _capsule_step_metrics(
        tmp_path / "capsule_step_metrics_trial_00.jsonl"
    )
    generation = rows[2]["recovery_generations"][0]

    assert rows[2]["event_status"] == "invalid"
    assert rows[2]["edit_rejection_reason"] == "recovery_generation_pending"
    assert generation["observation_satisfied"] is True
    assert generation["authorized_group_keys"] == ["group_key_000003"]
```

**Step 2: Add an unrelated-group execution regression**

Create a test with an unexecuted original group and a pending recovery. Attempt
the original group while the recovery observation group is the only permitted
recovery action:

```python
def test_pending_recovery_blocks_unrelated_runnable_group(tmp_path):
    trial_module._run_capsule_loop(
        FakeRewardDropCapsuleEnv(),
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 2,
            "capsule_max_regions_per_group": 1,
        },
        initial_code='move_to("original")\n',
        scripted_actions=[
            {
                "action": "append_recovery",
                "args": {
                    "source": 'obs = get_observation()\nmove_to("recover")'
                },
            },
            {"action": "run_group", "args": {"group_id": "group_1"}},
        ],
    )

    rows = _capsule_step_metrics(
        tmp_path / "capsule_step_metrics_trial_00.jsonl"
    )
    assert rows[1]["event_status"] == "invalid"
    assert rows[1]["safety_failure"] == "recovery_generation_pending"
    assert rows[1]["event_evidence"]["runnable_recovery_group_ids"] == [
        "group_2"
    ]
```

**Step 3: Sync and run the tests to verify RED**

Run from elevated PowerShell:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'cp /mnt/f/code/cap-x/tests/test_runtime_control_trial_loop.py /home/capx/code/cap-x/tests/test_runtime_control_trial_loop.py; export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_trial_loop.py::test_pending_recovery_groups_block_later_append_after_new_trace tests/test_runtime_control_trial_loop.py::test_pending_recovery_blocks_unrelated_runnable_group -q'
```

Expected: both tests fail because a newer trace reopens append and unrelated
dependency-runnable groups are not guarded.

**Step 4: Commit the failing tests**

```bash
git add tests/test_runtime_control_trial_loop.py
git commit -m "Test pending Capsule recovery transactions"
```

### Task 2: Make recovery completion authoritative

**Files:**
- Modify: `capx/envs/trial.py:1206-1285`
- Modify: `capx/envs/trial.py:2019-2075`
- Modify: `capx/envs/trial.py:2168-2185`
- Modify: `capx/envs/trial.py:3322-3370`
- Modify: `tests/test_runtime_control_trial_loop.py`

**Step 1: Extend recovery action state**

Add pending and patchable IDs and remove `trace_revision` from the state
calculator:

```python
@dataclass(frozen=True)
class _RecoveryActionState:
    append_recovery_available: bool
    append_recovery_block_reason: str | None
    pending_recovery_group_ids: tuple[str, ...]
    runnable_recovery_group_ids: tuple[str, ...]


def _recovery_generation_complete(generation: RecoveryGeneration) -> bool:
    return generation.observation_satisfied and not generation.authorized_group_keys
```

When there is no generation or the latest generation is complete, return an
available state. Otherwise map the latest generation's unexecuted observation
and authorized stable keys back to current IDs. `pending_recovery_group_ids`
must not apply dependency filtering; `runnable_recovery_group_ids` must retain
the current dependency and observation-order filtering.

Use the stable block reason `recovery_generation_pending`.

**Step 2: Add authoritative action guards**

Before `_no_rollback_guard_event`, add a guard that rejects:

```python
if action.action == "run_group" and group_id not in state.runnable_recovery_group_ids:
    # invalid recovery_generation_pending event

if action.action == "patch_group" and group_id not in state.pending_recovery_group_ids:
    # invalid recovery_generation_pending event
```

The guard applies only while `append_recovery_available` is false. Evidence
must include:

```python
{
    "safety_failure": "recovery_generation_pending",
    "edit_rejection_reason": "recovery_generation_pending",
    "pending_recovery_group_ids": list(state.pending_recovery_group_ids),
    "runnable_recovery_group_ids": list(state.runnable_recovery_group_ids),
}
```

Update `_append_recovery_guard_event` to use the same block reason and evidence.

**Step 3: Narrow prompt execution IDs at the loop boundary**

While recovery is pending, pass `runnable_recovery_group_ids` as the prompt's
`runnable_group_ids`. Otherwise pass the ordinary dependency-runnable IDs. Keep
the full dependency state for the runtime dependency guard.

**Step 4: Add completion transition coverage**

Add a test that appends `get_observation` plus two `move_to` groups, executes all
three, then successfully appends a second recovery. Assert that append remains
blocked after the first and second groups and becomes available only after the
last authorized group is consumed.

**Step 5: Sync and run focused tests to verify GREEN**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'cp /mnt/f/code/cap-x/capx/envs/trial.py /home/capx/code/cap-x/capx/envs/trial.py; cp /mnt/f/code/cap-x/tests/test_runtime_control_trial_loop.py /home/capx/code/cap-x/tests/test_runtime_control_trial_loop.py; export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_trial_loop.py -q'
```

Expected: all runtime-control trial-loop tests pass.

**Step 6: Commit**

```bash
git add capx/envs/trial.py tests/test_runtime_control_trial_loop.py
git commit -m "Enforce Capsule recovery transactions"
```

### Task 3: Align prompt guidance with the transaction guard

**Files:**
- Modify: `capx/runtime_control/prompts.py:90-180`
- Modify: `capx/runtime_control/prompts.py:276-313`
- Modify: `capx/runtime_control/prompts.py:1197-1215`
- Modify: `tests/test_runtime_control_prompts.py:420-485`

**Step 1: Write prompt RED tests**

Extend the blocked-recovery prompt test with pending IDs and assert:

```python
prompt = build_capsule_prompt(
    task="recover",
    regions=regions,
    groups=groups,
    history=[],
    trace_summary={},
    append_recovery_available=False,
    append_recovery_block_reason="recovery_generation_pending",
    pending_recovery_group_ids=["group_3", "group_4"],
    runnable_recovery_group_ids=["group_3"],
)
text = prompt[1]["content"][0]["text"]
allowed = next(line for line in text.splitlines() if line.startswith("Allowed actions:"))

assert "append_recovery" not in allowed
assert '"pending_recovery_group_ids": [\n    "group_3",' in text
assert '"runnable_group_ids": [\n    "group_3"\n  ]' in text
assert "finish the existing recovery transaction" in text
```

Add a second assertion proving the normal append recommendation returns when
`append_recovery_available=True`.

**Step 2: Run prompt tests to verify RED**

Sync `tests/test_runtime_control_prompts.py` and run its new exact node IDs in
WSL. Expected: FAIL because pending recovery group IDs are not prompt inputs and
the guidance still describes fresh trace as the unlock condition.

**Step 3: Implement minimal prompt changes**

- Add `pending_recovery_group_ids` to `build_capsule_prompt` and normalized
  recovery availability.
- Replace fresh-trace wording with transaction-completion wording.
- Explain that `run_group` must use runnable recovery IDs and `patch_group`
  must use pending recovery IDs.
- Continue omitting `append_recovery` from allowed actions while blocked.
- Preserve the normal append guidance only after completion.

**Step 4: Run prompt and loop suites**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'cp /mnt/f/code/cap-x/capx/runtime_control/prompts.py /home/capx/code/cap-x/capx/runtime_control/prompts.py; cp /mnt/f/code/cap-x/tests/test_runtime_control_prompts.py /home/capx/code/cap-x/tests/test_runtime_control_prompts.py; export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_prompts.py tests/test_runtime_control_trial_loop.py -q'
```

Expected: both suites pass.

**Step 5: Commit**

```bash
git add capx/runtime_control/prompts.py tests/test_runtime_control_prompts.py
git commit -m "Guide Capsule recovery transactions"
```

### Task 4: Simplify and verify the complete fix

**Files:**
- Verify: `capx/envs/trial.py`
- Verify: `capx/runtime_control/prompts.py`
- Verify: `tests/test_runtime_control_trial_loop.py`
- Verify: `tests/test_runtime_control_prompts.py`

**Step 1: Review only the modified sections for simplification**

Remove duplicate set-to-ID mapping or repeated evidence construction if a
small helper improves clarity. Do not change transaction behavior or unrelated
runtime code.

**Step 2: Sync all modified runtime files and tests to WSL**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'cp /mnt/f/code/cap-x/capx/envs/trial.py /home/capx/code/cap-x/capx/envs/trial.py; cp /mnt/f/code/cap-x/capx/runtime_control/prompts.py /home/capx/code/cap-x/capx/runtime_control/prompts.py; cp /mnt/f/code/cap-x/tests/test_runtime_control_trial_loop.py /home/capx/code/cap-x/tests/test_runtime_control_trial_loop.py; cp /mnt/f/code/cap-x/tests/test_runtime_control_prompts.py /home/capx/code/cap-x/tests/test_runtime_control_prompts.py; export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync ruff check capx/envs/trial.py capx/runtime_control/prompts.py tests/test_runtime_control_trial_loop.py tests/test_runtime_control_prompts.py; uv run --no-sync pytest tests/test_runtime_control_prompts.py tests/test_runtime_control_trial_loop.py tests/test_runtime_control_lineage.py tests/test_runtime_control_feedback.py -q'
```

Expected: Ruff reports `All checks passed!` and all focused recovery-control
tests pass.

**Step 3: Verify diff scope**

```powershell
git diff --check
git status --short --branch
git diff HEAD~3 -- capx/envs/trial.py capx/runtime_control/prompts.py tests/test_runtime_control_trial_loop.py tests/test_runtime_control_prompts.py docs/plans
```

Expected: no whitespace errors and no unrelated source, configuration, or
experiment-result changes. Existing untracked model-cache files remain
untouched.

**Step 4: Commit any final test-only cleanup if needed**

```bash
git add <only files changed by the cleanup>
git commit -m "Test Capsule recovery transactions"
```
