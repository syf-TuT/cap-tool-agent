# Effect-Bounded Forward-Only Recovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an `auto_forward` Capsule runtime mode that executes effect-bounded units without per-unit LLM calls and only invokes the LLM for forward-only recovery.

**Architecture:** Keep the existing stepwise LLM runtime as a compatibility path. Add a deterministic execution driver that walks grouped source units in order, records a no-replay side-effect ledger, and builds compact recovery prompts only when execution fails or a guard blocks unsafe replay. Defer task-local predicate evaluators; use exceptions, invalid actions, task completion, and existing coarse guards only.

**Tech Stack:** Python 3.12-compatible runtime code, dataclasses, pytest, existing `capx.runtime_control` and `capx.envs.trial` modules.

---

### Task 1: Add Mode Selection Tests

**Files:**
- Modify: `tests/test_runtime_control_trial_loop.py`
- Inspect: `capx/envs/trial.py`

**Step 1: Write failing tests**

Add tests that configure `capsule_control_mode`:

```python
def test_capsule_auto_forward_runs_groups_without_capsule_action_llm(tmp_path):
    config = {
        "use_runtime_control": True,
        "capsule_control_mode": "auto_forward",
        "max_capsule_steps": 8,
    }
    # Use a script or monkeypatched model that only supplies initial code.
    # Assert the generated side-effect groups execute in order.
    # Assert no "capsule_action" LLM stage is recorded for normal execution.


def test_capsule_llm_step_mode_keeps_existing_action_loop(tmp_path):
    config = {
        "use_runtime_control": True,
        "capsule_control_mode": "llm_step",
        "scripted_actions": [{"action": "run_group", "args": {"group_id": "group_1"}}],
    }
    # Assert the existing scripted action path still runs.
```

**Step 2: Run tests to verify failure**

Run in WSL per repository guidance:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_trial_loop.py -q'
```

Expected: new tests fail because `capsule_control_mode` is not implemented.

**Step 3: Implement mode dispatch**

In `capx/envs/trial.py`, split the current runtime-control loop into:

- `_run_capsule_llm_step_loop(...)` for current behavior.
- `_run_capsule_auto_forward_loop(...)` for the new deterministic normal path.

Route by:

```python
mode = str(config.get("capsule_control_mode", "llm_step"))
if mode == "auto_forward":
    return _run_capsule_auto_forward_loop(...)
if mode == "llm_step":
    return _run_capsule_llm_step_loop(...)
raise ValueError(f"Unsupported capsule_control_mode: {mode}")
```

**Step 4: Run tests**

Run the same pytest command. Expected: mode dispatch tests pass.

**Step 5: Commit**

```bash
git add capx/envs/trial.py tests/test_runtime_control_trial_loop.py
git commit -m "Add Capsule control mode dispatch"
```

### Task 2: Implement Auto-Forward Execution

**Files:**
- Modify: `capx/envs/trial.py`
- Modify: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write failing tests**

Add coverage for ordered group execution:

```python
def test_capsule_auto_forward_executes_effect_bounded_groups_in_source_order(tmp_path):
    # Generate source with two groups.
    # Assert group_1 executes before group_2.
    # Assert capsule_step_metrics rows are written for both groups.
```

Add coverage for no replay:

```python
def test_capsule_auto_forward_never_replays_successful_side_effect_group(tmp_path):
    # Make group_1 succeed and mutate robot state.
    # Force a later recovery attempt to target group_1 again.
    # Assert the no-rollback guard rejects it.
```

**Step 2: Run tests to verify failure**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_trial_loop.py -q'
```

Expected: auto-forward execution is incomplete.

**Step 3: Implement minimal auto-forward loop**

In `_run_capsule_auto_forward_loop(...)`:

- Build `regions`, `groups`, `region_by_id`, and `group_by_id` once from the generated source.
- Iterate `groups` in source order.
- For each group, snapshot state before execution.
- Run `_no_rollback_guard_event(...)`.
- If no guard event exists, execute `RuntimeAction("run_group", {"group_id": group.group_id})`.
- Update `executed_side_effect_groups` and `executed_side_effect_regions` on successful side-effect execution.
- Build feedback and step metrics with the existing helpers.
- Stop when task completion or reward `>= 1.0` is observed.

**Step 4: Run tests**

Expected: ordered execution and no-replay tests pass.

**Step 5: Commit**

```bash
git add capx/envs/trial.py tests/test_runtime_control_trial_loop.py
git commit -m "Run Capsule units with auto-forward execution"
```

### Task 3: Add Recovery-On-Failure

**Files:**
- Modify: `capx/envs/trial.py`
- Modify: `capx/runtime_control/prompts.py`
- Modify: `tests/test_runtime_control_trial_loop.py`
- Modify: `tests/test_runtime_control_prompts.py`

**Step 1: Write failing tests**

Add tests for one recovery call after a failed group:

```python
def test_capsule_auto_forward_calls_llm_only_after_group_failure(tmp_path):
    # group_1 succeeds, group_2 raises.
    # Assert no capsule_action LLM call before the failure.
    # Assert exactly one recovery prompt is built after the failure.
```

Add prompt tests:

```python
def test_recovery_prompt_is_local_and_bounded():
    # Build a recovery prompt with many groups/history/trace events.
    # Assert it contains current failed group and recent trace only.
    # Assert it does not include every group or full history.
```

**Step 2: Run tests to verify failure**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_trial_loop.py tests/test_runtime_control_prompts.py -q'
```

Expected: recovery prompt path is missing.

**Step 3: Implement recovery prompt builder**

Add `build_capsule_recovery_prompt(...)` in `capx/runtime_control/prompts.py`.

Inputs:

- `task`
- `failed_unit`
- `history_tail`
- `trace_summary`
- `side_effect_ledger`
- `recovery_observation_functions`

Allowed actions should be limited to inspection, patching, `append_recovery`,
`resume_from_region`, and `finish` as appropriate. Keep the existing fresh-state
rule for `append_recovery`.

**Step 4: Execute recovery action**

In `_run_capsule_auto_forward_loop(...)`, when a group fails or guard blocks:

- Build recovery prompt.
- Call `_query_model` under `llm_call_stage("capsule_recovery")`.
- Parse `RuntimeAction`.
- Run the same guard checks before executing the recovery action.
- Preserve the existing `recovery_side_effect_budget = 1` behavior after
  successful `append_recovery`.
- Re-segment source after successful patch or append.

**Step 5: Run tests**

Expected: failure-triggered recovery tests pass.

**Step 6: Commit**

```bash
git add capx/envs/trial.py capx/runtime_control/prompts.py tests/test_runtime_control_trial_loop.py tests/test_runtime_control_prompts.py
git commit -m "Add forward-only recovery prompts for Capsule auto-forward"
```

### Task 4: Rename User-Facing Group Language

**Files:**
- Modify: `capx/runtime_control/normalizer.py`
- Modify: `capx/runtime_control/schema.py`
- Modify: `capx/runtime_control/prompts.py`
- Modify: `tests/test_runtime_control_normalizer.py`
- Modify: `tests/test_runtime_control_prompts.py`

**Step 1: Write failing tests**

Add assertions that prompt text uses "effect-bounded execution unit" rather than
"semantic group".

**Step 2: Run prompt and normalizer tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_normalizer.py tests/test_runtime_control_prompts.py -q'
```

Expected: tests fail until wording is updated.

**Step 3: Update terminology**

Keep class names stable if a broad API rename would be too risky, but update:

- docstrings
- prompt text
- config descriptions
- metric labels where practical

Do not claim explicit semantic pre/postconditions.

**Step 4: Run tests**

Expected: terminology tests pass.

**Step 5: Commit**

```bash
git add capx/runtime_control/normalizer.py capx/runtime_control/schema.py capx/runtime_control/prompts.py tests/test_runtime_control_normalizer.py tests/test_runtime_control_prompts.py
git commit -m "Use effect-bounded unit terminology for Capsule groups"
```

### Task 5: Bound Trace Summaries and Inspect Trace

**Files:**
- Modify: `capx/runtime_control/trace.py`
- Modify: `capx/envs/trial.py`
- Modify: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write failing tests**

```python
def test_runtime_trace_summary_is_bounded():
    trace = RuntimeTrace()
    for index in range(50):
        trace.log({"name": "goto_pose", "status": "success", "index": index})
    summary = trace.summary(max_events=5)
    assert summary["event_count"] == 50
    assert len(summary["recent_events"]) == 5
    assert "events" not in summary


def test_inspect_trace_supports_last_n():
    action = RuntimeAction("inspect_trace", {"last_n": 3})
    # Assert only 3 recent events are returned.
```

**Step 2: Run tests to verify failure**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_trial_loop.py -q'
```

Expected: trace summary currently returns full `events`.

**Step 3: Implement bounded trace summary**

Change `RuntimeTrace.summary()` to accept optional parameters:

```python
def summary(self, *, max_events: int = 8, failed_only: bool = False) -> dict[str, Any]:
    ...
```

Return:

- `event_count`
- `primitive_call_counts`
- `failed_event_count`
- `recent_events`
- `failed_events`

Update `inspect_trace` to honor `last_n` and `failed_only`.

**Step 4: Run tests**

Expected: trace tests pass.

**Step 5: Commit**

```bash
git add capx/runtime_control/trace.py capx/envs/trial.py tests/test_runtime_control_trial_loop.py
git commit -m "Bound Capsule trace summaries"
```

### Task 6: Standardize Experiment Reporting Schema

**Files:**
- Inspect and modify the active remote result aggregation scripts under `scripts/` or existing run harnesses used for `remote_results`.
- Add tests under `tests/` for pure summary aggregation helpers if they exist or are factored out.

**Step 1: Locate aggregation code**

Run:

```powershell
rg -n "selection_policy|latest run per seed|task_completed_count|llm|attempt_count|retry_count|failure_kind" scripts capx tests
```

**Step 2: Factor a pure reporting helper**

Create or update a helper that computes:

- `first_attempt_success_count`
- `first_attempt_success_rate`
- `success_by_retry_budget`
- `total_attempt_count`
- `llm_logical_call_count`
- `llm_attempt_count`
- `llm_elapsed_seconds`
- `trial_elapsed_seconds`
- `provider_failure_count`
- `algorithm_failure_count`
- `budget_exhausted_count`

**Step 3: Write unit tests**

Use synthetic rows with repeated seeds and mixed provider/algorithm failures.
Assert first-attempt and latest-per-seed numbers differ when retries succeed.

**Step 4: Run tests**

Run focused pytest for the new helper.

**Step 5: Commit**

```bash
git add scripts capx tests
git commit -m "Report first-attempt and retry-aware Capsule metrics"
```

### Task 7: Verification Run

**Files:**
- No code changes expected.

**Step 1: Run focused unit tests**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_trial_loop.py tests/test_runtime_control_prompts.py tests/test_runtime_control_normalizer.py -q'
```

Expected: all pass.

**Step 2: Run a minimal Robosuite smoke test**

After syncing Windows edits into `/home/capx/code/cap-x`, run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; export MUJOCO_GL=egl; uv run --no-sync capx/envs/launch.py --config-path env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml --use-oracle-code True --total-trials 1 --num-workers 1 --record-video False'
```

Expected: no runtime-control regression in the known passing smoke path.

**Step 3: Run one auto-forward Capsule smoke trial**

Use a short seed run with `capsule_control_mode=auto_forward`, low trial count,
and video disabled. Expected: normal successful groups do not create one
`capsule_action` LLM call per group; recovery calls use `capsule_recovery`.

**Step 4: Commit final fixes if needed**

```bash
git add <changed-files>
git commit -m "Verify Capsule auto-forward recovery"
```
