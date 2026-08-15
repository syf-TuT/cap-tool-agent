# Capsule Runtime Guardrails Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent trials from starting with unavailable robot services, block out-of-order semantic groups and repeated recovery appends, and classify/retry transient service failures without prompting source repair.

**Architecture:** Add a small shared infrastructure-error module used by launch, trial execution, and result persistence. Compute group and recovery availability from existing source-analysis, executor, lineage, and trace state, pass the bounded state into the action prompt, and enforce the same state with runtime guards. Keep retries inside one `run_group` decision and abort when a robot side effect may have been attempted.

**Tech Stack:** Python 3.10-3.12, dataclasses, sockets/URL parsing, existing Capsule runtime-control types, pytest, Ruff, WSL2 Ubuntu with `uv --no-sync`.

---

### Task 1: Add typed infrastructure failures and a terminal run outcome

**Files:**
- Create: `capx/envs/infrastructure.py`
- Modify: `capx/envs/trial_results.py:21-33`
- Modify: `capx/envs/runner.py:22-48,285-385`
- Test: `tests/test_runner_resilience.py`
- Test: `tests/test_trial_results.py`

**Step 1: Write the failing result-schema test**

Add a test proving the schema accepts the new terminal outcome:

```python
def test_writer_accepts_infrastructure_failed_outcome(tmp_path):
    writer = TrialResultWriter(tmp_path)
    path = writer.start(trial=21, started_at=STARTED_AT)
    writer.finalize(
        _finished_result(
            run_outcome=RunOutcome.INFRASTRUCTURE_FAILED,
            failure_kind="service_http_5xx",
        )
    )

    assert _load(path)["run_outcome"] == "infrastructure_failed"
```

**Step 2: Sync the test and run it to verify it fails**

Run from elevated PowerShell:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'cp /mnt/f/code/cap-x/tests/test_trial_results.py /home/capx/code/cap-x/tests/test_trial_results.py; export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_trial_results.py::test_writer_accepts_infrastructure_failed_outcome -q'
```

Expected: FAIL because `RunOutcome.INFRASTRUCTURE_FAILED` does not exist.

**Step 3: Define the shared exception types and outcome**

Create `capx/envs/infrastructure.py` with typed failures that preserve a stable
kind and safe metadata:

```python
from __future__ import annotations

from typing import Any


class InfrastructureFailure(RuntimeError):
    def __init__(
        self,
        kind: str,
        message: str,
        *,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        self.kind = kind
        self.message = message
        self.evidence = evidence or {}
        super().__init__(message)


class ServiceReadinessError(InfrastructureFailure):
    pass
```

Add `INFRASTRUCTURE_FAILED = "infrastructure_failed"` to `RunOutcome`.

In `_run_single_trial_with_timeout`, catch `InfrastructureFailure` before the
generic `BaseException` branch and finalize with:

```python
outcome=RunOutcome.INFRASTRUCTURE_FAILED
failure_kind=exc.kind
failure_message=exc.message
```

**Step 4: Add and run the runner-classification test**

```python
def test_typed_infrastructure_failure_is_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner,
        "_run_single_trial",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            InfrastructureFailure("service_timeout", "SAM3 timed out")
        ),
    )

    summary = runner._run_trial_with_retries(
        object(), 1, _args(), _config(tmp_path), None
    )

    result = json.loads((tmp_path / "trial_1_result.json").read_text())
    assert summary.run_outcome == "infrastructure_failed"
    assert summary.failure_kind == "service_timeout"
    assert result["run_outcome"] == "infrastructure_failed"
```

Sync the modified files and run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'cp /mnt/f/code/cap-x/capx/envs/infrastructure.py /home/capx/code/cap-x/capx/envs/infrastructure.py; cp /mnt/f/code/cap-x/capx/envs/trial_results.py /home/capx/code/cap-x/capx/envs/trial_results.py; cp /mnt/f/code/cap-x/capx/envs/runner.py /home/capx/code/cap-x/capx/envs/runner.py; cp /mnt/f/code/cap-x/tests/test_trial_results.py /home/capx/code/cap-x/tests/test_trial_results.py; cp /mnt/f/code/cap-x/tests/test_runner_resilience.py /home/capx/code/cap-x/tests/test_runner_resilience.py; export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_trial_results.py::test_writer_accepts_infrastructure_failed_outcome tests/test_runner_resilience.py::test_typed_infrastructure_failure_is_persisted -q'
```

Expected: 2 passed.

**Step 5: Commit**

```bash
git add capx/envs/infrastructure.py capx/envs/trial_results.py capx/envs/runner.py tests/test_trial_results.py tests/test_runner_resilience.py
git commit -m "Classify infrastructure trial failures"
```

### Task 2: Make required-service startup a fail-fast preflight

**Files:**
- Modify: `capx/envs/infrastructure.py`
- Modify: `capx/envs/runner.py:70-123`
- Modify: `capx/envs/launch.py:203-218`
- Test: `tests/test_runner_resilience.py`

**Step 1: Write failing endpoint-discovery tests**

Add tests for configured servers and conditional Molmo discovery:

```python
def test_required_service_endpoints_include_libero_molmo():
    endpoints = runner._required_service_endpoints(
        [{"host": "127.0.0.1", "port": 8114}],
        {
            "cfg": {
                "apis": ["FrankaLiberoApi"],
                "molmo_base_url": "http://127.0.0.1:8122/v1",
            }
        },
    )

    assert [(item.host, item.port) for item in endpoints] == [
        ("127.0.0.1", 8114),
        ("127.0.0.1", 8122),
    ]


def test_required_service_endpoints_ignore_molmo_for_other_apis():
    endpoints = runner._required_service_endpoints(
        [],
        {"cfg": {"apis": ["OtherApi"], "molmo_base_url": "http://host:8122/v1"}},
    )

    assert endpoints == []
```

**Step 2: Write failing timeout-cleanup test**

Use fake processes and a fake readiness probe:

```python
def test_start_api_servers_stops_started_processes_on_readiness_timeout(monkeypatch):
    proc = SimpleNamespace(terminate=Mock(), join=Mock())
    monkeypatch.setattr(runner, "run_server_proc", lambda config: proc)
    monkeypatch.setattr(runner, "_service_endpoint_ready", lambda endpoint: False)
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)

    with pytest.raises(ServiceReadinessError, match="not ready"):
        runner._start_api_servers(
            [{"host": "127.0.0.1", "port": 8114}],
            required_endpoints=[ServiceEndpoint("sam3", "127.0.0.1", 8114)],
            wait_timeout=0,
        )

    proc.terminate.assert_called_once_with()
    proc.join.assert_called_once_with(timeout=5.0)
```

**Step 3: Run the tests to verify they fail**

Sync `tests/test_runner_resilience.py`, then run its three new service tests in
WSL with `uv run --no-sync pytest ... -q`.

Expected: FAIL because endpoint discovery, `ServiceEndpoint`, and fail-fast
readiness do not exist.

**Step 4: Implement endpoint discovery and common-deadline readiness**

In `capx/envs/infrastructure.py`, add:

```python
@dataclass(frozen=True)
class ServiceEndpoint:
    name: str
    host: str
    port: int
```

In `runner.py`:

- Parse `env_factory["cfg"]["molmo_base_url"]` with `urllib.parse.urlparse`
  only when `FrankaLiberoApi` is present.
- Deduplicate endpoints by `(host, port)` while retaining a useful name.
- Probe all endpoints against one `time.monotonic() + wait_timeout` deadline.
- Add already-running endpoints to the required set instead of skipping their
  readiness verification.
- Wrap process startup and readiness in `try/except`; call
  `_stop_api_servers(procs)` before re-raising.
- Raise `ServiceReadinessError("service_not_ready", ...)` with the unresolved
  endpoint list rather than printing a warning.

Update `launch.main`:

```python
required_endpoints = _required_service_endpoints(api_servers, env_factory)
server_procs = _start_api_servers(
    api_servers,
    required_endpoints=required_endpoints,
)
```

**Step 5: Run the focused preflight tests**

Sync `infrastructure.py`, `runner.py`, `launch.py`, and the test file. Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runner_resilience.py -q'
```

Expected: all runner resilience tests pass.

**Step 6: Commit**

```bash
git add capx/envs/infrastructure.py capx/envs/runner.py capx/envs/launch.py tests/test_runner_resilience.py
git commit -m "Require robot services before trials"
```

### Task 3: Gate semantic groups on runtime dependencies

**Files:**
- Modify: `capx/envs/trial.py:1934-2125,2745-2785,3200-3230`
- Modify: `capx/runtime_control/prompts.py:96-260,985-1022,1084-1134`
- Test: `tests/test_runtime_control_prompts.py`
- Test: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write the failing runtime guard test**

Use source whose second group consumes a value from the first group and script
the actions out of order:

```python
def test_run_group_blocks_missing_source_dependencies(tmp_path):
    trial_module._run_capsule_loop(
        FakeCapsuleEnv(),
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 2,
            "capsule_max_regions_per_group": 1,
        },
        initial_code='target = get_pose("basket")\nmove_to(target)\n',
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_2"}},
            {"action": "finish", "args": {}},
        ],
    )

    rows = _capsule_step_metrics(tmp_path / "capsule_step_metrics_trial_00.jsonl")
    assert rows[0]["event_status"] == "invalid"
    assert rows[0]["safety_failure"] == "missing_group_dependencies"
    assert rows[0]["event_evidence"]["missing_dependencies"] == ["target"]
```

**Step 2: Write the failing prompt-state test**

Call `build_capsule_prompt` with explicit runtime availability:

```python
prompt = build_capsule_prompt(
    task="move",
    regions=regions,
    groups=groups,
    history=[],
    trace_summary={},
    runnable_group_ids=["group_1"],
    blocked_group_dependencies={"group_2": ["target"]},
)
text = prompt[1]["content"][0]["text"]
assert '"runnable_group_ids": ["group_1"]' in text
assert '"group_2": ["target"]' in text
assert '"group_id": "group_1"' in text
```

**Step 3: Run both tests to verify they fail**

Sync both test files and run their exact node IDs in WSL.

Expected: FAIL because the prompt parameters and dependency guard are absent.

**Step 4: Implement one shared dependency-state calculation**

In `trial.py`, add a frozen state object and calculator:

```python
@dataclass(frozen=True)
class _GroupDependencyState:
    runnable_group_ids: tuple[str, ...]
    missing_by_group_id: dict[str, tuple[str, ...]]


def _group_dependency_state(
    groups: list[CodeRegionGroup],
    runtime_globals: Mapping[str, Any],
) -> _GroupDependencyState:
    source_defined_names = {
        name for group in groups for name in group.defined_names
    }
    missing_by_group_id = {
        group.group_id: tuple(
            name
            for name in group.used_names
            if name in source_defined_names and name not in runtime_globals
        )
        for group in groups
    }
    return _GroupDependencyState(
        runnable_group_ids=tuple(
            group.group_id
            for group in groups
            if not missing_by_group_id[group.group_id]
        ),
        missing_by_group_id={
            group_id: missing
            for group_id, missing in missing_by_group_id.items()
            if missing
        },
    )
```

Compute the state immediately before each action prompt. Add
`_group_dependency_guard_event(action, state)` before `_execute_runtime_action`.
The invalid event must use:

```python
evidence={
    "safety_failure": "missing_group_dependencies",
    "missing_dependencies": list(missing),
    "runnable_group_ids": list(state.runnable_group_ids),
}
```

**Step 5: Add bounded dependency state to the prompt**

Extend `build_capsule_prompt` and `_build_capsule_prompt_text` with
`runnable_group_ids` and `blocked_group_dependencies`. Include a compact JSON
runtime-availability section, annotate blocked group units with
`execution_state="blocked_missing_dependencies"`, `run_allowed=False`, and
bounded `missing_dependencies`, and choose the first runnable group for the
`run_group` example.

Update compact-unit bounding so `missing_dependencies` cannot exceed the
existing list limits.

**Step 6: Run focused prompt and loop tests**

Sync the four modified files and run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_prompts.py tests/test_runtime_control_trial_loop.py -q'
```

Expected: all prompt and trial-loop tests pass.

**Step 7: Commit**

```bash
git add capx/envs/trial.py capx/runtime_control/prompts.py tests/test_runtime_control_prompts.py tests/test_runtime_control_trial_loop.py
git commit -m "Gate Capsule groups on dependencies"
```

### Task 4: Disable repeated append and expose runnable recovery groups

**Files:**
- Modify: `capx/envs/trial.py:1230-1267,1934-2146,3232-3374`
- Modify: `capx/runtime_control/prompts.py:96-260`
- Test: `tests/test_runtime_control_prompts.py`
- Test: `tests/test_runtime_control_trial_loop.py:6450-6525`

**Step 1: Write the failing prompt transition test**

Capture the second LLM action prompt after a scripted append or call the prompt
builder directly with the new state:

```python
prompt = build_capsule_prompt(
    task="recover",
    regions=regions,
    groups=groups,
    history=[],
    trace_summary={},
    append_recovery_available=False,
    append_recovery_block_reason="no_new_physical_state_since_last_append",
    runnable_recovery_group_ids=["group_3"],
)
text = prompt[1]["content"][0]["text"]
allowed = next(line for line in text.splitlines() if line.startswith("Allowed actions:"))
assert "append_recovery" not in allowed
assert '"runnable_recovery_group_ids": ["group_3"]' in text
assert "no_new_physical_state_since_last_append" in text
```

**Step 2: Write the failing early-guard test**

Extend the existing no-new-physical-state test to assert the second append is
rejected before `_prepare_capsule_source_edit` is invoked, while retaining:

```python
assert rows[2]["edit_rejection_reason"] == (
    "no_new_physical_state_since_last_append"
)
assert rows[2]["source_edit_committed"] is False
```

**Step 3: Run the two tests to verify they fail**

Sync both tests and run their exact node IDs in WSL.

Expected: FAIL because append availability and recovery group IDs are not prompt
inputs and the only current check lives inside source-edit preparation.

**Step 4: Compute recovery availability from authoritative state**

In `trial.py`, add a helper that:

- Returns append available when there is no recovery generation or
  `executor.trace.mark()` is newer than the latest `append_trace_revision`.
- Maps the latest generation's `authorized_group_keys` through
  `lineage.group_key_by_id`.
- Intersects those IDs with dependency-runnable group IDs.
- Returns the stable reason `no_new_physical_state_since_last_append` when
  append is unavailable.

Pass this state to every LLM action prompt.

Add `_append_recovery_guard_event` before `_execute_runtime_action`. Its invalid
event must include `edit_rejection_reason`, `source_edit_committed=False`, and
the runnable recovery group IDs so scripted actions receive the same hard
boundary.

Keep `_prepare_capsule_source_edit`'s existing check as defense in depth.

**Step 5: Update prompt action availability**

Only append `append_recovery` to `allowed_actions` when:

```python
recovery_functions and not repair_pending and append_recovery_available
```

When blocked, replace the normal append guidance and example with a concise
instruction to run or patch one of `runnable_recovery_group_ids`, then exercise
it to create fresh physical trace evidence.

**Step 6: Run all append and prompt tests**

Run in WSL:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_prompts.py tests/test_runtime_control_trial_loop.py -q'
```

Expected: all tests pass, including the pre-existing append-generation and
physical-trace tests.

**Step 7: Commit**

```bash
git add capx/envs/trial.py capx/runtime_control/prompts.py tests/test_runtime_control_prompts.py tests/test_runtime_control_trial_loop.py
git commit -m "Guide Capsule recovery execution"
```

### Task 5: Retry only safe runtime infrastructure failures

**Files:**
- Modify: `capx/envs/infrastructure.py`
- Modify: `capx/envs/trial.py:2090-2145,2745-2785`
- Test: `tests/test_runtime_control_trial_loop.py`
- Test: `tests/test_runner_resilience.py`

**Step 1: Write classifier tests**

Add parameterized tests for trace messages containing connection refusal,
connect/read timeout, and HTTP 500/502/503/504, plus negative cases for
`NameError` and ordinary `RuntimeError`.

The classifier API should be:

```python
failure = classify_runtime_infrastructure_failure(event)
assert failure is not None
assert failure.kind == "service_connection_refused"
```

**Step 2: Write the safe-retry trial-loop test**

Monkeypatch `_execute_runtime_action` to return two service failures with no
side-effect trace and then success. Script one `run_group` and `finish`.

Assert:

```python
assert execute_calls == 3
assert rows[0]["event_status"] == "success"
assert trial_metrics["logical_decision_count"] == 2
assert trial_metrics["llm_decision_count"] == 0
```

**Step 3: Write the unsafe-retry test**

Return an infrastructure failure whose trace contains a robot side-effect API:

```python
RuntimeEvent(
    action="run_group",
    status="failed",
    evidence={
        "exception_type": "RuntimeError",
        "trace_events": [
            {
                "name": "move_to_pose",
                "status": "failed",
                "message": "HTTP 503",
            }
        ],
    },
)
```

Assert only one execution attempt occurs and the returned summary/result has
`run_outcome == "infrastructure_failed"`.

**Step 4: Run the new tests to verify they fail**

Sync the test files and run the exact node IDs in WSL.

Expected: FAIL because the classifier and retry wrapper do not exist.

**Step 5: Implement conservative classification**

In `infrastructure.py`, inspect the event message, stderr, exception type, and
trace-event messages. Recognize only:

- connection refused / `ConnectionError`
- connect or read timeout / `Timeout`
- HTTP status 500, 502, 503, or 504

Return an `InfrastructureFailure` with a stable kind. Do not classify arbitrary
`RuntimeError`, parse errors, `NameError`, or API validation failures.

**Step 6: Implement same-decision safe retries**

Add `_execute_runtime_action_with_infrastructure_retries` in `trial.py`:

```python
for attempt in range(1, 4):
    event = _execute_runtime_action(...)
    failure = classify_runtime_infrastructure_failure(event)
    if failure is None:
        return event
    if _event_has_side_effect_trace(event, side_effect_calls):
        raise failure
    if attempt == 3:
        raise failure
    time.sleep(backoff_seconds)
```

Use bounded backoff values `0.5` and `1.0` seconds. Replace only the
`run_group` execution path; non-execution actions continue through the existing
single-call function. Preserve trace evidence across attempts in the final
exception evidence for diagnostics.

Because the typed exception escapes before history construction, it cannot be
turned into source-repair feedback or consume another Capsule decision.

**Step 7: Run the focused regression suites**

Sync all changed files and run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runner_resilience.py tests/test_trial_results.py tests/test_runtime_control_prompts.py tests/test_runtime_control_trial_loop.py -q'
```

Expected: all tests pass.

**Step 8: Commit**

```bash
git add capx/envs/infrastructure.py capx/envs/trial.py tests/test_runtime_control_trial_loop.py tests/test_runner_resilience.py
git commit -m "Retry transient Capsule service failures"
```

### Task 6: Verify scope, formatting, and the LIBERO configuration

**Files:**
- Verify: `env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml`
- Verify: all files modified in Tasks 1-5

**Step 1: Confirm the three visual switches remain false**

Run:

```powershell
rg -n "^(use_visual_feedback|use_wrist_camera|capsule_action_visual_feedback):" env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml
```

Expected: every present visual switch is `false`; no YAML edit is required.

**Step 2: Run Ruff in WSL**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync ruff check capx/envs/infrastructure.py capx/envs/runner.py capx/envs/launch.py capx/envs/trial.py capx/envs/trial_results.py capx/runtime_control/prompts.py tests/test_runner_resilience.py tests/test_trial_results.py tests/test_runtime_control_prompts.py tests/test_runtime_control_trial_loop.py'
```

Expected: `All checks passed!`

**Step 3: Run the complete focused regression set in WSL**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runner_resilience.py tests/test_trial_results.py tests/test_runtime_control_prompts.py tests/test_runtime_control_trial_loop.py tests/test_libero_molmo_config.py -q'
```

Expected: all tests pass.

**Step 4: Inspect the final diff and tracked status**

Run:

```powershell
git diff --check
git status --short --branch
git diff HEAD~4 -- capx tests env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml docs/plans
```

Expected: no whitespace errors, no YAML changes, and only the four approved
runtime concerns plus their tests and plan documents are present. Existing
untracked Molmo cache files remain untouched.

**Step 5: Commit any final test-only cleanup if needed**

```bash
git add <only files changed by the cleanup>
git commit -m "Test Capsule runtime guardrails"
```
