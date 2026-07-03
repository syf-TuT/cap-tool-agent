# Capsule Trace Feedback Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Connect capsule-mode API tracing, source-bound feedback, and scripted smoke artifacts without changing the robot primitive interface seen by generated CaP code.

**Architecture:** Extend `RuntimeTrace` with trace windows, bind traced wrappers through `CodeExecutionEnvBase._build_capsule_globals(trace=...)`, and generate `RuntimeFeedback` objects after every capsule action. Keep `_run_capsule_trial()` as the orchestration point and use fake env/API tests for smoke coverage.

**Tech Stack:** Python dataclasses, existing `capx.runtime_control`, pytest, WSL `uv run --no-sync`, JSON artifacts.

---

## Working Rules

Use the Windows checkout for source edits. Run Python tests through WSL:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; <command>'
```

Before WSL tests, sync touched files from `/mnt/f/code/cap-x` into `/home/capx/code/cap-x`.

Do not run simulator rollouts, model servers, Robosuite launches, `uv sync`, or dependency installs.

---

### Task 1: Trace Windows

**Files:**
- Modify: `capx/runtime_control/trace.py`
- Modify: `tests/test_runtime_control_trace.py`

**Step 1: Write the failing test**

Add:

```python
def test_trace_window_returns_events_since_mark():
    trace = RuntimeTrace()
    start = trace.mark()
    trace.log({"name": "first"})
    trace.log({"name": "second"})

    assert [event["name"] for event in trace.events_since(start)] == ["first", "second"]
```

**Step 2: Run test to verify it fails**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; cp /mnt/f/code/cap-x/tests/test_runtime_control_trace.py tests/test_runtime_control_trace.py; uv run --no-sync pytest tests/test_runtime_control_trace.py -q'
```

Expected: FAIL with `AttributeError: 'RuntimeTrace' object has no attribute 'mark'`.

**Step 3: Implement trace windows**

Add:

```python
def mark(self) -> int:
    return len(self.events)

def events_since(self, index: int) -> list[dict[str, Any]]:
    return list(self.events[index:])
```

**Step 4: Run test to verify it passes**

Run the same pytest command.

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/runtime_control/trace.py tests/test_runtime_control_trace.py
git commit -m "Track capsule trace windows"
```

---

### Task 2: Bind Traced API Functions

**Files:**
- Modify: `capx/envs/tasks/base.py`
- Create: `tests/test_runtime_control_globals.py`

**Step 1: Write the failing test**

Use a fake environment object without constructing Robosuite:

```python
from capx.runtime_control.trace import RuntimeTrace
from capx.envs.tasks.base import CodeExecutionEnvBase


class FakeApi:
    def functions(self):
        return {"get_pose": self.get_pose}

    def get_pose(self, name):
        return {"name": name}


def test_capsule_globals_can_bind_traced_api_functions():
    env = object.__new__(CodeExecutionEnvBase)
    env.low_level_env = object()
    env._apis = {"fake": FakeApi()}

    trace = RuntimeTrace()
    globals_dict = env._build_capsule_globals(trace=trace)

    assert globals_dict["get_pose"]("cube") == {"name": "cube"}
    assert trace.events[0]["name"] == "get_pose"
```

**Step 2: Run test to verify it fails**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; cp /mnt/f/code/cap-x/tests/test_runtime_control_globals.py tests/test_runtime_control_globals.py; uv run --no-sync pytest tests/test_runtime_control_globals.py -q'
```

Expected: FAIL because `_build_capsule_globals()` does not accept `trace`.

**Step 3: Implement traced binding**

In `capx/envs/tasks/base.py`, import:

```python
from capx.runtime_control.trace import RuntimeTrace, wrap_function_for_trace
```

Change:

```python
def _build_capsule_globals(self, trace: RuntimeTrace | None = None) -> dict[str, Any]:
    ...
    g[fn_name] = wrap_function_for_trace(fn_name, fn, trace) if trace is not None else fn
```

**Step 4: Run test to verify it passes**

Run the same pytest command.

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/envs/tasks/base.py tests/test_runtime_control_globals.py
git commit -m "Bind traced APIs for capsule execution"
```

---

### Task 3: Attach Region Trace Events To RuntimeEvent

**Files:**
- Modify: `capx/runtime_control/executor.py`
- Modify: `tests/test_runtime_control_executor.py`

**Step 1: Write the failing test**

Add:

```python
from capx.runtime_control.trace import RuntimeTrace, wrap_function_for_trace


def test_executor_event_includes_region_trace_events():
    trace = RuntimeTrace()
    globals_dict = {"get_pose": wrap_function_for_trace("get_pose", lambda: [1, 2, 3], trace)}
    regions = segment_python_code("pose = get_pose()\n")
    executor = CapsuleExecutor(base_globals=globals_dict, trace=trace)

    event = executor.run_region(regions[0])

    assert event.evidence["trace_events"][0]["name"] == "get_pose"
```

**Step 2: Run test to verify it fails**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; cp /mnt/f/code/cap-x/tests/test_runtime_control_executor.py tests/test_runtime_control_executor.py; uv run --no-sync pytest tests/test_runtime_control_executor.py -q'
```

Expected: FAIL because `trace_events` is missing.

**Step 3: Implement trace event capture**

In `CapsuleExecutor.run_region()`:

```python
trace_start = self.trace.mark() if self.trace is not None else None
...
trace_events = self.trace.events_since(trace_start) if trace_start is not None else []
```

Include `trace_events` in success and failure `RuntimeEvent.evidence`.

**Step 4: Run test to verify it passes**

Run the same pytest command.

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/runtime_control/executor.py tests/test_runtime_control_executor.py
git commit -m "Attach trace events to capsule region results"
```

---

### Task 4: Source-Bound Feedback Builder

**Files:**
- Create: `capx/runtime_control/feedback.py`
- Modify: `capx/runtime_control/__init__.py`
- Create: `tests/test_runtime_control_feedback.py`

**Step 1: Write the failing tests**

```python
from capx.runtime_control.feedback import build_runtime_feedback
from capx.runtime_control.schema import CodeRegion, RuntimeAction, RuntimeEvent


def test_feedback_binds_failed_region_to_source_span():
    region = CodeRegion("region_2", 5, 7, "raise RuntimeError('bad')")
    action = RuntimeAction("run_region", {"region_id": "region_2"})
    event = RuntimeEvent(
        action="run_region",
        status="failed",
        region_id="region_2",
        message="bad",
        evidence={"exception_type": "RuntimeError"},
    )

    feedback = build_runtime_feedback(
        step_id=1,
        action=action,
        event=event,
        region=region,
        trace_events=[{"name": "goto_pose", "status": "failed"}],
        before_state={"reward": 0.0, "task_completed": False},
        after_state={"reward": 0.0, "task_completed": False},
    )

    assert feedback.status == "failed"
    assert feedback.region_id == "region_2"
    assert feedback.evidence["source_span"]["start_line"] == 5
    assert feedback.patch_scope == "region_2"


def test_feedback_warns_when_region_has_no_task_progress():
    region = CodeRegion("region_1", 1, 1, "x = 1")
    feedback = build_runtime_feedback(
        step_id=1,
        action=RuntimeAction("run_region", {"region_id": "region_1"}),
        event=RuntimeEvent(action="run_region", status="success", region_id="region_1"),
        region=region,
        trace_events=[],
        before_state={"reward": 0.0, "task_completed": False},
        after_state={"reward": 0.0, "task_completed": False},
    )

    assert feedback.status == "warning"
```

**Step 2: Run test to verify it fails**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; cp /mnt/f/code/cap-x/tests/test_runtime_control_feedback.py tests/test_runtime_control_feedback.py; uv run --no-sync pytest tests/test_runtime_control_feedback.py -q'
```

Expected: FAIL with missing `capx.runtime_control.feedback`.

**Step 3: Implement feedback builder**

Rules:

- Copy `source_span` from `CodeRegion`.
- Include `primitive_calls` as `[event["name"] for event in trace_events]`.
- Mark failed/invalid event statuses as failed/invalid.
- Mark success with unchanged reward and incomplete task as warning.
- Mark success with reward increase or task completion as success.

**Step 4: Run test to verify it passes**

Run the same pytest command.

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/runtime_control/feedback.py capx/runtime_control/__init__.py tests/test_runtime_control_feedback.py
git commit -m "Build source-bound capsule feedback"
```

---

### Task 5: Feed Trace And Feedback Through Capsule Trial Loop

**Files:**
- Modify: `capx/envs/trial.py`
- Modify: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write the failing test**

Extend the fake env with fake API functions:

```python
class FakeApi:
    def __init__(self):
        self.moved = False

    def functions(self):
        return {"get_pose": self.get_pose, "move_to": self.move_to}

    def get_pose(self, name):
        return [1, 2, 3]

    def move_to(self, pose):
        self.moved = True


class FakeCapsuleEnv:
    ...
    def __init__(self):
        self.api = FakeApi()
        self.low_level_env = object()
        self._apis = {"fake": self.api}
```

Add:

```python
def test_capsule_trial_writes_trace_and_feedback_artifact(tmp_path):
    _run_capsule_trial(
        env=FakeCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={...},
        initial_code='pose = get_pose("cube")\nmove_to(pose)\nRESULT = "done"\n',
        scripted_actions=[
            {"action": "run_region", "args": {"region_id": "region_1"}},
            {"action": "run_region", "args": {"region_id": "region_2"}},
            {"action": "inspect_trace", "args": {}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())
    assert trace[0]["feedback"]["region_id"] == "region_1"
    assert trace[0]["trace_events"][0]["name"] == "get_pose"
    assert trace[2]["event"]["evidence"]["events"][0]["name"] == "get_pose"
```

**Step 2: Run test to verify it fails**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; cp /mnt/f/code/cap-x/tests/test_runtime_control_trial_loop.py tests/test_runtime_control_trial_loop.py; uv run --no-sync pytest tests/test_runtime_control_trial_loop.py -q'
```

Expected: FAIL because history lacks feedback and real trace events.

**Step 3: Implement loop integration**

In `_run_capsule_trial()`:

```python
trace = RuntimeTrace()
executor = CapsuleExecutor(
    base_globals=env._build_capsule_globals(trace=trace),
    trace=trace,
)
```

Before and after each action:

```python
before_state = _capsule_state_snapshot(env)
event = ...
after_state = _capsule_state_snapshot(env)
trace_events = event.evidence.get("trace_events", [])
feedback = build_runtime_feedback(...)
```

Add helper:

```python
def _capsule_state_snapshot(env):
    return {"reward": _safe_compute_reward(env), "task_completed": _safe_task_completed(env)}
```

History entry shape:

```python
{
    "step_id": step_id,
    "action": ...,
    "event": ...,
    "feedback": feedback.to_dict(),
    "trace_events": trace_events,
    "state_before": before_state,
    "state_after": after_state,
}
```

**Step 4: Run test to verify it passes**

Run the same pytest command.

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/envs/trial.py tests/test_runtime_control_trial_loop.py
git commit -m "Record capsule trace feedback artifacts"
```

---

### Task 6: Focused Verification

**Files:**
- Modify only files needed to fix test or syntax failures.

**Step 1: Sync all touched files to WSL**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; cp -r /mnt/f/code/cap-x/capx/runtime_control capx/; cp /mnt/f/code/cap-x/capx/envs/tasks/base.py capx/envs/tasks/base.py; cp /mnt/f/code/cap-x/capx/envs/trial.py capx/envs/trial.py; cp /mnt/f/code/cap-x/tests/test_runtime_control_*.py tests/'
```

**Step 2: Run runtime-control tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_*.py -q'
```

Expected: all PASS.

**Step 3: Run import smoke**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync python -c "import capx.runtime_control; import capx.envs.trial; import capx.envs.tasks.base; print(123)"'
```

Expected: prints `123` with only optional dependency warnings.

**Step 4: Run syntax check**

Use this because Ruff was not available in the prepared WSL `.venv` during the previous run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync python -m py_compile capx/runtime_control/*.py capx/envs/trial.py capx/envs/tasks/base.py'
```

Expected: exit 0.

**Step 5: Commit fixes if needed**

```bash
git add capx/runtime_control capx/envs/trial.py capx/envs/tasks/base.py tests/test_runtime_control_*.py
git commit -m "Stabilize capsule trace feedback"
```

Skip this commit if no verification fixes were needed.

---

## Final Checklist

- Generated code still calls original API names.
- Traced wrappers are internal runtime bindings only.
- `inspect_trace` returns real primitive call events.
- Region run events include region-local trace evidence.
- Feedback points to `region_id` and source lines.
- `capsule_trace_trial_XX.json` includes action, event, feedback, trace events, and state snapshots.
- Runtime-control tests pass in WSL.
- Syntax check passes in WSL.

