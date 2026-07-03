# Capsule Trace Feedback Design

## Goal

Close the first practical loop for capsule runtime control:

1. Generated CaP-style Python still calls the original environment API names.
2. Runtime wraps those API functions internally and records primitive call traces.
3. Region execution feedback is bound to `region_id` and source lines.
4. Scripted smoke tests prove trace, feedback, and artifacts work without model or simulator services.

This keeps the research boundary clear: the system toolifies code execution control,
not robot primitives.

## Current State

The repository already has a capsule-mode skeleton:

- `capx/runtime_control/trace.py` records function calls but is not connected to capsule globals.
- `capx/runtime_control/executor.py` executes `CodeRegion` objects and returns `RuntimeEvent`.
- `capx/envs/tasks/base.py` exposes `_build_capsule_globals()`, but it currently binds raw API functions.
- `capx/envs/trial.py` saves capsule artifacts, but feedback is only an event log.

The next change should make those pieces work together rather than adding new planner
surface area.

## Trace Integration

Change `_build_capsule_globals()` to accept an optional `RuntimeTrace`:

```python
def _build_capsule_globals(self, trace: RuntimeTrace | None = None) -> dict[str, Any]:
    ...
    for api in self._apis.values():
        for fn_name, fn in api.functions().items():
            g[fn_name] = wrap_function_for_trace(fn_name, fn, trace) if trace else fn
```

The generated code still sees the same names, such as `get_object_pose` and
`goto_pose`. Only the runtime binding changes.

`RuntimeTrace` should add trace-window helpers:

```python
def mark(self) -> int:
    return len(self.events)

def events_since(self, index: int) -> list[dict[str, Any]]:
    return self.events[index:]
```

`CapsuleExecutor.run_region()` should mark the trace before execution and include the
region-local trace events in the returned `RuntimeEvent.evidence`.

## Source-Bound Feedback

Add `capx/runtime_control/feedback.py`.

The main API should be:

```python
def build_runtime_feedback(
    *,
    step_id: int,
    action: RuntimeAction,
    event: RuntimeEvent,
    region: CodeRegion | None,
    trace_events: list[dict[str, Any]],
    before_state: dict[str, Any],
    after_state: dict[str, Any],
) -> RuntimeFeedback:
    ...
```

Rules for the first version:

- Failed `run_region`: `status="failed"`, include `region_id`, `source_span`,
  exception evidence, stderr summary, and region-local primitive calls.
- Successful `run_region` with unchanged reward and incomplete task:
  `status="warning"`, message says execution succeeded but no local task progress was observed.
- Successful `run_region` with reward increase or task completion: `status="success"`.
- Failed primitive calls remain evidence, not planner actions. Feedback wording points
  to source lines and patch scope.

The feedback object should not tell the model to call `goto_pose` as a tool. It should
say which source region should be inspected or patched.

## Trial Loop Changes

In `_run_capsule_trial()`:

1. Create `trace = RuntimeTrace()`.
2. Build globals with `env._build_capsule_globals(trace=trace)`.
3. Create `CapsuleExecutor(..., trace=trace)`.
4. For each action, capture lightweight state before and after:

```python
{"reward": ..., "task_completed": ...}
```

5. After each event, call `build_runtime_feedback()`.
6. Save history entries with:

```json
{
  "step_id": 1,
  "action": {},
  "event": {},
  "feedback": {},
  "trace_events": [],
  "state_before": {},
  "state_after": {}
}
```

The artifact file remains `capsule_trace_trial_XX.json`, but it now contains actual
primitive call evidence and feedback.

## Scripted Smoke

Use fake APIs and scripted runtime actions. Do not run model, Robosuite, Pyroki, SAM,
or GraspNet.

The smoke should run code like:

```python
pose = get_pose("cube")
move_to(pose)
RESULT = "done"
```

With actions:

```json
[
  {"action": "run_region", "args": {"region_id": "region_1"}},
  {"action": "run_region", "args": {"region_id": "region_2"}},
  {"action": "inspect_trace", "args": {}},
  {"action": "finish", "args": {}}
]
```

Assertions:

- `capsule_trace_trial_01.json` exists.
- `get_pose` and `move_to` appear in trace events.
- Feedback entries include `region_id` and `source_span`.
- `inspect_trace` returns real events.
- Generated source remains Python, not JSON tool calls.

## Non-Goals

- Do not implement simulator state rollback here.
- Do not run live Robosuite or API server checks here.
- Do not expose robot primitives as planner-selectable tools.
- Do not add a new DSL for task progress; use reward/task completion and trace evidence only.

