# Runtime-Control Capsules Design

## Goal

Replace the low-level primitive toolification path with a runtime-control layer for
CaP-Agent0. The agent should continue to generate CaP-style Python code against the
original environment primitives, while the system makes that code interruptible,
inspectable, checkpointable, and locally patchable.

The key boundary is strict: tools must control code execution, not robot actions.

## Decision

Delete the existing primitive tool-agent path instead of keeping it as an ablation.
The current `agent_mode: tool` route exposes environment primitives such as
segmentation, IK, motion, and gripper control as planner-selected tools. That changes
the action interface seen by the agent and mixes with the original low-level and
high-level primitive abstractions.

The replacement mode is:

```yaml
agent_mode: capsule
```

In capsule mode, the LLM still writes Python. Runtime actions operate on code regions:

- `run_region`
- `inspect_trace`
- `inspect_variables`
- `patch_region`
- `rollback_to_checkpoint`
- `resume_from_region`
- `finish`

These actions do not perform manipulation directly.

## Current Code To Remove

Remove the primitive toolification implementation:

- `capx/tools/`
- `agent_mode: tool` branch and `_run_tool_trial` in `capx/envs/trial.py`
- Tool-facing methods in `capx/envs/tasks/base.py`
- `env_configs/*/*_tool_*.yaml`
- `tests/test_tool_*.py`
- `tests/test_franka_tool_metadata.py`
- Outdated toolification plans under `docs/plans/`

The normal code-generation path remains the baseline.

## Architecture

Add a new package:

```text
capx/runtime_control/
  schema.py
  segmenter.py
  trace.py
  checkpoints.py
  executor.py
  patching.py
  prompts.py
```

`schema.py` defines code-region and runtime event dataclasses. `segmenter.py` parses
generated Python with `ast` and splits it into executable regions. `trace.py` wraps the
existing API functions only to log calls and outputs. `executor.py` executes selected
regions in the existing persistent namespace. `patching.py` applies local patches to a
specific source span. `checkpoints.py` stores Python namespace snapshots first, with
optional simulator-state snapshots later.

## Data Flow

```text
Task instruction
  -> LLM generates complete CaP-style Python code
  -> Runtime segments code into CodeRegions
  -> Planner chooses runtime-control action
  -> Executor runs or inspects a source region
  -> Trace captures primitive calls without changing primitive names
  -> Verifier binds feedback to region_id and source lines
  -> LLM patches only the implicated source region
  -> Runtime resumes from an allowed checkpoint
```

The LLM never receives a tool list containing robot primitives such as `solve_ik` or
`move_to_joints`. It receives code-region status, traces, variable summaries, and
patch scopes.

## Code Regions

The first version should segment top-level Python statements into coarse regions:

- imports and constants
- perception binding
- geometry or pose computation
- pick or grasp execution
- placement computation
- placement execution
- final verification or cleanup

Explicit markers can refine this later:

```python
# capx: checkpoint after_grasp_pose
```

MVP segmentation should be deterministic and conservative. If segmentation is unsure,
it should create fewer, larger regions rather than fragmenting side-effectful code.

## Tracing

The runtime wraps existing API functions inside the execution namespace:

```python
def traced_fn(*args, **kwargs):
    trace.log_call(name, args, kwargs)
    result = original_fn(*args, **kwargs)
    trace.log_return(name, result)
    return result
```

This wrapper is internal. The generated code still calls the original function names.
Feedback may mention that a primitive call occurred, but repair guidance must point to
source lines and code regions, not to a new robot-tool action.

## Checkpoints And Rollback

Use two tiers:

1. Python namespace checkpoints for MVP.
   Store deepcopy-able globals, variable summaries, stdout/stderr, reward, completion
   state, and trace ranges.

2. Optional simulator checkpoints for Robosuite.
   Add an adapter later for MuJoCo state such as `qpos`, `qvel`, and `ctrl`. If an
   environment does not support simulator rollback, return structured
   `rollback_unsupported` feedback.

The MVP should not promise arbitrary physical rollback across all environments.

## Feedback

Feedback must be source-bound:

```text
Failure in region_4, source lines 18-23.
Primitive calls completed, but the local postcondition failed.
Observed: reward unchanged and object not in gripper.
Likely local cause: grasp pose z is too high.
Patch scope: region_3 grasp pose computation only.
```

This keeps the research claim focused on shorter feedback cycles, better failure
localization, and smaller patches.

## Error Handling

Syntax errors should stop before segmentation and request a full code regeneration.
Runtime exceptions should attach traceback summaries to the active region. Invalid
runtime-control actions should produce structured invalid-action feedback and consume a
retry budget. Patch failures should preserve the last valid program and report the
failed hunk.

Rollback failures must not silently continue from a mismatched state. If rollback is
unsupported or partial, the runtime should say so and either continue from current
state or request full regeneration according to config.

## Testing

The first test suite should be code-level only and run in WSL without simulator or
model servers:

- Code segmentation tests with small Python programs.
- Region execution tests with fake APIs.
- Trace wrapper tests proving call logging does not change return values.
- Patch application tests scoped to one region.
- Capsule trial-loop tests with scripted runtime-control actions.
- Config parsing tests for `agent_mode: capsule`.
- Regression tests that `agent_mode: code` remains unchanged.

Runtime Robosuite verification should remain an explicit optional check, run only in
the prepared WSL or remote experiment environment.

