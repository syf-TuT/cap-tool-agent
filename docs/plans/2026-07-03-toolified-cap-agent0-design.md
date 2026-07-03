# Toolified CaP-Agent0 Design

## Goal

Transform CaP-Agent0 from a program-level repair loop into a tool-level execution loop:
instead of asking the planner to generate a large Python program, the planner selects one
structured tool call at a time. The system validates the call, executes it, verifies the
local effect, and returns structured feedback before the next step.

The first implementation targets Robosuite Franka manipulation tasks, starting with
cube stacking, cube lifting, and cube restacking. It reuses the existing reduced Franka
API and keeps the current code-generation runner as the baseline path.

## Current Context

The existing execution stack already has the right substrate:

- `capx/envs/tasks/base.py` exposes `CodeExecutionEnvBase`, which executes generated
  Python code with a persistent namespace.
- `capx/integrations/base_api.py` defines `ApiBase.functions()`, which exposes callable
  robot APIs and prompt documentation.
- `capx/integrations/franka/control_reduced.py` and
  `capx/integrations/franka/control_reduced_skill_library.py` provide the semantic
  primitives needed for perception, grasp planning, IK, and gripper/motion control.
- `capx/envs/trial.py` already runs a multi-turn repair loop after each executed code
  block and can capture visual or video differencing feedback.
- `env_configs/*/*multiturn_vdm_reduced_api*.yaml` already define reduced API and
  visual feedback baselines for ablation.

The missing capability is not a lack of functions. The missing capability is that the
system cannot observe and verify each individual function call when those calls are
hidden inside LLM-generated Python.

## Approved Scope

Implement the first version for Robosuite Franka tasks only:

- Cube stack
- Cube lifting
- Cube restack

Do not include real robot control, LIBERO, R1Pro, MCP integration, or external tool
servers in the first version. Those can reuse the same abstractions later once the
Robosuite path is validated.

## Non-Goals

- Do not replace the current code-generation runner.
- Do not expose joint-level incremental control to the LLM.
- Do not require full OpenAI-style native function calling in the first version.
- Do not rewrite the existing Franka API implementation.
- Do not make skills opaque macro-actions; skills should remain expandable tool-call
  traces.

## Recommended Approach

Use a native tool-call trial loop beside the existing code trial loop.

The new flow is:

```text
Task
  -> Planner LLM
  -> Tool Router
  -> Atomic Tool Executor
  -> Step Verifier
  -> Structured Feedback
  -> Planner LLM selects next tool or finishes
```

This preserves existing baselines while adding a new agent mode that makes control and
feedback explicit at the tool-call level.

## Architecture

### Tool Schema Layer

Create `capx/tools/schema.py` with small dataclasses:

- `ToolSpec`: name, description, JSON-like input schema, output policy,
  preconditions, postconditions, failure modes, tags.
- `ToolCall`: step id, tool name, arguments, optional rationale.
- `ToolResult`: status, raw output, serialized output, stdout, stderr, duration,
  exception metadata.
- `StepFeedback`: status, failure stage, failure type, evidence, repair hints,
  recommended next tools.

The schema should be serializable to JSON so it can be saved in trial artifacts and
used directly in prompts.

### Tool Registry

Create `capx/tools/registry.py`.

The registry should build tool specs from existing `ApiBase.functions()` entries. The
first version can infer function names, signatures, and docstrings with `inspect`, then
apply a curated metadata overlay for Franka tools that need stronger semantics.

Example metadata overlay:

```python
{
    "segment_sam3_text_prompt": {
        "tags": ["perception"],
        "postconditions": ["non_empty_mask"],
        "failure_modes": ["object_not_found", "low_confidence_mask"],
    },
    "solve_ik": {
        "tags": ["planning"],
        "preconditions": ["target_pose_available"],
        "postconditions": ["joint_solution_valid"],
        "failure_modes": ["unreachable_pose", "invalid_pose", "ik_nonconvergence"],
    },
    "move_to_joints": {
        "tags": ["execution"],
        "preconditions": ["joint_solution_valid"],
        "postconditions": ["robot_reached_target"],
        "failure_modes": ["motion_timeout", "collision", "controller_error"],
    },
}
```

The registry should allow only explicitly registered tools. The planner must not call
arbitrary Python or access `env` directly in tool mode.

### Tool Execution Environment

Extend `CodeExecutionEnvBase` with a small tool-facing API:

- `tool_specs() -> list[ToolSpec]`
- `call_tool(tool_call: ToolCall) -> ToolResult`
- `snapshot_state() -> dict[str, Any]`
- `task_status() -> dict[str, Any]`

The implementation should reuse existing API instances in `self._apis`. It should call
the same functions that are currently injected into the Python execution namespace, but
through the registry and executor.

### State Store

Create `capx/tools/state.py`.

The tool loop needs references across steps without forcing the LLM to paste large
arrays into JSON. Tool outputs should be summarized and large values should be stored
behind state refs.

Examples:

```json
{
  "ref": "mask.red_cube_0",
  "summary": {
    "shape": [480, 640],
    "area": 1240,
    "score": 0.91
  }
}
```

The planner sees summaries and refs. The executor resolves refs before calling Python
functions. This avoids passing base64 images, masks, point clouds, and grasp matrices
through the LLM context.

### Tool Executor

Create `capx/tools/executor.py`.

Responsibilities:

- Validate tool name.
- Resolve state refs.
- Coerce simple JSON arguments into Python types.
- Reject unknown arguments.
- Capture stdout, stderr, exceptions, duration, and raw result.
- Store large outputs in `ToolState`.
- Return compact serialized outputs for the planner.

The executor should not swallow exceptions silently. Exceptions become `ToolResult`
with `status="failed"` and structured exception metadata.

### Step Verifier

Create `capx/tools/verifiers.py`.

The verifier runs after every tool call and uses:

- Tool metadata.
- Tool result.
- State snapshot before the call.
- State snapshot after the call.
- Low-level environment reward and task-completion signal.
- Optional image/video differencing feedback for execution tools.

Initial verifier coverage:

- Perception: mask exists, area threshold, score threshold, valid depth coverage.
- Planning: output shape, no NaN/Inf, IK output size, workspace sanity.
- Execution: robot pose changed when expected, gripper state changed when expected,
  no environment truncation, no exception.
- Task relations: object lifted, object on target, object placed near target, task
  completed.

For the first pass, relation verifiers can be task-aware and Robosuite-specific. A
general relation DSL can come later.

### Planner Prompt

Create `capx/tools/prompts.py`.

The planner prompt should ask for exactly one JSON object:

```json
{
  "thought": "brief reason",
  "tool": "tool_name",
  "args": {}
}
```

It should also allow a finish action:

```json
{
  "thought": "brief reason",
  "tool": "finish",
  "args": {}
}
```

The prompt should include:

- Task instruction.
- Current state summary.
- Available tool specs.
- Recent tool history.
- Last structured feedback.
- A strict instruction not to write Python code.

### Tool Trial Loop

Add a new loop in `capx/envs/trial.py`, for example `_run_tool_trial()`, selected by
config:

```yaml
agent_mode: tool
```

Pseudo-flow:

```python
obs, _ = env.reset(options={"trial": trial}, seed=trial)
tool_state = ToolState()
history = []

for step_id in range(max_tool_steps):
    before = env.snapshot_state()
    prompt = build_tool_prompt(obs, tool_specs, tool_state.summary(), history)
    model_response = query_model(args, prompt)
    tool_call = parse_tool_call(model_response)

    routed = router.validate(tool_call, tool_specs, tool_state)
    if not routed.ok:
        feedback = feedback_for_invalid_call(routed)
        history.append(feedback)
        continue

    result = env.call_tool(tool_call)
    after = env.snapshot_state()
    feedback = verifier.verify(tool_call, result, before, after)
    history.append({"call": tool_call, "result": result, "feedback": feedback})

    if task_verifier.success(env, feedback):
        break
```

The loop should save artifacts similarly to the current code loop:

- `tool_trace.json`
- `tool_prompts.json`
- `tool_feedback.json`
- final summary log
- videos/images if enabled

### YAML Configuration

Add tool-mode configs beside existing baselines:

```text
env_configs/cube_stack/franka_robosuite_cube_stack_tool_vdm.yaml
env_configs/cube_lifting/franka_robosuite_cube_lifting_tool_vdm.yaml
env_configs/cube_restack/franka_robosuite_cube_restack_tool_vdm.yaml
```

Example config fields:

```yaml
agent_mode: tool
max_tool_steps: 20
tool_feedback_level: repair_hint
use_img_differencing: true
record_video: true
output_dir: ./outputs/franka_robosuite_cube_stack_tool_vdm
```

Existing configs continue using the default code mode.

## Tool Granularity

Use semantic atomic tools. The LLM should control manipulation-level decisions, not
joint-level servoing.

First-version tool set:

- `get_observation`
- `segment_sam3_text_prompt`
- `segment_sam3_point_prompt`
- `point_prompt_molmo`
- `mask_to_world_points`
- `get_oriented_bounding_box_from_3d_points`
- `plan_grasp`
- `select_top_down_grasp`
- `solve_ik`
- `move_to_joints`
- `open_gripper`
- `close_gripper`
- `verify_grasp`
- `verify_place_relation`
- `verify_task_success`

The first twelve mostly reuse existing APIs. The verifier tools are system-owned and
should be exposed as callable checks, not as arbitrary Python functions.

## Structured Feedback

Every step should produce a compact JSON feedback object:

```json
{
  "step_id": 5,
  "tool": "move_to_joints",
  "status": "failed",
  "failure_stage": "execution",
  "failure_type": "motion_timeout",
  "evidence": {
    "target_ref": "joints.red_cube_grasp_0",
    "reward_before": 0.0,
    "reward_after": 0.0
  },
  "repair_hints": [
    "try a higher pregrasp pose",
    "sample another grasp candidate"
  ],
  "recommended_next_tools": [
    "select_top_down_grasp",
    "solve_ik"
  ]
}
```

Feedback levels should be configurable for ablation:

- `binary`: status only.
- `failure_type`: status plus failure type.
- `repair_hint`: failure type plus repair hints.
- `vdm`: repair hints plus visual/video differencing summary.

## Skill Library Design

Keep the existing Python-function skill library unchanged for code-mode baselines.

Add a tool-trace skill representation later in the same package:

```json
{
  "name": "pick_object",
  "source_tasks": ["cube_lifting"],
  "steps": [
    {"tool": "get_observation", "args": {}},
    {"tool": "segment_sam3_text_prompt", "args": {"text_prompt": "$object_name"}},
    {"tool": "plan_grasp", "args": {"segmentation": "$mask_ref"}},
    {"tool": "solve_ik", "args": {"position": "$grasp_position"}},
    {"tool": "move_to_joints", "args": {"joints": "$joints_ref"}},
    {"tool": "close_gripper", "args": {}},
    {"tool": "verify_grasp", "args": {"object": "$object_name"}}
  ]
}
```

A skill is therefore not a black-box action. It is an expandable, inspectable, and
interruptible subgraph of tool calls.

## Experiments

Compare:

- Baseline A: single-turn full Python program.
- Baseline B: multi-turn full Python program plus VDM.
- Ours 1: tool call plus binary success/failure feedback.
- Ours 2: tool call plus failure type.
- Ours 3: tool call plus failure type and repair hint.
- Ours 4: tool call plus repair hint, verifier, and step VDM.

Primary metrics:

- Task success rate.
- Average LLM turns.
- Average tool calls.
- Recovery success rate after first failure.
- Failure localization accuracy.
- Number of full-program regenerations.
- Time to first useful correction.
- Feedback latency, defined as the number of steps from failure occurrence to usable
  diagnostic feedback.

## Risks

- Tool mode may increase LLM calls. Mitigation: keep tools semantic and add reusable
  tool-trace skills after the loop works.
- Argument serialization may become brittle for arrays and poses. Mitigation: use
  state refs for large values and explicit coercion for common types.
- Verifiers may be task-specific at first. Mitigation: make verifier selection explicit
  by task family and only generalize after Robosuite cube tasks are stable.
- Planner may emit invalid JSON. Mitigation: strict parser, invalid-call feedback, and
  a retry budget.
- VDM cost may grow if run after every tool. Mitigation: only run VDM after execution
  tools or when a verifier reports uncertainty.

## Success Criteria

The design is successful when:

- Existing code-mode configs still run unchanged.
- Tool-mode configs can execute at least one Robosuite Franka cube task end to end.
- Every tool call is logged with inputs, summarized outputs, verifier result, and
  structured feedback.
- The first ablation can compare existing multi-turn VDM against tool-mode feedback
  on the same task family.

