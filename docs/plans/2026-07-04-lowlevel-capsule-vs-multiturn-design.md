# Low-Level Capsule vs Multiturn Robosuite Benchmark Design

## Objective

Compare Capsule-style region execution against the original multiturn regeneration
baseline on Robosuite tasks while holding the available robot primitives fixed.

The first pilot answers whether Capsule improves task completion or efficiency
relative to regenerating the remaining task code after each execution step.

## Scope

Pilot tasks:

- `cube_stack`
- `nut_assembly`

Methods:

- `mt_regenerate_lowlevel`: execute generated code blocks and ask the model to
  regenerate the remaining code or finish.
- `capsule_lowlevel`: execute code regions through the runtime-control Capsule
  loop with `max_capsule_steps: 60`.

All pilot runs use:

- Model: `deepseek-v4-flash`
- Server URL: `https://www.packyapi.com/v1/chat/completions`
- Temperature: `0.2`
- Max tokens: `8192`
- Trial timeout: `720s`
- Workers: `1`
- Videos: enabled
- Seeds: Robosuite trial ids `1..5`

## Low-Level Primitive Constraint

Every compared method must expose the same low-level primitive surface for a
given task. The pilot uses privileged state access only to avoid mixing in
visual perception model quality.

Disallowed in the pilot:

- `FrankaControlApiReducedSkillLibrary`
- SAM/contact grasp servers
- VDM/image/video differencing
- Visual-only APIs such as `FrankaControlNutAssemblyVisualApi`
- Oracle code

Allowed primitive examples:

- `get_object_pose(...)`
- `sample_grasp_pose(...)`
- `goto_pose(...)`
- `open_gripper()`
- `close_gripper()`
- `goto_home_joint_position()` when the task API exposes it

## Metrics

Primary:

- Success rate: `task_completed=True`
- Average reward

Secondary:

- Wall-clock seconds
- Timeout rate
- Sandbox error rate
- Number of LLM calls
- Capsule action counts: `run_region`, `inspect_variables`, `patch_region`, `finish`
- Multiturn regeneration and finish counts
- Video availability

## Execution Design

Pilot matrix:

| Task | Method | Trials |
| --- | --- | --- |
| cube_stack | mt_regenerate_lowlevel | 1..5 |
| cube_stack | capsule_lowlevel | 1..5 |
| nut_assembly | mt_regenerate_lowlevel | 1..5 |
| nut_assembly | capsule_lowlevel | 1..5 |

The runner randomizes these 20 single-trial runs with a fixed order seed to
reduce time-of-day and provider-load confounds.

Each single-trial run uses a temporary YAML with:

- `trials: <trial_id>`
- `resume_idx: <trial_id>`
- `num_workers: 1`
- a unique output directory under `./outputs/bench_lowlevel/...`

## Analysis Plan

For each task and method, compute success rate, average reward, median runtime,
timeout count, and error count. Then compare methods within each task on the
same trial ids.

The first decision point is practical rather than statistical: if Capsule wins
on both tasks or clearly improves the hard task without regressing the easy task,
expand to 20 trials per task and add `cube_restack`.
