# Cube Stack Non-Privileged Prompt-Parity Design

## Goal

Create a fair `cube_stack` comparison between the multiturn and Capsule
execution methods.  Both variants must expose the same non-privileged API and
receive the same initial task instruction.  Their only intentional differences
are the execution and recovery mechanisms of each method.

## Scope

The existing VDM configurations are historical experiment definitions and must
remain unchanged.  Add a new, clearly named paired configuration for the
matched comparison:

- `franka_robosuite_cube_stack_multiturn_vdm_matched.yaml`
- `franka_robosuite_cube_stack_capsule_vdm_matched.yaml`

Both configurations use `privileged: false` and
`FrankaControlApiReducedSkillLibrary`.  They share byte-for-byte identical
initial prompt text that directs the agent to use only the exposed
non-privileged APIs.

## Configuration Design

The common prompt states the cube-stacking goal, forbids privileged state APIs,
and requires executable Python.  It does not contain any method-specific
instruction:

- The multiturn configuration retains `multi_turn_prompt`; that later prompt
  controls `REGENERATE` / `FINISH` decisions after execution.
- The Capsule configuration retains its `agent_mode`, checkpoint, rollback,
  and repair settings; the Capsule runtime supplies its own control prompts.

The matched multiturn configuration intentionally omits
`use_legacy_multi_turn_decision_prompt`. That top-level YAML field is not
loaded into the runtime CLI arguments, so retaining it would be dead
configuration rather than an execution control. Omitting it uses the actual
default, non-legacy multiturn decision path. This matched-benchmark decision
does not change loader behavior or any legacy configuration.

Neither configuration uses `FrankaPickPlaceCodeEnv.PROMPT`, because that
prompt instructs the agent to use privileged state calls such as
`get_object_pose` and `sample_grasp_pose`, which are unavailable in this
non-privileged benchmark.

All other comparable settings remain aligned: simulator, API servers, video
recording, image differencing, trial count, and worker count.  Output
directories remain distinct to avoid mixing artifacts.

## Validation

Add a focused configuration test that loads both YAML files and asserts:

1. `privileged` is false in both variants.
2. Both expose exactly `FrankaControlApiReducedSkillLibrary`.
3. Their initial `cfg.prompt` values are identical.
4. The matched multiturn YAML omits the dead legacy decision field, while the
   multiturn-only and Capsule-only control fields remain in their respective
   configurations.

Run this focused test before rerunning either benchmark.  Historical VDM
results remain labelled as non-matched and are not used for the new comparison.
