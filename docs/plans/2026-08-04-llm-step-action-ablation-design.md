# Cube-Stack Action Ablation Design

## Goal

Make the cube-stack action ablation a controlled four-group experiment and remove geometric
answer hints from the task's default prompt. The experiment should isolate one-shot generation,
patching, and append-only recovery without changing the initial simulator state in this change.

## Experiment Groups

| Group | Execution mode | Patch | `append_recovery` |
| --- | --- | --- | --- |
| A1 | one-shot `code` agent | unavailable | unavailable |
| A2 | Capsule `llm_step` | disabled | enabled |
| A3 | Capsule `llm_step` | enabled | disabled |
| A4 | Capsule `llm_step` | enabled | enabled |

A1 must use `agent_mode: code`. Setting both permissions to `false` while retaining
`agent_mode: capsule` would not be a one-shot baseline: the model would still receive Capsule
action prompts, choose semantic groups to execute, and observe execution feedback. The ordinary
code agent performs one initial model generation and has no multi-turn prompt in this config.

For a uniform, auditable matrix, A1 retains the Capsule-only keys with both permissions set to
`false`; those keys are inert in code-agent mode. A2 through A4 are identical except for the two
permission booleans and their output directories. Relative to those groups, A1 additionally
differs in `agent_mode` and its output directory.

## Prompt

Restore the `FrankaPickPlaceCodeEnv` default prompt to the generic prompt used by the original
comparison experiment. It states the task, the non-privileged perception and planning stack,
and the required output format, but does not prescribe:

- the stacking-height formula;
- red or green cube half-height calculations;
- grasp-quaternion reuse;
- a minimum post-grasp lift distance.

All four configurations inherit this same class-level prompt. No group-specific problem-solving
hint is added to YAML.

## Shared Configuration

All groups use the same:

- non-privileged `FrankaControlApi` environment;
- local SAM3, Contact-GraspNet, and Pyroki servers;
- disabled VDM, image differencing, and visual feedback;
- video recording, trial count, worker count, and Capsule budget fields;
- task prompt inherited from `FrankaPickPlaceCodeEnv`.

The model endpoint, model name, temperature, seed selection, and per-run timeout remain launch
parameters rather than group-specific YAML differences.

## Scope

This change adds A1 and A4 YAML files, updates A2/A3 only as needed for exact matrix identity,
and restores the prompt. It does not change runtime-control implementation, simulator reset
behavior, object placement, seed handling, or the existing permission semantics.

## Testing

Regression tests will:

- assert the default prompt equals the intended generic text and contains none of the leaked
  geometric instructions;
- assert the exact A1-A4 execution-mode and permission matrix;
- assert every environment, API, perception, feedback, recording, budget, trial, and worker
  setting is shared;
- assert A2-A4 differ only in permission switches and output paths;
- assert A1 differs from the shared Capsule configuration only in `agent_mode`, permission
  switches, and output path.
