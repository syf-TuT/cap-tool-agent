# LLM-Step Action Ablation Design

## Goal

Add independently configurable patch and append-recovery permissions to Capsule's
`llm_step` control mode, then provide non-VDM cube-stack configurations for the A2 and A3
ablation groups.

## Experiment Groups

| Group | `patch_group` / `patch_region` | `append_recovery` |
| --- | --- | --- |
| A2 | disabled | enabled |
| A3 | enabled | disabled |

The two groups use the same non-VDM cube-stack environment, API servers, trial count, and
worker count. Their action permissions and output directories are the only intended
differences.

## Configuration

Add two configuration fields:

- `capsule_llm_step_allow_patch`, defaulting to `true`;
- `capsule_llm_step_allow_append_recovery`, defaulting to `true`.

Defaults preserve the existing `llm_step` behavior. Patch permission applies to both
`patch_group` and `patch_region`. It does not disable `resume_from_region`, which executes
existing source rather than replacing source text.

Create two runnable configurations based on the non-VDM
`env_configs/cube_stack/franka_robosuite_cube_stack.yaml` setup:

- `franka_robosuite_cube_stack_capsule_llm_step_a2_no_patch.yaml`;
- `franka_robosuite_cube_stack_capsule_llm_step_a3_no_append_recovery.yaml`.

Both explicitly select `agent_mode: capsule` and `capsule_control_mode: llm_step` and use
separate output directories.

## Prompt Behavior

The Capsule action prompt receives both permissions. Disabled actions are removed from the
allowed-action list, JSON examples, recovery guidance, and action-specific rules. This avoids
encouraging the model to select an action that the experiment forbids.

When patching is disabled, neither `patch_group` nor `patch_region` is advertised. When
append recovery is disabled, `append_recovery` is not advertised even if the active API
provides fresh-state observation functions.

## Runtime Enforcement

Prompt filtering is not treated as enforcement. Before the existing no-rollback and reward-drop
guards, the `llm_step` loop checks the selected action against the configured permissions.
A forbidden action produces a structured `invalid` event and does not execute or modify the
saved source. This also covers scripted actions and malformed model behavior while preserving
the attempted action in the trace.

The step metric records both resolved permission values so experiment artifacts can be audited.

## Compatibility and Scope

The global runtime-action schema remains unchanged because the actions are still valid in other
Capsule configurations. `auto_forward` behavior is unchanged. Existing `llm_step`
configurations continue to allow both action families through the default values.

## Testing

Tests cover:

- default and explicit configuration loading;
- prompt omission of disabled action names, examples, and instructions;
- runtime rejection of forbidden patch and append-recovery actions;
- continued execution of the action family that remains enabled in A2 and A3;
- step-metric permission fields;
- YAML group identity, non-VDM settings, and the exact A2/A3 permission matrix.

