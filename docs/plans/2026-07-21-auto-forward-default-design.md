# Auto-Forward Default Design

## Goal

Make Capsule trials use `auto_forward` when `capsule_control_mode` is omitted,
while preserving explicit `llm_step` selection for compatibility and ablation
runs.

## Design

Change both fallback sites that define the effective default:

- `capx/utils/launch_utils.py` must place `auto_forward` into the merged launch
  configuration when YAML does not specify a control mode.
- `capx/envs/trial.py` must dispatch directly to the auto-forward loop when a
  caller bypasses launch configuration merging and omits the mode.

No mode is removed. An explicit `capsule_control_mode: llm_step` continues to
select the existing stepwise LLM loop.

## Testing

Add an assertion that configuration loading defaults to `auto_forward`, and a
focused dispatch test that proves `_run_capsule_trial` selects auto-forward for
an empty configuration. Retain the existing explicit `llm_step` behavior test.

