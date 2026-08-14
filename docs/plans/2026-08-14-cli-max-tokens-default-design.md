# CLI Max Tokens Default Design

## Problem

The CLI evaluation entry points default model completions to `2048 * 10` tokens. This
causes ordinary CaP-X experiments to request up to 20,480 tokens even when the YAML
does not opt into a large completion budget. In the LIBERO PackyAPI run, the real
initial-code request timed out twice before receiving an HTTP response.

## Decision

Use `2048` as the default completion budget throughout the CLI experiment path:

- `LaunchArgs` in `capx/envs/launch.py`;
- the two defensive Capsule-action fallbacks in `capx/envs/trial.py`;
- `LiberoBatchLaunchArgs` in `capx/envs/scripts/run_libero_batch.py`;
- `BatchLaunchArgs` in `capx/envs/scripts/run_batch.py`.

Explicit CLI or YAML overrides remain unchanged. Web API defaults remain at 20,480
tokens because changing them would broaden the behavior change beyond CLI experiments.

## Verification

Add focused tests that instantiate the three CLI argument dataclasses and assert a
2,048-token default. Verify the Capsule-action fallback source no longer introduces
the old 20,480-token value, then run the focused test suite. Finally, sync the committed
change to the SeeTaCloud worktree and run one LIBERO trial without a `--max-tokens`
override.
