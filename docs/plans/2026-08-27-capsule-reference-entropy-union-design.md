# Capsule reference entropy union fix design

## Context

The first formal multi-step Cube Lift LoRA run completed five actor updates and then
failed while merging the sixth adapter-disabled reference log-probability result. The
pinned VeRL `FSDPWorker.compute_log_prob` returns both `old_log_probs` and diagnostic
`entropys`. Capsule calls the same method twice: once with the LoRA adapter enabled for
the old policy and once with the adapter disabled for the frozen-base reference policy.

Capsule currently unions both complete outputs into the training batch. Once LoRA has
changed the actor, the two `entropys` tensors are no longer equal. VeRL's strict
`DataProto.union` therefore rejects the duplicate field. Official VeRL PPO code removes
`entropys` before unioning the old-log-probability output, and its ordinary reference
worker returns only `ref_log_prob`.

## Decision

Match the official VeRL data contract at the Capsule boundary:

- Remove optional diagnostic `entropys` from the actor old-log-probability output before
  unioning it into the training batch.
- Remove optional diagnostic `entropys` from the adapter-disabled reference output before
  renaming `old_log_probs` to `ref_log_prob`.
- Preserve the required `old_log_probs`, `ref_log_prob`, adapter-disable metadata cleanup,
  and update ordering.

The training batch does not consume entropy diagnostics, so retaining or renaming them
would add memory and create an unsupported contract. Relaxing the generic merge helper
would be broader and could hide unrelated duplicate-field corruption.

## Test strategy

Add a regression test with a strict DataProto-like `union` implementation. The actor
returns different entropy tensors for adapter-enabled and adapter-disabled calls, as it
does after real LoRA updates. Before the fix the test must fail with VeRL's duplicate
`entropys` assertion. After the fix it must prove that the update receives
`old_log_probs` and `ref_log_prob`, receives no `entropys`, and leaves no adapter-disable
metadata behind.

Run the focused trainer test, the complete Capsule test suite, then repeat the remote
canonical Gate 1--7 workflow and adapter reload. Because the failed formal run did not
publish a checkpoint, archive its five group artifacts and log, create a new immutable
Gate/bundle identity at the fixed Git commit, and restart the 20-seed training from step
zero. Monitor it with foreground shell commands rather than an automation.
