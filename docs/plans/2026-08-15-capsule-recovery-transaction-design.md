# Capsule Recovery Transaction Design

## Context

The `libero_object_0` seed 2 run exposed a recovery-control loop. After an
`append_recovery`, the model executed only the first recovery group, then
appended another nearly identical recovery instead of executing the remaining
groups. The pending groups were preserved in lineage, but the runtime made a
new append available as soon as any later primitive trace existed.

The current gate therefore protects only against two consecutive appends with
no intervening trace. It does not protect the recovery generation as a unit of
work.

## Goals

- Treat the latest recovery generation as a transaction that must be completed
  or repaired before another recovery can be appended.
- Keep the model in the latest recovery generation while it has pending work.
- Enforce the same restriction in the prompt and in authoritative runtime
  guards.
- Preserve LLM-step control: the model still selects each recovery group and
  may patch a pending recovery group.
- Add regression tests for the exact seed 2 state transition.

## Non-goals

- Automatically execute every group in a recovery generation.
- Add rollback after robot side effects.
- Add a separate recovery-abandon action.
- Change semantic grouping, public APIs, visual feedback, task configuration,
  or Capsule step accounting.
- Repair the placement geometry generated in the seed 2 program.

## Considered approaches

### Prompt-only guidance

The prompt could tell the model to continue pending recovery groups. This is a
small change, but scripted actions and stale or non-compliant model responses
could still append again or execute unrelated groups. It does not provide an
authoritative state transition.

### Automatic recovery execution

The runtime could enqueue and execute all recovery groups without asking the
model again. This prevents recovery thrashing but changes the meaning of
LLM-step control and removes the model's ability to inspect or patch between
physical actions.

### Strict recovery transaction

The selected approach makes pending recovery state authoritative. Prompt
availability and runtime guards both restrict actions to the latest recovery
until it is complete. This fixes the state-machine bug without changing the
LLM-step execution model.

## Recovery lifecycle

The latest recovery generation is complete only when both conditions hold:

1. Its required fresh observation has been successfully traced.
2. Its `authorized_group_keys` set is empty.

While the generation is incomplete:

- `append_recovery` is unavailable.
- `run_group` may target only dependency-runnable, unexecuted groups belonging
  to the latest recovery generation.
- `patch_group` may target only unexecuted groups belonging to that generation,
  including a blocked group that must be repaired before it can run.
- `inspect_variables` remains available.
- `finish` remains available and is still subject to the existing task-success
  guard.

Once the generation is complete, normal dependency-runnable groups and
`append_recovery` become available again.

The recovery completion decision no longer depends on whether the global trace
revision is newer than the append revision. Trace revision remains diagnostic
evidence, not a transaction-completion signal.

## Runtime enforcement

The recovery action state will expose separate sets for:

- Pending recovery group IDs: all unexecuted groups in the latest generation
  that still belong to its observation or authorized side-effect plan.
- Runnable recovery group IDs: pending group IDs whose source dependencies are
  currently satisfied and whose observation ordering permits execution.

The runtime will reject:

- `append_recovery` while the latest generation is incomplete.
- `run_group` outside `runnable_recovery_group_ids` while recovery is pending.
- `patch_group` outside the latest pending recovery group IDs while recovery is
  pending.

These guards run before source execution or source-edit preparation, so an
invalid action cannot change the namespace, trace, source revision, or recovery
lineage.

## Prompt behavior

When recovery is pending, the prompt will:

- Remove `append_recovery` from allowed actions and examples.
- Pass only `runnable_recovery_group_ids` as runnable execution choices.
- Identify the pending recovery groups that may be patched.
- Instruct the model to run or patch the existing recovery transaction rather
  than append a replacement.

The normal advice to prefer `append_recovery` after prior side effects will be
shown only when no recovery transaction is pending.

## Error handling

An out-of-transaction run or patch produces an invalid event with a stable
`safety_failure` and evidence containing the allowed recovery group IDs. A
blocked append retains a stable edit-rejection reason but its message describes
pending recovery work rather than missing trace evidence.

If a recovery group fails before producing a side-effect trace, it remains
pending and may be patched. Existing no-rollback lineage rules continue to
govern failures after physical side effects.

## Testing strategy

Focused tests will prove that:

- Running only the first group does not unlock another append when later
  recovery groups remain authorized.
- An unrelated, dependency-runnable group cannot be executed while recovery is
  pending.
- A patch outside the pending recovery generation is rejected, while a patch
  inside it is accepted.
- Append becomes available only after the observation requirement and all
  authorized recovery groups are completed.
- Prompt allowed actions and runnable IDs match the runtime guards.
- Existing observation ordering, lineage reconciliation, replay prevention,
  source repair, and successful completion behavior remain intact.

Runtime tests will be executed in the prepared WSL checkout after synchronizing
the changed Windows files, as required by the repository guidelines.
