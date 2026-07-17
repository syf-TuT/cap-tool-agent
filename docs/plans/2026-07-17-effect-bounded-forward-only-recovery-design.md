# Effect-Bounded Forward-Only Recovery Design

## Problem

The current Capsule runtime has a strong mechanism for physical side effects:
robot-effect regions and groups that have already executed cannot be replayed,
and recovery code must call a fresh-state observation function before continuing.
This should be the central contribution rather than an incidental safety check.

The implementation still spends too much of the normal execution path on LLM
control. Each micro-step reconstructs a prompt and asks the model to choose the
next action, even when the generated program can simply proceed. The prompt also
includes full region/group metadata, recent history, and a growing trace summary,
which makes `inspect_trace` redundant and grows context with every action.

## Contribution Name

Use the conservative contribution name:

**Effect-bounded forward-only recovery**

This avoids overclaiming that the current grouping is semantic. The execution
units are bounded by sensing/effect structure and source dependencies, not by
explicit symbolic goals or task-local pre/postconditions.

## Core Claim

For robot programs with irreversible physical side effects, recovery should be
forward-only:

- A side-effect execution unit can be executed at most once.
- Previously executed robot side effects cannot be rolled back or replayed from
  their original preconditions.
- Repairs must continue from the current physical state.
- Appended recovery code must call a fresh-state observation function such as
  `get_observation()` before issuing new side effects.

## In Scope

1. Add an automatic normal execution path that runs effect-bounded execution
   units sequentially without one LLM call per unit.
2. Keep the existing LLM-controlled step mode as a compatibility and ablation
   path.
3. Rename user-facing and paper-facing language from "semantic group" to
   "effect-bounded execution unit" unless explicit semantic metadata is added
   later.
4. Make recovery prompts compact and local: failed unit, failed event, fresh
   observation guidance, recent bounded trace, and allowed repair actions.
5. Change trace summaries from full event replay to bounded summaries.
6. Standardize experiment reporting so retry behavior is visible.

## Out of Scope

Task-local predicate evaluators are intentionally deferred. Local progress should
not be claimed as reliable task semantics in this version. The runtime may keep
coarse task completion and optional reward-drop guards, but reward increase is
not treated as a local postcondition.

## Runtime Behavior

The recommended runtime mode is `auto_forward`:

1. Generate the initial program using the existing initial-code prompt.
2. Segment the program into effect-bounded execution units.
3. Execute units in source order.
4. Record a side-effect ledger for every successful robot-side-effect unit.
5. If a unit succeeds, continue to the next unit without asking the LLM.
6. If a unit fails, becomes invalid, hits the no-replay guard, or hits an
   enabled coarse guard, call the LLM once for a recovery action.
7. Prefer `append_recovery` for recovery after executed side effects.
8. Allow one recovery side-effect execution after successful `append_recovery`,
   preserving the existing forward-only budget behavior.

The existing stepwise LLM mode remains available as `llm_step` for old runs and
ablation studies.

## Prompt Policy

Normal execution does not build a capsule action prompt.

Recovery prompt input should be bounded:

- task text
- current unit metadata and source
- failure event and feedback
- side-effect ledger
- last N trace events or failed trace events
- fresh-state recovery rule
- allowed actions

The prompt should not include all groups, all regions, full history, or the full
trace by default.

## Trace Policy

`RuntimeTrace.summary()` should no longer return the full event list. It should
return scalar and bounded data such as:

- total event count
- primitive call counts
- last N events
- failed events
- last side-effect calls

`inspect_trace` can remain, but it should become a selective query action rather
than a duplicate of prompt context.

## Experiment Reporting

Every reported run should separate algorithm performance from retry/provider
effects:

- first-attempt success
- success by retry budget
- total LLM logical calls and attempts
- total LLM wall-clock time
- total trial wall-clock time
- robot primitive executions
- provider failure count
- algorithm failure count
- timeout or budget-exhaustion count

For aggregate files that select the latest run per seed, the report must also
state the total number of attempts used to obtain the selected seed outcomes.
