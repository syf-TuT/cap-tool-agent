# Capsule LLM-Step-Only Runtime Design

## Status

Approved on 2026-08-14.

This design supersedes the active runtime recommendations in:

- `2026-07-17-effect-bounded-forward-only-recovery-design.md`;
- `2026-07-21-auto-forward-default-design.md`;
- the auto-forward compatibility requirements in
  `2026-08-01-llm-step-compact-context-design.md`;
- the pending-recovery execution behavior described in
  `2026-08-13-libero-object-capsule-llm-step-design.md`.

Those files remain historical design records. The runtime implementation described here
has one Capsule control policy: strict LLM-step execution.

## Motivation

The current Capsule runtime has two control policies:

- `auto_forward`, which runs effect-bounded groups in source order and asks an LLM only
  for recovery; and
- `llm_step`, which normally asks an Action LLM to select one runtime action per step.

The two paths duplicate execution, recovery, source-edit, safety, and telemetry logic.
They have also diverged semantically. In particular, the current `llm_step` path switches
temporarily to auto-forward behavior after `append_recovery`: it fills
`pending_recovery_actions` and executes every appended group without another LLM
decision.

The LIBERO task-0 audit exposed two related failures:

1. Four consecutive `inspect_trace` actions returned byte-identical evidence. Compact
   prompt history omitted the inspection evidence, so no new decision evidence reached
   the Action LLM.
2. Reanalysis after repeated append operations remapped the executed side-effect ledger
   through temporary group IDs and duplicate source text. Newly appended groups were
   incorrectly marked executed and rejected as `side_effect_replay`.

The desired policy is unambiguous: every runtime action is selected explicitly by the
Action LLM, each selected action executes at most one runtime unit, and the host records
one automatic post-action observation before the next decision.

## Goals

- Remove the `auto_forward` implementation completely.
- Remove `capsule_control_mode` as an algorithm-selection option.
- Keep one Capsule runtime loop with strict per-action LLM decisions.
- Ensure `append_recovery` only edits source; it must never trigger hidden group
  execution.
- Record one automatic post-group observation after every attempted group execution.
- Remove `inspect_trace` as an LLM-selectable action because the host already supplies
  the new trace evidence.
- Give source units stable internal identities that survive display-ID renumbering.
- Make source patch and append operations transactional.
- Preserve forward-only no-replay safety and fresh-state recovery.
- Preserve non-privileged strict-subset and program-contract enforcement.

## Non-Goals

- Do not retain compatibility execution for `auto_forward`.
- Do not unify Capsule with the traditional non-Capsule execution path.
- Do not expose privileged simulator object state to an Action LLM.
- Do not add a separate VLM call after every group.
- Do not claim local task progress from LIBERO's sparse reward.
- Do not guarantee task reward success; this design guarantees scheduling, evidence, and
  ledger correctness.

## Single Runtime Policy

`_run_capsule_trial()` dispatches directly to the sole Capsule loop. The implementation
should rename `_run_capsule_llm_step_loop()` to `_run_capsule_loop()` because no second
mode remains.

The loop follows this state machine:

```text
DECIDE
  -> VALIDATE
  -> EXECUTE AT MOST ONE ACTION
  -> AUTOMATIC POST-ACTION OBSERVATION
  -> RECORD HISTORY / TRACE / METRICS
  -> TERMINATE OR DECIDE AGAIN
```

The invariants are:

- one logical Action LLM decision selects at most one runtime action;
- one `run_group` executes at most one effect-bounded group;
- every attempted group execution produces exactly one automatic observation;
- `append_recovery` commits source but performs no robot manipulation;
- the host never selects or queues a later group on behalf of the LLM;
- success is checked before any later LLM call.

For example:

```text
step 6: Action LLM selects append_recovery
        host validates and commits candidate source; robot does not move

step 7: Action LLM sees the new groups and selects one run_group
        host executes that group and records one observation

step 8: Action LLM sees the observation and selects the next action
```

There is no `pending_recovery_actions` queue and no `forced_recovery_action` origin.

## Auto-Forward Removal

Delete the following runtime behavior:

- `_run_capsule_auto_forward_loop()`;
- `capsule_control_mode` dispatch and default selection;
- automatic group-index advancement;
- failure-only and terminal-only auto-forward recovery prompts;
- automatic execution of appended recovery code;
- insertion of recovery source after an old group followed by automatic continuation.

Delete helpers that become unused, including:

- `_validate_recovery_action()`;
- `_coerce_terminal_append_recovery_action()`;
- `_terminal_python_payload()`;
- `_group_index_by_id()`;
- `_group_index_for_region()`;
- `_first_group_index_starting_after_line()`;
- `_insert_recovery_source_after_line()`;
- `build_capsule_recovery_prompt()`;
- `build_capsule_terminal_recovery_prompt()`.

Simplify `_execute_runtime_action()` so `append_recovery` always appends at the end of
the current source. Remove `append_recovery_insert_after_line`.

## Configuration Migration

Remove `capsule_control_mode` from merged launch configuration and repository YAML files.
All Capsule trials use the sole strict stepwise loop.

For a short migration period, configuration loading should reject a supplied legacy
`capsule_control_mode` field with an explicit message:

```text
capsule_control_mode has been removed. Capsule now always uses strict per-action LLM
control. Remove this configuration field.
```

The validation is a migration error only; it must not retain an auto-forward execution
path. Silently ignoring an old `auto_forward` field is unsafe because it changes the
experiment algorithm without notice.

## Stable Unit Lineage

Temporary display IDs such as `group_11` are unsuitable as safety identities because
reanalyzing edited source can renumber them. Duplicate source text is also unsuitable:
two separately appended `open_gripper()` calls are distinct physical actions.

Maintain stable internal identity outside generated Python source:

```python
@dataclass
class UnitLineage:
    next_region_key: int
    next_group_key: int
    region_key_by_id: dict[str, str]
    group_key_by_id: dict[str, str]
    executed_region_keys: set[str]
    executed_group_keys: set[str]
```

`region_id` and `group_id` remain the model-facing action identifiers. Guards translate
them to stable keys before checking the executed ledger. Trace and metric artifacts
record both the display ID and stable key.

Newly created units always receive new monotonic keys, even when their text exactly
matches an older unit.

## Source Revisions

Record source provenance separately from source text:

```python
@dataclass
class SourceRevision:
    revision: int
    source_sha256: str
    edit_kind: Literal["initial", "patch_region", "patch_group", "append_recovery"]
    parent_revision: int | None
    old_line_count: int
```

Revision metadata supports audit, append-boundary validation, lineage reconciliation,
and stale-analysis detection. It is not inserted into generated Python code.

## Transactional Source Editing

Patch and append actions operate on a candidate state:

```text
apply edit to candidate source
  -> analyze candidate regions, groups, strict subset, and contract
  -> reconcile stable lineage
  -> validate edit invariants
  -> atomically commit all candidate structures
```

A prepared edit contains:

```python
@dataclass
class PreparedSourceEdit:
    source: str
    analysis: CapsuleSourceAnalysis
    revision: SourceRevision
    lineage: UnitLineage
```

Only a successful commit replaces the live source, regions, groups, maps, violations,
lineage, and revision. Any failure leaves every live structure unchanged.

Candidate edit failure reasons include:

- `candidate_syntax_error`;
- `strict_subset_violation`;
- `program_contract_violation`;
- `lineage_ambiguous`;
- `append_boundary_crossed`;
- `executed_unit_edit_attempt`.

The old fail-closed behavior that marks all current units executed when any lineage is
unresolved must be removed. An ambiguous candidate edit is rejected atomically instead.

## Lineage Reconciliation

### Append

Append leaves the old source prefix unchanged. Therefore:

- an old unit can map only to a candidate unit ending at or before the old line count;
- old start line, end line, and source must match exactly;
- no global duplicate-source fallback is allowed;
- every unit wholly inside appended lines receives a new key;
- a group crossing the old/new boundary invalidates the candidate append;
- an unresolved or ambiguous executed unit invalidates the candidate append.

New append units must have no key overlap with the remapped executed ledger.

### Patch

- Units before the edit retain keys only through exact span and source matching.
- Units after the edit retain keys only through exact line-delta-adjusted span and source
  matching.
- Unexecuted units intersecting the patch span receive new keys.
- Guards reject patches to executed side-effect units before candidate construction.
- Ambiguous reconciliation rejects the candidate patch atomically.

## Recovery Generations

Replace the scalar `recovery_side_effect_budget` with authorization bound to stable unit
keys:

```python
@dataclass
class RecoveryGeneration:
    generation_id: str
    source_revision: int
    start_line: int
    end_line: int
    observation_functions: tuple[str, ...]
    authorized_group_keys: set[str]
    executed_group_keys: set[str]
```

Successful append creates a generation and authorizes its new side-effect group keys.
Inspection and other decisions do not consume authorization. Executing an authorized
group removes its key from the authorized set and adds it to both the generation and
global executed ledgers.

Patching an unexecuted recovery group reanalyzes the complete generation, revalidates its
fresh-state observation requirement, assigns new keys to edited units, and atomically
replaces the generation's authorization set.

To prevent repeated no-information append chains, another append is rejected until a
group execution or physical trace event has occurred after the preceding append. The
error is `no_new_physical_state_since_last_append`. The LLM can execute or patch the
existing recovery source instead.

## Automatic Post-Action Observation

Every attempted group execution records trace position and public state before execution:

```python
trace_start = executor.trace.event_count
state_before = snapshot()
```

After execution, the host captures:

```python
state_after = snapshot()
new_trace_events = executor.trace.events[trace_start:]
```

The latest observation has a stable prompt view:

```python
@dataclass
class PostActionObservation:
    step_id: int
    action: str
    unit_id: str | None
    unit_key: str | None
    event_status: str
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    reward_before: float | None
    reward_after: float | None
    task_completed: bool
    new_trace_events: list[dict[str, Any]]
    trace_revision: int
```

`latest_post_action_observation` is a dedicated prompt section. It is not stored only in
the bounded history window and must survive every compact-prompt fallback phase.

In `sparse_terminal` progress mode, reward `0 -> 0` remains successful execution while
the observation states `terminal_progress_unverified=true`.

## Inspection Actions

Remove `inspect_trace` from:

- runtime action schema;
- normal and recovery prompt action lists;
- prompt examples;
- `_execute_runtime_action()`;
- tests.

The automatic post-action observation already provides trace events for the selected
group. Keeping `inspect_trace` duplicates prompt evidence and permits no-information
loops.

Retain `inspect_variables`. Prevent an equivalent variable-inspection loop by caching the
tuple of source, trace, and namespace revisions plus the normalized requested name set.
Repeating the same inspection without a revision change returns
`no_new_variable_state`.

## Runtime Error Semantics

### Invalid model action

- Record an invalid action and feedback.
- Execute no source.
- Change no ledger or revision.
- Consume one logical decision step.
- Continue if budget remains.

### Group exception

If trace evidence shows a robot side effect occurred, record the stable key as executed
even if the group event failed later. It cannot be replayed. If failure occurred before
any side effect, the unit remains patchable subject to existing guards.

### Premature finish

When task success is required, `finish` before `task_completed` or reward 1 is rejected
without ending the loop. A successful group terminates the trial before another model
query.

## Step Budget

`max_capsule_steps` counts logical Action LLM decisions. Each decision triggers at most
one runtime action.

The following consume one step:

- `run_group` and `run_region`;
- `patch_group` and `patch_region`;
- `append_recovery`;
- `inspect_variables`;
- `finish`;
- invalid model output or a rejected action.

Automatic observations consume no step. Provider retries increase attempt metrics but do
not create additional logical decisions.

Budget exhaustion before task success produces `sandbox_rc=1` and
`loop_exit_reason=budget_exhausted`.

## Telemetry

Each step metric should record at least:

- decision ID and action origin (`llm` or `scripted`);
- source revision before and after;
- target display ID and stable unit key;
- trace revision before and after;
- new trace event count;
- whether the post-action observation was recorded;
- whether a robot side effect executed;
- whether a source edit committed and, if not, its rejection reason;
- lineage reconciliation status;
- budget-exhaustion status.

Remove pending-recovery and auto-forward telemetry, including:

- `action_origin=pending_recovery`;
- `forced_recovery_action`;
- pending recovery queue length;
- auto-forward group indexes and recovery prompt counters.

Trial summaries should include:

- logical LLM decisions and provider attempts;
- attempted group executions and post-group observations;
- source edits and append operations;
- blocked replays and duplicate variable inspections;
- budget exhaustion.

The number of post-group observations must equal the number of attempted group
executions.

## Test Strategy

### Removal and migration

- `_run_capsule_auto_forward_loop` no longer exists.
- `_run_capsule_trial` directly calls the sole loop.
- launch config does not emit `capsule_control_mode`.
- repository YAML files do not contain the field.
- a legacy field produces the migration error.
- auto-forward-only prompt builders and helpers no longer exist.

### Strict stepwise scheduling

Use a scripted or mocked model sequence containing append, one group, then another group.
Assert three separate logical decisions, no robot primitive during append, one group per
later decision, and no pending-recovery action origin.

### Automatic observation

- Every attempted group emits exactly one observation.
- The observation contains only the new trace events from that group.
- The next prompt contains the latest observation.
- History rollover and compact fallback do not remove it.
- Task success prevents another model call.
- `inspect_trace` is absent from schema, prompt, and executor.

### Append and lineage

- Identical source in separate appends receives distinct stable keys.
- Each new key can execute once; a second execution of the same key is rejected.
- Append never maps a new group into the old executed ledger.
- Display-ID renumbering preserves old stable keys.
- Ambiguous lineage leaves source and ledger unchanged.
- A group crossing the append boundary rejects the candidate.
- A second append without new physical evidence is rejected.
- Group execution or physical failure permits later forward recovery.

### Patch and authorization

- Unaffected units retain stable keys.
- Edited, unexecuted units receive new keys.
- Executed side-effect units cannot be patched.
- Recovery patches recompute authorization and fresh-observation validity.
- Failed patches leave every live source structure unchanged.

### Existing safety behavior

Port applicable auto-forward tests to the sole loop instead of deleting their safety
assertions. Preserve coverage for strict subset, program contract, fresh observation,
reward-drop guard, premature finish, sparse reward, success short-circuit, visual flags,
diagnostic ground-truth isolation, and sticky safety failure.

Delete only tests whose asserted policy was specifically auto-forward, such as no normal
Action LLM calls, automatic source-order execution, failure-only recovery prompts, and
automatic terminal recovery.

## Documentation

Update active paper and experiment documentation to remove auto-forward as the recommended
method. Historical plan documents remain unchanged for provenance. This document states
which historical recommendations it supersedes.

## Verification

Run focused and complete runtime-control tests in the prepared WSL environment, not the
Windows checkout. After unit acceptance, repeat the SeeTaCloud LIBERO task-0 trial with
the same model and visual settings used in the audit.

The remote trace must show:

- one new Action LLM decision before every `run_group`;
- one automatic observation after every attempted group;
- no pending-recovery actions;
- no `inspect_trace` actions;
- no false `side_effect_replay` for distinct appended units;
- no partial commit after a failed source edit.

Reward success remains an experiment outcome. Scheduler, observation, and ledger
correctness are acceptance requirements independent of reward.
