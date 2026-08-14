# Capsule Single-Effect Group Normalization Design

## Problem

The semantic group normalizer can place consecutive robot side-effect regions in the same
group, while the program contract rejects any group containing more than one robot side
effect. In LIBERO LLM-step mode this makes every candidate execution group invalid, so the
model repeatedly patches source without ever executing a robot action.

Historical Robosuite LLM-step runs do not resolve this contradiction: successful traces were
allowed to execute groups containing multiple robot side effects. The current Robosuite
default avoids most patch loops through `auto_forward`, which executes groups deterministically
and only asks the LLM for local recovery after an execution failure. That is a separate control
policy and should not be mixed into the grouping fix.

## Scope

This change establishes the smallest executable loop:

- every normalized semantic group contains at most one declared robot side effect;
- consecutive side-effect regions become consecutive groups;
- setup and observation regions may remain attached to the following side effect;
- group-count bounding never merges two effectful groups;
- the configured maximum group count becomes a soft limit when meeting it would violate the
  single-effect invariant.

This change does not modify the trial loop, prompts, token budgets, YAML configuration, recovery
policy, or patch acceptance policy.

## Normalization Rules

While scanning regions in source order, the normalizer starts a new group before adding a
side-effect region whenever the current group already contains a side effect. This preserves
the original source partition and execution order.

When reducing the number of groups, two adjacent groups may be merged only if the result still
contains at most one side effect. If no safe merge exists, normalization returns more than the
requested maximum rather than constructing a group that the contract must reject.

For a LIBERO pick-and-place program, a sequence such as open, approach, descend, close, transfer,
release therefore produces one executable effect per group in the same source order.

## Verification

Focused tests will cover:

1. consecutive side effects are split into separate groups;
2. safe non-effect regions still attach to effect groups;
3. group-count pressure cannot merge two effectful groups;
4. a LIBERO-like pick-and-place program has no `multiple_effects_in_group` violation;
5. normalized groups remain a lossless, ordered partition of the original source.

After local unit tests pass in the prepared WSL environment, the exact one-trial LIBERO command
will be rerun remotely. The first acceptance criterion is observable execution (`run_group` or
equivalent robot side-effect execution), not merely successful patching. Reward, completion,
sandbox return code, errors, and the output directory will be reported.

## Deferred Strategy Gate

No patch-degradation guard or separate LLM-step patch/execution budget is added in this change.
If the remote trial still consumes its action budget through consecutive successful patches
without execution, a follow-up change will adapt the Robosuite `auto_forward` recovery boundary:
execution remains the forward path, and LLM patching is restricted to a failed local unit. That
follow-up will be tested and reviewed independently so its effect is distinguishable from the
normalizer correction.
