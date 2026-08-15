# Capsule Quarantined Repair Draft Design

## Problem

`_prepare_capsule_source_edit()` currently rejects a `patch_group` candidate whenever the
complete candidate program still contains any strict-subset or program-contract violation.
Because one runtime action can replace only one semantic group, an initial program with
violations in multiple groups cannot make progress: every local repair leaves another
violation elsewhere and is rejected.

The implementation conflates two separate guarantees:

- source edits must update source, analysis, groups, lineage, and revision atomically;
- only a completely validated program may execute.

Atomic editing does not require every intermediate repair draft to be executable.

## Chosen Design

Represent an AST-valid but contract-invalid program as a quarantined repair draft. Preserve
its normal semantic groups so the Action LLM can repair one group per decision. A successful
local edit atomically updates the draft and its derived analysis, but runtime execution stays
blocked until the complete draft passes strict-subset and program-contract validation.

The existing whole-source fallback remains reserved for source that cannot safely reach
normal segmentation, including syntax errors and strict preflight failures.

No new public runtime action is added. `patch_group` retains its existing schema.

## State and Transitions

The loop derives `repair_pending` from the current source analysis:

```text
repair_pending = syntax_error or strict_subset_violations or contract_violations
```

For normally segmented source, the loop supports these transitions:

```text
valid source --invalid patch--> reject; keep valid source
invalid draft --non-progressing patch--> reject; keep previous draft
invalid draft --improving patch--> atomically commit updated draft
invalid draft --fully valid patch--> atomically commit and leave repair mode
```

While `repair_pending` is true, all execution actions are rejected. Source repair actions
remain available. `append_recovery` is rejected because it is a physical-state recovery
mechanism rather than an initial source-repair mechanism.

## Partial-Repair Admission

A partial draft repair is allowed only when no robot side effect has been executed and the
candidate is a strict diagnostic improvement:

- candidate Python parses successfully;
- candidate analysis and lineage reconciliation succeed;
- the edit does not target an executed side-effect unit;
- the candidate introduces no new normalized violation;
- at least one previous normalized violation is removed.

Violation comparison uses a multiset fingerprint that excludes unstable line numbers and
temporary region/group ids, while retaining the violation code, normalized message,
helper name, and side-effect calls. A simple total-count comparison is insufficient because
it could trade several minor violations for a new, different violation.

If the current source is valid, the existing fail-closed rule remains: any candidate strict
or contract violation rejects the edit atomically.

## Execution Safety

Repair-draft commits never authorize execution. A dedicated repair-pending guard rejects
`run_group`, `run_region`, and `resume_from_region` before dispatch. This preserves the
safety invariant even though the draft source and its analysis advance across local repairs.

The accepted patch event reports that the source edit committed and includes the remaining
violation count, but remaining draft diagnostics are not recorded as an executed safety
failure. An attempted execution during repair mode remains an invalid safety event.

## Prompt Behavior

AST-valid invalid programs keep their normal semantic groups and updated violation list.
The prompt explains that execution is quarantined and instructs the Action LLM to continue
with `patch_group`. After the final violation is removed, the next prompt returns to normal
execution guidance.

Syntax-invalid and strict-preflight-invalid source continues to use the existing temporary
whole-source group because reliable semantic group boundaries are unavailable.

## Testing

Regression coverage must prove:

- two violations in different groups can be repaired by two sequential `patch_group`
  actions;
- the first accepted patch advances the draft revision but execution remains blocked and
  produces no API calls;
- the second patch clears the remaining violation and enables execution;
- a candidate that introduces a new violation is rejected without changing the draft;
- a valid current source cannot be degraded into an invalid draft;
- syntax-invalid candidates and executed-unit edits remain atomic rejections;
- prompt and metrics expose repair-pending state and remaining diagnostics.
