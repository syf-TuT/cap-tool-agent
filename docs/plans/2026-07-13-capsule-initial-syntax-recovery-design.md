# Capsule Initial Syntax Recovery Design

## Problem

Runtime-control Capsule parses the initial generated Python source before it
enters the Capsule action loop. A `SyntaxError` from that parse prevents region
and group construction, so the model never receives a `capsule_action` request
and cannot use `patch_group` to repair the source.

Patch actions have a related failure mode: replacement source is accepted
before the complete patched program is parsed. If the replacement is invalid,
the subsequent regrouping parse escapes the Capsule loop and fails the trial.

## Chosen design

Reuse the existing `patch_group` protocol. When initial parsing fails, represent
the complete source as one temporary `region_1` and one temporary `group_1`.
Seed runtime history with a step-zero parse-failure diagnostic containing the
syntax error location and an explicit instruction to replace all of `group_1`.
The first real Capsule action can then patch the source and consumes the normal
Capsule step budget.

Validate the complete candidate program for both `patch_region` and
`patch_group` before publishing the replacement in a successful event. An
invalid candidate returns an `invalid` event with structured `SyntaxError`
evidence, retains the previous source, and lets the next Capsule action retry.

No new runtime action is added and valid initial programs keep the existing
execution path.

## Data flow

1. Generate and extract initial Python source.
2. Attempt normal AST segmentation and semantic grouping.
3. On `SyntaxError`, create whole-source fallback units and add an initial parse
   diagnostic to runtime history.
4. Build the normal Capsule prompt; it exposes temporary `group_1` and the
   diagnostic.
5. Accept a `patch_group` action, construct the full candidate source, and
   validate it with `ast.parse`.
6. If valid, publish the candidate and perform normal segmentation/grouping.
   If invalid, return feedback and retain the previous source.

## Error handling

- Initial syntax errors are recoverable and do not execute any source.
- Invalid region or group patches produce `invalid` runtime events.
- Syntax-error evidence includes exception type, line, offset, and source text.
- Other segmentation failures retain existing exception behavior.
- Exhausting the Capsule step budget with invalid source remains an ordinary
  unsuccessful trial and preserves Capsule artifacts for diagnosis.

## Tests

- An invalid initial program can be replaced through `patch_group`, regrouped,
  executed, and finished.
- An invalid patch produces feedback without ending the trial; a later valid
  patch succeeds.
- Existing runtime-control trial-loop tests remain green.

