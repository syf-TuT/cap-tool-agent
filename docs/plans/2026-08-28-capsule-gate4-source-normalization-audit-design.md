# Capsule Gate4 Source-Normalization Audit Design

## Problem

Clean replay now executes the same canonical source as ordinary evaluation while preserving the
raw Program response in `ProgramReplayResultV1`. A fenced Program response therefore records
`source_normalized=true` and can fail for the enclosed program's real semantic error instead of
the outer Markdown fence.

Gate4 still carries the previous contract: every fenced P0 must remain a first-line
`SyntaxError` with zero reward. The first real post-fix Gate4 run reached artifact verification
and failed on that stale assertion even though the evaluator preserved the raw source and
executed the normalized source as designed.

## Goals

- Make the Gate4 verifier accept fenced semantic P0 failures produced by canonical clean replay.
- Require auditable proof that raw source identity was preserved and the expected canonical
  source was executed.
- Preserve the existing explicit fence/suffix repair-unit lineage contract.
- Keep the change narrow enough to rerun the existing Gate1--7 workflow without changing the
  Controller, repair schema, or training objective.

## Non-goals

- Do not remove fence repair units or migrate repair traces to canonical source.
- Do not accept successful, infrastructure-error, or evaluator-error P0 results.
- Do not relax raw source/hash provenance or explicit protocol deletion ordering.
- Do not change ordinary evaluation, clean replay, rollout token identity, or reward semantics.

## Chosen Approach

When a repair trace contains the supported outer fence units, Gate4 will continue to rederive
the exact raw protocol units and require one explicit deletion for each protocol target. Instead
of requiring a fence-caused `SyntaxError`, it will require the replay diagnostics to prove:

- `source_normalized` is exactly `true`;
- `raw_source_sha256` equals the immutable raw P0 source hash;
- `executed_source_sha256` equals the hash of `normalize_program_source(raw_source)`.

The enclosing collector verifier already requires every P0 to be a typed semantic clean-replay
failure with binary reward zero, so the specialized fence verifier does not duplicate the
outcome check.

## Alternatives Considered

1. Remove explicit fence repair units and repair only canonical source. This is cleaner long
   term but requires coordinated changes to Controller prompts, repair traces, Gate4, Gate5,
   documentation, and historical assumptions.
2. Disable normalization for Gate4. This restores the old audit but reintroduces the train/eval
   mismatch that caused fenced valid programs to receive false zero rewards.

## Error Handling

Missing or false normalization flags, malformed hash fields, raw-hash mismatches, and
executed-hash mismatches remain hard Gate4 failures. Unfenced P0 behavior is unchanged because
the specialized protocol branch is entered only when supported fence units are present.

## Testing

- Update the fenced-P0 fixture to represent a canonicalized semantic failure and include both
  source hashes.
- Observe the existing explicit-deletion acceptance test fail under the stale SyntaxError rule.
- Add focused rejection tests for a missing normalization flag and a mismatched executed hash.
- Run the focused Gate4 verifier tests, broader Capsule script tests, and the existing remote
  focused test suite before rerunning Gate1--7.

## Acceptance Criteria

- A fenced, normalized, binary-zero semantic P0 passes the protocol portion of Gate4 when all
  explicit protocol deletions and hashes are correct.
- Gate4 rejects forged or missing source-normalization evidence.
- Existing whole-program cleanup rejection remains intact.
- The real Gate1--7 workflow advances beyond Gate4 on the fixed commit.
