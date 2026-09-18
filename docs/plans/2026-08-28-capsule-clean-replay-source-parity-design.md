# Capsule Clean-Replay Source Parity Design

## Problem

Ordinary CaP-X evaluation extracts Python from an outer ` ```python ` Markdown fence before
calling `environment.step()`. Capsule clean replay currently executes the raw Program response.
Most Qwen responses in the cube-lift training run were fenced, so clean replay classified valid
programs as first-line syntax errors even though the ordinary evaluator could execute them.

The same run scheduled twenty seed-local groups but published a checkpoint after two groups were
discarded by strict VeRL response validation. Only eighteen groups were persisted, with no attempt
to replace the two missing groups.

## Goals

- Execute exactly the same canonical program source in ordinary evaluation and clean replay.
- Preserve the raw Program response for rollout tokens, PPO identity, repair lineage, and hashing.
- Make source normalization explicit and auditable in clean-replay diagnostics.
- Require every scheduled seed to produce exactly one group before a checkpoint can be published.
- Retry a discarded group for the same seed at most three times, then fail the run without a
  checkpoint while preserving a complete discard audit.

## Non-goals

- Do not broaden Markdown parsing beyond the existing ` ```python\n...``` ` behavior.
- Do not normalize or rewrite the response token IDs used by VeRL.
- Do not relax decode/retokenize identity checks.
- Do not change binary reward semantics or the 7+1 group objective.

## Chosen Approach

Extract the existing fence-removal behavior into a small shared pure function. Keep
`launch_utils._extract_code()` as a compatibility wrapper and use the same function inside the
clean-replay evaluator. The evaluator passes canonical source to the replay backend while its
typed result continues to store the raw source.

This is preferred to normalizing only in `CandidateCleanReplayAdapter`, which would leave direct
clean-replay callers inconsistent, and to normalizing in the Robosuite environment, which would
put model-response parsing at the wrong abstraction boundary.

## Source Data Flow

1. VeRL returns a raw Program response and token IDs.
2. Candidate collection retains the raw response unchanged.
3. Clean replay derives `executed_source` with the shared normalization function.
4. The replay backend executes `executed_source`.
5. `ProgramReplayResultV1.source` and `source_sha256` continue to identify the raw response, so
   existing candidate provenance validation remains valid.
6. Replay diagnostics record `raw_source_sha256`, `executed_source_sha256`, and
   `source_normalized`.

Unfenced input is unchanged. Fenced input follows the existing evaluator behavior exactly,
including whitespace stripping and use of the final closing fence.

## Group Retry and Checkpoint Contract

Add the strict configuration value `capsule.max_group_attempts: 3`. For every scheduled task,
`CapsuleCritiqueRayTrainer.fit()` attempts assembly up to that limit:

- Every rejected attempt is recorded with its task identity, attempt index, reason, replay
  history, and partial repair evidence.
- A successful attempt produces one and only one training result for that seed.
- If all three attempts fail, training raises a typed exhaustion error.
- The server runtime writes the accumulated discard audit before surfacing the failure.
- Checkpoint publication remains after successful `fit()`, so exhaustion cannot publish a partial
  checkpoint.
- Before publication, the runtime asserts that the number of completed steps equals the number of
  scheduled groups.

Successful retries remain visible as discarded *attempts*, but do not reduce the completed group
count. Result metadata will distinguish discarded attempts from completed seed-local groups.

## Error Handling

Only `GroupDiscarded` triggers a same-seed retry. Programming defects and unexpected exceptions
still propagate immediately. The final exhaustion error includes the task identity and attempt
budget. Existing worker-level clean-replay retry behavior remains unchanged.

## Testing

Tests will be written before production changes and observed failing for the intended reason.
Coverage will include:

- exact legacy fence normalization and unchanged unfenced input;
- ordinary evaluation and clean replay sharing the same canonical source;
- raw source/hash preservation plus executed-source audit fields;
- two discarded attempts followed by one successful group;
- three discarded attempts producing a typed terminal failure;
- server-side discard-audit persistence and absence of checkpoint publication after exhaustion;
- a full-group-count guard that refuses partial checkpoint publication;
- focused unit regressions followed by the broader Capsule test suite and a remote clean-replay
  smoke test.

## Acceptance Criteria

- A fenced response that succeeds in ordinary evaluation receives the same clean-replay outcome.
- Clean replay never executes the outer supported Markdown fence.
- Raw response identity used for PPO and provenance is unchanged.
- Twenty scheduled groups cannot complete with fewer than twenty persisted training steps.
- A seed that cannot produce a valid group in three attempts fails the run with a complete audit
  and no newly published checkpoint.
