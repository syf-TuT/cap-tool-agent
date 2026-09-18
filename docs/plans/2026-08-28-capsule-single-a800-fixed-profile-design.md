# Capsule Single-A800 Fixed Profile Design

## Problem

The owned-services launcher currently treats five progressively more memory-conservative VeRL
profiles as an OOM fallback ladder. On the target A800 host, only the final
`fsdp_base_bf16_vllm_util_045` profile is known to run. Retaining the four earlier profiles wastes
startup time, preserves obsolete configuration branches, and allows old profile identities in
new audit artifacts.

## Goals

- Run every Single-A800 owned-services attempt with
  `fsdp_base_bf16_vllm_util_045` from the start.
- Remove the first four profile transformations and all compatibility paths that accept them.
- Keep the existing fixed-profile provenance, controller-seed retry, gate ordering, cleanup, and
  final-audit guarantees.
- Verify the fixed profile on the designated SeeTaCloud host.

## Non-goals

- Do not rewrite archived experiment reports that describe earlier failed attempts.
- Do not change the 10240-token cap, 7+1 group structure, KL/GRPO semantics, controller contract,
  hardware admission thresholds, or Gate 1--7 success criteria.
- Do not introduce a replacement fallback profile.

## Chosen Approach

Replace the ladder contract with one fixed profile identity. The canonical VeRL YAML will contain
the final profile's settings directly: fixed actor/rollout/reference microbatches of one, BF16
actor/reference FSDP model dtype, and vLLM GPU memory utilization 0.45. The workflow YAML will
declare a scalar `oom_profile` instead of an `oom_ladder` list.

The launcher will validate that scalar, materialize only the canonical fixed profile, and execute
the existing controller-seed attempts against it. An identified GPU OOM will terminate the run
instead of selecting another profile. Guided randomness may still consume up to three new run IDs
because that is a retry of sampling under the same immutable profile, not a configuration fallback.

The common and Gate 7 artifact validators will accept only
`fsdp_base_bf16_vllm_util_045`. Artifacts naming any removed profile will fail closed. Current tests
and documentation will describe the fixed-profile contract; archived dated result summaries will
remain unchanged as historical evidence.

## Alternatives Considered

1. Keep a one-element `OOM_LADDER`. This minimizes code churn but retains misleading fallback
   abstractions and index-based profile transformation code.
2. Keep a generic list-valued configuration with only the final profile. This makes future profile
   additions easier but does not satisfy the requested removal of compatibility behavior.

## Error Handling

- A workflow whose `oom_profile` differs from the fixed identity fails configuration validation.
- A resolved profile whose settings or `capsule_runtime.oom_profile` differ from the fixed contract
  fails before gates run.
- Any GPU OOM is recorded using the existing failure evidence and then propagated immediately.
- Final audit rejects initial-audit or resolved-profile evidence containing a removed identity.

## Testing and Remote Verification

- First change focused tests to require the scalar fixed-profile contract and direct final settings;
  run them against the old code to observe the expected failures.
- Add or update rejection coverage for removed profile identities and for OOM-without-fallback.
- Implement the minimal launcher, validator, YAML, and documentation changes; run focused and broad
  Capsule tests in the prepared WSL environment.
- Sync the exact changed runtime files to `/root/autodl-tmp/cap-x` on the user-designated SeeTaCloud
  host, run focused tests and a launcher dry-run, then run the applicable real fixed-profile gate
  workflow when required credentials and immutable inputs are available.

## Acceptance Criteria

- No production path constructs, accepts, or advances through any of the first four profiles.
- A normal or dry-run attempt starts with `fsdp_base_bf16_vllm_util_045` and records that identity.
- GPU OOM does not create a second profile attempt.
- Existing same-profile Controller seed retries and audit provenance remain intact.
- Focused local/remote tests and the selected remote runtime verification complete successfully.
