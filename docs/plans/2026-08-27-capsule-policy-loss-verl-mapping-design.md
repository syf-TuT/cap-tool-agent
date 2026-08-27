# Capsule Policy-Loss VeRL Mapping Compatibility Design

## Problem

The Cube Lift single-A800 LoRA run passed Gates 1--5 and failed in Gate 6 before the
optimizer step. VeRL passes an `FSDPActorConfig` object to the registered Capsule policy loss.
That object implements `collections.abc.Mapping`, but its `__getitem__` raises
`AttributeError` rather than `KeyError` for an absent field. The current `_config_get()` uses
`key in config`; the inherited Mapping containment check therefore propagates
`AttributeError` for the absent top-level `capsule_gamma` field instead of falling back to
`config.policy_loss.capsule_gamma`.

## Selected approach

Keep the compatibility boundary in `capx/rl/capsule/policy_loss.py`. For Mapping inputs,
perform one direct item lookup and treat both `KeyError` and `AttributeError` as a missing key.
Then preserve the existing nested-policy lookup and default behavior unchanged.

This is preferred over adding `capsule_gamma` to `FSDPActorConfig`, which violates VeRL's
structured config, and over modifying the pinned VeRL checkout, which would break its immutable
SHA provenance.

## Test design

Add a focused regression test in `tests/test_capsule_policy_loss.py` with a minimal
VeRL-style Mapping whose missing `__getitem__` lookup raises `AttributeError` and whose
`policy_loss` entry contains `capsule_gamma=0.2`. The existing public
`verl_capsule_critique_policy_loss()` wrapper must read the nested value and match the direct
Capsule loss result. The test must fail on the current implementation with the same
`AttributeError` observed remotely before production code changes.

## Validation and rerun

Run the focused policy-loss test, the complete policy-loss module, and the remote Capsule/Cube
Lift regression set. Commit the fix, upload only the new commits, and prepare a new immutable
seed-5 r02 bundle so the launcher creates a distinct run ID. Preserve the failed r01 Gate 1--6
artifacts without modification. A successful rerun must produce exactly one optimizer step,
LoRA-only trainable evidence, a checkpoint and adapter reload smoke before Gate 7 can claim
`runtime_verified: true`.
