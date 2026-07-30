# Cube Restack Seed Reproducibility Design

## Problem

`FrankaRobosuiteCubesRestackLowLevel.reset(seed=...)` currently replaces only the
wrapper's `_rng`. Robosuite owns a separate `rng`, and the custom
`StackedObjectRandomSampler` keeps the Robosuite RNG object that existed when the
sampler was constructed. Consequently, resetting two trials with the same seed
does not guarantee the same initial cube placement.

This invalidates seed-to-seed comparisons with earlier experiments. In the
observed runs, seed 5 produced materially different cube positions despite an
unchanged task prompt.

## Scope

This change makes cube-restack environment initialization reproducible for a
given seed. It does not change:

- Contact-GraspNet sampling
- LLM model aliases, temperature, or response handling
- Capsule recovery behavior
- reward or task-success logic
- video capture

Those sources of variance can be addressed separately after environment
reproducibility is established.

## Design

Add a protected reseeding helper to `RobosuiteBaseEnv`. Given an integer seed,
the helper creates one NumPy generator and synchronizes it to:

1. the CaP-X wrapper's `_rng`;
2. the underlying Robosuite environment's `seed` and `rng`;
3. the environment's placement initializer; and
4. any nested placement samplers exposed through a `samplers` collection.

Using one generator preserves Robosuite's expected ordering of random draws
while preventing stale sampler references. Recursive sampler traversal supports
both the current custom sampler and Robosuite composite samplers.

`FrankaRobosuiteCubesRestackLowLevel` will:

- pass the constructor seed to every `Stack` construction path; and
- call the base reseeding helper before `robosuite_env.reset()` whenever
  `reset(seed=...)` receives a seed.

Calling `reset()` without a seed retains the existing behavior and continues
from the current generator state.

## Verification

Tests will be written before production changes:

- a focused unit test will use fake environment and nested sampler objects to
  prove that the same generator reaches every RNG owner;
- constructor tests will verify that all cube-restack `Stack` paths receive the
  requested seed;
- a WSL Robosuite integration test will reset with the same seed and compare the
  initial cube poses, then verify that a different seed changes the placement.

All simulator and pytest execution will occur in the prepared WSL checkout at
`/home/capx/code/cap-x`. Only source and documentation are edited in the Windows
worktree.

## Failure Handling

If a placement initializer has no RNG attribute or no nested samplers, the
helper skips that capability. Unexpected sampler collection shapes are not
silently interpreted; traversal is limited to mapping values and ordinary
iterables used by Robosuite sampler composites.

