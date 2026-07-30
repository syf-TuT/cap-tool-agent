# Cube Restack Seed Reproducibility Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make cube-restack initial cube placement deterministic for a supplied seed.

**Architecture:** Centralize Robosuite RNG synchronization in a protected base-environment
helper, including recursive placement-sampler propagation. Have cube-restack pass the seed
at construction and invoke the helper before each seeded reset.

**Tech Stack:** Python 3.12, NumPy Generator, Robosuite/MuJoCo, pytest, Ruff, WSL2 Ubuntu.

---

**Execution note:** The prepared WSL checkout is on a different Git commit from the Windows
source worktree. To avoid mixed-version tests, run its Python environment against the mounted
isolated worktree with `PYTHONPATH` set to that worktree, rather than copying individual source
files into `/home/capx/code/cap-x`.

### Task 1: Lock down base reseeding behavior

**Files:**

- Modify: `tests/test_robosuite_observation_privilege.py`
- Modify: `capx/envs/simulators/robosuite_base.py`

**Step 1: Write the failing test**

Add fake placement samplers containing both mapping-backed and sequence-backed nested
samplers. Construct `RobosuiteBaseEnv`, attach a fake Robosuite environment, call
`_reseed_robosuite(17)`, and assert:

- wrapper `_rng`, environment `rng`, and every sampler `rng` are the same object;
- environment `seed` is `17`; and
- the generator produces the sequence expected from `np.random.default_rng(17)`.

**Step 2: Run the test to verify it fails**

From WSL, set `PYTHONPATH` to the mounted worktree and run:

```bash
cd /mnt/f/code/cap-x/.worktrees/cube-restack-seed-reproducibility
PYTHONPATH="$PWD" /home/capx/code/cap-x/.venv/bin/python -m pytest \
  tests/test_robosuite_observation_privilege.py \
  -k reseed_robosuite -q
```

Expected: FAIL because `_reseed_robosuite` does not exist.

**Step 3: Write the minimal implementation**

Add `_reseed_robosuite(seed)` and a small recursive sampler traversal helper to
`robosuite_base.py`. Support mapping values and non-string iterables, and guard against
cycles by object identity.

**Step 4: Run the test to verify it passes**

Rerun the command from Step 2 against the mounted worktree.

Expected: PASS.

### Task 2: Propagate constructor and reset seeds in cube-restack

**Files:**

- Modify: `tests/test_robosuite_observation_privilege.py`
- Modify: `capx/envs/simulators/robosuite_cubes_restack.py`

**Step 1: Write the failing constructor test**

Add a parameterized test over all `privileged` / `enable_render` branches. Intercept the
Robosuite `Stack` constructor and assert that a sentinel seed reaches its `seed` keyword.

**Step 2: Write the failing reset delegation test**

Create a cube-restack instance without invoking its constructor, attach a fake Robosuite
environment, stub observation-dependent methods, call `reset(seed=23)`, and verify reseeding
occurs before the underlying environment reset.

**Step 3: Run the focused tests to verify they fail**

```bash
uv run --no-sync pytest tests/test_robosuite_observation_privilege.py \
  -k "cube_restack and seed" -q
```

Expected: FAIL because constructor branches omit `seed` and reset updates only `_rng`.

**Step 4: Write the minimal implementation**

Pass `seed=seed` to all three `Stack` constructor calls. Replace the direct wrapper RNG
assignment in `reset()` with `self._reseed_robosuite(seed)`.

**Step 5: Run the focused tests to verify they pass**

Rerun the command from Step 3.

Expected: PASS.

### Task 3: Verify real simulator reproducibility

**Files:**

- Create: `tests/integrations/test_robosuite_cube_restack_seed.py`

**Step 1: Write the integration regression test**

Create a privileged, rendering-disabled cube-restack environment. Reset it with seed 5,
capture the primary and secondary cube poses, advance with a reset using another seed, then
reset again with seed 5. Assert the two seed-5 poses match within tight floating-point
tolerance and the alternate seed changes at least one cube position.

**Step 2: Run the integration test**

```bash
MUJOCO_GL=egl uv run --no-sync pytest \
  tests/integrations/test_robosuite_cube_restack_seed.py -q
```

Expected: PASS.

### Task 4: Regression and quality checks

**Files:**

- Review: all changed source and test files

**Step 1: Run focused unit regression**

```bash
uv run --no-sync pytest tests/test_robosuite_observation_privilege.py -q
```

Expected: PASS.

**Step 2: Run lint**

```bash
uv run --no-sync ruff check \
  capx/envs/simulators/robosuite_base.py \
  capx/envs/simulators/robosuite_cubes_restack.py \
  tests/test_robosuite_observation_privilege.py \
  tests/integrations/test_robosuite_cube_restack_seed.py
```

Expected: PASS.

**Step 3: Inspect the diff**

Confirm that only seed propagation, tests, and plan documentation changed. Confirm no
experiment configuration, reward logic, model settings, or result artifacts changed.

**Step 4: Commit**

```bash
git add capx/envs/simulators/robosuite_base.py \
  capx/envs/simulators/robosuite_cubes_restack.py \
  tests/test_robosuite_observation_privilege.py \
  tests/integrations/test_robosuite_cube_restack_seed.py \
  docs/plans/2026-07-30-cube-restack-seed-reproducibility.md
git commit -m "Fix cube restack seed reproducibility"
```
