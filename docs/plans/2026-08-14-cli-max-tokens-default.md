# CLI Max Tokens Default Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Set the default completion budget for all CLI experiment entry points to 4,096 tokens without changing explicit overrides or Web API defaults.

**Architecture:** Keep the existing dataclass-based configuration flow and change only its CLI defaults. Add a focused regression test that imports the real argument dataclasses, and retain YAML/CLI override behavior unchanged. Align the two defensive Capsule-action fallbacks so they cannot silently reintroduce the old value when an argument object lacks `max_tokens`.

**Tech Stack:** Python 3.12, dataclasses, pytest, Tyro CLI, Git, SeeTaCloud Linux runtime.

---

### Task 1: Add a failing CLI-default regression test

**Files:**
- Create: `tests/test_cli_max_tokens_defaults.py`

**Step 1: Write the failing test**

```python
from capx.envs.launch import LaunchArgs
from capx.envs.scripts.run_batch import BatchLaunchArgs
from capx.envs.scripts.run_libero_batch import LiberoBatchLaunchArgs


def test_cli_experiment_entry_points_default_to_4096_tokens() -> None:
    assert LaunchArgs(config_path="config.yaml").max_tokens == 4096
    assert BatchLaunchArgs().max_tokens == 4096
    assert LiberoBatchLaunchArgs().max_tokens == 4096
```

**Step 2: Run the test against commit `8b92e5a`**

Run on SeeTaCloud after copying only the test to `/tmp`:

```bash
cd /root/autodl-tmp/cap-x-libero-capsule-llm-step-8b92e5a
source .venv-libero/bin/activate
python -m pytest /tmp/codex_test_cli_max_tokens_defaults.py -q
```

Expected: FAIL because each default is currently `2048`.

### Task 2: Change the CLI defaults and fallbacks

**Files:**
- Modify: `capx/envs/launch.py`
- Modify: `capx/envs/trial.py`
- Modify: `capx/envs/scripts/run_libero_batch.py`
- Modify: `capx/envs/scripts/run_batch.py`

**Step 1: Implement the minimal change**

Replace the five CLI/fallback values `2048` with `4096`. Do not change
`capx/web/models.py`, `capx/web/server.py`, or explicit YAML/CLI override handling.

**Step 2: Run the focused test**

```bash
python -m pytest tests/test_cli_max_tokens_defaults.py -q
```

Expected: `1 passed`.

**Step 3: Run nearby regression tests**

```bash
python -m pytest tests/test_run_libero_batch.py tests/test_runtime_control_trial_loop.py -q
```

Expected: all selected tests pass.

**Step 4: Commit**

```bash
git add docs/plans/2026-08-14-cli-max-tokens-default.md \
  tests/test_cli_max_tokens_defaults.py \
  capx/envs/launch.py capx/envs/trial.py \
  capx/envs/scripts/run_libero_batch.py capx/envs/scripts/run_batch.py
git commit -m "Reduce CLI completion token defaults"
```

### Task 3: Synchronize and verify the SeeTaCloud experiment

**Files:**
- Runtime output only under `.codex_remote_runs/` and `outputs/`.

**Step 1: Fast-forward the remote worktree to the committed local branch**

Transfer a Git bundle containing commits after `8b92e5a`, fetch it remotely, and use a
fast-forward merge. Confirm tracked status is clean before running.

**Step 2: Verify the effective default**

Run `--help` or instantiate `LaunchArgs` in the remote environment and confirm
`max_tokens == 4096` without a CLI override.

**Step 3: Run one trial without `--max-tokens`**

Use the requested LIBERO YAML, PackyAPI DeepSeek model, offline model caches, and Molmo
8122. Store results in a unique output directory.

**Step 4: Report evidence**

Report reward, task completion, sandbox rc, launcher rc, error details, output directory,
and confirmation that the recorded LLM request used the new 4,096-token default.
