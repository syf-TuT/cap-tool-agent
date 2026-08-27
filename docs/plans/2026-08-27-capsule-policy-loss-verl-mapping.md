# Capsule Policy-Loss VeRL Mapping Compatibility Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the Capsule policy loss read nested `capsule_gamma` from VeRL's Mapping-based `FSDPActorConfig`, then rerun the immutable Cube Lift LoRA workflow.

**Architecture:** Keep the compatibility behavior inside the project's `_config_get()` helper. Treat `KeyError` and VeRL's nonstandard missing-key `AttributeError` equivalently for Mapping lookup, preserving the existing nested `policy_loss` fallback and pinned VeRL source unchanged.

**Tech Stack:** Python 3.12, PyTorch, pytest, VeRL v0.6.1, Ray, vLLM, Robosuite, LoRA/PEFT.

---

### Task 1: Reproduce and fix VeRL Mapping lookup

**Files:**
- Modify: `tests/test_capsule_policy_loss.py`
- Modify: `capx/rl/capsule/policy_loss.py:163-181`

**Step 1: Write the failing test**

Add a minimal `Mapping` whose absent `__getitem__` lookup raises `AttributeError`, matching
VeRL `BaseConfig`, and verify the public wrapper reads nested gamma:

```python
class VerlStyleActorConfig(Mapping[str, object]):
    def __init__(self) -> None:
        self.clip_ratio = 0.2
        self.clip_ratio_low = None
        self.clip_ratio_high = None
        self.clip_ratio_c = 3.0
        self.policy_loss = {"capsule_gamma": 0.2}

    def __getitem__(self, key: str) -> object:
        return getattr(self, key)

    def __iter__(self):
        return iter(("clip_ratio", "clip_ratio_low", "clip_ratio_high", "clip_ratio_c", "policy_loss"))

    def __len__(self) -> int:
        return 5
```

Call `verl_capsule_critique_policy_loss()` with a guided token and compare it to
`capsule_critique_policy_loss(..., capsule_gamma=0.2)`.

**Step 2: Run the focused test and verify RED**

Sync the test into the prepared WSL runtime and run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cp /mnt/f/code/cap-x/tests/test_capsule_policy_loss.py /home/capx/code/cap-x/tests/test_capsule_policy_loss.py; cd /home/capx/code/cap-x; uv run --no-sync pytest -p no:cacheprovider tests/test_capsule_policy_loss.py::test_verl_wrapper_reads_nested_gamma_from_verl_style_mapping -q'
```

Expected: FAIL with `AttributeError: ... capsule_gamma` from `_config_get()`.

**Step 3: Implement the minimal compatibility fix**

Replace Mapping membership probing with one guarded lookup:

```python
if isinstance(config, Mapping):
    try:
        return config[key]
    except (KeyError, AttributeError):
        pass
```

Leave non-Mapping getter/attribute behavior and defaults unchanged.

**Step 4: Run focused and module tests and verify GREEN**

Sync both modified files into WSL, then run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cp /mnt/f/code/cap-x/capx/rl/capsule/policy_loss.py /home/capx/code/cap-x/capx/rl/capsule/policy_loss.py; cp /mnt/f/code/cap-x/tests/test_capsule_policy_loss.py /home/capx/code/cap-x/tests/test_capsule_policy_loss.py; cd /home/capx/code/cap-x; uv run --no-sync pytest -p no:cacheprovider tests/test_capsule_policy_loss.py -q'
```

Expected: all policy-loss tests PASS.

**Step 5: Commit**

```bash
git add capx/rl/capsule/policy_loss.py tests/test_capsule_policy_loss.py docs/plans/2026-08-27-capsule-policy-loss-verl-mapping.md
git commit -m "Fix VeRL Capsule policy loss config lookup"
```

### Task 2: Verify and rerun Cube Lift LoRA remotely

**Files:**
- Create remotely: `artifacts/cube_lift_capsule_rl_prepare_seed5_20260827_r02/`
- Create remotely: new immutable `artifacts/capsule_single_a800/<run-id>/`

**Step 1: Upload and apply only the new commits**

Generate patches after `d00be63`, upload them to the approved SeeTaCloud host, validate them in
a detached worktree, and apply them with `git am --ignore-space-change` because the remote import
stores CRLF blobs.

**Step 2: Run remote regression tests**

Run the focused policy-loss module and the existing Cube Lift/Capsule suite. Expected: all PASS.

**Step 3: Prepare a new immutable seed-5 bundle**

Run `scripts.capsule_rl.prepare_dataset_config` with output
`artifacts/cube_lift_capsule_rl_prepare_seed5_20260827_r02`. Do not modify or reuse r01.

**Step 4: Run dry-run, then Gate 1--7**

Use the existing owned-service workflow/profile and r02 config. Keep Controller
`qwen3.7-plus`, `stream=false`, `enable_thinking=false`, and set
`CAPX_FORCE_STREAMING_CHAT_COMPLETIONS=0`. Credentials remain only in the live shell environment.

**Step 5: Verify final evidence**

Require Gate 6 to report exactly one optimizer step, a finite nonzero gradient norm, LoRA-only
trainable parameters, a real checkpoint and adapter. Require adapter reload smoke and Gate 7
`runtime_verified: true`; otherwise preserve the failure artifacts and report the exact blocker.
