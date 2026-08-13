# Non-Privileged Capsule Strict Subset Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace denylist-based protection with an always-on strict Python subset for non-privileged Capsule execution.

**Architecture:** Add a deterministic allowlist validator to the existing runtime-control analysis, restrict non-privileged Capsule builtins, and apply the combined analysis in both `llm_step` and `auto_forward` before execution and after rewrites. Preserve traditional execution and privileged legacy Capsule behavior.

**Tech Stack:** Python AST, immutable builtin mappings, existing Capsule contract/details analysis, pytest, Ruff, WSL2.

---

## Task 1: Implement the Strict-Subset Analyzer

**Files:**

- Modify: `capx/runtime_control/contract.py`
- Modify: `capx/runtime_control/__init__.py`
- Modify: `tests/test_runtime_control_contract.py`

**Step 1: Write failing tests**

Add tests proving strict analysis rejects imports, classes, lambdas, callable aliases,
attribute/subscript calls, unknown calls, decorators, dunder/private attributes, and
conditional rebinding. Add positive tests for direct public API calls, allowed builtins,
pure helpers, ordinary data attribute reads, and pure control flow.

**Step 2: Run red tests in WSL**

```bash
uv run --no-sync pytest tests/test_runtime_control_contract.py -k strict_subset -q
```

Expected: failures because no strict-subset API exists.

**Step 3: Implement the analyzer**

Expose a narrow API such as:

```python
def analyze_capsule_strict_subset(
    source: str,
    regions: list[CodeRegion],
    groups: list[CodeRegionGroup],
    *,
    public_api_calls: set[str],
    side_effect_calls: set[str],
    safe_builtin_calls: set[str],
) -> list[ProgramContractViolation]:
    ...
```

Parse once, bind spans to region/group IDs, emit deterministic `strict_subset_violation`
items, and reuse the existing helper purity/effect summary rather than adding another
call graph. Calls are valid only when `Call.func` is a direct allowed `Name`.

**Step 4: Run analyzer regressions**

```bash
uv run --no-sync pytest \
  tests/test_runtime_control_contract.py \
  tests/test_runtime_control_segmenter.py \
  tests/test_runtime_control_normalizer.py -q
```

**Step 5: Commit**

```bash
git add capx/runtime_control/contract.py capx/runtime_control/__init__.py tests/test_runtime_control_contract.py
git commit -m "Define non-privileged Capsule Python subset"
```

## Task 2: Restrict Non-Privileged Execution Globals

**Files:**

- Modify: `capx/envs/tasks/base.py`
- Modify: `capx/envs/trial.py`
- Modify: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write failing tests**

Verify non-privileged Capsule globals have an immutable safe `__builtins__` mapping and
lack import, eval, exec, globals, locals, getattr, vars, dir, open, and frame access.
Verify direct public API functions remain traced. Verify privileged legacy and traditional
execution retain their prior globals.

**Step 2: Run red tests**

```bash
uv run --no-sync pytest tests/test_runtime_control_trial_loop.py -k strict_builtins -q
```

**Step 3: Implement restricted globals**

Define one reviewed safe builtin mapping and insert it explicitly as `__builtins__` when
`include_internal_handles=False`. Do not rely on `exec` to populate default builtins.
Keep the mapping immutable and keep raw API/environment objects out of public globals.

**Step 4: Run loop regressions**

```bash
uv run --no-sync pytest tests/test_runtime_control_trial_loop.py tests/test_runtime_control_side_effects.py -q
```

**Step 5: Commit**

```bash
git add capx/envs/tasks/base.py capx/envs/trial.py tests/test_runtime_control_trial_loop.py
git commit -m "Restrict non-privileged Capsule globals"
```

## Task 3: Enforce Strict Analysis in Both Capsule Modes

**Files:**

- Modify: `capx/envs/trial.py`
- Modify: `capx/runtime_control/prompts.py`
- Modify: `tests/test_runtime_control_trial_loop.py`
- Modify: `tests/test_runtime_control_prompts.py`

**Step 1: Write failing execution tests**

Use real fake APIs to show that conditional rebinding, `sys._getframe`, imports,
callable attributes, aliases, lambdas, and classes are rejected before execution with
zero API calls even when `capsule_validate_program_contract` is false on a
non-privileged environment. Verify a patch to compliant direct code then succeeds.
Add an auto-forward test proving unsafe initial source never executes.

**Step 2: Run red tests**

```bash
uv run --no-sync pytest tests/test_runtime_control_trial_loop.py -k strict_nonprivileged -q
```

**Step 3: Integrate combined analysis**

Always run strict analysis when `env.cfg.privileged` is false. Merge its violations with
the existing Capsule-ready analysis, recompute after patch/append, send violations to the
repair prompt, and guard unsafe units. Validate auto-forward before its first group and
after recovery rewrites.

**Step 4: Update prompt constraints**

State the direct-call, no-import, no-reflection, and no-callable-alias rules compactly in
the non-optional execution constraints.

**Step 5: Run green regressions**

```bash
uv run --no-sync pytest \
  tests/test_runtime_control_contract.py \
  tests/test_runtime_control_prompts.py \
  tests/test_runtime_control_trial_loop.py \
  tests/test_runtime_control_side_effects.py \
  tests/test_runtime_control_segmenter.py \
  tests/test_runtime_control_normalizer.py -q
```

**Step 6: Commit**

```bash
git add capx/envs/trial.py capx/runtime_control/prompts.py tests/test_runtime_control_trial_loop.py tests/test_runtime_control_prompts.py
git commit -m "Enforce strict non-privileged Capsule execution"
```

## Task 4: Update the LIBERO Protocol and Verify

**Files:**

- Modify: `env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml`
- Modify: `docs/libero-tasks.md`
- Modify: `tests/test_runtime_control_config.py`

**Step 1: Add failing protocol assertions**

Assert the prompt explicitly forbids imports, dynamic calls, callable aliases, and
attribute calls, while directing the model to use public API functions and pure helpers.

**Step 2: Update YAML and documentation**

Document that non-privileged Capsule is an enforced strict Python subset and that legacy
arbitrary Python is not available in this mode.

**Step 3: Run the full focused suite and Ruff**

Use the existing Task 10 exact pytest command. Run cached Ruff 0.15.1 offline over every
changed Python file, including `capx/runtime_control/__init__.py` and
`capx/envs/tasks/base.py`.

**Step 4: Run leak and diff checks**

Confirm no saved prompt test contains simulator truth or embedded base64, then run
`git diff --check`.

**Step 5: Commit**

```bash
git add env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml docs/libero-tasks.md tests/test_runtime_control_config.py
git commit -m "Document strict LIBERO Capsule execution"
```
