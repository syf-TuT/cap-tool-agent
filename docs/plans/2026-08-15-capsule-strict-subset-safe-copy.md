# Capsule Strict-Subset Local Names and Safe Copy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Permit underscore-prefixed local names and zero-argument `.copy()` in non-privileged Capsule source without relaxing private capability access or any repair and robot-execution safety rule.

**Architecture:** Make two narrow AST-validator exceptions in `capx/runtime_control/contract.py`: local name/parameter validation no longer treats an underscore prefix as private capability access, and `visit_Call` recognizes only a zero-argument method named `copy` while recursively validating its receiver. Update strict-subset prompt wording to match. Keep repair fingerprints, quarantine, budgets, and all other method calls unchanged.

**Tech Stack:** Python 3.12, `ast`, pytest, Ruff, WSL2 Ubuntu 22.04.

---

### Task 1: Allow underscore-prefixed local variables and parameters

**Files:**
- Modify: `tests/test_runtime_control_contract.py:232-253`
- Modify: `capx/runtime_control/contract.py:1001-1157`

**Step 1: Write the failing contract tests**

Remove `_hidden = 1` from `test_strict_subset_rejects_private_and_sensitive_runtime_access` and add:

```python
def test_strict_subset_allows_underscore_prefixed_local_names_and_parameters():
    source = """\
def select(_value):
    return _value
position, _ = detect_object("bowl")
_soup = select(position)
"""

    assert _strict_analyze(source) == []


@pytest.mark.parametrize(
    "source",
    [
        "value = pose._position\n",
        "value = pose.__class__\n",
        "def _helper(value):\n    return value\n_helper(1)\n",
        "__builtins__ = {}\n",
    ],
)
def test_strict_subset_keeps_private_capabilities_and_helpers_closed(source):
    assert _strict_analyze(source)
```

The first test demonstrates the newly allowed local forms. The second protects private
attributes, private helper names, and sensitive runtime identifiers.

**Step 2: Sync the changed test to WSL**

From elevated PowerShell/Codex execution context:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'cp /mnt/f/code/cap-x/tests/test_runtime_control_contract.py /home/capx/code/cap-x/tests/test_runtime_control_contract.py'
```

**Step 3: Run the focused tests and verify RED**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; .venv/bin/python -m pytest tests/test_runtime_control_contract.py -k "underscore_prefixed_local or private_capabilities" -q'
```

Expected: the allowed-local test fails because `_value`, `_`, and `_soup` still produce
`strict_subset_violation`. The negative cases pass.

**Step 4: Implement the minimal local-identifier distinction**

Change the visitor helper to accept an explicit local-name allowance:

```python
def _identifier_violation(
    self,
    name: str,
    *,
    allow_private_local: bool = False,
) -> str | None:
    if name in _STRICT_CAPSULE_SENSITIVE_NAMES:
        return f"sensitive runtime name '{name}' is not available"
    if name.startswith("_") and not allow_private_local:
        return f"private name '{name}' is not available"
    return None
```

Use `allow_private_local=True` only for:

- `visit_Name`, covering local loads, stores, loop targets, and unpacking targets;
- function parameter validation in `visit_FunctionDef`;
- exception binding validation, which is another local binding even though the strict language
  currently rejects `try` statements.

Keep the default for function names, attribute names, direct call targets, and keyword names.
Existing protected-callable alias and rebinding checks remain in place after identifier
validation.

**Step 5: Sync the implementation and verify GREEN**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'cp /mnt/f/code/cap-x/capx/runtime_control/contract.py /home/capx/code/cap-x/capx/runtime_control/contract.py; cd /home/capx/code/cap-x; .venv/bin/python -m pytest tests/test_runtime_control_contract.py -k "underscore_prefixed_local or private_capabilities or private_and_sensitive" -q'
```

Expected: all selected tests pass.

**Step 6: Commit**

```bash
git add capx/runtime_control/contract.py tests/test_runtime_control_contract.py
git commit -m "Allow Capsule underscore local names"
```

### Task 2: Allow only zero-argument `.copy()` calls

**Files:**
- Modify: `tests/test_runtime_control_contract.py:186-204,232-253`
- Modify: `tests/test_runtime_control_trial_loop.py:2165-2258`
- Modify: `capx/runtime_control/contract.py:82-133,1090-1138`

**Step 1: Write the failing positive tests**

Add:

```python
@pytest.mark.parametrize(
    "expression",
    [
        "position.copy()",
        'observation["rgb"].copy()',
        "detect_object(\"bowl\")[0].copy()",
    ],
)
def test_strict_subset_allows_zero_argument_data_copy(expression):
    assert _strict_analyze(f"copied = {expression}\n") == []
```

**Step 2: Write the negative safety tests**

Add:

```python
@pytest.mark.parametrize(
    "source",
    [
        "value.copy(1)\n",
        'value.copy(order="K")\n',
        "value.tolist()\n",
        "value._copy()\n",
        "value._private.copy()\n",
        "__builtins__.copy()\n",
    ],
)
def test_strict_subset_rejects_non_whitelisted_attribute_calls(source):
    assert _strict_analyze(source)
```

Add a non-privileged repair regression using the same `_`/`.copy()` source shape as seed 4:

```python
def test_nonprivileged_repair_accepts_underscore_locals_and_copy(tmp_path):
    env = FakeSuccessfulNonPrivilegedCapsuleEnv()
    initial_source = (
        "def execute_task():\n"
        "    values = [1, 2, 3]\n"
        "    _soup = values.copy()\n"
        "    close_gripper()\n"
        "execute_task()\n"
    )
    repaired_source = (
        "values = [1, 2, 3]\n"
        "_soup = values.copy()\n"
        "close_gripper()\n"
    )

    trial_module._run_capsule_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 2,
            "capsule_validate_program_contract": True,
        },
        initial_code=initial_source,
        scripted_actions=[
            {
                "action": "patch_group",
                "args": {"group_id": "group_1", "source": repaired_source},
            },
            {"action": "run_group", "args": {"group_id": "group_1"}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_00.json").read_text())
    metrics = _capsule_step_metrics(tmp_path / "capsule_step_metrics_trial_00.jsonl")

    assert [entry["event"]["status"] for entry in trace] == ["success", "success"]
    assert trace[0]["event"]["evidence"]["source_revision_after"] == 1
    assert env.api.calls == ["close_gripper"]
    assert all(row["strict_subset_valid"] is True for row in metrics)
```

Also assert that the first generated action prompt's contract violations do not contain
`private name` or the generic attribute-call rejection, while retaining the helper/program
contract violations that quarantine the initial source.

**Step 3: Sync the tests and verify RED**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'cp /mnt/f/code/cap-x/tests/test_runtime_control_contract.py /home/capx/code/cap-x/tests/test_runtime_control_contract.py; cp /mnt/f/code/cap-x/tests/test_runtime_control_trial_loop.py /home/capx/code/cap-x/tests/test_runtime_control_trial_loop.py; cd /home/capx/code/cap-x; .venv/bin/python -m pytest tests/test_runtime_control_contract.py tests/test_runtime_control_trial_loop.py -k "data_copy or non_whitelisted_attribute or underscore_locals_and_copy" -q'
```

Expected: the three positive contract cases and the trial-loop regression fail because
`.copy()` still hits the current direct-call restriction. Negative cases continue to pass.

**Step 4: Implement the exact safe-call predicate**

Add a narrow helper near the strict-subset constants:

```python
def _is_zero_argument_copy_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "copy"
        and not node.args
        and not node.keywords
    )
```

Handle it before the existing direct-name call branch:

```python
def visit_Call(self, node: ast.Call) -> None:
    if _is_zero_argument_copy_call(node):
        self.visit(node.func.value)
        return
    if not isinstance(node.func, ast.Name):
        ...
```

Recursive receiver validation is mandatory. Do not visit `node.func` itself for the allowed
form because that would reclassify the approved method call, and do not generalize this to a
set of arbitrary methods.

**Step 5: Sync the implementation and verify GREEN**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'cp /mnt/f/code/cap-x/capx/runtime_control/contract.py /home/capx/code/cap-x/capx/runtime_control/contract.py; cd /home/capx/code/cap-x; .venv/bin/python -m pytest tests/test_runtime_control_contract.py tests/test_runtime_control_trial_loop.py -k "data_copy or non_whitelisted_attribute or non_direct_or_unknown or underscore_locals_and_copy" -q'
```

Expected: all selected tests pass.

**Step 6: Commit**

```bash
git add capx/runtime_control/contract.py tests/test_runtime_control_contract.py tests/test_runtime_control_trial_loop.py
git commit -m "Allow zero-argument Capsule copy calls"
```

### Task 3: Keep strict-subset prompts consistent

**Files:**
- Modify: `tests/test_runtime_control_prompts.py:1226-1250`
- Modify: `tests/test_runtime_control_config.py:112-121`
- Modify: `capx/runtime_control/prompts.py:55-60`
- Modify: `env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml:25-27`

**Step 1: Tighten the failing prompt assertions**

In both prompt tests, retain the assertion that attribute calls are generally restricted and
add an explicit exception assertion, for example:

```python
assert "zero-argument .copy()" in text
assert "other attribute calls" in text
```

For the YAML/config-derived prompt, assert the lower-cased equivalent.

**Step 2: Sync the tests and verify RED**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'cp /mnt/f/code/cap-x/tests/test_runtime_control_prompts.py /home/capx/code/cap-x/tests/test_runtime_control_prompts.py; cp /mnt/f/code/cap-x/tests/test_runtime_control_config.py /home/capx/code/cap-x/tests/test_runtime_control_config.py; cd /home/capx/code/cap-x; .venv/bin/python -m pytest tests/test_runtime_control_prompts.py::test_capsule_prompt_states_strict_python_subset_constraints tests/test_runtime_control_config.py -k "strict or libero" -q'
```

Expected: assertions fail because both prompt sources still forbid every attribute call.

**Step 3: Update prompt wording minimally**

Change the shared runtime-control constraint to say:

```text
Use no imports, classes, lambdas, try, while, async, dynamic or reflective calls,
callable aliases, or attribute calls other than zero-argument .copy().
```

Apply the same exception to the non-privileged LIBERO initial-code prompt. Do not modify any
other execution constraint.

**Step 4: Sync prompt sources and verify GREEN**

Copy the exact changed prompt/YAML files to their matching WSL paths, then rerun the command
from Step 2. Expected: all selected tests pass.

**Step 5: Commit**

```bash
git add capx/runtime_control/prompts.py env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml tests/test_runtime_control_prompts.py tests/test_runtime_control_config.py
git commit -m "Document Capsule copy allowance"
```

### Task 4: Run focused regressions, lint, and final diff review

**Files:**
- Verify all files modified in Tasks 1-3

**Step 1: Sync all exact changed files to WSL**

Copy only the changed Python, test, and exact prompt/config files from `/mnt/f/code/cap-x` to
their corresponding paths under `/home/capx/code/cap-x`.

**Step 2: Run the focused regression suite**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; .venv/bin/python -m pytest tests/test_runtime_control_contract.py tests/test_runtime_control_prompts.py tests/test_runtime_control_trial_loop.py tests/test_runtime_control_config.py -q'
```

Expected: all tests pass with no new warnings.

**Step 3: Run Ruff**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'cd /home/capx/code/cap-x; .venv/bin/python -m ruff check capx/runtime_control/contract.py capx/runtime_control/prompts.py tests/test_runtime_control_contract.py tests/test_runtime_control_prompts.py tests/test_runtime_control_trial_loop.py tests/test_runtime_control_config.py'
```

Expected: exit code 0.

**Step 4: Review scope and safety invariants**

```bash
git diff --check
git status --short
git diff --stat
git diff -- capx/runtime_control/contract.py capx/runtime_control/prompts.py tests/test_runtime_control_contract.py tests/test_runtime_control_prompts.py tests/test_runtime_control_trial_loop.py tests/test_runtime_control_config.py
```

Confirm that:

- repair fingerprint code is unchanged;
- no general attribute-call or private-attribute allowance was introduced;
- no budget, quarantine, lineage, or robot-side-effect logic changed;
- unrelated model-cache and experiment artifacts remain unstaged.

**Step 5: Commit any final test-only cleanup**

If final cleanup was needed:

```bash
git add <exact changed files only>
git commit -m "Test Capsule strict subset relaxation"
```

Do not create an empty commit.
