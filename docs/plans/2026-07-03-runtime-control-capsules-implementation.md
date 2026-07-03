# Runtime-Control Capsules Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove primitive toolification and add a runtime-control capsule mode where CaP-Agent0 writes normal Python code, while the runtime executes, inspects, checkpoints, and locally patches code regions.

**Architecture:** Delete the existing `capx.tools` primitive-tool path. Add `capx.runtime_control` for schema, segmentation, tracing, checkpointing, patching, prompts, and region execution. Add `agent_mode: capsule` beside the unchanged `agent_mode: code` baseline.

**Tech Stack:** Python dataclasses, `ast`, `copy`, existing `CodeExecutionEnvBase`, pytest, Ruff, WSL `uv run --no-sync`.

---

## Working Rules

Use the Windows checkout for source edits only. Run Python tests through WSL:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; <command>'
```

Before WSL tests, sync touched files from `/mnt/f/code/cap-x` into `/home/capx/code/cap-x`.

Do not run simulator rollouts, model servers, Robosuite launches, `uv sync`, or dependency installs from the Windows checkout.

---

### Task 1: Remove Primitive Toolification Surface

**Files:**
- Delete: `capx/tools/__init__.py`
- Delete: `capx/tools/schema.py`
- Delete: `capx/tools/state.py`
- Delete: `capx/tools/registry.py`
- Delete: `capx/tools/executor.py`
- Delete: `capx/tools/verifiers.py`
- Delete: `capx/tools/prompts.py`
- Delete: `capx/tools/planner.py`
- Delete: `capx/tools/franka_metadata.py`
- Delete: `tests/test_tool_*.py`
- Delete: `tests/test_franka_tool_metadata.py`
- Delete: `env_configs/cube_stack/franka_robosuite_cube_stack_tool_vdm.yaml`
- Delete: `env_configs/cube_stack/franka_robosuite_cube_stack_tool_state_first.yaml`
- Delete: `env_configs/cube_lifting/franka_robosuite_cube_lifting_tool_vdm.yaml`
- Delete: `env_configs/cube_restack/franka_robosuite_cube_restack_tool_vdm.yaml`
- Modify: `capx/envs/trial.py`
- Modify: `capx/envs/tasks/base.py`
- Modify: `capx/utils/launch_utils.py`

**Step 1: Remove imports and code paths**

In `capx/envs/trial.py`, remove:

```python
from capx.tools...
def _run_tool_trial(...):
...
if config.get("agent_mode", "code") == "tool":
    return _run_tool_trial(...)
```

In `capx/envs/tasks/base.py`, remove tool-state, tool-registry, and tool-executor initialization plus:

```python
tool_specs()
call_tool()
tool_state_summary()
snapshot_state()
```

In `capx/utils/launch_utils.py`, remove `tool_feedback_level`, `max_tool_steps`, and `scripted_tool_calls` config plumbing unless no longer present.

**Step 2: Verify imports**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync python -c "import capx.envs.trial; import capx.envs.tasks.base; print(\"imports ok\")"'
```

Expected: `imports ok`.

**Step 3: Commit**

```bash
git add capx/envs/trial.py capx/envs/tasks/base.py capx/utils/launch_utils.py capx/tools tests env_configs
git commit -m "Remove primitive toolification path"
```

---

### Task 2: Add Runtime-Control Schema

**Files:**
- Create: `capx/runtime_control/__init__.py`
- Create: `capx/runtime_control/schema.py`
- Create: `tests/test_runtime_control_schema.py`

**Step 1: Write tests**

```python
from capx.runtime_control.schema import CodeRegion, RuntimeAction, RuntimeEvent


def test_code_region_exports_source_span():
    region = CodeRegion(region_id="region_1", start_line=2, end_line=4, source="x = 1")

    assert region.to_dict()["region_id"] == "region_1"
    assert region.to_dict()["source_span"] == {"start_line": 2, "end_line": 4}


def test_runtime_action_validates_args_mapping():
    action = RuntimeAction.from_mapping({"action": "run_region", "args": {"region_id": "region_1"}})

    assert action.action == "run_region"
    assert action.args["region_id"] == "region_1"


def test_runtime_event_is_jsonable():
    event = RuntimeEvent(
        action="run_region",
        status="failed",
        region_id="region_2",
        message="boom",
        evidence={"exception_type": "ValueError"},
    )

    assert event.to_dict()["status"] == "failed"
```

**Step 2: Implement dataclasses**

Create `CodeRegion`, `RuntimeAction`, `RuntimeEvent`, and `RuntimeFeedback` with `to_dict()` methods. Supported action strings are `run_region`, `inspect_trace`, `inspect_variables`, `patch_region`, `rollback_to_checkpoint`, `resume_from_region`, and `finish`.

**Step 3: Run tests**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_schema.py -q'
```

Expected: PASS.

**Step 4: Commit**

```bash
git add capx/runtime_control tests/test_runtime_control_schema.py
git commit -m "Add runtime-control schema"
```

---

### Task 3: Add Python Code Segmenter

**Files:**
- Create: `capx/runtime_control/segmenter.py`
- Create: `tests/test_runtime_control_segmenter.py`

**Step 1: Write tests**

```python
from capx.runtime_control.segmenter import segment_python_code


def test_segmenter_groups_top_level_statements():
    source = "import numpy as np\nx = 1\ny = x + 2\nprint(y)\n"

    regions = segment_python_code(source)

    assert [r.region_id for r in regions] == ["region_1", "region_2", "region_3", "region_4"]
    assert regions[0].source == "import numpy as np"
    assert regions[-1].start_line == 4


def test_segmenter_keeps_compound_statement_together():
    source = "if True:\n    x = 1\n    y = 2\nprint(x)\n"

    regions = segment_python_code(source)

    assert len(regions) == 2
    assert "y = 2" in regions[0].source
```

**Step 2: Implement with `ast.parse`**

Use node `lineno` and `end_lineno` to slice source lines. If parsing fails, raise a `SyntaxError` with the original location.

**Step 3: Run tests**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_segmenter.py tests/test_runtime_control_schema.py -q'
```

Expected: PASS.

**Step 4: Commit**

```bash
git add capx/runtime_control/segmenter.py tests/test_runtime_control_segmenter.py
git commit -m "Segment generated code into runtime regions"
```

---

### Task 4: Add Primitive Call Trace Wrappers

**Files:**
- Create: `capx/runtime_control/trace.py`
- Create: `tests/test_runtime_control_trace.py`

**Step 1: Write tests**

```python
from capx.runtime_control.trace import RuntimeTrace, wrap_function_for_trace


def test_trace_wrapper_preserves_return_value():
    trace = RuntimeTrace()

    def add(x, y):
        return x + y

    wrapped = wrap_function_for_trace("add", add, trace)

    assert wrapped(2, 3) == 5
    assert trace.events[0]["name"] == "add"
    assert trace.events[0]["status"] == "success"


def test_trace_wrapper_records_exception():
    trace = RuntimeTrace()

    def explode():
        raise ValueError("bad")

    wrapped = wrap_function_for_trace("explode", explode, trace)

    try:
        wrapped()
    except ValueError:
        pass

    assert trace.events[0]["status"] == "failed"
```

**Step 2: Implement trace**

Summarize args and return values with type, shape, dtype, and short repr. Do not serialize large arrays into prompts.

**Step 3: Run tests**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_trace.py -q'
```

Expected: PASS.

**Step 4: Commit**

```bash
git add capx/runtime_control/trace.py tests/test_runtime_control_trace.py
git commit -m "Trace primitive calls inside code capsules"
```

---

### Task 5: Add Capsule Executor

**Files:**
- Create: `capx/runtime_control/executor.py`
- Create: `tests/test_runtime_control_executor.py`
- Modify: `capx/envs/tasks/base.py`

**Step 1: Write fake executor tests**

```python
from capx.runtime_control.executor import CapsuleExecutor
from capx.runtime_control.segmenter import segment_python_code


def test_executor_runs_regions_in_persistent_namespace():
    source = "x = 1\ny = x + 2\n"
    regions = segment_python_code(source)
    executor = CapsuleExecutor(base_globals={})

    first = executor.run_region(regions[0])
    second = executor.run_region(regions[1])

    assert first.status == "success"
    assert second.status == "success"
    assert executor.globals["y"] == 3


def test_executor_binds_feedback_to_failed_region():
    regions = segment_python_code("x = 1\nraise ValueError('bad')\n")
    executor = CapsuleExecutor(base_globals={})
    executor.run_region(regions[0])

    event = executor.run_region(regions[1])

    assert event.status == "failed"
    assert event.region_id == "region_2"
    assert event.evidence["exception_type"] == "ValueError"
```

**Step 2: Implement executor**

`CapsuleExecutor` should:

- accept a base global namespace
- execute one `CodeRegion`
- capture stdout and stderr
- preserve globals across regions
- return a `RuntimeEvent`
- support optional `RuntimeTrace` wrappers

**Step 3: Add base-env preparation hook**

In `CodeExecutionEnvBase`, add a method such as:

```python
def _build_capsule_globals(self) -> dict[str, Any]:
    ...
```

It should mirror `_init_exec_globals()` and bind existing API functions by their original names.

**Step 4: Run tests**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/bin:/usr/local/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_executor.py -q'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/runtime_control/executor.py capx/envs/tasks/base.py tests/test_runtime_control_executor.py
git commit -m "Execute generated code by runtime region"
```

---

### Task 6: Add Checkpoints And Local Patching

**Files:**
- Create: `capx/runtime_control/checkpoints.py`
- Create: `capx/runtime_control/patching.py`
- Create: `tests/test_runtime_control_checkpoints.py`
- Create: `tests/test_runtime_control_patching.py`

**Step 1: Write checkpoint tests**

```python
from capx.runtime_control.checkpoints import NamespaceCheckpointStore


def test_checkpoint_restores_copyable_globals():
    store = NamespaceCheckpointStore()
    namespace = {"x": 1, "__name__": "__main__"}
    checkpoint_id = store.save("before_region_2", namespace)
    namespace["x"] = 99

    restored = store.restore(checkpoint_id)

    assert restored["x"] == 1
```

**Step 2: Write patch tests**

```python
from capx.runtime_control.patching import replace_region_source
from capx.runtime_control.segmenter import segment_python_code


def test_replace_region_source_only_changes_target_region():
    source = "x = 1\ny = x + 2\nprint(y)\n"
    regions = segment_python_code(source)

    patched = replace_region_source(source, regions[1], "y = x + 3")

    assert "y = x + 3" in patched
    assert "print(y)" in patched
```

**Step 3: Implement MVP checkpointing**

Use `copy.deepcopy` where possible. Skip uncopyable values and record skipped variable names.

**Step 4: Implement source replacement**

Use `CodeRegion.start_line` and `CodeRegion.end_line` to replace only that source span.

**Step 5: Run tests**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_checkpoints.py tests/test_runtime_control_patching.py -q'
```

Expected: PASS.

**Step 6: Commit**

```bash
git add capx/runtime_control/checkpoints.py capx/runtime_control/patching.py tests/test_runtime_control_checkpoints.py tests/test_runtime_control_patching.py
git commit -m "Add capsule checkpoints and local patching"
```

---

### Task 7: Add Runtime-Control Prompts And Parser

**Files:**
- Create: `capx/runtime_control/prompts.py`
- Create: `tests/test_runtime_control_prompts.py`

**Step 1: Write tests**

```python
from capx.runtime_control.prompts import build_capsule_prompt, parse_runtime_action_response
from capx.runtime_control.schema import CodeRegion


def test_parse_runtime_action_response():
    action = parse_runtime_action_response('{"action": "run_region", "args": {"region_id": "region_1"}}')

    assert action.action == "run_region"


def test_capsule_prompt_excludes_robot_tool_list():
    prompt = build_capsule_prompt(
        task="stack cubes",
        regions=[CodeRegion(region_id="region_1", start_line=1, end_line=1, source="x = 1")],
        history=[],
        trace_summary={},
    )
    text = str(prompt)

    assert "run_region" in text
    assert "solve_ik" not in text
    assert "move_to_joints" not in text
```

**Step 2: Implement prompt**

The prompt should require one JSON object:

```json
{"action": "run_region", "args": {"region_id": "region_1"}}
```

It must explicitly say not to request robot primitives as tools and to patch only source regions.

**Step 3: Run tests**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_prompts.py -q'
```

Expected: PASS.

**Step 4: Commit**

```bash
git add capx/runtime_control/prompts.py tests/test_runtime_control_prompts.py
git commit -m "Prompt runtime-control capsule actions"
```

---

### Task 8: Add Capsule Trial Loop

**Files:**
- Modify: `capx/envs/trial.py`
- Create: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write scripted trial-loop test**

Use a fake env with a generated source program and scripted runtime actions:

```python
def test_capsule_trial_runs_scripted_regions(tmp_path):
    ...
    summary = _run_capsule_trial(
        env=fake_env,
        trial=1,
        args=SimpleNamespace(model="test"),
        config={"output_dir": str(tmp_path), "max_capsule_steps": 4},
        initial_code="x = 1\nRESULT = x + 1\n",
        scripted_actions=[
            {"action": "run_region", "args": {"region_id": "region_1"}},
            {"action": "run_region", "args": {"region_id": "region_2"}},
            {"action": "finish", "args": {}},
        ],
    )

    assert summary.sandbox_rc == 0
```

**Step 2: Implement `_run_capsule_trial`**

Flow:

1. Reset env.
2. Generate initial Python code through existing model path unless `initial_code` is injected by tests.
3. Segment code.
4. Build capsule globals from env.
5. Loop over runtime actions.
6. Run, inspect, patch, rollback, or finish.
7. Save artifacts:
   - `capsule_code_trial_XX.py`
   - `capsule_trace_trial_XX.json`
   - `capsule_prompts_trial_XX.json`

**Step 3: Route config**

In `_run_single_trial`:

```python
if config.get("agent_mode", "code") == "capsule":
    return _run_capsule_trial(env, trial, args, config)
```

**Step 4: Run tests**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_trial_loop.py -q'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/envs/trial.py tests/test_runtime_control_trial_loop.py
git commit -m "Add capsule-mode trial loop"
```

---

### Task 9: Add Capsule Config Loading And YAML

**Files:**
- Modify: `capx/utils/launch_utils.py`
- Create: `env_configs/cube_stack/franka_robosuite_cube_stack_capsule_vdm.yaml`
- Create: `tests/test_runtime_control_config.py`

**Step 1: Write config tests**

```python
from pathlib import Path
import yaml


def test_capsule_yaml_uses_code_primitives_not_robot_tools():
    data = yaml.safe_load(Path("env_configs/cube_stack/franka_robosuite_cube_stack_capsule_vdm.yaml").read_text())

    assert data["agent_mode"] == "capsule"
    assert data["max_capsule_steps"] > 0
    assert "Write Python code" in data["env"]["cfg"]["prompt"]
    assert "selecting one JSON tool call" not in data["env"]["cfg"]["prompt"]
```

**Step 2: Implement config fields**

Support:

```yaml
agent_mode: capsule
max_capsule_steps: 12
checkpoint_policy: region
rollback_policy: best_effort
capsule_feedback_level: source_region_repair_hint
```

**Step 3: Add YAML**

Base it on the current code-generation cube stack config, not the deleted tool YAML. The prompt must ask for executable Python code using the listed APIs.

**Step 4: Run tests**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_config.py -q'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/utils/launch_utils.py env_configs/cube_stack/franka_robosuite_cube_stack_capsule_vdm.yaml tests/test_runtime_control_config.py
git commit -m "Add capsule-mode config"
```

---

### Task 10: Focused Verification

**Files:**
- Modify only files needed to fix test or lint failures.

**Step 1: Sync touched files to WSL**

Copy `capx/runtime_control`, modified `capx/envs`, modified `capx/utils`, tests, and capsule YAML into `/home/capx/code/cap-x`.

**Step 2: Run focused tests**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_*.py tests/test_environments.py -q'
```

Expected: PASS.

**Step 3: Run Ruff**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync ruff check capx/runtime_control capx/envs/trial.py capx/envs/tasks/base.py capx/utils/launch_utils.py'
```

Expected: PASS.

**Step 4: Run import smoke**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync python -c "import capx.runtime_control; import capx.envs.trial; import capx.envs.tasks.base; print(\"capsule imports ok\")"'
```

Expected: `capsule imports ok`.

**Step 5: Commit fixes**

```bash
git add capx/runtime_control capx/envs/trial.py capx/envs/tasks/base.py capx/utils/launch_utils.py tests env_configs
git commit -m "Stabilize capsule runtime control"
```

Skip this commit if no verification fixes were needed.

---

## Final Checklist

- Primitive toolification files and configs are removed.
- Existing code-generation mode remains the default.
- Capsule mode never exposes robot primitives as planner tools.
- Runtime actions operate on source regions.
- Primitive calls are traced without renaming or replacing the API surface.
- Feedback references region ids and source lines.
- Local patches are scoped to a single region.
- Checkpoint rollback reports unsupported states explicitly.
- Focused tests and Ruff pass in WSL.
- Simulator rollouts are reported separately and not treated as local verification.

