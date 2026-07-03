# Toolified CaP-Agent0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Robosuite Franka tool-call agent mode that executes one structured tool call per LLM turn, verifies the local effect, and logs structured feedback while preserving the existing code-generation baseline.

**Architecture:** Add a `capx.tools` package for schemas, registry, state refs, execution, verification, prompts, and planning. Extend `CodeExecutionEnvBase` with a tool-facing interface that reuses existing `ApiBase.functions()` callables. Route `agent_mode: tool` configs through a new `_run_tool_trial()` loop in `capx/envs/trial.py`.

**Tech Stack:** Python dataclasses, `inspect`, `json`, `numpy`, existing CaP-X `ApiBase`, `CodeExecutionEnvBase`, pytest. Local verification is code-level only: unit tests, imports, YAML parsing, and Ruff. Do not start model servers, API servers, Robosuite simulators, Vite servers, or any long-running services on the local machine.

---

## Working Rules

Use the Windows checkout only for editing. Before running tests in WSL, sync touched files into `/home/capx/code/cap-x`.

Local verification policy:

- Allowed locally: focused pytest tests that use fakes/mocks, import checks, config parsing checks, and Ruff.
- Forbidden locally: `capx/envs/launch.py`, Robosuite rollouts, model-server calls, VDM calls, SAM/GraspNet/Pyroki API server startup, frontend dev servers, dependency installs, and `uv sync`.
- Keep simulator/service smoke checks as optional remote or high-capacity-environment checks only.
- If a task would require a real simulator or service to verify behavior, stop at code-level verification and record the skipped runtime check explicitly.

Suggested sync pattern for each task:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; mkdir -p capx/tools tests env_configs/cube_stack env_configs/cube_lifting env_configs/cube_restack; cp -r /mnt/f/code/cap-x/capx/tools/* capx/tools/ 2>/dev/null || true; cp /mnt/f/code/cap-x/tests/test_tool_*.py tests/ 2>/dev/null || true'
```

Run tests through WSL:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest <tests> -q'
```

Commit after each task with only the files touched by that task.

---

### Task 1: Tool Schema Primitives

**Files:**
- Create: `capx/tools/__init__.py`
- Create: `capx/tools/schema.py`
- Create: `tests/test_tool_schema.py`

**Step 1: Write the failing tests**

```python
from capx.tools.schema import StepFeedback, ToolCall, ToolResult, ToolSpec


def test_tool_call_parses_mapping():
    call = ToolCall.from_mapping({"tool": "solve_ik", "args": {"x": 1}, "thought": "try ik"})

    assert call.tool == "solve_ik"
    assert call.args == {"x": 1}
    assert call.thought == "try ik"


def test_tool_spec_exports_prompt_dict():
    spec = ToolSpec(
        name="solve_ik",
        description="Solve IK",
        input_schema={"position": "array[3]"},
        tags=["planning"],
        preconditions=["target_pose_available"],
        postconditions=["joint_solution_valid"],
        failure_modes=["unreachable_pose"],
    )

    prompt_dict = spec.to_prompt_dict()

    assert prompt_dict["name"] == "solve_ik"
    assert prompt_dict["input_schema"] == {"position": "array[3]"}
    assert prompt_dict["failure_modes"] == ["unreachable_pose"]


def test_tool_result_and_feedback_are_jsonable():
    result = ToolResult.failed(
        tool="solve_ik",
        failure_type="exception",
        message="bad pose",
        exception_type="ValueError",
    )
    feedback = StepFeedback(
        step_id=3,
        tool="solve_ik",
        status="failed",
        failure_stage="planning",
        failure_type="exception",
        evidence={"message": result.message},
        repair_hints=["choose another target"],
        recommended_next_tools=["solve_ik"],
    )

    assert result.to_dict()["status"] == "failed"
    assert feedback.to_dict()["repair_hints"] == ["choose another target"]
```

**Step 2: Run test to verify it fails**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_schema.py -q'
```

Expected: FAIL with `ModuleNotFoundError: No module named 'capx.tools'`.

**Step 3: Implement schema dataclasses**

Implement `capx/tools/schema.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ToolStatus = Literal["success", "failed", "warning", "invalid"]


@dataclass
class ToolSpec:
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)

    def to_prompt_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    thought: str = ""
    step_id: int | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ToolCall":
        if "tool" not in data:
            raise ValueError("Tool call must include 'tool'")
        args = data.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError("Tool call 'args' must be an object")
        return cls(
            tool=str(data["tool"]),
            args=args,
            thought=str(data.get("thought", "")),
            step_id=data.get("step_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    tool: str
    status: ToolStatus
    output_ref: str | None = None
    output_summary: Any = None
    message: str = ""
    failure_type: str | None = None
    stdout: str = ""
    stderr: str = ""
    duration_s: float | None = None
    exception_type: str | None = None

    @classmethod
    def failed(
        cls,
        *,
        tool: str,
        failure_type: str,
        message: str,
        exception_type: str | None = None,
        stderr: str = "",
    ) -> "ToolResult":
        return cls(
            tool=tool,
            status="failed",
            failure_type=failure_type,
            message=message,
            exception_type=exception_type,
            stderr=stderr,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepFeedback:
    step_id: int
    tool: str
    status: ToolStatus
    failure_stage: str | None = None
    failure_type: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    repair_hints: list[str] = field(default_factory=list)
    recommended_next_tools: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

Implement `capx/tools/__init__.py`:

```python
from capx.tools.schema import StepFeedback, ToolCall, ToolResult, ToolSpec

__all__ = ["StepFeedback", "ToolCall", "ToolResult", "ToolSpec"]
```

**Step 4: Run test to verify it passes**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_schema.py -q'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/tools/__init__.py capx/tools/schema.py tests/test_tool_schema.py
git commit -m "Add tool call schema primitives"
```

---

### Task 2: Tool State References

**Files:**
- Create: `capx/tools/state.py`
- Create: `tests/test_tool_state.py`
- Modify: `capx/tools/__init__.py`

**Step 1: Write the failing tests**

```python
import numpy as np

from capx.tools.state import ToolState


def test_state_stores_large_value_by_ref():
    state = ToolState()
    value = np.ones((2, 3))

    ref = state.put("mask", value, summary={"shape": [2, 3], "area": 6})

    assert ref.startswith("mask.")
    assert state.get(ref) is value
    assert state.summary()[ref]["area"] == 6


def test_state_resolves_nested_refs():
    state = ToolState()
    arr = np.array([1, 2, 3])
    ref = state.put("position", arr, summary={"shape": [3]})

    resolved = state.resolve_refs({"position": {"state_ref": ref}, "scale": 2})

    assert resolved["position"] is arr
    assert resolved["scale"] == 2


def test_state_rejects_missing_ref():
    state = ToolState()

    try:
        state.resolve_refs({"state_ref": "missing.ref"})
    except KeyError as exc:
        assert "missing.ref" in str(exc)
    else:
        raise AssertionError("missing state ref should fail")
```

**Step 2: Run test to verify it fails**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_state.py -q'
```

Expected: FAIL with missing `capx.tools.state`.

**Step 3: Implement `ToolState`**

```python
from __future__ import annotations

from collections import defaultdict
from typing import Any


class ToolState:
    def __init__(self) -> None:
        self._values: dict[str, Any] = {}
        self._summaries: dict[str, Any] = {}
        self._counts: defaultdict[str, int] = defaultdict(int)

    def put(self, namespace: str, value: Any, *, summary: Any = None) -> str:
        idx = self._counts[namespace]
        self._counts[namespace] += 1
        ref = f"{namespace}.{idx}"
        self._values[ref] = value
        self._summaries[ref] = summary if summary is not None else self._default_summary(value)
        return ref

    def get(self, ref: str) -> Any:
        if ref not in self._values:
            raise KeyError(f"Unknown state ref: {ref}")
        return self._values[ref]

    def summary(self) -> dict[str, Any]:
        return dict(self._summaries)

    def resolve_refs(self, value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {"state_ref"}:
                return self.get(str(value["state_ref"]))
            return {k: self.resolve_refs(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.resolve_refs(v) for v in value]
        return value

    def _default_summary(self, value: Any) -> Any:
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is not None:
            return {"type": type(value).__name__, "shape": list(shape), "dtype": str(dtype)}
        return {"type": type(value).__name__, "repr": repr(value)[:200]}
```

Export `ToolState` from `capx/tools/__init__.py`.

**Step 4: Run test to verify it passes**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_state.py -q'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/tools/__init__.py capx/tools/state.py tests/test_tool_state.py
git commit -m "Add tool state references"
```

---

### Task 3: Tool Registry From Existing APIs

**Files:**
- Create: `capx/tools/registry.py`
- Create: `tests/test_tool_registry.py`
- Modify: `capx/tools/__init__.py`

**Step 1: Write the failing tests**

```python
from capx.tools.registry import ToolRegistry, build_registry_from_apis


class FakeApi:
    def add(self, x: int, y: int = 1) -> int:
        """Add two numbers."""
        return x + y

    def functions(self):
        return {"add": self.add}


def test_registry_builds_specs_from_api_functions():
    registry = build_registry_from_apis({"fake": FakeApi()})

    spec = registry.spec("add")

    assert spec.name == "add"
    assert "Add two numbers" in spec.description
    assert "x" in spec.input_schema
    assert registry.get("add")(2, y=3) == 5


def test_registry_applies_metadata_overlay():
    registry = build_registry_from_apis(
        {"fake": FakeApi()},
        metadata={"add": {"tags": ["math"], "failure_modes": ["bad_input"]}},
    )

    spec = registry.spec("add")

    assert spec.tags == ["math"]
    assert spec.failure_modes == ["bad_input"]


def test_registry_rejects_unknown_tool():
    registry = ToolRegistry()

    try:
        registry.spec("missing")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown tool should fail")
```

**Step 2: Run test to verify it fails**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_registry.py -q'
```

Expected: FAIL with missing `capx.tools.registry`.

**Step 3: Implement registry**

Implement:

- `ToolRegistry.register(spec, fn)`
- `ToolRegistry.spec(name)`
- `ToolRegistry.get(name)`
- `ToolRegistry.prompt_specs()`
- `build_registry_from_apis(apis, metadata=None)`

Use `inspect.signature(fn)` to build a simple schema:

```python
def _schema_from_signature(fn):
    schema = {}
    for name, param in inspect.signature(fn).parameters.items():
        if name == "self":
            continue
        entry = {"required": param.default is inspect.Parameter.empty}
        if param.annotation is not inspect.Parameter.empty:
            entry["type"] = getattr(param.annotation, "__name__", str(param.annotation))
        if param.default is not inspect.Parameter.empty:
            entry["default"] = param.default
        schema[name] = entry
    return schema
```

Keep metadata overlay explicit and shallow:

```python
for key, value in metadata.get(name, {}).items():
    setattr(spec, key, value)
```

**Step 4: Run tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_registry.py tests/test_tool_schema.py -q'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/tools/__init__.py capx/tools/registry.py tests/test_tool_registry.py
git commit -m "Build tool registry from API functions"
```

---

### Task 4: Tool Executor

**Files:**
- Create: `capx/tools/executor.py`
- Create: `tests/test_tool_executor.py`
- Modify: `capx/tools/__init__.py`

**Step 1: Write the failing tests**

```python
import numpy as np

from capx.tools.executor import ToolExecutor
from capx.tools.registry import ToolRegistry
from capx.tools.schema import ToolCall, ToolSpec
from capx.tools.state import ToolState


def test_executor_calls_registered_tool_with_resolved_refs():
    state = ToolState()
    arr_ref = state.put("array", np.array([1, 2]), summary={"shape": [2]})
    registry = ToolRegistry()
    registry.register(ToolSpec(name="sum_array"), lambda arr: int(arr.sum()))

    result = ToolExecutor(registry, state).run(
        ToolCall(tool="sum_array", args={"arr": {"state_ref": arr_ref}})
    )

    assert result.status == "success"
    assert result.output_summary == 3


def test_executor_rejects_unknown_tool():
    result = ToolExecutor(ToolRegistry(), ToolState()).run(ToolCall(tool="missing"))

    assert result.status == "invalid"
    assert result.failure_type == "unknown_tool"


def test_executor_wraps_exception():
    registry = ToolRegistry()

    def explode():
        raise ValueError("bad")

    registry.register(ToolSpec(name="explode"), explode)

    result = ToolExecutor(registry, ToolState()).run(ToolCall(tool="explode"))

    assert result.status == "failed"
    assert result.exception_type == "ValueError"
    assert "bad" in result.message
```

**Step 2: Run test to verify it fails**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_executor.py -q'
```

Expected: FAIL with missing executor.

**Step 3: Implement executor**

Implementation requirements:

- Resolve refs using `ToolState.resolve_refs`.
- Reject unknown tools before function invocation.
- Capture stdout/stderr with `contextlib.redirect_stdout/redirect_stderr`.
- Store large outputs in state only when needed later. In this first task, return scalar
  outputs directly as `output_summary`.
- Store numpy arrays as refs by default.

Use this output summary rule:

```python
def _store_or_summarize(self, tool: str, output: Any) -> tuple[str | None, Any]:
    if hasattr(output, "shape"):
        ref = self.state.put(tool, output)
        return ref, self.state.summary()[ref]
    return None, output
```

**Step 4: Run tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_executor.py tests/test_tool_state.py tests/test_tool_registry.py -q'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/tools/__init__.py capx/tools/executor.py tests/test_tool_executor.py
git commit -m "Add structured tool executor"
```

---

### Task 5: Feedback Verifier

**Files:**
- Create: `capx/tools/verifiers.py`
- Create: `tests/test_tool_verifiers.py`
- Modify: `capx/tools/__init__.py`

**Step 1: Write the failing tests**

```python
from capx.tools.schema import ToolCall, ToolResult
from capx.tools.verifiers import StepVerifier


def test_verifier_passes_successful_perception_result():
    verifier = StepVerifier()
    result = ToolResult(
        tool="segment_sam3_text_prompt",
        status="success",
        output_summary={"count": 1, "best_score": 0.92, "mask_area": 1200},
    )

    feedback = verifier.verify(
        step_id=1,
        tool_call=ToolCall(tool="segment_sam3_text_prompt"),
        result=result,
        before={},
        after={},
    )

    assert feedback.status == "success"


def test_verifier_flags_low_confidence_mask():
    verifier = StepVerifier()
    result = ToolResult(
        tool="segment_sam3_text_prompt",
        status="success",
        output_summary={"count": 1, "best_score": 0.2, "mask_area": 10},
    )

    feedback = verifier.verify(
        step_id=1,
        tool_call=ToolCall(tool="segment_sam3_text_prompt"),
        result=result,
        before={},
        after={},
    )

    assert feedback.status == "warning"
    assert feedback.failure_type == "low_confidence_mask"
    assert "segment_sam3_text_prompt" in feedback.recommended_next_tools


def test_verifier_converts_failed_result_to_feedback():
    verifier = StepVerifier()
    result = ToolResult.failed(tool="solve_ik", failure_type="exception", message="bad")

    feedback = verifier.verify(
        step_id=2,
        tool_call=ToolCall(tool="solve_ik"),
        result=result,
        before={},
        after={},
    )

    assert feedback.status == "failed"
    assert feedback.failure_type == "exception"
```

**Step 2: Run test to verify it fails**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_verifiers.py -q'
```

Expected: FAIL with missing verifier.

**Step 3: Implement verifier**

Implement `StepVerifier.verify(...)`.

Start with deterministic rule-based checks:

- If `ToolResult.status in {"failed", "invalid"}`, pass through failure.
- For segmentation tools, require `best_score >= 0.4` and `mask_area >= 50` when
  those keys are present.
- For IK tools, check output summaries do not report invalid shape or NaN.
- For execution tools, compare `before.get("reward")` and `after.get("reward")`;
  include reward delta in evidence.

Keep thresholds module-level constants so later tasks can tune them.

**Step 4: Run tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_verifiers.py tests/test_tool_schema.py -q'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/tools/__init__.py capx/tools/verifiers.py tests/test_tool_verifiers.py
git commit -m "Add step-level tool verifier"
```

---

### Task 6: Tool Prompt Builder and Parser

**Files:**
- Create: `capx/tools/prompts.py`
- Create: `tests/test_tool_prompts.py`
- Modify: `capx/tools/__init__.py`

**Step 1: Write the failing tests**

```python
from capx.tools.prompts import build_tool_planner_prompt, parse_tool_call_response
from capx.tools.schema import ToolCall, ToolSpec


def test_parse_tool_call_response_accepts_json_object():
    call = parse_tool_call_response('{"thought": "look", "tool": "get_observation", "args": {}}')

    assert isinstance(call, ToolCall)
    assert call.tool == "get_observation"


def test_parse_tool_call_response_rejects_python_code():
    try:
        parse_tool_call_response("```python\nmove_to_joints(joints)\n```")
    except ValueError as exc:
        assert "JSON" in str(exc)
    else:
        raise AssertionError("Python code should be rejected")


def test_prompt_contains_tools_and_last_feedback():
    prompt = build_tool_planner_prompt(
        task="stack red cube on green cube",
        tool_specs=[ToolSpec(name="get_observation", description="Capture obs")],
        state_summary={"reward": 0.0},
        history=[{"feedback": {"status": "failed", "failure_type": "low_confidence_mask"}}],
    )

    text = prompt[-1]["content"][0]["text"]

    assert "Do not write Python code" in text
    assert "get_observation" in text
    assert "low_confidence_mask" in text
```

**Step 2: Run test to verify it fails**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_prompts.py -q'
```

Expected: FAIL with missing prompt module.

**Step 3: Implement prompt builder and parser**

Requirements:

- Parser strips Markdown fences only if the fenced content is JSON.
- Parser requires a JSON object.
- Parser delegates validation to `ToolCall.from_mapping`.
- Prompt returns the same chat-message shape used elsewhere:

```python
[
    {"role": "system", "content": "You select one robot tool call at a time."},
    {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
]
```

Prompt text must include:

- Task.
- Tool specs as JSON.
- State summary.
- Recent history.
- Exact output contract.
- "Do not write Python code."

**Step 4: Run tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_prompts.py tests/test_tool_schema.py -q'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/tools/__init__.py capx/tools/prompts.py tests/test_tool_prompts.py
git commit -m "Add tool planner prompt parsing"
```

---

### Task 7: Extend `CodeExecutionEnvBase` With Tool Interface

**Files:**
- Modify: `capx/envs/tasks/base.py`
- Create: `tests/test_tool_env_interface.py`

**Step 1: Write the failing tests**

```python
from gymnasium import Env

from capx.envs.tasks.base import CodeExecEnvConfig, CodeExecutionEnvBase
from capx.tools.schema import ToolCall


class FakeLowLevelEnv(Env):
    def reset(self, *, seed=None, options=None):
        return {"value": 1}, {}

    def get_observation(self):
        return {"value": 1}

    def compute_reward(self):
        return 0.0

    def task_completed(self):
        return False


class FakeApi:
    def __init__(self, env):
        self.env = env

    def functions(self):
        return {"add": self.add}

    def combined_doc(self):
        return "add(x, y)"

    def add(self, x: int, y: int = 1) -> int:
        """Add values."""
        return x + y


def test_code_env_exposes_tool_specs_and_call(monkeypatch):
    monkeypatch.setattr(
        "capx.envs.tasks.base.get_api",
        lambda name: (lambda env: FakeApi(env)),
    )
    env = CodeExecutionEnvBase(
        CodeExecEnvConfig(low_level=FakeLowLevelEnv(), apis=["FakeApi"], prompt="test")
    )

    specs = env.tool_specs()
    result = env.call_tool(ToolCall(tool="add", args={"x": 2, "y": 3}))

    assert [spec.name for spec in specs] == ["add"]
    assert result.status == "success"
    assert result.output_summary == 5


def test_code_env_snapshot_includes_reward_and_task_completed(monkeypatch):
    monkeypatch.setattr(
        "capx.envs.tasks.base.get_api",
        lambda name: (lambda env: FakeApi(env)),
    )
    env = CodeExecutionEnvBase(
        CodeExecEnvConfig(low_level=FakeLowLevelEnv(), apis=["FakeApi"], prompt="test")
    )

    snapshot = env.snapshot_state()

    assert snapshot["reward"] == 0.0
    assert snapshot["task_completed"] is False
```

**Step 2: Run test to verify it fails**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_env_interface.py -q'
```

Expected: FAIL with missing `tool_specs` or `call_tool`.

**Step 3: Implement env tool interface**

In `CodeExecutionEnvBase.__init__`, initialize:

```python
from capx.tools.executor import ToolExecutor
from capx.tools.registry import build_registry_from_apis
from capx.tools.state import ToolState

self._tool_state = ToolState()
self._tool_registry = build_registry_from_apis(self._apis)
self._tool_executor = ToolExecutor(self._tool_registry, self._tool_state)
```

In `_init_exec_globals` or `reset`, reset `_tool_state` and executor to avoid leakage
across episodes.

Add methods:

```python
def tool_specs(self):
    return self._tool_registry.specs()

def call_tool(self, tool_call):
    return self._tool_executor.run(tool_call)

def tool_state_summary(self):
    return self._tool_state.summary()

def snapshot_state(self):
    reward = float(self.compute_reward())
    completed = self.low_level_env.task_completed() if hasattr(self.low_level_env, "task_completed") else None
    return {
        "reward": reward,
        "task_completed": completed,
        "step_count": self._step_count,
        "observation_keys": list(self.low_level_env.get_observation().keys()),
    }
```

**Step 4: Run tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_env_interface.py tests/test_tool_executor.py tests/test_tool_registry.py -q'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/envs/tasks/base.py tests/test_tool_env_interface.py
git commit -m "Expose tool calls from code execution env"
```

---

### Task 8: Tool Planner Abstraction

**Files:**
- Create: `capx/tools/planner.py`
- Create: `tests/test_tool_planner.py`
- Modify: `capx/tools/__init__.py`

**Step 1: Write the failing tests**

```python
from capx.tools.planner import LlmToolPlanner, ScriptedToolPlanner
from capx.tools.schema import ToolCall


def test_scripted_tool_planner_returns_calls_in_order():
    planner = ScriptedToolPlanner([
        {"tool": "get_observation", "args": {}},
        {"tool": "finish", "args": {}},
    ])

    assert planner.next_call([], {}) == ToolCall(tool="get_observation", args={})
    assert planner.next_call([], {}).tool == "finish"


def test_llm_tool_planner_parses_query_result():
    calls = []

    def fake_query(args, prompt):
        calls.append(prompt)
        return {"content": '{"tool": "finish", "args": {}}', "reasoning": ""}

    planner = LlmToolPlanner(query_model=fake_query, args=object())

    call = planner.next_call(prompt=[{"role": "user", "content": [{"type": "text", "text": "x"}]}])

    assert call.tool == "finish"
    assert calls
```

**Step 2: Run test to verify it fails**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_planner.py -q'
```

Expected: FAIL with missing planner module.

**Step 3: Implement planner classes**

Implement:

- `ScriptedToolPlanner(script: list[dict])`
- `LlmToolPlanner(query_model, args)`

`LlmToolPlanner.next_call(prompt=...)` should call the existing `_query_model` compatible
function and parse `content` through `parse_tool_call_response`.

Keep this small; do not add retry logic yet.

**Step 4: Run tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_planner.py tests/test_tool_prompts.py -q'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/tools/__init__.py capx/tools/planner.py tests/test_tool_planner.py
git commit -m "Add tool planner abstraction"
```

---

### Task 9: Tool Trial Loop

**Files:**
- Modify: `capx/envs/trial.py`
- Create: `tests/test_tool_trial_loop.py`

**Step 1: Write the failing tests**

```python
from types import SimpleNamespace

from capx.envs.trial import _run_tool_trial
from capx.tools.schema import ToolResult, ToolSpec


class FakeToolEnv:
    oracle_code = ""

    def __init__(self):
        self.calls = []
        self.completed = False

    def reset(self, *, seed=None, options=None):
        return {"full_prompt": [{"role": "system", "content": "x"}, {"role": "user", "content": [{"type": "text", "text": "task"}]}]}, {}

    def tool_specs(self):
        return [ToolSpec(name="finish", description="Finish")]

    def tool_state_summary(self):
        return {}

    def snapshot_state(self):
        return {"reward": 1.0 if self.completed else 0.0, "task_completed": self.completed}

    def call_tool(self, tool_call):
        self.calls.append(tool_call.tool)
        if tool_call.tool == "finish":
            self.completed = True
        return ToolResult(tool=tool_call.tool, status="success", output_summary=None)


def test_tool_trial_loop_finishes_with_scripted_planner(tmp_path):
    env = FakeToolEnv()
    args = SimpleNamespace(model="test")
    config = {
        "output_dir": str(tmp_path),
        "max_tool_steps": 3,
        "record_video": False,
        "use_img_differencing": False,
        "use_video_differencing": False,
    }

    summary = _run_tool_trial(
        env=env,
        trial=1,
        args=args,
        config=config,
        scripted_tool_calls=[{"tool": "finish", "args": {}}],
    )

    assert summary.success is True
    assert summary.task_completed is True
    assert env.calls == ["finish"]
```

**Step 2: Run test to verify it fails**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_trial_loop.py -q'
```

Expected: FAIL with missing `_run_tool_trial`.

**Step 3: Implement `_run_tool_trial`**

Add `_run_tool_trial(...)` near `_run_single_trial`.

Requirements:

- Reset env.
- Build prompt each step with `build_tool_planner_prompt`.
- Use `ScriptedToolPlanner` when `scripted_tool_calls` is provided or config contains
  `scripted_tool_calls`.
- Use `LlmToolPlanner` otherwise.
- Run `StepVerifier` for every call.
- Treat `finish` specially: do not route through registry if env does not have a finish
  tool; compute task status and exit.
- Save `tool_trace.json` in `config["output_dir"]`.
- Return `TrialSummary` with:
  - `success = task_completed or reward == 1.0`
  - `num_code_blocks = number of tool calls`
  - `sandbox_rc = 0 if no invalid/fatal failure else 1`

At the top of `_run_single_trial`, add:

```python
if config.get("agent_mode", "code") == "tool":
    return _run_tool_trial(env, trial, args, config)
```

**Step 4: Run tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_trial_loop.py tests/test_tool_planner.py tests/test_tool_verifiers.py -q'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/envs/trial.py tests/test_tool_trial_loop.py
git commit -m "Add tool-mode trial loop"
```

---

### Task 10: Config Loading for Tool Mode

**Files:**
- Modify: `capx/utils/launch_utils.py`
- Create: `tests/test_tool_config_loading.py`

**Step 1: Write the failing tests**

```python
from types import SimpleNamespace

import yaml

from capx.utils.launch_utils import _load_config


def test_load_config_reads_tool_mode_fields(tmp_path):
    cfg_path = tmp_path / "tool.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "env": {"_target_": "fake.Env"},
                "agent_mode": "tool",
                "max_tool_steps": 7,
                "tool_feedback_level": "repair_hint",
                "trials": 1,
            }
        )
    )
    args = SimpleNamespace(
        config_path=str(cfg_path),
        total_trials=None,
        num_workers=None,
        record_video=None,
        output_dir=None,
        use_oracle_code=None,
        use_visual_feedback=None,
        use_img_differencing=None,
        use_video_differencing=None,
        use_wrist_camera=None,
        use_parallel_ensemble=None,
        use_multimodel=None,
        web_ui=None,
        web_ui_port=None,
        server_url="http://127.0.0.1:8110/chat/completions",
        visual_differencing_model="google/gemini-3.1-pro-preview",
        visual_differencing_model_server_url="http://127.0.0.1:8110/chat/completions",
        visual_differencing_model_api_key=None,
    )

    _, config, _ = _load_config(args)

    assert config["agent_mode"] == "tool"
    assert config["max_tool_steps"] == 7
    assert config["tool_feedback_level"] == "repair_hint"
```

**Step 2: Run test to verify it fails**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_config_loading.py -q'
```

Expected: FAIL because fields are not copied into merged config.

**Step 3: Implement config merge**

In `_load_config`, add:

```python
"agent_mode": configs_dict.get("agent_mode", "code"),
"max_tool_steps": configs_dict.get("max_tool_steps", 20),
"tool_feedback_level": configs_dict.get("tool_feedback_level", "repair_hint"),
"scripted_tool_calls": configs_dict.get("scripted_tool_calls", None),
```

Do not add CLI arguments for these fields in the first version unless needed.

**Step 4: Run tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_config_loading.py tests/test_tool_trial_loop.py -q'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/utils/launch_utils.py tests/test_tool_config_loading.py
git commit -m "Load tool-mode config fields"
```

---

### Task 11: Robosuite Franka Tool Metadata

**Files:**
- Create: `capx/tools/franka_metadata.py`
- Modify: `capx/envs/tasks/base.py`
- Create: `tests/test_franka_tool_metadata.py`

**Step 1: Write the failing tests**

```python
from capx.tools.franka_metadata import FRANKA_TOOL_METADATA


def test_franka_tool_metadata_marks_core_tools():
    assert FRANKA_TOOL_METADATA["segment_sam3_text_prompt"]["tags"] == ["perception"]
    assert "low_confidence_mask" in FRANKA_TOOL_METADATA["segment_sam3_text_prompt"]["failure_modes"]
    assert FRANKA_TOOL_METADATA["solve_ik"]["tags"] == ["planning"]
    assert FRANKA_TOOL_METADATA["move_to_joints"]["tags"] == ["execution"]
```

**Step 2: Run test to verify it fails**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_franka_tool_metadata.py -q'
```

Expected: FAIL with missing metadata module.

**Step 3: Implement metadata overlay**

Implement metadata for:

- `get_observation`
- `segment_sam3_text_prompt`
- `segment_sam3_point_prompt`
- `point_prompt_molmo`
- `mask_to_world_points`
- `get_oriented_bounding_box_from_3d_points`
- `plan_grasp`
- `select_top_down_grasp`
- `solve_ik`
- `move_to_joints`
- `open_gripper`
- `close_gripper`

In `CodeExecutionEnvBase`, pass `FRANKA_TOOL_METADATA` to `build_registry_from_apis`.
This is acceptable for the first version because tool mode is scoped to Franka.

**Step 4: Run tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_franka_tool_metadata.py tests/test_tool_env_interface.py -q'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/tools/franka_metadata.py capx/envs/tasks/base.py tests/test_franka_tool_metadata.py
git commit -m "Add Franka tool metadata"
```

---

### Task 12: Tool-Mode YAML Configs

**Files:**
- Create: `env_configs/cube_stack/franka_robosuite_cube_stack_tool_vdm.yaml`
- Create: `env_configs/cube_lifting/franka_robosuite_cube_lifting_tool_vdm.yaml`
- Create: `env_configs/cube_restack/franka_robosuite_cube_restack_tool_vdm.yaml`
- Create: `tests/test_tool_yaml_configs.py`

**Step 1: Write the failing tests**

```python
from pathlib import Path

import yaml


CONFIGS = [
    "env_configs/cube_stack/franka_robosuite_cube_stack_tool_vdm.yaml",
    "env_configs/cube_lifting/franka_robosuite_cube_lifting_tool_vdm.yaml",
    "env_configs/cube_restack/franka_robosuite_cube_restack_tool_vdm.yaml",
]


def test_tool_yaml_configs_define_tool_mode():
    for path in CONFIGS:
        data = yaml.safe_load(Path(path).read_text())
        assert data["agent_mode"] == "tool"
        assert data["max_tool_steps"] > 0
        assert data["env"]["cfg"]["apis"] == ["FrankaControlApiReducedSkillLibrary"]
```

**Step 2: Run test to verify it fails**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_yaml_configs.py -q'
```

Expected: FAIL because configs do not exist.

**Step 3: Add configs**

Base each config on the corresponding `*_multiturn_vdm_reduced_api_skill_lib.yaml`,
but change:

```yaml
agent_mode: tool
max_tool_steps: 20
tool_feedback_level: repair_hint
```

Prompt text must say:

```text
You are controlling a Franka Emika robot by selecting one JSON tool call at a time.
Do not write Python code.
```

Keep the same `api_servers`, `record_video`, `use_img_differencing`, and task goal.

**Step 4: Run tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_yaml_configs.py tests/test_tool_config_loading.py -q'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add env_configs/cube_stack/franka_robosuite_cube_stack_tool_vdm.yaml env_configs/cube_lifting/franka_robosuite_cube_lifting_tool_vdm.yaml env_configs/cube_restack/franka_robosuite_cube_restack_tool_vdm.yaml tests/test_tool_yaml_configs.py
git commit -m "Add Franka tool-mode configs"
```

---

### Task 13: Artifacts and Metrics

**Files:**
- Modify: `capx/envs/trial.py`
- Create: `tests/test_tool_artifacts.py`

**Step 1: Write the failing tests**

```python
import json
from types import SimpleNamespace

from capx.envs.trial import _run_tool_trial
from tests.test_tool_trial_loop import FakeToolEnv


def test_tool_trial_writes_trace_artifact(tmp_path):
    summary = _run_tool_trial(
        env=FakeToolEnv(),
        trial=1,
        args=SimpleNamespace(model="test"),
        config={
            "output_dir": str(tmp_path),
            "max_tool_steps": 3,
            "record_video": False,
            "use_img_differencing": False,
            "use_video_differencing": False,
        },
        scripted_tool_calls=[{"tool": "finish", "args": {}}],
    )

    trace_path = tmp_path / "tool_trace_trial_01.json"

    assert summary.success is True
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text())
    assert trace[0]["call"]["tool"] == "finish"
    assert trace[0]["feedback"]["status"] == "success"
```

**Step 2: Run test to verify it fails**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_artifacts.py -q'
```

Expected: FAIL if trace is not saved with this name and shape.

**Step 3: Implement artifact saving**

In `_run_tool_trial`, write:

- `tool_trace_trial_{trial:02d}.json`
- `tool_prompts_trial_{trial:02d}.json`

Each trace entry should include:

```json
{
  "step_id": 1,
  "call": {},
  "result": {},
  "feedback": {},
  "state_before": {},
  "state_after": {}
}
```

Also add metrics into `TrialSummary.log`:

- Number of tool calls.
- Number of invalid calls.
- First failure step.
- Feedback latency placeholder. For first version, set `0` when verifier feedback is
  produced immediately after failure.

**Step 4: Run tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_artifacts.py tests/test_tool_trial_loop.py -q'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/envs/trial.py tests/test_tool_artifacts.py
git commit -m "Save tool-mode trace artifacts"
```

---

### Task 14: Full Unit Test Pass and Lint

**Files:**
- Modify only files needed to fix test or lint failures.

**Step 1: Sync all touched files to WSL**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; cp -r /mnt/f/code/cap-x/capx/tools capx/; cp /mnt/f/code/cap-x/capx/envs/tasks/base.py capx/envs/tasks/base.py; cp /mnt/f/code/cap-x/capx/envs/trial.py capx/envs/trial.py; cp /mnt/f/code/cap-x/capx/utils/launch_utils.py capx/utils/launch_utils.py; cp /mnt/f/code/cap-x/tests/test_tool_*.py tests/; cp /mnt/f/code/cap-x/env_configs/cube_stack/franka_robosuite_cube_stack_tool_vdm.yaml env_configs/cube_stack/; cp /mnt/f/code/cap-x/env_configs/cube_lifting/franka_robosuite_cube_lifting_tool_vdm.yaml env_configs/cube_lifting/; cp /mnt/f/code/cap-x/env_configs/cube_restack/franka_robosuite_cube_restack_tool_vdm.yaml env_configs/cube_restack/'
```

**Step 2: Run focused tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_schema.py tests/test_tool_state.py tests/test_tool_registry.py tests/test_tool_executor.py tests/test_tool_verifiers.py tests/test_tool_prompts.py tests/test_tool_planner.py tests/test_tool_env_interface.py tests/test_tool_trial_loop.py tests/test_tool_config_loading.py tests/test_franka_tool_metadata.py tests/test_tool_yaml_configs.py tests/test_tool_artifacts.py -q'
```

Expected: all PASS.

**Step 3: Run Ruff on touched Python files**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync ruff check capx/tools capx/envs/tasks/base.py capx/envs/trial.py capx/utils/launch_utils.py'
```

Expected: PASS.

**Step 4: Fix failures**

If tests or lint fail, make minimal changes, sync again, and rerun the failing command.

**Step 5: Commit**

```bash
git add capx/tools capx/envs/tasks/base.py capx/envs/trial.py capx/utils/launch_utils.py tests/test_tool_*.py tests/test_franka_tool_metadata.py env_configs/cube_stack/franka_robosuite_cube_stack_tool_vdm.yaml env_configs/cube_lifting/franka_robosuite_cube_lifting_tool_vdm.yaml env_configs/cube_restack/franka_robosuite_cube_restack_tool_vdm.yaml
git commit -m "Stabilize tool-mode unit tests"
```

Skip this commit if there were no changes after Task 13.

---

### Task 15: Local Code-Level Verification Only

**Files:**
- No source changes expected unless code-level verification reveals a bug.

**Step 1: Confirm no local service or simulator command is required**

Do not run:

- `capx/envs/launch.py`
- Any config with `api_servers` through the launcher
- Robosuite rollouts
- Model, SAM, GraspNet, Pyroki, VDM, or frontend servers

The YAML files may still contain `api_servers` for future runtime experiments; local
verification must only parse them.

**Step 2: Run focused unit tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_tool_schema.py tests/test_tool_state.py tests/test_tool_registry.py tests/test_tool_executor.py tests/test_tool_verifiers.py tests/test_tool_prompts.py tests/test_tool_planner.py tests/test_tool_env_interface.py tests/test_tool_trial_loop.py tests/test_tool_config_loading.py tests/test_franka_tool_metadata.py tests/test_tool_yaml_configs.py tests/test_tool_artifacts.py -q'
```

Expected: all PASS.

**Step 3: Run Ruff on touched Python files**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync ruff check capx/tools capx/envs/tasks/base.py capx/envs/trial.py capx/utils/launch_utils.py'
```

Expected: PASS.

**Step 4: Run import and config parsing smoke without instantiating environments**

This command must not call `instantiate`, `launch.py`, or any API server startup.

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync python - <<'"'"'PY'"'"'\nfrom capx.envs.configs.loader import DictLoader\nfrom capx.tools import ToolCall, ToolSpec, ToolState\npaths = [\n    "env_configs/cube_stack/franka_robosuite_cube_stack_tool_vdm.yaml",\n    "env_configs/cube_lifting/franka_robosuite_cube_lifting_tool_vdm.yaml",\n    "env_configs/cube_restack/franka_robosuite_cube_restack_tool_vdm.yaml",\n]\nfor path in paths:\n    cfg = DictLoader.load(path)\n    assert cfg["agent_mode"] == "tool", path\n    assert cfg["max_tool_steps"] > 0, path\nassert ToolCall(tool="finish").tool == "finish"\nassert ToolSpec(name="get_observation").name == "get_observation"\nstate = ToolState()\nref = state.put("value", 1, summary={"value": 1})\nassert state.get(ref) == 1\nprint("code-level smoke ok")\nPY'
```

Expected: `code-level smoke ok`.

**Step 5: Record skipped runtime checks**

In the final implementation summary, explicitly state that the following were not run
locally because they require services or simulator resources:

- Existing Robosuite oracle rollout.
- LLM-backed tool-mode rollout.
- VDM/image/video differencing.
- SAM/GraspNet/Pyroki server-backed tools.

**Step 6: Commit bug fixes if needed**

```bash
git add <fixed files>
git commit -m "Fix tool-mode code-level verification issues"
```

Skip if no changes were needed.

### Optional Remote/High-Capacity Runtime Checks

Do not run these on the local machine. Use only on a machine with the required
Robosuite, GPU, model server, and API server resources.

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; export MUJOCO_GL=egl; uv run --no-sync capx/envs/launch.py --config-path env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml --use-oracle-code True --total-trials 1 --num-workers 1 --record-video False'
```

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; export MUJOCO_GL=egl; uv run --no-sync capx/envs/launch.py --config-path env_configs/cube_stack/franka_robosuite_cube_stack_tool_vdm.yaml --total-trials 1 --num-workers 1 --record-video False'
```

---

## Final Review Checklist

Before marking implementation complete:

- `agent_mode` defaults to `code`, so existing configs are unchanged.
- Tool mode never executes LLM-generated Python.
- Unknown tool names return structured invalid-call feedback.
- Exceptions in tools become `ToolResult(status="failed")`.
- Large arrays use state refs instead of prompt serialization.
- `tool_trace_trial_XX.json` records call, result, feedback, and state snapshots.
- Focused pytest suite passes in WSL without service or simulator startup.
- Ruff passes on touched Python files.
- Import and YAML parsing smoke passes without instantiating environments.
- Runtime rollouts are explicitly skipped locally unless a remote/high-capacity
  environment is provided.
