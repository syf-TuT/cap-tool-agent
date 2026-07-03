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
