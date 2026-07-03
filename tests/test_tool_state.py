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
