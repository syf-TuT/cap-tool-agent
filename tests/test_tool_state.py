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


def test_state_resolves_common_model_ref_aliases():
    state = ToolState()
    joints = np.array([1, 2, 3])
    ref = state.put("solve_ik", joints, summary={"shape": [3]})

    resolved = state.resolve_refs(
        {
            "canonical": {"state_ref": ref},
            "json_ref": {"$ref": ref},
            "string_ref": f"${ref}",
        }
    )

    assert resolved["canonical"] is joints
    assert resolved["json_ref"] is joints
    assert resolved["string_ref"] is joints


def test_state_resolves_first_output_when_model_uses_one_based_index():
    state = ToolState()
    first = np.array([1, 2, 3])
    second = np.array([4, 5, 6])
    state.put("solve_ik", first, summary={"shape": [3]})

    assert state.resolve_refs({"state_ref": "solve_ik.1"}) is first
    assert state.resolve_refs({"$ref": "solve_ik.1"}) is first
    assert state.resolve_refs("$solve_ik.1") is first

    state.put("solve_ik", second, summary={"shape": [3]})

    assert state.resolve_refs({"state_ref": "solve_ik.1"}) is second
    assert state.resolve_refs("$solve_ik.2") is second


def test_state_rejects_missing_ref():
    state = ToolState()

    try:
        state.resolve_refs({"state_ref": "missing.ref"})
    except KeyError as exc:
        assert "missing.ref" in str(exc)
    else:
        raise AssertionError("missing state ref should fail")
