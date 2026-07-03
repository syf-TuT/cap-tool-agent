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


def test_state_applies_add_operation_to_ref_value():
    state = ToolState()
    cube_pos = np.array([0.1, 0.2, 0.3])
    state.put("get_observation", {"cubeA_pos": cube_pos})

    resolved = state.resolve_refs(
        {
            "state_ref": "get_observation.0.cubeA_pos",
            "operation": "add",
            "value": [0, 0, 0.1],
        }
    )

    np.testing.assert_allclose(resolved, np.array([0.1, 0.2, 0.4]))


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


def test_state_indexes_nested_mapping_outputs_by_path():
    state = ToolState()
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.ones((2, 2), dtype=np.float32)
    observation = {
        "robot0_robotview": {
            "images": {
                "rgb": rgb,
                "depth": depth,
            }
        },
        "robot0_robotview_image": rgb,
    }

    ref = state.put("get_observation", observation)

    assert state.get(ref) is observation
    assert state.get("get_observation.0.robot0_robotview.images.rgb") is rgb
    assert state.get("robot0_robotview.images.rgb") is rgb
    assert state.get("robot0_robotview_image") is rgb
    assert state.resolve_refs({"state_ref": "robot0_robotview.images.rgb"}) is rgb

    summary = state.summary()
    assert "get_observation.0.robot0_robotview.images.rgb" in summary
    assert summary["get_observation.0"]["nested_refs"]["robot0_robotview.images.rgb"] == {
        "ref": "get_observation.0.robot0_robotview.images.rgb",
        "type": "ndarray",
        "shape": [2, 2, 3],
        "dtype": "uint8",
    }


def test_state_updates_unversioned_nested_aliases_to_latest_mapping_output():
    state = ToolState()
    first_rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    second_rgb = np.ones((2, 2, 3), dtype=np.uint8)

    state.put("get_observation", {"robot0_robotview": {"images": {"rgb": first_rgb}}})
    state.put("get_observation", {"robot0_robotview": {"images": {"rgb": second_rgb}}})

    assert state.get("get_observation.0.robot0_robotview.images.rgb") is first_rgb
    assert state.get("get_observation.1.robot0_robotview.images.rgb") is second_rgb
    assert state.get("robot0_robotview.images.rgb") is second_rgb


def test_state_rejects_missing_ref():
    state = ToolState()

    try:
        state.resolve_refs({"state_ref": "missing.ref"})
    except KeyError as exc:
        assert "missing.ref" in str(exc)
    else:
        raise AssertionError("missing state ref should fail")
