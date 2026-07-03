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


def test_state_control_api_exposes_only_state_and_motion_tools():
    from capx.integrations.franka.control_reduced import FrankaStateControlApi

    api = FrankaStateControlApi.__new__(FrankaStateControlApi)

    assert set(api.functions()) == {
        "get_observation",
        "solve_ik",
        "move_to_joints",
        "open_gripper",
        "close_gripper",
    }


def test_state_control_api_constructor_skips_vision_initializers(monkeypatch):
    from capx.integrations.franka import control_reduced

    pyroki_client = object()
    monkeypatch.setattr(control_reduced, "init_pyroki", lambda: pyroki_client)

    def fail_visual_init(*_args, **_kwargs):
        raise AssertionError("state-first API must not initialize vision or grasp services")

    for initializer in [
        "init_contact_graspnet",
        "init_sam3",
        "init_sam3_point_prompt",
        "init_owlvit",
        "init_sam2",
        "init_molmo",
        "init_pyroki_trajopt",
    ]:
        monkeypatch.setattr(control_reduced, initializer, fail_visual_init)

    api = control_reduced.FrankaStateControlApi(env=object())

    assert api.ik_solve_fn is pyroki_client


def test_state_control_api_is_registered():
    import capx.integrations  # noqa: F401
    from capx.integrations.base_api import list_apis

    assert "FrankaStateControlApi" in list_apis()
