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
