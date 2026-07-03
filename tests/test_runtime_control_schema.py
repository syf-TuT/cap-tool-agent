from capx.runtime_control.schema import CodeRegion, RuntimeAction, RuntimeEvent


def test_code_region_exports_source_span():
    region = CodeRegion(region_id="region_1", start_line=2, end_line=4, source="x = 1")

    assert region.to_dict()["region_id"] == "region_1"
    assert region.to_dict()["source_span"] == {"start_line": 2, "end_line": 4}


def test_runtime_action_validates_args_mapping():
    action = RuntimeAction.from_mapping(
        {"action": "run_region", "args": {"region_id": "region_1"}}
    )

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
