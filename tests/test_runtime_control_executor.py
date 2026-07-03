from capx.runtime_control.executor import CapsuleExecutor
from capx.runtime_control.segmenter import segment_python_code
from capx.runtime_control.trace import RuntimeTrace, wrap_function_for_trace


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


def test_executor_event_includes_region_trace_events():
    trace = RuntimeTrace()
    globals_dict = {
        "get_pose": wrap_function_for_trace("get_pose", lambda: [1, 2, 3], trace)
    }
    regions = segment_python_code("pose = get_pose()\n")
    executor = CapsuleExecutor(base_globals=globals_dict, trace=trace)

    event = executor.run_region(regions[0])

    assert event.evidence["trace_events"][0]["name"] == "get_pose"
