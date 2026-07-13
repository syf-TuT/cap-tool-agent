import numpy as np

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


def test_trace_window_returns_events_since_mark():
    trace = RuntimeTrace()
    start = trace.mark()
    trace.log({"name": "first"})
    trace.log({"name": "second"})

    assert [event["name"] for event in trace.events_since(start)] == ["first", "second"]


def test_trace_includes_values_for_small_numeric_arrays():
    trace = RuntimeTrace()
    wrapped = wrap_function_for_trace("pose", lambda: np.array([0.1, 0.2, 0.3]), trace)

    wrapped()

    assert trace.events[0]["result"]["value"] == [0.1, 0.2, 0.3]


def test_trace_omits_values_for_large_numeric_arrays():
    trace = RuntimeTrace()
    wrapped = wrap_function_for_trace("image", lambda: np.zeros((8, 8)), trace)

    wrapped()

    assert trace.events[0]["result"]["shape"] == [8, 8]
    assert "value" not in trace.events[0]["result"]


def test_trace_safely_summarizes_non_numpy_shaped_values():
    class TensorLike:
        shape = (3,)
        dtype = "float32"

        def size(self):
            return 3

    trace = RuntimeTrace()
    wrapped = wrap_function_for_trace("tensor", TensorLike, trace)

    wrapped()

    assert trace.events[0]["result"]["shape"] == [3]
    assert "value" not in trace.events[0]["result"]
