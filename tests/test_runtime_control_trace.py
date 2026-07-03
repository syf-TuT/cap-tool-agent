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
