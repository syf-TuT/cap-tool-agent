import json
import os
import queue as stdlib_queue
import threading
import time
from types import SimpleNamespace

import pytest
import requests

from capx.llm.client import (
    _call_with_deadline,
    query_model,
    query_model_ensemble,
    query_model_streaming,
    query_single_model_ensemble,
)
from capx.llm.context import TelemetryWriteError, trial_llm_context
from capx.llm.errors import LLMErrorKind, LLMQueryError


class _Clock:
    def __init__(self, value=100.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class _NonStreamingResponse:
    def __init__(self, status_code=200, body=None, *, chunks=None, headers=None, clock=None):
        self.status_code = status_code
        self.headers = headers or {}
        if body is None:
            body = {"choices": [{"message": {"content": "ok"}}]}
        encoded = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.chunks = [(0, encoded)] if chunks is None else chunks
        self.clock = clock
        self.closed = False
        self.iterated = False

    def iter_content(self, chunk_size=8192):
        del chunk_size
        self.iterated = True
        for delay, chunk in self.chunks:
            if self.clock is not None:
                self.clock.advance(delay)
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk

    def close(self):
        self.closed = True


def _args(**overrides):
    values = {
        "model": "deepseek-v4-flash",
        "server_url": "https://example.invalid/v1/chat/completions",
        "api_key": "test-key",
        "temperature": 0.2,
        "max_tokens": 256,
        "reasoning_effort": "minimal",
        "debug": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _non_streaming_policy(monkeypatch, *, attempts=2, timeout=60, backoff=0):
    monkeypatch.delenv("CAPX_FORCE_STREAMING_CHAT_COMPLETIONS", raising=False)
    monkeypatch.setenv("CAPX_LLM_MAX_ATTEMPTS", str(attempts))
    monkeypatch.setenv("CAPX_LLM_REQUEST_TIMEOUT_SECONDS", str(timeout))
    monkeypatch.setenv("CAPX_LLM_RETRY_BACKOFF_SECONDS", str(backoff))
    monkeypatch.setenv("CAPX_DISABLE_REASONING", "1")


class _StreamingResponse:
    headers = {"content-type": "text/event-stream"}

    def __init__(self):
        self.status_code = 200
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "hidden thought"}}]},
            {"choices": [{"delta": {"content": "answer"}}]},
        ]
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}".encode()
        yield b"data: [DONE]"

    def close(self):
        self.closed = True


class _ScriptedStreamingResponse:
    def __init__(self, *, status_code=200, lines=(), clock=None, headers=None):
        self.status_code = status_code
        self.headers = headers or {"content-type": "text/event-stream"}
        self.lines = list(lines)
        self.clock = clock
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def raise_for_status(self):
        if not 200 <= self.status_code < 300:
            error = requests.exceptions.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error

    def iter_lines(self):
        for delay, line in self.lines:
            if self.clock is not None:
                self.clock.advance(delay)
            if isinstance(line, BaseException):
                raise line
            yield line

    def close(self):
        self.closed = True


class _BlockingStreamingResponse(_ScriptedStreamingResponse):
    def __init__(self, *, delayed_line=None, delay_seconds=None):
        super().__init__()
        self.delayed_line = delayed_line
        self.delay_seconds = delay_seconds
        self.released = threading.Event()

    def iter_lines(self):
        if self.delay_seconds is None:
            self.released.wait(timeout=1)
        else:
            time.sleep(self.delay_seconds)
        if self.delayed_line is not None:
            yield self.delayed_line

    def close(self):
        self.closed = True
        self.released.set()


class _DripStreamingResponse(_ScriptedStreamingResponse):
    def iter_lines(self):
        yield _sse({"content": "partial"})
        for index in range(6):
            time.sleep(0.015)
            if index % 2:
                yield _sse({"content": " drip"})
            else:
                yield b": heartbeat"


class _SlowJSONStreamingResponse(_ScriptedStreamingResponse):
    def __init__(self, *, delay_seconds, content):
        super().__init__(headers={"content-type": "application/json"})
        self.delay_seconds = delay_seconds
        self.content = content

    def json(self):
        time.sleep(self.delay_seconds)
        return {"choices": [{"message": {"content": self.content}}]}


def _sse(delta):
    return f"data: {json.dumps({'choices': [{'delta': delta}]})}".encode()


def _streaming_policy(monkeypatch, *, attempts=2, timeout=60, first_content=5, backoff=0):
    monkeypatch.setenv("CAPX_FORCE_STREAMING_CHAT_COMPLETIONS", "1")
    monkeypatch.setenv("CAPX_LLM_MAX_ATTEMPTS", str(attempts))
    monkeypatch.setenv("CAPX_LLM_REQUEST_TIMEOUT_SECONDS", str(timeout))
    monkeypatch.setenv("CAPX_STREAMING_FIRST_CONTENT_TIMEOUT_SECONDS", str(first_content))
    monkeypatch.setenv("CAPX_LLM_RETRY_BACKOFF_SECONDS", str(backoff))
    monkeypatch.setenv("CAPX_DISABLE_REASONING", "1")
    monkeypatch.setattr("capx.llm.client.random.uniform", lambda *args: 0.0)


def test_streaming_query_drops_reasoning_when_disabled(monkeypatch):
    captured_payloads = []

    def fake_post(*args, data, **kwargs):
        captured_payloads.append(json.loads(data))
        return _StreamingResponse()

    monkeypatch.setenv("CAPX_DISABLE_REASONING", "1")
    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)

    args = SimpleNamespace(
        model="deepseek-v4-flash",
        server_url="https://example.invalid/v1/chat/completions",
        api_key="test-key",
        temperature=0.2,
        max_tokens=256,
        reasoning_effort="minimal",
    )
    chunks = list(query_model_streaming(args, [{"role": "user", "content": "hi"}]))

    assert captured_payloads[0]["thinking"] == {"type": "disabled"}
    assert chunks[-1] == {"type": "done", "content": "answer", "reasoning": None}


def test_first_content_timeout_falls_back_to_non_streaming_once(monkeypatch):
    _streaming_policy(monkeypatch, first_content=1)
    clock = _Clock()
    streaming_response = _ScriptedStreamingResponse(
        lines=[(1.1, _sse({"reasoning_content": "still thinking"}))], clock=clock
    )
    responses = [
        streaming_response,
        _NonStreamingResponse(body={"choices": [{"message": {"content": "fallback"}}]}),
    ]
    payloads = []

    def fake_post(*args, data, **kwargs):
        payloads.append(json.loads(data))
        return responses.pop(0)

    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)
    monkeypatch.setattr("capx.llm.client.time.monotonic", clock)

    result = query_model(_args(), [{"role": "user", "content": "hi"}])

    assert result["content"] == "fallback"
    assert [payload.get("stream", False) for payload in payloads] == [True, False]
    assert len(payloads) == 2
    assert streaming_response.closed is True


def test_empty_stream_falls_back_to_non_streaming_once(monkeypatch):
    _streaming_policy(monkeypatch)
    responses = [
        _ScriptedStreamingResponse(lines=[(0, b"data: [DONE]")]),
        _NonStreamingResponse(body={"choices": [{"message": {"content": "fallback"}}]}),
    ]
    payloads = []

    def fake_post(*args, data, **kwargs):
        payloads.append(json.loads(data))
        return responses.pop(0)

    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)

    assert query_model(_args(), [{"role": "user", "content": "hi"}])["content"] == "fallback"
    assert [payload.get("stream", False) for payload in payloads] == [True, False]


@pytest.mark.parametrize(
    "interruption",
    [
        requests.exceptions.ReadTimeout("stream stalled"),
        requests.exceptions.ConnectionError("stream reset"),
    ],
)
def test_partial_stream_disconnect_discards_partial_and_falls_back(monkeypatch, interruption):
    _streaming_policy(monkeypatch)
    streaming_response = _ScriptedStreamingResponse(
        lines=[(0, _sse({"content": "partial"})), (0, interruption)]
    )
    responses = [
        streaming_response,
        _NonStreamingResponse(body={"choices": [{"message": {"content": "replacement"}}]}),
    ]
    payloads = []

    def fake_post(*args, data, **kwargs):
        payloads.append(json.loads(data))
        return responses.pop(0)

    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)

    result = query_model(_args(), [{"role": "user", "content": "hi"}])

    assert result["content"] == "replacement"
    assert "partial" not in result["content"]
    assert [payload.get("stream", False) for payload in payloads] == [True, False]
    assert streaming_response.closed is True


def test_query_model_retries_non_streaming_timeout_once(monkeypatch):
    request_timeouts = []

    class NonStreamingResponse:
        status_code = 200
        headers = {}

        def iter_content(self, chunk_size=8192):
            del chunk_size
            yield json.dumps({"choices": [{"message": {"content": "retry ok"}}]}).encode()

        def close(self):
            return None

    def fake_post(*args, timeout, **kwargs):
        request_timeouts.append(timeout)
        if len(request_timeouts) == 1:
            raise requests.exceptions.Timeout("request timed out")
        return NonStreamingResponse()

    monkeypatch.delenv("CAPX_FORCE_STREAMING_CHAT_COMPLETIONS", raising=False)
    monkeypatch.setenv("CAPX_DISABLE_REASONING", "1")
    monkeypatch.setenv("CAPX_NONSTREAMING_REQUEST_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("CAPX_NONSTREAMING_REQUEST_RETRIES", "1")
    monkeypatch.setenv("CAPX_LLM_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)

    args = SimpleNamespace(
        model="deepseek-v4-flash",
        server_url="https://example.invalid/v1/chat/completions",
        api_key="test-key",
        temperature=0.2,
        max_tokens=256,
        reasoning_effort="minimal",
        debug=False,
    )

    assert query_model(args, [{"role": "user", "content": "hi"}]) == {
        "content": "retry ok",
        "reasoning": None,
    }
    assert request_timeouts == [60.0, 60.0]


def test_streaming_query_times_out_before_first_content(monkeypatch):
    clock = _Clock(0.0)
    response = _ScriptedStreamingResponse(
        lines=[
            (0.4, _sse({"reasoning_content": "thinking"})),
            (0.4, _sse({"reasoning_content": "thinking"})),
            (0.4, _sse({"reasoning_content": "thinking"})),
        ],
        clock=clock,
    )

    def fake_post(*args, data, **kwargs):
        return response

    monkeypatch.setenv("CAPX_DISABLE_REASONING", "1")
    monkeypatch.setenv("CAPX_STREAMING_FIRST_CONTENT_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)
    monkeypatch.setattr("capx.llm.client.time.monotonic", clock)

    args = SimpleNamespace(
        model="deepseek-v4-flash",
        server_url="https://example.invalid/v1/chat/completions",
        api_key="test-key",
        temperature=0.2,
        max_tokens=256,
        reasoning_effort="minimal",
    )

    with pytest.raises(LLMQueryError) as raised:
        list(query_model_streaming(args, [{"role": "user", "content": "hi"}]))

    assert raised.value.kind is LLMErrorKind.NO_CONTENT


def test_streaming_503_retries_in_streaming_mode(monkeypatch):
    _streaming_policy(monkeypatch)
    responses = [
        _ScriptedStreamingResponse(status_code=503),
        _ScriptedStreamingResponse(
            lines=[(0, _sse({"content": "ok"})), (0, b"data: [DONE]")]
        ),
    ]
    payloads = []

    def fake_post(*args, data, **kwargs):
        payloads.append(json.loads(data))
        return responses.pop(0)

    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)
    monkeypatch.setattr("capx.llm.client.random.uniform", lambda *args: 0.0)

    assert query_model(_args(), [{"role": "user", "content": "hi"}])["content"] == "ok"
    assert [payload["stream"] for payload in payloads] == [True, True]


def test_streaming_503_twice_raises_typed_http_5xx(monkeypatch):
    _streaming_policy(monkeypatch)
    all_responses = [
        _ScriptedStreamingResponse(status_code=503),
        _ScriptedStreamingResponse(status_code=503),
    ]
    responses = list(all_responses)
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)
    monkeypatch.setattr("capx.llm.client.random.uniform", lambda *args: 0.0)

    with pytest.raises(LLMQueryError) as raised:
        query_model(_args(), [{"role": "user", "content": "hi"}])

    assert raised.value.kind is LLMErrorKind.HTTP_5XX
    assert raised.value.attempt == 2
    assert raised.value.status_code == 503
    assert len(calls) == 2
    assert all(response.closed for response in all_responses)


@pytest.mark.parametrize("status", [401, 404])
def test_streaming_non_retryable_http_error_uses_one_attempt(monkeypatch, status):
    _streaming_policy(monkeypatch)
    response = _ScriptedStreamingResponse(status_code=status)
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        return response

    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)

    with pytest.raises(LLMQueryError) as raised:
        query_model(_args(), [{"role": "user", "content": "hi"}])

    assert raised.value.kind is (
        LLMErrorKind.AUTH_ERROR if status == 401 else LLMErrorKind.REQUEST_REJECTED
    )
    assert raised.value.attempt == 1
    assert len(calls) == 1
    assert response.closed is True


def test_streaming_failure_with_no_budget_does_not_fallback(monkeypatch, tmp_path):
    _streaming_policy(monkeypatch)
    clock = _Clock()
    response = _ScriptedStreamingResponse(lines=[(0, b"data: [DONE]")], clock=clock)
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        return response

    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)
    monkeypatch.setattr("capx.llm.client.time.monotonic", clock)
    with trial_llm_context(
        trial=1,
        deadline_monotonic=clock() + 4.5,
        telemetry_path=tmp_path / "calls.jsonl",
        monotonic=clock,
    ):
        with pytest.raises(LLMQueryError) as raised:
            query_model(_args(), [{"role": "user", "content": "hi"}])

    assert raised.value.kind is LLMErrorKind.TRIAL_BUDGET_EXHAUSTED
    assert len(calls) == 1


def test_streaming_telemetry_shares_call_index_and_tracks_mode_transition(
    monkeypatch, tmp_path
):
    _streaming_policy(monkeypatch)
    clock = _Clock()
    responses = [
        _ScriptedStreamingResponse(
            lines=[
                (0.2, b": heartbeat"),
                (0.3, _sse({"content": "partial"})),
                (0.1, requests.exceptions.ReadTimeout("stalled")),
            ],
            clock=clock,
        ),
        _NonStreamingResponse(
            body={"choices": [{"message": {"content": "replacement"}}]},
            chunks=[
                (
                    0.4,
                    json.dumps(
                        {"choices": [{"message": {"content": "replacement"}}]}
                    ).encode(),
                )
            ],
            clock=clock,
        ),
    ]
    monkeypatch.setattr("capx.llm.client.requests.post", lambda *args, **kwargs: responses.pop(0))
    monkeypatch.setattr("capx.llm.client.time.monotonic", clock)
    telemetry = tmp_path / "calls.jsonl"

    with trial_llm_context(
        trial=4,
        deadline_monotonic=clock() + 30,
        telemetry_path=telemetry,
        monotonic=clock,
    ):
        query_model(_args(), [{"role": "user", "content": "secret prompt"}])

    records = [json.loads(line) for line in telemetry.read_text(encoding="utf-8").splitlines()]
    assert [record["call_index"] for record in records] == [1, 1]
    assert [record["attempt"] for record in records] == [1, 2]
    assert [record["mode"] for record in records] == ["streaming", "non_streaming"]
    assert records[0]["ttfb_ms"] == 200
    assert records[0]["first_content_ms"] == 500
    assert records[0]["retry_scheduled"] is True
    assert records[1]["outcome"] == "success"
    assert "secret prompt" not in telemetry.read_text(encoding="utf-8")


def test_streaming_heartbeat_lines_cannot_bypass_first_content_watchdog(monkeypatch):
    _streaming_policy(monkeypatch, attempts=1, first_content=1)
    clock = _Clock()
    response = _ScriptedStreamingResponse(
        lines=[(0.4, b": heartbeat"), (0.4, b""), (0.4, b'data: {"choices": []}')],
        clock=clock,
    )
    monkeypatch.setattr("capx.llm.client.requests.post", lambda *args, **kwargs: response)
    monkeypatch.setattr("capx.llm.client.time.monotonic", clock)

    with pytest.raises(LLMQueryError) as raised:
        query_model(_args(), [{"role": "user", "content": "hi"}])

    assert raised.value.kind is LLMErrorKind.NO_CONTENT
    assert response.closed is True


def test_completely_silent_stream_hits_first_content_deadline_and_falls_back(monkeypatch):
    _streaming_policy(monkeypatch, first_content=0.03)
    streaming_response = _BlockingStreamingResponse()
    responses = [streaming_response, _NonStreamingResponse()]
    payloads = []

    def fake_post(*args, data, **kwargs):
        payloads.append(json.loads(data))
        return responses.pop(0)

    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)

    started = time.monotonic()
    result = query_model(_args(), [{"role": "user", "content": "hi"}])
    elapsed = time.monotonic() - started

    assert result["content"] == "ok"
    assert elapsed < 0.5
    assert [payload.get("stream", False) for payload in payloads] == [True, False]
    assert streaming_response.closed is True


def test_content_arriving_after_first_content_deadline_is_not_yielded(monkeypatch):
    _streaming_policy(monkeypatch, first_content=0.02)
    streaming_response = _BlockingStreamingResponse(
        delayed_line=_sse({"content": "too late"}), delay_seconds=0.05
    )
    responses = [streaming_response, _NonStreamingResponse()]

    monkeypatch.setattr(
        "capx.llm.client.requests.post", lambda *args, **kwargs: responses.pop(0)
    )

    result = query_model(_args(), [{"role": "user", "content": "hi"}])

    assert result["content"] == "ok"
    assert "too late" not in result["content"]
    assert streaming_response.closed is True


def test_continuous_stream_data_cannot_exceed_total_attempt_budget(monkeypatch):
    _streaming_policy(monkeypatch, timeout=0.05, first_content=0.02)
    streaming_response = _DripStreamingResponse()
    responses = [streaming_response, _NonStreamingResponse()]

    monkeypatch.setattr(
        "capx.llm.client.requests.post", lambda *args, **kwargs: responses.pop(0)
    )

    result = query_model(_args(), [{"role": "user", "content": "hi"}])

    assert result["content"] == "ok"
    assert "partial" not in result["content"]
    assert streaming_response.closed is True


def test_attempt_deadline_before_first_content_deadline_retries_streaming(monkeypatch):
    _streaming_policy(monkeypatch, timeout=0.03, first_content=0.1)
    first = _BlockingStreamingResponse()
    second = _ScriptedStreamingResponse(
        lines=[(0, _sse({"content": "ok"})), (0, b"data: [DONE]")]
    )
    responses = [first, second]
    payloads = []

    def fake_post(*args, data, **kwargs):
        payloads.append(json.loads(data))
        return responses.pop(0)

    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)

    result = query_model(_args(), [{"role": "user", "content": "hi"}])

    assert result["content"] == "ok"
    assert [payload["stream"] for payload in payloads] == [True, True]
    assert first.closed is True


def test_slow_response_headers_cannot_exceed_attempt_deadline(monkeypatch):
    _streaming_policy(monkeypatch, timeout=0.03, first_content=0.1)
    late_response = _ScriptedStreamingResponse(
        lines=[(0, _sse({"content": "late"})), (0, b"data: [DONE]")]
    )
    retry_response = _ScriptedStreamingResponse(
        lines=[(0, _sse({"content": "ok"})), (0, b"data: [DONE]")]
    )
    calls = 0
    lock = threading.Lock()

    def fake_post(*args, **kwargs):
        nonlocal calls
        with lock:
            calls += 1
            call = calls
        if call == 1:
            time.sleep(0.08)
            return late_response
        return retry_response

    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)

    started = time.monotonic()
    result = query_model(_args(), [{"role": "user", "content": "hi"}])
    elapsed = time.monotonic() - started
    time.sleep(0.07)

    assert result["content"] == "ok"
    assert elapsed < 0.07
    assert calls == 2
    assert late_response.closed is True


def test_slow_json_body_cannot_succeed_after_attempt_deadline(monkeypatch):
    _streaming_policy(monkeypatch, timeout=0.03, first_content=0.1)
    late_json = _SlowJSONStreamingResponse(delay_seconds=0.08, content="late")
    retry_response = _ScriptedStreamingResponse(
        lines=[(0, _sse({"content": "ok"})), (0, b"data: [DONE]")]
    )
    responses = [late_json, retry_response]

    monkeypatch.setattr(
        "capx.llm.client.requests.post", lambda *args, **kwargs: responses.pop(0)
    )

    started = time.monotonic()
    result = query_model(_args(), [{"role": "user", "content": "hi"}])
    elapsed = time.monotonic() - started

    assert result["content"] == "ok"
    assert elapsed < 0.07
    assert late_json.closed is True


def test_deadline_cancellation_race_reclaims_queued_response(monkeypatch):
    queue_class = stdlib_queue.Queue
    result_queue = queue_class(maxsize=1)
    put_started = threading.Event()
    allow_put = threading.Event()
    original_put = result_queue.put_nowait

    def blocked_put(item):
        put_started.set()
        allow_put.wait(timeout=1)
        original_put(item)

    monkeypatch.setattr(result_queue, "put_nowait", blocked_put)
    monkeypatch.setattr(
        "capx.llm.client.queue.Queue", lambda *args, **kwargs: result_queue
    )
    response = _ScriptedStreamingResponse()
    errors = []

    def invoke():
        try:
            _call_with_deadline(
                lambda: response,
                deadline=time.monotonic() + 0.02,
                timeout_error=requests.exceptions.ReadTimeout("deadline"),
                on_late_result=lambda late_response: late_response.close(),
            )
        except requests.exceptions.ReadTimeout:
            return
        except Exception as error:
            errors.append(error)

    caller = threading.Thread(target=invoke)
    caller.start()
    assert put_started.wait(timeout=0.5)
    time.sleep(0.04)
    allow_put.set()
    caller.join(timeout=0.5)

    assert not errors
    assert not caller.is_alive()
    assert response.closed is True


def test_direct_streaming_generator_remains_single_attempt_and_event_compatible(monkeypatch):
    response = _ScriptedStreamingResponse(
        lines=[
            (0, _sse({"reasoning_content": "thought"})),
            (0, _sse({"content": "answer"})),
            (0, b"data: [DONE]"),
        ]
    )
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        return response

    monkeypatch.delenv("CAPX_DISABLE_REASONING", raising=False)
    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)

    events = list(query_model_streaming(_args(), [{"role": "user", "content": "hi"}]))

    assert [event["type"] for event in events] == ["reasoning_delta", "content_delta", "done"]
    assert len(calls) == 1
    assert response.closed is True


def test_direct_streaming_records_trial_telemetry(monkeypatch, tmp_path):
    _streaming_policy(monkeypatch, attempts=1)
    response = _ScriptedStreamingResponse(
        lines=[(0, _sse({"content": "answer"})), (0, b"data: [DONE]")]
    )
    monkeypatch.setattr("capx.llm.client.requests.post", lambda *args, **kwargs: response)
    telemetry = tmp_path / "calls.jsonl"

    with trial_llm_context(trial=9, telemetry_path=telemetry):
        assert list(query_model_streaming(_args(), [{"role": "user", "content": "hi"}]))[-1][
            "content"
        ] == "answer"

    records = [json.loads(line) for line in telemetry.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["call_index"] == 1
    assert records[0]["attempt"] == 1
    assert records[0]["mode"] == "streaming"
    assert records[0]["outcome"] == "success"


def test_streaming_eof_after_partial_content_falls_back_once(monkeypatch):
    _streaming_policy(monkeypatch)
    partial = _ScriptedStreamingResponse(lines=[(0, _sse({"content": "partial"}))])
    fallback = _NonStreamingResponse(body={"choices": [{"message": {"content": "replacement"}}]})
    responses = [partial, fallback]
    payloads = []

    def fake_post(*args, data, **kwargs):
        payloads.append(json.loads(data))
        return responses.pop(0)

    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)

    result = query_model(_args(), [{"role": "user", "content": "hi"}])

    assert result["content"] == "replacement"
    assert [payload.get("stream", False) for payload in payloads] == [True, False]


def test_non_streaming_slow_body_cannot_exceed_attempt_deadline(monkeypatch):
    _non_streaming_policy(monkeypatch, timeout=0.03, backoff=0)

    class SlowResponse(_NonStreamingResponse):
        def iter_content(self, chunk_size=8192):
            del chunk_size
            time.sleep(0.08)
            yield json.dumps({"choices": [{"message": {"content": "late"}}]}).encode()

    slow = SlowResponse()
    fallback = _NonStreamingResponse(body={"choices": [{"message": {"content": "ok"}}]})
    responses = [slow, fallback]
    monkeypatch.setattr("capx.llm.client.requests.post", lambda *args, **kwargs: responses.pop(0))
    monkeypatch.setattr("capx.llm.client.random.uniform", lambda *_args: 0.0)

    started = time.monotonic()
    result = query_model(_args(), [{"role": "user", "content": "hi"}])

    assert result["content"] == "ok"
    assert time.monotonic() - started < 0.07
    assert slow.closed is True


@pytest.mark.parametrize("ensemble", [query_model_ensemble, query_single_model_ensemble])
def test_all_llm_failed_ensemble_reraises_typed_error(monkeypatch, ensemble):
    error = LLMQueryError(
        kind=LLMErrorKind.HTTP_5XX,
        call_index=3,
        attempt=2,
        status_code=503,
        elapsed_seconds=1.0,
        message="provider unavailable",
    )
    monkeypatch.setattr(
        "capx.llm.client.query_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr("capx.llm.client.ENSEMBLE_CONFIGS", [("model", [0.1])])

    with pytest.raises(LLMQueryError) as raised:
        if ensemble is query_single_model_ensemble:
            ensemble(_args(), [{"role": "user", "content": "hi"}], model="model")
        else:
            ensemble(_args(), [{"role": "user", "content": "hi"}])

    assert raised.value is error


def test_non_streaming_503_then_200_retries_once_and_succeeds(monkeypatch):
    _non_streaming_policy(monkeypatch)
    responses = [_NonStreamingResponse(503), _NonStreamingResponse(200)]
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)
    monkeypatch.setattr("capx.llm.client.random.uniform", lambda *args: 0.0)

    assert query_model(_args(), [{"role": "user", "content": "hi"}]) == {
        "content": "ok",
        "reasoning": None,
    }
    assert len(calls) == 2
    assert all(call["stream"] is True for call in calls)


def test_non_streaming_503_twice_raises_typed_http_5xx(monkeypatch):
    _non_streaming_policy(monkeypatch)
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        return _NonStreamingResponse(503)

    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)
    monkeypatch.setattr("capx.llm.client.random.uniform", lambda *args: 0.0)

    with pytest.raises(LLMQueryError) as raised:
        query_model(_args(), [{"role": "user", "content": "hi"}])

    assert raised.value.kind is LLMErrorKind.HTTP_5XX
    assert raised.value.attempt == 2
    assert raised.value.status_code == 503
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, LLMErrorKind.REQUEST_REJECTED),
        (401, LLMErrorKind.AUTH_ERROR),
        (403, LLMErrorKind.AUTH_ERROR),
        (404, LLMErrorKind.REQUEST_REJECTED),
    ],
)
def test_non_streaming_rejected_4xx_does_not_retry(monkeypatch, status, kind):
    _non_streaming_policy(monkeypatch)
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        return _NonStreamingResponse(status)

    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)

    with pytest.raises(LLMQueryError) as raised:
        query_model(_args(), [{"role": "user", "content": "hi"}])

    assert raised.value.kind is kind
    assert raised.value.attempt == 1
    assert raised.value.status_code == status
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("exception", "expected_kind"),
    [
        (requests.exceptions.ConnectTimeout("connect timed out"), LLMErrorKind.CONNECT_TIMEOUT),
        (requests.exceptions.ReadTimeout("read timed out"), LLMErrorKind.READ_TIMEOUT),
        (requests.exceptions.Timeout("request timed out"), LLMErrorKind.READ_TIMEOUT),
    ],
)
def test_non_streaming_timeout_retries_once_then_succeeds(
    monkeypatch, tmp_path, exception, expected_kind
):
    _non_streaming_policy(monkeypatch)
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise exception
        return _NonStreamingResponse(200)

    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)
    monkeypatch.setattr("capx.llm.client.random.uniform", lambda *args: 0.0)

    telemetry = tmp_path / "calls.jsonl"
    with trial_llm_context(trial=1, telemetry_path=telemetry):
        assert query_model(_args(), [{"role": "user", "content": "hi"}])["content"] == "ok"
    assert len(calls) == 2
    records = [json.loads(line) for line in telemetry.read_text(encoding="utf-8").splitlines()]
    assert records[0]["error_kind"] == expected_kind.value
    assert records[0]["retry_scheduled"] is True


def test_non_streaming_connection_error_is_typed_after_final_attempt(monkeypatch):
    _non_streaming_policy(monkeypatch)
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        raise requests.exceptions.ConnectionError("connection reset")

    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)
    monkeypatch.setattr("capx.llm.client.random.uniform", lambda *args: 0.0)

    with pytest.raises(LLMQueryError) as raised:
        query_model(_args(), [{"role": "user", "content": "hi"}])

    assert raised.value.kind is LLMErrorKind.CONNECTION_ERROR
    assert raised.value.attempt == 2
    assert len(calls) == 2


def test_non_streaming_429_retry_after_is_capped(monkeypatch):
    _non_streaming_policy(monkeypatch)
    responses = [
        _NonStreamingResponse(429, headers={"Retry-After": "90"}),
        _NonStreamingResponse(200),
    ]
    sleeps = []
    monkeypatch.setattr("capx.llm.client.requests.post", lambda *args, **kwargs: responses.pop(0))
    monkeypatch.setattr("capx.llm.client.time.sleep", sleeps.append)
    monkeypatch.setattr("capx.llm.client.random.uniform", lambda *args: 0.0)

    assert query_model(_args(), [{"role": "user", "content": "hi"}])["content"] == "ok"
    assert sleeps == [10.0]


def test_non_streaming_low_trial_budget_suppresses_second_attempt(monkeypatch, tmp_path):
    _non_streaming_policy(monkeypatch)
    clock = _Clock()
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        return _NonStreamingResponse(503, clock=clock)

    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)
    monkeypatch.setattr("capx.llm.client.time.monotonic", clock)
    with trial_llm_context(
        trial=1,
        deadline_monotonic=clock() + 4.5,
        telemetry_path=tmp_path / "calls.jsonl",
        monotonic=clock,
    ):
        with pytest.raises(LLMQueryError) as raised:
            query_model(_args(), [{"role": "user", "content": "hi"}])

    assert raised.value.kind is LLMErrorKind.TRIAL_BUDGET_EXHAUSTED
    assert raised.value.attempt == 1
    assert len(calls) == 1


def test_non_streaming_request_timeout_is_capped_by_trial_budget(monkeypatch):
    _non_streaming_policy(monkeypatch, attempts=1, timeout=60)
    clock = _Clock()
    timeouts = []

    def fake_post(*args, timeout, **kwargs):
        timeouts.append(timeout)
        return _NonStreamingResponse(200, clock=clock)

    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)
    monkeypatch.setattr("capx.llm.client.time.monotonic", clock)
    with trial_llm_context(trial=1, deadline_monotonic=clock() + 7.25, monotonic=clock):
        query_model(_args(), [{"role": "user", "content": "hi"}])

    assert timeouts == [pytest.approx(7.25)]


def test_non_streaming_ttfb_uses_first_non_empty_body_chunk(monkeypatch, tmp_path):
    _non_streaming_policy(monkeypatch, attempts=1)
    clock = _Clock()
    response = _NonStreamingResponse(
        200,
        chunks=[
            (0.2, b""),
            (0.3, b'{"choices":[{"message":{"content":"ok"}}]}'),
        ],
        clock=clock,
    )
    monkeypatch.setattr("capx.llm.client.requests.post", lambda *args, **kwargs: response)
    monkeypatch.setattr("capx.llm.client.time.monotonic", clock)
    telemetry = tmp_path / "calls.jsonl"

    with trial_llm_context(
        trial=1,
        deadline_monotonic=clock() + 30,
        telemetry_path=telemetry,
        monotonic=clock,
    ):
        query_model(_args(), [{"role": "user", "content": "hi"}])

    record = json.loads(telemetry.read_text(encoding="utf-8"))
    assert record["ttfb_ms"] == 500
    assert record["duration_ms"] == 500


def test_non_streaming_attempt_telemetry_shares_call_index_and_is_safe(monkeypatch, tmp_path):
    _non_streaming_policy(monkeypatch)
    clock = _Clock()
    responses = [
        _NonStreamingResponse(503, clock=clock),
        _NonStreamingResponse(200, clock=clock),
    ]
    monkeypatch.setattr("capx.llm.client.requests.post", lambda *args, **kwargs: responses.pop(0))
    monkeypatch.setattr("capx.llm.client.time.monotonic", clock)
    monkeypatch.setattr("capx.llm.client.random.uniform", lambda *args: 0.0)
    telemetry = tmp_path / "calls.jsonl"

    with trial_llm_context(
        trial=9,
        deadline_monotonic=clock() + 30,
        telemetry_path=telemetry,
        monotonic=clock,
    ):
        query_model(_args(api_key="super-secret"), [{"role": "user", "content": "private"}])

    records = [json.loads(line) for line in telemetry.read_text(encoding="utf-8").splitlines()]
    assert [record["attempt"] for record in records] == [1, 2]
    assert records[0]["call_index"] == records[1]["call_index"] == 1
    assert records[0]["http_status"] == 503
    assert records[0]["outcome"] == "retryable_http_error"
    assert records[0]["error_kind"] == "http_5xx"
    assert records[0]["retry_scheduled"] is True
    assert records[0]["ttfb_ms"] is None
    assert records[0]["duration_ms"] == 0
    assert records[0]["trial_remaining_ms_before"] == 30_000
    assert records[0]["trial_remaining_ms_after"] == 30_000
    assert records[1]["outcome"] == "success"
    assert records[1]["error_kind"] is None
    assert records[1]["retry_scheduled"] is False
    encoded = json.dumps(records)
    assert "super-secret" not in encoded
    assert "private" not in encoded
    assert "authorization" not in encoded.lower()
    assert "response" not in encoded.lower()
    assert "payload" not in encoded.lower()


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": None}}]},
    ],
)
def test_non_streaming_invalid_response_is_typed_and_not_retried(monkeypatch, body):
    _non_streaming_policy(monkeypatch)
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        return _NonStreamingResponse(200, body=body)

    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)

    with pytest.raises(LLMQueryError) as raised:
        query_model(_args(), [{"role": "user", "content": "hi"}])

    assert raised.value.kind is LLMErrorKind.INVALID_RESPONSE
    assert raised.value.attempt == 1
    assert len(calls) == 1


def test_non_streaming_without_context_uses_safe_call_index(monkeypatch):
    _non_streaming_policy(monkeypatch, attempts=1)
    monkeypatch.setattr(
        "capx.llm.client.requests.post", lambda *args, **kwargs: _NonStreamingResponse(503)
    )

    with pytest.raises(LLMQueryError) as raised:
        query_model(_args(), [{"role": "user", "content": "hi"}])

    assert raised.value.call_index == 1


def test_non_streaming_telemetry_write_failure_is_terminal(monkeypatch):
    _non_streaming_policy(monkeypatch)
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        return _NonStreamingResponse(503)

    def fail_telemetry(**kwargs):
        del kwargs
        raise TelemetryWriteError()

    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)
    with trial_llm_context(trial=1) as context:
        monkeypatch.setattr(context, "record_attempt", fail_telemetry)
        with pytest.raises(TelemetryWriteError):
            query_model(_args(), [{"role": "user", "content": "hi"}])

    assert len(calls) == 1


def test_non_streaming_closes_response_after_success(monkeypatch):
    _non_streaming_policy(monkeypatch, attempts=1)
    response = _NonStreamingResponse(200)
    monkeypatch.setattr("capx.llm.client.requests.post", lambda *args, **kwargs: response)

    query_model(_args(), [{"role": "user", "content": "hi"}])

    assert response.closed is True
    assert response.iterated is True


def test_non_streaming_closes_each_response_without_reading_http_error_body(monkeypatch):
    _non_streaming_policy(monkeypatch)
    responses = [_NonStreamingResponse(503), _NonStreamingResponse(200)]
    monkeypatch.setattr("capx.llm.client.requests.post", lambda *args, **kwargs: responses.pop(0))
    monkeypatch.setattr("capx.llm.client.random.uniform", lambda *args: 0.0)
    first, second = responses

    query_model(_args(), [{"role": "user", "content": "hi"}])

    assert first.closed is True
    assert first.iterated is False
    assert second.closed is True
    assert second.iterated is True


def test_non_streaming_closes_terminal_http_error_without_reading_body(monkeypatch):
    _non_streaming_policy(monkeypatch)
    response = _NonStreamingResponse(401)
    monkeypatch.setattr("capx.llm.client.requests.post", lambda *args, **kwargs: response)

    with pytest.raises(LLMQueryError):
        query_model(_args(), [{"role": "user", "content": "hi"}])

    assert response.closed is True
    assert response.iterated is False


def test_non_streaming_closes_response_after_invalid_json(monkeypatch):
    _non_streaming_policy(monkeypatch)
    response = _NonStreamingResponse(200, body=b"not json")
    monkeypatch.setattr("capx.llm.client.requests.post", lambda *args, **kwargs: response)

    with pytest.raises(LLMQueryError):
        query_model(_args(), [{"role": "user", "content": "hi"}])

    assert response.closed is True


def test_non_streaming_closes_response_when_body_iteration_times_out(monkeypatch):
    _non_streaming_policy(monkeypatch, attempts=1)
    response = _NonStreamingResponse(
        200,
        chunks=[(0, requests.exceptions.ReadTimeout("body timed out"))],
    )
    monkeypatch.setattr("capx.llm.client.requests.post", lambda *args, **kwargs: response)

    with pytest.raises(LLMQueryError) as raised:
        query_model(_args(), [{"role": "user", "content": "hi"}])

    assert raised.value.kind is LLMErrorKind.READ_TIMEOUT
    assert response.closed is True


def test_non_streaming_retry_budget_equality_reserves_delay_and_minimum_attempt(
    monkeypatch,
):
    _non_streaming_policy(monkeypatch, backoff=1)
    clock = _Clock()
    responses = [_NonStreamingResponse(503), _NonStreamingResponse(200)]
    timeouts = []

    def fake_post(*args, timeout, **kwargs):
        timeouts.append(timeout)
        return responses.pop(0)

    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)
    monkeypatch.setattr("capx.llm.client.random.uniform", lambda *args: 0.5)
    monkeypatch.setattr("capx.llm.client.time.monotonic", clock)
    monkeypatch.setattr("capx.llm.client.time.sleep", clock.advance)
    with trial_llm_context(
        trial=1,
        deadline_monotonic=clock() + 6.5,
        monotonic=clock,
    ):
        query_model(_args(), [{"role": "user", "content": "hi"}])

    assert timeouts == [pytest.approx(6.5), pytest.approx(5.0)]


def test_non_streaming_retry_after_requires_delay_plus_minimum_attempt_budget(monkeypatch):
    _non_streaming_policy(monkeypatch)
    clock = _Clock()
    response = _NonStreamingResponse(429, headers={"Retry-After": "90"})
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        return response

    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)
    monkeypatch.setattr("capx.llm.client.random.uniform", lambda *args: 0.0)
    monkeypatch.setattr("capx.llm.client.time.monotonic", clock)
    with trial_llm_context(
        trial=1,
        deadline_monotonic=clock() + 14.999,
        monotonic=clock,
    ):
        with pytest.raises(LLMQueryError) as raised:
            query_model(_args(), [{"role": "user", "content": "hi"}])

    assert raised.value.kind is LLMErrorKind.TRIAL_BUDGET_EXHAUSTED
    assert len(calls) == 1
