import json
import os
from types import SimpleNamespace

import pytest
import requests

from capx.llm.client import query_model, query_model_streaming
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

    def iter_content(self, chunk_size=8192):
        del chunk_size
        for delay, chunk in self.chunks:
            if self.clock is not None:
                self.clock.advance(delay)
            yield chunk


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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
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


def test_query_model_retries_streaming_timeout(monkeypatch):
    calls = []

    def fake_streaming(args, prompt):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("no content delta")
        yield {"type": "content_delta", "content": "ok"}
        yield {"type": "done", "content": "ok", "reasoning": None}

    monkeypatch.setenv("CAPX_FORCE_STREAMING_CHAT_COMPLETIONS", "1")
    monkeypatch.setenv("CAPX_STREAMING_CHAT_COMPLETIONS_RETRIES", "2")
    monkeypatch.setattr("capx.llm.client.query_model_streaming", fake_streaming)

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
        "content": "ok",
        "reasoning": None,
    }
    assert len(calls) == 2


def test_query_model_retries_empty_streaming_content(monkeypatch):
    calls = []

    def fake_streaming(args, prompt):
        calls.append(1)
        if len(calls) == 1:
            yield {"type": "done", "content": "", "reasoning": None}
            return
        yield {"type": "content_delta", "content": "ok"}
        yield {"type": "done", "content": "ok", "reasoning": None}

    monkeypatch.setenv("CAPX_FORCE_STREAMING_CHAT_COMPLETIONS", "1")
    monkeypatch.setenv("CAPX_STREAMING_CHAT_COMPLETIONS_RETRIES", "2")
    monkeypatch.setenv("CAPX_STREAMING_REQUIRE_CONTENT", "1")
    monkeypatch.setattr("capx.llm.client.query_model_streaming", fake_streaming)

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
        "content": "ok",
        "reasoning": None,
    }
    assert len(calls) == 2


def test_query_model_falls_back_to_non_streaming_after_empty_streaming_content(monkeypatch):
    stream_calls = []
    post_payloads = []

    def fake_streaming(args, prompt):
        stream_calls.append(1)
        yield {"type": "done", "content": "", "reasoning": None}

    class NonStreamingResponse:
        status_code = 200
        headers = {}

        def iter_content(self, chunk_size=8192):
            del chunk_size
            yield json.dumps(
                {"choices": [{"message": {"content": "fallback ok"}}]}
            ).encode()

    def fake_post(*args, data, **kwargs):
        post_payloads.append(json.loads(data))
        return NonStreamingResponse()

    monkeypatch.setenv("CAPX_FORCE_STREAMING_CHAT_COMPLETIONS", "1")
    monkeypatch.setenv("CAPX_STREAMING_CHAT_COMPLETIONS_RETRIES", "2")
    monkeypatch.setenv("CAPX_STREAMING_REQUIRE_CONTENT", "1")
    monkeypatch.setenv("CAPX_DISABLE_REASONING", "1")
    monkeypatch.setattr("capx.llm.client.query_model_streaming", fake_streaming)
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
        "content": "fallback ok",
        "reasoning": None,
    }
    assert len(stream_calls) == 2
    assert post_payloads[0]["thinking"] == {"type": "disabled"}
    assert os.environ["CAPX_FORCE_STREAMING_CHAT_COMPLETIONS"] == "1"


def test_query_model_retries_non_streaming_timeout_once(monkeypatch):
    request_timeouts = []

    class NonStreamingResponse:
        status_code = 200
        headers = {}

        def iter_content(self, chunk_size=8192):
            del chunk_size
            yield json.dumps({"choices": [{"message": {"content": "retry ok"}}]}).encode()

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
    class ReasoningOnlyResponse(_StreamingResponse):
        def iter_lines(self):
            for _ in range(3):
                yield b'data: {"choices": [{"delta": {"reasoning_content": "thinking"}}]}'

    def fake_post(*args, data, **kwargs):
        return ReasoningOnlyResponse()

    times = iter([0.0, 0.0, 2.0])

    monkeypatch.setenv("CAPX_DISABLE_REASONING", "1")
    monkeypatch.setenv("CAPX_STREAMING_FIRST_CONTENT_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)
    monkeypatch.setattr("capx.llm.client.time.time", lambda: next(times))

    args = SimpleNamespace(
        model="deepseek-v4-flash",
        server_url="https://example.invalid/v1/chat/completions",
        api_key="test-key",
        temperature=0.2,
        max_tokens=256,
        reasoning_effort="minimal",
    )

    try:
        list(query_model_streaming(args, [{"role": "user", "content": "hi"}]))
    except TimeoutError as exc:
        assert "No content delta" in str(exc)
    else:
        raise AssertionError("expected first-content timeout")


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
    assert records[0]["ttfb_ms"] == 0
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
