import json
import os
from types import SimpleNamespace

import requests

from capx.llm.client import query_model, query_model_streaming


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

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "fallback ok"}}]}

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

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "retry ok"}}]}

    def fake_post(*args, timeout, **kwargs):
        request_timeouts.append(timeout)
        if len(request_timeouts) == 1:
            raise requests.exceptions.Timeout("request timed out")
        return NonStreamingResponse()

    monkeypatch.delenv("CAPX_FORCE_STREAMING_CHAT_COMPLETIONS", raising=False)
    monkeypatch.setenv("CAPX_DISABLE_REASONING", "1")
    monkeypatch.setenv("CAPX_NONSTREAMING_REQUEST_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("CAPX_NONSTREAMING_REQUEST_RETRIES", "1")
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
