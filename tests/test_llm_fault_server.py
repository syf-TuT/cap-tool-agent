"""Local HTTP fault injection coverage for the bounded LLM client.

These tests exercise Requests against a real localhost socket.  They deliberately
never contact an LLM provider and therefore cannot consume provider quota.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest

from capx.envs.trial_results import RunOutcome, TrialResultWriter
from capx.llm.client import query_model
from capx.llm.context import trial_llm_context
from capx.llm.errors import LLMErrorKind
from capx.utils.experiment_results import aggregate_trial_results


class _FaultHandler(BaseHTTPRequestHandler):
    """A configurable sequence of local OpenAI-compatible responses."""

    protocol_version = "HTTP/1.1"
    scenarios: list[str] = []
    request_bodies: list[dict] = []
    lock = threading.Lock()

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def do_POST(self) -> None:  # noqa: N802 - required stdlib callback name
        size = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(size))
        with self.lock:
            self.request_bodies.append(body)
            scenario = self.scenarios.pop(0)

        if scenario == "503":
            self._json(503, {"error": "temporarily unavailable"})
        elif scenario == "429":
            self._json(429, {"error": "slow down"}, retry_after="3600")
        elif scenario == "json":
            self._json(200, _success_body("ok"))
        elif scenario == "delayed_json":
            time.sleep(0.04)
            self._json(200, _success_body("delayed"))
        elif scenario == "heartbeats":
            self._sse(b": heartbeat\n\n")
            time.sleep(0.15)
        elif scenario == "partial_close":
            payload = b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            # Complete the first chunk, then truncate the second.  Requests can
            # deliver the first SSE delta before surfacing the connection error.
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n20\r\ntruncated")
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
        else:  # pragma: no cover - protects the fixture itself
            raise AssertionError(f"unknown scenario: {scenario}")

    def _json(self, status: int, body: dict, retry_after: str | None = None) -> None:
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        if retry_after is not None:
            self.send_header("Retry-After", retry_after)
        self.end_headers()
        self.wfile.write(encoded)

    def _sse(self, payload: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()


def _success_body(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


@contextmanager
def _fault_server(*scenarios: str) -> Iterator[str]:
    _FaultHandler.scenarios = list(scenarios)
    _FaultHandler.request_bodies = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FaultHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    finally:
        server.shutdown()
        thread.join(timeout=1)
        server.server_close()


def _args(server_url: str) -> SimpleNamespace:
    return SimpleNamespace(
        model="local-test-model",
        server_url=server_url,
        api_key="not-a-real-secret",
        temperature=0.2,
        max_tokens=32,
        reasoning_effort="minimal",
    )


def _policy(monkeypatch: pytest.MonkeyPatch, *, streaming: bool = False) -> None:
    monkeypatch.setenv("CAPX_LLM_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("CAPX_LLM_REQUEST_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("CAPX_LLM_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setenv("CAPX_LLM_RETRY_JITTER_SECONDS", "0")
    monkeypatch.setenv("CAPX_LLM_RETRY_AFTER_CAP_SECONDS", "0.05")
    monkeypatch.setenv("CAPX_LLM_MINIMUM_RETRY_BUDGET_SECONDS", "0.01")
    monkeypatch.setenv("CAPX_STREAMING_FIRST_CONTENT_TIMEOUT_SECONDS", "0.05")
    monkeypatch.setenv("CAPX_DISABLE_REASONING", "1")
    if streaming:
        monkeypatch.setenv("CAPX_FORCE_STREAMING_CHAT_COMPLETIONS", "1")
    else:
        monkeypatch.delenv("CAPX_FORCE_STREAMING_CHAT_COMPLETIONS", raising=False)


def _telemetry(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_real_requests_503_retries_once_and_writes_complete_telemetry(monkeypatch, tmp_path):
    _policy(monkeypatch)
    telemetry_path = tmp_path / "llm_calls_trial_17.jsonl"
    with _fault_server("503", "json") as server_url:
        with trial_llm_context(
            trial=17,
            deadline_monotonic=time.monotonic() + 7,
            telemetry_path=telemetry_path,
        ):
            result = query_model(_args(server_url), [{"role": "user", "content": "hello"}])

    assert result == {"content": "ok", "reasoning": None}
    records = _telemetry(telemetry_path)
    assert [(row["call_index"], row["attempt"], row["http_status"]) for row in records] == [
        (1, 1, 503),
        (1, 2, 200),
    ]
    assert records[0]["retry_scheduled"] is True
    assert records[1]["outcome"] == "success"
    assert records[1]["ttfb_ms"] is not None
    expected_fields = {
        "trial", "call_index", "stage", "attempt", "mode", "http_status", "ttfb_ms",
        "first_content_ms", "duration_ms", "trial_remaining_ms_before",
        "trial_remaining_ms_after", "outcome", "error_kind", "retry_scheduled",
    }
    assert set(records[0]) == expected_fields


def test_real_requests_delayed_first_byte_measures_ttfb(monkeypatch, tmp_path):
    _policy(monkeypatch)
    telemetry_path = tmp_path / "llm_calls_trial_18.jsonl"
    with _fault_server("delayed_json") as server_url:
        with trial_llm_context(trial=18, telemetry_path=telemetry_path):
            result = query_model(
                _args(server_url), [{"role": "user", "content": "hello"}]
            )
    assert result["content"] == "delayed"

    record = _telemetry(telemetry_path)[0]
    assert record["http_status"] == 200
    assert record["ttfb_ms"] >= 25
    assert record["duration_ms"] >= record["ttfb_ms"]


def test_real_requests_oversized_retry_after_is_capped(monkeypatch):
    _policy(monkeypatch)
    with _fault_server("429", "json") as server_url:
        started = time.monotonic()
        result = query_model(_args(server_url), [{"role": "user", "content": "hello"}])
        elapsed = time.monotonic() - started

    assert result["content"] == "ok"
    assert elapsed >= 0.04
    assert elapsed < 0.5


@pytest.mark.parametrize("scenario", ["heartbeats", "partial_close"])
def test_real_stream_breakages_fall_back_once_to_non_streaming(monkeypatch, scenario):
    _policy(monkeypatch, streaming=True)
    with _fault_server(scenario, "json") as server_url:
        result = query_model(_args(server_url), [{"role": "user", "content": "hello"}])

    assert result == {"content": "ok", "reasoning": None}
    assert len(_FaultHandler.request_bodies) == 2
    assert _FaultHandler.request_bodies[0]["stream"] is True
    assert "stream" not in _FaultHandler.request_bodies[1]


def test_result_outcomes_and_aggregation_keep_finished_metrics_separate(tmp_path):
    started_at = datetime(2026, 7, 10, tzinfo=timezone.utc)
    finished_writer = TrialResultWriter(tmp_path)
    finished_writer.start(trial=1, started_at=started_at)
    finished_writer.finalize(
        {
            "run_outcome": RunOutcome.FINISHED,
            "finished_at": started_at + timedelta(seconds=1),
            "elapsed_seconds": 1.0,
            "reward": 1.0,
            "task_completed": True,
            "sandbox_rc": 0,
        }
    )
    failed_writer = TrialResultWriter(tmp_path)
    failed_writer.start(trial=2, started_at=started_at)
    failed_writer.finalize(
        {
            "run_outcome": RunOutcome.LLM_FAILED,
            "failure_kind": LLMErrorKind.HTTP_5XX.value,
            "finished_at": started_at + timedelta(seconds=1),
            "elapsed_seconds": 1.0,
            "reward": 0.0,
            "task_completed": False,
            "sandbox_rc": 1,
        }
    )

    summary = aggregate_trial_results(
        [json.loads(finished_writer.path.read_text()), json.loads(failed_writer.path.read_text())]
    )
    assert summary["outcome_counts"] == {"finished": 1, "llm_failed": 1}
    assert summary["average_reward"] == 1.0
    assert summary["task_completion_rate"] == 1.0
