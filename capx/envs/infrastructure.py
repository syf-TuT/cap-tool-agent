"""Typed failures shared by environment launch and trial execution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ServiceEndpoint:
    """A TCP endpoint required before an environment can be constructed."""

    name: str
    host: str
    port: int


class InfrastructureFailure(RuntimeError):
    """Signals a non-program failure in a required runtime service."""

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        self.kind = kind
        self.message = message
        self.evidence = evidence or {}
        super().__init__(message)


class ServiceReadinessError(InfrastructureFailure):
    """Signals that required services did not become ready before launch."""


_CONNECTION_EXCEPTION_TYPES = {
    "ConnectError",
    "ConnectionError",
    "ConnectionRefusedError",
    "NewConnectionError",
}
_TIMEOUT_EXCEPTION_TYPES = {
    "ConnectTimeout",
    "ReadTimeout",
    "Timeout",
    "TimeoutError",
}
_HTTP_5XX_PATTERN = re.compile(
    r"(?:http(?:\s+error)?|status(?:\s+code)?|server\s+error)[^\d]{0,16}(5\d{2})"
    r"|(5\d{2})\s+server\s+error",
    re.IGNORECASE,
)


def classify_runtime_infrastructure_failure(
    event: Any,
) -> InfrastructureFailure | None:
    """Classify transient service failures returned by Capsule execution."""
    if getattr(event, "status", None) != "failed":
        return None

    evidence = getattr(event, "evidence", {})
    evidence = evidence if isinstance(evidence, dict) else {}
    trace_events = evidence.get("trace_events", [])
    trace_events = trace_events if isinstance(trace_events, list) else []
    exception_types = {
        str(value)
        for value in [
            evidence.get("exception_type"),
            *[
                trace_event.get("exception_type")
                for trace_event in trace_events
                if isinstance(trace_event, dict)
            ],
        ]
        if value
    }
    text = "\n".join(
        str(value)
        for value in [
            getattr(event, "message", ""),
            getattr(event, "stderr", ""),
            *[
                trace_event.get("message", "")
                for trace_event in trace_events
                if isinstance(trace_event, dict)
            ],
        ]
        if value
    )
    lowered_text = text.casefold()

    if exception_types & _CONNECTION_EXCEPTION_TYPES or any(
        marker in lowered_text
        for marker in (
            "connection refused",
            "failed to establish a new connection",
            "max retries exceeded",
        )
    ):
        kind = "service_connection_refused"
    elif exception_types & _TIMEOUT_EXCEPTION_TYPES or any(
        marker in lowered_text
        for marker in ("timed out", "timeout", "time-out")
    ):
        kind = "service_timeout"
    elif _HTTP_5XX_PATTERN.search(text):
        kind = "service_http_5xx"
    else:
        return None

    event_data = event.to_dict() if callable(getattr(event, "to_dict", None)) else {}
    message = getattr(event, "message", "") or text or kind
    return InfrastructureFailure(
        kind,
        f"Runtime service failure during {getattr(event, 'action', 'execution')}: {message}",
        evidence={"runtime_event": event_data},
    )
