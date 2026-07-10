"""Typed, safely serializable failures raised by LLM requests."""

from __future__ import annotations

import re
from enum import Enum


class LLMErrorKind(str, Enum):
    """Stable categories used for LLM failure accounting."""

    CONNECT_TIMEOUT = "connect_timeout"
    READ_TIMEOUT = "read_timeout"
    CONNECTION_ERROR = "connection_error"
    RATE_LIMITED = "rate_limited"
    HTTP_5XX = "http_5xx"
    NO_CONTENT = "no_content"
    INVALID_RESPONSE = "invalid_response"
    AUTH_ERROR = "auth_error"
    REQUEST_REJECTED = "request_rejected"
    TRIAL_BUDGET_EXHAUSTED = "trial_budget_exhausted"


_MAX_SAFE_MESSAGE_LENGTH = 512
_AUTHORIZATION_PATTERN = re.compile(
    r"(?i)(?P<prefix>(?P<key_quote>[\"']?)authorization(?P=key_quote)\s*[:=]\s*)"
    r"(?P<value>\"[^\"]*\"|'[^']*'|(?:bearer\s+)?[^\s,;&}\]]+)"
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_CREDENTIAL_PATTERN = re.compile(
    r"(?i)(?P<prefix>(?P<key_quote>[\"']?)"
    r"(?:(?:[a-z0-9]+_)*api[_-]?key|access[_-]?token|token|secret|password)"
    r"(?P=key_quote)\s*[:=]\s*)"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^\s,;&}\]]+)"
)


def _redact_matched_value(match: re.Match[str]) -> str:
    value = match.group("value")
    if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
        redacted_value = f"{value[0]}[REDACTED]{value[0]}"
    else:
        redacted_value = "[REDACTED]"
    return f"{match.group('prefix')}{redacted_value}"


def _sanitize_message(message: object) -> str:
    safe = str(message).replace("\r", " ").replace("\n", " ")
    safe = _AUTHORIZATION_PATTERN.sub(_redact_matched_value, safe)
    safe = _BEARER_PATTERN.sub("[REDACTED]", safe)
    safe = _CREDENTIAL_PATTERN.sub(_redact_matched_value, safe)
    safe = " ".join(safe.split())
    return safe[:_MAX_SAFE_MESSAGE_LENGTH]


class LLMQueryError(Exception):
    """An LLM request failure containing only bounded, safe scalar metadata."""

    def __init__(
        self,
        *,
        kind: LLMErrorKind,
        call_index: int,
        attempt: int,
        status_code: int | None,
        elapsed_seconds: float,
        message: str,
    ) -> None:
        self.kind = LLMErrorKind(kind)
        self.call_index = int(call_index)
        self.attempt = int(attempt)
        self.status_code = None if status_code is None else int(status_code)
        self.elapsed_seconds = float(elapsed_seconds)
        self.message = _sanitize_message(message)
        super().__init__(self.message)

    def to_safe_dict(self) -> dict[str, str | int | float | None]:
        """Return metadata suitable for logs and structured trial results."""

        return {
            "kind": self.kind.value,
            "call_index": self.call_index,
            "attempt": self.attempt,
            "status_code": self.status_code,
            "elapsed_seconds": self.elapsed_seconds,
            "message": self.message,
        }
