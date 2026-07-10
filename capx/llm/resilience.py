"""Configuration for bounded LLM request attempts."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Mapping


def _first_value(environ: Mapping[str, str], *names: str) -> tuple[str, str] | None:
    for name in names:
        if name in environ:
            return name, environ[name]
    return None


def _parse_int(name: str, value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _parse_float(name: str, value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return parsed


def _positive_float(name: str, value: str) -> float:
    parsed = _parse_float(name, value)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value!r}")
    return parsed


def _non_negative_float(name: str, value: str) -> float:
    parsed = _parse_float(name, value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return parsed


@dataclass(frozen=True)
class LLMRetryPolicy:
    """Immutable request limits shared by streaming and non-streaming calls."""

    max_attempts: int = 2
    request_timeout_seconds: float = 60.0
    retry_backoff_seconds: float = 1.0
    retry_jitter_seconds: float = 0.5
    retry_after_cap_seconds: float = 10.0
    minimum_retry_budget_seconds: float = 5.0
    first_content_timeout_seconds: float = 45.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_attempts", min(2, max(1, int(self.max_attempts))))
        for name in (
            "request_timeout_seconds",
            "retry_backoff_seconds",
            "retry_jitter_seconds",
            "retry_after_cap_seconds",
            "minimum_retry_budget_seconds",
            "first_content_timeout_seconds",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be greater than zero")
        if self.first_content_timeout_seconds <= 0:
            raise ValueError("first_content_timeout_seconds must be greater than zero")
        for name in (
            "retry_backoff_seconds",
            "retry_jitter_seconds",
            "retry_after_cap_seconds",
            "minimum_retry_budget_seconds",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @classmethod
    def from_env(cls) -> "LLMRetryPolicy":
        """Build a policy while preserving legacy retry-count semantics."""

        environ = os.environ
        streaming = environ.get("CAPX_FORCE_STREAMING_CHAT_COMPLETIONS") == "1"

        max_attempts_value = _first_value(environ, "CAPX_LLM_MAX_ATTEMPTS")
        if max_attempts_value is not None:
            name, raw_value = max_attempts_value
            max_attempts = _parse_int(name, raw_value)
        else:
            streaming_legacy = _first_value(
                environ, "CAPX_STREAMING_CHAT_COMPLETIONS_RETRIES"
            )
            non_streaming_legacy = _first_value(
                environ,
                "CAPX_NONSTREAMING_REQUEST_RETRIES",
                "CAPX_NON_STREAMING_REQUEST_RETRIES",
            )
            if streaming or (non_streaming_legacy is None and streaming_legacy is not None):
                max_attempts = (
                    2 if streaming_legacy is None else _parse_int(*streaming_legacy)
                )
            else:
                max_attempts = (
                    2
                    if non_streaming_legacy is None
                    else _parse_int(*non_streaming_legacy) + 1
                )

        request_timeout_value = _first_value(environ, "CAPX_LLM_REQUEST_TIMEOUT_SECONDS")
        if request_timeout_value is None:
            streaming_timeout = _first_value(
                environ, "CAPX_STREAMING_REQUEST_TIMEOUT_SECONDS"
            )
            non_streaming_timeout = _first_value(
                environ,
                "CAPX_NONSTREAMING_REQUEST_TIMEOUT_SECONDS",
                "CAPX_NON_STREAMING_REQUEST_TIMEOUT_SECONDS",
            )
            if streaming:
                request_timeout_value = streaming_timeout or non_streaming_timeout
            else:
                request_timeout_value = non_streaming_timeout or streaming_timeout
        request_timeout_seconds = (
            60.0
            if request_timeout_value is None
            else _positive_float(*request_timeout_value)
        )

        backoff_value = _first_value(environ, "CAPX_LLM_RETRY_BACKOFF_SECONDS")
        retry_backoff_seconds = (
            1.0 if backoff_value is None else _non_negative_float(*backoff_value)
        )

        retry_after_value = _first_value(environ, "CAPX_LLM_RETRY_AFTER_CAP_SECONDS")
        retry_after_cap_seconds = (
            10.0 if retry_after_value is None else _non_negative_float(*retry_after_value)
        )

        first_content_value = _first_value(
            environ, "CAPX_STREAMING_FIRST_CONTENT_TIMEOUT_SECONDS"
        )
        first_content_timeout_seconds = (
            45.0
            if first_content_value is None
            else _positive_float(*first_content_value)
        )

        return cls(
            max_attempts=max_attempts,
            request_timeout_seconds=request_timeout_seconds,
            retry_backoff_seconds=retry_backoff_seconds,
            retry_after_cap_seconds=retry_after_cap_seconds,
            first_content_timeout_seconds=first_content_timeout_seconds,
        )
