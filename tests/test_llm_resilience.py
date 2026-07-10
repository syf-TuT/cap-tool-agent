import json

import pytest

from capx.llm.errors import LLMErrorKind, LLMQueryError
from capx.llm.resilience import LLMRetryPolicy


POLICY_ENV_VARS = (
    "CAPX_LLM_MAX_ATTEMPTS",
    "CAPX_LLM_REQUEST_TIMEOUT_SECONDS",
    "CAPX_LLM_RETRY_BACKOFF_SECONDS",
    "CAPX_LLM_RETRY_AFTER_CAP_SECONDS",
    "CAPX_FORCE_STREAMING_CHAT_COMPLETIONS",
    "CAPX_STREAMING_CHAT_COMPLETIONS_RETRIES",
    "CAPX_STREAMING_REQUEST_TIMEOUT_SECONDS",
    "CAPX_STREAMING_FIRST_CONTENT_TIMEOUT_SECONDS",
    "CAPX_NONSTREAMING_REQUEST_RETRIES",
    "CAPX_NON_STREAMING_REQUEST_RETRIES",
    "CAPX_NONSTREAMING_REQUEST_TIMEOUT_SECONDS",
    "CAPX_NON_STREAMING_REQUEST_TIMEOUT_SECONDS",
)


def clear_llm_policy_env(monkeypatch):
    for name in POLICY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_error_kinds_have_stable_string_values():
    assert {kind.value for kind in LLMErrorKind} == {
        "connect_timeout",
        "read_timeout",
        "connection_error",
        "rate_limited",
        "http_5xx",
        "no_content",
        "invalid_response",
        "auth_error",
        "request_rejected",
        "trial_budget_exhausted",
    }
    assert isinstance(LLMErrorKind.HTTP_5XX, str)


def test_retry_policy_uses_approved_defaults(monkeypatch):
    clear_llm_policy_env(monkeypatch)

    policy = LLMRetryPolicy.from_env()

    assert policy == LLMRetryPolicy(
        max_attempts=2,
        request_timeout_seconds=60.0,
        retry_backoff_seconds=1.0,
        retry_jitter_seconds=0.5,
        retry_after_cap_seconds=10.0,
        minimum_retry_budget_seconds=5.0,
        first_content_timeout_seconds=45.0,
    )


def test_canonical_policy_variables_override_legacy_aliases(monkeypatch):
    clear_llm_policy_env(monkeypatch)
    monkeypatch.setenv("CAPX_FORCE_STREAMING_CHAT_COMPLETIONS", "1")
    monkeypatch.setenv("CAPX_LLM_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("CAPX_STREAMING_CHAT_COMPLETIONS_RETRIES", "1")
    monkeypatch.setenv("CAPX_NONSTREAMING_REQUEST_RETRIES", "0")
    monkeypatch.setenv("CAPX_LLM_REQUEST_TIMEOUT_SECONDS", "17")
    monkeypatch.setenv("CAPX_STREAMING_REQUEST_TIMEOUT_SECONDS", "98")
    monkeypatch.setenv("CAPX_NONSTREAMING_REQUEST_TIMEOUT_SECONDS", "99")
    monkeypatch.setenv("CAPX_LLM_RETRY_BACKOFF_SECONDS", "2.5")
    monkeypatch.setenv("CAPX_LLM_RETRY_AFTER_CAP_SECONDS", "7")

    policy = LLMRetryPolicy.from_env()

    assert policy.max_attempts == 2
    assert policy.request_timeout_seconds == 17.0
    assert policy.retry_backoff_seconds == 2.5
    assert policy.retry_after_cap_seconds == 7.0


@pytest.mark.parametrize(
    ("configured_attempts", "expected_attempts"),
    [("0", 1), ("1", 1), ("2", 2), ("9", 2)],
)
def test_legacy_streaming_value_is_an_attempt_count(
    monkeypatch, configured_attempts, expected_attempts
):
    clear_llm_policy_env(monkeypatch)
    monkeypatch.setenv("CAPX_FORCE_STREAMING_CHAT_COMPLETIONS", "1")
    monkeypatch.setenv("CAPX_STREAMING_CHAT_COMPLETIONS_RETRIES", configured_attempts)

    assert LLMRetryPolicy.from_env().max_attempts == expected_attempts


def test_legacy_streaming_attempt_count_is_recognized_without_force_flag(monkeypatch):
    clear_llm_policy_env(monkeypatch)
    monkeypatch.setenv("CAPX_STREAMING_CHAT_COMPLETIONS_RETRIES", "1")

    assert LLMRetryPolicy.from_env().max_attempts == 1


@pytest.mark.parametrize(
    ("configured_retries", "expected_attempts"),
    [("-1", 1), ("0", 1), ("1", 2), ("8", 2)],
)
def test_legacy_non_streaming_value_is_converted_from_retries_to_attempts(
    monkeypatch, configured_retries, expected_attempts
):
    clear_llm_policy_env(monkeypatch)
    monkeypatch.setenv("CAPX_NONSTREAMING_REQUEST_RETRIES", configured_retries)

    assert LLMRetryPolicy.from_env().max_attempts == expected_attempts


@pytest.mark.parametrize(
    ("configured_attempts", "expected"),
    [("-2", 1), ("0", 1), ("1", 1), ("2", 2), ("3", 2)],
)
def test_canonical_max_attempts_is_clamped(monkeypatch, configured_attempts, expected):
    clear_llm_policy_env(monkeypatch)
    monkeypatch.setenv("CAPX_LLM_MAX_ATTEMPTS", configured_attempts)

    assert LLMRetryPolicy.from_env().max_attempts == expected


def test_legacy_non_streaming_underscore_aliases_are_supported(monkeypatch):
    clear_llm_policy_env(monkeypatch)
    monkeypatch.setenv("CAPX_NON_STREAMING_REQUEST_RETRIES", "1")
    monkeypatch.setenv("CAPX_NON_STREAMING_REQUEST_TIMEOUT_SECONDS", "23")

    policy = LLMRetryPolicy.from_env()

    assert policy.max_attempts == 2
    assert policy.request_timeout_seconds == 23.0


def test_streaming_legacy_timeout_values_are_supported(monkeypatch):
    clear_llm_policy_env(monkeypatch)
    monkeypatch.setenv("CAPX_FORCE_STREAMING_CHAT_COMPLETIONS", "1")
    monkeypatch.setenv("CAPX_STREAMING_REQUEST_TIMEOUT_SECONDS", "31")
    monkeypatch.setenv("CAPX_STREAMING_FIRST_CONTENT_TIMEOUT_SECONDS", "12")

    policy = LLMRetryPolicy.from_env()

    assert policy.request_timeout_seconds == 31.0
    assert policy.first_content_timeout_seconds == 12.0


def test_streaming_legacy_request_timeout_is_recognized_without_force_flag(monkeypatch):
    clear_llm_policy_env(monkeypatch)
    monkeypatch.setenv("CAPX_STREAMING_REQUEST_TIMEOUT_SECONDS", "31")

    assert LLMRetryPolicy.from_env().request_timeout_seconds == 31.0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CAPX_LLM_REQUEST_TIMEOUT_SECONDS", "0"),
        ("CAPX_LLM_REQUEST_TIMEOUT_SECONDS", "-1"),
        ("CAPX_NONSTREAMING_REQUEST_TIMEOUT_SECONDS", "0"),
        ("CAPX_STREAMING_REQUEST_TIMEOUT_SECONDS", "-0.5"),
        ("CAPX_STREAMING_FIRST_CONTENT_TIMEOUT_SECONDS", "0"),
    ],
)
def test_non_positive_timeout_values_raise_value_error(monkeypatch, name, value):
    clear_llm_policy_env(monkeypatch)
    if name.startswith("CAPX_STREAMING"):
        monkeypatch.setenv("CAPX_FORCE_STREAMING_CHAT_COMPLETIONS", "1")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        LLMRetryPolicy.from_env()


@pytest.mark.parametrize(
    "name",
    [
        "CAPX_LLM_REQUEST_TIMEOUT_SECONDS",
        "CAPX_LLM_RETRY_BACKOFF_SECONDS",
        "CAPX_LLM_RETRY_AFTER_CAP_SECONDS",
        "CAPX_STREAMING_FIRST_CONTENT_TIMEOUT_SECONDS",
    ],
)
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_non_finite_environment_numbers_raise_value_error(monkeypatch, name, value):
    clear_llm_policy_env(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        LLMRetryPolicy.from_env()


@pytest.mark.parametrize(
    "field",
    [
        "request_timeout_seconds",
        "retry_backoff_seconds",
        "retry_jitter_seconds",
        "retry_after_cap_seconds",
        "minimum_retry_budget_seconds",
        "first_content_timeout_seconds",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_direct_policy_construction_rejects_non_finite_numbers(field, value):
    with pytest.raises(ValueError, match=field):
        LLMRetryPolicy(**{field: value})


def test_llm_query_error_exposes_only_safe_scalar_metadata():
    error = LLMQueryError(
        kind=LLMErrorKind.HTTP_5XX,
        call_index=4,
        attempt=2,
        status_code=503,
        elapsed_seconds=1.25,
        message=(
            "Authorization: Bearer sk-secret-token api_key=also-secret "
            "OPENAI_API_KEY=sk-openai-secret "
            "OPENROUTER_API_KEY=sk-openrouter-secret "
            + "provider unavailable " * 100
        ),
    )

    safe = error.to_safe_dict()
    encoded = json.dumps(safe)

    assert safe.keys() == {
        "kind",
        "call_index",
        "attempt",
        "status_code",
        "elapsed_seconds",
        "message",
    }
    assert safe["kind"] == "http_5xx"
    assert safe["call_index"] == 4
    assert safe["attempt"] == 2
    assert safe["status_code"] == 503
    assert safe["elapsed_seconds"] == 1.25
    assert len(safe["message"]) <= 512
    assert "sk-secret-token" not in encoded
    assert "also-secret" not in encoded
    assert "sk-openai-secret" not in encoded
    assert "sk-openrouter-secret" not in encoded
    assert error.kind is LLMErrorKind.HTTP_5XX
    assert str(error) == safe["message"]


def test_llm_query_error_allows_missing_optional_http_metadata():
    error = LLMQueryError(
        kind=LLMErrorKind.CONNECTION_ERROR,
        call_index=1,
        attempt=1,
        status_code=None,
        elapsed_seconds=0.1,
        message="connection reset",
    )

    safe = error.to_safe_dict()

    assert safe["status_code"] is None
    assert all(value is None or isinstance(value, (str, int, float)) for value in safe.values())


@pytest.mark.parametrize(
    ("message", "secret", "preserved_syntax"),
    [
        (
            '{"api_key": "sk-json-secret", "model": "x"}',
            "sk-json-secret",
            '{"api_key": "[REDACTED]", "model": "x"}',
        ),
        (
            "{'OPENAI_API_KEY': 'sk-python-secret', 'x': 1}",
            "sk-python-secret",
            "{'OPENAI_API_KEY': '[REDACTED]', 'x': 1}",
        ),
        (
            '{"Authorization": "Bearer sk-auth-secret", "x": 1}',
            "sk-auth-secret",
            '{"Authorization": "[REDACTED]", "x": 1}',
        ),
        (
            "https://provider.test/chat?api_key=sk-query-secret&x=1",
            "sk-query-secret",
            "api_key=[REDACTED]&x=1",
        ),
    ],
)
def test_llm_query_error_redacts_common_credential_syntax(
    message, secret, preserved_syntax
):
    error = LLMQueryError(
        kind=LLMErrorKind.AUTH_ERROR,
        call_index=1,
        attempt=1,
        status_code=401,
        elapsed_seconds=0.2,
        message=message,
    )

    safe_message = error.to_safe_dict()["message"]

    assert secret not in safe_message
    assert preserved_syntax in safe_message
