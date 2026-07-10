import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from capx.llm.context import (
    get_trial_llm_context,
    llm_call_stage,
    trial_llm_context,
)


class FakeClock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _record_attempt(context, *, call_index: int, attempt: int = 1, **overrides) -> None:
    fields = {
        "mode": "streaming",
        "http_status": 503,
        "ttfb_ms": 318,
        "first_content_ms": None,
        "started_monotonic": 100.0,
        "finished_monotonic": 100.427,
        "remaining_before_ms": 450_000,
        "outcome": "retryable_http_error",
        "error_kind": "http_5xx",
        "retry_scheduled": attempt == 1,
    }
    fields.update(overrides)
    context.record_attempt(call_index=call_index, attempt=attempt, **fields)


def test_remaining_budget_uses_monotonic_clock_and_clamps_to_zero():
    clock = FakeClock(100.0)

    with trial_llm_context(trial=17, deadline_monotonic=105.0, monotonic=clock) as context:
        assert context.remaining_seconds() == 5.0
        clock.value = 106.0
        assert context.remaining_seconds() == 0.0

    with trial_llm_context(trial=18, monotonic=clock) as context:
        assert context.remaining_seconds() is None


def test_call_indices_are_unique_and_monotonic_under_concurrency():
    with trial_llm_context(trial=17) as context:
        with ThreadPoolExecutor(max_workers=8) as executor:
            indices = list(executor.map(lambda _: context.next_call_index(), range(100)))

    assert sorted(indices) == list(range(1, 101))


def test_stage_defaults_and_nested_stages_restore(tmp_path):
    telemetry_path = tmp_path / "calls.jsonl"

    with trial_llm_context(trial=17, telemetry_path=telemetry_path) as context:
        first = context.next_call_index()
        _record_attempt(context, call_index=first)
        with llm_call_stage("initial_code"):
            second = context.next_call_index()
            _record_attempt(context, call_index=second)
            with llm_call_stage("visual_feedback"):
                third = context.next_call_index()
                _record_attempt(context, call_index=third)
            fourth = context.next_call_index()
            _record_attempt(context, call_index=fourth)
        fifth = context.next_call_index()
        _record_attempt(context, call_index=fifth)

    stages = [json.loads(line)["stage"] for line in telemetry_path.read_text().splitlines()]
    assert stages == [
        "unknown",
        "initial_code",
        "visual_feedback",
        "initial_code",
        "unknown",
    ]


def test_attempt_record_uses_exact_field_contract_and_nullable_values(tmp_path):
    clock = FakeClock(100.427)
    telemetry_path = tmp_path / "nested" / "llm_calls_trial_17.jsonl"

    with trial_llm_context(
        trial=17,
        deadline_monotonic=550.0,
        telemetry_path=telemetry_path,
        monotonic=clock,
    ) as context:
        with llm_call_stage("capsule_action"):
            call_index = context.next_call_index()
            _record_attempt(context, call_index=call_index)

    record = json.loads(telemetry_path.read_text().splitlines()[0])
    assert set(record) == {
        "trial",
        "call_index",
        "stage",
        "attempt",
        "mode",
        "http_status",
        "ttfb_ms",
        "first_content_ms",
        "duration_ms",
        "trial_remaining_ms_before",
        "trial_remaining_ms_after",
        "outcome",
        "error_kind",
        "retry_scheduled",
    }
    assert record == {
        "trial": 17,
        "call_index": 1,
        "stage": "capsule_action",
        "attempt": 1,
        "mode": "streaming",
        "http_status": 503,
        "ttfb_ms": 318,
        "first_content_ms": None,
        "duration_ms": 427,
        "trial_remaining_ms_before": 450_000,
        "trial_remaining_ms_after": 449_573,
        "outcome": "retryable_http_error",
        "error_kind": "http_5xx",
        "retry_scheduled": True,
    }


def test_attempt_records_append_and_update_summary(tmp_path):
    telemetry_path = tmp_path / "calls.jsonl"

    with trial_llm_context(trial="seed-17", telemetry_path=telemetry_path) as context:
        first = context.next_call_index()
        _record_attempt(context, call_index=first, retry_scheduled=True)
        _record_attempt(
            context,
            call_index=first,
            attempt=2,
            started_monotonic=101.0,
            finished_monotonic=102.25,
            retry_scheduled=False,
        )
        second = context.next_call_index()
        _record_attempt(
            context,
            call_index=second,
            started_monotonic=103.0,
            finished_monotonic=103.5,
            retry_scheduled=False,
        )
        summary = context.summary()

    records = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    assert len(records) == 3
    assert [record["attempt"] for record in records] == [1, 2, 1]
    assert summary == {
        "logical_call_count": 2,
        "attempt_count": 3,
        "retry_count": 1,
        "elapsed_seconds": pytest.approx(2.177),
        "last_call_index": 2,
    }


def test_trial_context_is_restored_after_nested_context():
    assert get_trial_llm_context() is None

    with trial_llm_context(trial="outer") as outer:
        assert get_trial_llm_context() is outer
        with trial_llm_context(trial="inner") as inner:
            assert get_trial_llm_context() is inner
        assert get_trial_llm_context() is outer

    assert get_trial_llm_context() is None


def test_record_attempt_rejects_arbitrary_sensitive_fields():
    with trial_llm_context(trial=17) as context:
        call_index = context.next_call_index()
        with pytest.raises(TypeError):
            _record_attempt(context, call_index=call_index, prompt="secret")
        with pytest.raises(TypeError):
            _record_attempt(context, call_index=call_index, authorization="Bearer secret")
        with pytest.raises(TypeError):
            _record_attempt(context, call_index=call_index, response="secret")


def test_missing_telemetry_path_still_updates_counters():
    with trial_llm_context(trial=17, telemetry_path=None) as context:
        call_index = context.next_call_index()
        _record_attempt(context, call_index=call_index, retry_scheduled=False)

        assert context.summary() == {
            "logical_call_count": 1,
            "attempt_count": 1,
            "retry_count": 0,
            "elapsed_seconds": pytest.approx(0.427),
            "last_call_index": 1,
        }
