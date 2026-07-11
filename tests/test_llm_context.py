import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import capx.llm.client as client_module
import capx.llm.context as context_module
from capx.llm.context import (
    get_trial_llm_context,
    llm_call_stage,
    trial_llm_context,
)
from capx.llm.client import ModelQueryArgs


TELEMETRY_FIELDS = {
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
    assert set(record) == TELEMETRY_FIELDS
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


def test_remaining_after_uses_attempt_finish_boundary_not_later_clock(tmp_path):
    clock = FakeClock(200.0)
    telemetry_path = tmp_path / "calls.jsonl"

    with trial_llm_context(
        trial=17,
        deadline_monotonic=550.0,
        telemetry_path=telemetry_path,
        monotonic=clock,
    ) as context:
        call_index = context.next_call_index()
        _record_attempt(context, call_index=call_index)

    record = json.loads(telemetry_path.read_text().splitlines()[0])
    assert record["duration_ms"] == 427
    assert record["trial_remaining_ms_after"] == 449_573


def test_nullable_diagnostics_remain_null(tmp_path):
    telemetry_path = tmp_path / "calls.jsonl"

    with trial_llm_context(trial=17, telemetry_path=telemetry_path) as context:
        call_index = context.next_call_index()
        _record_attempt(
            context,
            call_index=call_index,
            http_status=None,
            ttfb_ms=None,
            first_content_ms=None,
            remaining_before_ms=None,
            error_kind=None,
        )

    record = json.loads(telemetry_path.read_text().splitlines()[0])
    assert record["http_status"] is None
    assert record["ttfb_ms"] is None
    assert record["first_content_ms"] is None
    assert record["trial_remaining_ms_before"] is None
    assert record["trial_remaining_ms_after"] is None
    assert record["error_kind"] is None


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


def test_trial_context_is_restored_when_nested_context_raises():
    with trial_llm_context(trial="outer") as outer:
        with pytest.raises(RuntimeError, match="boom"):
            with trial_llm_context(trial="inner"):
                raise RuntimeError("boom")
        assert get_trial_llm_context() is outer

    assert get_trial_llm_context() is None


def test_call_stage_is_restored_when_nested_stage_raises(tmp_path):
    telemetry_path = tmp_path / "calls.jsonl"

    with trial_llm_context(trial=17, telemetry_path=telemetry_path) as context:
        with pytest.raises(RuntimeError, match="boom"):
            with llm_call_stage("capsule_action"):
                raise RuntimeError("boom")
        call_index = context.next_call_index()
        _record_attempt(context, call_index=call_index)

    record = json.loads(telemetry_path.read_text().splitlines()[0])
    assert record["stage"] == "unknown"


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


def test_open_failure_raises_safe_error_without_mutating_attempt_accounting(
    tmp_path, monkeypatch
):
    telemetry_path = tmp_path / "calls.jsonl"

    def fail_open(*args, **kwargs):
        raise OSError("disk failure at /secret/path")

    with trial_llm_context(trial=17, telemetry_path=telemetry_path) as context:
        call_index = context.next_call_index()
        monkeypatch.setattr(Path, "open", fail_open)
        with pytest.raises(context_module.TelemetryWriteError) as error_info:
            _record_attempt(context, call_index=call_index)

        assert str(error_info.value) == "failed to persist LLM attempt telemetry"
        assert isinstance(error_info.value.__cause__, OSError)
        assert "/secret/path" not in str(error_info.value)
        assert context.summary() == {
            "logical_call_count": 1,
            "attempt_count": 0,
            "retry_count": 0,
            "elapsed_seconds": 0.0,
            "last_call_index": 1,
        }


def test_fsync_failure_does_not_double_count_after_successful_retry(tmp_path, monkeypatch):
    telemetry_path = tmp_path / "calls.jsonl"
    real_fsync = context_module.os.fsync
    calls = 0

    def fail_once(file_descriptor):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("fsync failed")
        return real_fsync(file_descriptor)

    monkeypatch.setattr(context_module.os, "fsync", fail_once)
    with trial_llm_context(trial=17, telemetry_path=telemetry_path) as context:
        call_index = context.next_call_index()
        with pytest.raises(context_module.TelemetryWriteError):
            _record_attempt(context, call_index=call_index)
        assert context.summary()["attempt_count"] == 0

        _record_attempt(context, call_index=call_index)
        assert context.summary() == {
            "logical_call_count": 1,
            "attempt_count": 1,
            "retry_count": 0,
            "elapsed_seconds": pytest.approx(0.427),
            "last_call_index": 1,
        }


@pytest.mark.parametrize("attempt", [0, 3])
def test_attempt_number_must_be_one_or_two(attempt):
    with trial_llm_context(trial=17) as context:
        call_index = context.next_call_index()
        with pytest.raises(ValueError, match="attempt"):
            _record_attempt(context, call_index=call_index, attempt=attempt)


def test_call_index_must_be_positive():
    with trial_llm_context(trial=17) as context:
        with pytest.raises(ValueError, match="call_index"):
            _record_attempt(context, call_index=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("started_monotonic", -1.0),
        ("finished_monotonic", -1.0),
        ("ttfb_ms", -1),
        ("first_content_ms", -1),
        ("remaining_before_ms", -1),
    ],
)
def test_timing_and_latency_diagnostics_cannot_be_negative(field, value):
    with trial_llm_context(trial=17) as context:
        call_index = context.next_call_index()
        with pytest.raises(ValueError, match=field):
            _record_attempt(context, call_index=call_index, **{field: value})


def test_attempt_finish_cannot_precede_start():
    with trial_llm_context(trial=17) as context:
        call_index = context.next_call_index()
        with pytest.raises(ValueError, match="finished_monotonic"):
            _record_attempt(
                context,
                call_index=call_index,
                started_monotonic=101.0,
                finished_monotonic=100.0,
            )


@pytest.mark.parametrize("retry_scheduled", [0, 1, "yes", None])
def test_retry_scheduled_must_be_boolean(retry_scheduled):
    with trial_llm_context(trial=17) as context:
        call_index = context.next_call_index()
        with pytest.raises(ValueError, match="retry_scheduled"):
            _record_attempt(
                context,
                call_index=call_index,
                retry_scheduled=retry_scheduled,
            )


def test_concurrent_attempt_records_are_complete_and_not_interleaved(tmp_path):
    telemetry_path = tmp_path / "calls.jsonl"

    with trial_llm_context(trial=17, telemetry_path=telemetry_path) as context:
        def record_one(_):
            call_index = context.next_call_index()
            _record_attempt(context, call_index=call_index, retry_scheduled=False)
            return call_index

        with ThreadPoolExecutor(max_workers=8) as executor:
            call_indices = list(executor.map(record_one, range(50)))
        summary = context.summary()

    records = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    assert len(records) == 50
    assert all(set(record) == TELEMETRY_FIELDS for record in records)
    assert sorted(record["call_index"] for record in records) == list(range(1, 51))
    assert sorted(call_indices) == list(range(1, 51))
    assert summary == {
        "logical_call_count": 50,
        "attempt_count": 50,
        "retry_count": 0,
        "elapsed_seconds": pytest.approx(21.35),
        "last_call_index": 50,
    }


def test_ensemble_workers_inherit_trial_context_with_unique_call_indices(
    tmp_path, monkeypatch
):
    telemetry_path = tmp_path / "ensemble_calls.jsonl"
    candidate_barrier = threading.Barrier(2)

    def fake_query_model(args, prompt):
        context = get_trial_llm_context()
        assert context is not None
        if args.model == "candidate-model":
            candidate_barrier.wait(timeout=2)
        call_index = context.next_call_index()
        _record_attempt(
            context,
            call_index=call_index,
            http_status=200,
            outcome="success",
            error_kind=None,
            retry_scheduled=False,
        )
        return {"content": "candidate", "reasoning": None}

    monkeypatch.setattr(client_module, "query_model", fake_query_model)
    monkeypatch.setattr(
        client_module,
        "ENSEMBLE_CONFIGS",
        [("candidate-model", [0.1, 0.2])],
    )
    args = ModelQueryArgs(
        model="unused",
        server_url="http://example.test",
        api_key=None,
        temperature=0.2,
        max_tokens=100,
    )

    with trial_llm_context(trial="trial-ensemble", telemetry_path=telemetry_path):
        with llm_call_stage("initial_code"):
            result = client_module.query_model_ensemble(
                args,
                [{"role": "user", "content": "task"}],
                synthesis_model="synthesis-model",
            )

    records = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    assert result["content"] == "candidate"
    assert len(records) == 3
    assert {record["trial"] for record in records} == {"trial-ensemble"}
    assert {record["stage"] for record in records} == {"initial_code"}
    assert sorted(record["call_index"] for record in records) == [1, 2, 3]


def test_single_model_ensemble_workers_inherit_trial_context(tmp_path, monkeypatch):
    telemetry_path = tmp_path / "single_model_ensemble_calls.jsonl"

    def fake_query_model(args, prompt):
        context = get_trial_llm_context()
        assert context is not None
        call_index = context.next_call_index()
        _record_attempt(
            context,
            call_index=call_index,
            http_status=200,
            outcome="success",
            error_kind=None,
            retry_scheduled=False,
        )
        return {"content": "candidate", "reasoning": None}

    monkeypatch.setattr(client_module, "query_model", fake_query_model)
    args = ModelQueryArgs(
        model="candidate-model",
        server_url="http://example.test",
        api_key=None,
        temperature=0.2,
        max_tokens=100,
    )

    with trial_llm_context(trial="trial-single", telemetry_path=telemetry_path):
        with llm_call_stage("multi_turn"):
            result = client_module.query_single_model_ensemble(
                args,
                [{"role": "user", "content": "task"}],
                "candidate-model",
            )

    records = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    assert result["content"] == "candidate"
    assert len(records) == 10
    assert {record["trial"] for record in records} == {"trial-single"}
    assert {record["stage"] for record in records} == {"multi_turn"}
    assert sorted(record["call_index"] for record in records) == list(range(1, 11))
