from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from capx.envs.trial_results import RunOutcome, TrialResultWriter
from capx.llm.errors import LLMErrorKind
from capx.utils.launch_utils import TrialSummary


STARTED_AT = datetime(2026, 7, 10, 10, 0, tzinfo=timezone.utc)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _finished_result(**overrides):
    result = {
        "run_outcome": RunOutcome.FINISHED,
        "finished_at": STARTED_AT + timedelta(seconds=12.5),
        "elapsed_seconds": 12.5,
        "reward": 1.0,
        "task_completed": True,
        "sandbox_rc": 0,
        "llm": {
            "call_count": 2,
            "attempt_count": 3,
            "retry_count": 1,
            "elapsed_seconds": 4.25,
            "last_call_index": 2,
        },
    }
    result.update(overrides)
    return result


def test_writer_creates_running_result_before_trial_work(tmp_path):
    writer = TrialResultWriter(tmp_path)

    path = writer.start(trial=17, started_at=STARTED_AT)

    assert path == tmp_path / "trial_17_result.json"
    assert _load(path) == {
        "schema_version": 1,
        "trial": 17,
        "run_outcome": "running",
        "failure_kind": None,
        "failure_stage": None,
        "failure_message": None,
        "started_at": "2026-07-10T10:00:00Z",
        "finished_at": None,
        "elapsed_seconds": 0.0,
        "reward": None,
        "task_completed": None,
        "sandbox_rc": None,
        "llm": {
            "call_count": 0,
            "attempt_count": 0,
            "retry_count": 0,
            "elapsed_seconds": 0.0,
            "last_call_index": 0,
        },
    }


def test_writer_atomically_finalizes_finished_result(tmp_path):
    writer = TrialResultWriter(tmp_path)
    path = writer.start(trial=3, started_at=STARTED_AT)

    writer.finalize(_finished_result())

    result = _load(path)
    assert result["schema_version"] == 1
    assert result["trial"] == 3
    assert result["run_outcome"] == "finished"
    assert result["started_at"] == "2026-07-10T10:00:00Z"
    assert result["finished_at"] == "2026-07-10T10:00:12.500000Z"
    assert result["llm"]["attempt_count"] == 3
    assert not list(tmp_path.glob("*.tmp"))


def test_writer_persists_sanitized_llm_failure_and_accounting(tmp_path):
    writer = TrialResultWriter(tmp_path)
    path = writer.start(trial=4, started_at=STARTED_AT)
    unsafe_message = "Authorization: Bearer top-secret\n" + "x" * 1_000

    writer.finalize(
        _finished_result(
            run_outcome=RunOutcome.LLM_FAILED,
            failure_kind="http_5xx",
            failure_stage="capsule_action",
            failure_message=unsafe_message,
            reward=0.24,
            task_completed=False,
            sandbox_rc=1,
        )
    )

    result = _load(path)
    assert result["run_outcome"] == "llm_failed"
    assert result["failure_kind"] == "http_5xx"
    assert result["failure_stage"] == "capsule_action"
    assert "top-secret" not in result["failure_message"]
    assert "\n" not in result["failure_message"]
    assert len(result["failure_message"]) <= 512
    assert result["llm"] == {
        "call_count": 2,
        "attempt_count": 3,
        "retry_count": 1,
        "elapsed_seconds": 4.25,
        "last_call_index": 2,
    }


def test_writer_serializes_typed_llm_failure_kind_as_stable_value(tmp_path):
    writer = TrialResultWriter(tmp_path)
    path = writer.start(trial=5, started_at=STARTED_AT)

    writer.finalize(
        _finished_result(
            run_outcome=RunOutcome.LLM_FAILED,
            failure_kind=LLMErrorKind.HTTP_5XX,
        )
    )

    assert _load(path)["failure_kind"] == "http_5xx"


def test_writer_marks_only_residual_running_result_parent_guard_killed(tmp_path):
    writer = TrialResultWriter(tmp_path)
    path = writer.start(trial=8, started_at=STARTED_AT)

    writer.mark_parent_guard_killed(process_rc=124, elapsed_seconds=480.25)

    result = _load(path)
    assert result["run_outcome"] == "parent_guard_killed"
    assert result["failure_kind"] == "parent_guard_killed"
    assert result["elapsed_seconds"] == 480.25
    assert result["sandbox_rc"] == 124
    assert result["finished_at"].endswith("Z")


def test_writer_rejects_parent_guard_overwrite_of_finished_result(tmp_path):
    writer = TrialResultWriter(tmp_path)
    writer.start(trial=9, started_at=STARTED_AT)
    writer.finalize(_finished_result())

    with pytest.raises(ValueError, match="terminal"):
        writer.mark_parent_guard_killed(process_rc=124, elapsed_seconds=480.0)


def test_finalize_is_idempotent_but_rejects_a_different_terminal_result(tmp_path):
    writer = TrialResultWriter(tmp_path)
    path = writer.start(trial=10, started_at=STARTED_AT)
    result = _finished_result()
    writer.finalize(result)
    original = path.read_bytes()

    writer.finalize(result)
    assert path.read_bytes() == original

    with pytest.raises(ValueError, match="terminal"):
        writer.finalize(_finished_result(run_outcome=RunOutcome.CANCELLED))


def test_finalize_rejects_running_as_a_terminal_outcome(tmp_path):
    writer = TrialResultWriter(tmp_path)
    writer.start(trial=11, started_at=STARTED_AT)

    with pytest.raises(ValueError, match="terminal outcome"):
        writer.finalize({"run_outcome": RunOutcome.RUNNING})


def test_replace_failure_preserves_running_result_and_removes_temp_file(tmp_path, monkeypatch):
    writer = TrialResultWriter(tmp_path)
    path = writer.start(trial=12, started_at=STARTED_AT)
    original = path.read_bytes()

    def fail_replace(_source, _destination):
        raise OSError("replace failed")

    monkeypatch.setattr("capx.envs.trial_results.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        writer.finalize(_finished_result())

    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_start_does_not_overwrite_an_existing_terminal_result(tmp_path):
    writer = TrialResultWriter(tmp_path)
    path = writer.start(trial=13, started_at=STARTED_AT)
    writer.finalize(_finished_result())
    original = path.read_bytes()

    with pytest.raises(ValueError, match="already exists"):
        TrialResultWriter(tmp_path).start(trial=13, started_at=STARTED_AT)

    assert path.read_bytes() == original


def test_old_trial_summary_constructor_remains_valid():
    summary = TrialSummary(
        trial=1,
        success=True,
        reward=1.0,
        terminated=True,
        truncated=False,
        sandbox_rc=0,
        log="done",
    )

    assert summary.run_outcome is None
    assert summary.failure_kind is None
    assert summary.failure_stage is None
    assert summary.failure_message is None
    assert summary.llm_call_count == 0
    assert summary.llm_attempt_count == 0
    assert summary.llm_retry_count == 0
    assert summary.llm_elapsed_seconds == 0.0
