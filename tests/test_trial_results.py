from __future__ import annotations

import json
import threading
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


def test_concurrent_finalizers_cannot_read_running_during_first_terminal_write(
    tmp_path, monkeypatch
):
    writer = TrialResultWriter(tmp_path)
    path = writer.start(trial=14, started_at=STARTED_AT)
    first_write_entered = threading.Event()
    release_first_write = threading.Event()
    second_started = threading.Event()
    second_read_entered = threading.Event()
    original_atomic_write = writer._atomic_write
    original_read = writer._read

    def controlled_atomic_write(target, value):
        if (
            threading.current_thread().name == "first-finalizer"
            and value["run_outcome"] == RunOutcome.FINISHED.value
        ):
            first_write_entered.set()
            assert release_first_write.wait(timeout=2)
        original_atomic_write(target, value)

    def observed_read():
        if threading.current_thread().name == "second-finalizer":
            second_read_entered.set()
        return original_read()

    monkeypatch.setattr(writer, "_atomic_write", controlled_atomic_write)
    monkeypatch.setattr(writer, "_read", observed_read)
    errors = []

    first = threading.Thread(
        name="first-finalizer", target=lambda: writer.finalize(_finished_result())
    )

    def finalize_second():
        second_started.set()
        try:
            writer.finalize(
                _finished_result(
                    run_outcome=RunOutcome.CANCELLED,
                    failure_kind="cancelled",
                )
            )
        except Exception as error:
            errors.append(error)

    second = threading.Thread(name="second-finalizer", target=finalize_second)
    first.start()
    assert first_write_entered.wait(timeout=2)
    second.start()
    assert second_started.wait(timeout=2)
    try:
        assert not second_read_entered.wait(timeout=0.2)
    finally:
        release_first_write.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert _load(path)["run_outcome"] == "finished"
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert "terminal" in str(errors[0])


def test_parent_guard_try_mark_returns_false_when_child_wins_finalization_race(
    tmp_path, monkeypatch
):
    writer = TrialResultWriter(tmp_path)
    path = writer.start(trial=18, started_at=STARTED_AT)
    child = TrialResultWriter.open_existing(path)
    parent = TrialResultWriter.open_existing(path)
    child_write_entered = threading.Event()
    release_child_write = threading.Event()
    original_atomic_write = child._atomic_write

    def controlled_atomic_write(target, value):
        if value["run_outcome"] == RunOutcome.FINISHED.value:
            child_write_entered.set()
            assert release_child_write.wait(timeout=2)
        original_atomic_write(target, value)

    monkeypatch.setattr(child, "_atomic_write", controlled_atomic_write)
    child_thread = threading.Thread(target=lambda: child.finalize(_finished_result()))
    child_thread.start()
    assert child_write_entered.wait(timeout=2)

    parent_result = []
    parent_thread = threading.Thread(
        target=lambda: parent_result.append(
            parent.try_mark_parent_guard_killed(process_rc=124, elapsed_seconds=480.0)
        )
    )
    parent_thread.start()
    release_child_write.set()
    child_thread.join(timeout=2)
    parent_thread.join(timeout=2)

    assert not child_thread.is_alive()
    assert not parent_thread.is_alive()
    assert parent_result == [False]
    assert _load(path)["run_outcome"] == RunOutcome.FINISHED.value


@pytest.mark.parametrize("trial", [True, -1, 1.5, "1"])
def test_start_rejects_invalid_trial_identifiers(tmp_path, trial):
    writer = TrialResultWriter(tmp_path)

    with pytest.raises((TypeError, ValueError), match="trial"):
        writer.start(trial=trial, started_at=STARTED_AT)


@pytest.mark.parametrize(
    "reward", [float("nan"), float("inf"), -float("inf"), True, "1.0"]
)
def test_finalize_rejects_invalid_reward(tmp_path, reward):
    writer = TrialResultWriter(tmp_path)
    writer.start(trial=15, started_at=STARTED_AT)

    with pytest.raises(ValueError, match="reward"):
        writer.finalize(_finished_result(reward=reward))


@pytest.mark.parametrize("task_completed", [0, 1, "yes"])
def test_finalize_rejects_nonboolean_task_completed(tmp_path, task_completed):
    writer = TrialResultWriter(tmp_path)
    writer.start(trial=16, started_at=STARTED_AT)

    with pytest.raises(TypeError, match="task_completed"):
        writer.finalize(_finished_result(task_completed=task_completed))


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
