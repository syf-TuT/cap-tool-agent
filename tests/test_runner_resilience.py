from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from capx.envs import runner
from capx.llm.context import llm_call_stage
from capx.llm.errors import LLMErrorKind, LLMQueryError
from capx.utils.launch_utils import TrialSummary, _print_and_save_summary


def _summary(*, reward: float = 1.0, completed: bool = True) -> TrialSummary:
    return TrialSummary(
        trial=1,
        success=completed,
        reward=reward,
        terminated=completed,
        truncated=False,
        sandbox_rc=0 if completed else 1,
        log="normal trial",
        task_completed=completed,
    )


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        model="test-model",
        visual_differencing_model="visual-model",
        config_path="test.yaml",
    )


def _config(tmp_path) -> dict:
    return {"output_dir": str(tmp_path), "use_img_differencing": False}


def test_default_complete_trial_timeout_is_450_seconds(monkeypatch):
    monkeypatch.delenv("CAPX_TRIAL_TIMEOUT_SECONDS", raising=False)

    assert runner._trial_timeout_seconds() == 450


def test_runner_writes_running_result_before_invoking_trial(tmp_path, monkeypatch):
    observed = {}

    def fake_trial(*args, **kwargs):
        path = tmp_path / "trial_1_result.json"
        observed["outcome"] = json.loads(path.read_text())["run_outcome"]
        return _summary()

    monkeypatch.setattr(runner, "_run_single_trial", fake_trial)

    summary = runner._run_single_trial_with_timeout(
        object(), 1, _args(), _config(tmp_path), None, timeout_s=450
    )

    assert observed["outcome"] == "running"
    assert summary.run_outcome == "finished"
    assert json.loads((tmp_path / "trial_1_result.json").read_text())["run_outcome"] == "finished"


def test_typed_llm_failure_is_persisted_without_retrying_whole_trial(tmp_path, monkeypatch):
    calls = 0

    def fail_with_llm(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise LLMQueryError(
            kind=LLMErrorKind.HTTP_5XX,
            call_index=2,
            attempt=2,
            status_code=503,
            elapsed_seconds=3.0,
            message="provider unavailable",
        )

    monkeypatch.setattr(runner, "_run_single_trial", fail_with_llm)

    summary = runner._run_trial_with_retries(object(), 1, _args(), _config(tmp_path), None)

    result = json.loads((tmp_path / "trial_1_result.json").read_text())
    assert calls == 1
    assert summary.run_outcome == "llm_failed"
    assert summary.failure_kind == "http_5xx"
    assert result["run_outcome"] == "llm_failed"
    assert result["failure_kind"] == "http_5xx"


def test_typed_trial_budget_llm_error_is_classified_as_budget_exhausted(tmp_path, monkeypatch):
    def fail_with_budget_error(*args, **kwargs):
        raise LLMQueryError(
            kind=LLMErrorKind.TRIAL_BUDGET_EXHAUSTED,
            call_index=2,
            attempt=1,
            status_code=None,
            elapsed_seconds=3.0,
            message="trial budget exhausted",
        )

    monkeypatch.setattr(runner, "_run_single_trial", fail_with_budget_error)

    summary = runner._run_trial_with_retries(object(), 1, _args(), _config(tmp_path), None)

    result = json.loads((tmp_path / "trial_1_result.json").read_text())
    assert summary.run_outcome == "trial_budget_exhausted"
    assert result["run_outcome"] == "trial_budget_exhausted"


def test_llm_failure_retains_active_call_stage_in_summary_and_result(tmp_path, monkeypatch):
    def fail_from_capsule_action(*args, **kwargs):
        with llm_call_stage("capsule_action"):
            raise LLMQueryError(
                kind=LLMErrorKind.HTTP_5XX,
                call_index=2,
                attempt=2,
                status_code=503,
                elapsed_seconds=3.0,
                message="provider unavailable",
            )

    monkeypatch.setattr(runner, "_run_single_trial", fail_from_capsule_action)

    summary = runner._run_trial_with_retries(object(), 1, _args(), _config(tmp_path), None)

    result = json.loads((tmp_path / "trial_1_result.json").read_text())
    assert summary.failure_stage == "capsule_action"
    assert result["failure_stage"] == "capsule_action"


def test_timeout_is_persisted_as_trial_budget_exhausted(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner,
        "_run_single_trial",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("budget exhausted")),
    )

    summary = runner._run_trial_with_retries(object(), 1, _args(), _config(tmp_path), None)

    assert summary.run_outcome == "trial_budget_exhausted"
    assert json.loads((tmp_path / "trial_1_result.json").read_text())["run_outcome"] == (
        "trial_budget_exhausted"
    )


def test_unexpected_exception_is_execution_failed_not_llm_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner,
        "_run_single_trial",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("environment blew up")),
    )

    summary = runner._run_trial_with_retries(object(), 1, _args(), _config(tmp_path), None)

    assert summary.run_outcome == "execution_failed"
    assert summary.failure_kind == "RuntimeError"


def test_normal_task_failure_is_finished_not_infrastructure_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner, "_run_single_trial", lambda *args, **kwargs: _summary(reward=0.0, completed=False)
    )

    summary = runner._run_trial_with_retries(object(), 1, _args(), _config(tmp_path), None)

    assert summary.run_outcome == "finished"
    assert summary.task_completed is False
    assert summary.failure_kind is None


def test_summary_uses_only_finished_trials_for_reward_and_reports_outcomes(tmp_path, capsys):
    summaries = [
        _summary(reward=1.0, completed=True),
        TrialSummary(
            trial=2, success=False, reward=99.0, terminated=False, truncated=True,
            sandbox_rc=1, log="llm failed", task_completed=False, run_outcome="llm_failed",
        ),
    ]

    _print_and_save_summary(summaries, _args(), _config(tmp_path), start_time=0.0)

    output = capsys.readouterr().out
    saved = (tmp_path / "summaries.txt").read_text()
    assert "1.000/1.000/1" in output
    assert "Outcome counts: finished=1, llm_failed=1" in output
    assert "Outcome counts: finished=1, llm_failed=1" in saved
