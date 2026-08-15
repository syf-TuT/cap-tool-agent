from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from capx.envs import launch as launch_module
from capx.envs import runner
from capx.envs.infrastructure import InfrastructureFailure, ServiceReadinessError
from capx.llm.context import llm_call_stage
from capx.llm.errors import LLMErrorKind, LLMQueryError
from capx.llm import client as llm_client
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


def test_infrastructure_failure_preserves_stable_diagnostics():
    module_spec = importlib.util.find_spec("capx.envs.infrastructure")
    assert module_spec is not None

    from capx.envs.infrastructure import InfrastructureFailure

    failure = InfrastructureFailure(
        "service_timeout",
        "SAM3 timed out",
        evidence={"service": "sam3"},
    )

    assert failure.kind == "service_timeout"
    assert failure.message == "SAM3 timed out"
    assert failure.evidence == {"service": "sam3"}


def test_required_service_endpoints_include_libero_molmo():
    endpoints = runner._required_service_endpoints(
        [{"host": "127.0.0.1", "port": 8114}],
        {
            "cfg": {
                "apis": ["FrankaLiberoApi"],
                "molmo_base_url": "http://127.0.0.1:8122/v1",
            }
        },
    )

    assert [(item.host, item.port) for item in endpoints] == [
        ("127.0.0.1", 8114),
        ("127.0.0.1", 8122),
    ]


def test_required_service_endpoints_ignore_molmo_for_other_apis():
    endpoints = runner._required_service_endpoints(
        [],
        {
            "cfg": {
                "apis": ["OtherApi"],
                "molmo_base_url": "http://molmo.example.test:8122/v1",
            }
        },
    )

    assert endpoints == []


def test_start_api_servers_stops_started_processes_on_readiness_timeout(monkeypatch):
    proc = SimpleNamespace(terminate=Mock(), join=Mock())
    endpoint = SimpleNamespace(name="sam3", host="127.0.0.1", port=8114)
    monkeypatch.setattr(runner, "run_server_proc", lambda config: proc)
    monkeypatch.setattr(runner, "_service_endpoint_ready", lambda item: False)
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)

    with pytest.raises(ServiceReadinessError, match="not ready"):
        runner._start_api_servers(
            [{"host": "127.0.0.1", "port": 8114}],
            required_endpoints=[endpoint],
            wait_timeout=0,
        )

    proc.terminate.assert_called_once_with()
    proc.join.assert_called_once_with(timeout=5.0)


def test_start_api_servers_checks_external_required_endpoint(monkeypatch):
    endpoint = SimpleNamespace(name="molmo", host="127.0.0.1", port=8122)
    checked = []
    monkeypatch.setattr(
        runner,
        "_service_endpoint_ready",
        lambda item: checked.append(item) or True,
    )

    assert runner._start_api_servers([], required_endpoints=[endpoint]) == []
    assert checked == [endpoint]


def test_launch_preflights_all_required_endpoints_before_trials(monkeypatch):
    env_factory = {"cfg": {"apis": ["FrankaLiberoApi"]}}
    api_servers = [{"host": "127.0.0.1", "port": 8114}]
    endpoints = [SimpleNamespace(name="sam3", host="127.0.0.1", port=8114)]
    calls = []

    monkeypatch.setattr(
        launch_module,
        "_load_config",
        lambda args: (env_factory, {"web_ui": False}, api_servers),
    )
    monkeypatch.setattr(
        runner,
        "_required_service_endpoints",
        lambda servers, factory: calls.append((servers, factory)) or endpoints,
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "_start_api_servers",
        lambda servers, *, required_endpoints: (
            calls.append((servers, required_endpoints)) or []
        ),
    )
    monkeypatch.setattr(
        runner,
        "_run_headless_trials",
        lambda *args: calls.append("trials"),
    )
    monkeypatch.setattr(runner, "_stop_api_servers", lambda procs: None)

    launch_module.main(SimpleNamespace())

    assert calls == [
        (api_servers, env_factory),
        (api_servers, endpoints),
        "trials",
    ]


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


def test_typed_infrastructure_failure_is_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner,
        "_run_single_trial",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            InfrastructureFailure("service_timeout", "SAM3 timed out")
        ),
    )

    summary = runner._run_trial_with_retries(
        object(), 1, _args(), _config(tmp_path), None
    )

    result = json.loads((tmp_path / "trial_1_result.json").read_text())
    assert summary.run_outcome == "infrastructure_failed"
    assert summary.failure_kind == "service_timeout"
    assert result["run_outcome"] == "infrastructure_failed"


def test_all_llm_failed_ensemble_is_classified_as_llm_failure(tmp_path, monkeypatch):
    error = LLMQueryError(
        kind=LLMErrorKind.HTTP_5XX,
        call_index=2,
        attempt=2,
        status_code=503,
        elapsed_seconds=1.0,
        message="provider unavailable",
    )
    llm_args = SimpleNamespace(
        model="test-model",
        server_url="http://localhost:1",
        api_key=None,
        temperature=0.2,
        max_tokens=32,
        reasoning_effort="minimal",
    )
    monkeypatch.setattr(llm_client, "ENSEMBLE_CONFIGS", [("test-model", [0.1])])
    monkeypatch.setattr(
        llm_client,
        "query_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        runner,
        "_run_single_trial",
        lambda *args, **kwargs: llm_client.query_model_ensemble(
            llm_args, [{"role": "user", "content": "hi"}]
        ),
    )

    summary = runner._run_trial_with_retries(object(), 1, _args(), _config(tmp_path), None)

    assert summary.run_outcome == "llm_failed"
    assert json.loads((tmp_path / "trial_1_result.json").read_text())["failure_kind"] == "http_5xx"


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


def test_timeout_summary_uses_the_same_resolved_budget_as_the_alarm(tmp_path, monkeypatch):
    def change_environment_then_timeout(*args, **kwargs):
        monkeypatch.setenv("CAPX_TRIAL_TIMEOUT_SECONDS", "999")
        raise TimeoutError("budget exhausted")

    monkeypatch.setattr(runner, "_run_single_trial", change_environment_then_timeout)

    summary = runner._run_single_trial_with_timeout(
        object(), 1, _args(), _config(tmp_path), None, timeout_s=17
    )

    assert "timed out after 17 seconds" in summary.log


def test_result_write_failure_after_finished_trial_propagates_without_reclassification(
    tmp_path, monkeypatch
):
    calls = 0

    def fail_finalize(self, result):
        nonlocal calls
        calls += 1
        raise OSError("disk full")

    monkeypatch.setattr(runner, "_run_single_trial", lambda *args, **kwargs: _summary())
    monkeypatch.setattr(runner.TrialResultWriter, "finalize", fail_finalize)

    with pytest.raises(OSError, match="disk full") as raised:
        runner._run_trial_with_retries(object(), 1, _args(), _config(tmp_path), None)

    assert calls == 1
    assert raised.value.__cause__ is None


def test_result_write_failure_chains_from_original_llm_failure(tmp_path, monkeypatch):
    original = LLMQueryError(
        kind=LLMErrorKind.HTTP_5XX,
        call_index=2,
        attempt=2,
        status_code=503,
        elapsed_seconds=3.0,
        message="provider unavailable",
    )

    monkeypatch.setattr(
        runner, "_run_single_trial", lambda *args, **kwargs: (_ for _ in ()).throw(original)
    )
    monkeypatch.setattr(
        runner.TrialResultWriter,
        "finalize",
        lambda self, result: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(LLMQueryError, match="provider unavailable") as raised:
        runner._run_trial_with_retries(object(), 1, _args(), _config(tmp_path), None)

    assert isinstance(raised.value.__cause__, OSError)
    assert str(raised.value.__cause__) == "disk full"


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


@pytest.mark.parametrize(
    "preclassified_outcome",
    ["execution_failed", "trial_budget_exhausted"],
)
def test_runner_preserves_preclassified_trial_summary_outcome(
    tmp_path, monkeypatch, preclassified_outcome
):
    def preclassified_trial(*args, **kwargs):
        summary = _summary(reward=0.0, completed=False)
        summary.run_outcome = preclassified_outcome
        return summary

    monkeypatch.setattr(runner, "_run_single_trial", preclassified_trial)

    summary = runner._run_single_trial_with_timeout(
        object(), 1, _args(), _config(tmp_path), None, timeout_s=450
    )
    result = json.loads((tmp_path / "trial_1_result.json").read_text())

    assert summary.run_outcome == preclassified_outcome
    assert result["run_outcome"] == preclassified_outcome


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
