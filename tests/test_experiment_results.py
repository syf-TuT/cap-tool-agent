"""Tests for loading versioned and historical experiment trial results."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from capx.envs.trial_results import RunOutcome, TrialResultWriter
from capx.utils.experiment_results import (
    aggregate_trial_results,
    finalize_parent_guard_exit,
    load_trial_result,
)


def _write_result(directory, trial, *, outcome, reward=None, task_completed=None):
    writer = TrialResultWriter(directory)
    path = writer.start(trial=trial, started_at=datetime(2026, 7, 10, tzinfo=timezone.utc))
    writer.finalize(
        {
            "run_outcome": outcome,
            "finished_at": datetime(2026, 7, 10, 0, 1, tzinfo=timezone.utc),
            "elapsed_seconds": 60.0,
            "reward": reward,
            "task_completed": task_completed,
        }
    )
    return path


def test_load_prefers_schema_v1_result_over_legacy_directory(tmp_path):
    _write_result(
        tmp_path,
        7,
        outcome=RunOutcome.LLM_FAILED,
        reward=0.8,
        task_completed=True,
    )
    (tmp_path / "trial_07_sandboxrc_0_reward_1.000_taskcompleted_1").mkdir()

    result = load_trial_result(tmp_path, 7)

    assert result["run_outcome"] == "llm_failed"
    assert result["result_source"] == "structured"
    assert result["reward"] == 0.8


def test_load_legacy_directory_as_finished_without_inventing_failure(tmp_path):
    (tmp_path / "trial_03_sandboxrc_1_reward_0.250_taskcompleted_0").mkdir()

    result = load_trial_result(tmp_path, 3)

    assert result == {
        "schema_version": None,
        "trial": 3,
        "run_outcome": "finished",
        "failure_kind": None,
        "failure_stage": None,
        "failure_message": None,
        "reward": 0.25,
        "task_completed": False,
        "sandbox_rc": 1,
        "result_source": "legacy",
        "diagnostic": "inferred from legacy trial directory name",
    }


@pytest.mark.parametrize(
    ("contents", "outcome", "source"),
    [
        (None, "missing_result", "missing"),
        ("{not json", "invalid_result", "corrupt"),
        (json.dumps({"schema_version": 99}), "invalid_result", "invalid_schema"),
    ],
)
def test_missing_or_invalid_result_is_explicit_and_keeps_diagnostic(
    tmp_path, contents, outcome, source
):
    path = tmp_path / "trial_9_result.json"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")

    result = load_trial_result(tmp_path, 9)

    assert result["run_outcome"] == outcome
    assert result["result_source"] == source
    assert result["diagnostic"]
    assert result["failure_kind"] == "unknown_legacy_failure"


def test_truncated_schema_v1_result_is_not_accepted_as_structured(tmp_path):
    (tmp_path / "trial_10_result.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trial": 10,
                "run_outcome": "running",
                "reward": None,
                "task_completed": None,
                "sandbox_rc": None,
            }
        ),
        encoding="utf-8",
    )

    result = load_trial_result(tmp_path, 10)

    assert result["run_outcome"] == "invalid_result"
    assert result["result_source"] == "invalid_schema"
    assert "schema" in result["diagnostic"]


def test_non_utf8_structured_result_is_reported_as_corrupt(tmp_path):
    (tmp_path / "trial_11_result.json").write_bytes(b"\xff\xfe")

    result = load_trial_result(tmp_path, 11)

    assert result["run_outcome"] == "invalid_result"
    assert result["result_source"] == "corrupt"


def test_terminal_result_without_finished_timestamp_is_invalid_schema(tmp_path):
    path = tmp_path / "trial_12_result.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trial": 12,
                "run_outcome": "finished",
                "failure_kind": None,
                "failure_stage": None,
                "failure_message": None,
                "started_at": "2026-07-10T00:00:00Z",
                "finished_at": None,
                "elapsed_seconds": 1.0,
                "reward": 1.0,
                "task_completed": True,
                "sandbox_rc": 0,
                "llm": {
                    "call_count": 0,
                    "attempt_count": 0,
                    "retry_count": 0,
                    "token_count": 0,
                    "elapsed_seconds": 0.0,
                    "last_call_index": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    result = load_trial_result(tmp_path, 12)

    assert result["run_outcome"] == "invalid_result"
    assert result["result_source"] == "invalid_schema"
    assert "finished_at" in result["diagnostic"]


def test_load_structured_result_accepts_llm_token_count(tmp_path):
    writer = TrialResultWriter(tmp_path)
    path = writer.start(trial=16, started_at=datetime(2026, 7, 10, tzinfo=timezone.utc))
    writer.finalize(
        {
            "run_outcome": RunOutcome.FINISHED,
            "finished_at": datetime(2026, 7, 10, 0, 1, tzinfo=timezone.utc),
            "elapsed_seconds": 60.0,
            "reward": 1.0,
            "task_completed": True,
            "sandbox_rc": 0,
            "llm": {
                "call_count": 1,
                "attempt_count": 1,
                "retry_count": 0,
                "token_count": 123,
                "elapsed_seconds": 2.0,
                "last_call_index": 1,
            },
        }
    )

    assert load_trial_result(path.parent, 16)["llm"]["token_count"] == 123


def test_parent_guard_finalizes_only_residual_running_result(tmp_path):
    writer = TrialResultWriter(tmp_path)
    running_path = writer.start(trial=1, started_at=datetime(2026, 7, 10, tzinfo=timezone.utc))
    finished_path = _write_result(
        tmp_path, 2, outcome=RunOutcome.FINISHED, reward=1.0, task_completed=True
    )

    assert finalize_parent_guard_exit(running_path, process_rc=124, elapsed_seconds=480.0) is True
    assert finalize_parent_guard_exit(finished_path, process_rc=124, elapsed_seconds=480.0) is False

    assert json.loads(running_path.read_text())["run_outcome"] == "parent_guard_killed"
    assert json.loads(finished_path.read_text())["run_outcome"] == "finished"


@pytest.mark.parametrize(
    "contents",
    [
        "{not json",
        json.dumps(
            {
                "schema_version": 1,
                "trial": 13,
                "run_outcome": "running",
            }
        ),
    ],
)
def test_parent_guard_returns_false_for_invalid_result_and_preserves_diagnostic(tmp_path, contents):
    path = tmp_path / "trial_13_result.json"
    path.write_text(contents, encoding="utf-8")

    assert finalize_parent_guard_exit(path, process_rc=124, elapsed_seconds=480.0) is False
    assert path.read_text(encoding="utf-8") == contents
    assert load_trial_result(tmp_path, 13)["result_source"] in {"corrupt", "invalid_schema"}


def test_parent_guard_returns_false_when_path_trial_differs_from_payload_trial(tmp_path):
    path = _write_result(
        tmp_path, 14, outcome=RunOutcome.FINISHED, reward=1.0, task_completed=True
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    value["trial"] = 15
    value["run_outcome"] = "running"
    value["finished_at"] = None
    path.write_text(json.dumps(value), encoding="utf-8")

    assert finalize_parent_guard_exit(path, process_rc=124, elapsed_seconds=480.0) is False
    assert json.loads(path.read_text(encoding="utf-8"))["trial"] == 15


def test_aggregate_counts_outcomes_and_uses_finished_only_denominators():
    summary = aggregate_trial_results(
        [
            {"run_outcome": "finished", "reward": 1.0, "task_completed": True},
            {"run_outcome": "finished", "reward": 0.5, "task_completed": False},
            {"run_outcome": "llm_failed", "reward": 100.0, "task_completed": True},
            {"run_outcome": "unknown_legacy_failure", "reward": 100.0, "task_completed": True},
        ]
    )

    assert summary["outcome_counts"] == {
        "finished": 2,
        "llm_failed": 1,
        "unknown_legacy_failure": 1,
    }
    assert summary["finished_count"] == 2
    assert summary["average_reward"] == pytest.approx(0.75)
    assert summary["task_completion_rate"] == pytest.approx(0.5)
    assert summary["infrastructure_failure_counts"] == {
        "llm_failed": 1,
        "unknown_legacy_failure": 1,
    }


def test_aggregate_reports_first_attempt_and_retry_aware_metrics():
    summary = aggregate_trial_results(
        [
            {
                "trial": 1,
                "attempt_index": 1,
                "run_outcome": "execution_failed",
                "failure_kind": "sandbox_rc_nonzero",
                "reward": 0.0,
                "task_completed": False,
                "elapsed_seconds": 10.0,
                "robot_execution_count": 1,
                "llm": {
                    "call_count": 2,
                    "attempt_count": 2,
                    "retry_count": 0,
                    "elapsed_seconds": 7.0,
                    "token_count": 100,
                },
            },
            {
                "trial": 1,
                "attempt_index": 2,
                "run_outcome": "finished",
                "failure_kind": None,
                "reward": 1.0,
                "task_completed": True,
                "elapsed_seconds": 12.0,
                "robot_execution_count": 2,
                "llm": {
                    "call_count": 1,
                    "attempt_count": 1,
                    "retry_count": 0,
                    "elapsed_seconds": 5.0,
                    "token_count": 80,
                },
            },
            {
                "trial": 2,
                "attempt_index": 1,
                "run_outcome": "llm_failed",
                "failure_kind": "http_5xx",
                "elapsed_seconds": 3.0,
                "llm": {
                    "call_count": 1,
                    "attempt_count": 3,
                    "retry_count": 2,
                    "elapsed_seconds": 3.0,
                    "token_count": 0,
                },
            },
            {
                "trial": 2,
                "attempt_index": 2,
                "run_outcome": "finished",
                "failure_kind": None,
                "reward": 1.0,
                "task_completed": True,
                "elapsed_seconds": 11.0,
                "robot_execution_count": 1,
                "llm": {
                    "call_count": 1,
                    "attempt_count": 1,
                    "retry_count": 0,
                    "elapsed_seconds": 4.0,
                    "token_count": 60,
                },
            },
            {
                "trial": 3,
                "attempt_index": 1,
                "run_outcome": "trial_budget_exhausted",
                "failure_kind": "trial_budget_exhausted",
                "elapsed_seconds": 20.0,
                "llm": {
                    "call_count": 4,
                    "attempt_count": 4,
                    "retry_count": 0,
                    "elapsed_seconds": 18.0,
                    "token_count": 200,
                },
            },
        ]
    )

    assert summary["selection_policy"] == "all_attempts_grouped_by_trial"
    assert summary["unique_trial_count"] == 3
    assert summary["total_attempt_count"] == 5
    assert summary["first_attempt_success_count"] == 0
    assert summary["first_attempt_success_rate"] == pytest.approx(0.0)
    assert summary["latest_attempt_success_count"] == 2
    assert summary["success_by_retry_budget"] == {
        "0": {"success_count": 0, "success_rate": pytest.approx(0.0)},
        "1": {"success_count": 2, "success_rate": pytest.approx(2 / 3)},
    }
    assert summary["llm_logical_call_count"] == 9
    assert summary["llm_attempt_count"] == 11
    assert summary["llm_retry_count"] == 2
    assert summary["llm_token_count"] == 440
    assert summary["llm_elapsed_seconds"] == pytest.approx(37.0)
    assert summary["trial_elapsed_seconds"] == pytest.approx(56.0)
    assert summary["robot_execution_count"] == 4
    assert summary["provider_failure_count"] == 1
    assert summary["algorithm_failure_count"] == 1
    assert summary["budget_exhausted_count"] == 1
    assert summary["experiment_infrastructure_failure_count"] == 0
    assert summary["unclassified_failure_count"] == 0


def test_aggregate_failure_taxonomy_covers_infrastructure_and_legacy_unknowns():
    summary = aggregate_trial_results(
        [
            {
                "trial": 1,
                "run_outcome": "execution_failed",
                "failure_kind": "unknown_legacy_failure",
            },
            {
                "trial": 2,
                "run_outcome": "parent_guard_killed",
                "failure_kind": "parent_guard_killed",
            },
            {
                "trial": 3,
                "run_outcome": "cancelled",
                "failure_kind": "cancelled",
            },
        ]
    )

    assert summary["algorithm_failure_count"] == 1
    assert summary["experiment_infrastructure_failure_count"] == 2
    assert summary["provider_failure_count"] == 0
    assert summary["budget_exhausted_count"] == 0
    assert summary["unclassified_failure_count"] == 0


def test_aggregate_loaded_missing_and_invalid_results_as_infrastructure(tmp_path):
    invalid_dir = tmp_path / "invalid"
    invalid_dir.mkdir()
    (invalid_dir / "trial_2_result.json").write_text("{not json", encoding="utf-8")

    summary = aggregate_trial_results(
        [
            load_trial_result(tmp_path / "missing", 1),
            load_trial_result(invalid_dir, 2),
        ]
    )

    assert summary["experiment_infrastructure_failure_count"] == 2
    assert summary["algorithm_failure_count"] == 0
    assert summary["unclassified_failure_count"] == 0


def test_failure_taxonomy_uses_single_priority_bucket_per_attempt():
    summary = aggregate_trial_results(
        [
            {
                "trial": 1,
                "run_outcome": "trial_budget_exhausted",
                "failure_kind": "timeout",
            },
            {
                "trial": 2,
                "run_outcome": "parent_guard_killed",
                "failure_kind": "http_5xx",
            },
        ]
    )

    assert summary["budget_exhausted_count"] == 1
    assert summary["experiment_infrastructure_failure_count"] == 1
    assert summary["provider_failure_count"] == 0
    assert summary["algorithm_failure_count"] == 0
    assert summary["unclassified_failure_count"] == 0
