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
