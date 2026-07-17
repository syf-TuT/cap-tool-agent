"""Load and aggregate resilient per-trial experiment result records."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from capx.envs.trial_results import RunOutcome, TrialResultWriter, validate_trial_result


_LEGACY_DIRECTORY = re.compile(
    r"^trial_(?P<trial>\d+)_sandboxrc_(?P<sandbox_rc>-?\d+)_reward_"
    r"(?P<reward>-?\d+(?:\.\d+)?)_taskcompleted_(?P<task_completed>[01])$"
)
def load_trial_result(seed_output_dir: Path, trial: int) -> dict[str, Any]:
    """Load a canonical result, falling back safely to a historical directory.

    A present structured file is authoritative: malformed files are reported as
    invalid rather than silently replaced by an inferred legacy result.
    """

    directory = Path(seed_output_dir)
    result_path = directory / f"trial_{trial}_result.json"
    if result_path.exists():
        return _load_structured_result(result_path, trial)

    for item in directory.glob(f"trial_{trial:02d}_sandboxrc_*_reward_*_taskcompleted_*"):
        match = _LEGACY_DIRECTORY.fullmatch(item.name)
        if match and item.is_dir() and int(match["trial"]) == trial:
            return {
                "schema_version": None,
                "trial": trial,
                "run_outcome": RunOutcome.FINISHED.value,
                "failure_kind": None,
                "failure_stage": None,
                "failure_message": None,
                "reward": float(match["reward"]),
                "task_completed": bool(int(match["task_completed"])),
                "sandbox_rc": int(match["sandbox_rc"]),
                "result_source": "legacy",
                "diagnostic": "inferred from legacy trial directory name",
            }
    return _unknown_result(trial, "missing", "no structured result or legacy trial directory found")


def finalize_parent_guard_exit(result_path: Path, process_rc: int, elapsed_seconds: float) -> bool:
    """Mark a residual running result when the 480-second parent guard exits 124."""

    path = Path(result_path)
    if process_rc != 124 or not path.is_file():
        return False
    try:
        trial = _trial_from_path(path)
    except ValueError:
        return False
    return TrialResultWriter.open_existing(path).try_mark_parent_guard_killed(
        process_rc=process_rc, elapsed_seconds=elapsed_seconds, expected_trial=trial
    )


def aggregate_trial_results(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate outcomes without allowing failures to bias task metrics."""

    materialized = list(results)
    outcome_counts = Counter(str(result.get("run_outcome", "invalid_result")) for result in materialized)
    finished = [
        result for result in materialized if result.get("run_outcome") == RunOutcome.FINISHED.value
    ]
    rewards = [float(result["reward"]) for result in finished if result.get("reward") is not None]
    completed = [bool(result["task_completed"]) for result in finished if result.get("task_completed") is not None]
    infrastructure_failure_counts = {
        outcome: count for outcome, count in sorted(outcome_counts.items()) if outcome != RunOutcome.FINISHED.value
    }
    return {
        "trial_count": len(materialized),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "finished_count": len(finished),
        "average_reward": sum(rewards) / len(rewards) if rewards else None,
        "task_completion_rate": sum(completed) / len(completed) if completed else None,
        "infrastructure_failure_counts": infrastructure_failure_counts,
        **_retry_aware_summary(materialized),
    }


def _retry_aware_summary(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    attempts_by_trial = _attempts_by_trial(results)
    first_attempts = [
        attempts[0] for attempts in attempts_by_trial.values() if attempts
    ]
    latest_attempts = [
        attempts[-1] for attempts in attempts_by_trial.values() if attempts
    ]
    unique_trial_count = len(attempts_by_trial)
    first_attempt_success_count = sum(_attempt_succeeded(result) for result in first_attempts)
    latest_attempt_success_count = sum(_attempt_succeeded(result) for result in latest_attempts)

    return {
        "selection_policy": "all_attempts_grouped_by_trial",
        "unique_trial_count": unique_trial_count,
        "total_attempt_count": len(results),
        "first_attempt_success_count": first_attempt_success_count,
        "first_attempt_success_rate": (
            first_attempt_success_count / unique_trial_count if unique_trial_count else None
        ),
        "latest_attempt_success_count": latest_attempt_success_count,
        "latest_attempt_success_rate": (
            latest_attempt_success_count / unique_trial_count if unique_trial_count else None
        ),
        "success_by_retry_budget": _success_by_retry_budget(attempts_by_trial),
        "llm_logical_call_count": sum(_llm_int(result, "call_count") for result in results),
        "llm_attempt_count": sum(_llm_int(result, "attempt_count") for result in results),
        "llm_retry_count": sum(_llm_int(result, "retry_count") for result in results),
        "llm_token_count": sum(_llm_int(result, "token_count") for result in results),
        "llm_elapsed_seconds": sum(_llm_float(result, "elapsed_seconds") for result in results),
        "trial_elapsed_seconds": sum(_nonnegative_number(result.get("elapsed_seconds")) for result in results),
        "robot_execution_count": sum(_nonnegative_int_value(result.get("robot_execution_count")) for result in results),
        "provider_failure_count": sum(_is_provider_failure(result) for result in results),
        "algorithm_failure_count": sum(_is_algorithm_failure(result) for result in results),
        "budget_exhausted_count": sum(_is_budget_exhausted(result) for result in results),
        "experiment_infrastructure_failure_count": sum(
            _is_experiment_infrastructure_failure(result) for result in results
        ),
        "unclassified_failure_count": sum(_is_unclassified_failure(result) for result in results),
    }


def _attempts_by_trial(results: list[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for index, result in enumerate(results):
        trial_id = str(result.get("trial", index + 1))
        grouped.setdefault(trial_id, []).append(result)
    for attempts in grouped.values():
        attempts.sort(key=_attempt_sort_key)
    return dict(sorted(grouped.items()))


def _attempt_sort_key(result: Mapping[str, Any]) -> tuple[int, int]:
    attempt_index = result.get("attempt_index", result.get("attempt", 1))
    try:
        normalized_attempt = int(attempt_index)
    except (TypeError, ValueError):
        normalized_attempt = 1
    llm = result.get("llm")
    last_call_index = (
        _nonnegative_int_value(llm.get("last_call_index"))
        if isinstance(llm, Mapping)
        else 0
    )
    return normalized_attempt, last_call_index


def _success_by_retry_budget(
    attempts_by_trial: dict[str, list[Mapping[str, Any]]],
) -> dict[str, dict[str, float | int | None]]:
    budgets: dict[str, dict[str, float | int | None]] = {}
    if not attempts_by_trial:
        return budgets
    max_retry_budget = max(max(len(attempts) - 1, 0) for attempts in attempts_by_trial.values())
    trial_count = len(attempts_by_trial)
    for budget in range(max_retry_budget + 1):
        success_count = 0
        for attempts in attempts_by_trial.values():
            visible_attempts = attempts[: budget + 1]
            success_count += int(any(_attempt_succeeded(attempt) for attempt in visible_attempts))
        budgets[str(budget)] = {
            "success_count": success_count,
            "success_rate": success_count / trial_count if trial_count else None,
        }
    return budgets


def _attempt_succeeded(result: Mapping[str, Any]) -> bool:
    return (
        result.get("run_outcome") == RunOutcome.FINISHED.value
        and bool(result.get("task_completed"))
    )


def _llm_int(result: Mapping[str, Any], field: str) -> int:
    llm = result.get("llm")
    if not isinstance(llm, Mapping):
        return 0
    return _nonnegative_int_value(llm.get(field))


def _llm_float(result: Mapping[str, Any], field: str) -> float:
    llm = result.get("llm")
    if not isinstance(llm, Mapping):
        return 0.0
    return _nonnegative_number(llm.get(field))


def _nonnegative_int_value(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(number, 0)


def _nonnegative_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(number, 0.0)


def _is_provider_failure(result: Mapping[str, Any]) -> bool:
    if result.get("run_outcome") == RunOutcome.LLM_FAILED.value:
        return True
    failure_kind = str(result.get("failure_kind") or "")
    return failure_kind.startswith(("http_", "rate_limit", "timeout", "provider"))


def _is_budget_exhausted(result: Mapping[str, Any]) -> bool:
    return (
        result.get("run_outcome") == RunOutcome.TRIAL_BUDGET_EXHAUSTED.value
        or result.get("failure_kind") == RunOutcome.TRIAL_BUDGET_EXHAUSTED.value
    )


def _is_algorithm_failure(result: Mapping[str, Any]) -> bool:
    if (
        _is_provider_failure(result)
        or _is_budget_exhausted(result)
        or _is_experiment_infrastructure_failure(result)
    ):
        return False
    outcome = result.get("run_outcome")
    if outcome == RunOutcome.EXECUTION_FAILED.value:
        return True
    if (
        result.get("failure_kind") == "unknown_legacy_failure"
        and outcome not in {"missing_result", "invalid_result"}
    ):
        return True
    return outcome == RunOutcome.FINISHED.value and not bool(result.get("task_completed"))


def _is_experiment_infrastructure_failure(result: Mapping[str, Any]) -> bool:
    outcome = result.get("run_outcome")
    return outcome in {
        RunOutcome.PARENT_GUARD_KILLED.value,
        RunOutcome.CANCELLED.value,
        "missing_result",
        "invalid_result",
    }


def _is_unclassified_failure(result: Mapping[str, Any]) -> bool:
    outcome = result.get("run_outcome")
    if outcome == RunOutcome.FINISHED.value:
        return False
    return not (
        _is_provider_failure(result)
        or _is_algorithm_failure(result)
        or _is_budget_exhausted(result)
        or _is_experiment_infrastructure_failure(result)
    )


def _load_structured_result(path: Path, trial: int) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _unknown_result(trial, "corrupt", f"cannot parse structured result: {exc}")
    try:
        normalized = validate_trial_result(result)
    except (TypeError, ValueError) as exc:
        return _unknown_result(trial, "invalid_schema", f"invalid structured result schema: {exc}")
    if normalized["trial"] != trial:
        return _unknown_result(trial, "invalid_schema", "structured result trial does not match its path")
    return {**normalized, "result_source": "structured", "diagnostic": None}


def _unknown_result(trial: int, source: str, diagnostic: str) -> dict[str, Any]:
    return {
        "schema_version": None,
        "trial": trial,
        "run_outcome": "invalid_result" if source != "missing" else "missing_result",
        "failure_kind": "unknown_legacy_failure",
        "failure_stage": None,
        "failure_message": None,
        "reward": None,
        "task_completed": None,
        "sandbox_rc": None,
        "result_source": source,
        "diagnostic": diagnostic,
    }


def _trial_from_path(path: Path) -> int:
    match = re.fullmatch(r"trial_(\d+)_result\.json", path.name)
    if match is None:
        raise ValueError(f"not a canonical trial result filename: {path.name}")
    return int(match.group(1))
