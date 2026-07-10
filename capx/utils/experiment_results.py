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
    _trial_from_path(path)
    return TrialResultWriter.open_existing(path).try_mark_parent_guard_killed(
        process_rc=process_rc, elapsed_seconds=elapsed_seconds
    )
    return True


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
    }


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
