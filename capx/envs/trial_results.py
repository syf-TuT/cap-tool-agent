"""Atomic, versioned persistence for per-trial execution outcomes."""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from capx.llm.errors import _sanitize_message


SCHEMA_VERSION = 1


class RunOutcome(str, Enum):
    """Stable execution outcomes, independent from task success."""

    RUNNING = "running"
    FINISHED = "finished"
    LLM_FAILED = "llm_failed"
    TRIAL_BUDGET_EXHAUSTED = "trial_budget_exhausted"
    EXECUTION_FAILED = "execution_failed"
    CANCELLED = "cancelled"
    PARENT_GUARD_KILLED = "parent_guard_killed"


_RESULT_FIELDS = {
    "schema_version",
    "trial",
    "run_outcome",
    "failure_kind",
    "failure_stage",
    "failure_message",
    "started_at",
    "finished_at",
    "elapsed_seconds",
    "reward",
    "task_completed",
    "sandbox_rc",
    "llm",
}
_LLM_FIELDS = {
    "call_count",
    "attempt_count",
    "retry_count",
    "elapsed_seconds",
    "last_call_index",
}


def _format_timestamp(value: datetime | str | None, *, field: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if not isinstance(value, datetime):
        raise TypeError(f"{field} must be a datetime, string, or None")
    if value.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _nonnegative_float(value: Any, *, field: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return number


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    number = int(value)
    if number < 0 or number != value:
        raise ValueError(f"{field} must be a non-negative integer")
    return number


def _empty_llm_accounting() -> dict[str, int | float]:
    return {
        "call_count": 0,
        "attempt_count": 0,
        "retry_count": 0,
        "elapsed_seconds": 0.0,
        "last_call_index": 0,
    }


def _safe_message(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value
    return _sanitize_message(value)


def _normalize_llm_accounting(value: Mapping[str, Any]) -> dict[str, int | float]:
    unknown = set(value) - _LLM_FIELDS
    if unknown:
        raise ValueError(f"unknown LLM accounting fields: {sorted(unknown)}")
    merged = {**_empty_llm_accounting(), **value}
    return {
        "call_count": _nonnegative_int(merged["call_count"], field="llm.call_count"),
        "attempt_count": _nonnegative_int(
            merged["attempt_count"], field="llm.attempt_count"
        ),
        "retry_count": _nonnegative_int(merged["retry_count"], field="llm.retry_count"),
        "elapsed_seconds": _nonnegative_float(
            merged["elapsed_seconds"], field="llm.elapsed_seconds"
        ),
        "last_call_index": _nonnegative_int(
            merged["last_call_index"], field="llm.last_call_index"
        ),
    }


def _normalize_result(value: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(value) - _RESULT_FIELDS
    if unknown:
        raise ValueError(f"unknown trial result fields: {sorted(unknown)}")

    outcome = RunOutcome(value["run_outcome"])
    failure_message = value.get("failure_message")
    return {
        "schema_version": int(value["schema_version"]),
        "trial": int(value["trial"]),
        "run_outcome": outcome.value,
        "failure_kind": (
            None if value.get("failure_kind") is None else _safe_message(value["failure_kind"])
        ),
        "failure_stage": (
            None if value.get("failure_stage") is None else _safe_message(value["failure_stage"])
        ),
        "failure_message": None if failure_message is None else _safe_message(failure_message),
        "started_at": _format_timestamp(value.get("started_at"), field="started_at"),
        "finished_at": _format_timestamp(value.get("finished_at"), field="finished_at"),
        "elapsed_seconds": _nonnegative_float(
            value.get("elapsed_seconds", 0.0), field="elapsed_seconds"
        ),
        "reward": None if value.get("reward") is None else float(value["reward"]),
        "task_completed": value.get("task_completed"),
        "sandbox_rc": None if value.get("sandbox_rc") is None else int(value["sandbox_rc"]),
        "llm": _normalize_llm_accounting(value.get("llm", {})),
    }


class TrialResultWriter:
    """Persist a single trial's running and terminal states atomically."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self._path: Path | None = None

    @property
    def path(self) -> Path:
        if self._path is None:
            raise RuntimeError("trial result has not been started")
        return self._path

    def start(self, *, trial: int, started_at: datetime) -> Path:
        """Create the canonical running record before any trial work begins."""

        if self._path is not None:
            raise RuntimeError("this writer has already been started")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"trial_{int(trial)}_result.json"
        if path.exists():
            raise ValueError(f"trial result already exists: {path}")

        result = _normalize_result(
            {
                "schema_version": SCHEMA_VERSION,
                "trial": trial,
                "run_outcome": RunOutcome.RUNNING,
                "failure_kind": None,
                "failure_stage": None,
                "failure_message": None,
                "started_at": started_at,
                "finished_at": None,
                "elapsed_seconds": 0.0,
                "reward": None,
                "task_completed": None,
                "sandbox_rc": None,
                "llm": _empty_llm_accounting(),
            }
        )
        self._atomic_write(path, result)
        self._path = path
        return path

    def finalize(self, result: Mapping[str, Any]) -> None:
        """Replace a running result with a terminal result.

        Replaying the same terminal result is idempotent. A different terminal
        result is rejected so cleanup code cannot overwrite established evidence.
        """

        requested_outcome = RunOutcome(result["run_outcome"])
        if requested_outcome is RunOutcome.RUNNING:
            raise ValueError("finalize requires a terminal outcome")

        current = self._read()
        candidate = _normalize_result({**current, **result})
        if current["run_outcome"] != RunOutcome.RUNNING.value:
            if candidate == current:
                return
            raise ValueError("trial result is already terminal")
        if candidate["schema_version"] != SCHEMA_VERSION or candidate["trial"] != current["trial"]:
            raise ValueError("schema_version and trial cannot change during finalization")
        if candidate["finished_at"] is None:
            raise ValueError("a terminal result requires finished_at")

        self._atomic_write(self.path, candidate)

    def mark_parent_guard_killed(self, *, process_rc: int, elapsed_seconds: float) -> None:
        """Finalize a residual running result after its parent guard kills it."""

        self.finalize(
            {
                "run_outcome": RunOutcome.PARENT_GUARD_KILLED,
                "failure_kind": RunOutcome.PARENT_GUARD_KILLED.value,
                "failure_stage": None,
                "failure_message": f"Parent process guard exited with return code {process_rc}",
                "finished_at": datetime.now(timezone.utc),
                "elapsed_seconds": elapsed_seconds,
                "sandbox_rc": process_rc,
            }
        )

    def _read(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return _normalize_result(value)

    @staticmethod
    def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
            TrialResultWriter._fsync_directory(path.parent)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        """Best-effort directory fsync; unsupported on some platforms."""

        descriptor: int | None = None
        try:
            descriptor = os.open(directory, os.O_RDONLY)
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            if descriptor is not None:
                os.close(descriptor)
