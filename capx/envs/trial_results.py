"""Atomic, versioned persistence for per-trial execution outcomes."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

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
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _trial_id(value: Any) -> int:
    return _nonnegative_int(value, field="trial")


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _optional_bool(value: Any, *, field: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be a boolean or None")
    return value


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
        "schema_version": _nonnegative_int(value["schema_version"], field="schema_version"),
        "trial": _trial_id(value["trial"]),
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
        "reward": (
            None
            if value.get("reward") is None
            else _finite_float(value["reward"], field="reward")
        ),
        "task_completed": _optional_bool(value.get("task_completed"), field="task_completed"),
        "sandbox_rc": None if value.get("sandbox_rc") is None else int(value["sandbox_rc"]),
        "llm": _normalize_llm_accounting(value.get("llm", {})),
    }


def validate_trial_result(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a complete on-disk schema-v1 result record.

    This is intentionally stricter than ``finalize`` merging a partial update:
    loaders must reject truncated or forward-incompatible files rather than
    silently manufacturing defaults.
    """

    if not isinstance(value, Mapping):
        raise TypeError("trial result must be a mapping")
    missing = _RESULT_FIELDS - set(value)
    if missing:
        raise ValueError(f"missing trial result fields: {sorted(missing)}")
    normalized = _normalize_result(value)
    if normalized["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {normalized['schema_version']}")
    return normalized


class TrialResultWriter:
    """Persist a single trial's running and terminal states atomically."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self._path: Path | None = None

    @classmethod
    def open_existing(cls, result_path: str | Path) -> "TrialResultWriter":
        """Attach a writer to an already-created canonical result file."""

        path = Path(result_path)
        writer = cls(path.parent)
        writer._path = path
        return writer

    @property
    def path(self) -> Path:
        if self._path is None:
            raise RuntimeError("trial result has not been started")
        return self._path

    def start(self, *, trial: int, started_at: datetime) -> Path:
        """Create the canonical running record before any trial work begins."""

        if self._path is not None:
            raise RuntimeError("this writer has already been started")
        trial = _trial_id(trial)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"trial_{trial}_result.json"

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
        with self._exclusive_lock(path):
            if path.exists():
                raise ValueError(f"trial result already exists: {path}")
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

        with self._exclusive_lock(self.path):
            current = self._read()
            candidate = _normalize_result({**current, **result})
            if current["run_outcome"] != RunOutcome.RUNNING.value:
                if candidate == current:
                    return
                raise ValueError("trial result is already terminal")
            if (
                candidate["schema_version"] != SCHEMA_VERSION
                or candidate["trial"] != current["trial"]
            ):
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
    @contextmanager
    def _exclusive_lock(result_path: Path) -> Iterator[None]:
        """Hold an OS-managed cross-process lock for one result transition."""

        lock_path = result_path.with_name(f".{result_path.name}.lock")
        # Keep this stable sidecar: unlinking it can let a waiter and a newcomer
        # lock different inodes, recreating the finalization race.
        with lock_path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

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
