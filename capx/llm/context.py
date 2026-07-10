"""Trial-scoped LLM budget, accounting, and attempt telemetry."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Callable, Iterator


_DEFAULT_STAGE = "unknown"
_active_context: ContextVar[TrialLLMContext | None] = ContextVar(
    "active_trial_llm_context", default=None
)
_active_stage: ContextVar[str] = ContextVar("active_llm_call_stage", default=_DEFAULT_STAGE)


@dataclass
class TrialLLMContext:
    """Mutable accounting shared by all LLM calls within one trial."""

    trial: int | str
    deadline_monotonic: float | None = None
    telemetry_path: Path | None = None
    monotonic: Callable[[], float] = time.monotonic
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _logical_call_count: int = field(default=0, init=False, repr=False)
    _attempt_count: int = field(default=0, init=False, repr=False)
    _retry_count: int = field(default=0, init=False, repr=False)
    _elapsed_seconds: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.telemetry_path is not None:
            self.telemetry_path = Path(self.telemetry_path)

    def remaining_seconds(self) -> float | None:
        """Return the non-negative time remaining in the trial budget."""
        if self.deadline_monotonic is None:
            return None
        return max(0.0, self.deadline_monotonic - self.monotonic())

    def next_call_index(self) -> int:
        """Allocate a trial-unique, monotonically increasing logical call index."""
        with self._lock:
            self._logical_call_count += 1
            return self._logical_call_count

    def record_attempt(
        self,
        *,
        call_index: int,
        attempt: int,
        mode: str,
        http_status: int | None,
        ttfb_ms: int | None,
        first_content_ms: int | None,
        started_monotonic: float,
        finished_monotonic: float,
        remaining_before_ms: int | None,
        outcome: str,
        error_kind: str | None,
        retry_scheduled: bool,
    ) -> None:
        """Account for and durably append one HTTP attempt."""
        duration_seconds = max(0.0, finished_monotonic - started_monotonic)
        remaining_after = self.remaining_seconds()
        remaining_after_ms = (
            None if remaining_after is None else int(round(remaining_after * 1000))
        )
        record = {
            "trial": self.trial,
            "call_index": call_index,
            "stage": _active_stage.get(),
            "attempt": attempt,
            "mode": mode,
            "http_status": http_status,
            "ttfb_ms": ttfb_ms,
            "first_content_ms": first_content_ms,
            "duration_ms": int(round(duration_seconds * 1000)),
            "trial_remaining_ms_before": remaining_before_ms,
            "trial_remaining_ms_after": remaining_after_ms,
            "outcome": outcome,
            "error_kind": error_kind,
            "retry_scheduled": retry_scheduled,
        }

        with self._lock:
            self._attempt_count += 1
            if attempt > 1:
                self._retry_count += 1
            self._elapsed_seconds += duration_seconds
            if self.telemetry_path is not None:
                self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
                with self.telemetry_path.open("a", encoding="utf-8") as telemetry_file:
                    telemetry_file.write(json.dumps(record, separators=(",", ":")) + "\n")
                    telemetry_file.flush()
                    os.fsync(telemetry_file.fileno())

    def summary(self) -> dict[str, int | float]:
        """Return a thread-safe scalar snapshot of LLM accounting."""
        with self._lock:
            return {
                "logical_call_count": self._logical_call_count,
                "attempt_count": self._attempt_count,
                "retry_count": self._retry_count,
                "elapsed_seconds": self._elapsed_seconds,
                "last_call_index": self._logical_call_count,
            }


@contextmanager
def trial_llm_context(
    *,
    trial: int | str,
    deadline_monotonic: float | None = None,
    telemetry_path: Path | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> Iterator[TrialLLMContext]:
    """Activate a new LLM context and restore the previous context on exit."""
    context = TrialLLMContext(
        trial=trial,
        deadline_monotonic=deadline_monotonic,
        telemetry_path=telemetry_path,
        monotonic=monotonic,
    )
    token = _active_context.set(context)
    try:
        yield context
    finally:
        _active_context.reset(token)


def get_trial_llm_context() -> TrialLLMContext | None:
    """Return the active trial context, if any."""
    return _active_context.get()


@contextmanager
def llm_call_stage(name: str) -> Iterator[None]:
    """Set the current call stage, restoring the enclosing stage on exit."""
    token = _active_stage.set(name)
    try:
        yield
    finally:
        _active_stage.reset(token)
