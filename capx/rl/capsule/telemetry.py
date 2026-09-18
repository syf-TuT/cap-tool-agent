"""Typed clean-replay telemetry shared by server gates and local audit artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from .schema import ProgramReplayResultV1, ReplayOutcome

_UNKNOWN_OUTCOMES = {ReplayOutcome.INFRA_ERROR, ReplayOutcome.EVALUATOR_ERROR}


def summarize_replay_results(
    results: Iterable[ProgramReplayResultV1],
    *,
    require_attempt_history: bool = False,
) -> dict[str, int]:
    """Derive retry/failure counters solely from immutable typed replay results.

    The evaluator retries only infrastructure/evaluator failures.  Detailed evaluator histories
    preserve which kind occurred on every attempt; legacy in-memory fixtures without histories
    use a conservative infrastructure-failure fallback and are never accepted by gate verifiers.
    """

    replay_event_count = 0
    attempt_event_count = 0
    retry_count = 0
    infra_failures = 0
    evaluator_failures = 0
    worker_replacements = 0
    for result in results:
        if not isinstance(result, ProgramReplayResultV1):
            raise TypeError("replay telemetry requires ProgramReplayResultV1 values")
        replay_event_count += 1
        history = result.diagnostics.get("evaluator_attempt_history")
        if history is None:
            if require_attempt_history:
                raise ValueError(
                    "typed replay omitted diagnostics.evaluator_attempt_history"
                )
            retries = result.attempts - 1
            retry_count += retries
            infra_failures += retries
            if result.outcome in _UNKNOWN_OUTCOMES:
                infra_failures += 1
            attempt_event_count += result.attempts
            worker_replacements += retries + int(result.outcome in _UNKNOWN_OUTCOMES)
            continue
        if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
            raise TypeError("evaluator_attempt_history must be a sequence")
        if len(history) != result.attempts:
            raise ValueError("evaluator_attempt_history length must equal attempts")
        parsed_outcomes: list[ReplayOutcome] = []
        for expected_attempt, event in enumerate(history, start=1):
            if not isinstance(event, Mapping):
                raise TypeError("evaluator_attempt_history events must be mappings")
            attempt = event.get("attempt")
            worker_replaced = event.get("worker_replaced")
            retry_scheduled = event.get("retry_scheduled")
            if attempt != expected_attempt or isinstance(attempt, bool):
                raise ValueError("evaluator attempt indices must be contiguous and one-based")
            if not isinstance(worker_replaced, bool) or not isinstance(retry_scheduled, bool):
                raise TypeError("evaluator attempt flags must be booleans")
            try:
                outcome = ReplayOutcome(event.get("outcome"))
            except (TypeError, ValueError) as error:
                raise ValueError("evaluator attempt outcome is invalid") from error
            expected_replacement = outcome in {
                ReplayOutcome.INFRA_ERROR,
                ReplayOutcome.EVALUATOR_ERROR,
                ReplayOutcome.PROGRAM_TIMEOUT,
            }
            if worker_replaced is not expected_replacement:
                raise ValueError("evaluator worker replacement evidence is inconsistent")
            should_retry = expected_attempt < len(history)
            if retry_scheduled is not should_retry:
                raise ValueError("evaluator retry_scheduled evidence is inconsistent")
            if should_retry and outcome not in _UNKNOWN_OUTCOMES:
                raise ValueError("only infra/evaluator failures may schedule clean-replay retry")
            parsed_outcomes.append(outcome)
            attempt_event_count += 1
            retry_count += int(retry_scheduled)
            infra_failures += int(outcome is ReplayOutcome.INFRA_ERROR)
            evaluator_failures += int(outcome is ReplayOutcome.EVALUATOR_ERROR)
            worker_replacements += int(worker_replaced)
        if parsed_outcomes[-1] is not result.outcome:
            raise ValueError("final evaluator attempt outcome must match typed replay outcome")
    return {
        "replay_event_count": replay_event_count,
        "attempt_event_count": attempt_event_count,
        "retry_count": retry_count,
        "infra_failures": infra_failures,
        "evaluator_failures": evaluator_failures,
        "worker_replacements": worker_replacements,
    }


__all__ = ["summarize_replay_results"]
