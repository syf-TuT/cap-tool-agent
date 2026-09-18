from __future__ import annotations

import os
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from capx.rl.capsule.evaluator import (
    CandidateCleanReplayAdapter,
    CleanReplayEvaluator,
    PersistentProcessReplayBackend,
    WorkerCrashedError,
    WorkerInfrastructureTimedOutError,
    WorkerTimedOutError,
    configure_default_evaluator,
    evaluate_program,
)
from capx.rl.capsule.schema import ReplayOutcome, TaskInstanceV1, source_sha256
from capx.rl.capsule.telemetry import summarize_replay_results

STATE_HASH = "a" * 64


def _task(seed: int = 5) -> TaskInstanceV1:
    return TaskInstanceV1(
        task_id="cube-stack:5",
        environment_seed=seed,
        prompt="stack the cubes",
        environment="fake-cube-stack",
        api="fake-privileged",
        privilege="privileged",
        initial_state_sha256=STATE_HASH,
        metadata={"program_sample_id": "sample-0"},
    )


def _payload(
    *,
    reward: float = 0.0,
    task_completed: bool = False,
    terminated: bool = False,
    truncated: bool = False,
    sandbox_rc: int = 0,
    error_type: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "reset_info": {"initial_state_sha256": STATE_HASH},
        "step": {
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": {
                "task_completed": task_completed,
                "sandbox_rc": sandbox_rc,
                "error_type": error_type,
                "error_message": error_message,
            },
        },
    }


class _FakeBackend:
    def __init__(self, responses: list[object]) -> None:
        self.responses = deque(responses)
        self.calls: list[tuple[TaskInstanceV1, str, int, float]] = []
        self.replacements = 0
        self.closed = False

    def execute(
        self, task: TaskInstanceV1, source: str, seed: int, timeout_s: float
    ) -> Mapping[str, Any]:
        self.calls.append((task, source, seed, timeout_s))
        response = self.responses.popleft()
        if isinstance(response, BaseException):
            raise response
        assert isinstance(response, Mapping)
        return response

    def replace_worker(self) -> None:
        self.replacements += 1

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (_payload(reward=1.0, task_completed=True, terminated=True), ReplayOutcome.SUCCESS),
        (_payload(reward=0.4), ReplayOutcome.TASK_FAILURE),
        (
            _payload(error_type="ValueError", error_message="bad source", sandbox_rc=1),
            ReplayOutcome.PROGRAM_ERROR,
        ),
        (_payload(truncated=True), ReplayOutcome.STEP_BUDGET_EXHAUSTED),
    ],
)
def test_evaluator_classifies_semantic_outcomes(
    payload: dict[str, Any], expected: ReplayOutcome
) -> None:
    backend = _FakeBackend([payload])
    evaluator = CleanReplayEvaluator(backend, timeout_s=3.5)

    result = evaluator.evaluate_program(_task(), "RESULT = None", seed=5)

    assert result.outcome is expected
    assert result.binary_reward == (1.0 if expected is ReplayOutcome.SUCCESS else 0.0)
    assert result.attempts == 1
    assert backend.calls == [(_task(), "RESULT = None", 5, 3.5)]


def test_sandbox_rc_is_diagnostic_only_and_does_not_override_success() -> None:
    backend = _FakeBackend(
        [_payload(reward=1.0, task_completed=True, terminated=True, sandbox_rc=19)]
    )

    result = CleanReplayEvaluator(backend).evaluate_program(_task(), "pass", seed=5)

    assert result.outcome is ReplayOutcome.SUCCESS
    assert result.sandbox_rc == 19


def test_evaluator_executes_normalized_source_but_preserves_raw_identity() -> None:
    backend = _FakeBackend([_payload(reward=1.0, task_completed=True, terminated=True)])
    evaluator = CleanReplayEvaluator(backend)
    raw_source = "```python\npass\n```"

    result = evaluator.evaluate_program(_task(), raw_source, seed=5)

    assert backend.calls == [(_task(), "pass", 5, 120.0)]
    assert result.source == raw_source
    assert result.source_sha256 == source_sha256(raw_source)
    assert result.diagnostics["raw_source_sha256"] == source_sha256(raw_source)
    assert result.diagnostics["executed_source_sha256"] == source_sha256("pass")
    assert result.diagnostics["source_normalized"] is True


def test_evaluator_reports_unchanged_unfenced_execution_source() -> None:
    backend = _FakeBackend([_payload(reward=0.2)])
    evaluator = CleanReplayEvaluator(backend)

    result = evaluator.evaluate_program(_task(), "pass", seed=5)

    assert backend.calls == [(_task(), "pass", 5, 120.0)]
    assert result.source == "pass"
    assert result.diagnostics["raw_source_sha256"] == source_sha256("pass")
    assert result.diagnostics["executed_source_sha256"] == source_sha256("pass")
    assert result.diagnostics["source_normalized"] is False


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            _payload(
                reward=1.0,
                task_completed=True,
                terminated=True,
                error_type="RuntimeError",
                error_message="failed after completing the physical task",
            ),
            ReplayOutcome.PROGRAM_ERROR,
        ),
        (
            _payload(reward=1.0, task_completed=True, truncated=True),
            ReplayOutcome.STEP_BUDGET_EXHAUSTED,
        ),
        (
            _payload(reward=0.75, task_completed=True, terminated=True),
            ReplayOutcome.TASK_FAILURE,
        ),
    ],
)
def test_non_success_outcomes_preserve_observed_completion_only_in_diagnostics(
    payload: dict[str, Any], expected: ReplayOutcome
) -> None:
    backend = _FakeBackend([payload])

    result = CleanReplayEvaluator(backend).evaluate_program(_task(), "pass", seed=5)

    assert result.outcome is expected
    assert result.binary_reward == 0.0
    assert result.task_completed is False
    assert result.attempts == 1
    assert result.diagnostics["observed_task_completed"] is True
    assert result.diagnostics["step_info"]["task_completed"] is True
    assert backend.replacements == 0


def test_timeout_is_semantic_program_timeout_and_replaces_worker_without_retry() -> None:
    backend = _FakeBackend([WorkerTimedOutError("watchdog expired")])

    result = CleanReplayEvaluator(backend).evaluate_program(_task(), "while True: pass", seed=5)

    assert result.outcome is ReplayOutcome.PROGRAM_TIMEOUT
    assert result.binary_reward == 0.0
    assert result.attempts == 1
    assert backend.replacements == 1
    assert len(backend.calls) == 1


def test_factory_or_reset_watchdog_timeout_is_retried_as_infrastructure_failure() -> None:
    backend = _FakeBackend(
        [WorkerInfrastructureTimedOutError("reset watchdog expired"), _payload(reward=0.2)]
    )

    result = CleanReplayEvaluator(backend).evaluate_program(_task(), "pass", seed=5)

    assert result.outcome is ReplayOutcome.TASK_FAILURE
    assert result.attempts == 2
    assert backend.replacements == 1


def test_infra_failure_retries_twice_and_replaces_each_poisoned_worker() -> None:
    backend = _FakeBackend(
        [WorkerCrashedError("exit 9"), WorkerCrashedError("exit 9"), _payload(reward=0.2)]
    )

    result = CleanReplayEvaluator(backend, max_failure_retries=2).evaluate_program(
        _task(), "pass", seed=5
    )

    assert result.outcome is ReplayOutcome.TASK_FAILURE
    assert result.attempts == 3
    assert backend.replacements == 2
    assert len(backend.calls) == 3


def test_candidate_adapter_history_drives_retry_and_infra_telemetry() -> None:
    backend = _FakeBackend(
        [
            WorkerCrashedError("transient crash"),
            _payload(reward=0.2),
            WorkerCrashedError("persistent crash"),
            WorkerCrashedError("persistent crash"),
            WorkerCrashedError("persistent crash"),
        ]
    )
    adapter = CandidateCleanReplayAdapter(
        CleanReplayEvaluator(backend, max_failure_retries=2)
    )
    candidate = SimpleNamespace(source="pass", program_sample_id="sample-a")

    recovered = adapter(_task(), candidate)
    candidate.program_sample_id = "sample-b"
    exhausted = adapter(_task(), candidate)

    history = adapter.drain_history()
    assert history == (recovered, exhausted)
    assert adapter.drain_history() == ()
    assert summarize_replay_results(history) == {
        "replay_event_count": 2,
        "attempt_event_count": 5,
        "retry_count": 3,
        "infra_failures": 4,
        "evaluator_failures": 0,
        "worker_replacements": 4,
    }
    assert [
        event["outcome"]
        for event in recovered.diagnostics["evaluator_attempt_history"]
    ] == ["infra_error", "task_failure"]


def test_exhausted_infra_retries_return_null_binary_reward() -> None:
    backend = _FakeBackend([WorkerCrashedError("crash")] * 3)

    result = CleanReplayEvaluator(backend, max_failure_retries=2).evaluate_program(
        _task(), "pass", seed=5
    )

    assert result.outcome is ReplayOutcome.INFRA_ERROR
    assert result.binary_reward is None
    assert result.attempts == 3
    assert backend.replacements == 3


def test_malformed_payload_is_evaluator_error_and_is_retried() -> None:
    backend = _FakeBackend([{"not_step": True}, _payload(reward=0.1)])

    result = CleanReplayEvaluator(backend, max_failure_retries=2).evaluate_program(
        _task(), "pass", seed=5
    )

    assert result.outcome is ReplayOutcome.TASK_FAILURE
    assert result.attempts == 2
    assert backend.replacements == 1
    assert summarize_replay_results((result,), require_attempt_history=True) == {
        "replay_event_count": 1,
        "attempt_event_count": 2,
        "retry_count": 1,
        "infra_failures": 0,
        "evaluator_failures": 1,
        "worker_replacements": 1,
    }


@pytest.mark.parametrize(
    ("container", "field_name"),
    [
        ("step", "terminated"),
        ("step", "truncated"),
        ("info", "task_completed"),
    ],
)
def test_missing_required_outcome_fields_are_evaluator_errors(
    container: str, field_name: str
) -> None:
    malformed = _payload(reward=0.2)
    target = malformed["step"] if container == "step" else malformed["step"]["info"]
    del target[field_name]
    backend = _FakeBackend([deepcopy(malformed) for _ in range(3)])

    result = CleanReplayEvaluator(backend).evaluate_program(_task(), "pass", seed=5)

    assert result.outcome is ReplayOutcome.EVALUATOR_ERROR
    assert result.binary_reward is None
    assert result.attempts == 3
    assert backend.replacements == 3


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("sandbox_rc", True),
        ("sandbox_rc", "zero"),
        ("error_type", 123),
        ("error_type", ""),
        ("error_message", 123),
    ],
)
def test_invalid_fatal_diagnostics_are_evaluator_errors(
    field_name: str, invalid_value: object
) -> None:
    malformed = _payload(reward=0.2)
    malformed["step"]["info"][field_name] = invalid_value
    backend = _FakeBackend([deepcopy(malformed) for _ in range(3)])

    result = CleanReplayEvaluator(backend).evaluate_program(_task(), "pass", seed=5)

    assert result.outcome is ReplayOutcome.EVALUATOR_ERROR
    assert result.binary_reward is None
    assert result.attempts == 3
    assert backend.replacements == 3


def test_exhausted_bad_payload_returns_evaluator_error() -> None:
    backend = _FakeBackend([{"not_step": True}] * 3)

    result = CleanReplayEvaluator(backend, max_failure_retries=2).evaluate_program(
        _task(), "pass", seed=5
    )

    assert result.outcome is ReplayOutcome.EVALUATOR_ERROR
    assert result.binary_reward is None
    assert result.attempts == 3


def test_module_level_evaluator_requires_explicit_or_configured_instance() -> None:
    backend = _FakeBackend([_payload(reward=0.0), _payload(reward=0.0)])
    evaluator = CleanReplayEvaluator(backend)

    explicit = evaluate_program(_task(), "pass", 5, evaluator=evaluator)
    configure_default_evaluator(evaluator)
    try:
        configured = evaluate_program(_task(), "pass", 5)
    finally:
        configure_default_evaluator(None)

    assert explicit.outcome is ReplayOutcome.TASK_FAILURE
    assert configured.outcome is ReplayOutcome.TASK_FAILURE


def test_explicit_program_sample_id_supports_group_candidate_identity() -> None:
    backend = _FakeBackend([_payload(reward=1.0, task_completed=True)])
    evaluator = CleanReplayEvaluator(backend)

    result = evaluator.evaluate_program(
        _task(),
        "pass",
        seed=5,
        program_sample_id="base-sample-3",
    )

    assert result.program_sample_id == "base-sample-3"


def test_explicit_program_sample_id_must_be_nonempty_text() -> None:
    evaluator = CleanReplayEvaluator(_FakeBackend([]))

    with pytest.raises(ValueError, match="program_sample_id"):
        evaluator.evaluate_program(_task(), "pass", seed=5, program_sample_id="")


def test_candidate_adapter_preserves_candidate_identity_for_group_assembler() -> None:
    backend = _FakeBackend([_payload(reward=0.0)])
    adapter = CandidateCleanReplayAdapter(CleanReplayEvaluator(backend))
    candidate = SimpleNamespace(program_sample_id="candidate-7", source="RESULT = False")

    result = adapter(_task(), candidate)

    assert result.program_sample_id == "candidate-7"
    assert backend.calls[0][1:3] == ("RESULT = False", 5)


class _ProcessFakeEnv:
    def __init__(self, task: TaskInstanceV1) -> None:
        self.task = task
        self.calls: list[tuple[Any, ...]] = []

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self.calls.append(("reset", seed))
        return {}, {"initial_state_sha256": self.task.initial_state_sha256}

    def step(self, source: str) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.calls.append(("step", source))
        return (
            {},
            1.0,
            True,
            False,
            {
                "task_completed": True,
                "sandbox_rc": 7,
                "error_type": None,
                "error_message": None,
                "worker_pid": os.getpid(),
                "call_order": list(self.calls),
            },
        )

    def close(self) -> None:
        return None


def _process_fake_env_factory(task: TaskInstanceV1) -> _ProcessFakeEnv:
    return _ProcessFakeEnv(task)


class _FakeConnection:
    def __init__(self) -> None:
        self.closed = False

    def send(self, _payload: object) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _StubbornProcess:
    def __init__(self) -> None:
        self.alive = True
        self.terminated = False
        self.killed = False
        self.closed = False

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        del timeout

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.alive = False

    def close(self) -> None:
        self.closed = True


def test_worker_replacement_escalates_from_terminate_to_kill() -> None:
    backend = PersistentProcessReplayBackend(_process_fake_env_factory)
    process = _StubbornProcess()
    connection = _FakeConnection()
    backend._process = process  # type: ignore[assignment]
    backend._connection = connection  # type: ignore[assignment]

    backend.replace_worker()

    assert process.terminated is True
    assert process.killed is True
    assert process.closed is True
    assert connection.closed is True
    assert backend._process is None
    assert backend._connection is None


class _StartFailureProcess:
    def start(self) -> None:
        raise OSError("process table full")

    def is_alive(self) -> bool:
        return False


class _StartFailureContext:
    def __init__(self) -> None:
        self.parent = _FakeConnection()
        self.child = _FakeConnection()

    def Pipe(self):
        return self.parent, self.child

    def Process(self, **_kwargs):
        return _StartFailureProcess()


def test_worker_start_failure_closes_pipes_and_is_typed_as_worker_crash() -> None:
    backend = PersistentProcessReplayBackend(_process_fake_env_factory)
    context = _StartFailureContext()
    backend._context = context  # type: ignore[assignment]

    with pytest.raises(WorkerCrashedError, match="start"):
        backend.execute(_task(), "pass", 5, 1.0)

    assert context.parent.closed is True
    assert context.child.closed is True


class _CancelledStartProcess:
    def __init__(self) -> None:
        self.closed = False

    def start(self) -> None:
        raise KeyboardInterrupt("cancel worker start")

    def is_alive(self) -> bool:
        return False

    def close(self) -> None:
        self.closed = True


class _CancelledStartContext(_StartFailureContext):
    def __init__(self) -> None:
        super().__init__()
        self.process = _CancelledStartProcess()

    def Process(self, **_kwargs):
        return self.process


def test_worker_start_cancellation_closes_process_and_pipes_without_wrapping() -> None:
    backend = PersistentProcessReplayBackend(_process_fake_env_factory)
    context = _CancelledStartContext()
    backend._context = context  # type: ignore[assignment]

    with pytest.raises(KeyboardInterrupt, match="cancel worker start"):
        backend.execute(_task(), "pass", 5, 1.0)

    assert context.parent.closed is True
    assert context.child.closed is True
    assert context.process.closed is True


class _FailingCloseConnection(_FakeConnection):
    def close(self) -> None:
        self.closed = True
        raise RuntimeError("child close failed")


class _CancelledStartWithCleanupFailureContext(_CancelledStartContext):
    def __init__(self) -> None:
        super().__init__()
        self.child = _FailingCloseConnection()


def test_worker_start_cleanup_failure_preserves_cancellation_and_attempts_all_cleanup() -> None:
    backend = PersistentProcessReplayBackend(_process_fake_env_factory)
    context = _CancelledStartWithCleanupFailureContext()
    backend._context = context  # type: ignore[assignment]

    with pytest.raises(KeyboardInterrupt, match="cancel worker start") as caught:
        backend.execute(_task(), "pass", 5, 1.0)

    assert context.child.closed is True
    assert context.parent.closed is True
    assert context.process.closed is True
    assert isinstance(caught.value.cleanup_error, RuntimeError)
    assert str(caught.value.cleanup_error) == "child close failed"


def test_non_boolean_step_flags_are_retried_as_evaluator_error() -> None:
    malformed = _payload(reward=1.0, task_completed=True)
    malformed["step"]["truncated"] = "false"
    backend = _FakeBackend([malformed, _payload(reward=0.2)])

    result = CleanReplayEvaluator(backend).evaluate_program(_task(), "pass", seed=5)

    assert result.outcome is ReplayOutcome.TASK_FAILURE
    assert result.attempts == 2


def test_persistent_process_backend_runs_only_reset_then_full_program_step() -> None:
    backend = PersistentProcessReplayBackend(_process_fake_env_factory, start_method="spawn")
    evaluator = CleanReplayEvaluator(backend, timeout_s=5.0)
    try:
        result = evaluator.evaluate_program(_task(seed=23), "RESULT = 'done'", seed=23)
    finally:
        evaluator.close()

    assert result.outcome is ReplayOutcome.SUCCESS
    assert result.diagnostics["step_info"]["call_order"] == (
        ("reset", 23),
        ("step", "RESULT = 'done'"),
    )
    assert result.sandbox_rc == 7


def test_persistent_process_backend_exposes_live_child_pid_for_sequential_replays() -> None:
    backend = PersistentProcessReplayBackend(_process_fake_env_factory, start_method="spawn")

    assert backend.worker_pid is None
    try:
        first = backend.execute(_task(seed=23), "RESULT = 'first'", 23, 5.0)
        first_pid = backend.worker_pid
        second = backend.execute(_task(seed=23), "RESULT = 'second'", 23, 5.0)
        second_pid = backend.worker_pid
    finally:
        backend.close()

    assert isinstance(first_pid, int)
    assert first_pid == first["step"]["info"]["worker_pid"]
    assert second_pid == first_pid
    assert second["step"]["info"]["worker_pid"] == first_pid
    assert backend.worker_pid is None


class _ExitedProcess:
    pid = 12345

    @staticmethod
    def is_alive() -> bool:
        return False


def test_persistent_process_backend_hides_exited_worker_pid() -> None:
    backend = PersistentProcessReplayBackend(_process_fake_env_factory)
    backend._process = _ExitedProcess()  # type: ignore[assignment]

    assert backend.worker_pid is None
