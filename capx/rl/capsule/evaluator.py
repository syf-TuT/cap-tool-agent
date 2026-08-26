"""Strict reset-and-step clean replay with a replaceable process worker."""

from __future__ import annotations

import multiprocessing as mp
import traceback
from collections.abc import Callable, Mapping
from dataclasses import replace
from multiprocessing.connection import Connection
from typing import Any, Protocol

from .schema import ProgramReplayResultV1, ReplayOutcome, TaskInstanceV1, source_sha256


class WorkerTimedOutError(TimeoutError):
    """The parent watchdog expired while the worker was executing a program."""


class WorkerCrashedError(RuntimeError):
    """The worker exited or could no longer communicate with its parent."""


class WorkerInfrastructureTimedOutError(WorkerCrashedError):
    """Environment construction or deterministic reset exceeded the watchdog."""


class ReplayBackend(Protocol):
    def execute(
        self,
        task: TaskInstanceV1,
        source: str,
        seed: int,
        timeout_s: float,
    ) -> Mapping[str, Any]: ...

    def replace_worker(self) -> None: ...

    def close(self) -> None: ...


class ProgramCandidateLike(Protocol):
    program_sample_id: str
    source: str


def _attach_cleanup_errors(
    primary_error: BaseException, cleanup_errors: list[BaseException]
) -> None:
    if not cleanup_errors:
        return
    try:
        setattr(primary_error, "cleanup_errors", tuple(cleanup_errors))
        setattr(primary_error, "cleanup_error", cleanup_errors[0])
    except BaseException:
        pass


def _process_worker_main(
    connection: Connection,
    environment_factory: Callable[[TaskInstanceV1], Any],
) -> None:
    environment: Any | None = None
    environment_key: tuple[str, str, str, str] | None = None
    try:
        while True:
            request = connection.recv()
            if request.get("operation") == "close":
                return
            task = request["task"]
            key = (task.task_id, task.environment, task.api, task.privilege)
            if environment is None or key != environment_key:
                if environment is not None and hasattr(environment, "close"):
                    environment.close()
                environment = environment_factory(task)
                environment_key = key

            _reset_observation, reset_info = environment.reset(seed=request["seed"])
            connection.send({"event": "step_started"})
            _observation, reward, terminated, truncated, step_info = environment.step(
                request["source"]
            )
            connection.send(
                {
                    "reset_info": reset_info,
                    "step": {
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": step_info,
                    },
                }
            )
    except EOFError:
        return
    except BaseException as error:
        try:
            connection.send(
                {
                    "worker_error": {
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                }
            )
        except BaseException:
            pass
    finally:
        if environment is not None and hasattr(environment, "close"):
            try:
                environment.close()
            except BaseException:
                pass
        connection.close()


class PersistentProcessReplayBackend:
    """One persistent child environment guarded by a parent-side watchdog."""

    def __init__(
        self,
        environment_factory: Callable[[TaskInstanceV1], Any],
        *,
        start_method: str = "spawn",
    ) -> None:
        self._environment_factory = environment_factory
        self._context = mp.get_context(start_method)
        self._connection: Connection | None = None
        self._process: mp.Process | None = None
        self._closed = False

    @property
    def worker_pid(self) -> int | None:
        process = self._process
        if process is None or not process.is_alive():
            return None
        pid = process.pid
        return pid if isinstance(pid, int) and not isinstance(pid, bool) else None

    def _start_worker(self) -> None:
        if self._closed:
            raise WorkerCrashedError("replay backend is closed")
        parent: Connection | None = None
        child: Connection | None = None
        process: mp.Process | None = None
        try:
            parent, child = self._context.Pipe()
            process = self._context.Process(
                target=_process_worker_main,
                args=(child, self._environment_factory),
                daemon=True,
            )
            process.start()
            child.close()
        except BaseException as error:
            cleanup_errors: list[BaseException] = []

            def attempt(cleanup: Callable[[], Any]) -> Any | None:
                try:
                    return cleanup()
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
                    return None

            for connection in (child, parent):
                if connection is not None:
                    attempt(connection.close)
            if process is not None:
                alive = attempt(process.is_alive)
                if alive is True:
                    attempt(process.terminate)
                    attempt(lambda: process.join(timeout=1.0))
                    alive = attempt(process.is_alive)
                    if alive is True:
                        attempt(process.kill)
                        attempt(lambda: process.join(timeout=1.0))
                        alive = attempt(process.is_alive)
                if alive is not True and hasattr(process, "close"):
                    attempt(process.close)
            _attach_cleanup_errors(error, cleanup_errors)
            if not isinstance(error, Exception):
                raise
            wrapped = WorkerCrashedError("failed to start replay worker")
            _attach_cleanup_errors(wrapped, cleanup_errors)
            raise wrapped from error
        assert parent is not None and process is not None
        self._connection = parent
        self._process = process

    def execute(
        self,
        task: TaskInstanceV1,
        source: str,
        seed: int,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        if self._process is None or self._connection is None:
            self._start_worker()
        assert self._process is not None and self._connection is not None
        if not self._process.is_alive():
            raise WorkerCrashedError(f"replay worker exited with code {self._process.exitcode}")
        try:
            self._connection.send(
                {"operation": "evaluate", "task": task, "source": source, "seed": seed}
            )
        except (BrokenPipeError, EOFError, OSError) as error:
            raise WorkerCrashedError("failed to send request to replay worker") from error
        if not self._connection.poll(timeout_s):
            raise WorkerInfrastructureTimedOutError(
                f"environment construction/reset exceeded {timeout_s:.3f}s"
            )
        try:
            phase_payload = self._connection.recv()
        except (EOFError, OSError) as error:
            raise WorkerCrashedError("replay worker closed before returning a result") from error
        if isinstance(phase_payload, Mapping) and "worker_error" in phase_payload:
            details = phase_payload["worker_error"]
            raise WorkerCrashedError(f"replay worker failed: {details}")
        if not isinstance(phase_payload, Mapping) or phase_payload.get("event") != "step_started":
            raise WorkerCrashedError("replay worker omitted the step-started handshake")
        if not self._connection.poll(timeout_s):
            raise WorkerTimedOutError(f"program step exceeded {timeout_s:.3f}s")
        try:
            payload = self._connection.recv()
        except (EOFError, OSError) as error:
            raise WorkerCrashedError("replay worker closed before returning a result") from error
        if isinstance(payload, Mapping) and "worker_error" in payload:
            details = payload["worker_error"]
            raise WorkerCrashedError(f"replay worker failed: {details}")
        if not isinstance(payload, Mapping):
            raise WorkerCrashedError("replay worker returned a non-mapping payload")
        return payload

    def _stop_worker(self) -> None:
        connection, process = self._connection, self._process
        self._connection = None
        self._process = None
        if connection is not None:
            if process is not None and process.is_alive():
                try:
                    connection.send({"operation": "close"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
            connection.close()
        if process is not None:
            process.join(timeout=0.5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
            if process.is_alive():
                self._process = process
                raise WorkerCrashedError("unable to stop poisoned replay worker")
            process.close()

    def replace_worker(self) -> None:
        self._stop_worker()

    def close(self) -> None:
        self._stop_worker()
        self._closed = True


class _MalformedPayloadError(ValueError):
    pass


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


class CleanReplayEvaluator:
    """Convert a single reset-direct-replay into the typed v1 reward contract."""

    def __init__(
        self,
        backend: ReplayBackend,
        *,
        timeout_s: float = 120.0,
        max_failure_retries: int = 2,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if max_failure_retries < 0 or max_failure_retries > 2:
            raise ValueError("max_failure_retries must be between zero and two")
        self.backend = backend
        self.timeout_s = float(timeout_s)
        self.max_failure_retries = max_failure_retries

    @staticmethod
    def _program_sample_id(task: TaskInstanceV1, source: str) -> str:
        configured = task.metadata.get("program_sample_id")
        if isinstance(configured, str) and configured:
            return configured
        return f"replay-{source_sha256(source)[:16]}"

    def _result(
        self,
        task: TaskInstanceV1,
        source: str,
        *,
        program_sample_id: str | None,
        outcome: ReplayOutcome,
        attempts: int,
        raw_reward: float | None,
        binary_reward: float | None,
        task_completed: bool,
        initial_state_sha256: str | None = None,
        terminated: bool = False,
        truncated: bool = False,
        sandbox_rc: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> ProgramReplayResultV1:
        return ProgramReplayResultV1(
            task_id=task.task_id,
            environment_seed=task.environment_seed,
            program_sample_id=(
                self._program_sample_id(task, source)
                if program_sample_id is None
                else program_sample_id
            ),
            source=source,
            initial_state_sha256=initial_state_sha256 or task.initial_state_sha256,
            outcome=outcome,
            raw_reward=raw_reward,
            binary_reward=binary_reward,
            task_completed=task_completed,
            terminated=terminated,
            truncated=truncated,
            sandbox_rc=sandbox_rc,
            error_type=error_type,
            error_message=error_message,
            attempts=attempts,
            diagnostics={} if diagnostics is None else diagnostics,
        )

    def _classify_payload(
        self,
        task: TaskInstanceV1,
        source: str,
        payload: Mapping[str, Any],
        attempts: int,
        program_sample_id: str | None,
    ) -> ProgramReplayResultV1:
        reset_info = payload.get("reset_info")
        step = payload.get("step")
        if not isinstance(reset_info, Mapping) or not isinstance(step, Mapping):
            raise _MalformedPayloadError("payload must contain reset_info and step mappings")
        initial_hash = reset_info.get("initial_state_sha256")
        if not isinstance(initial_hash, str) or initial_hash != task.initial_state_sha256:
            raise _MalformedPayloadError("reset initial_state_sha256 does not match the task")
        step_info = step.get("info")
        if not isinstance(step_info, Mapping):
            raise _MalformedPayloadError("step.info must be a mapping")
        reward = step.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise _MalformedPayloadError("step.reward must be numeric")
        raw_reward = float(reward)
        required_step_fields = ("terminated", "truncated")
        missing_step_fields = [field for field in required_step_fields if field not in step]
        if missing_step_fields:
            raise _MalformedPayloadError(
                "step omitted required fields: " + ", ".join(missing_step_fields)
            )
        if "task_completed" not in step_info:
            raise _MalformedPayloadError("step.info omitted required task_completed")
        terminated = step["terminated"]
        truncated = step["truncated"]
        task_completed = step_info["task_completed"]
        for field_name, value in (
            ("terminated", terminated),
            ("truncated", truncated),
            ("task_completed", task_completed),
        ):
            if not isinstance(value, bool):
                raise _MalformedPayloadError(f"{field_name} must be a boolean")
        sandbox_rc = step_info.get("sandbox_rc")
        error_type = step_info.get("error_type")
        error_message = step_info.get("error_message")
        if sandbox_rc is not None and (
            isinstance(sandbox_rc, bool) or not isinstance(sandbox_rc, int)
        ):
            raise _MalformedPayloadError("sandbox_rc must be an integer or null")
        if error_type is not None and (
            not isinstance(error_type, str) or not error_type.strip()
        ):
            raise _MalformedPayloadError("error_type must be non-empty text or null")
        if error_message is not None and not isinstance(error_message, str):
            raise _MalformedPayloadError("error_message must be text or null")
        diagnostics = {
            "reset_info": _json_safe(reset_info),
            "step_info": _json_safe(step_info),
            "observed_task_completed": task_completed,
        }

        if error_type is not None:
            outcome = ReplayOutcome.PROGRAM_ERROR
        elif truncated:
            outcome = ReplayOutcome.STEP_BUDGET_EXHAUSTED
        elif task_completed and raw_reward >= 1.0:
            outcome = ReplayOutcome.SUCCESS
        else:
            outcome = ReplayOutcome.TASK_FAILURE
        return self._result(
            task,
            source,
            program_sample_id=program_sample_id,
            outcome=outcome,
            attempts=attempts,
            raw_reward=raw_reward,
            binary_reward=1.0 if outcome is ReplayOutcome.SUCCESS else 0.0,
            # ProgramReplayResultV1.task_completed is the evaluator's accepted success
            # verdict, not the raw environment observation.  A program error, truncation,
            # or sub-threshold reward must remain a typed non-success even when the
            # environment briefly observed physical completion; the observation stays in
            # diagnostics for audit.
            task_completed=outcome is ReplayOutcome.SUCCESS,
            initial_state_sha256=initial_hash,
            terminated=terminated,
            truncated=truncated,
            sandbox_rc=sandbox_rc,
            error_type=error_type,
            error_message=error_message,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _attempt_event(
        *,
        attempt: int,
        outcome: ReplayOutcome,
        worker_replaced: bool,
        retry_scheduled: bool,
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        return {
            "attempt": attempt,
            "outcome": outcome.value,
            "worker_replaced": worker_replaced,
            "retry_scheduled": retry_scheduled,
            "error_type": None if error is None else type(error).__name__,
            "error_message": None if error is None else str(error),
        }

    @staticmethod
    def _with_attempt_history(
        result: ProgramReplayResultV1,
        attempt_history: list[dict[str, Any]],
    ) -> ProgramReplayResultV1:
        diagnostics = dict(result.diagnostics)
        diagnostics["evaluator_attempt_history"] = list(attempt_history)
        return replace(result, diagnostics=diagnostics)

    def evaluate_program(
        self,
        task: TaskInstanceV1,
        source: str,
        seed: int,
        *,
        program_sample_id: str | None = None,
    ) -> ProgramReplayResultV1:
        if seed != task.environment_seed:
            raise ValueError("clean replay seed must match task.environment_seed")
        if program_sample_id is not None and (
            not isinstance(program_sample_id, str) or not program_sample_id
        ):
            raise ValueError("program_sample_id must be non-empty text when provided")
        attempts = 0
        last_failure: tuple[ReplayOutcome, BaseException] | None = None
        attempt_history: list[dict[str, Any]] = []
        while attempts <= self.max_failure_retries:
            attempts += 1
            try:
                payload = self.backend.execute(task, source, seed, self.timeout_s)
                result = self._classify_payload(
                    task,
                    source,
                    payload,
                    attempts,
                    program_sample_id,
                )
                attempt_history.append(
                    self._attempt_event(
                        attempt=attempts,
                        outcome=result.outcome,
                        worker_replaced=False,
                        retry_scheduled=False,
                    )
                )
                return self._with_attempt_history(result, attempt_history)
            except WorkerTimedOutError as error:
                self.backend.replace_worker()
                result = self._result(
                    task,
                    source,
                    program_sample_id=program_sample_id,
                    outcome=ReplayOutcome.PROGRAM_TIMEOUT,
                    attempts=attempts,
                    raw_reward=None,
                    binary_reward=0.0,
                    task_completed=False,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
                attempt_history.append(
                    self._attempt_event(
                        attempt=attempts,
                        outcome=ReplayOutcome.PROGRAM_TIMEOUT,
                        worker_replaced=True,
                        retry_scheduled=False,
                        error=error,
                    )
                )
                return self._with_attempt_history(result, attempt_history)
            except WorkerCrashedError as error:
                self.backend.replace_worker()
                last_failure = (ReplayOutcome.INFRA_ERROR, error)
                attempt_history.append(
                    self._attempt_event(
                        attempt=attempts,
                        outcome=ReplayOutcome.INFRA_ERROR,
                        worker_replaced=True,
                        retry_scheduled=attempts <= self.max_failure_retries,
                        error=error,
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                self.backend.replace_worker()
                last_failure = (ReplayOutcome.EVALUATOR_ERROR, error)
                attempt_history.append(
                    self._attempt_event(
                        attempt=attempts,
                        outcome=ReplayOutcome.EVALUATOR_ERROR,
                        worker_replaced=True,
                        retry_scheduled=attempts <= self.max_failure_retries,
                        error=error,
                    )
                )

        assert last_failure is not None
        outcome, error = last_failure
        result = self._result(
            task,
            source,
            program_sample_id=program_sample_id,
            outcome=outcome,
            attempts=attempts,
            raw_reward=None,
            binary_reward=None,
            task_completed=False,
            error_type=type(error).__name__,
            error_message=str(error),
        )
        return self._with_attempt_history(result, attempt_history)

    def close(self) -> None:
        self.backend.close()


class CandidateCleanReplayAdapter:
    """Adapt the source API to ``CapsuleGroupAssembler`` while preserving sample identity."""

    def __init__(self, evaluator: CleanReplayEvaluator) -> None:
        if not isinstance(evaluator, CleanReplayEvaluator):
            raise TypeError("evaluator must be CleanReplayEvaluator")
        self.evaluator = evaluator
        self._history: list[ProgramReplayResultV1] = []

    def __call__(
        self,
        task: TaskInstanceV1,
        candidate: ProgramCandidateLike,
    ) -> ProgramReplayResultV1:
        result = self.evaluator.evaluate_program(
            task,
            candidate.source,
            task.environment_seed,
            program_sample_id=candidate.program_sample_id,
        )
        self._history.append(result)
        return result

    def drain_history(self) -> tuple[ProgramReplayResultV1, ...]:
        """Return and clear results recorded since the previous drain.

        Collection is seed-local and sequential.  Draining at each group-attempt boundary makes
        discarded attempts auditable without changing the assembler's framework-neutral API.
        """

        history = tuple(self._history)
        self._history.clear()
        return history


_DEFAULT_EVALUATOR: CleanReplayEvaluator | None = None


def configure_default_evaluator(evaluator: CleanReplayEvaluator | None) -> None:
    global _DEFAULT_EVALUATOR
    _DEFAULT_EVALUATOR = evaluator


def evaluate_program(
    task: TaskInstanceV1,
    source: str,
    seed: int,
    *,
    evaluator: CleanReplayEvaluator | None = None,
    program_sample_id: str | None = None,
) -> ProgramReplayResultV1:
    selected = evaluator if evaluator is not None else _DEFAULT_EVALUATOR
    if selected is None:
        raise RuntimeError("no clean replay evaluator has been configured")
    return selected.evaluate_program(
        task,
        source,
        seed,
        program_sample_id=program_sample_id,
    )


__all__ = [
    "CandidateCleanReplayAdapter",
    "CleanReplayEvaluator",
    "PersistentProcessReplayBackend",
    "WorkerCrashedError",
    "WorkerInfrastructureTimedOutError",
    "WorkerTimedOutError",
    "configure_default_evaluator",
    "evaluate_program",
]
