"""Trial orchestration for CaP-X environments.

Handles batch execution, parallel worker dispatch, retry logic, and
wall-clock timeouts for running code-generation trials.  The actual
single-trial execution lives in :mod:`capx.envs.trial`.
"""

from __future__ import annotations

import functools
import os
import signal
import socket
import sys
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from tqdm import tqdm

from capx.envs.configs.instantiate import instantiate
from capx.envs.infrastructure import (
    InfrastructureFailure,
    ServiceEndpoint,
    ServiceReadinessError,
)
from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.envs.trial import (
    _annotate_code_blocks,
    _build_log_lines,
    _run_single_trial,
)
from capx.envs.trial_results import RunOutcome, TrialResultWriter
from capx.llm.context import TrialLLMContext, trial_llm_context
from capx.llm.errors import LLMErrorKind, LLMQueryError
from capx.utils.launch_utils import (
    TrialSummary,
    _print_and_save_summary,
    _save_trial_artifacts,
    run_server_proc,
)
from capx.utils.parallel_eval import run_parallel_with_setup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRIAL_TIMEOUT_SECONDS = 450


class TrialResultPersistenceError(RuntimeError):
    """Signals that terminal-result persistence failed after trial execution."""

    def __init__(self, cause: Exception) -> None:
        self.cause = cause
        super().__init__("failed to persist structured trial result")


def _trial_timeout_seconds() -> int:
    """Read the complete per-trial budget once, with the approved default."""

    raw_value = os.getenv("CAPX_TRIAL_TIMEOUT_SECONDS")
    if raw_value is None:
        return TRIAL_TIMEOUT_SECONDS
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError("CAPX_TRIAL_TIMEOUT_SECONDS must be a positive number") from exc
    if value <= 0:
        raise ValueError("CAPX_TRIAL_TIMEOUT_SECONDS must be a positive number")
    return max(1, int(value))


# ---------------------------------------------------------------------------
# API server helpers
# ---------------------------------------------------------------------------


def _service_endpoint_name(api_server: Mapping[str, Any], port: int) -> str:
    target = str(api_server.get("_target_", ""))
    parts = target.split(".")
    if len(parts) > 1 and parts[-1] == "main":
        return parts[-2]
    return parts[-1] or f"api_server_{port}"


def _required_service_endpoints(
    api_servers: list | None,
    env_factory: Mapping[str, Any],
) -> list[ServiceEndpoint]:
    endpoints: list[ServiceEndpoint] = []
    for api_server in api_servers or []:
        port = api_server.get("port")
        if port is None:
            continue
        endpoint_port = int(port)
        endpoints.append(
            ServiceEndpoint(
                name=_service_endpoint_name(api_server, endpoint_port),
                host=str(api_server.get("host", "127.0.0.1")),
                port=endpoint_port,
            )
        )

    env_cfg = env_factory.get("cfg", {})
    apis = env_cfg.get("apis", []) if isinstance(env_cfg, Mapping) else []
    molmo_base_url = (
        env_cfg.get("molmo_base_url") if isinstance(env_cfg, Mapping) else None
    )
    if "FrankaLiberoApi" in apis and molmo_base_url:
        parsed = urlparse(str(molmo_base_url))
        if parsed.hostname is None:
            raise ValueError(f"Invalid molmo_base_url: {molmo_base_url}")
        endpoints.append(
            ServiceEndpoint(
                name="molmo",
                host=parsed.hostname,
                port=parsed.port or (443 if parsed.scheme == "https" else 80),
            )
        )

    deduplicated: list[ServiceEndpoint] = []
    seen: set[tuple[str, int]] = set()
    for endpoint in endpoints:
        key = (endpoint.host, endpoint.port)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(endpoint)
    return deduplicated


def _service_endpoint_ready(endpoint: ServiceEndpoint) -> bool:
    try:
        with socket.create_connection((endpoint.host, endpoint.port), timeout=1.0):
            return True
    except OSError:
        return False


def _wait_for_required_services(
    endpoints: list[ServiceEndpoint],
    *,
    wait_timeout: float,
) -> None:
    pending = list(endpoints)
    deadline = time.monotonic() + max(0.0, wait_timeout)
    while pending:
        unavailable: list[ServiceEndpoint] = []
        for endpoint in pending:
            if _service_endpoint_ready(endpoint):
                print(
                    f"API server {endpoint.name} on "
                    f"{endpoint.host}:{endpoint.port} is ready"
                )
            else:
                unavailable.append(endpoint)
        if not unavailable:
            return
        if time.monotonic() >= deadline:
            unresolved = ", ".join(
                f"{endpoint.name} ({endpoint.host}:{endpoint.port})"
                for endpoint in unavailable
            )
            raise ServiceReadinessError(
                "service_not_ready",
                f"Required services not ready after {wait_timeout}s: {unresolved}",
                evidence={
                    "unresolved_endpoints": [
                        {
                            "name": endpoint.name,
                            "host": endpoint.host,
                            "port": endpoint.port,
                        }
                        for endpoint in unavailable
                    ]
                },
            )
        pending = unavailable
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def _start_api_servers(
    api_servers: list | None,
    *,
    required_endpoints: list[ServiceEndpoint] | None = None,
    wait_timeout: float = 120.0,
) -> list:
    """Launch any API server sub-processes defined in the config.

    Skips servers whose port is already in use (e.g. started externally).
    After launching, waits until all servers are accepting connections.
    """
    procs = []
    endpoints = required_endpoints
    if endpoints is None:
        endpoints = _required_service_endpoints(api_servers, {})
    try:
        for api_server in api_servers or []:
            port = api_server.get("port")
            host = api_server.get("host", "127.0.0.1")
            if port is not None:
                endpoint = ServiceEndpoint(
                    name=_service_endpoint_name(api_server, int(port)),
                    host=str(host),
                    port=int(port),
                )
                if _service_endpoint_ready(endpoint):
                    print(f"API server on {host}:{port} already running, skipping")
                    continue
            proc = run_server_proc(api_server)
            procs.append(proc)
            print(f"API server {api_server} started")
        _wait_for_required_services(endpoints, wait_timeout=wait_timeout)
    except BaseException:
        _stop_api_servers(procs)
        raise

    return procs


def _stop_api_servers(server_procs: list) -> None:
    """Terminate API server sub-processes."""
    for proc in server_procs:
        proc.terminate()
        proc.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------

def _setup_output_dir(args, config: dict[str, Any]) -> None:
    """Create and normalize the output directory path.

    Inserts the model name into the output path so that results from
    different models are stored separately.
    """
    if args.use_oracle_code:
        args.model = "oracle"
    if config["output_dir"]:
        parts = config["output_dir"].split("/")
        parts.insert(-1, str(args.model).replace("/", "_"))
        new_out_dir = "/".join(parts)
        Path(new_out_dir).mkdir(parents=True, exist_ok=True)
        config["output_dir"] = new_out_dir


# ---------------------------------------------------------------------------
# Headless trial runner
# ---------------------------------------------------------------------------

def _run_headless_trials(
    args,
    env_factory: dict[str, Any],
    config: dict[str, Any],
    start_time: float,
) -> None:
    """Run all trials in headless CLI mode, then print a summary.

    Dispatches to parallel or sequential execution depending on ``num_workers``.
    """
    if config["total_trials"] <= 0:
        print("No trials requested; exiting.")
        return

    if config["record_video"] and not config["output_dir"]:
        raise ValueError("record_video requires --output-dir to save generated videos")

    _setup_output_dir(args, config)

    # Determine which trials to run (supports resume)
    trial_ids = list(range(1, config["total_trials"] + 1))
    if config.get("resume_idx") is not None:
        trial_ids = list(range(config["resume_idx"], config["total_trials"] + 1))

    # Execute trials
    if config["num_workers"] > 1:
        setup_fn = functools.partial(_worker_setup, env_factory=env_factory)
        trial_fn = functools.partial(_run_single_trial_worker, args=args, config=config)
        summaries = run_parallel_with_setup(
            trial_ids,
            num_workers=config["num_workers"],
            setup_fn=setup_fn,
            trial_fn=trial_fn,
        )
    else:
        summaries = _run_trial_batch(trial_ids, args=args, env_factory=env_factory, config=config)

    summaries.sort(key=lambda s: s.trial)
    _print_and_save_summary(summaries, args, config, start_time)

    # Write completion flag
    os.makedirs(os.path.join(config["output_dir"], "aaa_done_flag"), exist_ok=True)
    with open(os.path.join(config["output_dir"], "aaa_done_flag", "aaa_done_flag.txt"), "w") as f:
        f.write("1")


# ---------------------------------------------------------------------------
# Worker setup and trial dispatch
# ---------------------------------------------------------------------------

def _worker_setup(*, env_factory: dict[str, Any]) -> tuple[Any, str | None]:
    """Create the environment once per parallel worker.

    Returns:
        Tuple of (environment, multi_turn_prompt).
    """
    env = instantiate(env_factory)
    multi_turn_prompt = env_factory["cfg"].get("multi_turn_prompt", None)
    return (env, multi_turn_prompt)


def _run_single_trial_worker(
    state: tuple[Any, str | None],
    trial: int,
    *,
    args,
    config: dict[str, Any],
) -> TrialSummary:
    """Run a single trial using pre-initialized worker state (for parallel mode)."""
    env, multi_turn_prompt = state
    return _run_trial_with_retries(env, trial, args, config, multi_turn_prompt)


def _run_trial_with_retries(
    env: CodeExecutionEnvBase,
    trial: int,
    args,
    config: dict[str, Any],
    multi_turn_prompt: str | None,
) -> TrialSummary:
    """Run one complete trial; logical LLM retries belong to the LLM client."""

    return _run_single_trial_with_timeout(
        env=env,
        trial=trial,
        args=args,
        config=config,
        multi_turn_prompt=multi_turn_prompt,
        timeout_s=_trial_timeout_seconds(),
    )


def _run_trial_batch(
    trial_ids: Iterable[int],
    *,
    args,
    env_factory: dict[str, Any],
    config: dict[str, Any],
) -> list[TrialSummary]:
    """Run a batch of trials sequentially (single-worker mode).

    Creates one environment instance and reuses it for all trials.
    """
    trial_indices = list(trial_ids)
    if not trial_indices:
        return []

    # OmniGibson / Isaac Sim reads sys.argv and can crash on unknown flags.
    # Strip our CLI args while instantiating the env.
    original_sys_argv = sys.argv[:]
    try:
        sys.argv = sys.argv[:1]
        env = instantiate(env_factory)
    finally:
        sys.argv = original_sys_argv

    multi_turn_prompt = env_factory["cfg"].get("multi_turn_prompt", None)

    summaries: list[TrialSummary] = []
    for trial in tqdm(trial_indices, desc="Running Trials"):
        summary = _run_trial_with_retries(env, trial, args, config, multi_turn_prompt)
        summaries.append(summary)

    summaries.sort(key=lambda s: s.trial)
    return summaries


# ---------------------------------------------------------------------------
# Timeout wrapper
# ---------------------------------------------------------------------------

def _run_single_trial_with_timeout(
    env: CodeExecutionEnvBase,
    trial: int,
    args,
    config: dict[str, Any],
    multi_turn_prompt: str | None,
    timeout_s: float,
) -> TrialSummary:
    """Run a single trial with a wall-clock timeout via SIGALRM."""
    timeout_seconds = max(1, int(timeout_s))
    timed_out = False
    trial_started_monotonic = time.monotonic()
    started_at = datetime.now(timezone.utc)
    output_dir = config.get("output_dir")
    writer = TrialResultWriter(output_dir) if output_dir else None
    if writer is not None:
        writer.start(trial=trial, started_at=started_at)
    telemetry_path = (
        Path(output_dir) / f"llm_calls_trial_{trial}.jsonl" if output_dir else None
    )
    context_manager = trial_llm_context(
        trial=trial,
        deadline_monotonic=trial_started_monotonic + timeout_seconds,
        telemetry_path=telemetry_path,
    )

    def _timeout_handler(signum: int, frame) -> None:  # type: ignore[override]
        nonlocal timed_out
        timed_out = True
        raise TimeoutError(f"Trial {trial} exceeded {timeout_seconds} seconds")

    partial_artifacts: dict[str, Any] = {}
    llm_context: TrialLLMContext | None = None
    previous_handler = None
    alarm_started = False
    try:
        previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
        alarm_started = True
        with context_manager as llm_context:
            summary = _run_single_trial(
                env, trial, args, config, multi_turn_prompt, partial_artifacts=partial_artifacts
            )
            outcome = (
                RunOutcome(summary.run_outcome)
                if summary.run_outcome is not None
                else RunOutcome.FINISHED
            )
            return _finalize_trial(
                summary=summary,
                writer=writer,
                llm_context=llm_context,
                outcome=outcome,
                started_monotonic=trial_started_monotonic,
            )
    except TrialResultPersistenceError as exc:
        raise exc.cause from None
    except LLMQueryError as exc:
        outcome = (
            RunOutcome.TRIAL_BUDGET_EXHAUSTED
            if exc.kind is LLMErrorKind.TRIAL_BUDGET_EXHAUSTED
            else RunOutcome.LLM_FAILED
        )
        try:
            return _finalize_exception(
                trial=trial,
                config=config,
                writer=writer,
                outcome=outcome,
                failure_kind=exc.kind.value,
                failure_message=exc.message,
                partial_artifacts=partial_artifacts,
                timeout_seconds=timeout_seconds,
                started_monotonic=trial_started_monotonic,
                llm_context=llm_context,
            )
        except TrialResultPersistenceError as persistence_error:
            raise exc from persistence_error.cause
    except (TimeoutError, KeyboardInterrupt) as exc:
        outcome = (
            RunOutcome.TRIAL_BUDGET_EXHAUSTED
            if timed_out or isinstance(exc, TimeoutError)
            else RunOutcome.CANCELLED
        )
        try:
            return _finalize_exception(
                trial=trial,
                config=config,
                writer=writer,
                outcome=outcome,
                failure_kind=(
                    "trial_budget_exhausted"
                    if outcome is RunOutcome.TRIAL_BUDGET_EXHAUSTED
                    else "cancelled"
                ),
                failure_message=str(exc),
                partial_artifacts=partial_artifacts,
                timeout_seconds=timeout_seconds,
                started_monotonic=trial_started_monotonic,
                llm_context=llm_context,
            )
        except TrialResultPersistenceError as persistence_error:
            raise exc from persistence_error.cause
    except InfrastructureFailure as exc:
        try:
            return _finalize_exception(
                trial=trial,
                config=config,
                writer=writer,
                outcome=RunOutcome.INFRASTRUCTURE_FAILED,
                failure_kind=exc.kind,
                failure_message=exc.message,
                partial_artifacts=partial_artifacts,
                timeout_seconds=timeout_seconds,
                started_monotonic=trial_started_monotonic,
                llm_context=llm_context,
            )
        except TrialResultPersistenceError as persistence_error:
            raise exc from persistence_error.cause
    except BaseException as exc:
        is_timeout = timed_out or isinstance(exc, TimeoutError)
        try:
            import requests as _requests
            is_timeout = is_timeout or isinstance(exc, _requests.exceptions.Timeout)
        except Exception:
            pass
        if is_timeout:
            outcome = RunOutcome.TRIAL_BUDGET_EXHAUSTED
            failure_kind = "trial_budget_exhausted"
        else:
            outcome = RunOutcome.EXECUTION_FAILED
            failure_kind = type(exc).__name__
        try:
            return _finalize_exception(
                trial=trial,
                config=config,
                writer=writer,
                outcome=outcome,
                failure_kind=failure_kind,
                failure_message=str(exc),
                partial_artifacts=partial_artifacts,
                timeout_seconds=timeout_seconds,
                started_monotonic=trial_started_monotonic,
                llm_context=llm_context,
            )
        except TrialResultPersistenceError as persistence_error:
            raise exc from persistence_error.cause
    finally:
        if alarm_started:
            signal.alarm(0)
        if previous_handler is not None:
            signal.signal(signal.SIGALRM, previous_handler)


def _llm_accounting(context: TrialLLMContext | None) -> dict[str, int | float]:
    snapshot = context.summary() if context is not None else {}
    return {
        "call_count": int(snapshot.get("logical_call_count", 0)),
        "attempt_count": int(snapshot.get("attempt_count", 0)),
        "retry_count": int(snapshot.get("retry_count", 0)),
        "token_count": int(snapshot.get("token_count", 0)),
        "elapsed_seconds": float(snapshot.get("elapsed_seconds", 0.0)),
        "last_call_index": int(snapshot.get("last_call_index", 0)),
    }


def _finalize_trial(
    *,
    summary: TrialSummary,
    writer: TrialResultWriter | None,
    llm_context: TrialLLMContext | None,
    outcome: RunOutcome,
    started_monotonic: float,
    failure_kind: str | None = None,
    failure_stage: str | None = None,
    failure_message: str | None = None,
) -> TrialSummary:
    llm = _llm_accounting(llm_context)
    summary.run_outcome = outcome.value
    summary.failure_kind = failure_kind
    summary.failure_stage = failure_stage
    summary.failure_message = failure_message
    summary.llm_call_count = int(llm["call_count"])
    summary.llm_attempt_count = int(llm["attempt_count"])
    summary.llm_retry_count = int(llm["retry_count"])
    summary.llm_elapsed_seconds = float(llm["elapsed_seconds"])
    if writer is not None:
        try:
            writer.finalize(
                {
                    "run_outcome": outcome,
                    "failure_kind": failure_kind,
                    "failure_stage": failure_stage,
                    "failure_message": failure_message,
                    "finished_at": datetime.now(timezone.utc),
                    "elapsed_seconds": time.monotonic() - started_monotonic,
                    "reward": summary.reward,
                    "task_completed": summary.task_completed,
                    "sandbox_rc": summary.sandbox_rc,
                    "llm": llm,
                }
            )
        except Exception as exc:
            raise TrialResultPersistenceError(exc) from exc
    return summary


def _finalize_exception(
    *,
    trial: int,
    config: dict[str, Any],
    writer: TrialResultWriter | None,
    outcome: RunOutcome,
    failure_kind: str,
    failure_message: str,
    partial_artifacts: dict[str, Any],
    timeout_seconds: int,
    started_monotonic: float,
    llm_context: TrialLLMContext | None,
) -> TrialSummary:
    failure_stage = llm_context.last_stage() if llm_context is not None else None
    if failure_stage == "unknown":
        failure_stage = None
    if outcome is RunOutcome.TRIAL_BUDGET_EXHAUSTED:
        summary = _build_timeout_summary(
            trial,
            timeout_seconds,
            partial_artifacts,
            config,
            TimeoutError(failure_message),
        )
    else:
        summary = TrialSummary(
            trial=trial,
            success=False,
            reward=float(partial_artifacts.get("reward", 0.0)),
            terminated=bool(partial_artifacts.get("terminated", False)),
            truncated=bool(partial_artifacts.get("truncated", False)),
            sandbox_rc=1,
            log=f"Trial {trial} {outcome.value}: {failure_message}",
            task_completed=partial_artifacts.get("info_step", {}).get("task_completed", False),
            code_path=None,
            num_regenerations=int(partial_artifacts.get("num_regenerations", 0)),
            num_finishes=int(partial_artifacts.get("num_finishes", 0)),
            num_code_blocks=int(partial_artifacts.get("num_code_blocks", 0)),
        )
    return _finalize_trial(
        summary=summary,
        writer=writer,
        llm_context=llm_context,
        outcome=outcome,
        started_monotonic=started_monotonic,
        failure_kind=failure_kind,
        failure_stage=failure_stage,
        failure_message=failure_message,
    )


def _build_timeout_summary(
    trial: int,
    timeout_seconds: int,
    pa: dict[str, Any],
    config: dict[str, Any],
    exc: BaseException,
) -> TrialSummary:
    """Build a TrialSummary from partial artifacts after a timeout."""
    raw_code = pa.get("raw_code", "")
    code_blocks = pa.get("code_blocks", [])
    code_block_metadata = pa.get("code_block_metadata", [])
    final_code = _annotate_code_blocks(code_blocks, code_block_metadata)

    info_step = pa.get("info_step", {"sandbox_rc": 1, "stdout": "", "stderr": str(exc)})
    if info_step.get("stderr") == "":
        info_step["stderr"] = str(exc)
    else:
        info_step["stderr"] += f"\n\nTimeout Error: {exc}"

    reward = pa.get("reward", 0.0)
    terminated = pa.get("terminated", False)
    truncated = pa.get("truncated", False)
    num_regenerations = pa.get("num_regenerations", 0)
    num_finishes = pa.get("num_finishes", 0)
    num_code_blocks = pa.get("num_code_blocks", len(code_blocks))

    log_lines = _build_log_lines(
        final_code, info_step, reward, terminated, truncated,
        num_regenerations, num_finishes, num_code_blocks,
        prefix=f"Trial {trial} timed out after {timeout_seconds} seconds.",
    )

    code_path = _save_trial_artifacts(
        config, trial,
        sandbox_rc=1,
        reward=reward,
        task_completed=info_step.get("task_completed", False),
        final_code=final_code,
        raw_code=raw_code,
        all_responses=pa.get("all_responses", []),
        log_lines=log_lines,
        visual_feedback_imgs=pa.get("visual_feedback_imgs", []),
        ensemble_data=pa.get("ensemble_data"),
        multiturn_ensemble_data=pa.get("multiturn_ensemble_data", []),
    )

    return TrialSummary(
        trial=trial,
        success=False,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        sandbox_rc=1,
        log="\n".join(log_lines),
        task_completed=info_step.get("task_completed", False),
        code_path=code_path,
        num_regenerations=num_regenerations,
        num_finishes=num_finishes,
        num_code_blocks=num_code_blocks,
    )
