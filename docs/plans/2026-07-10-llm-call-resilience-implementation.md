# LLM Call Resilience Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make every CaP-X LLM call bounded and observable, and persist a structured reason for every trial that does not finish normally.

**Architecture:** Add typed LLM failures, a trial-scoped context with a monotonic deadline and append-only attempt telemetry, and a unified two-attempt request policy inside the existing client. Extend the headless runner with atomic per-trial results and outcome-aware aggregation while preserving the public `query_model()` response contract and legacy configuration aliases.

**Tech Stack:** Python 3.10-3.12, Requests, contextvars, dataclasses, pytest, PowerShell-to-WSL2 test execution.

---

## Execution Rules

- Work in an isolated Windows git worktree when implementation begins.
- Edit only the Windows checkout/worktree.
- Before running tests, copy touched files to `/home/capx/code/cap-x` in WSL.
- Run all Python, pytest, simulator, and `uv` commands in WSL, never in the Windows `.venv`.
- Use TDD for every behavior change: failing test, observed failure, minimal implementation, passing test.
- Do not call a paid external provider until local tests pass and the user approves the smoke-test cost.
- Do not log prompts, responses, images, authorization headers, or secrets.

The standard WSL test prefix is:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin MUJOCO_GL=egl /home/capx/.local/bin/uv run --no-sync
```

## Task 1: Add Typed Errors and the Unified Retry Policy

**Files:**

- Create: `capx/llm/errors.py`
- Create: `capx/llm/resilience.py`
- Create: `tests/test_llm_resilience.py`

**Step 1: Write failing tests for error metadata and policy parsing**

Add tests that require a stable string enum, a sanitized typed exception, canonical environment
variable precedence, legacy alias translation, and the approved defaults.

```python
def test_retry_policy_uses_approved_defaults(monkeypatch):
    clear_llm_policy_env(monkeypatch)

    policy = LLMRetryPolicy.from_env()

    assert policy.max_attempts == 2
    assert policy.request_timeout_seconds == 60.0
    assert policy.retry_backoff_seconds == 1.0
    assert policy.retry_after_cap_seconds == 10.0
    assert policy.minimum_retry_budget_seconds == 5.0


def test_canonical_policy_variables_override_legacy_aliases(monkeypatch):
    monkeypatch.setenv("CAPX_LLM_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("CAPX_STREAMING_CHAT_COMPLETIONS_RETRIES", "9")
    monkeypatch.setenv("CAPX_LLM_REQUEST_TIMEOUT_SECONDS", "17")
    monkeypatch.setenv("CAPX_NONSTREAMING_REQUEST_TIMEOUT_SECONDS", "99")

    policy = LLMRetryPolicy.from_env()

    assert policy.max_attempts == 2
    assert policy.request_timeout_seconds == 17.0


def test_llm_query_error_exposes_safe_metadata_only():
    error = LLMQueryError(
        kind=LLMErrorKind.HTTP_5XX,
        call_index=4,
        attempt=2,
        status_code=503,
        elapsed_seconds=1.25,
        message="provider unavailable",
    )

    assert error.kind is LLMErrorKind.HTTP_5XX
    assert error.to_safe_dict()["status_code"] == 503
    assert "Authorization" not in json.dumps(error.to_safe_dict())
```

**Step 2: Copy the tests to WSL and verify they fail**

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /bin/cp /mnt/f/code/cap-x/tests/test_llm_resilience.py tests/test_llm_resilience.py
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /home/capx/.local/bin/uv run --no-sync pytest tests/test_llm_resilience.py -q
```

Expected: collection fails because `capx.llm.errors` and `capx.llm.resilience` do not exist.

**Step 3: Implement the types and compatibility parser**

Implement `LLMErrorKind` as `class LLMErrorKind(str, Enum)`. Implement `LLMQueryError` with
only safe scalar fields and a bounded message. Implement this policy shape:

```python
@dataclass(frozen=True)
class LLMRetryPolicy:
    max_attempts: int = 2
    request_timeout_seconds: float = 60.0
    retry_backoff_seconds: float = 1.0
    retry_jitter_seconds: float = 0.5
    retry_after_cap_seconds: float = 10.0
    minimum_retry_budget_seconds: float = 5.0
    first_content_timeout_seconds: float = 45.0

    @classmethod
    def from_env(cls) -> "LLMRetryPolicy":
        ...
```

Normalize existing variables so the old streaming value remains an attempt count and the old
non-streaming value remains a retry count. Clamp attempts to `1..2`; reject non-positive timeouts
with a clear `ValueError`.

**Step 4: Copy the implementation to WSL and verify the tests pass**

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /bin/cp /mnt/f/code/cap-x/capx/llm/errors.py capx/llm/errors.py
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /bin/cp /mnt/f/code/cap-x/capx/llm/resilience.py capx/llm/resilience.py
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /home/capx/.local/bin/uv run --no-sync pytest tests/test_llm_resilience.py -q
```

Expected: all policy and exception tests pass.

**Step 5: Commit**

```bash
git add capx/llm/errors.py capx/llm/resilience.py tests/test_llm_resilience.py
git commit -m "Add typed LLM resilience policy"
```

## Task 2: Add Trial-scoped Context and Append-only Telemetry

**Files:**

- Create: `capx/llm/context.py`
- Create: `tests/test_llm_context.py`

**Step 1: Write failing context and telemetry tests**

Test monotonic remaining budget, shared logical call indices, per-attempt accounting, stage
labels, thread safety, JSONL append behavior, and redaction.

```python
def test_attempt_record_contains_required_diagnostics(tmp_path):
    clock = FakeClock(100.0)
    telemetry_path = tmp_path / "llm_calls_trial_17.jsonl"

    with trial_llm_context(
        trial=17,
        deadline_monotonic=550.0,
        telemetry_path=telemetry_path,
        monotonic=clock,
    ) as context:
        with llm_call_stage("capsule_action"):
            call_index = context.next_call_index()
            context.record_attempt(
                call_index=call_index,
                attempt=1,
                mode="streaming",
                http_status=503,
                ttfb_ms=318,
                first_content_ms=None,
                started_monotonic=100.0,
                finished_monotonic=100.427,
                remaining_before_ms=450_000,
                outcome="retryable_http_error",
                error_kind="http_5xx",
                retry_scheduled=True,
            )

    record = json.loads(telemetry_path.read_text().splitlines()[0])
    assert record["call_index"] == 1
    assert record["attempt"] == 1
    assert record["stage"] == "capsule_action"
    assert record["trial_remaining_ms_after"] == 449_573
```

Add a concurrency test that submits several `next_call_index()` calls and asserts unique,
monotonically increasing values.

Treat the following as the required, exact JSONL field contract. Tests must assert every key,
including nullable values, rather than accepting abbreviated internal names:

```text
trial
call_index
stage
attempt
mode
http_status
ttfb_ms
first_content_ms
duration_ms
trial_remaining_ms_before
trial_remaining_ms_after
outcome
error_kind
retry_scheduled
```

The writer may accept convenient internal argument names, but the serialized record must use
these exact keys.

**Step 2: Run the focused test and observe failure**

Copy `tests/test_llm_context.py` into WSL and run:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /home/capx/.local/bin/uv run --no-sync pytest tests/test_llm_context.py -q
```

Expected: import failure for `capx.llm.context`.

**Step 3: Implement `TrialLLMContext`**

Implement:

```python
@dataclass
class TrialLLMContext:
    trial: int
    deadline_monotonic: float | None
    telemetry_path: Path | None
    monotonic: Callable[[], float] = time.monotonic

    def remaining_seconds(self) -> float | None: ...
    def next_call_index(self) -> int: ...
    def record_attempt(self, **fields: Any) -> None: ...
    def summary(self) -> dict[str, int | float]: ...
```

Expose `trial_llm_context(...)`, `get_trial_llm_context()`, and `llm_call_stage(name)`. Protect
counters and JSONL append with a lock. Flush and `os.fsync()` each record. Serialize only the
explicit schema fields.

**Step 4: Verify the focused tests pass**

Copy the new module to WSL and rerun `tests/test_llm_context.py -q`.

Expected: all tests pass and the telemetry file contains exactly one JSON object per line.

**Step 5: Commit**

```bash
git add capx/llm/context.py tests/test_llm_context.py
git commit -m "Add trial-scoped LLM telemetry"
```

## Task 3: Replace Non-streaming Infinite Retries with a Bounded Attempt Loop

**Files:**

- Modify: `capx/llm/client.py:184-362`
- Modify: `tests/test_llm_client.py`
- Modify: `tests/test_llm_resilience.py`

**Step 1: Add failing request-state tests**

Add fake response tests for:

- 503 then 200 uses attempts `[1, 2]` and returns success;
- 503 twice raises `LLMErrorKind.HTTP_5XX`;
- 401, 403, and 404 do not retry;
- connection timeout and read timeout retry once;
- 429 honors a capped `Retry-After`;
- remaining trial budget below five seconds suppresses attempt 2;
- non-streaming TTFB is measured from the first non-empty response chunk;
- no attempt records contain payloads or authorization headers.

Use a response fake that implements `__enter__`, `__exit__`, `iter_content()`, headers, and
status code so production code can use `stream=True` even for a non-streaming JSON response.

**Step 2: Run the tests and verify the old loop fails them**

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /home/capx/.local/bin/uv run --no-sync pytest tests/test_llm_client.py tests/test_llm_resilience.py -q
```

Expected: tests show unbounded status retries, missing typed errors, and missing telemetry.

**Step 3: Implement one reusable logical-call loop**

In `query_model()`:

1. Allocate one `call_index` from the active context, or a local index when no context exists.
2. Resolve `LLMRetryPolicy.from_env()` once.
3. Execute at most `policy.max_attempts`.
4. Cap Requests timeout by remaining trial budget when a context is active.
5. Use `requests.post(..., stream=True)` and read `iter_content()` to capture first-byte time.
6. Classify response status before decoding JSON.
7. Append telemetry before success, retry, or failure.
8. Sleep only for the bounded delay allowed by remaining budget.
9. Raise `LLMQueryError` after the final failure.

Delete the `while response.status_code in [404, 500, 502, 503, 504]` loop and its 240-second
sleep. Keep payload construction and the public return dictionary unchanged.

**Step 4: Run focused tests**

Copy `client.py` and the changed tests into WSL, then rerun the command from Step 2.

Expected: all non-streaming retry, budget, TTFB, and redaction tests pass.

**Step 5: Commit**

```bash
git add capx/llm/client.py tests/test_llm_client.py tests/test_llm_resilience.py
git commit -m "Bound non-streaming LLM retries"
```

## Task 4: Integrate Streaming Telemetry and Non-streaming Fallback

**Files:**

- Modify: `capx/llm/client.py:254-552`
- Modify: `tests/test_llm_client.py`
- Modify: `tests/test_llm_resilience.py`

**Step 1: Add failing streaming state-machine tests**

Cover these exact cases:

```python
def test_first_content_timeout_falls_back_to_non_streaming_once(...): ...
def test_empty_stream_falls_back_to_non_streaming_once(...): ...
def test_partial_stream_disconnect_discards_partial_and_falls_back(...): ...
def test_streaming_503_retries_in_streaming_mode(...): ...
def test_streaming_attempt_records_ttfb_and_first_content(...): ...
def test_streaming_failure_with_no_budget_does_not_fallback(...): ...
```

Assert that attempts share one call index, have attempt values 1 and 2, and never create a third
request.

**Step 2: Run the streaming tests and observe failure**

Run `pytest tests/test_llm_client.py tests/test_llm_resilience.py -q` in WSL.

Expected: current code only falls back after a completed empty stream and lacks unified attempt
telemetry.

**Step 3: Implement streaming attempt behavior**

Keep `query_model_streaming()` as the single-attempt generator used by the Web UI. Add internal
hooks for call index, attempt, timing, policy, and telemetry. Let `query_model()` own the logical
two-attempt decision:

- HTTP/connect failures retry in the original mode;
- first-content timeout, empty content, or partial interruption use non-streaming attempt 2;
- discard partial content before fallback;
- use `time.monotonic()` for all budgets and latencies;
- preserve the existing generator event types for direct Web UI callers.

**Step 4: Verify focused tests pass**

Copy touched files to WSL and rerun the focused tests.

Expected: all streaming state-machine and compatibility tests pass.

**Step 5: Commit**

```bash
git add capx/llm/client.py tests/test_llm_client.py tests/test_llm_resilience.py
git commit -m "Harden streaming LLM fallback"
```

## Task 5: Propagate Call Stages and Context into Ensemble Threads

**Files:**

- Modify: `capx/envs/trial.py:329-759`
- Modify: `capx/llm/client.py:555-756`
- Modify: `tests/test_llm_context.py`
- Modify: `tests/test_runtime_control_trial_loop.py`

**Step 1: Add failing stage and thread-propagation tests**

Assert that initial generation, capsule actions, multi-turn decisions, and visual feedback use
their approved stage names. Add an ensemble test that executes calls on multiple threads and
asserts that all records share the trial identifier while call indices remain unique.

**Step 2: Run the tests and observe missing stages/context**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /home/capx/.local/bin/uv run --no-sync pytest tests/test_llm_context.py tests/test_runtime_control_trial_loop.py -q
```

Expected: records use the default stage and ensemble worker threads have no active context.

**Step 3: Add stage scopes and copy contexts per submitted thread**

Wrap only the actual model calls:

```python
with llm_call_stage("initial_code"):
    out = _query_model(args, obs["full_prompt"])

with llm_call_stage("capsule_action"):
    response = _query_model(action_query_args, prompt)
```

Use `visual_feedback` and `multi_turn` at the corresponding call sites. For each ensemble task,
create a fresh `contextvars.copy_context()` and submit `context.run(query_single, ...)`; never run
one copied `Context` concurrently in multiple threads.

**Step 4: Verify stage and thread tests pass**

Rerun the focused command from Step 2.

Expected: correct stages and unique call indices under thread execution.

**Step 5: Commit**

```bash
git add capx/envs/trial.py capx/llm/client.py tests/test_llm_context.py tests/test_runtime_control_trial_loop.py
git commit -m "Propagate LLM call context"
```

## Task 6: Add Atomic Structured Trial Results

**Files:**

- Create: `capx/envs/trial_results.py`
- Create: `tests/test_trial_results.py`
- Modify: `capx/utils/launch_utils.py:51-66`

**Step 1: Write failing schema and persistence tests**

Test these transitions and fields:

```python
def test_writer_creates_running_result_before_trial_work(tmp_path): ...
def test_writer_atomically_finalizes_finished_result(tmp_path): ...
def test_writer_persists_llm_failure_and_accounting(tmp_path): ...
def test_writer_marks_residual_running_result_parent_guard_killed(tmp_path): ...
def test_old_trial_summary_constructor_remains_valid(): ...
```

Assert schema version 1, stable path `trial_<trial>_result.json`, sanitized message length, and no
leftover temporary file after finalization.

**Step 2: Run tests and verify imports fail**

Run `pytest tests/test_trial_results.py -q` in WSL.

Expected: `capx.envs.trial_results` does not exist and `TrialSummary` lacks outcome fields.

**Step 3: Implement result types and atomic writer**

Implement:

```python
class RunOutcome(str, Enum):
    RUNNING = "running"
    FINISHED = "finished"
    LLM_FAILED = "llm_failed"
    TRIAL_BUDGET_EXHAUSTED = "trial_budget_exhausted"
    EXECUTION_FAILED = "execution_failed"
    CANCELLED = "cancelled"
    PARENT_GUARD_KILLED = "parent_guard_killed"


class TrialResultWriter:
    def start(self, *, trial: int, started_at: datetime) -> Path: ...
    def finalize(self, result: Mapping[str, Any]) -> None: ...
    def mark_parent_guard_killed(self, *, process_rc: int, elapsed_seconds: float) -> None: ...
```

Write JSON to a sibling temporary path, flush and fsync it, then use `os.replace()`. Extend
`TrialSummary` with optional defaults for `run_outcome`, `failure_kind`, `failure_stage`,
`failure_message`, `llm_call_count`, `llm_attempt_count`, `llm_retry_count`, and
`llm_elapsed_seconds`.

**Step 4: Verify persistence tests pass**

Copy files to WSL and run `pytest tests/test_trial_results.py -q`.

Expected: all atomic persistence and compatibility tests pass.

**Step 5: Commit**

```bash
git add capx/envs/trial_results.py capx/utils/launch_utils.py tests/test_trial_results.py
git commit -m "Persist structured trial results"
```

## Task 7: Integrate Trial Budget and Failure Classification into the Runner

**Files:**

- Modify: `capx/envs/runner.py:39-373`
- Modify: `capx/utils/launch_utils.py:479-566`
- Create: `tests/test_runner_resilience.py`
- Modify: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write failing runner tests**

Test:

- the default complete trial budget is 450 seconds;
- the result file exists as `running` before environment reset;
- a typed LLM failure returns `run_outcome="llm_failed"` and keeps partial accounting;
- SIGALRM timeout returns `trial_budget_exhausted`;
- a local exception returns `execution_failed` without being mislabeled as LLM failure;
- task failure after a normal run remains `finished` with `task_completed=False`;
- outcome aggregation computes reward only over `finished` trials;
- legacy human-readable summary output remains present.

Use fake environments and fake clocks/signals; do not wait for real timeouts.

**Step 2: Run the tests and verify current behavior fails**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /home/capx/.local/bin/uv run --no-sync pytest tests/test_runner_resilience.py tests/test_runtime_control_trial_loop.py -q
```

Expected: missing running result, missing typed outcomes, and old all-trial reward denominator.

**Step 3: Establish one 450-second context per trial**

Read `CAPX_TRIAL_TIMEOUT_SECONDS` once, defaulting to 450. Create `TrialResultWriter` and the
trial LLM context before invoking `_run_single_trial()`. Use the same monotonic deadline for all
LLM attempts. Keep whole-trial execution to one attempt by default; retain an explicit legacy
override only if an existing caller configures it.

**Step 4: Classify every exit and finalize the result**

Catch `LLMQueryError`, `TimeoutError`/Requests timeout, cancellation, and other exceptions in
separate branches. Build `TrialSummary` fields from the active LLM context summary. Finalize the
result atomically in every handled branch, and clear SIGALRM in `finally`.

Update `_print_and_save_summary()` to print outcome counts and compute reward/task metrics over
`run_outcome == "finished"` only.

**Step 5: Verify runner tests pass**

Copy touched files to WSL and rerun the command from Step 2.

Expected: all runner classification and aggregation tests pass.

**Step 6: Commit**

```bash
git add capx/envs/runner.py capx/utils/launch_utils.py tests/test_runner_resilience.py tests/test_runtime_control_trial_loop.py
git commit -m "Classify trial resilience failures"
```

## Task 8: Add Reusable Result Loading and Parent-guard Finalization

**Files:**

- Create: `capx/utils/experiment_results.py`
- Create: `tests/test_experiment_results.py`
- Modify: `docs/development.md`

**Step 1: Write failing new-schema and legacy fallback tests**

Create fixtures for:

- several schema-version-1 results with distinct outcomes;
- a residual `running` result plus process return code 124;
- a historical reward-named trial directory with no structured result;
- mixed new and legacy results.

Assert new files take precedence, legacy inference remains available, and a 124 exit updates
only a residual `running` result to `parent_guard_killed`.

**Step 2: Run tests and verify the loader is missing**

Run `pytest tests/test_experiment_results.py -q` in WSL.

Expected: import failure for `capx.utils.experiment_results`.

**Step 3: Implement the loader, aggregator, and guard helper**

Provide:

```python
def load_trial_result(seed_output_dir: Path, trial: int) -> dict[str, Any]: ...
def finalize_parent_guard_exit(result_path: Path, process_rc: int, elapsed_seconds: float) -> None: ...
def aggregate_trial_results(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]: ...
```

Return explicit outcome counts, finished-only average reward, finished-only task completion
rate, and infrastructure failure counts. Never convert an unknown legacy case directly to
`llm_failed`; retain an `unknown_legacy_failure` marker.

**Step 4: Document the 450/480 contract**

Add a batch-runner example to `docs/development.md`:

```python
TRIAL_TIMEOUT_SECONDS = 450
PARENT_TIMEOUT_SECONDS = TRIAL_TIMEOUT_SECONDS + 30
```

Document the canonical environment variables, stable result paths, result schema, and required
call to `finalize_parent_guard_exit()` after a process returns 124.

**Step 5: Verify tests pass**

Copy files to WSL and run `pytest tests/test_experiment_results.py -q`.

Expected: all new-schema, legacy, aggregation, and parent-guard tests pass.

**Step 6: Commit**

```bash
git add capx/utils/experiment_results.py tests/test_experiment_results.py docs/development.md
git commit -m "Add structured experiment result loading"
```

## Task 9: Add Local Fault-injection Coverage and Run Regressions

**Files:**

- Create: `tests/test_llm_fault_server.py`
- Modify: `tests/test_llm_client.py`
- Modify: `README.md`

**Step 1: Write a local HTTP fault server test**

Use `http.server.ThreadingHTTPServer` on an ephemeral localhost port. Implement handlers for:

- delayed first byte followed by JSON success;
- 503 followed by 200;
- SSE heartbeats/empty deltas followed by first-content timeout;
- partial SSE content followed by connection close;
- 429 with an oversized `Retry-After` value.

Assert real Requests behavior produces accurate TTFB, correct attempt counts, bounded retry
delay, and approved fallback behavior.

**Step 2: Run the fault-server test and fix only test-discovered gaps**

Run in WSL:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /home/capx/.local/bin/uv run --no-sync pytest tests/test_llm_fault_server.py -q
```

Expected: all localhost fault scenarios pass without external network access.

**Step 3: Document operator-facing behavior**

Update `README.md` with:

- the two-attempt maximum;
- canonical environment variables and legacy compatibility;
- telemetry and result file names;
- outcome meanings;
- the rule that 450 seconds includes all trial work;
- the 480-second external guard requirement.

**Step 4: Run all focused regressions**

Sync all touched files to WSL, then run:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin MUJOCO_GL=egl /home/capx/.local/bin/uv run --no-sync pytest tests/test_llm_client.py tests/test_llm_resilience.py tests/test_llm_context.py tests/test_trial_results.py tests/test_runner_resilience.py tests/test_experiment_results.py tests/test_llm_fault_server.py tests/test_runtime_control_trial_loop.py -q
```

Expected: all focused tests pass with zero failures.

**Step 5: Run static checks on touched Python files**

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /home/capx/.local/bin/uv run --no-sync ruff check capx/llm/errors.py capx/llm/resilience.py capx/llm/context.py capx/llm/client.py capx/envs/trial_results.py capx/envs/runner.py capx/envs/trial.py capx/utils/launch_utils.py capx/utils/experiment_results.py tests/test_llm_client.py tests/test_llm_resilience.py tests/test_llm_context.py tests/test_trial_results.py tests/test_runner_resilience.py tests/test_experiment_results.py tests/test_llm_fault_server.py
```

Expected: zero Ruff errors.

**Step 6: Verify Web UI compatibility without launching a paid request**

Run the existing Web UI-related Python tests that import and exercise the streaming APIs. At a
minimum, import `capx.web.async_trial_runner` in WSL and confirm direct
`query_model_streaming()` event types remain unchanged.

Expected: imports succeed and existing streaming event tests pass.

**Step 7: Optionally run one real single-trial smoke test**

Only after user approval for the provider call, use the prepared WSL experiment command with
`total-trials=1`, `num-workers=1`, and video disabled. Verify both files exist:

```text
trial_01_result.json
llm_calls_trial_01.jsonl
```

Confirm the result contains `run_outcome`, and every telemetry row contains call index, attempt,
HTTP status, TTFB, total duration, and remaining budget.

**Step 8: Commit**

```bash
git add tests/test_llm_fault_server.py tests/test_llm_client.py README.md
git commit -m "Verify LLM call resilience"
```

## Final Verification

Use the `verification-before-completion` skill. Run fresh focused tests and Ruff checks, inspect
`git diff --check`, and inspect the final diff against the approved design. Do not claim the
feature is complete unless all required commands exit successfully and structured artifacts are
verified from at least the local fault-injection test.
