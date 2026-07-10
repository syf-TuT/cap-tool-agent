# LLM Call Resilience and Trial Failure Accounting Design

**Status:** Approved on 2026-07-10

## Problem

CaP-X experiments currently collapse several different failure modes into missing or `NA`
results. Local evidence shows at least four distinct cases:

- cumulative LLM latency consumes the 450-second trial budget;
- an outer 450-second process guard kills the child before it can save a structured result;
- non-streaming 404/5xx responses enter an unbounded retry loop with 150-330 second sleeps;
- streaming connections can remain alive while producing no usable content.

The same seeds often complete when rerun with the same code and configuration, so these
failures must be tracked separately from task performance.

## Goals

- Bound every logical LLM call to at most two attempts.
- Keep the complete trial budget at 450 seconds, including LLM waiting, simulation, and
  execution.
- Persist enough evidence to explain every LLM attempt and every trial outcome.
- Preserve partial artifacts when a trial fails or exhausts its budget.
- Keep the existing `query_model()` input and `{content, reasoning}` output contract.
- Support all OpenAI-compatible providers rather than adding Packy-specific behavior.
- Preserve compatibility with existing YAML files, environment variables, Web UI calls, and
  `TrialSummary` construction.

## Non-goals

- Do not guarantee deterministic model output after a retry.
- Do not retry a logical LLM call more than once.
- Do not remove the 450-second experiment fairness budget.
- Do not rewrite historical result directories.
- Do not log complete prompts, responses, authorization headers, or API keys.
- Do not introduce a separate external resilience proxy.

## Chosen Approach

Add a small resilience layer to the existing core path. The LLM client owns request attempts,
retry decisions, timing, and typed errors. The runner owns the complete trial deadline and
turns typed errors into structured trial outcomes. Stable per-trial files replace inference
from reward-dependent directory names.

```text
runner creates 450s TrialContext
    -> trial executes environment and model calls
        -> LLM client performs at most two attempts
            -> each attempt appends telemetry
        -> success returns the existing response dictionary
        -> failure raises a typed LLMQueryError
    -> runner writes TrialSummary and trial_result.json

external process guard runs at 480s
    -> preserves 30s for structured shutdown and artifact persistence
```

## Components

### Trial-scoped LLM context

Create `capx/llm/context.py` with a context-managed trial record containing:

- trial identifier;
- monotonic deadline;
- stable telemetry path;
- logical call counter;
- attempt and retry counters;
- cumulative LLM elapsed time;
- current LLM stage.

Use `contextvars` so existing `query_model()` call signatures remain unchanged. Ensemble work
submitted to threads must explicitly copy the active context. Multiprocessing workers receive
independent contexts because each trial runs in its own process.

The runner starts the deadline before the trial begins, so environment reset, LLM calls,
simulation, and execution all consume the same budget.

### Typed LLM errors

Create `capx/llm/errors.py` with `LLMQueryError` and these stable error kinds:

```text
connect_timeout
read_timeout
connection_error
rate_limited
http_5xx
no_content
invalid_response
auth_error
request_rejected
trial_budget_exhausted
```

The exception carries safe metadata only: error kind, logical call index, final attempt,
HTTP status when available, elapsed time, and a bounded sanitized message.

### Request policy

The core client uses one policy object for streaming and non-streaming requests. The default
policy is:

```text
maximum attempts:             2
request timeout:             60s
base retry delay:             1s plus up to 0.5s jitter
Retry-After cap:             10s
minimum remaining retry time: 5s
```

The policy removes the existing unbounded 404/5xx loop.

## Retry State Machine

One logical call contains attempt 1 and, at most, attempt 2.

Retry once for:

- DNS and connection failures;
- connect and read timeouts;
- HTTP 408, 429, 500, 502, 503, and 504;
- streaming first-content timeout;
- streaming responses that end without content;
- a streaming connection that fails after partial content.

Do not retry:

- HTTP 400, 401, 403, and 404;
- invalid local arguments or serialization;
- invalid JSON or an incompatible response schema;
- an exhausted trial budget.

Retry delay consumes the trial budget. A retry is skipped when fewer than five seconds remain.
When the server supplies `Retry-After`, honor it up to ten seconds and only if the remaining
trial budget permits it.

For streaming calls, ordinary connection failures and retryable HTTP status codes retain the
requested mode. First-content timeout, empty content, partial-content interruption, or an
otherwise unusable stream switches attempt 2 to non-streaming. Partial streaming output is
discarded because the API has no safe resume semantics.

Non-streaming responses are read in chunks internally so TTFB can be measured accurately, but
callers still receive a complete response dictionary.

## Per-attempt Telemetry

Append one record per HTTP attempt to:

```text
<seed-output-dir>/llm_calls_trial_<trial>.jsonl
```

Example:

```json
{
  "trial": 17,
  "call_index": 4,
  "stage": "capsule_action",
  "attempt": 1,
  "mode": "streaming",
  "http_status": 503,
  "ttfb_ms": 318,
  "first_content_ms": null,
  "duration_ms": 427,
  "trial_remaining_ms_before": 382410,
  "trial_remaining_ms_after": 381983,
  "outcome": "retryable_http_error",
  "error_kind": "http_5xx",
  "retry_scheduled": true
}
```

Definitions:

- `call_index` identifies a logical call; both attempts share it.
- `attempt` is 1 or 2.
- `http_status` is null when no HTTP response was received.
- `ttfb_ms` measures time from request dispatch to the first non-empty response byte.
- `first_content_ms` measures time to the first usable streaming content delta.
- `duration_ms` measures the complete attempt.
- remaining-budget fields are sampled immediately before and after the attempt.
- `stage` identifies `initial_code`, `capsule_action`, `multi_turn`, `visual_feedback`, or
  another registered call site.

Append and flush each attempt before returning, retrying, or raising. This makes completed
attempts visible even if a later outer guard kills the process. Do not record prompt or response
content, authorization headers, or secrets.

## Structured Trial Results

Create a stable result file before environment reset:

```text
<seed-output-dir>/trial_<trial>_result.json
```

The file uses a versioned schema:

```json
{
  "schema_version": 1,
  "trial": 17,
  "run_outcome": "llm_failed",
  "failure_kind": "http_5xx",
  "failure_stage": "capsule_action",
  "failure_message": "HTTP 503 after 2 attempts",
  "started_at": "2026-07-10T10:00:00Z",
  "finished_at": "2026-07-10T10:04:12Z",
  "elapsed_seconds": 252.4,
  "reward": 0.24,
  "task_completed": false,
  "sandbox_rc": 1,
  "llm": {
    "call_count": 13,
    "attempt_count": 14,
    "retry_count": 1,
    "elapsed_seconds": 196.3,
    "last_call_index": 13
  }
}
```

`run_outcome` is independent from task success:

```text
finished
llm_failed
trial_budget_exhausted
execution_failed
cancelled
parent_guard_killed
```

The result starts as `running`. Final updates use a temporary file and atomic replacement.
If the outer 480-second guard still kills the child, the parent batch runner changes a residual
`running` result to `parent_guard_killed` and retains the last telemetry record.

Extend `TrialSummary` with optional, defaulted failure and LLM accounting fields so old
constructors continue to work. The structured result is canonical; existing human-readable
summaries remain available.

## Aggregation Semantics

Do not use `NA` as a failure reason. Report counts and rates for each `run_outcome` separately.
Compute reward averages and task completion rates over `finished` trials only. Report
infrastructure and budget failures beside task metrics so they cannot silently bias model
performance.

Historical results remain readable through the existing inference path. New aggregation first
looks for `trial_<trial>_result.json` and falls back only when it is absent.

## Configuration and Compatibility

Add canonical variables:

```text
CAPX_LLM_MAX_ATTEMPTS=2
CAPX_LLM_REQUEST_TIMEOUT_SECONDS=60
CAPX_LLM_RETRY_BACKOFF_SECONDS=1
CAPX_LLM_RETRY_AFTER_CAP_SECONDS=10
CAPX_TRIAL_TIMEOUT_SECONDS=450
CAPX_PARENT_TIMEOUT_GRACE_SECONDS=30
```

Continue accepting the existing streaming and non-streaming timeout/retry variables. Canonical
variables take precedence when set; otherwise the compatibility parser translates legacy
values into the unified policy. Existing model payloads and `query_model()` return values do
not change.

## Security and Privacy

- Redact authorization headers and known credential fields.
- Never serialize prompts, responses, images, or complete provider error bodies into telemetry.
- Bound failure messages to a small safe length.
- Use monotonic time for budgets and ISO-8601 UTC for human-facing timestamps.
- Keep telemetry append-only and result updates atomic.

## Testing

Unit tests use fake responses and a local HTTP server only. They cover:

- retryable 503 followed by success;
- two retryable failures producing a typed final error;
- non-retryable 401, 403, and 404;
- connection and read timeout recovery;
- capped `Retry-After` behavior;
- streaming empty content, first-content timeout, and partial interruption fallback;
- budget-aware retry suppression;
- accurate call index, attempt, status, TTFB, content latency, duration, and remaining budget;
- telemetry redaction;
- atomic running-to-final result transitions;
- structured LLM, execution, budget, cancellation, and parent-guard outcomes;
- compatibility with old `TrialSummary` constructors and `query_model()` callers.

Run focused tests first, then runtime-control regressions in the prepared WSL environment. A
local fault-injection server validates real Requests timing and SSE parsing without external
charges. A final optional one-trial provider smoke test verifies artifact creation but is not a
CI requirement.

## Rollout

1. Add types, context, telemetry, and tests without changing public call contracts.
2. Replace the client retry loops with the unified policy.
3. Add trial result persistence and parent-guard classification.
4. Update aggregation and documentation.
5. Run focused WSL tests and a fault-injection integration test.
6. Run one real single-trial smoke test after local verification.
