# Capsule Runtime Guardrails Design

## Context

The first text-only LIBERO Capsule run exposed four control-plane failures:

1. The launcher warned and continued when a configured perception or motion
   service did not become ready.
2. `run_group` accepted groups whose values had not yet been defined in the
   runtime namespace.
3. A successful `append_recovery` remained advertised even though another
   append could not be accepted without new physical trace evidence.
4. Transient service failures were reported to the model as ordinary program
   failures, encouraging irrelevant source patches and recovery appends.

This change is deliberately limited to those four failures. It does not alter
the LIBERO task, visual-feedback switches, generated program semantics, video
recording, artifact layout, or unrelated reporting.

## Goals

- Start a trial only after all configured robot services and the Molmo service
  used by `FrankaLiberoApi` are reachable.
- Expose and execute only semantic groups whose source dependencies are
  currently satisfied.
- Suppress repeated recovery appends until execution has produced new physical
  trace evidence, while showing the model which recovery groups can run.
- Retry infrastructure failures without spending another LLM decision step or
  suggesting source repair, and classify exhausted retries separately from
  program failures.

## Non-goals

- Automatically choose or execute the next semantic group.
- Change the three visual-feedback switches in the LIBERO YAML.
- Add new perception models or robot APIs.
- Change source-generation prompts beyond runtime availability guidance.
- Retry an action after a robot side effect may have been attempted.
- Address video length, token metrics, artifact naming, or other observations
  from the experiment.

## Selected approach

Use prompt guidance backed by runtime guards and typed infrastructure failures.
Prompt-only filtering is insufficient because scripted actions and malformed
model choices can bypass it. A fully automatic group scheduler would change the
meaning of LLM-step control and is outside scope.

## Service readiness preflight

The launcher will assemble required endpoints from two sources:

- Every entry in top-level `api_servers`.
- `env.cfg.molmo_base_url` when the active API list includes
  `FrankaLiberoApi`.

The launcher will start configured local services as today, then wait for every
required endpoint, including services that were already running. Readiness is
bounded by 120 seconds. A timeout raises a typed infrastructure-readiness error
instead of printing a warning and continuing. If preflight fails, all processes
started by this launch are stopped before the error propagates, so a partial
startup cannot leak daemon services.

Environment construction and trial dispatch happen only after this preflight
succeeds. TCP reachability is the common readiness contract because the local
FastAPI services do not expose a uniform health endpoint and Molmo's vLLM port
is opened only once the server is accepting requests.

## Semantic-group dependency gate

The runtime will derive source dependencies from the segmenter's existing
`defined_names` and `used_names` facts. A used name is a group dependency only
when some source group defines that name. API functions, initial globals, and
safe builtins therefore remain external runtime names rather than false group
dependencies.

For each group:

- A dependency is satisfied when its name exists in the executor's current
  globals.
- A group is runnable when all source-defined dependencies are satisfied and
  existing repair, replay, and safety guards also permit execution.
- An attempted `run_group` with missing dependencies returns an invalid event
  with `safety_failure=missing_group_dependencies` and the missing names. It
  never reaches `exec`.

The action prompt will include bounded `runnable_group_ids` and blocked-group
dependency information. The `run_group` example will select a runnable group
when one exists. The runtime guard remains authoritative for scripted actions
and stale model responses.

## Recovery-append gate

After an append commits, the latest recovery generation records the trace
revision at which it was created. While no newer physical trace exists:

- `append_recovery` is removed from allowed prompt actions.
- The prompt explains that the existing recovery must be run or patched and
  then exercised before another append.
- A direct or scripted append is rejected before source-edit preparation with
  `no_new_physical_state_since_last_append`.

The prompt also maps the latest generation's authorized stable group keys back
to current group IDs and displays them as `runnable_recovery_group_ids`, after
intersecting them with dependency-runnable groups. Patching recovery source is
allowed, but patching alone does not create physical evidence and therefore
does not re-enable another append. A later recovery execution that adds trace
evidence does.

## Infrastructure failure classification and retry

Runtime failures will be classified from the exception and trace evidence.
Connection refusal, connect/read timeout, and HTTP 5xx responses are
infrastructure failures. Ordinary Python errors such as `NameError`, invalid
arguments, and contract violations remain program failures.

A `run_group` infrastructure failure is retried inside the same Capsule action
up to three total attempts with bounded backoff. These attempts do not consume
additional LLM decisions. Automatic retry is permitted only when the failed
attempt contains no trace entry for a robot side-effect API. If a side effect
was attempted, the runner aborts immediately because the remote outcome may be
ambiguous and replay could duplicate motion.

If a retry succeeds, the normal event and trace commit path continues. If all
safe retries fail, or retry is unsafe, a typed infrastructure exception escapes
the Capsule loop. The trial runner records `run_outcome=infrastructure_failed`,
preserves the service error in `failure_kind` and `failure_message`, and ends the
trial. The failure is not appended to model-facing repair history, so it cannot
trigger patch or append behavior.

## Testing strategy

Focused tests will cover:

- Waiting for already-running and newly launched required services.
- Preflight timeout, typed failure, and cleanup of processes launched earlier
  in the same startup.
- Molmo endpoint discovery only for `FrankaLiberoApi`.
- Runnable group calculation, prompt exposure, and hard blocking of missing
  dependencies.
- Removal of `append_recovery` after append, recovery group ID exposure, and
  the existing no-new-trace hard guard.
- Safe infrastructure retries within one logical step, exhausted retry
  classification, and refusal to retry after a robot side-effect trace.
- Preservation of ordinary source-error and successful-trial behavior.

All Python tests and experiment-facing verification will run in the prepared
WSL checkout after syncing the edited Windows files. No dependencies will be
installed or experiments launched from the Windows checkout.
