# Capsule-RL External Controller and Explicit Fence Repair Design

## Goal

Run the single-A800 LoRA Gate 1--7 workflow with `qwen3.7-plus` as the frozen external
Controller, a 4096-token Controller response limit, and PyRoKi on the A800. Preserve the Actor
protocol boundary: a Markdown-fenced Actor response is executed unchanged, fails as P0, and can
only become executable through auditable Controller edits followed by an independent replay.

## Actor and repair protocol

The Actor source remains immutable input to the first clean replay. No extraction, trimming,
fence removal, or syntax normalization occurs before P0 execution. A fenced response therefore
produces the normal typed `SyntaxError` replay result with binary reward zero. The failed replay
identity, raw source, error type, error message, diagnostics, and reward remain attached to the
repair trajectory and are included in the Controller state.

After that failure, unit discovery may describe the immutable bytes without changing them. For a
Markdown code fence at the start of the Actor response it exposes stable `base:fence_open` and
`base:fence_close` targets, plus editable Python units from the enclosed body when the body
parses. If raw text follows the closing fence, those bytes remain available as a separate
`base:protocol_suffix` target. This avoids turning a fenced response with trailing explanation
into a whole-program cleanup target.

The Controller prompt identifies every protocol target as an Actor protocol error and instructs
the Controller to remove each one with an explicit `replace` action whose replacement source is
empty. Every action is recorded in the normal immutable edit ledger and must precede semantic
repairs. A replacement which leaves a target byte-identical is rejected into the audit stream and
does not create a revision. The repaired `PT/P_hat` is reconstructed only from committed edits;
it is then replayed from the same initial state and scored independently. Gate 4 independently
re-derives the expected units from the immutable P0, rejects whole-program fence cleanup, and
checks the typed SyntaxError, zero reward, deletions, and edit order. The unit discovery helper
never returns cleaned source and never commits an edit on the Controller's behalf.

Malformed or incomplete fences retain the existing whole-program fallback. They are still
executed unchanged as P0 and require an explicit Controller replacement if they are to be
repaired.

## External Controller contract

The single-A800 runtime profile uses the OpenAI-compatible endpoint
`https://coding.dashscope.aliyuncs.com/v1` with model `qwen3.7-plus`. Requests are synchronous and
set all of the following explicitly:

- `stream=false`
- `extra_body={"enable_thinking": false}`
- `max_tokens=4096`
- the existing JSON-object response contract
- the existing 300-second timeout and frozen-Controller invariant

The API key is read only from `CAPX_CONTROLLER_API_KEY`. It is never accepted as a literal YAML
value and is never written to a resolved config, launcher command, artifact, log, or repository
file. The remote operator injects it interactively into the launcher environment.

The owned-service workflow represents the Controller as an external service. It validates and
attests the endpoint, model, request mode, output limit, and credential-environment name, but does
not spawn or clean up a local Controller process. Cleanup evidence retains a `controller` entry
marked external/not-owned so Gate 7 can distinguish intentional non-ownership from a missing
service. Existing local Controller profiles remain supported and continue to own their process.

## GPU ownership and memory behavior

The single-A800 PyRoKi service receives `CUDA_VISIBLE_DEVICES=0` and `JAX_PLATFORMS=cuda`.
`XLA_PYTHON_CLIENT_PREALLOCATE=false` prevents JAX from reserving most of the A800 before VeRL and
vLLM initialize. Program identity remains CPU-only. VeRL actor/rollout/reference and MuJoCo EGL
retain GPU 0 as before, with the selected BF16 plus vLLM `gpu_memory_utilization=0.45` retry
profile.

Preflight and launcher attestations record the PyRoKi CUDA policy without treating PyRoKi as a
CPU-only exception. Continuous memory monitoring and the existing stop thresholds remain active;
an OOM still requires a new config hash and run ID and follows the existing immutable ladder.

## Configuration and audit data

The Controller config gains explicit `stream` and `enable_thinking` fields with fail-fast type
validation. The single-A800 Capsule config resolves to `qwen3.7-plus`, 4096 output tokens, disabled
streaming, and disabled thinking. The external-service definition contains no local binary, GGUF,
archive, or model-file hash.

Controller attestation records only non-secret configuration and whether the named credential was
present. Gate artifacts continue to hash the resolved config, so endpoint/model/request-mode drift
changes the binding. Fence repair evidence is the existing P0 replay plus committed edit ledger;
no parallel normalization artifact is introduced.

## Failure behavior

- A fenced Actor response that is never explicitly repaired remains reward zero.
- A Controller response that silently returns cleaned Python instead of a valid edit remains a
  `parse_failure` or invalid action; the collector does not infer an edit.
- A repeated empty replacement of an already-deleted protocol target is an invalid no-op audit,
  not a committed revision.
- External API timeout, authentication failure, invalid response shape, or multiple choices is an
  infrastructure/protocol failure and stops the affected Gate with immutable failure evidence.
- PyRoKi CUDA initialization failure is a service readiness failure; the launcher cleans up only
  processes it owns and preserves logs and artifacts.
- Secrets are redacted by construction because only environment-variable names enter rendered
  workflow data.

## Tests and remote acceptance

Pure tests first demonstrate that fenced source is unchanged, the raw replay failure reaches the
Controller, trailing prose remains a separate protocol target, explicit protocol edits produce
the repaired source, no-op replacements do not advance revision, and no edit means no cleaned
source. Gate-verifier tests reject whole-program cleanup and accept only the typed explicit-edit
lineage.
Transport tests assert synchronous requests, disabled thinking, 4096 tokens, dedicated credentials,
and no secret serialization. Owned-service tests cover external Controller non-spawn/non-cleanup,
local-mode compatibility, PyRoKi CUDA environment, resolved-config drift, and Gate 7 cleanup
evidence.

After focused WSL tests and the Capsule regression suite pass, commit code and documentation,
synchronize the clean commit to `/root/autodl-tmp/cap-x`, inject both service credentials
interactively, and launch a new immutable run. Acceptance remains Gate 1--7 plus adapter reload,
`runtime_verified: true`, exactly one optimizer step, nonzero LoRA gradient norm, valid 7+1
reward/mask and reference-KL contracts, reloadable adapter with changed finite logits, and CUDA
peak reserved at or below 70 GiB.
