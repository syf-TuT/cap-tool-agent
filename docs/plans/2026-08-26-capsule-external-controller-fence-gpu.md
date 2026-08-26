# Capsule-RL External Controller and Explicit Fence Repair Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make fenced Actor output fail unchanged as P0, require explicit Controller fence-removal edits, use synchronous non-thinking `qwen3.7-plus` responses up to 4096 tokens, and run PyRoKi on the single A800 before rerunning Gate 1--7.

**Architecture:** Preserve the clean-replay boundary and enhance only repair unit discovery so an already-failed fenced P0 exposes stable fence targets plus a trailing protocol-suffix target when needed. Extend the frozen OpenAI-compatible transport and the single-A800 owned launcher with a typed external-service mode, non-secret external Controller attestation, and PyRoKi CUDA policy while retaining local Controller compatibility.

**Tech Stack:** Python 3.10+, dataclasses, OpenAI Python SDK, YAML/JSON, pytest, WSL2 `uv run --no-sync`, SeeTaCloud SSH, VeRL/FSDP/vLLM, JAX/PyRoKi.

---

### Task 1: Explicit fence repair after unchanged P0 replay

**Files:**
- Modify: `tests/test_capsule_controller_collector.py`
- Modify: `tests/test_capsule_server_adapter.py`
- Modify: `capx/rl/capsule/controller.py`

**Step 1: Write failing unit tests for fenced source spans**

Add a test using the literal source `"```python\nprint('ok')\n```\n"`. Assert:

```python
spans = python_base_unit_spans(source)
assert source == "```python\nprint('ok')\n```\n"
assert [(span.unit_id, span.expected_source) for span in spans] == [
    ("fence_open", "```python\n"),
    ("group_0", "print('ok')"),
    ("fence_close", "```\n"),
]
```

Keep parameterized malformed-fence cases on the existing `base:program` fallback.

**Step 2: Write a failing collector test for explicit edits**

Construct a verified failed P0 whose source contains the fence and whose error is
`SyntaxError`. Script the Controller with two `replace` actions targeting
`base:fence_open` and `base:fence_close`, both with empty source, followed by `finish`.
Assert the trace retains the exact P0 source/error, contains two committed edits, and reconstructs
`"print('ok')\n"`. Also assert a `finish`-only Controller leaves the fenced source unchanged.

**Step 3: Verify RED in WSL**

Sync only the edited test file into `/home/capx/code/cap-x`, then run:

```bash
uv run --no-sync pytest \
  tests/test_capsule_controller_collector.py \
  tests/test_capsule_server_adapter.py -q
```

Expected: the new fenced-span and explicit-edit assertions fail because syntax errors currently
produce one whole-program unit.

**Step 4: Implement fenced-unit discovery and prompt guidance**

In `python_base_unit_spans`, detect only a single outer Markdown fence whose opener is the first
line and closer is the final nonempty line. Do not return transformed source. Parse the enclosed
body only to derive inner stable spans, translate offsets back into the immutable original, and
return opener/body/closer spans. If the wrapper or body is ambiguous, retain the whole-source
fallback.

Extend `_SYSTEM_PROMPT` to state that Markdown fences are Actor protocol errors already observed
in P0, and that removing them requires explicit empty-source `replace` actions against the two
fence targets. Do not add an automatic edit action or preprocessing path.

**Step 5: Verify GREEN and commit**

Run the Task 1 tests again, then:

```bash
git add capx/rl/capsule/controller.py \
  tests/test_capsule_controller_collector.py \
  tests/test_capsule_server_adapter.py
git commit -m "Require explicit controller edits for fenced actor output"
```

### Task 2: Synchronous non-thinking 4096-token external Controller requests

**Files:**
- Modify: `tests/test_capsule_controller_collector.py`
- Modify: `tests/test_capsule_config.py`
- Modify: `capx/rl/capsule/controller.py`
- Modify: `scripts/capsule_rl/common.py`
- Modify: `scripts/capsule_rl/server_adapter.py`
- Modify: `env_configs/cube_stack/capsule_rl/franka_robosuite_cube_stack_capsule_critique_grpo.yaml`

**Step 1: Write failing transport tests**

Change the strict default assertion to 4096 and capture the SDK call. Assert exactly:

```python
assert create_kwargs["model"] == "qwen3.7-plus"
assert create_kwargs["max_tokens"] == 4096
assert create_kwargs["stream"] is False
assert create_kwargs["extra_body"] == {"enable_thinking": False}
assert create_kwargs["response_format"] == {"type": "json_object"}
```

Add fail-fast cases rejecting non-boolean `stream`/`enable_thinking` and accepting only the
disabled values for this frozen runtime. Assert the client receives the key from
`CAPX_CONTROLLER_API_KEY`, while neither the config dataclass nor serialized call metadata
contains its value.

**Step 2: Write failing config and adapter tests**

Update the valid Capsule config fixture to contain:

```yaml
controller_service:
  max_output_tokens: 4096
  stream: false
  enable_thinking: false
```

Assert `validate_capsule_config` rejects missing/true/non-boolean request-mode fields and that the
server adapter passes all three values into `FrozenControllerConfig`.

**Step 3: Verify RED in WSL**

```bash
uv run --no-sync pytest \
  tests/test_capsule_controller_collector.py \
  tests/test_capsule_config.py \
  tests/test_capsule_server_adapter.py -q
```

Expected: new constructor/config assertions fail and the captured SDK call lacks `stream` and
`extra_body`.

**Step 4: Implement the request contract**

Set `FrozenControllerConfig.max_output_tokens=4096`; add `stream: bool=False` and
`enable_thinking: bool=False`; validate exact disabled values. Pass `stream=False` and
`extra_body={"enable_thinking": False}` explicitly in `chat.completions.create`. Require and
validate these fields in Capsule config, and forward them through `server_adapter` without a
fallback that could silently re-enable either feature.

Update the repository Capsule template to the external endpoint, `qwen3.7-plus`, 4096, disabled
streaming, and disabled thinking. Keep only the environment-variable name in YAML.

**Step 5: Verify GREEN and commit**

Run the Task 2 tests again, then commit the listed files with:

```bash
git commit -m "Use synchronous non-thinking external capsule controller"
```

### Task 3: External Controller service mode and audit chain

**Files:**
- Modify: `tests/test_capsule_owned_services.py`
- Modify: `tests/test_capsule_final_audit_contract.py`
- Modify: `tests/test_capsule_final_audit_consumers.py`
- Modify: `scripts/capsule_rl/launch_owned_services.py`
- Modify: `scripts/capsule_rl/analyze_artifacts.py`
- Modify: `env_configs/cube_stack/capsule_rl/franka_robosuite_cube_stack_capsule_single_a800_owned_services.yaml`

**Step 1: Write failing workflow tests**

Replace the single-A800 Controller service fixture with an external declaration containing
`mode`, endpoint, model, API-key environment name, timeout, output limit, stream, and thinking
fields. Assert dry-run rendering includes only `program` and `pyroki` commands, runtime spawn order
is `program`, `pyroki`, and no Controller PID is created or terminated.

Keep a focused synthetic local-mode test to prove the existing llama.cpp renderer/spawn cleanup
path still works.

**Step 2: Write failing attestation and cleanup tests**

Define an external Controller attestation with schema version 1, artifact type, endpoint, model,
request settings, API-key environment name, and `credential_present: true`; never include the
credential value. Assert resolved-config drift changes the hash and Gate 7 accepts a cleanup entry:

```json
{
  "name": "controller",
  "ownership": "external",
  "termination_confirmed": null
}
```

Program and PyRoKi entries remain owned PID/start-time identities with confirmed termination.
Assert Gate 7 rejects an external entry for either owned service, a missing credential-presence
proof, or a leaked credential-like field.

**Step 3: Verify RED in WSL**

```bash
uv run --no-sync pytest \
  tests/test_capsule_owned_services.py \
  tests/test_capsule_final_audit_contract.py \
  tests/test_capsule_final_audit_consumers.py -q
```

Expected: the loader still requires llama.cpp fields and exactly three spawned processes.

**Step 4: Implement typed local/external branches**

Validate `services.controller.mode` as `local` or `external`. Preserve the existing pinned
llama.cpp validation and attestation under `local`. Under `external`, reject local binary/model
fields, require the exact non-secret request contract, create a deterministic external attestation,
and skip Controller rendering/spawn/readiness/termination.

Generalize cleanup evidence to contain all three logical services with explicit ownership. Only
owned entries contain PID/start-time and must be confirmed dead. Update Gate 7 verification and
final hash bindings for the external attestation schema without weakening local attestation
recomputation.

Rename the post-Controller memory stage only if necessary for schema accuracy; otherwise document
that it means post-Controller-initialization and is immediate for an external Controller.

**Step 5: Verify GREEN and commit**

Run the Task 3 tests and commit with:

```bash
git commit -m "Support external controller in owned capsule workflow"
```

### Task 4: PyRoKi CUDA policy and single-A800 resolved configuration

**Files:**
- Modify: `tests/test_capsule_owned_services.py`
- Modify: `tests/test_capsule_scripts.py`
- Modify: `scripts/capsule_rl/launch_owned_services.py`
- Modify: `env_configs/cube_stack/capsule_rl/franka_robosuite_cube_stack_capsule_single_a800_owned_services.yaml`
- Modify: `env_configs/cube_stack/capsule_rl/franka_robosuite_cube_stack_capsule_single_a800_verl.yaml`

**Step 1: Write failing GPU-policy tests**

Assert the exact PyRoKi environment is:

```python
{
    "CUDA_VISIBLE_DEVICES": "0",
    "JAX_PLATFORMS": "cuda",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}
```

Assert Program identity remains CPU-only, all GPU Gates retain device 0/EGL, environment
credential isolation is unchanged, and the final BF16 retry resolves vLLM utilization to 0.45
without changing token caps or GRPO semantics.

**Step 2: Verify RED in WSL**

```bash
uv run --no-sync pytest tests/test_capsule_owned_services.py tests/test_capsule_scripts.py -q
```

Expected: exact workflow assertions still require the PyRoKi CPU environment.

**Step 3: Implement and validate the GPU policy**

Change only the single-A800 workflow and its strict loader contract. Preserve JAX non-preallocation
as a string environment value. Do not add a memory fraction or move Program identity onto GPU.
Keep the base profile and OOM ladder immutable except for the already-approved terminal 0.45
profile.

**Step 4: Verify GREEN and commit**

Run the Task 4 tests and commit with:

```bash
git commit -m "Run single-A800 PyRoKi service on CUDA"
```

### Task 5: Documentation, regression verification, and code simplification

**Files:**
- Modify: `docs/capsule_rl.md`
- Modify: `docs/capsule_rl_single_a800_seed5_20260825.md`
- Modify as needed: files changed in Tasks 1--4

**Step 1: Update operator documentation**

Document the unchanged-P0 fence failure, explicit Controller edit/replay sequence, external
Controller request contract, 4096-token limit, environment-only secret injection, PyRoKi CUDA
sharing, non-preallocation, and the prior stopped run's parse-failure evidence. Do not include an
API-key example value.

**Step 2: Simplify without behavior changes**

Apply the simplify skill to the scoped diff: remove duplicated local/external branching, keep
schema validation fail-closed, and preserve readable typed helpers. Do not broaden the feature.

**Step 3: Sync the scoped tree into WSL and run regression tests**

```bash
uv run --no-sync pytest \
  tests/test_capsule_controller_collector.py \
  tests/test_capsule_config.py \
  tests/test_capsule_server_adapter.py \
  tests/test_capsule_owned_services.py \
  tests/test_capsule_final_audit_contract.py \
  tests/test_capsule_final_audit_consumers.py \
  tests/test_capsule_scripts.py \
  tests/test_capsule_main_ppo.py \
  tests/test_capsule_trainer.py -q
ruff check capx/rl/capsule scripts/capsule_rl tests/test_capsule_*.py
```

Expected: all selected tests pass; only documented pre-existing warnings may remain.

**Step 4: Verify repository hygiene and commit**

Run `git diff --check`, inspect `git status --short`, confirm no credential literal appears with a
targeted secret scan, and commit documentation/simplification with:

```bash
git commit -m "Document external controller capsule rerun"
```

### Task 6: Remote deployment and immutable Gate 1--7 rerun

**Files:**
- Create remotely under ignored artifact roots only: new resolved configs, service logs, Gate
  artifacts, checkpoint, adapter, reload evidence, and final audit
- Modify locally after the run: `docs/capsule_rl_single_a800_seed5_20260825.md`

**Step 1: Verify local and remote cleanliness**

Require a clean local branch, upload the reviewed commits to `/root/autodl-tmp/cap-x`, and verify
the remote Git SHA and clean status. Pin the known SSH host fingerprint before authentication.

**Step 2: Run remote read-only preflight**

Confirm exactly one idle A800 80GB, at least 76 GiB free VRAM, no foreign process over 512 MiB,
120 GiB RAM, required `/dev/shm` and disk space, CUDA/JAX visibility, model-tree bytes, VeRL pin,
and no stale owned launcher processes.

**Step 3: Inject credentials interactively**

Use a remote interactive shell to read `CAPX_PROGRAM_API_KEY` and
`CAPX_CONTROLLER_API_KEY` without echo, export them only in the launcher process environment, and
start a new immutable run. Never place either value in command arguments, shell history, files, or
logs.

**Step 4: Execute and monitor Gates**

Run Gate 1--6, adapter reload smoke, Gate 7 candidate, owned cleanup, and Gate 7 finalizer. Preserve
all failed attempts and use a new config hash/run ID for every OOM-ladder transition. Allow at most
three new Controller-seed run IDs.

**Step 5: Verify final acceptance evidence**

Assert `runtime_verified: true`, one optimizer-step increment, nonzero LoRA gradient norm, exact
7+1 reward/mask and reference KL contracts, only LoRA trainability, adapter reload with finite
changed logits, Controller external attestation, explicit fence edit lineage, PyRoKi CUDA policy,
minimum host memory, and CUDA peak reserved no more than 70 GiB.

**Step 6: Curate the result and commit**

Update the concise local audit summary with run ID, commit SHA, config hash, commands, artifact
paths, metrics, and any retained failure evidence. Do not copy raw logs, checkpoints, models, or
secrets into Git. Run final verification and commit the summary.
