# Capsule Single-A800 Fixed Profile Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the four obsolete Single-A800 fallback profiles and execute, audit, and validate only `fsdp_base_bf16_vllm_util_045`.

**Architecture:** Replace the list-valued OOM ladder with one shared fixed-profile identity and store the final memory settings directly in the canonical VeRL YAML. Simplify the launcher to one immutable profile with same-profile Controller seed retries, and update initial/failure audit schemas so no `retry_name` compatibility surface remains.

**Tech Stack:** Python 3.12, PyYAML, pytest, VeRL/FSDP/vLLM configuration, SeeTaCloud A800 runtime.

---

### Task 1: Specify the fixed profile contract in tests

**Files:**
- Modify: `tests/test_capsule_owned_services.py`

**Step 1: Update the repository-owned configuration expectations**

Require the workflow to contain:

```python
assert workflow["oom_profile"] == "fsdp_base_bf16_vllm_util_045"
assert "oom_ladder" not in workflow
```

Require the canonical profile to contain fixed microbatches, disabled dynamic batching, BF16 actor
and reference FSDP model dtype, vLLM utilization 0.45, and the fixed `capsule_runtime.oom_profile`.

**Step 2: Replace ladder/materialization tests**

Remove the public `materialize_retry_profile` import and cumulative transformation test. Replace it
with a test that proves a dry-run and a real attempt both use the canonical fixed profile and fixed
run-ID prefix.

Replace the GPU OOM advancement test with:

```python
def test_gpu_oom_stops_after_the_fixed_profile(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    runtime.oom_failures_remaining = 1
    runtime.oom_failure_gate = "gate02_seed"

    with pytest.raises(GateCommandError):
        execute_owned_service_workflow(...)

    assert len(runtime.configured_attempts) == 1
    assert runtime.configured_attempts[0].oom_profile == (
        "fsdp_base_bf16_vllm_util_045"
    )
```

Update same-profile guided retry assertions to use `attempt.oom_profile` and require the one fixed
identity for all seed attempts.

**Step 3: Run focused tests to verify RED**

Run in WSL:

```bash
.venv/bin/python -m pytest \
  tests/test_capsule_owned_services.py::test_repository_single_a800_profile_matches_exact_contract \
  tests/test_capsule_owned_services.py::test_repository_owned_workflow_matches_exact_service_and_audit_contract \
  tests/test_capsule_owned_services.py::test_dry_run_renders_exact_commands_and_does_not_create_outputs \
  tests/test_capsule_owned_services.py::test_gpu_oom_stops_after_the_fixed_profile \
  tests/test_capsule_owned_services.py::test_guided_randomness_uses_at_most_three_new_run_ids -q
```

Expected: failures show the current FP32/dynamic/0.30 base profile, list-valued ladder, old run-ID
prefix, and missing `oom_profile` attempt attribute.

### Task 2: Make the canonical YAMLs fixed-profile-only

**Files:**
- Modify: `env_configs/cube_stack/capsule_rl/franka_robosuite_cube_stack_capsule_single_a800_verl.yaml`
- Modify: `env_configs/cube_stack/capsule_rl/franka_robosuite_cube_stack_capsule_single_a800_owned_services.yaml`

**Step 1: Store the final VeRL settings directly**

Set:

```yaml
actor_rollout_ref:
  actor:
    fsdp_config:
      model_dtype: bf16
    ppo_micro_batch_size_per_gpu: 1
    use_dynamic_bsz: false
  rollout:
    gpu_memory_utilization: 0.45
    log_prob_micro_batch_size_per_gpu: 1
    log_prob_use_dynamic_bsz: false
  ref:
    fsdp_config:
      model_dtype: bf16
    log_prob_micro_batch_size_per_gpu: 1
    log_prob_use_dynamic_bsz: false
capsule_runtime:
  oom_profile: fsdp_base_bf16_vllm_util_045
```

Keep the 10240-token limits and all unrelated policy/trainer settings unchanged.

**Step 2: Replace the workflow ladder**

Delete `oom_ladder` and add:

```yaml
oom_profile: fsdp_base_bf16_vllm_util_045
```

**Step 3: Re-run the focused tests**

Expected: YAML-value assertions pass; launcher-shape assertions remain RED until Task 3.

### Task 3: Simplify the launcher to one immutable profile

**Files:**
- Modify: `scripts/capsule_rl/common.py`
- Modify: `scripts/capsule_rl/launch_owned_services.py`
- Test: `tests/test_capsule_owned_services.py`

**Step 1: Introduce the shared identity**

Define in `scripts/capsule_rl/common.py`:

```python
SINGLE_A800_OOM_PROFILE = "fsdp_base_bf16_vllm_util_045"
```

Import it into the launcher. Remove `OOM_LADDER`, `_retry_profile`, and
`materialize_retry_profile`.

**Step 2: Enforce the final profile in the loader**

Change `load_single_a800_resolved_profile` to require fixed microbatches, disabled dynamic
batching, BF16 actor/reference FSDP model dtype, vLLM utilization 0.45, and exact
`capsule_runtime.oom_profile == SINGLE_A800_OOM_PROFILE`.

Change `load_owned_services_workflow` to require exact scalar `oom_profile` and reject an
`oom_ladder` field as an unexpected obsolete contract.

**Step 3: Remove retry terminology from runtime state**

Rename `RuntimeContext.retry_name` and `AttemptResult.retry_name` to `oom_profile`. Make
`_attempt_run_id` and `_hypothetical_attempt` operate directly on the validated canonical profile.
Write `oom_profile` rather than `retry_name` to launcher failure and initial-audit artifacts, and
bump those changed artifact schemas to version 2.

**Step 4: Collapse execution to one profile loop**

Create one base attempt and retain only the Controller seed loop. Continue on guided-randomness
failure while seed budget remains; write failure evidence and re-raise every OOM or other terminal
error immediately. Remove ladder advancement and the exhausted-ladder assertion.

**Step 5: Run focused tests to verify GREEN**

Run the focused command from Task 1. Expected: all selected tests pass.

### Task 4: Reject old artifact identities and schemas

**Files:**
- Modify: `scripts/capsule_rl/analyze_artifacts.py`
- Modify: `scripts/capsule_rl/common.py`
- Modify: `tests/test_capsule_scripts.py`
- Modify: `tests/test_capsule_final_audit_contract.py`

**Step 1: Update the Gate 7 fixtures first**

Change initial-audit fixtures to schema version 2 with `oom_profile`, resolved profiles to the fixed
identity, and final runtime-audit fixtures to `fsdp_base_bf16_vllm_util_045`. Add focused rejection
coverage for an old schema/`retry_name` initial audit and a non-fixed final `oom_profile`.

**Step 2: Run artifact tests to verify RED**

Run in WSL:

```bash
.venv/bin/python -m pytest \
  tests/test_capsule_scripts.py -k "gate7_finalization and (oom_profile or initial_audit)" \
  tests/test_capsule_final_audit_contract.py -q
```

Expected: the analyzer still requires schema version 1/`retry_name`, and common validation still
accepts removed profile identities.

**Step 3: Implement fixed-profile validation**

Import `SINGLE_A800_OOM_PROFILE` in the analyzer, remove `_OOM_PROFILES`, and require initial audit
schema version 2 with exact `oom_profile`. Compare the resolved profile's
`capsule_runtime.oom_profile` against that field. In `common.py`, replace the profile set with an
exact equality check against the shared identity.

**Step 4: Run artifact tests to verify GREEN**

Run the Task 4 focused command. Expected: all selected tests pass, including explicit rejection of
old artifacts.

### Task 5: Update current documentation and run local regressions

**Files:**
- Modify: `docs/capsule_rl.md`

**Step 1: Replace the fallback section**

Document that the launcher always uses BF16 FSDP, fixed microbatch one, and vLLM utilization 0.45;
an OOM fails the attempt without a profile transition, while same-profile Controller seed retries
remain bounded at three.

**Step 2: Search for obsolete compatibility surfaces**

Run:

```bash
rg -n "base_dynamic_fp32|vllm_util_026|fixed_microbatch_1|fsdp_base_bf16([^_]|$)|oom_ladder|retry_name" \
  scripts/capsule_rl env_configs/cube_stack/capsule_rl tests docs/capsule_rl.md
```

Expected: no current production/config/test/doc compatibility references. Dated historical result
summaries are outside this search and remain unchanged.

**Step 3: Run focused and broad tests in WSL**

```bash
.venv/bin/python -m pytest tests/test_capsule_owned_services.py -q
.venv/bin/python -m pytest tests/test_capsule_scripts.py tests/test_capsule_final_audit_contract.py -q
.venv/bin/python -m pytest \
  tests/test_capsule_config.py \
  tests/test_capsule_evaluator.py \
  tests/test_capsule_server_factory.py \
  tests/test_capsule_trainer.py \
  tests/test_program_source.py \
  tests/test_capsule_owned_services.py \
  tests/test_capsule_scripts.py \
  tests/test_capsule_final_audit_contract.py -q
```

Expected: zero failures.

**Step 4: Commit the implementation**

```bash
git add scripts/capsule_rl/common.py scripts/capsule_rl/launch_owned_services.py \
  scripts/capsule_rl/analyze_artifacts.py \
  env_configs/cube_stack/capsule_rl/franka_robosuite_cube_stack_capsule_single_a800_verl.yaml \
  env_configs/cube_stack/capsule_rl/franka_robosuite_cube_stack_capsule_single_a800_owned_services.yaml \
  tests/test_capsule_owned_services.py tests/test_capsule_scripts.py \
  tests/test_capsule_final_audit_contract.py docs/capsule_rl.md
git commit -m "Use fixed BF16 A800 Capsule profile"
```

### Task 6: Verify the fixed profile on SeeTaCloud

**Files:**
- Sync only the committed implementation files to `/root/autodl-tmp/cap-x`

**Step 1: Establish and pin the new host key**

Connect to `connect.nmb1.seetacloud.com:40755`, record the server-presented fingerprint, and use
that exact `-hostkey` value for every subsequent noninteractive `plink`/`pscp` call. Never store the
password in files or logs.

**Step 2: Inspect the remote checkout and prerequisites**

Verify repository path, Git state, Python environment, A800 identity/free VRAM, model/config inputs,
and required credential presence without printing secret values. Preserve unrelated remote changes.

**Step 3: Sync and run focused remote tests**

Transfer the exact committed runtime/config/test files and run:

```bash
.venv/bin/python -m pytest \
  tests/test_capsule_owned_services.py \
  tests/test_capsule_scripts.py \
  tests/test_capsule_final_audit_contract.py -q
```

Expected: zero failures.

**Step 4: Run launcher dry-run**

Invoke `scripts.capsule_rl.launch_owned_services` with the canonical workflow, canonical fixed VeRL
profile, and the existing resolved Capsule config. Verify the printed run ID starts with
`fsdp_base_bf16_vllm_util_045-` and rendered commands reference the fixed resolved profile.

**Step 5: Run the applicable real verification**

If the existing immutable Capsule input and required API credentials are present, run the actual
owned-services workflow and require its fresh gate artifacts to identify only the fixed profile.
If an external prerequisite is absent, report the exact completed remote tests/dry-run and the
specific blocker without claiming the full experiment passed.

### Task 7: Final evidence review

**Files:**
- Review all changed files and remote outputs

**Step 1: Verify repository state and diff**

Run `git status --short`, `git diff HEAD^ --check`, and inspect the committed diff for unrelated
changes.

**Step 2: Re-run the decisive local test command**

Run the broad Task 5 pytest command freshly and record the exit status and pass count.

**Step 3: Report evidence**

Report local commit IDs, exact passing test counts, remote host verification level, fixed profile
identity, and any remaining external prerequisite. Do not claim a real gate run passed unless its
fresh output proves it.
