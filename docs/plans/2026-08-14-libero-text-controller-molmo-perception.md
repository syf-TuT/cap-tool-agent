# LIBERO Text Controller with Molmo Perception Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Configure the LIBERO-Object Capsule experiment for a text-only decision model while preserving Molmo inside `FrankaLiberoApi`.

**Architecture:** Reuse the runtime's existing separation between prompt-attached visuals and API-internal perception. Disable initial/action prompt imagery and wrist capture only in the LIBERO preset; leave the Molmo service configuration unchanged.

**Tech Stack:** YAML, Python pytest, WSL2, Ruff.

---

### Task 1: Switch the LIBERO preset to text-only decisions

**Files:**
- Modify: `tests/test_runtime_control_config.py`
- Modify: `env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml`
- Modify: `docs/libero-tasks.md`

**Step 1: Write the failing test**

Change the LIBERO preset assertions to require:

```python
assert config["use_visual_feedback"] is False
assert config["capsule_action_visual_feedback"] is False
assert config["use_wrist_camera"] is False
assert config["env"]["cfg"]["molmo_base_url"] == "http://127.0.0.1:8122/v1"
assert config["env"]["cfg"]["molmo_model_name"] == "allenai/Molmo2-8B"
```

**Step 2: Run the test to verify it fails**

Run in `/home/capx/code/cap-x`:

```bash
uv run --no-sync pytest tests/test_runtime_control_config.py -q
```

Expected: FAIL because the three visual flags are still true.

**Step 3: Implement the minimal configuration and documentation change**

Set the three flags to `false` in the YAML. Update `docs/libero-tasks.md` to state
that the controller is text-only, main-camera video remains enabled, and Molmo
continues to run as an API-internal perception service.

**Step 4: Run focused and regression tests**

Run:

```bash
uv run --no-sync pytest tests/test_runtime_control_config.py tests/test_libero_molmo_config.py -q
uv run --no-sync pytest tests/test_runtime_control_*.py tests/test_run_libero_batch.py tests/integrations/test_libero_integration.py tests/test_libero_molmo_config.py tests/test_api_registry.py tests/test_eval_utils.py -q
```

Expected: all tests pass.

**Step 5: Verify and commit**

Run Ruff on the changed Python test and `git diff --check`, then commit:

```bash
git add tests/test_runtime_control_config.py env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml docs/libero-tasks.md
git commit -m "Use text-only LIBERO Capsule decisions"
```
