# Cube Stack Non-Privileged Prompt-Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a matched non-privileged `cube_stack` multiturn/Capsule benchmark pair whose initial prompt and exposed API are identical.

**Architecture:** Preserve existing VDM configurations as historical definitions. Add two sibling `*_matched.yaml` files with the same literal task prompt and `FrankaControlApiReducedSkillLibrary`; retain only execution-method fields in the configuration that needs them. A YAML-level pytest regression test prevents prompt/API drift.

**Tech Stack:** YAML experiment configuration, PyYAML, pytest, WSL2 `uv` environment.

---

### Task 1: Define the configuration-parity contract with a failing test

**Files:**

- Modify: `tests/test_runtime_control_config.py`
- Create: `env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm_matched.yaml`
- Create: `env_configs/cube_stack/franka_robosuite_cube_stack_capsule_vdm_matched.yaml`

**Step 1: Write the failing test**

Append this test:

```python
def test_cube_stack_nonprivileged_matched_configs_share_task_contract():
    multiturn = yaml.safe_load(
        Path("env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm_matched.yaml").read_text()
    )
    capsule = yaml.safe_load(
        Path("env_configs/cube_stack/franka_robosuite_cube_stack_capsule_vdm_matched.yaml").read_text()
    )

    multiturn_cfg = multiturn["env"]["cfg"]
    capsule_cfg = capsule["env"]["cfg"]
    assert multiturn_cfg["privileged"] is False
    assert capsule_cfg["privileged"] is False
    assert multiturn_cfg["apis"] == ["FrankaControlApiReducedSkillLibrary"]
    assert capsule_cfg["apis"] == ["FrankaControlApiReducedSkillLibrary"]
    assert multiturn_cfg["prompt"] == capsule_cfg["prompt"]
    assert "multi_turn_prompt" in multiturn_cfg
    assert capsule["agent_mode"] == "capsule"
```

**Step 2: Run the test and confirm the expected failure**

Run from WSL:

```bash
cd /home/capx/code/cap-x
uv run --no-sync pytest tests/test_runtime_control_config.py::test_cube_stack_nonprivileged_matched_configs_share_task_contract -q
```

Expected: `FileNotFoundError`, because neither matched YAML exists yet.

**Step 3: Commit the failing test**

```bash
git add tests/test_runtime_control_config.py
git commit -m "test: define cube stack prompt parity contract"
```

### Task 2: Add the matched multiturn configuration

**Files:**

- Create: `env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm_matched.yaml`
- Reference: `env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm.yaml`

**Step 1: Copy shared simulator and observation settings**

Start from the existing multiturn VDM configuration. Keep API servers, video
recording, image differencing, trial count, and worker count. Set a distinct
output directory: `./outputs/franka_robosuite_cube_stack_multiturn_vdm_matched`.

**Step 2: Replace the environment contract**

Use this exact block:

```yaml
env:
  _target_: capx.envs.tasks.franka.franka_pick_place.FrankaPickPlaceCodeEnv
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level: franka_robosuite_cubes_low_level
    privileged: false
    apis:
      - FrankaControlApiReducedSkillLibrary
    prompt: |
      You are controlling a Franka Emika robot with the API described below.
      Goal: place the red cube on top of the green cube and then open the gripper.
      Use only the provided non-privileged environment APIs.
      Do not use privileged state APIs.
      Write ONLY executable Python code. Do not use code fences. Import numpy explicitly if needed.
```

Retain the current `multi_turn_prompt` and
`use_legacy_multi_turn_decision_prompt: true`. These are multiturn-only
execution controls, not task information.

**Step 3: Run the focused test**

Run the Task 1 pytest command again.

Expected: FAIL because the matched Capsule configuration is not present yet.

### Task 3: Add the matched Capsule configuration

**Files:**

- Create: `env_configs/cube_stack/franka_robosuite_cube_stack_capsule_vdm_matched.yaml`
- Reference: `env_configs/cube_stack/franka_robosuite_cube_stack_capsule_vdm.yaml`

**Step 1: Copy the exact environment contract from Task 2**

The new Capsule YAML must use byte-for-byte identical `env.cfg.prompt`,
`privileged`, and `apis` values from the multiturn YAML. Copy the same service,
video, differencing, trial, and worker settings; set output directory
`./outputs/franka_robosuite_cube_stack_capsule_vdm_matched`.

**Step 2: Retain only Capsule runtime controls**

```yaml
agent_mode: capsule
max_capsule_steps: 12
checkpoint_policy: region
rollback_policy: none
capsule_feedback_level: source_region_repair_hint
```

Do not add Capsule execution/repair instructions to `env.cfg.prompt`: Capsule
constructs its runtime prompts separately.

**Step 3: Run the focused test and confirm it passes**

```bash
cd /home/capx/code/cap-x
uv run --no-sync pytest tests/test_runtime_control_config.py::test_cube_stack_nonprivileged_matched_configs_share_task_contract -q
```

Expected: `1 passed`.

**Step 4: Run the complete configuration test module**

```bash
cd /home/capx/code/cap-x
uv run --no-sync pytest tests/test_runtime_control_config.py -q
```

Expected: all tests pass.

**Step 5: Commit the implementation**

```bash
git add tests/test_runtime_control_config.py \
  env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm_matched.yaml \
  env_configs/cube_stack/franka_robosuite_cube_stack_capsule_vdm_matched.yaml
git commit -m "Add matched nonprivileged cube stack benchmarks"
```

### Task 4: Verify parsed configuration without launching the simulator

**Files:**

- Verify: the two new matched YAML files

**Step 1: Load through the configuration test**

The Task 3 pytest covers YAML parsing and the parity contract. It deliberately
does not instantiate Robosuite or start API servers.

**Step 2: Rerun evaluations only after tests pass**

Use the two new paths with identical model, seed policy, trial count, and worker
count. Treat older VDM output directories as historical non-matched results.
