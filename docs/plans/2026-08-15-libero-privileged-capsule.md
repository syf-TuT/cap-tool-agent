# LIBERO Privileged Capsule Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a separate LIBERO Object Capsule experiment configuration that uses
`FrankaLiberoPrivilegedApi` and ground-truth state without changing the existing
non-privileged baseline.

**Architecture:** Keep the generic Capsule runtime unchanged and add one declarative YAML
configuration. Protect the configuration contract with a source-only pytest test, and document
the distinct privileged experiment workflow. Perform all pytest and Ruff commands in the
prepared WSL project after copying only the edited files from the Windows checkout.

**Tech Stack:** YAML, Python 3.12, pytest, Ruff, Markdown, WSL2 Ubuntu, uv

---

### Task 1: Add the failing privileged Capsule configuration test

**Files:**
- Modify: `tests/test_runtime_control_config.py`
- Test: `tests/test_runtime_control_config.py`

**Step 1: Write the failing test**

Append this test after the existing non-privileged LIBERO Capsule configuration test:

```python
def test_libero_object_privileged_capsule_yaml_uses_ground_truth_api():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (
        repo_root
        / "env_configs"
        / "libero"
        / "franka_libero_object_0_privileged_capsule_llm_step.yaml"
    )
    data = yaml.safe_load(config_path.read_text())
    cfg = data["env"]["cfg"]
    low_level = cfg["low_level"]

    assert low_level["suite_name"] == "libero_object"
    assert low_level["task_id"] == 0
    assert low_level["privileged"] is True
    assert cfg["privileged"] is True
    assert cfg["apis"] == ["FrankaLiberoPrivilegedApi"]
    assert "molmo_base_url" not in cfg
    assert "molmo_model_name" not in cfg

    assert data["agent_mode"] == "capsule"
    assert data["max_capsule_steps"] == 24
    assert data["capsule_execution_granularity"] == "semantic_group"
    assert data["capsule_progress_mode"] == "sparse_terminal"
    assert data["capsule_require_task_success_for_finish"] is True
    assert data["capsule_validate_program_contract"] is True
    assert data["capsule_action_visual_feedback"] is False
    assert data["capsule_prompt_state_level"] == "full"
    assert data["capsule_diagnostic_state_level"] == "full"
    assert data["use_visual_feedback"] is False
    assert data["use_wrist_camera"] is False
    assert data["use_parallel_ensemble"] is False
    assert data["record_video"] is True
    assert data["trials"] == 1
    assert data["num_workers"] == 1

    assert data["api_servers"] == [
        {
            "_target_": "capx.serving.launch_pyroki_server.main",
            "port": 8116,
            "host": "127.0.0.1",
            "robot": "panda_description",
            "target_link": "panda_hand",
        }
    ]

    prompt = cfg["prompt"].lower()
    assert "ground-truth object poses" in prompt
    assert "public api functions" in prompt
    assert "no imports" in prompt
    assert "env" in prompt
    assert "apis" in prompt
```

**Step 2: Copy the edited test into WSL**

Run from PowerShell:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/cp `
  /mnt/f/code/cap-x/tests/test_runtime_control_config.py `
  tests/test_runtime_control_config.py
```

Expected: exit code 0.

**Step 3: Run the test to verify RED**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env `
  PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin `
  /home/capx/.local/bin/uv run --no-sync pytest `
  tests/test_runtime_control_config.py::test_libero_object_privileged_capsule_yaml_uses_ground_truth_api `
  -q
```

Expected: FAIL with `FileNotFoundError` for
`franka_libero_object_0_privileged_capsule_llm_step.yaml`.

### Task 2: Add the privileged Capsule YAML

**Files:**
- Create: `env_configs/libero/franka_libero_object_0_privileged_capsule_llm_step.yaml`
- Test: `tests/test_runtime_control_config.py`

**Step 1: Create the minimal configuration**

Create this file:

```yaml
# LIBERO Object Task 0 with privileged Capsule LLM-step control.
# Usage: uv run --no-sync capx/envs/launch.py \
#   --config-path env_configs/libero/franka_libero_object_0_privileged_capsule_llm_step.yaml

env:
  _target_: capx.envs.tasks.franka.franka_libero_env.FrankaLiberoCodeEnv
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level:
      _target_: capx.envs.simulators.libero.FrankaLiberoEnv
      suite_name: libero_object
      task_id: 0
      privileged: true
      max_steps: 8000
      control_freq: 20
    privileged: true
    apis:
      - FrankaLiberoPrivilegedApi
    prompt: |
      You are controlling a Franka Emika robot with the API described below.
      Goal: {libero_environment_goal}
      Generate one complete executable Python program for the goal.
      Ground-truth object poses are available through the privileged public API functions.
      Use no imports and do not access the internal env or APIS handles.
      Use only direct calls to the public API functions and safe builtins.
      Robot side effects must be top-level.
      Use at most one robot side-effect API call per semantic group.
      You may write Python code comments for reasoning.
      Write only executable Python code and do not use code fences.
      The public API functions below are already available in the restricted execution environment.

agent_mode: capsule
max_capsule_steps: 24
capsule_execution_granularity: semantic_group
capsule_progress_mode: sparse_terminal
capsule_require_task_success_for_finish: true
capsule_validate_program_contract: true
capsule_action_visual_feedback: false
capsule_prompt_state_level: full
capsule_diagnostic_state_level: full

api_servers:
  - _target_: capx.serving.launch_pyroki_server.main
    port: 8116
    host: 127.0.0.1
    robot: panda_description
    target_link: panda_hand

record_video: true
use_visual_feedback: false
use_wrist_camera: false
use_parallel_ensemble: false
output_dir: ./outputs/franka_libero_object_0_privileged_capsule_llm_step

trials: 1
num_workers: 1
```

**Step 2: Copy the configuration into WSL**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/cp `
  /mnt/f/code/cap-x/env_configs/libero/franka_libero_object_0_privileged_capsule_llm_step.yaml `
  env_configs/libero/franka_libero_object_0_privileged_capsule_llm_step.yaml
```

Expected: exit code 0.

**Step 3: Run the focused test to verify GREEN**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env `
  PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin `
  /home/capx/.local/bin/uv run --no-sync pytest `
  tests/test_runtime_control_config.py::test_libero_object_privileged_capsule_yaml_uses_ground_truth_api `
  -q
```

Expected: PASS.

**Step 4: Run both LIBERO Capsule configuration tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env `
  PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin `
  /home/capx/.local/bin/uv run --no-sync pytest tests/test_runtime_control_config.py -q `
  -k libero_object
```

Expected: both LIBERO Object configuration tests pass.

**Step 5: Commit the test and configuration**

```bash
git add tests/test_runtime_control_config.py \
  env_configs/libero/franka_libero_object_0_privileged_capsule_llm_step.yaml
git commit -m "Add privileged LIBERO Capsule config"
```

### Task 3: Document the privileged experiment

**Files:**
- Modify: `docs/libero-tasks.md`

**Step 1: Add a privileged Capsule section**

Immediately after the existing non-privileged Capsule section, document:

- the new configuration path;
- that it uses ground-truth object poses and `FrankaLiberoPrivilegedApi`;
- that only PyRoKi is required and Molmo, SAM3, and Contact-GraspNet are not used;
- that full ground-truth state is visible to Capsule Action prompts and diagnostics; and
- this one-trial smoke command:

```bash
source .venv-libero/bin/activate
MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/libero/franka_libero_object_0_privileged_capsule_llm_step.yaml \
  --total-trials 1 \
  --num-workers 1
```

**Step 2: Check documentation and YAML references**

Run from PowerShell:

```powershell
rg -n "privileged_capsule_llm_step|FrankaLiberoPrivilegedApi|PyRoKi" `
  docs/libero-tasks.md `
  env_configs/libero/franka_libero_object_0_privileged_capsule_llm_step.yaml
git diff --check
```

Expected: the documentation and YAML matches are present and `git diff --check` exits 0.

**Step 3: Commit the documentation**

```bash
git add docs/libero-tasks.md
git commit -m "Document privileged LIBERO Capsule run"
```

### Task 4: Run final source verification

**Files:**
- Verify: `tests/test_runtime_control_config.py`
- Verify: `tests/test_runtime_control_side_effects.py`
- Verify: `env_configs/libero/franka_libero_object_0_privileged_capsule_llm_step.yaml`
- Verify: `docs/libero-tasks.md`

**Step 1: Resync all executable source files to WSL**

Copy the final test and YAML files from `/mnt/f/code/cap-x` to the matching paths under
`/home/capx/code/cap-x` using `/usr/bin/cp`.

**Step 2: Run focused pytest verification in WSL**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env `
  PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin `
  /home/capx/.local/bin/uv run --no-sync pytest `
  tests/test_runtime_control_config.py tests/test_runtime_control_side_effects.py -q
```

Expected: all selected tests pass.

**Step 3: Run Ruff in WSL**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env `
  PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin `
  /home/capx/.local/bin/uv run --no-sync ruff check tests/test_runtime_control_config.py
```

Expected: `All checks passed!`.

**Step 4: Inspect the final change set**

Run:

```powershell
git status --short
git diff --check HEAD
git diff HEAD -- tests/test_runtime_control_config.py `
  env_configs/libero/franka_libero_object_0_privileged_capsule_llm_step.yaml `
  docs/libero-tasks.md
```

Expected: only the intended implementation files remain changed relative to the design/plan
commits, with no whitespace errors. Existing unrelated untracked model-cache files remain
untouched.

**Step 5: Record the unrun runtime smoke test**

Do not launch LIBERO from the Windows checkout. Report that source/configuration validation was
completed and that the documented one-trial simulator smoke test remains the next runtime step in
a dedicated LIBERO environment.
