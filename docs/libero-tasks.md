# LIBERO-PRO Tasks

CaP-X supports LIBERO-PRO tasks by adding additional environment configs. 

## Setup

LIBERO requires a **separate virtual environment** because its Robosuite fork conflicts with the standard Robosuite used by other tasks.

```bash
# 1. Create a dedicated LIBERO venv
uv venv .venv-libero --python 3.12
source .venv-libero/bin/activate

# 2. Install LIBERO dependencies (--active targets the activated venv)
uv sync --active --extra libero --extra contactgraspnet

# 3. Set up an LLM proxy (needed for code generation)
echo "sk-or-v1-your-key" > .openrouterkey
python capx/serving/openrouter_server.py --key-file .openrouterkey --port 8110
```

> **Note:** The PyRoKi IK server (port 8116) is auto-started by the YAML config. No manual setup needed.

### Headless / non-interactive setup

On headless servers or in non-interactive environments (CI, Docker), LIBERO needs a config file at `~/.libero/config.yaml` pointing to the submodule paths. Create it manually if the interactive setup prompt is not available:

```bash
mkdir -p ~/.libero
cat > ~/.libero/config.yaml << EOF
assets: $(pwd)/capx/third_party/LIBERO-PRO/libero/libero/assets
bddl_files: $(pwd)/capx/third_party/LIBERO-PRO/libero/libero/bddl_files
benchmark_root: $(pwd)/capx/third_party/LIBERO-PRO/libero/libero
datasets: $(pwd)/capx/third_party/LIBERO-PRO/libero/libero/../datasets
init_states: $(pwd)/capx/third_party/LIBERO-PRO/libero/libero/init_files
EOF
```

## Running a Task

### Web UI (recommended for exploration)

```bash
source .venv-libero/bin/activate
python capx/envs/launch.py \
    --config-path env_configs/libero/franka_libero_spatial_0.yaml \
    --web-ui True
```

### Headless evaluation

```bash
source .venv-libero/bin/activate
python capx/envs/launch.py \
    --config-path env_configs/libero/franka_libero_spatial_0.yaml \
    --total-trials 10
```

### Capsule LLM-step for standard LIBERO-Object

The `franka_libero_object_0_capsule_llm_step.yaml` configuration targets standard
`libero_object` task 0. It uses the non-privileged `FrankaLiberoApi`, not a reduced or
ground-truth API. The initial generation prompt and every Action LLM step receive the
current main- and wrist-camera images. Prompt-visible state is limited to
proprioception, while full object state is written only to diagnostic artifacts and is
never added to the LLM prompt or runtime history.

Because this configuration sets `privileged: false`, every generated Capsule program
must pass the strict Python-subset preflight. This enforcement is independent of
`capsule_validate_program_contract`; setting that flag to `false` does not disable the
strict subset. Use no imports, classes, lambdas, `try`, `while`, or async constructs.
Dynamic or reflective calls, callable aliases, and attribute calls are unavailable. The
execution globals contain only restricted safe builtins and approved public API
bindings. Programs may directly call those bindings, safe builtins, or proven-pure
top-level helpers. They may use only statically bounded `for` loops, and total
computation must remain within the static budget. Legacy arbitrary-Python Capsule
programs are not available in non-privileged mode.

For a local source-only check, run the focused configuration test below from the
prepared WSL project at `/home/capx/code/cap-x`; do not run this command from the Windows
checkout. It parses the YAML and verifies the non-privileged prompt contract; it does
not start LIBERO, MuJoCo, model servers, or a robot trial.

```bash
cd /home/capx/code/cap-x
uv run --no-sync pytest tests/test_runtime_control_config.py -q \
  -k libero_object_capsule_llm_step_yaml_uses_approved_capabilities
```

Run the simulator commands below only on a server or dedicated LIBERO runtime with the
LIBERO environment and required LLM, PyRoKi, SAM3, and Contact-GraspNet services
available. They are not local Windows validation commands. Validate task 0 first, then
expand to all ten tasks.

Task-0 smoke test:

```bash
source .venv-libero/bin/activate
MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml \
  --total-trials 1 \
  --num-workers 1
```

Task 0 over five initial states:

```bash
source .venv-libero/bin/activate
MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
uv run --no-sync --active python -m capx.envs.scripts.run_libero_batch \
  --base-config-path env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml \
  --suites libero_object \
  --task-ids 0 \
  --total-trials 5 \
  --num-workers 1 \
  --output-dir ./outputs/libero_object_capsule_llm_step_task0
```

All ten tasks over five initial states each:

```bash
source .venv-libero/bin/activate
MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
uv run --no-sync --active python -m capx.envs.scripts.run_libero_batch \
  --base-config-path env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml \
  --suites libero_object \
  --total-trials 5 \
  --num-workers 1 \
  --output-dir ./outputs/libero_object_capsule_llm_step_all
```

Each run directory contains the generated Capsule program, sanitized initial/step
prompts, runtime trace JSON, per-step metrics, full-state diagnostic JSONL, and a
`capsule_visuals_trial_<NN>/` directory with the main/wrist PNG files. Persisted prompt
artifacts contain image metadata and relative PNG paths rather than inline base64 image
payloads.

## Choosing a Task

Each LIBERO task is specified by a **suite name** and **task index**. The YAML config's `low_level` field follows the pattern:

```
low_level:
  _target_: capx.envs.simulators.libero.FrankaLiberoEnv
  suite_name: <suite_name>
  task_id: <task_id>
```

For example, to run `libero_goal` task 2 ("put the wine bottle on top of the cabinet"), set:

```yaml
low_level:
  _target_: capx.envs.simulators.libero.FrankaLiberoEnv
  suite_name: libero_goal
  task_id: 2
```

### Available Suites, LIBERO

| Suite | Tasks | Description |
|-------|-------|-------------|
| `libero_10` | 10 | Core benchmark tasks (multi-object, multi-step) |
| `libero_90` | 90 | Extended benchmark (diverse kitchen/living room scenes) |
| `libero_object` | 10 | Object generalization (same scene, different objects) |
| `libero_spatial` | 10 | Spatial generalization (same objects, different positions) |
| `libero_goal` | 10 | Goal generalization (same scene, different goals) |

### Available Suites, LIBERO-PRO
| `libero_10_swap` | 10 | Position perturbation on `libero_10` |
| `libero_10_task` | 10 | Task perturbation on `libero_10` |
| `libero_object_swap` | 10 | Position perturbation on `libero_object` |
| `libero_object_task` | 10 | Task perturbation on `libero_object` |
| `libero_spatial_swap` | 10 | Position perturbation on `libero_spatial` |
| `libero_spatial_task` | 10 | Task perturbation on `libero_spatial` |
| `libero_goal_swap` | 10 | Position perturbation on `libero_goal` |
| `libero_goal_task` | 10 | Task perturbation on `libero_goal` |

All other LIBERO-PRO perturbations (i.e. Obj, Sem, Env) are also supported. To view the full list with the exact suite names please run the following in your terminal.

```bash
source .venv-libero/bin/activate
python -c "
from libero import benchmark  # type: ignore[import-not-found]
benchmark_dict = benchmark.get_benchmark_dict(help=True)"
```

### Example Tasks

**libero_10:**
| Index | Task |
|-------|------|
| 0 | Put both the alphabet soup and the tomato sauce in the basket |
| 1 | Put both the cream cheese box and the butter in the basket |
| 2 | Turn on the stove and put the moka pot on it |

**libero_spatial:**
| Index | Task |
|-------|------|
| 0 | Pick up the black bowl between the plate and the ramekin and place it on the plate |
| 1 | Pick up the black bowl next to the ramekin and place it on the plate |
| 2 | Pick up the black bowl from table center and place it on the plate |

**libero_goal:**
| Index | Task |
|-------|------|
| 0 | Open the middle drawer of the cabinet |
| 1 | Put the bowl on the stove |
| 2 | Put the wine bottle on top of the cabinet |

To list all tasks in a suite:

```bash
source .venv-libero/bin/activate
python -c "
from libero import benchmark
suite = benchmark.get_benchmark_dict()['libero_spatial']()
for i in range(suite.n_tasks):
    print(f'  [{i}] {suite.get_task(i).language}')
"
```

## Creating a Config for Any Task

Copy an existing config and change `suite_name` and `task_id`:

```bash
cp env_configs/libero/franka_libero_spatial_0.yaml env_configs/libero/franka_libero_goal_5.yaml
```

Then edit `suite_name` and `task_id` in the new file:

```yaml
low_level:
  _target_: capx.envs.simulators.libero.FrankaLiberoEnv
  suite_name: libero_goal
  task_id: 5
```

The task prompt is automatically populated from LIBERO's task language description via the `{libero_environment_goal}` placeholder.

## Config Reference

```yaml
env:
  _target_: capx.envs.tasks.franka.franka_libero_env.FrankaLiberoCodeEnv
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level:
      _target_: capx.envs.simulators.libero.FrankaLiberoEnv
      suite_name: libero_goal
      task_id: 5
    privileged: true                                       # true = ground-truth state
    apis:
      - FrankaLiberoPrivilegedApi                          # privileged API
    prompt: |
      You are controlling a Franka Emika robot with API described below.
      Goal: {libero_environment_goal}
      ...
```

## Available APIs

| API | Description |
|-----|-------------|
| `FrankaLiberoPrivilegedApi` | Ground-truth object poses, IK-based control (privileged) |
| `FrankaLiberoApi` | Perception-based control with SAM3 + GraspNet (requires servers) |
| `FrankaLiberoApiReduced` | Low-level abstractions for perception and control functions (requires servers) |
| `FrankaLiberoApiReducedSkillLibrary` | Low-level abstractions for perception and control functions + extra utility functions from automatically synthesized skill library (requires servers) |

## Using CuRobo
To use CuRobo uncomment the following functions in the API you want to use (i.e. `capx/integrations/franka/libero.py`, `capx/integrations/franka/libero_reduced.py`).

```python
def functions(self) -> dict[str, Any]:
  fns =  {
      ...
      # # CuRobo, uncomment these for the coding agent to use them!
      # "plan_grasp_trajectory": self.plan_grasp_trajectory,
      # "plan_with_grasped_object": self.plan_with_grasped_object,
      # "execute_joint_trajectory": self.execute_joint_trajectory,
  }
```
