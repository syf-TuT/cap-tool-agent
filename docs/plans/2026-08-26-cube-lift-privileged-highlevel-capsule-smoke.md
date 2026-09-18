# Cube Lift Privileged High-Level Capsule Smoke Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a privileged Cube Lift Capsule profile that exposes only the existing high-level Franka primitives, then prove deterministic `5,6,5` reset identity and two successful same-worker oracle clean replays without starting Program sampling, Controller repair, VeRL, Ray, or an optimizer.

**Architecture:** Replace the Cube-Stack-only task checks with a pure task-profile registry shared by static configuration validators. Give Cube Lift its own one-cube state hash and deterministic Robosuite reset path, then add a lightweight replay-smoke entrypoint that validates the profile and environment YAML, probes `5,6,5`, and runs the existing oracle twice through one retry-disabled persistent worker.

**Tech Stack:** Python 3.10-3.12, dataclasses, PyYAML, NumPy, Robosuite/MuJoCo EGL, PyRoKi, multiprocessing, pytest, Ruff, WSL2 Ubuntu, `uv run --no-sync`.

---

## Execution constraints

- Edit only the Windows checkout at `F:\code\cap-x`.
- Run Python, pytest, Ruff, Robosuite, PyRoKi, and smoke commands only in the prepared WSL copy at `/home/capx/code/cap-x`.
- Before each WSL test, copy the files touched by that task from `/mnt/f/code/cap-x` into `/home/capx/code/cap-x`.
- Use this command prefix from the elevated PowerShell/Codex context:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin MUJOCO_GL=egl
```

- Do not run `uv sync`, install dependencies, initialize submodules, or launch experiments from Windows.
- The real smoke artifact belongs under ignored `artifacts/`; only its concise reproducibility summary belongs under `docs/`.
- Keep the existing two-cube Cube Stack hash bytes and legacy profile behavior unchanged.

### Task 1: Introduce the task-profile registry

**Files:**

- Create: `capx/rl/capsule/task_profiles.py`
- Modify: `capx/rl/capsule/compat.py`
- Modify: `tests/test_capsule_config.py`

**Step 1: Write the failing profile-selection tests**

Extend `tests/test_capsule_config.py` with these cases:

- the existing Cube Stack tuple without `task.profile` remains valid;
- explicit `robosuite_cube_stack_privileged` remains valid;
- explicit `robosuite_cube_lift_privileged_highlevel` with
  `environment=robosuite_cube_lift`, `api=franka_control_privileged`, and
  `privilege=privileged` is valid;
- Cube Lift without an explicit profile is rejected;
- an unknown profile is rejected;
- profile mismatches in environment, API, privilege, render, record-video, EGL, or PyRoKi fail;
- boolean `task.privilege: true` remains invalid; the task-level contract is the string
  `privileged`.

Use a helper that derives the Lift config from the existing `valid_config()` fixture, so every
unrelated 7+1 invariant remains identical.

**Step 2: Run the focused test and confirm RED**

Sync the changed test, then run:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /bin/cp /mnt/f/code/cap-x/tests/test_capsule_config.py tests/test_capsule_config.py
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin MUJOCO_GL=egl /home/capx/.local/bin/uv run --no-sync pytest -p no:cacheprovider tests/test_capsule_config.py -q
```

Expected: the Lift acceptance test and profile-specific rejection tests fail because
`validate_capsule_config()` still hard-codes Cube Stack and knows no `task.profile`.

**Step 3: Implement the pure registry**

Create a side-effect-free module with no YAML, Hydra, Robosuite, or simulator imports. Define a
frozen profile value object containing at least:

```python
@dataclass(frozen=True)
class CapsuleTaskProfile:
    name: str
    environment: str
    api: str
    privilege: str
    render: bool
    record_video: bool
    env_target: str
    env_config_target: str
    low_level: str
    api_classes: tuple[str, ...]
    required_runtime: tuple[tuple[str, bool], ...]
    required_server_targets: tuple[str, ...]
```

Register exactly these initial profiles:

- `robosuite_cube_stack_privileged` -> `FrankaPickPlaceCodeEnv`,
  `franka_robosuite_cubes_low_level`, privileged Franka API, EGL + PyRoKi;
- `robosuite_cube_lift_privileged_highlevel` -> `FrankaLiftCodeEnv`,
  `franka_robosuite_cube_lift_low_level`, privileged Franka API, EGL + PyRoKi.

Expose a resolver plus an error-collection function. Missing `task.profile` may infer only the
exact legacy Cube Stack environment/API/privilege tuple. Never infer Cube Lift, and never bind a
profile to one literal `task.config_path`, because materialized bundles may relocate the YAML.

**Step 4: Integrate the registry into `validate_capsule_config()`**

Remove only the Cube-Stack-specific task/runtime entries from `exact_values`. Append profile
validation errors to the existing aggregate error list. Preserve every schema, 7+1, GRPO,
Controller-frozen, and VeRL invariant unchanged.

**Step 5: Verify GREEN and the legacy contract**

Sync `task_profiles.py`, `compat.py`, and the test, then rerun the Step 2 pytest command. Expected:
all tests pass, including the original unprofiled Cube Stack fixture.

**Step 6: Commit the profile layer**

```powershell
git add capx/rl/capsule/task_profiles.py capx/rl/capsule/compat.py tests/test_capsule_config.py
git commit -m "Add Capsule task profile registry"
```

### Task 2: Validate referenced environment YAML against the selected profile

**Files:**

- Modify: `capx/rl/capsule/task_profiles.py`
- Modify: `scripts/capsule_rl/common.py`
- Modify: `capx/rl/capsule/main_ppo.py`
- Modify: `tests/test_capsule_scripts.py`
- Modify: `tests/test_capsule_main_ppo.py`

**Step 1: Write failing environment-contract tests**

In `tests/test_capsule_scripts.py`, create temporary, parse-only environment YAML fixtures and
test that runtime-path validation:

- accepts a matching Lift YAML even when it lives at a staged temporary path;
- rejects drift in `env._target_`, `env.cfg._target_`, `low_level`, `privileged`,
  `enable_render`, `viser_debug`, API list, `record_video`, `num_workers`, or PyRoKi service;
- requires exactly one PyRoKi server with host `127.0.0.1`, port `8116`, robot
  `panda_description`, and target link `panda_hand`;
- does not open `task.config_path` when `check_runtime_paths=False`.

In `tests/test_capsule_main_ppo.py`, add a focused test showing that the stable environment
snapshot is rejected when its YAML contents disagree with the selected profile. Assert a typed
`TrainerFactoryError`, not a simulator import or untyped YAML exception.

**Step 2: Confirm RED**

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /bin/cp /mnt/f/code/cap-x/tests/test_capsule_scripts.py tests/test_capsule_scripts.py
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /bin/cp /mnt/f/code/cap-x/tests/test_capsule_main_ppo.py tests/test_capsule_main_ppo.py
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin MUJOCO_GL=egl /home/capx/.local/bin/uv run --no-sync pytest -p no:cacheprovider tests/test_capsule_scripts.py tests/test_capsule_main_ppo.py -q
```

Expected: drifted YAML passes because current code checks only that the path exists.

**Step 3: Add pure environment-payload validation**

In `task_profiles.py`, add a function that accepts an already parsed mapping and a resolved
profile, returning all contract errors without file I/O. Check exact high-level target,
`CodeExecEnvConfig`, low-level environment, privilege flag, disabled render/Viser/video, exact API
class list, one worker, and the exact required service declaration.

Do not validate `output_dir` or require one repository-relative YAML path. Those are deployment
details rather than task identity.

**Step 4: Use the validator in both runtime-path paths**

In `scripts/capsule_rl/common.py`, after the existing path check and only when
`check_runtime_paths=True`, decode the referenced YAML, require a mapping root, and validate it
against the resolved profile. Convert UTF-8, YAML, shape, and drift failures into
`ConfigValidationError`.

In `main_ppo.py`, parse `environment_snapshot.raw_bytes` inside
`_snapshot_runtime_dependencies()` and apply the same pure validator before accepting the
snapshot hash. Preserve the stable-file snapshot; do not reopen the path. Convert failures into
`TrainerFactoryError`.

**Step 5: Verify GREEN and unchecked-path compatibility**

Sync the five changed files and rerun the Step 2 command. Expected: matching staged YAML passes,
every drift case fails before simulator/model startup, and unchecked placeholder paths still pass.

**Step 6: Commit environment contract validation**

```powershell
git add capx/rl/capsule/task_profiles.py scripts/capsule_rl/common.py capx/rl/capsule/main_ppo.py tests/test_capsule_scripts.py tests/test_capsule_main_ppo.py
git commit -m "Validate Capsule environment profiles"
```

### Task 3: Add the privileged high-level Cube Lift configuration artifacts

**Files:**

- Create: `env_configs/cube_lifting/capsule_rl/franka_robosuite_cube_lift_privileged_clean_replay.yaml`
- Create: `env_configs/cube_lifting/capsule_rl/franka_robosuite_cube_lift_capsule_smoke.yaml`
- Create: `env_configs/cube_lifting/capsule_rl/cube_lift_capsule_source_tasks.jsonl`
- Modify: `tests/test_capsule_config.py`
- Modify: `tests/test_capsule_scripts.py`

**Step 1: Write failing artifact tests**

Add tests that load the repository files by their final names and assert:

- the Capsule template selects `robosuite_cube_lift_privileged_highlevel` and the exact
  Lift/API/privilege tuple;
- it preserves the existing 7+1 Capsule-Critique and GRPO invariants;
- its referenced environment YAML passes the new profile validator;
- the environment uses `FrankaLiftCodeEnv`,
  `franka_robosuite_cube_lift_low_level`, `privileged: true`, and exactly
  `FrankaControlPrivilegedApi`;
- rendering, Viser, and video are false; trials and workers are both one; PyRoKi is the only
  service;
- the source JSONL contains exactly one row with `task_id` and `prompt`;
- the prompt explicitly names only the exposed high-level functions
  `get_object_pose`, `sample_grasp_pose`, `goto_pose`, `open_gripper`, and
  `close_gripper`, asks for one executable Python program, and does not advertise raw environment,
  joint-control, `grasp`, `lift`, `pick_and_lift`, or `home_pose` APIs.

**Step 2: Confirm RED**

Run the focused config and script tests. Expected: repository-file tests fail because the three
artifacts do not exist.

**Step 3: Add the clean-replay environment YAML**

Use the exact contract from the design:

```yaml
env:
  _target_: capx.envs.tasks.franka.franka_lift.FrankaLiftCodeEnv
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level: franka_robosuite_cube_lift_low_level
    privileged: true
    enable_render: false
    viser_debug: false
    apis:
      - FrankaControlPrivilegedApi
api_servers:
  - _target_: capx.serving.launch_pyroki_server.main
    port: 8116
    host: 127.0.0.1
    robot: panda_description
    target_link: panda_hand
record_video: false
trials: 1
num_workers: 1
```

Give it an ignored `outputs/cube_lift_privileged_clean_replay` output directory.

**Step 4: Add the Capsule template and source task**

Clone the proven Cube Stack Capsule template's non-task invariants. Change only the profile, task
identity, environment path, output path, and Program prompt wording. Retain model/service paths as
safe placeholders, because this milestone does not start them.

The one-line JSONL prompt must describe “pick up the red cube and lift it,” name the five existing
functions and their important arguments, state that quaternions are WXYZ, and require Python code
without code fences. Do not copy a new oracle into the dataset; the smoke reads
`FrankaLiftCodeEnv.oracle_code`.

**Step 5: Verify GREEN**

Sync the three artifacts and changed tests, then run:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin MUJOCO_GL=egl /home/capx/.local/bin/uv run --no-sync pytest -p no:cacheprovider tests/test_capsule_config.py tests/test_capsule_scripts.py -q
```

Expected: both legacy Stack and explicit Lift configuration tests pass without importing Robosuite
or contacting PyRoKi.

**Step 6: Commit the experiment contract**

```powershell
git add env_configs/cube_lifting/capsule_rl tests/test_capsule_config.py tests/test_capsule_scripts.py
git commit -m "Add privileged Cube Lift Capsule config"
```

### Task 4: Add a task-specific one-cube initial-state hash

**Files:**

- Modify: `capx/rl/capsule/initial_state.py`
- Modify: `tests/test_capsule_initial_state.py`

**Step 1: Write failing one-cube hash tests**

Add tests for:

- exactly one `primary` pose plus seven joints;
- mapping order, negative zero, insignificant float noise, quaternion normalization, and
  quaternion sign equivalence;
- changed cube position or joint value changes the hash;
- missing/extra cube names, wrong pose length, zero quaternion, wrong joint count, booleans,
  NaN, and infinity raise `ValueError`;
- an observation without `cube_poses` returns `None`, preserving old non-privileged reset behavior;
- an explicit fixed regression vector proves the existing Cube Stack hash is byte-for-byte
  unchanged.

**Step 2: Confirm RED**

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /bin/cp /mnt/f/code/cap-x/tests/test_capsule_initial_state.py tests/test_capsule_initial_state.py
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin MUJOCO_GL=egl /home/capx/.local/bin/uv run --no-sync pytest -p no:cacheprovider tests/test_capsule_initial_state.py -q
```

Expected: imports for the Cube Lift hash helpers fail.

**Step 3: Implement independent Cube Lift helpers**

Add and export:

```python
canonicalize_cube_lift_initial_state(...)
cube_lift_initial_state_sha256(...)
cube_lift_initial_state_sha256_from_observation(...)
```

Reuse the existing finite-float, pose, JSON, and SHA-256 mechanics, but require exactly
`{"primary"}`. Do not make the existing Stack function accept variable cube counts or change its
serialized field order/content.

**Step 4: Verify GREEN**

Sync both files and rerun the focused hash test. Expected: all Lift cases pass and the Stack
regression vector is unchanged.

**Step 5: Commit the state identity**

```powershell
git add capx/rl/capsule/initial_state.py tests/test_capsule_initial_state.py
git commit -m "Add Cube Lift initial state hash"
```

### Task 5: Make Cube Lift reset deterministic and fix pose semantics

**Files:**

- Modify: `capx/envs/simulators/robosuite_cube_lift.py`
- Modify: `tests/test_capsule_deterministic_reset.py`
- Modify: `tests/test_robosuite_observation_privilege.py`

**Step 1: Write failing constructor/reset AST tests**

Mirror the existing Cube Stack AST tests to assert that all three Robosuite `Lift(...)`
constructor branches receive `seed=seed` and that `reset()` calls `_reseed_robosuite(seed)` before
`robosuite_env.reset()`.

**Step 2: Write failing semantic reset tests**

Using a fake Robosuite environment and an instance created without the heavy constructor, test that:

- a seeded reset routes through `_reseed_robosuite` and synchronizes the placement sampler RNG;
- privileged reset returns a lowercase 64-character `info["initial_state_sha256"]`;
- non-privileged reset remains usable and omits the hash;
- the task prompt says to lift the red/primary cube, not stack two cubes;
- Robosuite's XYZW `cube_quat` is converted to WXYZ before constructing `viser.transforms.SE3`;
- identity XYZW `[0, 0, 0, 1]` therefore produces identity WXYZ `[1, 0, 0, 0]` in the
  robot-base pose.

**Step 3: Confirm RED**

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /bin/cp /mnt/f/code/cap-x/tests/test_capsule_deterministic_reset.py tests/test_capsule_deterministic_reset.py
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /bin/cp /mnt/f/code/cap-x/tests/test_robosuite_observation_privilege.py tests/test_robosuite_observation_privilege.py
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin MUJOCO_GL=egl /home/capx/.local/bin/uv run --no-sync pytest -p no:cacheprovider tests/test_capsule_deterministic_reset.py tests/test_robosuite_observation_privilege.py -q
```

Expected: seed-forwarding/reseed/hash/prompt/quaternion assertions fail against the current Lift
wrapper.

**Step 4: Implement the minimal simulator changes**

- pass `seed=seed` to each of the three `Lift(...)` constructors;
- replace the wrapper-only RNG assignment with `_reseed_robosuite(seed)`;
- convert `cube_quat` from XYZW to WXYZ before creating `SE3`;
- after settling and obtaining the privileged observation, compute the one-cube hash from the
  pose and `_current_joints`, adding it only when available;
- correct the low-level fallback task prompt to Cube Lift.

Do not expose this hash in the Program prompt or change the non-privileged observation surface.

**Step 5: Verify GREEN**

Sync the simulator and both tests, then rerun the Step 3 command. Expected: all deterministic
reset and privilege-observation tests pass.

**Step 6: Commit deterministic Cube Lift reset**

```powershell
git add capx/envs/simulators/robosuite_cube_lift.py tests/test_capsule_deterministic_reset.py tests/test_robosuite_observation_privilege.py
git commit -m "Make Cube Lift reset deterministic"
```

### Task 6: Expose and verify persistent replay worker identity

**Files:**

- Modify: `capx/rl/capsule/evaluator.py`
- Modify: `scripts/capsule_rl/server_adapter.py`
- Modify: `tests/test_capsule_evaluator.py`
- Modify: `tests/test_capsule_server_adapter.py`

**Step 1: Write failing worker-identity tests**

Add a real `spawn` test with a lightweight fake environment that executes two programs through
one `PersistentProcessReplayBackend`. Assert both executions report the same child PID and that
closing the backend clears the public worker identity.

Add an adapter test that expects `ConcreteGateRuntime.oracle()` to read the public worker PID and
uses the task-neutral error text “configured environment does not expose oracle_code.” Preserve
the existing Gate retry policy; the new dedicated smoke will disable retries explicitly.

**Step 2: Confirm RED**

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /bin/cp /mnt/f/code/cap-x/tests/test_capsule_evaluator.py tests/test_capsule_evaluator.py
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /bin/cp /mnt/f/code/cap-x/tests/test_capsule_server_adapter.py tests/test_capsule_server_adapter.py
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin MUJOCO_GL=egl /home/capx/.local/bin/uv run --no-sync pytest -p no:cacheprovider tests/test_capsule_evaluator.py tests/test_capsule_server_adapter.py -q
```

Expected: the public PID accessor is absent and the adapter still uses a private process field and
Cube-Stack-specific error text.

**Step 3: Add the read-only worker accessor**

Expose `PersistentProcessReplayBackend.worker_pid -> int | None` without changing worker startup,
task-key caching, reset, timeout, or replacement behavior. Return `None` before start and after
close. Update the adapter to consume this accessor and make only the oracle-source error text
task-neutral.

**Step 4: Verify GREEN**

Sync all four files and rerun the Step 2 command. Expected: the fake backend proves two sequential
requests share one process, and all existing retry/replacement tests remain green.

**Step 5: Commit replay identity support**

```powershell
git add capx/rl/capsule/evaluator.py scripts/capsule_rl/server_adapter.py tests/test_capsule_evaluator.py tests/test_capsule_server_adapter.py
git commit -m "Expose persistent replay worker identity"
```

### Task 7: Add the lightweight Cube Lift replay smoke entrypoint

**Files:**

- Create: `scripts/capsule_rl/cube_lift_privileged_replay_smoke.py`
- Create: `tests/test_capsule_cube_lift_smoke.py`
- Modify: `tests/test_capsule_scripts.py`
- Modify: `tests/test_capsule_scripts_package.py`

**Step 1: Write failing orchestration and artifact tests**

Design the script around small pure validators plus injectable factory/evaluator boundaries. Test:

- `--validate-only` parses the Capsule YAML, resolves the explicit Lift profile, validates the
  referenced environment YAML and one-row source task, then exits without sockets, simulator,
  processes, or output files;
- a non-ready PyRoKi endpoint fails before environment construction;
- `5,6,5` accepts only `h5a == h5b` and `h5a != h6`, and reports all three hashes on failure;
- the source row plus the real seed-5 hash constructs the exact `TaskInstanceV1` identity;
- one backend and one `CleanReplayEvaluator(max_failure_retries=0)` are used for both calls;
- PID drift, retries, `worker_replaced: true`, hash drift, non-success outcome, non-binary-one
  reward, or `task_completed: false` makes the smoke fail;
- both reset-evidence flags prove a fresh namespace and cleared API state;
- probe and evaluator close on success and exception paths;
- output collision is rejected by `common.atomic_write_json`;
- the artifact is not named like a formal Gate artifact and explicitly records
  `program_actor_used: false`, `controller_used: false`, `ray_used: false`,
  `verl_used: false`, and `optimizer_used: false`.

Add the module to the repository script import/entrypoint checks.

**Step 2: Confirm RED**

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /bin/cp /mnt/f/code/cap-x/tests/test_capsule_cube_lift_smoke.py tests/test_capsule_cube_lift_smoke.py
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /bin/cp /mnt/f/code/cap-x/tests/test_capsule_scripts.py tests/test_capsule_scripts.py
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /bin/cp /mnt/f/code/cap-x/tests/test_capsule_scripts_package.py tests/test_capsule_scripts_package.py
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin MUJOCO_GL=egl /home/capx/.local/bin/uv run --no-sync pytest -p no:cacheprovider tests/test_capsule_cube_lift_smoke.py tests/test_capsule_scripts.py tests/test_capsule_scripts_package.py -q
```

Expected: import and behavior tests fail because the smoke module does not exist.

**Step 3: Implement side-effect-free validation and CLI parsing**

Support these arguments:

```text
--config
--source-task
--seed-sequence 5,6,5
--replay-seed 5
--replays 2
--timeout-s 180
--output
--validate-only
```

Load the main config with runtime asset checks disabled, require the explicit
`robosuite_cube_lift_privileged_highlevel` profile, then independently parse and validate its
referenced environment YAML. Require exactly one source JSONL row. `--validate-only` stops here.

**Step 4: Implement the real smoke path**

For normal execution:

1. Check the YAML-declared PyRoKi TCP endpoint before importing/constructing Robosuite.
2. Create one probe using `YamlEnvironmentFactory` and run resets `5,6,5` in that instance.
3. Validate `h5a == h5b != h6` and obtain `FrankaLiftCodeEnv.oracle_code` from the probe.
4. Build one `TaskInstanceV1` for seed 5 using the canonical source row, profile tuple, and `h5a`.
5. Create exactly one `PersistentProcessReplayBackend(start_method="spawn")` and one
   `CleanReplayEvaluator(timeout_s=180, max_failure_retries=0)`.
6. Evaluate the same oracle twice, reading `backend.worker_pid` after each result.
7. Require identical PIDs, identical state hashes, two `ReplayOutcome.SUCCESS` values, two binary
   rewards of 1, two completed tasks, exactly one attempt per result, no worker replacement, and
   valid reset evidence.
8. Always close evaluator and probe.
9. Atomically publish a JSON artifact with mode
   `cube_lift_privileged_replay_smoke_v1`, config/source/environment hashes, reset hashes, replay
   PIDs/results, and the five false “unused subsystem” flags.

The script must not call `ConcreteGateRuntime._open_collection_session`, Program services,
Controller transports, `start_verl_workers`, Ray, or trainer code.

**Step 5: Verify GREEN**

Sync the new script and tests, then rerun the Step 2 command. Expected: all unit tests pass without
requiring Robosuite, PyRoKi, a model, or a GPU process.

**Step 6: Commit the smoke runner**

```powershell
git add scripts/capsule_rl/cube_lift_privileged_replay_smoke.py tests/test_capsule_cube_lift_smoke.py tests/test_capsule_scripts.py tests/test_capsule_scripts_package.py
git commit -m "Add Cube Lift privileged replay smoke"
```

### Task 8: Run focused WSL verification and the real PyRoKi smoke

**Files:**

- Create after the real run: `docs/cube-lift-privileged-capsule-smoke.md`

**Step 1: Sync the complete scoped change into WSL**

From elevated PowerShell, preserve relative paths while copying only scoped files:

```powershell
wsl.exe -d Ubuntu-22.04 --exec /bin/bash --noprofile --norc -lc 'cd /mnt/f/code/cap-x && cp --parents capx/rl/capsule/task_profiles.py capx/rl/capsule/compat.py capx/rl/capsule/main_ppo.py capx/rl/capsule/initial_state.py capx/rl/capsule/evaluator.py capx/envs/simulators/robosuite_cube_lift.py scripts/capsule_rl/common.py scripts/capsule_rl/server_adapter.py scripts/capsule_rl/cube_lift_privileged_replay_smoke.py tests/test_capsule_config.py tests/test_capsule_scripts.py tests/test_capsule_main_ppo.py tests/test_capsule_initial_state.py tests/test_capsule_deterministic_reset.py tests/test_robosuite_observation_privilege.py tests/test_capsule_evaluator.py tests/test_capsule_server_adapter.py tests/test_capsule_cube_lift_smoke.py tests/test_capsule_scripts_package.py env_configs/cube_lifting/capsule_rl/franka_robosuite_cube_lift_privileged_clean_replay.yaml env_configs/cube_lifting/capsule_rl/franka_robosuite_cube_lift_capsule_smoke.yaml env_configs/cube_lifting/capsule_rl/cube_lift_capsule_source_tasks.jsonl /home/capx/code/cap-x/'
```

**Step 2: Run focused regression tests**

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin MUJOCO_GL=egl /home/capx/.local/bin/uv run --no-sync pytest -p no:cacheprovider tests/test_capsule_config.py tests/test_capsule_scripts.py tests/test_capsule_main_ppo.py tests/test_capsule_initial_state.py tests/test_capsule_deterministic_reset.py tests/test_robosuite_observation_privilege.py tests/test_capsule_evaluator.py tests/test_capsule_server_adapter.py tests/test_capsule_cube_lift_smoke.py tests/test_capsule_scripts_package.py -q
```

Expected: every selected test passes. No test may start a Program actor, Controller, VeRL, Ray, or
optimizer.

**Step 3: Run Ruff on the scoped Python files**

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /home/capx/.local/bin/uv run --no-sync ruff check capx/rl/capsule/task_profiles.py capx/rl/capsule/compat.py capx/rl/capsule/main_ppo.py capx/rl/capsule/initial_state.py capx/rl/capsule/evaluator.py capx/envs/simulators/robosuite_cube_lift.py scripts/capsule_rl/common.py scripts/capsule_rl/server_adapter.py scripts/capsule_rl/cube_lift_privileged_replay_smoke.py tests/test_capsule_config.py tests/test_capsule_scripts.py tests/test_capsule_main_ppo.py tests/test_capsule_initial_state.py tests/test_capsule_deterministic_reset.py tests/test_robosuite_observation_privilege.py tests/test_capsule_evaluator.py tests/test_capsule_server_adapter.py tests/test_capsule_cube_lift_smoke.py tests/test_capsule_scripts_package.py
```

Expected: Ruff exits zero.

**Step 4: Validate the smoke inputs without services**

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin MUJOCO_GL=egl /home/capx/.local/bin/uv run --no-sync python -m scripts.capsule_rl.cube_lift_privileged_replay_smoke --config env_configs/cube_lifting/capsule_rl/franka_robosuite_cube_lift_capsule_smoke.yaml --source-task env_configs/cube_lifting/capsule_rl/cube_lift_capsule_source_tasks.jsonl --validate-only
```

Expected: validation succeeds and creates no artifact, subprocess, or network connection.

**Step 5: Start only PyRoKi in terminal A**

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin MUJOCO_GL=egl /home/capx/.local/bin/uv run --no-sync capx/serving/launch_servers.py --config-path env_configs/cube_lifting/capsule_rl/franka_robosuite_cube_lift_privileged_clean_replay.yaml --timeout 120
```

Wait until port 8116 is reported ready. Keep this terminal open.

**Step 6: Run the real smoke in terminal B**

Use a new output path; immutable evidence must never be overwritten:

```powershell
wsl.exe -d Ubuntu-22.04 --cd /home/capx/code/cap-x --exec /usr/bin/env PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin MUJOCO_GL=egl /home/capx/.local/bin/uv run --no-sync python -m scripts.capsule_rl.cube_lift_privileged_replay_smoke --config env_configs/cube_lifting/capsule_rl/franka_robosuite_cube_lift_capsule_smoke.yaml --source-task env_configs/cube_lifting/capsule_rl/cube_lift_capsule_source_tasks.jsonl --seed-sequence 5,6,5 --replay-seed 5 --replays 2 --timeout-s 180 --output artifacts/cube_lift_privileged_smoke_20260826/smoke.json
```

Expected artifact conditions:

- reset hashes satisfy `hashes[0] == hashes[2]` and `hashes[0] != hashes[1]`;
- both worker IDs are the same non-empty PID;
- both outcomes are `success`, both binary rewards are `1.0`, and both tasks completed;
- both replay hashes equal the probed seed-5 hash;
- each result used exactly one attempt with no worker replacement;
- Program actor, Controller, Ray, VeRL, and optimizer markers are all false;
- render and video remain disabled.

Stop terminal A with Ctrl-C after the artifact is safely written.

**Step 7: Record reproducibility evidence**

Create `docs/cube-lift-privileged-capsule-smoke.md` containing:

- date, Windows Git SHA, Capsule YAML SHA-256, environment YAML SHA-256, and source JSONL SHA-256;
- the exact validation, service, test, Ruff, and smoke commands;
- all three reset hashes;
- both replay PIDs, outcomes, binary rewards, completion flags, attempt counts, and state hashes;
- the artifact path;
- an explicit statement that no Program actor sampling, Controller repair, 7+1 group assembly,
  Ray, VeRL, Gate 4-7, or optimizer step ran, so this is readiness evidence rather than a training
  result.

**Step 8: Run final hygiene checks**

From Windows:

```powershell
git diff --check
git status --short
```

Inspect that ignored `artifacts/` output is not staged and that no unrelated user changes are
included.

**Step 9: Commit the verified result record**

```powershell
git add docs/cube-lift-privileged-capsule-smoke.md
git commit -m "Record Cube Lift privileged replay smoke"
```

## Final acceptance checklist

- The Program contract exposes only the five existing high-level privileged Franka functions.
- Legacy unprofiled privileged Cube Stack configuration still validates unchanged.
- Profile/YAML drift fails before simulator or model startup.
- Cube Lift pose semantics are WXYZ in robot-base coordinates.
- Seeded Cube Lift reset supplies a valid task-specific state hash.
- Real `5,6,5` reset evidence satisfies same/different/same identity.
- Two real oracle clean replays succeed through one persistent, unreplaced worker.
- No perception stack, Program model, Controller, VeRL, Ray, or optimizer participates.
- Focused tests and Ruff pass in WSL, and the observed evidence is documented.
