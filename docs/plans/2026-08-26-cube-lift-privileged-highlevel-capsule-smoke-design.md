# Cube Lift Privileged High-Level Capsule Smoke Design

## Goal

Add a Cube Lift experiment profile to the Capsule-RL configuration and clean-replay
path. The Program continues to generate Python against the existing object-centric
Franka primitives, while privileged simulator state removes perception failures from
the experiment signal. The first milestone validates configuration, deterministic
reset, and real clean replay without running Program sampling, Controller repair, or
VeRL training.

## Experiment Contract

The Cube Lift profile is fixed to the following contract:

- task: Robosuite Cube Lift;
- high-level environment: `FrankaLiftCodeEnv`;
- low-level environment: `franka_robosuite_cube_lift_low_level`;
- API: `FrankaControlPrivilegedApi`;
- privilege: `privileged: true`;
- Program output: one complete executable Python program;
- robot primitives: `get_object_pose`, `sample_grasp_pose`, `goto_pose`,
  `open_gripper`, and `close_gripper`;
- services: PyRoKi only;
- rendering and video: disabled for replay.

This replaces the initially considered non-privileged profile. The purpose of the
privileged contract is to isolate Program and Capsule code-strategy behavior from
SAM3 and Contact-GraspNet failures. No new `grasp`, `lift`, or `pick_and_lift`
primitive is introduced.

## Scope

The milestone includes:

1. a profile-aware Capsule task contract;
2. a Cube Lift privileged clean-replay environment YAML;
3. a Cube Lift Capsule smoke configuration template and canonical source task;
4. deterministic Cube Lift reset and a canonical single-cube initial-state hash;
5. focused configuration and state-identity tests;
6. a WSL smoke using the real persistent replay backend and existing Cube Lift oracle;
7. a concise reproducibility record containing commands and observed results.

The milestone excludes:

- real Program actor sampling;
- Controller repair collection;
- 7+1 guided-group construction;
- owned-service changes for a single A800;
- VeRL optimizer, Gate 4-7, and training validation;
- non-privileged perception or visual-service orchestration.

## Architecture

### Task profiles

Introduce a core task-profile registry shared by Capsule configuration validation.
It initially describes two supported contracts:

- the existing privileged Cube Stack contract;
- the new privileged high-level Cube Lift contract.

The new Cube Lift configuration declares its profile explicitly. Existing Cube Stack
configuration that predates the profile field remains compatible and resolves only to
the original exact Cube Stack tuple. A profile does not loosen validation: its task
environment, API label, privilege, environment YAML, rendering flags, and runtime
requirements must agree.

When runtime paths are checked, validation also parses the referenced environment YAML
without instantiating Robosuite. It verifies the high-level target, low-level environment,
API class, privilege flag, render/video flags, and declared PyRoKi service. A misleading
profile label therefore cannot hide a different execution contract.

### Configuration artifacts

Add a Capsule-RL directory under `env_configs/cube_lifting/` containing:

- a server-only privileged Cube Lift clean-replay environment;
- a Capsule smoke/template configuration that selects the new task profile;
- one canonical Cube Lift source task whose prompt names the permitted existing
  object-centric primitives.

The environment configuration uses `FrankaLiftCodeEnv`,
`franka_robosuite_cube_lift_low_level`, `FrankaControlPrivilegedApi`,
`enable_render: false`, and `record_video: false`. Its only API server is PyRoKi.

### Deterministic initial-state identity

Cube Lift must synchronize the wrapper RNG, Robosuite RNG, and placement sampler on
every seeded reset, matching the deterministic reset behavior already used by Cube
Stack.

After settling the environment, Cube Lift computes a canonical initial-state SHA-256
from:

- the single cube position and normalized WXYZ quaternion in the robot-base frame;
- the seven robot joint positions.

Canonicalization rejects non-finite values and incorrect dimensions, normalizes the
quaternion sign, rounds insignificant floating-point noise, and serializes with a stable
key order. The resulting hash is returned only as
`info["initial_state_sha256"]`. Privileged object state remains available through the
selected privileged experiment API as intended.

The existing two-cube Cube Stack hash remains unchanged. Task-specific state layouts
must not be conflated.

## Data Flow

```text
source task + Capsule YAML
          |
          v
profile-aware static validation
          |
          v
YamlEnvironmentFactory -> FrankaLiftCodeEnv -> seeded Cube Lift reset
          |                                      |
          |                                      +-> canonical initial-state SHA-256
          v
existing Cube Lift oracle Python
          |
          v
FrankaControlPrivilegedApi -> PyRoKi -> Robosuite
          |
          v
reward + reset hash -> PersistentProcessReplayBackend result
```

The replay identity binds task ID, environment seed, program sample ID, source hash,
and initial-state hash. Replaying the same source and seed in the same persistent worker
must start from the same initial state.

## Failure Handling

- Unknown profiles and profile/task/API/privilege mismatches fail before simulator or
  model startup.
- Environment YAML drift from its declared profile fails static validation.
- A missing or malformed initial-state hash fails seed resolution and clean replay;
  random hashes and all-zero placeholders are not accepted as runtime evidence.
- A PyRoKi readiness failure stops the integration smoke before oracle replay.
- Seed determinism failure reports the three observed hashes for the `5,6,5` sequence.
- Oracle execution errors remain typed replay failures and are not converted into
  successful evidence.
- The current privileged Cube Stack profile and its exact validation contract remain
  supported.

## Verification

### Focused tests

Add tests that prove:

1. the new Cube Lift profile and environment YAML pass validation;
2. mismatched profile, environment, API, privilege, rendering, or service fields fail;
3. the existing privileged Cube Stack configuration still passes;
4. the single-cube hash is stable across mapping/float noise and changes with state;
5. invalid pose, quaternion, or joint inputs are rejected;
6. Cube Lift reset reseeds Robosuite and nested placement samplers;
7. Cube Lift reset returns a valid initial-state SHA-256.

All Python tests run from the prepared WSL checkout, never from the Windows checkout.

### Real WSL smoke

Run PyRoKi, construct the new environment through its YAML, and verify:

1. `reset(5)`, `reset(6)`, `reset(5)` produces `h5a == h5b` and `h5a != h6`;
2. the real `PersistentProcessReplayBackend` uses one worker for two consecutive oracle
   clean replays at the same seed;
3. both replays return `ReplayOutcome.SUCCESS`, `binary_reward == 1`, the same
   `initial_state_sha256`, and the same worker identity;
4. the environment records no video and opens no interactive renderer.

The smoke does not start a Program actor, Controller, guided collector, trainer, or
optimizer.

## Completion Criteria

The milestone is complete when the focused WSL tests pass, the `5,6,5` reset check is
deterministic, two same-worker privileged Cube Lift oracle replays succeed, and the
commands plus observed hashes/replay outcomes are recorded under `docs/`. These results
establish readiness for a later real 7+1 sampling milestone; they do not claim that
sampling or training has run.
