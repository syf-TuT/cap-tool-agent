# LIBERO Privileged Capsule Design

## Goal

Add an official LIBERO Object Capsule experiment configuration that uses
`FrankaLiberoPrivilegedApi` without changing the existing non-privileged Capsule baseline.

## Approach

Create a separate
`env_configs/libero/franka_libero_object_0_privileged_capsule_llm_step.yaml` configuration.
The configuration will reuse the current Capsule LLM-step behavior while enabling privileged
state in both the low-level LIBERO environment and `CodeExecEnvConfig`. It will expose only
`FrankaLiberoPrivilegedApi` to generated programs.

This avoids adding API-selection flags or YAML inheritance. The Capsule runtime already
instantiates APIs from `cfg.apis`, exposes their public functions, and records their declared
side effects, so no runtime Python changes are needed.

## Configuration

The new configuration will:

- target `libero_object` task 0;
- set both privileged flags to `true`;
- select `FrankaLiberoPrivilegedApi`;
- keep `agent_mode: capsule`, semantic-group execution, sparse terminal progress, task-success
  completion, and program-contract validation;
- use full prompt and diagnostic state so the Capsule Action model can inspect ground-truth
  object poses;
- disable visual feedback and wrist-camera prompt capture while retaining main-camera video;
- start only the PyRoKi server on port 8116;
- omit Molmo, SAM3, and Contact-GraspNet configuration; and
- write to a distinct privileged Capsule output directory.

The prompt will describe a complete public-API-only Capsule program. It will prohibit imports
and internal environment handles because program-contract validation builds a restricted public
execution namespace, even though the simulator state itself is privileged.

## Data Flow

The launcher instantiates `FrankaLiberoEnv` with privileged state enabled, then constructs
`FrankaLiberoPrivilegedApi` from `cfg.apis`. Capsule receives the API documentation in its
initial prompt and binds the API's public functions into its execution namespace. Calls such as
`get_object_pose`, `goto_pose`, and gripper operations run through the existing trace wrappers;
declared robot side effects continue to drive semantic-group and recovery safeguards.

## Failure Handling

Configuration tests will reject accidental fallback to the perception API, either privileged
flag being disabled, reintroduction of perception services, or loss of the Capsule safety and
state settings. Runtime service readiness remains handled by the existing launcher; only PyRoKi
is required by this API.

## Testing and Documentation

Add a focused YAML regression test in `tests/test_runtime_control_config.py`. Run it first before
the new YAML exists to prove that it fails for the missing feature, then add the configuration and
rerun the focused configuration tests in the prepared WSL environment.

Document the privileged Capsule command and its differences from the non-privileged Molmo/SAM3
configuration in `docs/libero-tasks.md`. A simulator trial is outside the source-adaptation scope;
the documented next step is a one-task, one-trial smoke test in a dedicated LIBERO runtime.
