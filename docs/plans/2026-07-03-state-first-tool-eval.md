# State-First Tool Evaluation Implementation Plan

**Goal:** Add a state-first CaP-X tool-agent evaluation path that avoids image segmentation services and lets the agent use simulator-provided object poses.

**Architecture:** Add a general Franka state control API that exposes observation and motion primitives without initializing vision/grasp services. Add a cube-stack state-first YAML config that selects this API and states that object poses are available in `get_observation`, without encoding a task-specific action recipe.

**Tech Stack:** Python, pytest, YAML experiment config, remote SeeTaCloud Robosuite runtime.

---

### Task 1: Add State-First API Registration

**Files:**
- Modify: `capx/integrations/franka/control_reduced.py`
- Modify: `capx/integrations/__init__.py`
- Test: `tests/test_tool_registry.py`

**Steps:**
1. Write a failing test that constructs a fake state-first API and verifies its functions omit vision tools.
2. Implement a `FrankaStateControlApi` that initializes only motion dependencies and exposes `get_observation`, `solve_ik`, `move_to_joints`, `open_gripper`, and `close_gripper`.
3. Register it as `FrankaStateControlApi`.
4. Run the focused tests.

### Task 2: Add State-First Config

**Files:**
- Add: `env_configs/cube_stack/franka_robosuite_cube_stack_tool_state_first.yaml`
- Modify: `tests/test_tool_yaml_configs.py`

**Steps:**
1. Write a failing YAML test that asserts the state-first config uses `FrankaStateControlApi`, starts only the Pyroki API server, and mentions state pose availability without an action recipe.
2. Add the YAML config.
3. Run YAML tests.

### Task 3: Verify and Run Remote Evaluation

**Files:**
- No source changes.

**Steps:**
1. Run WSL focused tests and compile checks.
2. Commit and push.
3. Pull on SeeTaCloud.
4. Run 5 trials with `franka_robosuite_cube_stack_tool_state_first.yaml`, `deepseek-v4-flash`, `num_workers=1`, `record_video=False`.
5. Report success rate, average reward, per-trial tool calls, warnings, and failures.
