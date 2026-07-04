# Capsule Semantic Groups Design

## Goal

Make Capsule runtime-control execute and repair semantically meaningful code chunks instead
of single top-level Python statements, so `max_capsule_steps` measures task-level progress
rather than source-line traversal.

## Problem

The current Capsule segmenter splits generated Python with one `CodeRegion` per top-level
AST statement. For robot programs this often means a single assignment, `solve_ik`, or
`move_to_joints` call consumes one Capsule action. In the strict cube-stack pilot,
Capsule exhausted 30 actions before reaching green-cube placement in some seeds.

The current feedback also marks every successful region with no immediate reward gain as
`warning`. That is too noisy for perception, planning, and pre-grasp setup steps where no
reward change is expected. The warning encourages unnecessary local patches and contributes
to undefined-variable failures.

## Recommended Architecture

Keep `CodeRegion` as the atomic source representation, then add `CodeRegionGroup` as the
execution unit presented to the model. A group combines adjacent atomic regions until the
source reaches a meaningful boundary such as a robot side-effect call, a perception or
planning phase boundary, or a maximum group size.

Capsule prompts should present groups first and actions should prefer `run_group` and
`patch_group`. `run_region` and `patch_region` remain available for compatibility and
debugging, but benchmark configs should use semantic groups by default.

## Grouping Rules

The initial grouping should be deterministic and heuristic, with no task-specific hardcoded
cube-stack state machine:

- Parse top-level Python statements into atomic `CodeRegion` objects.
- Extract defined names, used names, primitive calls, and whether a statement has robot
  side effects.
- Merge adjacent statements into a group until one of these boundaries is reached:
  - a robot side-effect primitive such as `move_to_joints`, `move_to_pose`,
    `close_gripper`, or `open_gripper`;
  - a group already contains a side-effect statement;
  - a group reaches a conservative maximum number of statements;
  - the next statement starts a new major perception or planning phase.
- Preserve source order and stable group ids.

This creates groups like observe/segment, grasp planning, move-and-grasp, lift, target
perception, place, and release without embedding task-specific object names.

## Runtime Actions

Add two actions to the schema:

- `run_group`: executes a complete `CodeRegionGroup`.
- `patch_group`: replaces the source span of a complete `CodeRegionGroup`.

`patch_group` is preferred over `patch_region` because most NameError failures are caused by
patching a single statement without the variables that define its context.

## Feedback

Feedback should distinguish expected no-progress steps from suspicious no-progress steps.

- Pure computation, perception, and planning groups can succeed without reward gain.
- Robot side-effect groups that do not change reward can produce a warning.
- Failed groups should include missing variable names when Python raises `NameError`.
- Patch hints should recommend patching a group by default and only patching an atomic
  region when the source span is self-contained.

## Evaluation Impact

After this change, `max_capsule_steps=15-20` should represent 15-20 semantic runtime-control
decisions. The baseline can keep `max_regenerations=5`. The benchmark should report both
semantic groups and atomic regions executed so the comparison remains transparent.

## Validation

Unit tests should cover deterministic grouping, `run_group`, `patch_group`, prompt content,
and feedback severity. A short remote pilot should then rerun cube-stack seeds 1-5 with:

- DeepSeek V4 Flash through Packy API;
- streaming enabled;
- reasoning disabled;
- privileged disabled;
- `FrankaControlApiReduced`;
- Capsule semantic groups with `max_capsule_steps=20`;
- baseline multi-turn regeneration with `max_regenerations=5`.
