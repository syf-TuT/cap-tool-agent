# Standard LIBERO-Object Capsule LLM-Step Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a multimodal, non-privileged Capsule `llm_step` path for standard `libero_object`, with sparse-reward semantics, enforceable phase boundaries, ground-truth isolation, and task-0-to-all-task batch support.

**Architecture:** Keep the existing generic Capsule loop and add opt-in capabilities around it: a source-contract analyzer, public/diagnostic state views, multimodal prompt assembly with sanitized artifacts, and sparse-terminal finish semantics. LIBERO enables those capabilities through a new task-0 YAML using `FrankaLiberoApi`; existing Capsule configurations retain their defaults.

**Tech Stack:** Python 3.10-3.12, AST analysis, Pillow, YAML, pytest, Ruff, existing CaP-X Capsule runtime, WSL2 Ubuntu 22.04 prepared environment.

---

## Execution Rules

- Use @test-driven-development for every behavior change.
- Use @systematic-debugging for unexpected failures.
- Use @run-capx-webui-experiment for WSL sync and pytest commands.
- Use @simplify after focused tests pass.
- Use @verification-before-completion before claiming completion.
- Edit only the Windows checkout at `F:\code\cap-x`.
- Run Python, pytest, and Ruff only in `/home/capx/code/cap-x` inside WSL.
- Sync only files touched by the current task from `/mnt/f/code/cap-x` to WSL.
- Do not start LIBERO, MuJoCo, SAM3, GraspNet, PyRoKi, or an LLM server locally.
- Keep commits scoped and do not amend design commit `ed20080`.

Use this PowerShell-to-WSL command shape:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; <command>'
```

## Task 1: Load Capsule Capability Configuration

**Files:**

- Modify: `capx/utils/launch_utils.py:140-205`
- Modify: `tests/test_runtime_control_config.py`

**Step 1: Write failing tests**

Add a small `_args_for_config(path)` helper if it removes the repeated test
`SimpleNamespace` safely, then add:

```python
def test_load_config_reads_libero_capsule_capabilities(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
env:
  _target_: tests.fake.Env
agent_mode: capsule
capsule_progress_mode: sparse_terminal
capsule_require_task_success_for_finish: true
capsule_validate_program_contract: true
capsule_action_visual_feedback: true
capsule_prompt_state_level: proprioceptive
capsule_diagnostic_state_level: full
"""
    )
    _, config, _ = _load_config(_args_for_config(config_path))
    assert config["capsule_progress_mode"] == "sparse_terminal"
    assert config["capsule_require_task_success_for_finish"] is True
    assert config["capsule_validate_program_contract"] is True
    assert config["capsule_action_visual_feedback"] is True
    assert config["capsule_prompt_state_level"] == "proprioceptive"
    assert config["capsule_diagnostic_state_level"] == "full"


def test_load_config_defaults_preserve_existing_capsule_behavior(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("env:\n  _target_: tests.fake.Env\nagent_mode: capsule\n")
    _, config, _ = _load_config(_args_for_config(config_path))
    assert config["capsule_progress_mode"] == "dense"
    assert config["capsule_require_task_success_for_finish"] is False
    assert config["capsule_validate_program_contract"] is False
    assert config["capsule_action_visual_feedback"] is False
    assert config["capsule_prompt_state_level"] == "full"
    assert config["capsule_diagnostic_state_level"] == "none"
```

**Step 2: Sync and run red tests**

```bash
cp /mnt/f/code/cap-x/tests/test_runtime_control_config.py tests/test_runtime_control_config.py
uv run --no-sync pytest tests/test_runtime_control_config.py::test_load_config_reads_libero_capsule_capabilities tests/test_runtime_control_config.py::test_load_config_defaults_preserve_existing_capsule_behavior -q
```

Expected: FAIL with missing keys.

**Step 3: Implement minimal config loading**

Add to `_load_config`:

```python
"capsule_progress_mode": configs_dict.get("capsule_progress_mode", "dense"),
"capsule_require_task_success_for_finish": configs_dict.get(
    "capsule_require_task_success_for_finish", False
),
"capsule_validate_program_contract": configs_dict.get(
    "capsule_validate_program_contract", False
),
"capsule_action_visual_feedback": configs_dict.get(
    "capsule_action_visual_feedback", False
),
"capsule_prompt_state_level": configs_dict.get("capsule_prompt_state_level", "full"),
"capsule_diagnostic_state_level": configs_dict.get(
    "capsule_diagnostic_state_level", "none"
),
```

Do not add CLI flags; these fields define the experiment protocol in YAML.

**Step 4: Sync and run all config tests**

```bash
cp /mnt/f/code/cap-x/capx/utils/launch_utils.py capx/utils/launch_utils.py
uv run --no-sync pytest tests/test_runtime_control_config.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/utils/launch_utils.py tests/test_runtime_control_config.py
git commit -m "Load LIBERO Capsule capability settings"
```

## Task 2: Implement the Capsule-Ready Contract Analyzer

**Files:**

- Create: `capx/runtime_control/contract.py`
- Create: `tests/test_runtime_control_contract.py`
- Modify: `capx/runtime_control/__init__.py`

**Step 1: Write failing analyzer tests**

Build real regions and groups with:

```python
SIDE_EFFECTS = {"goto_pose", "open_gripper", "close_gripper"}


def _analyze(source: str):
    regions = segment_python_code(source)
    groups = segment_python_code_groups(
        source, regions, side_effect_calls=SIDE_EFFECTS
    )
    return analyze_capsule_program_contract(
        source, regions, groups, side_effect_calls=SIDE_EFFECTS
    )
```

Required tests:

```python
def test_contract_allows_pure_helper_and_single_top_level_effect_groups():
    violations = _analyze(
        "def offset(p):\n"
        "    return p + 0.1\n"
        "target = offset(0.2)\n"
        "goto_pose(target, [1, 0, 0, 0])\n"
        "verified = True\n"
        "open_gripper()\n"
    )
    assert violations == []


def test_contract_rejects_direct_and_transitive_effectful_helpers():
    violations = _analyze(
        "def inner():\n"
        "    close_gripper()\n"
        "def outer():\n"
        "    inner()\n"
        "outer()\n"
    )
    helpers = {
        item.helper_name for item in violations if item.code == "effectful_helper"
    }
    assert helpers == {"inner", "outer"}


def test_contract_rejects_effectful_loop_and_try():
    violations = _analyze(
        "for _ in range(2):\n"
        "    open_gripper()\n"
        "try:\n"
        "    close_gripper()\n"
        "except RuntimeError:\n"
        "    pass\n"
    )
    assert "effectful_control_flow" in {item.code for item in violations}


def test_contract_counts_repeated_effect_calls_in_one_group():
    violations = _analyze(
        "goto_pose([0, 0, 0], [1, 0, 0, 0])\n"
        "goto_pose([0, 0, 1], [1, 0, 0, 0])\n"
    )
    multi = [item for item in violations if item.code == "multiple_effects_in_group"]
    assert len(multi) == 1
    assert multi[0].side_effect_calls == ("goto_pose", "goto_pose")
```

Also verify `to_dict()` contains `code`, `message`, `source_span`, `region_ids`,
`group_ids`, `side_effect_calls`, and `helper_name`.

**Step 2: Sync and run red tests**

```bash
cp /mnt/f/code/cap-x/tests/test_runtime_control_contract.py tests/test_runtime_control_contract.py
uv run --no-sync pytest tests/test_runtime_control_contract.py -q
```

Expected: collection FAIL because the module does not exist.

**Step 3: Implement the analyzer**

Create:

```python
@dataclass(frozen=True)
class ProgramContractViolation:
    code: str
    message: str
    start_line: int
    end_line: int
    region_ids: tuple[str, ...] = ()
    group_ids: tuple[str, ...] = ()
    side_effect_calls: tuple[str, ...] = ()
    helper_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "source_span": {"start_line": self.start_line, "end_line": self.end_line},
            "region_ids": list(self.region_ids),
            "group_ids": list(self.group_ids),
            "side_effect_calls": list(self.side_effect_calls),
            "helper_name": self.helper_name,
        }
```

Public analyzer:

```python
def analyze_capsule_program_contract(
    source: str,
    regions: list[CodeRegion],
    groups: list[CodeRegionGroup],
    *,
    side_effect_calls: set[str],
) -> list[ProgramContractViolation]:
    ...
```

Implementation requirements:

1. Parse the module once.
2. Collect direct effect calls and helper-to-helper calls for top-level functions while
   skipping nested definitions during body traversal.
3. Resolve transitive helper effects with cycle protection.
4. Emit one `effectful_helper` item per resolved effectful helper.
5. Inspect `For`, `AsyncFor`, `While`, `Try`, and Python 3.11+ `TryStar` nodes and emit
   `effectful_control_flow` when their executable body can call an effect.
6. Parse each group and count call occurrences, not unique names; emit
   `multiple_effects_in_group` when count is greater than one.
7. Bind each source span to overlapping region/group IDs.
8. Deduplicate exact items and sort by `(start_line, end_line, code, helper_name or "")`.

Export the dataclass and analyzer from `capx/runtime_control/__init__.py`.

**Step 4: Sync and run green tests**

```bash
cp /mnt/f/code/cap-x/capx/runtime_control/contract.py capx/runtime_control/contract.py
cp /mnt/f/code/cap-x/capx/runtime_control/__init__.py capx/runtime_control/__init__.py
uv run --no-sync pytest tests/test_runtime_control_contract.py tests/test_runtime_control_segmenter.py tests/test_runtime_control_normalizer.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/runtime_control/contract.py capx/runtime_control/__init__.py tests/test_runtime_control_contract.py
git commit -m "Validate Capsule-ready program structure"
```

## Task 3: Put Contract Violations in the Prompt and Guard Effects

**Files:**

- Modify: `capx/runtime_control/prompts.py:31-275`
- Modify: `capx/envs/trial.py:1333-1685`
- Modify: `tests/test_runtime_control_prompts.py`
- Modify: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write failing prompt and loop tests**

Prompt test: call `build_capsule_prompt(..., contract_violations=[...])` and assert the
text contains the violation code/message, source lines, group IDs, and an instruction to
patch before running robot effects.

Loop test: start with an effectful helper and scripted actions that first try to run it,
then patch it into a compliant top-level phase, then run the repaired group. Assert:

- the first run is invalid and moves nothing;
- event evidence contains serialized contract violations;
- patch succeeds and triggers reanalysis;
- the compliant side effect executes;
- `capsule_validate_program_contract: false` retains old behavior.

**Step 2: Sync and run red tests**

```bash
cp /mnt/f/code/cap-x/tests/test_runtime_control_prompts.py tests/test_runtime_control_prompts.py
cp /mnt/f/code/cap-x/tests/test_runtime_control_trial_loop.py tests/test_runtime_control_trial_loop.py
uv run --no-sync pytest tests/test_runtime_control_prompts.py -k contract_violations tests/test_runtime_control_trial_loop.py -k program_contract -q
```

Expected: FAIL because the prompt parameter and guard are missing.

**Step 3: Extend the prompt**

Add:

```python
contract_violations: list[dict[str, Any]] | None = None,
```

to `build_capsule_prompt`. Render violations in a compact JSON section titled
`Capsule-ready program contract violations`. Preserve this section in the most compact
prompt fallback; it is safety context, not optional history.

**Step 4: Add the guard**

Implement:

```python
def _program_contract_guard_event(
    action: RuntimeAction,
    violations: list[ProgramContractViolation],
    region_by_id: dict[str, CodeRegion],
    group_by_id: dict[str, CodeRegionGroup],
    side_effect_calls: set[str],
) -> RuntimeEvent | None:
    if not violations:
        return None
    if not _runtime_action_targets_side_effect_unit(
        action, region_by_id, group_by_id, side_effect_calls
    ):
        return None
    return RuntimeEvent(
        action=action.action,
        status="invalid",
        region_id=str(
            action.args.get("group_id") or action.args.get("region_id") or ""
        ) or None,
        message=(
            "Robot side-effect execution is blocked until the Capsule-ready "
            "program contract is repaired."
        ),
        evidence={
            "program_contract_violations": [item.to_dict() for item in violations]
        },
    )
```

Do not block patch, inspect, finish, append source, or pure groups.

**Step 5: Integrate and revalidate**

- Analyze after initial segmentation when the config flag is true.
- Pass serialized violations into the Action LLM prompt.
- Apply this guard before no-replay and reward guards.
- Recompute after every successful patch or append that rebuilds source/groups.
- Add `program_contract_valid`, count, and violation codes to step metrics.
- Keep `auto_forward` unchanged.

**Step 6: Sync and run green tests**

```bash
cp /mnt/f/code/cap-x/capx/runtime_control/prompts.py capx/runtime_control/prompts.py
cp /mnt/f/code/cap-x/capx/envs/trial.py capx/envs/trial.py
uv run --no-sync pytest tests/test_runtime_control_prompts.py tests/test_runtime_control_trial_loop.py -q
```

Expected: PASS.

**Step 7: Commit**

```bash
git add capx/runtime_control/prompts.py capx/envs/trial.py tests/test_runtime_control_prompts.py tests/test_runtime_control_trial_loop.py
git commit -m "Guard Capsule execution with program contract"
```

## Task 4: Add Sparse-Terminal Feedback, Finish Guard, and Budget Status

**Files:**

- Modify: `capx/runtime_control/feedback.py`
- Modify: `capx/envs/trial.py:1588-1717`
- Modify: `tests/test_runtime_control_feedback.py`
- Modify: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write failing feedback tests**

Add a sparse-terminal test for a successful effect with reward `0 -> 0`:

```python
feedback = build_runtime_feedback(
    step_id=1,
    action=RuntimeAction("run_group", {"group_id": "group_1"}),
    event=RuntimeEvent(action="run_group", status="success", region_id="group_1"),
    region=_effect_group(),
    trace_events=[{"name": "goto_pose", "status": "success"}],
    before_state={"reward": 0.0, "task_completed": False},
    after_state={"reward": 0.0, "task_completed": False},
    progress_mode="sparse_terminal",
)
assert feedback.status == "success"
assert feedback.evidence["terminal_progress_unverified"] is True
assert "no local task progress" not in feedback.message
```

Add a dense-default regression that still expects the existing warning.

**Step 2: Write failing finish and budget tests**

Cover:

1. Guarded premature `finish` returns a warning, does not increment `num_finishes`, and
   does not stop the next scripted action.
2. A later action that makes the environment successful stops without consuming another
   model/script action.
3. Exhausting all steps without success gives `summary.truncated is True`, marks the last
   metric `budget_exhausted: true`, and logs the flag.
4. Existing unguarded finish behavior remains unchanged.

**Step 3: Run red tests**

```bash
cp /mnt/f/code/cap-x/tests/test_runtime_control_feedback.py tests/test_runtime_control_feedback.py
cp /mnt/f/code/cap-x/tests/test_runtime_control_trial_loop.py tests/test_runtime_control_trial_loop.py
uv run --no-sync pytest tests/test_runtime_control_feedback.py -k sparse_terminal tests/test_runtime_control_trial_loop.py -k 'premature_finish or budget_exhausted or stops_immediately_on_success' -q
```

Expected: FAIL.

**Step 4: Implement feedback mode**

Add `progress_mode: str = "dense"` to `build_runtime_feedback`. Validate `dense` and
`sparse_terminal`. In sparse mode, successful nonterminal effect execution remains
`success`; add `progress_mode` and `terminal_progress_unverified=True` to evidence and
use a neutral message. Other call sites keep the dense default.

**Step 5: Implement the finish guard**

```python
def _finish_success_guard_event(
    action: RuntimeAction,
    state: dict[str, Any],
    *,
    require_task_success: bool,
) -> RuntimeEvent | None:
    if action.action != "finish" or not require_task_success:
        return None
    reward = _state_reward(state)
    if state.get("task_completed") is True or (reward is not None and reward >= 1.0):
        return None
    return RuntimeEvent(
        action="finish",
        status="warning",
        message=(
            "Finish rejected because the environment success predicate is not satisfied."
        ),
        evidence={"task_completed": state.get("task_completed"), "reward": reward},
    )
```

Apply it before other guards. Break on finish only when its event is successful. When the
guard option is enabled, stop immediately after any action whose after-state is successful.

**Step 6: Track budget exhaustion**

Track exit reason. If the loop consumes its range without accepted finish or environment
success, set `budget_exhausted=True`, `TrialSummary.truncated=True`, mark the last metric,
and include it in the log. Do not set `failed=True` solely for budget exhaustion.

**Step 7: Sync and run green regressions**

```bash
cp /mnt/f/code/cap-x/capx/runtime_control/feedback.py capx/runtime_control/feedback.py
cp /mnt/f/code/cap-x/capx/envs/trial.py capx/envs/trial.py
uv run --no-sync pytest tests/test_runtime_control_feedback.py tests/test_runtime_control_trial_loop.py -q
```

Expected: PASS.

**Step 8: Commit**

```bash
git add capx/runtime_control/feedback.py capx/envs/trial.py tests/test_runtime_control_feedback.py tests/test_runtime_control_trial_loop.py
git commit -m "Support sparse-terminal Capsule progress"
```

## Task 5: Separate Prompt State from Diagnostic State

**Files:**

- Modify: `capx/envs/trial.py:2427-2595`
- Modify: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write failing state tests**

Create a fake LIBERO-like low-level environment with robot fields and:

```python
def _get_all_object_poses(self):
    return {
        "alphabet_soup": (
            np.array([0.2, 0.1, 0.05]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
    }
```

Assert:

- `_capsule_state_snapshot(env, state_level="proprioceptive")` has robot state and no
  `object_poses`;
- `state_level="full"` contains normalized `pos` and `quat_wxyz`;
- invalid levels raise `ValueError`;
- public and diagnostic nested values are independent;
- LLM-step trace/prompt artifacts contain no `alphabet_soup` truth pose when public level
  is proprioceptive;
- `capsule_diagnostics_trial_XX.jsonl` does contain it when diagnostic level is full.

**Step 2: Run red tests**

```bash
cp /mnt/f/code/cap-x/tests/test_runtime_control_trial_loop.py tests/test_runtime_control_trial_loop.py
uv run --no-sync pytest tests/test_runtime_control_trial_loop.py -k 'state_level or diagnostic_state_artifact' -q
```

Expected: FAIL.

**Step 3: Implement state levels**

Change the helper to:

```python
def _capsule_state_snapshot(
    env: CodeExecutionEnvBase,
    *,
    state_level: str = "full",
) -> dict[str, Any]:
```

Always collect reward, completion, counters, gripper, joints, and end-effector pose. Only
`full` collects object poses. Extend `_capsule_object_pose_snapshot` to prefer callable
`low_level_env._get_all_object_poses()` and normalize each value as:

```python
{"pos": [...], "quat_wxyz": [...]}
```

Retain the existing Robosuite fallback.

**Step 4: Wire public and diagnostic views**

- Read prompt level with default `full` and diagnostic level with default `none`.
- Build history and feedback only from public snapshots.
- Independently capture diagnostic before/after states when diagnostic level is not
  `none`.
- Write `{step_id, state_before, state_after}` rows to
  `capsule_diagnostics_trial_XX.jsonl`.
- Never place diagnostic dictionaries in prompts, history, feedback, or side-effect
  ledgers.

**Step 5: Run green regressions**

```bash
cp /mnt/f/code/cap-x/capx/envs/trial.py capx/envs/trial.py
uv run --no-sync pytest tests/test_runtime_control_trial_loop.py -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add capx/envs/trial.py tests/test_runtime_control_trial_loop.py
git commit -m "Separate Capsule prompt and diagnostic state"
```

## Task 6: Add Multimodal Prompt and Sanitized Artifact Helpers

**Files:**

- Modify: `capx/envs/trial.py:225-315,1333-1717`
- Modify: `capx/utils/launch_utils.py:365-420`
- Modify: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write failing visual helper tests**

Use tiny RGB arrays and test:

1. `_capture_capsule_visuals(env, use_wrist_camera=True)` returns main and wrist records
   containing camera name, data URL, PIL image, dimensions, and SHA-256.
2. Missing wrist rendering returns the current main image plus an explicit wrist capture
   error; no old wrist image is reused.
3. `_attach_capsule_visuals(prompt, records, errors)` adds labeled image entries and any
   error notice.
4. `_sanitize_multimodal_prompt(prompt, artifact_by_sha256)` contains no data URL/base64
   and includes camera, path, dimensions, and hash.
5. Sanitization does not mutate the live prompt.

Add an initial-query test proving `initial_prompt.txt` stores a hash reference instead of
an image payload when `_query_initial_code` receives a multimodal prompt.

**Step 2: Run red tests**

```bash
cp /mnt/f/code/cap-x/tests/test_runtime_control_trial_loop.py tests/test_runtime_control_trial_loop.py
uv run --no-sync pytest tests/test_runtime_control_trial_loop.py -k 'capsule_visual or sanitizes_initial_prompt' -q
```

Expected: FAIL because the helpers do not exist and initial prompt persistence is raw.

**Step 3: Implement visual capture**

Wrap existing `_get_visual_feedback(env, use_wrist_camera=...)`, normalizing its scalar
and list return shapes. Use:

```python
@dataclass(frozen=True)
class CapsuleVisual:
    camera: str
    data_url: str
    image: Image.Image
    width: int
    height: int
    sha256: str

    def metadata(self) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "width": self.width,
            "height": self.height,
            "sha256": self.sha256,
        }
```

Compute the hash from decoded PNG bytes. Never carry a prior record through a failed
current capture.

**Step 4: Implement attachment and sanitization**

- Prompt attachment modifies only the newly built per-step prompt.
- Add `Current main-camera view` and `Current wrist-camera view` labels.
- Sanitization deep-copies and replaces every image item with `image_reference` metadata.
- Accept optional `sha256 -> relative path` artifact mapping.
- Unknown image URLs keep media type/hash only, never raw content.
- Make `_query_initial_code` persist the sanitized prompt while preserving return values
  and callers.

**Step 5: Save visual files**

Write, when `output_dir` exists:

```text
capsule_visuals_trial_XX/step_YY_main.png
capsule_visuals_trial_XX/step_YY_wrist.png
```

Return the mapping used by sanitization. Surface save failures in metadata instead of
silently dropping them.

**Step 6: Run green tests**

```bash
cp /mnt/f/code/cap-x/capx/envs/trial.py capx/envs/trial.py
cp /mnt/f/code/cap-x/capx/utils/launch_utils.py capx/utils/launch_utils.py
uv run --no-sync pytest tests/test_runtime_control_trial_loop.py -q
```

Expected: PASS.

**Step 7: Commit**

```bash
git add capx/envs/trial.py capx/utils/launch_utils.py tests/test_runtime_control_trial_loop.py
git commit -m "Add multimodal Capsule prompt artifacts"
```

## Task 7: Integrate Goal Injection and Per-Step Visual Feedback

**Files:**

- Modify: `capx/envs/trial.py:1333-1717`
- Modify: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write failing goal initialization test**

Create a dummy low-level environment with
`handle.task_language = "Pick the alphabet soup and place it in the basket"`; reset
returns a prompt containing `{libero_environment_goal}`. Mock `_query_initial_code` and
assert its prompt contains the task language and no placeholder.

Assert the loop deep-copies `obs["full_prompt"]` before mutation so repeated resets do not
accumulate image items.

**Step 2: Write failing per-step visual tests**

Mock capture with step-0 `initial-main/initial-wrist` images and next-step
`after-main/after-wrist` images. Mock model responses and assert:

- initial generation sees both initial images;
- Action step 1 sees the initial pair;
- Action step 2 sees only the fresh after-action pair;
- data URLs do not contribute to `action_prompt_chars`;
- metrics contain count, dimensions, hashes, and capture errors;
- disabling `capsule_action_visual_feedback` keeps Action prompts text-only.

Add an artifact integration test proving:

- model requests contain real `image_url` data URLs;
- `capsule_prompts_trial_XX.json` contains references and paths;
- no saved prompt contains `data:image`;
- expected PNGs exist.

**Step 3: Run red tests**

```bash
cp /mnt/f/code/cap-x/tests/test_runtime_control_trial_loop.py tests/test_runtime_control_trial_loop.py
uv run --no-sync pytest tests/test_runtime_control_trial_loop.py -k 'libero_goal_before_initial or llm_step_visual_feedback or sanitized_visual_artifact' -q
```

Expected: FAIL.

**Step 4: Integrate trial initialization**

Immediately after reset:

```python
obs["full_prompt"] = copy.deepcopy(obs["full_prompt"])
_patch_libero_goal(env, obs)
```

Capture step-0 visuals once when either initial or Action visual feedback is enabled.
Attach them to initial generation only when `use_visual_feedback` is true. Retain them as
the current state for the first Action decision.

**Step 5: Integrate current images into Action prompts**

- Build and measure the text-only prompt first.
- Attach images after `action_prompt_chars` and budget overflow are computed.
- Send the multimodal live prompt to `_query_model`.
- Save only a sanitized prompt copy.
- Freshly capture after every executed, inspected, patched, rejected, or invalid action.
- If a camera fails, clear that camera and present the explicit error next time.
- Forced appended-recovery actions do not query the Action LLM but refresh images after
  execution.

Add metrics:

```python
metric["action_prompt_image_count"] = len(current_visuals)
metric["action_prompt_images"] = [item.metadata() for item in current_visuals]
metric["visual_capture_errors"] = list(current_visual_errors)
```

No metric may contain a data URL.

**Step 6: Run green regressions**

```bash
cp /mnt/f/code/cap-x/capx/envs/trial.py capx/envs/trial.py
uv run --no-sync pytest tests/test_runtime_control_trial_loop.py -q
```

Expected: PASS.

**Step 7: Commit**

```bash
git add capx/envs/trial.py tests/test_runtime_control_trial_loop.py
git commit -m "Use current images for Capsule llm-step"
```

## Task 8: Add the Standard LIBERO-Object Task-0 Configuration

**Files:**

- Create: `env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml`
- Modify: `tests/test_runtime_control_config.py`

**Step 1: Write the failing YAML test**

Load the new YAML and assert:

```python
cfg = data["env"]["cfg"]
low_level = cfg["low_level"]
assert low_level["suite_name"] == "libero_object"
assert low_level["task_id"] == 0
assert low_level["privileged"] is False
assert cfg["privileged"] is False
assert cfg["apis"] == ["FrankaLiberoApi"]
assert data["agent_mode"] == "capsule"
assert data["capsule_control_mode"] == "llm_step"
assert data["max_capsule_steps"] == 24
assert data["capsule_progress_mode"] == "sparse_terminal"
assert data["capsule_require_task_success_for_finish"] is True
assert data["capsule_validate_program_contract"] is True
assert data["capsule_action_visual_feedback"] is True
assert data["capsule_prompt_state_level"] == "proprioceptive"
assert data["capsule_diagnostic_state_level"] == "full"
assert data["use_visual_feedback"] is True
assert data["use_wrist_camera"] is True
assert data["use_parallel_ensemble"] is False
assert data["trials"] == 1
assert data["num_workers"] == 1
```

Also assert the initial prompt requires a complete program, pure-only helpers, no
effectful loops/try blocks, and top-level one-effect phases.

**Step 2: Run red test**

```bash
cp /mnt/f/code/cap-x/tests/test_runtime_control_config.py tests/test_runtime_control_config.py
uv run --no-sync pytest tests/test_runtime_control_config.py -k libero_object_capsule -q
```

Expected: FAIL because the YAML is missing.

**Step 3: Create the YAML**

Start from `env_configs/libero/franka_libero_spatial_0.yaml`, switch to standard
`libero_object` task 0 and `FrankaLiberoApi`, and add the approved Capsule fields. Retain
the PyRoKi, Contact-GraspNet, and SAM3 server definitions.

The prompt must include:

```text
Generate one complete executable Python program for the goal.
Pure computation, geometry, and perception helpers are allowed.
Do not place robot motion or gripper calls inside helper functions, loops, or try blocks.
Write robot side effects as top-level phases with at most one side-effect API call per
phase, separated by calculations or fresh observations needed by the next phase.
```

Use one trial, one worker, no ensemble, `record_video: true`, and a dedicated output path.

**Step 4: Sync and run all config tests**

```bash
cp /mnt/f/code/cap-x/env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml
uv run --no-sync pytest tests/test_runtime_control_config.py -q
```

Expected: PASS without instantiating LIBERO.

**Step 5: Commit**

```bash
git add env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml tests/test_runtime_control_config.py
git commit -m "Configure LIBERO-Object Capsule llm-step"
```

## Task 9: Add Dependency-Free Batch Task Filtering

**Files:**

- Modify: `capx/envs/scripts/run_libero_batch.py`
- Create: `tests/test_run_libero_batch.py`

**Step 1: Write failing pure tests**

The test module must import the batch module without LIBERO installed. Cover:

```python
def test_select_task_ids_defaults_to_all_tasks():
    assert _select_task_ids(4, None) == [0, 1, 2, 3]


def test_select_task_ids_preserves_order_and_removes_duplicates():
    assert _select_task_ids(10, [7, 0, 7]) == [7, 0]


@pytest.mark.parametrize("task_ids", [[-1], [10], [0, 11]])
def test_select_task_ids_rejects_out_of_range_values(task_ids):
    with pytest.raises(ValueError, match="task ID"):
        _select_task_ids(10, task_ids)
```

Add a fake benchmark suite test: task ID 0 yields one collected task; `None` yields all.
The fake suite supplies `n_tasks`, `get_task`, and `get_task_init_states` without Torch or
LIBERO.

**Step 2: Run red tests**

```bash
cp /mnt/f/code/cap-x/tests/test_run_libero_batch.py tests/test_run_libero_batch.py
uv run --no-sync pytest tests/test_run_libero_batch.py -q
```

Expected: collection FAIL because the module exits when LIBERO is absent or helpers do
not exist.

**Step 3: Defer optional import**

Remove top-level import/exit and add:

```python
def _load_benchmark_dict():
    try:
        from libero import benchmark
    except ImportError as exc:
        raise RuntimeError(
            "LIBERO is not installed; run this command in the dedicated LIBERO environment."
        ) from exc
    return benchmark.get_benchmark_dict()
```

Call it from `main`.

**Step 4: Implement filtering**

- Add `task_ids: list[int] | None = None` to `LiberoBatchLaunchArgs`.
- Implement `_select_task_ids` exactly as tested.
- Extract task collection into a helper that accepts an already loaded benchmark dict.
- Validate IDs before making output directories or launching experiments.
- Retain the 50-state fallback.
- Pass `use_wrist_camera` through when constructing `LaunchArgs`.
- Do not override YAML Capsule protocol fields.

**Step 5: Run green tests**

```bash
cp /mnt/f/code/cap-x/capx/envs/scripts/run_libero_batch.py capx/envs/scripts/run_libero_batch.py
uv run --no-sync pytest tests/test_run_libero_batch.py tests/integrations/test_libero_integration.py -q
```

Expected: PASS without a real LIBERO installation.

**Step 6: Commit**

```bash
git add capx/envs/scripts/run_libero_batch.py tests/test_run_libero_batch.py
git commit -m "Filter LIBERO batch tasks by ID"
```

## Task 10: Integrated Verification, Simplification, and Server Documentation

**Files:**

- Modify as needed: files from Tasks 1-9
- Modify: `docs/libero-tasks.md`

**Step 1: Sync all touched files explicitly**

Copy these Windows paths to matching WSL paths:

```text
capx/runtime_control/contract.py
capx/runtime_control/__init__.py
capx/runtime_control/prompts.py
capx/runtime_control/feedback.py
capx/envs/trial.py
capx/utils/launch_utils.py
capx/envs/scripts/run_libero_batch.py
env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml
tests/test_runtime_control_contract.py
tests/test_runtime_control_prompts.py
tests/test_runtime_control_feedback.py
tests/test_runtime_control_config.py
tests/test_runtime_control_trial_loop.py
tests/test_run_libero_batch.py
```

Do not sync `.git`, `.venv`, outputs, or the whole checkout.

**Step 2: Run the focused regression suite**

```bash
uv run --no-sync pytest \
  tests/test_runtime_control_contract.py \
  tests/test_runtime_control_prompts.py \
  tests/test_runtime_control_feedback.py \
  tests/test_runtime_control_config.py \
  tests/test_runtime_control_trial_loop.py \
  tests/test_runtime_control_side_effects.py \
  tests/test_runtime_control_segmenter.py \
  tests/test_runtime_control_normalizer.py \
  tests/test_run_libero_batch.py \
  tests/integrations/test_libero_integration.py \
  -q
```

Expected: zero failures.

**Step 3: Run Ruff**

```bash
uv run --no-sync ruff check \
  capx/runtime_control/contract.py \
  capx/runtime_control/prompts.py \
  capx/runtime_control/feedback.py \
  capx/envs/trial.py \
  capx/utils/launch_utils.py \
  capx/envs/scripts/run_libero_batch.py \
  tests/test_runtime_control_contract.py \
  tests/test_runtime_control_prompts.py \
  tests/test_runtime_control_feedback.py \
  tests/test_runtime_control_config.py \
  tests/test_runtime_control_trial_loop.py \
  tests/test_run_libero_batch.py
```

Expected: exit 0.

**Step 4: Simplify and reverify**

Apply @simplify. Remove duplicated reanalysis, visual metadata construction, and state
branching through narrow helpers. Do not combine public/diagnostic state objects or add
new configuration modes. Re-run Steps 2 and 3 unchanged.

**Step 5: Document server-only commands**

Add a Capsule LLM-step section to `docs/libero-tasks.md` and label these as server/runtime
validation, not local Windows commands.

Task-0 smoke:

```bash
source .venv-libero/bin/activate
MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml \
  --total-trials 1 \
  --num-workers 1
```

Task 0 over five states:

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

All 10 tasks over five states:

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

Do not execute these locally.

**Step 6: Run documentation and leak checks**

```powershell
rg -n "libero_object|capsule_action_visual_feedback|task-ids" docs/libero-tasks.md env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml
rg -n "data:image|base64," docs/plans/2026-08-13-libero-object-capsule-llm-step.md env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml
git diff --check
```

Expected: protocol references are present, no embedded payload exists, and diff check is
clean.

**Step 7: Commit documentation and final cleanup**

If simplification changed implementation after prior commits, commit it separately.
Then:

```bash
git add docs/libero-tasks.md
git commit -m "Document LIBERO-Object Capsule validation"
```

**Step 8: Final verification record**

Using @verification-before-completion, report:

- exact pytest command and pass count;
- exact Ruff command and exit status;
- `git status --short --branch`;
- commits created;
- explicit note that simulator/server experiments were not run locally.

Local implementation is complete only when the focused suite and Ruff pass and artifact
tests find neither embedded base64 nor simulator truth in model prompts.
