# Capsule LLM-Step-Only Runtime Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove Capsule auto-forward execution and make the sole Capsule runtime perform one explicit LLM-selected action followed by one automatic observation, with transactional source edits and stable forward-only side-effect identities.

**Architecture:** Keep one loop in `capx/envs/trial.py`, backed by a small lineage module in `capx/runtime_control/lineage.py`. Model-facing region/group IDs remain temporary, while replay guards and recovery authorization use stable monotonic keys. Patch and append actions prepare and validate a complete candidate state before atomically replacing live source structures.

**Tech Stack:** Python 3.10-3.12, dataclasses, AST-based Capsule segmentation, pytest, Ruff, uv, WSL2 Ubuntu for all Python test execution.

---

## Execution Preconditions

- Use @using-git-worktrees before implementation. Create a dedicated worktree from the
  branch containing design commit `bb54012`.
- Use @test-driven-development for every behavior change below.
- Edit source only in the Windows worktree. Do not install dependencies or run Python in
  the Windows checkout.
- Sync each changed file into `/home/capx/code/cap-x` and run tests in WSL2.
- Use @verification-before-completion before the final commit or any success claim.
- Do not add compatibility execution for `auto_forward`.
- Preserve unrelated untracked model-cache files.

The standard test invocation, run from elevated PowerShell, is:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest <tests> -q'
```

Before each invocation, sync the exact changed files. For example:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'set -e; cd /mnt/f/code/cap-x; cp --parents capx/envs/trial.py tests/test_runtime_control_trial_loop.py /home/capx/code/cap-x; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_trial_loop.py -q'
```

Adapt `/mnt/f/code/cap-x` if the implementation worktree uses a different Windows path.

### Task 1: Make Capsule Dispatch LLM-Step-Only

**Files:**
- Modify: `capx/envs/trial.py:1349-1376`
- Modify: `capx/utils/launch_utils.py:98-195`
- Modify: `env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml:38-46`
- Modify: `tests/test_runtime_control_trial_loop.py:1640-1691`
- Modify: `tests/test_runtime_control_config.py:188-225,338-373`

**Step 1: Write failing dispatch and migration tests**

Replace the default-auto-forward test with a sole-loop dispatch test:

```python
def test_capsule_trial_dispatches_directly_to_capsule_loop(monkeypatch):
    expected = object()

    def fake_loop(**kwargs):
        assert kwargs["trial"] == 3
        return expected

    monkeypatch.setattr(trial_module, "_run_capsule_loop", fake_loop)

    result = _run_capsule_trial(
        env=FakeCapsuleEnv(),
        trial=3,
        args=SimpleNamespace(),
        config={},
        initial_code="x = 1",
    )

    assert result is expected
```

Replace the control-mode config test with:

```python
def test_load_config_rejects_removed_capsule_control_mode(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
env:
  _target_: tests.fake.Env
agent_mode: capsule
capsule_control_mode: auto_forward
"""
    )

    with pytest.raises(ValueError, match="capsule_control_mode has been removed"):
        _load_config(_args_for_config(config_path))
```

Update `test_load_config_reads_capsule_fields` to assert the merged config does not
contain `capsule_control_mode`.

**Step 2: Run the focused tests and verify failure**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_config.py tests/test_runtime_control_trial_loop.py::test_capsule_trial_dispatches_directly_to_capsule_loop -q'
```

Expected: FAIL because `_run_capsule_loop` does not exist and legacy mode is accepted.

**Step 3: Implement the sole dispatch and migration error**

Rename `_run_capsule_llm_step_loop` to `_run_capsule_loop` and make dispatch direct:

```python
def _run_capsule_trial(...):
    return _run_capsule_loop(
        env=env,
        trial=trial,
        args=args,
        config=config,
        initial_code=initial_code,
        scripted_actions=scripted_actions,
    )
```

In `_load_config`, immediately after loading `configs_dict`, add:

```python
if "capsule_control_mode" in configs_dict:
    raise ValueError(
        "capsule_control_mode has been removed. Capsule now always uses strict "
        "per-action LLM control. Remove this configuration field."
    )
```

Delete the merged `capsule_control_mode` field and remove the explicit `llm_step` field
from the LIBERO YAML.

Mechanically rename direct test calls from `_run_capsule_llm_step_loop` to
`_run_capsule_loop`; do not change their assertions in this step.

**Step 4: Run the focused tests and verify pass**

Run the same command as Step 2.

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/envs/trial.py capx/utils/launch_utils.py \
  env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml \
  tests/test_runtime_control_trial_loop.py tests/test_runtime_control_config.py
git commit -m "Remove Capsule control mode selection"
```

### Task 2: Delete Auto-Forward Runtime and Recovery-Only Prompts

**Files:**
- Modify: `capx/envs/trial.py:1399-2100,2806-2843,3243-3428,3637-3667`
- Modify: `capx/runtime_control/prompts.py:578-794`
- Modify: `capx/envs/trial.py:52-76` imports
- Modify: `tests/test_runtime_control_trial_loop.py:1692-2580`
- Modify: `tests/test_runtime_control_prompts.py:1-15,1230-1430`

**Step 1: Write failing removal tests**

Add:

```python
def test_auto_forward_runtime_is_removed():
    assert not hasattr(trial_module, "_run_capsule_auto_forward_loop")
    assert not hasattr(trial_module, "_validate_recovery_action")
    assert not hasattr(trial_module, "_insert_recovery_source_after_line")
```

In prompt tests, assert the package exposes only the normal action prompt builder:

```python
def test_recovery_only_prompt_builders_are_removed():
    assert not hasattr(prompt_module, "build_capsule_recovery_prompt")
    assert not hasattr(prompt_module, "build_capsule_terminal_recovery_prompt")
```

**Step 2: Run tests and verify failure**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_trial_loop.py::test_auto_forward_runtime_is_removed tests/test_runtime_control_prompts.py::test_recovery_only_prompt_builders_are_removed -q'
```

Expected: FAIL because the symbols still exist.

**Step 3: Delete dead implementation**

- Delete `_run_capsule_auto_forward_loop` completely.
- Delete all helpers listed in the test and approved design when `rg` proves they have no
  remaining caller.
- Delete the two recovery-only prompt builders and their imports.
- Simplify `_execute_runtime_action` by removing
  `append_recovery_insert_after_line`; append always targets the end of source.
- Remove only tests named `test_*auto_forward*` and tests solely covering removed recovery
  prompts. Keep shared fake environments and port safety assertions in later tasks.

Run:

```bash
rg -n "auto_forward|build_capsule_recovery_prompt|build_capsule_terminal_recovery_prompt|append_recovery_insert_after_line" capx tests env_configs
```

Expected: no runtime/config references. Historical `docs/plans` references are allowed.

**Step 4: Run the focused removal and prompt tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_prompts.py tests/test_runtime_control_trial_loop.py::test_auto_forward_runtime_is_removed -q'
```

Expected: PASS. The complete loop suite runs after Task 8 replaces pending recovery.

**Step 5: Commit**

```bash
git add capx/envs/trial.py capx/runtime_control/prompts.py \
  tests/test_runtime_control_trial_loop.py tests/test_runtime_control_prompts.py
git commit -m "Delete Capsule auto-forward runtime"
```

### Task 3: Remove `inspect_trace` as a Runtime Action

**Files:**
- Modify: `capx/runtime_control/schema.py:6-27`
- Modify: `capx/runtime_control/prompts.py:76-160,390-450`
- Modify: `capx/envs/trial.py:2846-2920`
- Modify: `tests/test_runtime_control_schema.py`
- Modify: `tests/test_runtime_control_prompts.py`
- Modify: `tests/test_runtime_control_trial_loop.py:5070-5130`

**Step 1: Write failing schema and prompt tests**

```python
def test_inspect_trace_is_not_a_supported_runtime_action():
    with pytest.raises(ValueError, match="Unsupported runtime action"):
        RuntimeAction.from_mapping({"action": "inspect_trace", "args": {}})


def test_capsule_prompt_does_not_offer_inspect_trace():
    text = _prompt_text(build_capsule_prompt(...))
    assert "inspect_trace" not in text
```

**Step 2: Run tests and verify failure**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_schema.py tests/test_runtime_control_prompts.py -q'
```

Expected: FAIL because `inspect_trace` remains supported and documented.

**Step 3: Remove the action**

- Remove `inspect_trace` from `RuntimeActionName` and `SUPPORTED_ACTIONS`.
- Remove it from allowed action lists and prompt examples.
- Delete its branch from `_execute_runtime_action`.
- Delete old selective trace inspection tests.
- Keep `RuntimeTrace.summary()` because prompts and audit artifacts still use bounded
  summaries.

**Step 4: Run schema, prompt, and loop tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_schema.py tests/test_runtime_control_prompts.py tests/test_runtime_control_trial_loop.py -q'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/runtime_control/schema.py capx/runtime_control/prompts.py \
  capx/envs/trial.py tests/test_runtime_control_schema.py \
  tests/test_runtime_control_prompts.py tests/test_runtime_control_trial_loop.py
git commit -m "Remove redundant Capsule trace inspection action"
```

### Task 4: Add Stable Unit Lineage Data Structures

**Files:**
- Create: `capx/runtime_control/lineage.py`
- Create: `tests/test_runtime_control_lineage.py`
- Modify: `capx/runtime_control/__init__.py`

**Step 1: Write failing allocation tests**

```python
def test_initial_lineage_assigns_monotonic_region_and_group_keys():
    regions = [
        CodeRegion("region_1", 1, 1, "x = 1"),
        CodeRegion("region_2", 2, 2, "move_to(x)"),
    ]
    groups = [
        CodeRegionGroup("group_1", 1, 2, "x = 1\nmove_to(x)", ["region_1", "region_2"])
    ]

    lineage = UnitLineage.create(regions, groups)

    assert lineage.region_key_by_id == {
        "region_1": "region_key_000001",
        "region_2": "region_key_000002",
    }
    assert lineage.group_key_by_id == {"group_1": "group_key_000001"}


def test_duplicate_source_units_receive_distinct_keys():
    regions = [
        CodeRegion("region_1", 1, 1, "open_gripper()"),
        CodeRegion("region_2", 2, 2, "open_gripper()"),
    ]
    lineage = UnitLineage.create(regions, [])
    assert lineage.region_key_by_id["region_1"] != lineage.region_key_by_id["region_2"]
```

**Step 2: Run tests and verify import failure**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_lineage.py -q'
```

Expected: FAIL because the module does not exist.

**Step 3: Implement the data structures**

Create:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from capx.runtime_control.schema import CodeRegion, CodeRegionGroup


class LineageAmbiguityError(ValueError):
    pass


@dataclass(frozen=True)
class SourceRevision:
    revision: int
    source_sha256: str
    edit_kind: Literal["initial", "patch_region", "patch_group", "append_recovery"]
    parent_revision: int | None
    old_line_count: int


@dataclass
class UnitLineage:
    next_region_key: int = 1
    next_group_key: int = 1
    region_key_by_id: dict[str, str] = field(default_factory=dict)
    group_key_by_id: dict[str, str] = field(default_factory=dict)
    executed_region_keys: set[str] = field(default_factory=set)
    executed_group_keys: set[str] = field(default_factory=set)

    @classmethod
    def create(
        cls,
        regions: list[CodeRegion],
        groups: list[CodeRegionGroup],
    ) -> "UnitLineage":
        lineage = cls()
        for region in regions:
            lineage.region_key_by_id[region.region_id] = lineage.allocate_region_key()
        for group in groups:
            lineage.group_key_by_id[group.group_id] = lineage.allocate_group_key()
        return lineage

    def allocate_region_key(self) -> str:
        key = f"region_key_{self.next_region_key:06d}"
        self.next_region_key += 1
        return key

    def allocate_group_key(self) -> str:
        key = f"group_key_{self.next_group_key:06d}"
        self.next_group_key += 1
        return key
```

Export the public types from `capx.runtime_control`.

**Step 4: Run tests and verify pass**

Run the command from Step 2.

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/runtime_control/lineage.py capx/runtime_control/__init__.py \
  tests/test_runtime_control_lineage.py
git commit -m "Add stable Capsule unit lineage"
```

### Task 5: Reconcile Lineage Across Append and Patch

**Files:**
- Modify: `capx/runtime_control/lineage.py`
- Modify: `tests/test_runtime_control_lineage.py`
- Reference then remove: `capx/envs/trial.py:3445-3619`

**Step 1: Write failing reconciliation tests**

Add four focused tests:

```python
def test_append_preserves_prefix_keys_and_allocates_fresh_duplicate_keys(): ...
def test_append_rejects_group_crossing_old_source_boundary(): ...
def test_patch_preserves_shifted_unaffected_unit_keys(): ...
def test_ambiguous_mapping_raises_without_mutating_previous_lineage(): ...
```

The duplicate test must append the exact text of an already executed
`open_gripper()` and assert the new group key differs from the executed key.

**Step 2: Run tests and verify failure**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_lineage.py -q'
```

Expected: FAIL because reconciliation is not implemented.

**Step 3: Implement exact edit-aware reconciliation**

Add:

```python
def reconcile_lineage(
    *,
    edit_kind: str,
    previous_source: str,
    current_source: str,
    previous_regions: list[CodeRegion],
    current_regions: list[CodeRegion],
    previous_groups: list[CodeRegionGroup],
    current_groups: list[CodeRegionGroup],
    previous_lineage: UnitLineage,
    edit_start_line: int,
    edit_end_line: int,
    line_delta: int,
) -> UnitLineage:
    ...
```

Implementation rules:

- Copy counters and executed-key sets into a new object; never mutate the input.
- Append maps only exact old-prefix spans and source.
- Patch maps only exact unchanged or line-delta-shifted spans and source.
- Never fall back to every identical source candidate.
- Allocate a fresh key for every unmatched current unit.
- Raise `LineageAmbiguityError` for missing executed lineage, multiple exact candidates,
  or a group crossing the append boundary.
- Derive executed group keys through stable keys, not intersections of temporary IDs.

Delete `_remap_executed_side_effect_ledger` and `_lineage_matches_after_edit` only after
all callers migrate in Tasks 6-7.

**Step 4: Run lineage tests**

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/runtime_control/lineage.py tests/test_runtime_control_lineage.py
git commit -m "Reconcile Capsule lineage across source edits"
```

### Task 6: Use Stable Keys in Replay Guards and Execution Recording

**Files:**
- Modify: `capx/envs/trial.py:2102-2725,3055-3348`
- Modify: `tests/test_runtime_control_trial_loop.py:3334-3520,4370-4560`

**Step 1: Rewrite replay tests around stable keys**

Port the existing renumbering and duplicate-source tests. Add an end-to-end assertion:

```python
def test_distinct_appended_duplicate_group_can_execute_once_each(tmp_path):
    actions = [
        {"action": "run_group", "args": {"group_id": "group_1"}},
        {"action": "append_recovery", "args": {"source": recovery_source}},
        {"action": "run_group", "args": {"group_id": "group_2"}},
    ]
    ...
    assert api.calls == ["open_gripper", "open_gripper"]
    assert trace[2]["event"]["status"] == "success"
    assert trace[0]["unit_key"] != trace[2]["unit_key"]
```

Add a fourth action rerunning the same current group and assert
`side_effect_replay`.

**Step 2: Run focused tests and verify failure**

Run the exact new test and the two existing lineage safety tests.

Expected: FAIL because guards still use temporary IDs.

**Step 3: Integrate `UnitLineage`**

- Initialize lineage after initial source analysis.
- Change `_no_rollback_guard_event` to translate target IDs to stable keys.
- Change `_record_runtime_side_effect_execution` to record stable region/group keys.
- Build prompt ledger display IDs by reversing the current ID-to-key maps for executed
  keys.
- Add `unit_key` and `source_revision` to history entries and step metrics.
- Preserve trace-based recording when a group fails after a robot side effect.

Do not alter reward, strict-subset, or contract guards.

**Step 4: Run focused and full loop tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_trial_loop.py tests/test_runtime_control_lineage.py -q'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/envs/trial.py tests/test_runtime_control_trial_loop.py
git commit -m "Use stable keys for Capsule replay safety"
```

### Task 7: Make Patch and Append Transactions Atomic

**Files:**
- Modify: `capx/envs/trial.py:1167-1300,2595-2660,2922-3040,3445-3667`
- Modify: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write failing atomicity tests**

Add tests proving that syntax failure, lineage ambiguity, and append-boundary crossing do
not change source revision, groups, or executed keys. Capture those values from trace
metrics before and after the rejected action.

```python
assert event.status == "invalid"
assert event.evidence["edit_rejection_reason"] == "lineage_ambiguous"
assert event.evidence["source_revision_before"] == event.evidence["source_revision_after"]
assert final_source == original_source
```

**Step 2: Run tests and verify failure**

Expected: FAIL because the current path updates source and then remaps fail-closed.

**Step 3: Implement candidate preparation and commit**

Add private structures and helpers near `_CapsuleSourceAnalysis`:

```python
@dataclass
class _PreparedSourceEdit:
    source: str
    analysis: _CapsuleSourceAnalysis
    revision: SourceRevision
    lineage: UnitLineage


def _prepare_capsule_source_edit(...) -> _PreparedSourceEdit:
    candidate_source = _candidate_source_for_action(...)
    candidate_analysis = _analyze_capsule_source(candidate_source, ...)
    candidate_lineage = reconcile_lineage(...)
    _validate_candidate_source_edit(...)
    return _PreparedSourceEdit(...)
```

The loop commits every candidate field together only after this helper returns. Convert
`LineageAmbiguityError` and analysis failures into invalid `RuntimeEvent`s with explicit
reasons. Remove the old remap helper and fail-closed branch after the last caller is gone.

**Step 4: Run source-edit, contract, and loop tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_trial_loop.py tests/test_runtime_control_contract.py tests/test_runtime_control_lineage.py -q'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/envs/trial.py tests/test_runtime_control_trial_loop.py
git commit -m "Make Capsule source edits transactional"
```

### Task 8: Remove Hidden Pending-Recovery Execution

**Files:**
- Modify: `capx/envs/trial.py:2257-2660,3380-3415`
- Modify: `tests/test_runtime_control_trial_loop.py:4580-4750`

**Step 1: Replace the auto-execution regression test**

Replace `test_capsule_llm_step_auto_executes_appended_recovery_before_next_action` with:

```python
def test_capsule_append_requires_new_decision_before_each_group(tmp_path, monkeypatch):
    responses = iter(
        [
            {"content": json.dumps({"action": "append_recovery", "args": {"source": recovery}})},
            {"content": json.dumps({"action": "run_group", "args": {"group_id": "group_2"}})},
            {"content": json.dumps({"action": "run_group", "args": {"group_id": "group_3"}})},
        ]
    )
    monkeypatch.setattr(trial_module, "_query_model", lambda *_: next(responses))
    ...
    assert action_names == ["append_recovery", "run_group", "run_group"]
    assert query_count == 3
    assert primitive_calls_after_append == 0
    assert all(row.get("action_origin") != "pending_recovery" for row in metrics)
```

**Step 2: Run the new test and verify failure**

Expected: FAIL because append populates and drains `pending_recovery_actions`.

**Step 3: Remove the queue**

- Delete `pending_recovery_actions` and `forced_recovery_action`.
- Delete `_runtime_actions_for_appended_recovery` and its count-based side-effect budget
  update.
- Always construct a decision prompt unless `scripted_actions` supplies that exact step.
- After append commit, return to the top of the loop without executing a group.
- Add `action_origin="llm"` or `"scripted"` to every step metric.

**Step 4: Run scheduling and loop tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_trial_loop.py -q'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/envs/trial.py tests/test_runtime_control_trial_loop.py
git commit -m "Require an LLM decision for every Capsule action"
```

### Task 9: Add Automatic Post-Action Observations to Prompts

**Files:**
- Modify: `capx/runtime_control/schema.py`
- Modify: `capx/runtime_control/prompts.py:76-450,1292-1353`
- Modify: `capx/envs/trial.py:2296-2593`
- Modify: `tests/test_runtime_control_prompts.py`
- Modify: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write failing observation tests**

Add a prompt test proving the latest observation survives a one-character prompt budget
and a loop test proving one observation per attempted group:

```python
def test_compact_prompt_always_keeps_latest_post_action_observation(): ...


def test_capsule_records_one_post_action_observation_per_group(tmp_path):
    ...
    assert metrics[0]["post_action_observation_recorded"] is True
    assert metrics[0]["new_trace_event_count"] == 1
    assert next_prompt.count("Latest post-action observation") == 1
```

**Step 2: Run tests and verify failure**

Expected: FAIL because no dedicated latest-observation prompt field exists.

**Step 3: Implement `PostActionObservation`**

Add the approved dataclass to `schema.py` and export it. In the loop:

```python
trace_mark = executor.trace.mark()
before_state = _capsule_state_snapshot(...)
event = _execute_runtime_action(...)
after_state = _capsule_state_snapshot(...)
new_trace_events = executor.trace.events_since(trace_mark)
latest_observation = PostActionObservation(...)
```

Pass `latest_observation` and `source_revision` to `build_capsule_prompt`. Render it as a
dedicated compact section before bounded history. Apply explicit scalar/list bounds but
never remove the entire section during fallback compaction.

For non-group actions, record normal feedback but set
`post_action_observation_recorded=False`; the required equality concerns attempted group
execution.

**Step 4: Run prompt and loop tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_prompts.py tests/test_runtime_control_trial_loop.py -q'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/runtime_control/schema.py capx/runtime_control/prompts.py \
  capx/envs/trial.py tests/test_runtime_control_prompts.py \
  tests/test_runtime_control_trial_loop.py
git commit -m "Add automatic Capsule post-action observations"
```

### Task 10: Bind Recovery Authorization to Stable Group Keys

**Files:**
- Modify: `capx/runtime_control/lineage.py`
- Modify: `capx/envs/trial.py:2270-2660,3120-3235`
- Modify: `tests/test_runtime_control_lineage.py`
- Modify: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write failing recovery-generation tests**

Cover:

```python
def test_append_authorizes_each_new_side_effect_group_by_stable_key(): ...
def test_inspection_does_not_consume_recovery_authorization(): ...
def test_executing_recovery_group_consumes_only_its_key(): ...
def test_patch_recomputes_unexecuted_recovery_keys_atomically(): ...
def test_second_append_without_new_physical_evidence_is_rejected(): ...
def test_physical_trace_after_append_allows_later_append(): ...
```

**Step 2: Run tests and verify failure**

Expected: FAIL because authorization is a scalar counter and repeated append is allowed.

**Step 3: Implement `RecoveryGeneration`**

Add:

```python
@dataclass
class RecoveryGeneration:
    generation_id: str
    source_revision: int
    start_line: int
    end_line: int
    observation_functions: tuple[str, ...]
    authorized_group_keys: set[str] = field(default_factory=set)
    executed_group_keys: set[str] = field(default_factory=set)
```

Replace `recovery_side_effect_budget` with key lookup in the reward-drop guard. Track the
trace/world revision at successful append. Reject a later append with
`no_new_physical_state_since_last_append` until group execution produces new physical
trace evidence. Recompute generation membership and authorization transactionally when an
unexecuted recovery group is patched.

**Step 4: Run lineage, reward-guard, and loop tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_lineage.py tests/test_runtime_control_trial_loop.py tests/test_runtime_control_feedback.py -q'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/runtime_control/lineage.py capx/envs/trial.py \
  tests/test_runtime_control_lineage.py tests/test_runtime_control_trial_loop.py
git commit -m "Bind Capsule recovery to stable group keys"
```

### Task 11: Prevent Duplicate Variable Inspections and Finalize Metrics

**Files:**
- Modify: `capx/envs/trial.py:2257-2725,2904-2920`
- Modify: `capx/runtime_control/prompts.py:1292-1353`
- Modify: `tests/test_runtime_control_trial_loop.py`
- Modify: `tests/test_runtime_control_prompts.py`

**Step 1: Write failing inspection and metric tests**

```python
def test_repeated_variable_inspection_without_revision_change_is_invalid(tmp_path): ...
def test_variable_inspection_is_allowed_after_group_changes_namespace(tmp_path): ...
def test_trial_metrics_count_decisions_attempts_groups_and_observations(tmp_path): ...
```

Assert the repeated event message contains `no_new_variable_state`. Assert trial metrics
contain no pending-recovery or auto-forward keys.

**Step 2: Run tests and verify failure**

Expected: FAIL because repeated inspections are unguarded and metrics are incomplete.

**Step 3: Implement the context-revision guard and metrics**

Normalize requested variable names to a sorted tuple. Cache:

```python
inspection_key = (
    source_revision.revision,
    len(executor.trace.events),
    namespace_revision,
    tuple(sorted(names)),
)
```

Do not advance `namespace_revision` for inspection itself. Advance it after successful
group/region execution or a committed source edit. Record the approved per-step and trial
metric fields, and assert observation count equals attempted group count at finalization.

**Step 4: Run focused tests**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_trial_loop.py tests/test_runtime_control_prompts.py -q'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/envs/trial.py capx/runtime_control/prompts.py \
  tests/test_runtime_control_trial_loop.py tests/test_runtime_control_prompts.py
git commit -m "Guard Capsule diagnostic actions and metrics"
```

### Task 12: Port Safety Coverage and Update Active Documentation

**Files:**
- Modify: `tests/test_runtime_control_trial_loop.py`
- Modify: `docs/paper/capsule-paper-draft-zh.md:130-210`
- Modify: `docs/paper/capsule-open-items.md:1-35`
- Review: `docs/plans/2026-08-14-capsule-llm-step-only-runtime-design.md`

**Step 1: Add a safety-coverage checklist test selection**

Ensure sole-loop tests explicitly cover:

- strict subset and program contract rejection;
- fresh-observation requirement;
- replay prevention after successful and partially failed side effects;
- reward-drop guard and recovery authorization;
- premature finish and task-success short-circuit;
- sparse terminal feedback;
- visual/diagnostic state isolation;
- sticky safety failure.

Port any missing assertion from deleted auto-forward tests using scripted LLM actions.

**Step 2: Run safety tests before documentation edits**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_contract.py tests/test_runtime_control_executor.py tests/test_runtime_control_feedback.py tests/test_runtime_control_trial_loop.py -q'
```

Expected: PASS.

**Step 3: Update active documentation**

Replace active claims that auto-forward is recommended with the sole-loop policy:

```text
Every Capsule action is selected by the Action LLM. The host executes at most one action,
captures a bounded post-action observation, and returns that evidence for the next
decision. Recovery remains append-only and forward-only.
```

Remove auto-forward as an open ablation item. Do not rewrite historical plan documents.

**Step 4: Check documentation and references**

Run:

```bash
rg -n "auto_forward|capsule_control_mode" capx env_configs tests docs/paper
```

Expected: no active code/config/test/paper references. Historical `docs/plans` references
are intentionally excluded.

**Step 5: Commit**

```bash
git add tests/test_runtime_control_trial_loop.py \
  docs/paper/capsule-paper-draft-zh.md docs/paper/capsule-open-items.md
git commit -m "Document the Capsule LLM-step-only policy"
```

### Task 13: Full Local Verification

**Files:**
- No source changes expected
- Inspect: all files changed by Tasks 1-12

**Step 1: Run formatting and static checks**

Run in WSL:

```bash
uv run --no-sync ruff format --check capx tests
uv run --no-sync ruff check capx tests
```

Expected: both exit 0. If formatting is needed, run `ruff format` only on changed Python
files, sync those exact files back to the Windows worktree through `apply_patch`-equivalent
edits, and rerun. Do not bulk-format unrelated files.

**Step 2: Run the complete runtime-control suite**

```bash
uv run --no-sync pytest \
  tests/test_runtime_control_schema.py \
  tests/test_runtime_control_trace.py \
  tests/test_runtime_control_segmenter.py \
  tests/test_runtime_control_normalizer.py \
  tests/test_runtime_control_contract.py \
  tests/test_runtime_control_executor.py \
  tests/test_runtime_control_feedback.py \
  tests/test_runtime_control_prompts.py \
  tests/test_runtime_control_lineage.py \
  tests/test_runtime_control_trial_loop.py \
  tests/test_runtime_control_config.py -q
```

Expected: all pass.

**Step 3: Run repository environment regressions**

```bash
uv run --no-sync pytest tests/test_environments.py -q
```

Expected: PASS or documented environment-specific skips only.

**Step 4: Inspect removal and diff quality**

```bash
rg -n "auto_forward|pending_recovery_actions|forced_recovery_action|inspect_trace|capsule_control_mode" capx env_configs tests docs/paper
git diff --check
git status --short
```

Expected: no removed runtime symbols; no whitespace errors; only intended changes and
pre-existing untracked cache files.

**Step 5: Commit any verification-only corrections**

If no corrections were needed, do not create an empty commit. Otherwise:

```bash
git add <only-corrected-files>
git commit -m "Fix Capsule runtime verification issues"
```

### Task 14: SeeTaCloud LIBERO Task-0 Validation

**Files:**
- No source edits unless a new reproducible defect is found
- Inspect remote output artifacts only

**Step 1: Sync the completed branch to the remote worktree**

Use the approved SeeTaCloud connection and a clean remote worktree. Verify the exact
commit before running:

```bash
git rev-parse HEAD
git status --short
```

Expected: the intended implementation commit and no source modifications. Do not include
credentials in files or logs.

**Step 2: Verify services and configuration**

- Text LLM: packyapi `deepseek-v4-flash`, reasoning disabled.
- Molmo: `allenai/Molmo2-8B` on port 8122, used only by `FrankaLiberoApi`.
- All three visual prompt switches remain false for this ablation.
- Pyroki and other configured API servers are ready.

**Step 3: Run one trial**

```bash
source .venv-libero/bin/activate
MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/libero/franka_libero_object_0_capsule_llm_step.yaml \
  --total-trials 1 --num-workers 1
```

**Step 4: Verify scheduler and ledger invariants in artifacts**

Assert from trace, prompts, and metrics:

- every `run_group` is preceded by a new LLM decision;
- every attempted group has exactly one post-action observation;
- append performs no primitive call;
- no action has `pending_recovery` origin;
- no action is `inspect_trace`;
- distinct appended duplicate source receives distinct unit keys;
- no false `side_effect_replay` occurs for a fresh append unit;
- rejected edits leave source revision unchanged.

**Step 5: Report results**

Report:

- reward;
- task completed;
- sandbox rc;
- loop exit reason;
- LLM decisions and attempts;
- group executions and observations;
- append/patch commits and rejections;
- replay blocks;
- error log excerpts;
- exact output directory.

Reward 0 is not by itself an implementation failure if all scheduler and ledger
invariants pass. Any invariant failure requires a new failing local test before a source
fix.

---

## Completion Criteria

Implementation is complete only when:

- active code, config, tests, and paper docs contain no auto-forward mode;
- Capsule has one strict per-action LLM loop;
- append never triggers hidden execution;
- every attempted group produces one automatic observation visible in the next prompt;
- `inspect_trace` is not a model action;
- repeated variable inspection without new context is rejected;
- stable keys, not temporary IDs or duplicate text, drive replay safety;
- candidate source edits are atomic;
- recovery authorization is bound to stable keys;
- all focused and complete WSL tests pass;
- the remote one-trial artifact audit satisfies scheduler and ledger invariants.
