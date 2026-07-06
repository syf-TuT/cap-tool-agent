# Capsule Group Normalization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move capsule group boundary decisions into a pure metadata normalizer while preserving original executable Python source exactly.

**Architecture:** `capx.runtime_control.segmenter` becomes the structural-facts layer: source parsing, source-preserving regions, and per-region analysis. A new `capx.runtime_control.normalizer` owns all group boundary decisions and returns `CodeRegionGroup` metadata whose source is a direct concatenation of member region sources. `CapsuleExecutor` continues to compile and execute only the original region/group source slices.

**Tech Stack:** Python 3.10-3.12, `ast`, dataclasses, pytest, WSL Ubuntu runtime with `uv run --no-sync`.

---

## Ground Rules

- Do not rewrite executable Python source.
- Do not inline helper bodies into executable source.
- Do not split compound statements into separately executable nested statements.
- Do not reintroduce hardcoded robot primitive names in normalization policy.
- All tests and runtime checks must run through the WSL project copy at
  `/home/capx/code/cap-x`, after syncing edited files from `F:\code\cap-x`.

Use this WSL command shape for tests:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; <command>'
```

Before each WSL test run, sync touched files, for example:

```bash
cp /mnt/f/code/cap-x/capx/runtime_control/segmenter.py capx/runtime_control/segmenter.py
cp /mnt/f/code/cap-x/capx/runtime_control/normalizer.py capx/runtime_control/normalizer.py
cp /mnt/f/code/cap-x/capx/envs/trial.py capx/envs/trial.py
cp /mnt/f/code/cap-x/tests/test_runtime_control_segmenter.py tests/test_runtime_control_segmenter.py
cp /mnt/f/code/cap-x/tests/test_runtime_control_normalizer.py tests/test_runtime_control_normalizer.py
cp /mnt/f/code/cap-x/tests/test_runtime_control_trial_loop.py tests/test_runtime_control_trial_loop.py
```

Create `capx/runtime_control/normalizer.py` only when the first test requires it. Until
then, avoid production code changes.

### Task 1: Lock Metadata-Only Invariants

**Files:**
- Create: `tests/test_runtime_control_normalizer.py`
- Test: `tests/test_runtime_control_normalizer.py`

**Step 1: Write the failing tests**

Add tests that express the hard invariants before adding the normalizer:

```python
from capx.runtime_control.normalizer import segment_python_code_groups
from capx.runtime_control.segmenter import segment_python_code


def test_groups_preserve_original_source_bytes():
    source = "\n".join(
        [
            "x = 1",
            "move_to(x)",
            "y = x + 1",
            "move_to(y)",
        ]
    )

    regions = segment_python_code(source)
    groups = segment_python_code_groups(
        source,
        regions,
        side_effect_calls={"move_to"},
    )

    assert "".join(group.source for group in groups) == source


def test_groups_partition_regions_without_gaps_or_reordering():
    source = "\n".join(
        [
            "x = 1",
            "move_to(x)",
            "y = x + 1",
            "move_to(y)",
        ]
    )
    regions = segment_python_code(source)
    groups = segment_python_code_groups(
        source,
        regions,
        side_effect_calls={"move_to"},
    )

    grouped_region_ids = [
        region_id
        for group in groups
        for region_id in group.region_ids
    ]

    assert grouped_region_ids == [region.region_id for region in regions]
    assert len(grouped_region_ids) == len(set(grouped_region_ids))
```

**Step 2: Run tests to verify failure**

Run in WSL after syncing the new test file:

```bash
uv run --no-sync pytest tests/test_runtime_control_normalizer.py -q
```

Expected: FAIL because `capx.runtime_control.normalizer` does not exist.

**Step 3: Commit the failing tests only**

```bash
git add tests/test_runtime_control_normalizer.py
git commit -m "Test capsule group metadata invariants"
```

### Task 2: Make Segmenter Source-Preserving and Analysis-Only

**Files:**
- Modify: `capx/runtime_control/segmenter.py`
- Modify: `tests/test_runtime_control_segmenter.py`
- Test: `tests/test_runtime_control_segmenter.py`

**Step 1: Write failing segmenter tests**

Add tests for exact source partitioning and public analysis:

```python
from capx.runtime_control.segmenter import analyze_python_regions, segment_python_code


def test_segmenter_regions_partition_source_bytes():
    source = "x = 1\nmove_to(x)\ny = x + 1"

    regions = segment_python_code(source)

    assert "".join(region.source for region in regions) == source


def test_segmenter_exposes_region_analysis_facts():
    source = "x = 1\nmove_to(x)\n"
    regions = segment_python_code(source)

    analyses = analyze_python_regions(
        source,
        regions,
        side_effect_calls={"move_to"},
    )

    assert [analysis.region_id for analysis in analyses] == ["region_1", "region_2"]
    assert analyses[0].defined_names == ["x"]
    assert analyses[1].primitive_calls == ["move_to"]
    assert analyses[1].has_robot_side_effect is True
```

Run:

```bash
uv run --no-sync pytest tests/test_runtime_control_segmenter.py -q
```

Expected: FAIL because regions do not yet preserve direct source concatenation and
`analyze_python_regions` is not public.

**Step 2: Implement source-preserving regions**

In `segment_python_code(source)`:

1. Use `source.splitlines(keepends=True)` so newline bytes stay attached to chunks.
2. For each top-level AST node, build a region source slice that preserves the original
   text for that region. A practical first pass is:
   - region start is the AST node `lineno`;
   - region end is the line before the next top-level node starts;
   - the final region ends at the end of source.
3. Keep `start_line` and `end_line` consistent with the source slice used for patching.

Do not change execution behavior. A trailing newline in a compiled region is safe.

**Step 3: Expose `RegionAnalysis` and `analyze_python_regions`**

Rename `_RegionAnalysis` to public `RegionAnalysis`:

```python
@dataclass(frozen=True)
class RegionAnalysis:
    region_id: str
    primitive_calls: list[str]
    defined_names: list[str]
    used_names: list[str]
    has_robot_side_effect: bool
    has_structural_effect: bool
```

Add:

```python
def analyze_python_regions(
    source: str,
    regions: list[CodeRegion],
    *,
    side_effect_calls: set[str],
) -> list[RegionAnalysis]:
    return [_analyze_region(region, side_effect_calls) for region in regions]
```

Update `_analyze_region` to set `region_id` and rename `is_effect` to
`has_structural_effect`.

**Step 4: Remove group-boundary ownership from segmenter**

Delete or stop exporting the group-building implementation from `segmenter.py`.
If removal is too large for this step, leave a temporary internal helper unexported, but
do not keep `segmenter` as the long-term owner of `segment_python_code_groups`.

**Step 5: Run segmenter tests**

```bash
uv run --no-sync pytest tests/test_runtime_control_segmenter.py -q
```

Expected: PASS for structural region and analysis tests. Existing group tests may still
fail until Task 3 migrates them to the normalizer; update imports only after the normalizer
exists.

**Step 6: Commit**

```bash
git add capx/runtime_control/segmenter.py tests/test_runtime_control_segmenter.py
git commit -m "Expose capsule region analysis facts"
```

### Task 3: Add Normalizer With Current Sense-to-Act Behavior

**Files:**
- Create: `capx/runtime_control/normalizer.py`
- Modify: `capx/runtime_control/__init__.py`
- Modify: `tests/test_runtime_control_segmenter.py`
- Modify: `tests/test_runtime_control_normalizer.py`
- Test: `tests/test_runtime_control_segmenter.py`
- Test: `tests/test_runtime_control_normalizer.py`

**Step 1: Move group tests to the normalizer**

In `tests/test_runtime_control_segmenter.py`, keep only region and analysis tests.
Move group-specific tests to `tests/test_runtime_control_normalizer.py`, importing:

```python
from capx.runtime_control.normalizer import segment_python_code_groups
from capx.runtime_control.segmenter import segment_python_code
```

Run:

```bash
uv run --no-sync pytest tests/test_runtime_control_segmenter.py tests/test_runtime_control_normalizer.py -q
```

Expected: FAIL because `normalizer.py` is not implemented.

**Step 2: Implement `GroupingPolicy` and group construction**

Create `capx/runtime_control/normalizer.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from capx.runtime_control.schema import CodeRegion, CodeRegionGroup
from capx.runtime_control.segmenter import (
    ROBOT_SIDE_EFFECT_CALLS,
    RegionAnalysis,
    analyze_python_regions,
    segment_python_code,
)


@dataclass(frozen=True)
class GroupingPolicy:
    max_regions_per_group: int = 20
    min_groups: int = 3
    max_groups: int = 8


def segment_python_code_groups(
    source: str,
    regions: list[CodeRegion] | None = None,
    *,
    max_regions_per_group: int = 20,
    side_effect_calls: set[str] | None = None,
) -> list[CodeRegionGroup]:
    if regions is None:
        regions = segment_python_code(source)
    if not regions:
        return []
    if side_effect_calls is None:
        side_effect_calls = ROBOT_SIDE_EFFECT_CALLS

    analyses = analyze_python_regions(
        source,
        regions,
        side_effect_calls=side_effect_calls,
    )
    return normalize_python_code_groups(
        source,
        regions,
        analyses,
        policy=GroupingPolicy(max_regions_per_group=max_regions_per_group),
    )
```

Add `normalize_python_code_groups(...)` using the existing sense-to-act rule:

- start a new group when the current group already has an effect and the next region does
  not have an effect;
- also split when `len(current) >= policy.max_regions_per_group`;
- build group source with `"".join(region.source for region in regions)`;
- build group metadata from the analyses for member regions.

**Step 3: Update package exports**

In `capx/runtime_control/__init__.py`, import `segment_python_code_groups` from
`capx.runtime_control.normalizer`, not `segmenter`.

**Step 4: Run migrated tests**

```bash
uv run --no-sync pytest tests/test_runtime_control_segmenter.py tests/test_runtime_control_normalizer.py -q
```

Expected: PASS. Existing expected source strings may need trailing-newline-aware assertions,
but do not weaken source identity invariants.

**Step 5: Commit**

```bash
git add capx/runtime_control/normalizer.py capx/runtime_control/__init__.py tests/test_runtime_control_segmenter.py tests/test_runtime_control_normalizer.py
git commit -m "Move capsule group boundaries to normalizer"
```

### Task 4: Normalize Helper and Control-Flow Effect Signals

**Files:**
- Modify: `capx/runtime_control/normalizer.py`
- Modify: `capx/runtime_control/segmenter.py`
- Modify: `tests/test_runtime_control_normalizer.py`
- Test: `tests/test_runtime_control_normalizer.py`

**Step 1: Write failing helper/control-flow tests**

Add tests:

```python
def test_helper_definition_alone_is_not_a_robot_side_effect():
    source = "\n".join(
        [
            "def pick():",
            "    move_to([1, 2, 3])",
            "x = 1",
        ]
    )

    groups = segment_python_code_groups(
        source,
        side_effect_calls={"move_to"},
    )

    assert all(group.has_robot_side_effect is False for group in groups)


def test_helper_call_inherits_declared_side_effect_for_group_metadata():
    source = "\n".join(
        [
            "def pick():",
            "    move_to([1, 2, 3])",
            "pick()",
            "x = 1",
            "move_to([4, 5, 6])",
        ]
    )

    groups = segment_python_code_groups(
        source,
        side_effect_calls={"move_to"},
    )

    assert groups[0].has_robot_side_effect is True
    assert "move_to" in groups[0].primitive_calls
    assert groups[1].has_robot_side_effect is True


def test_control_flow_region_containing_side_effect_is_effect_metadata():
    source = "\n".join(
        [
            "ready = True",
            "if ready:",
            "    move_to([1, 2, 3])",
            "next_target = [4, 5, 6]",
            "move_to(next_target)",
        ]
    )

    groups = segment_python_code_groups(
        source,
        side_effect_calls={"move_to"},
    )

    assert [group.region_ids for group in groups] == [
        ["region_1", "region_2"],
        ["region_3", "region_4"],
    ]
```

Run:

```bash
uv run --no-sync pytest tests/test_runtime_control_normalizer.py -q
```

Expected: FAIL. The helper definition currently contributes lexical robot calls even when
not executed, and helper call reachability is not modeled explicitly.

**Step 2: Extend analysis facts without creating boundary policy in segmenter**

In `RegionAnalysis`, add facts needed by the normalizer:

```python
defined_functions: list[str]
top_level_call_names: list[str]
lexical_side_effect_calls: list[str]
```

Keep these as facts only. Do not let `segmenter.py` choose group boundaries.

**Step 3: Implement normalizer reachability**

In `normalizer.py`:

1. Build a map from locally defined helper name to that function-definition region's
   `lexical_side_effect_calls`.
2. For a top-level helper-call region, add the helper's side-effect calls to that region's
   normalized effect metadata.
3. For control-flow regions, mark them as normalized effect regions when lexical
   side-effect calls are present.
4. For helper-definition regions, do not mark the group as robot-side-effectful unless the
   definition is executed through a top-level call in the same group.
5. Leave unresolved dynamic calls conservative: no crash, no inferred helper reachability.

The execution source remains unchanged. Only group metadata changes.

**Step 4: Run tests**

```bash
uv run --no-sync pytest tests/test_runtime_control_normalizer.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/runtime_control/segmenter.py capx/runtime_control/normalizer.py tests/test_runtime_control_normalizer.py
git commit -m "Normalize capsule helper effect metadata"
```

### Task 5: Add Broad-Band Guardrails With Def-Use Safety

**Files:**
- Modify: `capx/runtime_control/normalizer.py`
- Modify: `tests/test_runtime_control_normalizer.py`
- Test: `tests/test_runtime_control_normalizer.py`

**Step 1: Write failing broad-band tests**

Add a test for explosion control that creates many alternating single-effect phases and
expects the normalizer to reduce group count into the policy band without dropping source:

```python
def test_normalizer_reduces_many_single_effect_groups_within_policy_band():
    source = "\n".join(
        line
        for i in range(10)
        for line in [
            f"target_{i} = [{i}, {i}, {i}]",
            f"move_to(target_{i})",
        ]
    )

    groups = segment_python_code_groups(
        source,
        side_effect_calls={"move_to"},
    )

    assert 3 <= len(groups) <= 8
    assert "".join(group.source for group in groups) == source
```

Add a safety test that ensures the normalizer keeps a boundary when ambiguity would hide
repair granularity:

```python
def test_normalizer_keeps_ambiguous_dependency_boundary():
    source = "\n".join(
        [
            "target = plan_a()",
            "move_to(target)",
            "target = observe_and_replan()",
            "move_to(target)",
        ]
    )

    groups = segment_python_code_groups(
        source,
        side_effect_calls={"move_to"},
    )

    assert [group.region_ids for group in groups] == [
        ["region_1", "region_2"],
        ["region_3", "region_4"],
    ]
```

Run:

```bash
uv run --no-sync pytest tests/test_runtime_control_normalizer.py -q
```

Expected: FAIL for the explosion-control test until broad-band merging is implemented.

**Step 2: Implement policy-controlled merging**

In `GroupingPolicy`, keep:

```python
min_groups: int = 3
max_groups: int = 8
```

After initial sense-to-act grouping:

1. If `len(groups) <= policy.max_groups`, return groups unchanged.
2. Otherwise, consider adjacent single-effect groups for merging.
3. Merge only when `_can_merge_adjacent_groups(left, right)` returns true.
4. Stop once `len(groups) <= policy.max_groups` or no safe merge remains.

Use existing `defined_names` and `used_names` only. The safe default is no merge when
dependency facts are ambiguous.

**Step 3: Implement a narrow `_can_merge_adjacent_groups`**

Start conservative:

- allow merge when neither group redefines a name used by the other after a side effect;
- disallow merge when both groups define the same name and the right group uses that name;
- disallow merge when either group has no clear effect marker;
- do not infer new data dependencies.

This is intentionally imperfect. It should reduce obvious over-fragmentation without
pretending to solve full Python data flow.

**Step 4: Run tests**

```bash
uv run --no-sync pytest tests/test_runtime_control_normalizer.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add capx/runtime_control/normalizer.py tests/test_runtime_control_normalizer.py
git commit -m "Bound capsule group count with safe merges"
```

### Task 6: Wire Trial Loop to the Normalizer

**Files:**
- Modify: `capx/envs/trial.py`
- Modify: `capx/runtime_control/__init__.py`
- Modify: `tests/test_runtime_control_trial_loop.py`
- Test: `tests/test_runtime_control_trial_loop.py`

**Step 1: Write failing trial-loop source-preservation test**

Add:

```python
def test_capsule_trial_writes_original_source_after_group_normalization(tmp_path):
    source = "\n".join(
        [
            "def pick():",
            "    move_to([1, 2, 3])",
            "pick()",
            "x = 1",
        ]
    )

    summary = _run_capsule_trial(
        env=FakeCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 2,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code=source,
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "finish", "args": {}},
        ],
    )

    assert Path(summary.code_path).read_text() == source
```

Ensure `FakeApi.functions()` exposes `move_to`, which it already does.

Run:

```bash
uv run --no-sync pytest tests/test_runtime_control_trial_loop.py::test_capsule_trial_writes_original_source_after_group_normalization -q
```

Expected: FAIL until `trial.py` imports and uses the normalizer correctly after the move.

**Step 2: Update imports and calls**

In `capx/envs/trial.py`, import `segment_python_code_groups` from
`capx.runtime_control.normalizer` or from package exports that now point to the normalizer.
Keep `segment_python_code` from `segmenter`.

Use the same normalizer path for:

- initial grouping;
- regrouping after `patch_region`;
- regrouping after `patch_group`;
- regrouping after `append_recovery`.

**Step 3: Run trial-loop tests**

```bash
uv run --no-sync pytest tests/test_runtime_control_trial_loop.py -q
```

Expected: PASS.

**Step 4: Commit**

```bash
git add capx/envs/trial.py capx/runtime_control/__init__.py tests/test_runtime_control_trial_loop.py
git commit -m "Use normalizer for capsule trial groups"
```

### Task 7: Focused Regression Verification

**Files:**
- No code changes expected.
- Test: `tests/test_runtime_control_segmenter.py`
- Test: `tests/test_runtime_control_normalizer.py`
- Test: `tests/test_runtime_control_trial_loop.py`
- Test: `tests/test_runtime_control_config.py`
- Test: `tests/test_runtime_control_side_effects.py`

**Step 1: Run focused runtime-control tests**

Run in WSL:

```bash
uv run --no-sync pytest tests/test_runtime_control_segmenter.py tests/test_runtime_control_normalizer.py tests/test_runtime_control_trial_loop.py tests/test_runtime_control_config.py tests/test_runtime_control_side_effects.py -q
```

Expected: PASS. Existing deprecation warnings from dependencies are acceptable.

**Step 2: Run whitespace and import smoke checks**

From Windows:

```powershell
git diff --check
```

Run in WSL:

```bash
uv run --no-sync python -c "import capx.runtime_control; import capx.envs.trial; print('normalizer imports ok')"
```

Expected: exit 0 and `normalizer imports ok`.

**Step 3: Inspect final diff**

```powershell
git diff --stat
git diff -- capx/runtime_control/segmenter.py capx/runtime_control/normalizer.py capx/envs/trial.py tests/test_runtime_control_normalizer.py
```

Confirm:

- no executable source rewriting;
- groups are built from original region sources;
- `segmenter.py` no longer owns boundary decisions;
- normalizer uses injected `side_effect_calls`.

**Step 4: Final commit if needed**

If any cleanup changes were required:

```bash
git add capx/runtime_control/segmenter.py capx/runtime_control/normalizer.py capx/runtime_control/__init__.py capx/envs/trial.py tests/test_runtime_control_segmenter.py tests/test_runtime_control_normalizer.py tests/test_runtime_control_trial_loop.py
git commit -m "Stabilize capsule group normalization"
```

If no cleanup changes were required, do not create an empty commit.
