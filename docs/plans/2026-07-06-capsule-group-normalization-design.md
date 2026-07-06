# Capsule Group Normalization Design

## Goal

Reduce capsule semantic-group count variance caused by LLM coding style while keeping
execution semantics unchanged.

The normalization layer affects grouping metadata only. It must never rewrite, reorder,
expand, or execute modified Python source. Every returned group source is always the
byte-for-byte concatenation of its member `CodeRegion.source` values in original order.

## Context

Current capsule grouping is driven by top-level AST statement shape:

1. `segment_python_code(source)` parses source into top-level `CodeRegion` objects.
2. `segment_python_code_groups(...)` analyzes each region and decides sense-to-act
   group boundaries.
3. `_run_capsule_trial()` executes the original group or region source through
   `CapsuleExecutor`.

This makes grouping sensitive to superficial LLM style differences. A direct top-level
`goto_pose(...)` call and a helper-wrapped `pick()` call can describe the same phase but
produce different effect signals and group counts. Raising `capsule_max_regions_per_group`
to 20 only fixes the loose fallback cap; it does not address this root cause.

## Chosen Approach

Use a pure analysis-view normalization layer.

Rejected alternatives:

- Enhancing the current segmenter in place would mix structural facts, normalization,
  and boundary policy in one function, making pre-normalized and post-normalized behavior
  hard to observe and test independently.
- Prompting the model to write flatter code is useful as a hint, but not reliable enough
  to be the system-side fix. Prompt style and runtime segmentation are orthogonal.

## Responsibility Split

The implementation should make the split explicit:

- `segmenter`: structural facts only. It parses source into regions and computes per-region
  facts such as calls, definitions, uses, and direct structural effect signals. It does not
  decide group boundaries.
- `normalizer`: boundary policy only. It consumes regions, analyses, environment-declared
  `side_effect_calls`, and fallback bandwidth settings, then returns `CodeRegionGroup`
  metadata. Sense-to-act grouping, lower-bound splitting, upper-bound merging, and safety
  checks all live here.

This means `segment_python_code_groups` should move out of `segmenter.py` into a new
normalizer module. `_RegionAnalysis` should become a public, testable analysis object
rather than a private implementation detail.

## Metadata-Only Invariants

These invariants are mandatory and should be covered by the first RED tests:

```python
"".join(group.source for group in groups) == original_source
```

Group membership is also a partition over regions:

- no region id is missing;
- no region id appears twice;
- region order is preserved.

Any implementation that rewrites source, drops source bytes, duplicates regions, or changes
execution spans must fail these tests immediately.

## Grouping Target

The objective is not to force a fixed 4 to 6 group count. Observed distributions such as
`[1, 1, 19, 13, 1]` show that a hard target would be brittle.

The objective is:

- avoid collapse to one giant group;
- avoid explosion to 13+ tiny groups;
- compress typical programs into a broad, stable band, initially around 3 to 8 groups.

The band is a policy knob, not a semantic truth. The implementation should prefer stable,
safe metadata over chasing an exact count.

## Boundary Signals

Normalization should use environment-declared side-effect names, collected via
`collect_side_effect_calls(...)`, rather than hardcoded robot vocabulary.

Conservative first-pass rules:

- A top-level bare call to a side-effect primitive is an effect region.
- A top-level control-flow region (`if`, `for`, `while`, `try`, `with`) containing a
  side-effect primitive call is an effect region for grouping metadata.
- A top-level helper definition is not itself an effect region.
- A top-level call to a locally defined helper inherits the helper's normalized effect
  facts when the helper body can be statically analyzed.
- Unresolved dynamic calls such as `globals()[name]()` fall back to the structural facts
  already produced by the segmenter.

This keeps conditional actions conditional at execution time. The metadata only says that
the region contains or reaches a robot side effect.

## Def-Use Safety

Merging adjacent single-effect groups requires a def-use safety check. The normalizer may
merge only when doing so does not hide a meaningful dependency split between phases.

The first implementation should use existing `defined_names` and `used_names` facts:

- if a later group uses names defined by an earlier adjacent group, the merge is usually
  safe because the execution order stays unchanged inside the larger group;
- if a planned merge would cross a boundary where repair or inspection granularity depends
  on a value produced after a side effect, keep the boundary;
- never introduce new data-flow facts beyond the existing analysis object in the first pass.

The safe default is to keep a boundary when dependency information is ambiguous.

## Data Flow

Initial segmentation:

```text
source
  -> segment_python_code(source)
  -> analyze_python_regions(source, regions, side_effect_calls)
  -> normalize_python_code_groups(source, regions, analyses, policy)
  -> CodeRegionGroup metadata
```

Patch and recovery paths use the same flow after source changes. The executor continues to
compile and execute only the original `CodeRegion.source` or `CodeRegionGroup.source`.

## Public API Sketch

```python
@dataclass(frozen=True)
class RegionAnalysis:
    region_id: str
    primitive_calls: list[str]
    defined_names: list[str]
    used_names: list[str]
    has_robot_side_effect: bool
    has_structural_effect: bool
    normalized_has_effect: bool


@dataclass(frozen=True)
class GroupingPolicy:
    max_regions_per_group: int = 20
    min_groups: int = 3
    max_groups: int = 8


def analyze_python_regions(
    source: str,
    regions: list[CodeRegion],
    *,
    side_effect_calls: set[str],
) -> list[RegionAnalysis]:
    ...


def normalize_python_code_groups(
    source: str,
    regions: list[CodeRegion],
    analyses: list[RegionAnalysis],
    *,
    policy: GroupingPolicy,
) -> list[CodeRegionGroup]:
    ...
```

Names may change during implementation, but the separation should not.

## Testing Strategy

Core unit tests:

- metadata-only source identity and region partition invariants;
- behavior-preserving migration of current semantic group tests;
- helper-wrapped side-effect calls produce effect boundaries at call sites;
- helper definitions alone do not create effect boundaries;
- control-flow regions containing side effects are marked as effect metadata;
- unresolved dynamic calls fall back without crashing;
- def-use guarded merging does not cross unsafe boundaries.

Trial-loop regression tests:

- `_run_capsule_trial()` writes the original source to `capsule_code_trial_XX.py`;
- patch and append recovery regroup through the normalizer while preserving source spans.

Verification should run focused runtime-control tests in WSL:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_segmenter.py tests/test_runtime_control_trial_loop.py tests/test_runtime_control_config.py -q'
```

## Non-Goals

- Do not rewrite executable Python source.
- Do not inline helper functions into source.
- Do not split a compound statement into separately executable nested statements.
- Do not make a fixed group count a correctness condition.
- Do not add new robot primitive hardcoding.
