# LLM-Step Compact Context Design

## Goal

Reduce token usage in Capsule `llm_step` mode while preserving its strict
stepwise control semantics: every runtime step still asks the LLM to choose the
next action. The change only compacts the prompt context sent for each action
decision.

## Problem

`llm_step` currently rebuilds a full runtime-control prompt on every step. The
prompt includes complete generated code regions and complete effect-bounded
groups, and recent history entries can include duplicated trace, feedback,
state, and patched-source payloads. This makes token use scale with both program
size and step count. In long trials or after source patches, repeated prompt
payloads can dominate experiment cost.

The runtime must keep strict stepwise control semantics: every normal execution
step remains selected by the LLM rather than hidden behind deterministic source
order execution.

## Requirements

- Keep `llm_step` as a strict per-step LLM decision loop.
- Keep full source and full runtime history available for execution, trace
  files, metrics, and post-run audit artifacts.
- Remove redundant full-source and duplicated trace payloads from action
  prompts.
- Ensure failed or patchable units can still expose enough source for repair.
- Add tests that prevent regressions toward unbounded prompt growth.

## Proposed Approach

Use compact prompt views for `llm_step` action selection:

1. Send compact region and group maps instead of full `to_dict()` payloads.
   Each entry should include ids, source spans, primitive calls, defined names,
   used names, side-effect flags, and a short source preview.
2. Summarize prompt history before insertion. Keep only recent action-relevant
   fields: step id, action, unit id, status, message, exception type, reward
   before/after, task-completion before/after, and primitive calls.
3. Strip full patched program text from prompt history. In particular,
   `event.evidence.source` from successful `patch_*` and `append_recovery`
   actions must not be serialized into later action prompts.
4. Bound trace context by passing a compact trace summary with a small recent
   event budget. Avoid repeating the same trace events through `event`,
   `feedback`, and top-level history fields in the prompt view.
5. Provide targeted full-source context only when repair requires it. If the
   previous step failed or was invalid, include the failed source unit's full
   source as an explicit focused section. The runtime continues to hold the full
   source in memory for execution and patching.
6. Add an action-prompt size guard. The guard should estimate serialized prompt
   size, apply deterministic fallback compaction when needed, and record prompt
   size metadata for audit.

## Data Flow

The runtime loop still maintains the existing full structures:

- `source`
- `regions`
- `groups`
- `history`
- `RuntimeTrace`
- step metrics and artifact files

Before each `capsule_action` query, `llm_step` builds a prompt-only view:

1. Convert regions/groups to compact descriptors.
2. Summarize recent history.
3. Bound trace summary.
4. Add focused full source for the most recent failed or invalid unit, if any.
5. Apply prompt-size guard.
6. Query the LLM and execute the selected action as before.

Execution, patching, append recovery, rollback guards, reward-drop guards, and
artifact writing continue to use the full runtime state.

## Prompt Schema

Compact region entry:

```json
{
  "region_id": "region_3",
  "source_span": {"start_line": 12, "end_line": 18},
  "source_preview": "pose = get_pose(\"cube\") ...",
  "primitive_calls": ["get_pose"],
  "defined_names": ["pose"],
  "used_names": ["get_pose"]
}
```

Compact group entry:

```json
{
  "group_id": "group_2",
  "source_span": {"start_line": 8, "end_line": 18},
  "source_preview": "pose = get_pose(\"cube\") ... move_to(pose)",
  "region_ids": ["region_2", "region_3"],
  "primitive_calls": ["get_pose", "move_to"],
  "defined_names": ["pose"],
  "used_names": ["get_pose", "move_to"],
  "has_robot_side_effect": true
}
```

Summarized history entry:

```json
{
  "step_id": 4,
  "action": "run_group",
  "unit_id": "group_2",
  "status": "failed",
  "message": "object not reachable",
  "exception_type": "RuntimeError",
  "reward_before": 0.0,
  "reward_after": 0.0,
  "task_completed_before": false,
  "task_completed_after": false,
  "primitive_calls": ["move_to"]
}
```

## Configuration

Add conservative optional settings:

- `capsule_llm_step_compact_context`: default `true`.
- `capsule_action_history_max_entries`: default `4`.
- `capsule_action_trace_max_events`: default `5`.
- `capsule_action_source_preview_chars`: default `240`.
- `capsule_action_prompt_char_budget`: default `60000`.

`capsule_action_prompt_char_budget` is a fallback threshold for compacting prompt
context, not a hard truncation limit. Step metrics should record the serialized
prompt size, configured threshold, and whether the final prompt remains over
threshold after fallback.

The settings should affect only `llm_step` action prompts unless explicitly
reused by recovery prompt builders later.

## Testing

Focused tests should cover:

- `llm_step` still issues one `capsule_action` model query per runtime step.
- Compact action prompts omit full long region/group source by default.
- Compact action prompts include source previews and enough ids/spans for
  selection.
- Prompt history omits `event.evidence.source` after patch or append recovery.
- Failed or invalid units can expose focused full source in the next action
  prompt.

## Risks

Compact prompts may reduce the model's ability to choose a patch without seeing
all source. The mitigation is to include focused full source for failed or
invalid units and keep ids/spans/previews visible for normal selection.

Prompt-size guards can hide useful context if set too aggressively. Defaults
should be high enough for normal trials and only activate when repeated source or
history growth would otherwise create expensive prompts.
