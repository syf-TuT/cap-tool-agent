# LLM-Step Compact Context Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce Capsule `llm_step` action-prompt token usage while preserving strict per-step LLM action selection.

**Architecture:** Keep full runtime state in `trial.py` for execution and artifacts, but build a compact prompt-only view before each `capsule_action` query. `prompts.py` owns source previews, history summaries, focused failed-unit source, and prompt-size fallback compaction; `trial.py` owns config defaults and per-step prompt-size metrics.

**Tech Stack:** Python 3.10-3.12, pytest, existing Capsule runtime-control modules under `capx/runtime_control`, WSL Ubuntu test environment at `/home/capx/code/cap-x`.

---

## Environment Notes

Source edits happen in the Windows checkout:

```powershell
cd F:\code\cap-x
```

Tests must run in the prepared WSL project copy:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; <command>'
```

Before each WSL test run, sync the touched files from Windows to WSL. Use a
small file list for this change:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'mkdir -p /home/capx/code/cap-x/capx/runtime_control /home/capx/code/cap-x/capx/envs /home/capx/code/cap-x/tests'
```

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'cp /mnt/f/code/cap-x/capx/runtime_control/prompts.py /home/capx/code/cap-x/capx/runtime_control/prompts.py'
```

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'cp /mnt/f/code/cap-x/capx/envs/trial.py /home/capx/code/cap-x/capx/envs/trial.py'
```

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'cp /mnt/f/code/cap-x/tests/test_runtime_control_prompts.py /home/capx/code/cap-x/tests/test_runtime_control_prompts.py'
```

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'cp /mnt/f/code/cap-x/tests/test_runtime_control_trial_loop.py /home/capx/code/cap-x/tests/test_runtime_control_trial_loop.py'
```

---

### Task 1: Add Compact Prompt Unit Tests

**Files:**
- Modify: `tests/test_runtime_control_prompts.py`
- Test: `tests/test_runtime_control_prompts.py`

**Step 1: Write the failing tests**

Add tests near the existing `build_capsule_prompt` tests:

```python
def test_capsule_prompt_compact_context_omits_full_region_and_group_source():
    long_region_source = "x = 1\n" + "\n".join(f"value_{idx} = {idx}" for idx in range(80))
    long_group_source = long_region_source + "\nmove_to(value_79)"

    prompt = build_capsule_prompt(
        task="stack cubes",
        regions=[
            CodeRegion(
                region_id="region_1",
                start_line=1,
                end_line=81,
                source=long_region_source,
            )
        ],
        groups=[
            CodeRegionGroup(
                group_id="group_1",
                start_line=1,
                end_line=82,
                source=long_group_source,
                region_ids=["region_1"],
                primitive_calls=["move_to"],
                defined_names=["x", "value_79"],
                used_names=["move_to"],
                has_robot_side_effect=True,
            )
        ],
        history=[],
        trace_summary={},
        compact_context=True,
        source_preview_chars=80,
    )

    text = prompt[1]["content"][0]["text"]

    assert "Compact generated code regions" in text
    assert "Compact effect-bounded execution units" in text
    assert "source_preview" in text
    assert "value_79" in text
    assert long_region_source not in text
    assert long_group_source not in text
```

```python
def test_capsule_prompt_compact_history_strips_full_patched_source():
    patched_source = "\n".join(f"line_{idx} = {idx}" for idx in range(120))
    history = [
        {
            "step_id": 1,
            "action": {
                "action": "patch_group",
                "args": {"group_id": "group_1", "source": "replacement"},
            },
            "event": {
                "action": "patch_group",
                "status": "success",
                "region_id": "group_1",
                "evidence": {"source": patched_source},
            },
            "feedback": {
                "status": "success",
                "region_id": "group_1",
                "evidence": {"trace_events": [{"name": "move_to"}]},
            },
            "trace_events": [{"name": "move_to"}],
            "state_before": {"reward": 0.0, "task_completed": False},
            "state_after": {"reward": 0.1, "task_completed": False},
        }
    ]

    prompt = build_capsule_prompt(
        task="stack cubes",
        regions=[CodeRegion(region_id="region_1", start_line=1, end_line=1, source="x = 1")],
        history=history,
        trace_summary={},
        compact_context=True,
    )

    text = prompt[1]["content"][0]["text"]

    assert "Recent runtime history summary" in text
    assert patched_source not in text
    assert "reward_before" in text
    assert "reward_after" in text
    assert "primitive_calls" in text
```

```python
def test_capsule_prompt_compact_context_includes_focused_failed_unit_source():
    failed_source = 'pose = get_pose("cube")\nmove_to(pose)'
    prompt = build_capsule_prompt(
        task="stack cubes",
        regions=[
            CodeRegion(
                region_id="region_1",
                start_line=1,
                end_line=2,
                source=failed_source,
            )
        ],
        groups=[
            CodeRegionGroup(
                group_id="group_1",
                start_line=1,
                end_line=2,
                source=failed_source,
                region_ids=["region_1"],
                primitive_calls=["get_pose", "move_to"],
                defined_names=["pose"],
                used_names=["get_pose", "move_to"],
                has_robot_side_effect=True,
            )
        ],
        history=[
            {
                "step_id": 1,
                "action": {"action": "run_group", "args": {"group_id": "group_1"}},
                "event": {
                    "action": "run_group",
                    "status": "failed",
                    "region_id": "group_1",
                    "message": "boom",
                    "evidence": {"exception_type": "RuntimeError"},
                },
            }
        ],
        trace_summary={},
        compact_context=True,
        focused_source_max_units=1,
    )

    text = prompt[1]["content"][0]["text"]

    assert "Focused source for recent failed or invalid units" in text
    assert failed_source in text
```

**Step 2: Sync and run tests to verify failure**

Sync `tests/test_runtime_control_prompts.py` to WSL, then run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_prompts.py::test_capsule_prompt_compact_context_omits_full_region_and_group_source tests/test_runtime_control_prompts.py::test_capsule_prompt_compact_history_strips_full_patched_source tests/test_runtime_control_prompts.py::test_capsule_prompt_compact_context_includes_focused_failed_unit_source -q'
```

Expected: FAIL because `build_capsule_prompt` does not accept compact-context
arguments yet.

**Step 3: Commit tests**

Do not commit failing tests separately unless the team wants strict red-green
history. Otherwise keep them staged for the implementation commit in Task 2.

---

### Task 2: Implement Compact Prompt Builders

**Files:**
- Modify: `capx/runtime_control/prompts.py`
- Test: `tests/test_runtime_control_prompts.py`

**Step 1: Extend `build_capsule_prompt` signature**

Add keyword-only parameters with backwards-compatible defaults:

```python
def build_capsule_prompt(
    *,
    task: str,
    regions: list[CodeRegion],
    groups: list[CodeRegionGroup] | None = None,
    history: list[dict[str, Any]],
    trace_summary: dict[str, Any],
    recovery_observation_functions: set[str] | None = None,
    compact_context: bool = False,
    history_max_entries: int = 8,
    trace_max_events: int = 8,
    source_preview_chars: int = 240,
    focused_source_max_units: int = 0,
    prompt_char_budget: int | None = None,
) -> list[dict[str, Any]]:
```

Keep current behavior when `compact_context=False`.

**Step 2: Add source preview helper**

Add below `summarize_terminal_state_for_recovery` or near other private helpers:

```python
def _source_preview(source: str, *, max_chars: int) -> str:
    normalized = " ".join(source.strip().split())
    if max_chars <= 0 or len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 4)].rstrip() + " ..."
```

**Step 3: Add compact region and group helpers**

```python
def _compact_region_for_prompt(region: CodeRegion, *, source_preview_chars: int) -> dict[str, Any]:
    data: dict[str, Any] = {
        "region_id": region.region_id,
        "source_span": {"start_line": region.start_line, "end_line": region.end_line},
        "source_preview": _source_preview(region.source, max_chars=source_preview_chars),
    }
    return data
```

```python
def _compact_group_for_prompt(
    group: CodeRegionGroup, *, source_preview_chars: int
) -> dict[str, Any]:
    return {
        "group_id": group.group_id,
        "source_span": {"start_line": group.start_line, "end_line": group.end_line},
        "source_preview": _source_preview(group.source, max_chars=source_preview_chars),
        "region_ids": list(group.region_ids),
        "primitive_calls": list(group.primitive_calls),
        "defined_names": list(group.defined_names),
        "used_names": list(group.used_names),
        "has_robot_side_effect": group.has_robot_side_effect,
    }
```

Do not attempt AST analysis for `CodeRegion` here; existing `CodeRegion` does
not carry primitive/name metadata.

**Step 4: Add action and history summarizers**

```python
def _action_unit_id(action: dict[str, Any], event: dict[str, Any], feedback: dict[str, Any]) -> str | None:
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    return (
        args.get("group_id")
        or args.get("region_id")
        or event.get("region_id")
        or feedback.get("region_id")
    )
```

```python
def _history_state_value(entry: dict[str, Any], state_key: str, value_key: str) -> Any:
    state = entry.get(state_key)
    if isinstance(state, dict):
        return state.get(value_key)
    return None
```

```python
def _primitive_calls_from_history(entry: dict[str, Any]) -> list[str]:
    feedback = entry.get("feedback")
    if isinstance(feedback, dict):
        evidence = feedback.get("evidence")
        if isinstance(evidence, dict) and isinstance(evidence.get("primitive_calls"), list):
            return [str(name) for name in evidence["primitive_calls"]]
    trace_events = entry.get("trace_events")
    if isinstance(trace_events, list):
        return [
            str(event["name"])
            for event in trace_events
            if isinstance(event, dict) and "name" in event
        ]
    return []
```

```python
def _summarize_history_for_prompt(
    history: list[dict[str, Any]], *, max_entries: int
) -> list[dict[str, Any]]:
    bounded = history[-max(0, int(max_entries)) :] if max_entries else []
    summaries: list[dict[str, Any]] = []
    for entry in bounded:
        action = entry.get("action") if isinstance(entry.get("action"), dict) else {}
        event = entry.get("event") if isinstance(entry.get("event"), dict) else {}
        feedback = entry.get("feedback") if isinstance(entry.get("feedback"), dict) else {}
        evidence = event.get("evidence") if isinstance(event.get("evidence"), dict) else {}
        summary = {
            "step_id": entry.get("step_id"),
            "action": action.get("action") or event.get("action"),
            "unit_id": _action_unit_id(action, event, feedback),
            "status": event.get("status") or feedback.get("status"),
            "message": event.get("message") or feedback.get("message"),
            "exception_type": evidence.get("exception_type"),
            "reward_before": _history_state_value(entry, "state_before", "reward"),
            "reward_after": _history_state_value(entry, "state_after", "reward"),
            "task_completed_before": _history_state_value(
                entry, "state_before", "task_completed"
            ),
            "task_completed_after": _history_state_value(
                entry, "state_after", "task_completed"
            ),
            "primitive_calls": _primitive_calls_from_history(entry),
        }
        summaries.append({key: value for key, value in summary.items() if value not in (None, [], "")})
    return summaries
```

**Step 5: Add focused failed-source helper**

```python
def _focused_failed_units_for_prompt(
    *,
    history: list[dict[str, Any]],
    regions: list[CodeRegion],
    groups: list[CodeRegionGroup] | None,
    max_units: int,
) -> list[dict[str, Any]]:
    if max_units <= 0:
        return []
    region_by_id = {region.region_id: region for region in regions}
    group_by_id = {group.group_id: group for group in groups or []}
    focused: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in reversed(history):
        event = entry.get("event") if isinstance(entry.get("event"), dict) else {}
        feedback = entry.get("feedback") if isinstance(entry.get("feedback"), dict) else {}
        status = event.get("status") or feedback.get("status")
        if status not in {"failed", "invalid"}:
            continue
        action = entry.get("action") if isinstance(entry.get("action"), dict) else {}
        unit_id = _action_unit_id(action, event, feedback)
        if not isinstance(unit_id, str) or unit_id in seen:
            continue
        unit = group_by_id.get(unit_id) or region_by_id.get(unit_id)
        if unit is None:
            continue
        focused.append(unit.to_dict())
        seen.add(unit_id)
        if len(focused) >= max_units:
            break
    return list(reversed(focused))
```

**Step 6: Add compact prompt payload selection**

Inside `build_capsule_prompt`, replace the current `region_data`, `group_data`,
and history/trace insertion with branches:

```python
    if compact_context:
        region_data = [
            _compact_region_for_prompt(region, source_preview_chars=source_preview_chars)
            for region in regions
        ]
        group_data = [
            _compact_group_for_prompt(group, source_preview_chars=source_preview_chars)
            for group in groups or []
        ]
        history_data = _summarize_history_for_prompt(history, max_entries=history_max_entries)
        trace_data = _bound_trace_summary(trace_summary, max_events=trace_max_events)
        focused_source_data = _focused_failed_units_for_prompt(
            history=history,
            regions=regions,
            groups=groups,
            max_units=focused_source_max_units,
        )
        region_heading = "Compact generated code regions"
        group_heading = "Compact effect-bounded execution units (preferred run_group targets)"
        history_heading = "Recent runtime history summary"
        trace_heading = "Recent primitive call trace summary"
    else:
        region_data = [region.to_dict() for region in regions]
        group_data = [group.to_dict() for group in groups or []]
        history_data = history[-8:]
        trace_data = trace_summary
        focused_source_data = []
        region_heading = "Generated code regions"
        group_heading = "Effect-bounded execution units (preferred run_group targets)"
        history_heading = "Recent runtime history"
        trace_heading = "Primitive call trace summary"
```

Add the focused-source block only when non-empty:

```python
    focused_source_text = ""
    if focused_source_data:
        focused_source_text = (
            "Focused source for recent failed or invalid units:\n"
            f"{json.dumps(focused_source_data, indent=2, default=str)}\n\n"
        )
```

Then use the headings and payloads in `prompt_text`:

```python
        f"{group_heading}:\n"
        f"{json.dumps(group_data, indent=2, default=str)}\n\n"
```

Only emit the group block when groups are present, preserving current behavior
for non-group runs.

**Step 7: Add prompt-size fallback helper**

Keep this deterministic and simple:

```python
def _prompt_text_over_budget(prompt_text: str, prompt_char_budget: int | None) -> bool:
    return prompt_char_budget is not None and prompt_char_budget > 0 and len(prompt_text) > prompt_char_budget
```

If the first compact prompt is over budget, rebuild once with:

- `history_max_entries = min(history_max_entries, 2)`
- `trace_max_events = min(trace_max_events, 2)`
- `source_preview_chars = min(source_preview_chars, 80)`
- `focused_source_max_units = min(focused_source_max_units, 1)`

Do not apply this fallback in non-compact mode.

**Step 8: Sync and run prompt tests**

Sync `capx/runtime_control/prompts.py` and `tests/test_runtime_control_prompts.py`
to WSL, then run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_prompts.py -q'
```

Expected: PASS.

**Step 9: Commit**

```bash
git add capx/runtime_control/prompts.py tests/test_runtime_control_prompts.py
git commit -m "Compact Capsule llm-step action prompts"
```

---

### Task 3: Wire Compact Context Into `llm_step`

**Files:**
- Modify: `capx/envs/trial.py`
- Test: `tests/test_runtime_control_trial_loop.py`

**Step 1: Add failing trial-loop test for compact default**

Add near existing `llm_step` tests:

```python
def test_capsule_llm_step_uses_compact_action_prompt_by_default(tmp_path, monkeypatch):
    prompts = []
    long_source = "\n".join(f"value_{idx} = {idx}" for idx in range(100))

    def fake_query_model(args, prompt):
        prompts.append(prompt)
        return {"content": '{"action": "finish", "args": {}}'}

    monkeypatch.setattr("capx.envs.trial._query_model", fake_query_model)

    summary = trial_module._run_capsule_llm_step_loop(
        _CapsuleDummyEnv(),
        trial=0,
        args=_args(),
        config={
            "output_dir": str(tmp_path),
            "capsule_control_mode": "llm_step",
            "max_capsule_steps": 1,
        },
        initial_code=long_source,
    )

    assert summary.num_finishes == 1
    text = prompts[0][1]["content"][0]["text"]
    assert "Compact generated code regions" in text
    assert long_source not in text
```

Adjust `_CapsuleDummyEnv`, `_args`, or local test helper names to match the
existing file. If no suitable helper supports `finish`, create the smallest
local fake env using patterns already present in `tests/test_runtime_control_trial_loop.py`.

**Step 2: Add failing test for disabling compact context**

```python
def test_capsule_llm_step_can_disable_compact_action_prompt(tmp_path, monkeypatch):
    prompts = []
    source = "x = 1"

    def fake_query_model(args, prompt):
        prompts.append(prompt)
        return {"content": '{"action": "finish", "args": {}}'}

    monkeypatch.setattr("capx.envs.trial._query_model", fake_query_model)

    trial_module._run_capsule_llm_step_loop(
        _CapsuleDummyEnv(),
        trial=0,
        args=_args(),
        config={
            "output_dir": str(tmp_path),
            "capsule_control_mode": "llm_step",
            "max_capsule_steps": 1,
            "capsule_llm_step_compact_context": False,
        },
        initial_code=source,
    )

    text = prompts[0][1]["content"][0]["text"]
    assert "Generated code regions" in text
    assert "Compact generated code regions" not in text
```

**Step 3: Run tests to verify failure**

Sync `tests/test_runtime_control_trial_loop.py` to WSL, then run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_trial_loop.py::test_capsule_llm_step_uses_compact_action_prompt_by_default tests/test_runtime_control_trial_loop.py::test_capsule_llm_step_can_disable_compact_action_prompt -q'
```

Expected: FAIL because `trial.py` does not pass compact-context settings.

**Step 4: Wire config into `_run_capsule_llm_step_loop`**

Before the loop, add:

```python
    llm_step_compact_context = bool(config.get("capsule_llm_step_compact_context", True))
    action_history_max_entries = int(config.get("capsule_action_history_max_entries", 4))
    action_trace_max_events = int(config.get("capsule_action_trace_max_events", 5))
    action_source_preview_chars = int(config.get("capsule_action_source_preview_chars", 240))
    action_prompt_char_budget = int(config.get("capsule_action_prompt_char_budget", 60000))
```

Then pass these into `build_capsule_prompt`:

```python
            compact_context=llm_step_compact_context,
            history_max_entries=action_history_max_entries,
            trace_max_events=action_trace_max_events,
            source_preview_chars=action_source_preview_chars,
            focused_source_max_units=1 if llm_step_compact_context else 0,
            prompt_char_budget=action_prompt_char_budget if llm_step_compact_context else None,
```

**Step 5: Record action prompt size in metrics**

After building the prompt:

```python
        action_prompt_chars = len(json.dumps(prompt, default=str))
```

After `_capsule_step_metric(...)` returns:

```python
        metric["action_prompt_chars"] = action_prompt_chars
        metric["action_prompt_compact_context"] = llm_step_compact_context
```

This keeps prompt-size audit data in `capsule_step_metrics_trial_XX.jsonl`
without logging prompt contents.

**Step 6: Sync and run focused trial-loop tests**

Sync `capx/envs/trial.py` and `tests/test_runtime_control_trial_loop.py` to WSL,
then run:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_trial_loop.py::test_capsule_llm_step_uses_compact_action_prompt_by_default tests/test_runtime_control_trial_loop.py::test_capsule_llm_step_can_disable_compact_action_prompt -q'
```

Expected: PASS.

**Step 7: Commit**

```bash
git add capx/envs/trial.py tests/test_runtime_control_trial_loop.py
git commit -m "Use compact context for llm-step actions"
```

---

### Task 4: Add Regression Tests for Patched Source Growth

**Files:**
- Modify: `tests/test_runtime_control_trial_loop.py`
- Test: `tests/test_runtime_control_trial_loop.py`

**Step 1: Add failing/passing regression test**

Add a test that performs a patch, then inspects the next action prompt:

```python
def test_capsule_llm_step_compact_prompt_does_not_replay_full_patched_source(
    tmp_path, monkeypatch
):
    prompts = []
    patched_source = "\n".join(f"patched_{idx} = {idx}" for idx in range(200))
    responses = iter(
        [
            {
                "content": json.dumps(
                    {
                        "action": "patch_group",
                        "args": {"group_id": "group_1", "source": patched_source},
                    }
                )
            },
            {"content": '{"action": "finish", "args": {}}'},
        ]
    )

    def fake_query_model(args, prompt):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr("capx.envs.trial._query_model", fake_query_model)

    trial_module._run_capsule_llm_step_loop(
        _CapsuleDummyEnv(),
        trial=0,
        args=_args(),
        config={
            "output_dir": str(tmp_path),
            "capsule_control_mode": "llm_step",
            "max_capsule_steps": 2,
        },
        initial_code="x = 1",
    )

    assert len(prompts) == 2
    second_prompt_text = prompts[1][1]["content"][0]["text"]
    assert patched_source not in second_prompt_text
    assert "Recent runtime history summary" in second_prompt_text
```

Adapt helper names and initial source shape to existing test utilities. If the
dummy env marks patching as invalid because no group is available, use an
initial source that creates `group_1` under the real segmenter.

**Step 2: Run regression test**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_trial_loop.py::test_capsule_llm_step_compact_prompt_does_not_replay_full_patched_source -q'
```

Expected: PASS after Tasks 2 and 3.

**Step 3: Commit**

```bash
git add tests/test_runtime_control_trial_loop.py
git commit -m "Prevent llm-step prompt source replay"
```

---

### Task 5: Run Focused Regression Suite

**Files:**
- Test only.

**Step 1: Sync all touched files**

Run the four sync commands from the Environment Notes section.

**Step 2: Run runtime prompt tests**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_prompts.py -q'
```

Expected: PASS.

**Step 3: Run runtime trial-loop tests**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync pytest tests/test_runtime_control_trial_loop.py -q'
```

Expected: PASS.

**Step 4: Run lint on touched Python files if available**

```powershell
wsl.exe -d Ubuntu-22.04 -- bash --noprofile --norc -lc 'export PATH=/home/capx/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; cd /home/capx/code/cap-x; uv run --no-sync ruff check capx/runtime_control/prompts.py capx/envs/trial.py tests/test_runtime_control_prompts.py tests/test_runtime_control_trial_loop.py'
```

Expected: PASS.

**Step 5: Final commit if needed**

If verification fixes changed files:

```bash
git add capx/runtime_control/prompts.py capx/envs/trial.py tests/test_runtime_control_prompts.py tests/test_runtime_control_trial_loop.py
git commit -m "Verify llm-step compact context"
```

---

## Completion Criteria

- `llm_step` still queries the LLM for each runtime action step.
- Default `llm_step` action prompts use compact code and history views.
- Full source remains available for execution, patches, saved code, trace files,
  and focused failed-unit repair context.
- `auto_forward` behavior remains unchanged.
- Focused prompt and trial-loop tests pass in WSL.
