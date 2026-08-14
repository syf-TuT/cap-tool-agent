import json

import capx.runtime_control.prompts as prompt_module
from capx.runtime_control.prompts import (
    _summarize_history_for_prompt,
    build_capsule_prompt,
    parse_runtime_action_response,
)
from capx.runtime_control.schema import CodeRegion, CodeRegionGroup


def test_parse_runtime_action_response():
    action = parse_runtime_action_response(
        '{"action": "run_region", "args": {"region_id": "region_1"}}'
    )

    assert action.action == "run_region"


def test_capsule_prompt_excludes_robot_tool_list():
    prompt = build_capsule_prompt(
        task="stack cubes",
        regions=[CodeRegion(region_id="region_1", start_line=1, end_line=1, source="x = 1")],
        history=[],
        trace_summary={},
    )
    text = str(prompt)

    assert "run_region" in text
    assert "solve_ik" not in text
    assert "move_to_joints" not in text


def test_capsule_prompt_excludes_removed_trace_inspection_action():
    prompt = build_capsule_prompt(
        task="stack cubes",
        regions=[CodeRegion(region_id="region_1", start_line=1, end_line=1, source="x = 1")],
        history=[],
        trace_summary={"event_count": 1},
    )

    text = str(prompt)

    assert "inspect_" "trace" not in text
    assert '"event_count": 1' in text


def test_capsule_prompt_forbids_callable_reflection_and_dynamic_access():
    prompt = build_capsule_prompt(
        task="stack cubes",
        regions=[
            CodeRegion(
                region_id="region_1",
                start_line=1,
                end_line=1,
                source="close_gripper()",
            )
        ],
        history=[],
        trace_summary={},
    )

    text = str(prompt)

    assert "Do not use callable introspection" in text
    assert "__closure__" in text
    assert "globals()/eval()/exec()" in text


def test_capsule_prompt_documents_patch_region_source_schema():
    prompt = build_capsule_prompt(
        task="stack cubes",
        regions=[CodeRegion(region_id="region_1", start_line=1, end_line=1, source="x = 1")],
        history=[],
        trace_summary={},
    )
    text = str(prompt)

    assert '{"action": "patch_region", "args": {"region_id": "region_1", "source":' in text
    assert '{"action": "inspect_variables", "args": {"names": ["variable_name"]}}' in text
    assert "Do not pass region_id to inspect_variables." in text
    assert "Do not use new_source or patch for patch_region replacement text." in text


def test_capsule_prompt_prefers_group_actions_when_groups_are_available():
    prompt = build_capsule_prompt(
        task="stack cubes",
        regions=[CodeRegion(region_id="region_1", start_line=1, end_line=1, source="x = 1")],
        groups=[
            CodeRegionGroup(
                group_id="group_1",
                start_line=1,
                end_line=2,
                source="x = 1\nmove_to_joints(x)",
                region_ids=["region_1", "region_2"],
                primitive_calls=["move_to_joints"],
                defined_names=["x"],
                used_names=["move_to_joints"],
                has_robot_side_effect=True,
            )
        ],
        history=[],
        trace_summary={},
    )
    text = str(prompt)

    assert "Effect-bounded execution units" in text
    assert "semantic source chunk" not in text
    assert '{"action": "run_group", "args": {"group_id": "group_1"}}' in text
    assert '{"action": "patch_group", "args": {"group_id": "group_1", "source":' in text
    assert "Prefer run_group over run_region" in text
    assert "effect-bounded execution unit" in text


def test_capsule_prompt_marks_executed_side_effect_units_unrunnable():
    prompt = build_capsule_prompt(
        task="stack cubes",
        regions=[
            CodeRegion(
                region_id="region_1",
                start_line=1,
                end_line=1,
                source='pose = get_pose("cube")',
            ),
            CodeRegion(
                region_id="region_2",
                start_line=2,
                end_line=2,
                source="move_to(pose)",
            ),
        ],
        groups=[
            CodeRegionGroup(
                group_id="group_1",
                start_line=1,
                end_line=2,
                source='pose = get_pose("cube")\nmove_to(pose)',
                region_ids=["region_1", "region_2"],
                primitive_calls=["get_pose", "move_to"],
                defined_names=["pose"],
                used_names=["get_pose", "move_to"],
                has_robot_side_effect=True,
            ),
            CodeRegionGroup(
                group_id="group_2",
                start_line=3,
                end_line=3,
                source='move_to("next")',
                region_ids=["region_3"],
                primitive_calls=["move_to"],
                defined_names=[],
                used_names=["move_to"],
                has_robot_side_effect=True,
            ),
        ],
        history=[],
        trace_summary={},
        side_effect_ledger={
            "executed_side_effect_groups": ["group_1"],
            "executed_side_effect_regions": ["region_2"],
        },
        compact_context=True,
    )

    text = prompt[1]["content"][0]["text"]

    assert "Side-effect execution ledger" in text
    assert '"executed_side_effect_groups": [\n    "group_1"\n  ]' in text
    assert '"unit_id": "group_1"' in text
    assert '"execution_state": "executed_side_effect"' in text
    assert '"run_allowed": false' in text
    assert '"patch_allowed": false' in text
    assert "Do not choose run_group, run_region, patch_group, or patch_region" in text
    assert '{"action": "run_group", "args": {"group_id": "group_1"}}' not in text
    assert '{"action": "run_group", "args": {"group_id": "group_2"}}' in text


def test_capsule_prompt_uses_no_rollback_forward_recovery_semantics():
    prompt = build_capsule_prompt(
        task="stack cubes",
        regions=[CodeRegion(region_id="region_1", start_line=1, end_line=1, source="x = 1")],
        history=[],
        trace_summary={},
    )
    text = str(prompt)

    assert "rollback_to_checkpoint" not in text
    assert "Rollback is unavailable" in text
    assert "fresh observation" in text
    assert "current physical state" in text


def test_capsule_prompt_documents_append_recovery():
    prompt = build_capsule_prompt(
        task="stack cubes",
        regions=[CodeRegion(region_id="region_1", start_line=1, end_line=1, source="x = 1")],
        history=[],
        trace_summary={},
    )
    text = str(prompt)

    assert "append_recovery" in text
    assert '{"action": "append_recovery", "args": {"source":' in text
    assert "get_observation()" in text
    assert "current physical state" in text


def test_capsule_prompt_documents_task_specific_recovery_observation_functions():
    prompt = build_capsule_prompt(
        task="lift pot",
        regions=[CodeRegion(region_id="region_1", start_line=1, end_line=1, source="x = 1")],
        history=[],
        trace_summary={},
        recovery_observation_functions={"get_handle0_pos", "get_handle1_pos"},
    )
    text = str(prompt)

    assert "get_handle0_pos()" in text
    assert "get_handle1_pos()" in text
    assert "must call at least one" in text


def test_capsule_prompt_marks_recovery_unavailable_without_fresh_state_function():
    prompt = build_capsule_prompt(
        task="unknown task",
        regions=[CodeRegion(region_id="region_1", start_line=1, end_line=1, source="x = 1")],
        history=[],
        trace_summary={},
        recovery_observation_functions=set(),
    )

    text = prompt[1]["content"][0]["text"]
    allowed_actions = next(line for line in text.splitlines() if line.startswith("Allowed actions:"))

    assert "append_recovery is unavailable" in text
    assert "append_recovery" not in allowed_actions


def _json_string_payload(value: str) -> str:
    return json.dumps(value)[1:-1]


def _prompt_json_section(text: str, heading: str, next_heading: str) -> dict:
    section = text.split(f"{heading}:\n", 1)[1].split(f"\n\n{next_heading}:", 1)[0]
    return json.loads(section)


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
    assert "region_1" in text
    assert "group_1" in text
    assert "value_79" in text
    assert long_region_source not in text
    assert _json_string_payload(long_region_source) not in text
    assert long_group_source not in text
    assert _json_string_payload(long_group_source) not in text


def test_capsule_prompt_compact_history_strips_full_patched_source():
    patched_source = "\n".join(f"line_{idx} = {idx}" for idx in range(120))
    history = [
        {
            "step_id": 1,
            "action": {
                "action": "patch_group",
                "args": {"group_id": "group_1", "source": patched_source},
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
    assert _json_string_payload(patched_source) not in text
    assert "reward_before" in text
    assert "reward_after" in text
    assert "primitive_calls" in text
    assert "move_to" in text


def test_capsule_prompt_compact_history_prefers_warning_feedback_status():
    history = [
        {
            "step_id": 1,
            "action": {"action": "run_group", "args": {"group_id": "group_1"}},
            "event": {
                "action": "run_group",
                "status": "success",
                "region_id": "group_1",
                "message": "execution succeeded",
            },
            "feedback": {
                "status": "warning",
                "region_id": "group_1",
                "message": "no progress observed",
            },
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

    assert '"status": "warning"' in text
    assert '"event_status": "success"' in text
    assert '"feedback_status": "warning"' in text


def test_capsule_prompt_compact_history_keeps_sparse_terminal_evidence_after_fallback():
    history = [
        {
            "step_id": 1,
            "action": {"action": "run_group", "args": {"group_id": "group_1"}},
            "event": {
                "action": "run_group",
                "status": "success",
                "region_id": "group_1",
                "evidence": {
                    "source": "FULL_SOURCE_MUST_NOT_APPEAR",
                    "trace_events": [{"name": "FULL_TRACE_MUST_NOT_APPEAR"}],
                },
            },
            "feedback": {
                "status": "warning",
                "evidence": {
                    "terminal_progress_unverified": True,
                    "progress_mode": "sparse_terminal",
                    "unrelated": "UNRELATED_EVIDENCE_MUST_NOT_APPEAR",
                },
            },
        }
    ]

    prompt = build_capsule_prompt(
        task="stack cubes",
        regions=[CodeRegion(region_id="region_1", start_line=1, end_line=1, source="x = 1")],
        history=history,
        trace_summary={},
        compact_context=True,
        prompt_char_budget=1,
    )

    text = prompt[1]["content"][0]["text"]

    assert '"terminal_progress_unverified": true' in text
    assert '"progress_mode": "sparse_terminal"' in text
    assert "FULL_SOURCE_MUST_NOT_APPEAR" not in text
    assert "FULL_TRACE_MUST_NOT_APPEAR" not in text
    assert "UNRELATED_EVIDENCE_MUST_NOT_APPEAR" not in text


def test_capsule_prompt_compact_history_bounds_inspected_variable_summaries():
    variable_evidence = {
        "array": {
            "type": "ndarray",
            "shape": [1000, 1000],
            "value": list(range(200)),
            "source": "INSPECT_SOURCE_MUST_NOT_APPEAR",
        },
        "description": {"type": "str", "repr": "R" * 2000},
        "binary": {
            "type": "str",
            "repr": "data:image/png;base64," + "A" * 2000,
        },
        **{
            f"extra_{index}": {"type": "str", "repr": f"value_{index}"}
            for index in range(10)
        },
    }
    history = [
        {
            "step_id": 3,
            "action": {"action": "inspect_variables", "args": {"names": ["array"]}},
            "event": {
                "action": "inspect_variables",
                "status": "success",
                "evidence": variable_evidence,
            },
        },
        {
            "step_id": 4,
            "action": {"action": "run_region", "args": {"region_id": "region_1"}},
            "event": {
                "action": "run_region",
                "status": "success",
                "evidence": {"ordinary": "ORDINARY_EVIDENCE_MUST_NOT_APPEAR"},
            },
        },
    ]

    prompt = build_capsule_prompt(
        task="stack cubes",
        regions=[CodeRegion(region_id="region_1", start_line=1, end_line=1, source="x = 1")],
        history=history,
        trace_summary={},
        compact_context=True,
        prompt_char_budget=1,
    )

    text = prompt[1]["content"][0]["text"]

    assert '"inspected_variables"' in text
    assert '"array"' in text
    assert '"type": "ndarray"' in text
    assert "INSPECT_SOURCE_MUST_NOT_APPEAR" not in text
    assert "ORDINARY_EVIDENCE_MUST_NOT_APPEAR" not in text
    assert "data:image" not in text
    assert "base64," not in text
    assert "R" * 300 not in text
    assert len(text) < 20000


def test_compact_history_only_redacts_explicit_binary_markers():
    history = [
        {"step_id": 1, "event": {"message": "metadata: safe status"}},
        {"step_id": 2, "event": {"message": "literal base64,short remains"}},
        {
            "step_id": 3,
            "event": {"message": "image data:image/png;base64," + "A" * 100},
        },
        {
            "step_id": 4,
            "event": {"message": "blob BASE64, AAAAAAAA" + "B" * 100},
        },
    ]

    summary = _summarize_history_for_prompt(history, max_entries=4)

    assert summary[0]["message"] == "metadata: safe status"
    assert summary[1]["message"] == "literal base64,short remains"
    assert summary[2]["message"] == "image <redacted binary data>"
    assert summary[3]["message"] == "blob <redacted binary data>"
    assert "A" * 20 not in json.dumps(summary)
    assert "B" * 20 not in json.dumps(summary)


def test_compact_history_bounds_all_text_and_primitive_call_lists():
    payload = "data:image/png;base64," + "A" * 10000
    history = [
        {
            "step_id": "S" * 1000,
            "action": {
                "action": "run_region" + "X" * 1000,
                "args": {"group_id": "G" * 1000},
            },
            "event": {
                "status": "failed" + "Y" * 1000,
                "message": "event failed: " + payload,
                "evidence": {
                    "exception_type": "VeryLongError" * 1000,
                    "source": "FULL_SOURCE_MUST_NOT_APPEAR",
                },
            },
            "feedback": {
                "status": "warning" + "Z" * 1000,
                "message": "feedback failed: " + payload,
                "evidence": {
                    "primitive_calls": ["primitive_" + str(index) + "P" * 1000 for index in range(500)],
                },
            },
        }
    ]

    summary = _summarize_history_for_prompt(history, max_entries=1)[0]

    assert len(summary["step_id"]) <= 120
    assert len(summary["action"]) <= 120
    assert len(summary["unit_id"]) <= 120
    assert len(summary["status"]) <= 120
    assert len(summary["event_status"]) <= 120
    assert len(summary["feedback_status"]) <= 120
    assert len(summary["message"]) <= 240
    assert summary["message"].startswith("feedback failed:")
    assert len(summary["exception_type"]) <= 120
    assert len(summary["primitive_calls"]) <= 8
    assert all(len(name) <= 80 for name in summary["primitive_calls"])
    serialized = json.dumps(summary)
    assert "data:image" not in serialized
    assert "base64," not in serialized
    assert "FULL_SOURCE_MUST_NOT_APPEAR" not in serialized
    assert "A" * 300 not in serialized


def test_capsule_prompt_compact_second_fallback_respects_serialized_budget():
    payload = "data:image/png;base64," + "A" * 10000
    history = []
    for step_id in range(8):
        feedback = {
            "status": "warning",
            "evidence": {
                "primitive_calls": [
                    f"primitive_{step_id}_{index}_" + "P" * 1000
                    for index in range(500)
                ],
                "source": "FULL_FEEDBACK_SOURCE_MUST_NOT_APPEAR",
            },
        }
        if step_id == 6:
            feedback["message"] = "feedback failed: " + payload
        history.append(
            {
                "step_id": step_id,
                "action": {"action": "run_group", "args": {"group_id": "G" * 1000}},
                "event": {
                    "action": "run_group",
                    "status": "success",
                    "message": "event failed: " + payload,
                    "evidence": {
                        "exception_type": "VeryLongError" * 1000,
                        "source": "FULL_EVENT_SOURCE_MUST_NOT_APPEAR",
                    },
                },
                "feedback": feedback,
            }
        )

    prompt = build_capsule_prompt(
        task="stack cubes",
        regions=[CodeRegion(region_id="region_1", start_line=1, end_line=1, source="x = 1")],
        history=history,
        trace_summary={},
        compact_context=True,
        prompt_char_budget=6500,
    )

    serialized = json.dumps(prompt, default=str)

    assert len(serialized) <= 6500
    assert "feedback failed:" in serialized
    assert "event failed:" in serialized
    assert "data:image" not in serialized
    assert "base64," not in serialized
    assert "FULL_FEEDBACK_SOURCE_MUST_NOT_APPEAR" not in serialized
    assert "FULL_EVENT_SOURCE_MUST_NOT_APPEAR" not in serialized
    assert "A" * 300 not in serialized


def test_compact_inspected_variables_share_aggregate_budget():
    payload = "data:image/png;base64," + "A" * 10000
    leaf = [payload for _ in range(32)]
    level_two = [leaf for _ in range(32)]
    nested_bomb = [level_two for _ in range(32)]
    history = [
        {
            "step_id": 9,
            "action": {
                "action": "inspect_variables",
                "args": {"names": ["a_short", "z_bomb"]},
            },
            "event": {
                "action": "inspect_variables",
                "status": "success",
                "evidence": {
                    "z_bomb": {"type": "list", "value": nested_bomb},
                    "a_short": {"type": "int", "repr": "7"},
                },
            },
            "feedback": {
                "status": "warning",
                "evidence": {
                    "terminal_progress_unverified": True,
                    "progress_mode": "sparse_terminal",
                },
            },
        }
    ]

    prompt = build_capsule_prompt(
        task="stack cubes",
        regions=[CodeRegion(region_id="region_1", start_line=1, end_line=1, source="x = 1")],
        history=history,
        trace_summary={},
        compact_context=True,
        prompt_char_budget=6500,
    )

    serialized = json.dumps(prompt, default=str)
    text = prompt[1]["content"][0]["text"]

    assert len(serialized) <= 6500
    assert '"terminal_progress_unverified": true' in text
    assert '"progress_mode": "sparse_terminal"' in text
    assert '"a_short"' in text
    assert '"repr": "7"' in text
    assert "data:image" not in serialized
    assert "base64," not in serialized
    assert "A" * 300 not in serialized


def test_compact_inspected_variable_mapping_order_is_deterministic():
    def history_with_evidence(evidence):
        return [
            {
                "step_id": 1,
                "action": {"action": "inspect_variables", "args": {"names": list(evidence)}},
                "event": {
                    "action": "inspect_variables",
                    "status": "success",
                    "evidence": evidence,
                },
            }
        ]

    ascending = {
        "alpha": {"type": "dict", "value": {"a": 1, "b": 2}},
        "beta": {"type": "dict", "value": {"x": 3, "y": 4}},
    }
    descending = {
        "beta": {"value": {"y": 4, "x": 3}, "type": "dict"},
        "alpha": {"value": {"b": 2, "a": 1}, "type": "dict"},
    }

    first = _summarize_history_for_prompt(history_with_evidence(ascending), max_entries=1)
    second = _summarize_history_for_prompt(history_with_evidence(descending), max_entries=1)

    assert json.dumps(first) == json.dumps(second)


def test_compact_prompt_uses_third_fallback_without_dropping_safety_context(monkeypatch):
    calls = []
    original_build = prompt_module._build_capsule_prompt_text

    def tracking_build(**kwargs):
        calls.append(kwargs)
        return original_build(**kwargs)

    monkeypatch.setattr(prompt_module, "_build_capsule_prompt_text", tracking_build)
    history = []
    for step_id in range(2):
        history.append(
            {
                "step_id": step_id,
                "action": {"action": "inspect_variables", "args": {"names": ["a_short"]}},
                "event": {
                    "action": "inspect_variables",
                    "status": "success",
                    "evidence": {"a_short": {"type": "int", "repr": "7"}},
                },
                "feedback": {
                    "status": "warning",
                    "message": "M" * 1000,
                    "evidence": {
                        "terminal_progress_unverified": True,
                        "progress_mode": "sparse_terminal",
                        "primitive_calls": ["P" * 1000 for _ in range(100)],
                    },
                },
            }
        )

    prompt = build_capsule_prompt(
        task="stack cubes",
        regions=[CodeRegion(region_id="region_1", start_line=1, end_line=1, source="x = 1")],
        history=history,
        trace_summary={},
        contract_violations=[
            {
                "code": "effectful_helper",
                "message": "Patch this violation before running effects",
            }
        ],
        strict_subset=True,
        compact_context=True,
        prompt_char_budget=6500,
    )

    serialized = json.dumps(prompt, default=str)
    text = prompt[1]["content"][0]["text"]

    assert len(calls) == 3
    assert calls[-1]["history_max_entries"] == 1
    assert len(serialized) <= 6500
    assert "Strict Python subset" in text
    assert "Capsule-ready program contract violations" in text
    assert "effectful_helper" in text
    assert '"terminal_progress_unverified": true' in text
    assert '"a_short"' in text


def test_capsule_prompt_compact_context_includes_focused_failed_unit_source():
    unique_tail = "focused_tail_marker_final_line = 8675309"
    failed_source = "\n".join(
        [
            'pose = get_pose("cube")',
            "adjusted_pose = pose",
            "move_to(adjusted_pose)",
            unique_tail,
        ]
    )
    prompt = build_capsule_prompt(
        task="stack cubes",
        regions=[
            CodeRegion(
                region_id="region_1",
                start_line=1,
                end_line=4,
                source=failed_source,
            )
        ],
        groups=[
            CodeRegionGroup(
                group_id="group_1",
                start_line=1,
                end_line=4,
                source=failed_source,
                region_ids=["region_1"],
                primitive_calls=["get_pose", "move_to"],
                defined_names=["adjusted_pose", "focused_tail_marker_final_line", "pose"],
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
        source_preview_chars=8,
    )

    text = prompt[1]["content"][0]["text"]

    assert "Focused source for recent failed or invalid units" in text
    assert unique_tail in text
    assert failed_source in text or _json_string_payload(failed_source) in text


def test_capsule_prompt_compact_focused_source_ignores_resolved_failure():
    patched_body = "\n".join(f"resolved_patch_line_{idx} = {idx}" for idx in range(200))
    patched_source = f"PATCHED_SOURCE = {patched_body!r}\n"
    prompt = build_capsule_prompt(
        task="stack cubes",
        regions=[
            CodeRegion(
                region_id="region_1",
                start_line=1,
                end_line=1,
                source=patched_source,
            )
        ],
        groups=[
            CodeRegionGroup(
                group_id="group_1",
                start_line=1,
                end_line=1,
                source=patched_source,
                region_ids=["region_1"],
                primitive_calls=["move_to"],
                defined_names=["PATCHED_SOURCE"],
                used_names=["move_to"],
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
                    "evidence": {"exception_type": "RuntimeError"},
                },
            },
            {
                "step_id": 2,
                "action": {
                    "action": "patch_group",
                    "args": {"group_id": "group_1", "source": patched_source},
                },
                "event": {
                    "action": "patch_group",
                    "status": "success",
                    "region_id": "group_1",
                    "evidence": {"source": patched_source},
                },
                "feedback": {"status": "success", "region_id": "group_1"},
            },
        ],
        trace_summary={},
        compact_context=True,
        focused_source_max_units=1,
        source_preview_chars=8,
    )

    text = prompt[1]["content"][0]["text"]

    assert "Recent runtime history summary" in text
    assert "patch_group" in text
    assert "success" in text
    assert "Focused source for recent failed or invalid units" not in text
    assert patched_source not in text
    assert _json_string_payload(patched_source) not in text
    assert "resolved_patch_line_199 = 199" not in text


def test_capsule_prompt_budget_fallback_omits_large_focused_failed_source():
    unique_tail = "budget_fallback_tail_marker = 424242"
    failed_source = "\n".join(
        [f"oversized_failed_line_{idx} = {idx}" for idx in range(400)] + [unique_tail]
    )

    prompt = build_capsule_prompt(
        task="stack cubes",
        regions=[
            CodeRegion(
                region_id="region_1",
                start_line=1,
                end_line=401,
                source=failed_source,
            )
        ],
        groups=[
            CodeRegionGroup(
                group_id="group_1",
                start_line=1,
                end_line=401,
                source=failed_source,
                region_ids=["region_1"],
                primitive_calls=["move_to"],
                defined_names=[],
                used_names=["move_to"],
                has_robot_side_effect=True,
            )
        ],
        history=[
            {
                "step_id": 1,
                "action": {"action": "run_group", "args": {"group_id": "group_1"}},
                "event": {"action": "run_group", "status": "failed", "region_id": "group_1"},
            }
        ],
        trace_summary={},
        compact_context=True,
        focused_source_max_units=1,
        source_preview_chars=0,
        prompt_char_budget=1,
    )

    text = prompt[1]["content"][0]["text"]

    assert "Focused source for recent failed or invalid units" not in text
    assert unique_tail not in text
    assert failed_source not in text
    assert _json_string_payload(failed_source) not in text


def test_capsule_prompt_contract_violations_survive_compact_budget_fallback():
    violations = [
        {
            "code": "effectful_helper",
            "message": "Helper 'move_cube' can execute a robot side effect",
            "source_span": {"start_line": 3, "end_line": 5},
            "region_ids": ["region_2"],
            "group_ids": ["group_1"],
            "side_effect_calls": ["move_to"],
            "helper_name": "move_cube",
        }
    ]

    prompt = build_capsule_prompt(
        task="stack cubes",
        regions=[
            CodeRegion(
                region_id="region_1",
                start_line=1,
                end_line=1,
                source="x = 1",
            )
        ],
        history=[
            {
                "step_id": index,
                "event": {"status": "failed", "message": "x" * 1000},
            }
            for index in range(10)
        ],
        trace_summary={"events": [{"name": "move_to"}] * 20},
        contract_violations=violations,
        compact_context=True,
        prompt_char_budget=1,
    )

    text = prompt[1]["content"][0]["text"]

    assert "Capsule-ready program contract violations" in text
    assert '"code": "effectful_helper"' in text
    assert "Helper 'move_cube' can execute a robot side effect" in text
    assert '"start_line": 3' in text
    assert '"end_line": 5' in text
    assert '"group_ids": ["group_1"]' in text
    assert "before running any robot effects" in text


def test_capsule_prompt_states_strict_python_subset_constraints():
    prompt = build_capsule_prompt(
        task="close the gripper",
        regions=[
            CodeRegion(
                region_id="region_1",
                start_line=1,
                end_line=1,
                source="close_gripper()",
            )
        ],
        history=[],
        trace_summary={},
        strict_subset=True,
    )

    text = prompt[1]["content"][0]["text"]

    assert "Strict Python subset" in text
    assert "no imports, classes, lambdas, try, while, async" in text
    assert "callable aliases" in text
    assert "attribute calls" in text
    assert "direct public API functions" in text
    assert "bounded for loops" in text


def test_capsule_prompt_bounds_large_contract_safety_context():
    violations = [
        {
            "code": f"effectful_helper_{index}_" + "c" * 1000,
            "message": f"unsafe helper {index}: " + "m" * 4000,
            "source_span": {"start_line": index + 1, "end_line": index + 2},
            "region_ids": [f"region_{index}_" + "r" * 1000 for _ in range(20)],
            "group_ids": [f"group_{index}_" + "g" * 1000 for _ in range(20)],
            "side_effect_calls": ["close_gripper_" + "s" * 1000],
            "helper_name": "helper_" + "h" * 1000,
        }
        for index in range(1000)
    ]

    prompt = build_capsule_prompt(
        task="close the gripper",
        regions=[
            CodeRegion(
                region_id="region_1",
                start_line=1,
                end_line=1,
                source="x = 1",
            )
        ],
        history=[],
        trace_summary={},
        contract_violations=violations,
        compact_context=True,
        prompt_char_budget=60000,
    )

    text = prompt[1]["content"][0]["text"]
    serialized = json.dumps(prompt, default=str)

    assert len(serialized) <= 60000
    assert "Capsule-ready program contract violations" in text
    assert '"total_count": 1000' in text
    assert '"omitted_count":' in text
    assert '"violations": [' in text
    assert '"code": "effectful_helper_0_' in text
    assert '"message": "unsafe helper 0:' in text
    assert '"source_span": {"start_line": 1, "end_line": 2}' in text
    assert '"group_ids": ["group_0_' in text
    assert "before running any robot effects" in text
    assert violations[0]["message"] not in text


def test_capsule_prompt_dynamically_bounds_contract_context_with_strict_constraints():
    violations = [
        {
            "code": f"effectful_helper_{index}_" + "c" * 1000,
            "message": f"unsafe helper {index}: " + "m" * 4000,
            "source_span": {"start_line": index + 1, "end_line": index + 2},
            "region_ids": [f"region_{index}_" + "r" * 1000 for _ in range(20)],
            "group_ids": [f"group_{index}_" + "g" * 1000 for _ in range(20)],
            "side_effect_calls": ["close_gripper_" + "s" * 1000],
            "helper_name": "helper_" + "h" * 1000,
        }
        for index in range(1000)
    ]

    prompt = build_capsule_prompt(
        task="close the gripper",
        regions=[
            CodeRegion(
                region_id="region_1",
                start_line=1,
                end_line=1,
                source="x = 1",
            )
        ],
        history=[],
        trace_summary={},
        contract_violations=violations,
        strict_subset=True,
        compact_context=True,
        prompt_char_budget=6500,
    )

    text = prompt[1]["content"][0]["text"]
    serialized = json.dumps(prompt, default=str)

    assert len(serialized) <= 6500
    assert "Capsule-ready program contract violations" in text
    assert '"total_count": 1000' in text
    assert '"omitted_count": 999' in text
    assert '"code": "effectful_helper_0_' in text
    assert "Strict Python subset" in text
    assert "before running any robot effects" in text


def test_capsule_prompt_bounds_aggregate_compact_region_units():
    regions = [
        CodeRegion(
            region_id=f"region_{index}",
            start_line=index + 1,
            end_line=index + 1,
            source=f"value_{index} = {index}",
        )
        for index in range(1000)
    ]
    history = [
        {
            "step_id": 1,
            "action": {"action": "run_region", "args": {"region_id": "region_997"}},
            "event": {"status": "failed", "region_id": "region_997"},
        }
    ]

    prompt = build_capsule_prompt(
        task="close the gripper",
        regions=regions,
        history=history,
        trace_summary={},
        contract_violations=[
            {
                "code": "effectful_helper",
                "message": "repair before running robot effects",
            }
        ],
        side_effect_ledger={"executed_side_effect_regions": ["region_998"]},
        strict_subset=True,
        compact_context=True,
        prompt_char_budget=6500,
    )

    text = prompt[1]["content"][0]["text"]
    serialized = json.dumps(prompt, default=str)
    region_envelope = _prompt_json_section(
        text,
        "Compact generated code regions",
        "Recent runtime history summary",
    )
    region_ids = [unit["region_id"] for unit in region_envelope["units"]]

    assert len(serialized) <= 6500
    assert region_envelope["total_count"] == 1000
    assert region_envelope["omitted_count"] > 0
    assert region_ids[:3] == ["region_997", "region_0", "region_998"]
    assert "Side-effect execution ledger" in text
    assert "Capsule-ready program contract violations" in text
    assert "before running any robot effects" in text
    assert "Strict Python subset" in text


def test_capsule_prompt_bounds_groups_and_large_side_effect_ledger():
    long_name = "payload_" + "X" * 500
    groups = [
        CodeRegionGroup(
            group_id=f"group_{index}",
            start_line=index + 1,
            end_line=index + 1,
            source=f"value_{index} = {index}",
            region_ids=[f"region_{index}_{item}_{long_name}" for item in range(20)],
            primitive_calls=[f"primitive_{item}_{long_name}" for item in range(20)],
            defined_names=[f"defined_{item}_{long_name}" for item in range(20)],
            used_names=[f"used_{item}_{long_name}" for item in range(20)],
            has_robot_side_effect=True,
        )
        for index in range(1000)
    ]
    history = [
        {
            "step_id": 1,
            "action": {"action": "run_group", "args": {"group_id": "group_997"}},
            "feedback": {"status": "invalid", "region_id": "group_997"},
        }
    ]
    executed_groups = ["group_998", *[f"zz_executed_{index}_{long_name}" for index in range(999)]]
    executed_regions = [f"zz_region_{index}_{long_name}" for index in range(1000)]

    prompt = build_capsule_prompt(
        task="close the gripper",
        regions=[CodeRegion(region_id="region_0", start_line=1, end_line=1, source="x = 1")],
        groups=groups,
        history=history,
        trace_summary={},
        contract_violations=[
            {
                "code": "effectful_helper",
                "message": "repair before running robot effects",
            }
        ],
        side_effect_ledger={
            "executed_side_effect_groups": executed_groups,
            "executed_side_effect_regions": executed_regions,
        },
        strict_subset=True,
        compact_context=True,
        prompt_char_budget=6500,
    )

    text = prompt[1]["content"][0]["text"]
    serialized = json.dumps(prompt, default=str)
    group_envelope = _prompt_json_section(
        text,
        "Compact effect-bounded execution units (preferred run_group targets)",
        "Compact generated code regions",
    )
    group_ids = [unit["group_id"] for unit in group_envelope["units"]]
    executed_group = next(unit for unit in group_envelope["units"] if unit["group_id"] == "group_998")

    assert len(serialized) <= 6500
    assert group_envelope["total_count"] == 1000
    assert group_envelope["omitted_count"] > 0
    assert group_ids[:3] == ["group_997", "group_0", "group_998"]
    assert executed_group["unit_id"] == "group_998"
    assert executed_group["run_allowed"] is False
    assert '"executed_side_effect_groups_total_count": 1000' in text
    assert '"executed_side_effect_groups_omitted_count":' in text
    assert "X" * 300 not in serialized
    assert "Capsule-ready program contract violations" in text
    assert "before running any robot effects" in text
    assert "Strict Python subset" in text


def test_capsule_prompt_defaults_to_legacy_without_strict_constraints():
    prompt = build_capsule_prompt(
        task="close the gripper",
        regions=[
            CodeRegion(
                region_id="region_1",
                start_line=1,
                end_line=1,
                source="close_gripper()",
            )
        ],
        history=[],
        trace_summary={},
    )

    assert "Strict Python subset" not in prompt[1]["content"][0]["text"]
