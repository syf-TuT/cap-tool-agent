import json

from capx.runtime_control.prompts import (
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
