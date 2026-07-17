import json

from capx.runtime_control.prompts import (
    build_capsule_prompt,
    build_capsule_recovery_prompt,
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

    assert "Generated code groups" in text
    assert '{"action": "run_group", "args": {"group_id": "group_1"}}' in text
    assert '{"action": "patch_group", "args": {"group_id": "group_1", "source":' in text
    assert "Prefer run_group over run_region" in text


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


def test_recovery_prompt_is_local_and_bounded():
    groups = [
        CodeRegionGroup(
            group_id=f"group_{idx}",
            start_line=idx,
            end_line=idx,
            source=f"value_{idx} = {idx}",
            region_ids=[f"region_{idx}"],
            primitive_calls=[],
            defined_names=[f"value_{idx}"],
            used_names=[],
            has_robot_side_effect=False,
        )
        for idx in range(1, 12)
    ]
    history = [
        {
            "step_id": idx,
            "event": {"status": "success", "region_id": f"group_{idx}"},
            "feedback": {"message": f"history-{idx}"},
        }
        for idx in range(1, 12)
    ]
    trace_summary = {
        "events": [
            {"name": f"trace_{idx}", "args": [idx], "result": idx}
            for idx in range(1, 12)
        ]
    }

    prompt = build_capsule_recovery_prompt(
        task="stack cubes",
        failed_unit=groups[7],
        history_tail=history,
        trace_summary=trace_summary,
        side_effect_ledger={
            "executed_side_effect_groups": ["group_1"],
            "executed_side_effect_regions": ["region_1"],
        },
        recovery_observation_functions={"get_observation"},
    )

    text = prompt[1]["content"][0]["text"]
    allowed_actions = next(
        line for line in text.splitlines() if line.startswith("Allowed actions:")
    )

    assert "group_8" in text
    assert "value_8 = 8" in text
    assert "trace_11" in text
    assert '"name": "trace_1"' not in text
    assert "value_1 = 1" not in text
    assert '"message": "history-1"' not in text
    assert "history-11" in text
    assert "run_group" not in allowed_actions
    assert "run_region" not in allowed_actions
    assert "append_recovery" in allowed_actions
    assert "resume_from_region" in allowed_actions


def test_recovery_prompt_examples_are_valid_json():
    failed_group = CodeRegionGroup(
        group_id="group_2",
        start_line=2,
        end_line=3,
        source='raise RuntimeError("boom")',
        region_ids=["region_2"],
        primitive_calls=[],
        defined_names=[],
        used_names=[],
        has_robot_side_effect=False,
    )

    prompt = build_capsule_recovery_prompt(
        task="stack cubes",
        failed_unit=failed_group,
        history_tail=[],
        trace_summary={},
        side_effect_ledger={},
        recovery_observation_functions={"get_observation"},
    )

    text = prompt[1]["content"][0]["text"]
    example_lines = [
        line for line in text.splitlines() if line.startswith('{"action":')
    ]

    assert example_lines
    for line in example_lines:
        parsed = json.loads(line)
        assert "action" in parsed
        assert isinstance(parsed.get("args"), dict)
