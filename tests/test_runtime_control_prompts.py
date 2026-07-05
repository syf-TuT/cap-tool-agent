from capx.runtime_control.prompts import build_capsule_prompt, parse_runtime_action_response
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
