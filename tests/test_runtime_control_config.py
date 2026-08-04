from pathlib import Path
from types import SimpleNamespace

import yaml

from capx.envs.tasks.franka.franka_pick_place import PROMPT
from capx.utils.launch_utils import _load_config


def test_capsule_yaml_uses_code_primitives_not_robot_tools():
    data = yaml.safe_load(
        Path("env_configs/cube_stack/franka_robosuite_cube_stack_capsule_vdm.yaml").read_text()
    )

    assert data["agent_mode"] == "capsule"
    assert data["max_capsule_steps"] > 0
    assert "Write Python code" in data["env"]["cfg"]["prompt"]
    assert "selecting one JSON tool call" not in data["env"]["cfg"]["prompt"]


def test_strict_l1_cube_stack_capsule_uses_30_step_budget():
    data = yaml.safe_load(
        Path("env_configs/benchmarks/strict_l1/cube_stack_capsule.yaml").read_text()
    )

    assert data["agent_mode"] == "capsule"
    assert data["max_capsule_steps"] == 30


def test_cube_stack_capsule_benchmarks_use_loose_group_cap():
    for config_path in [
        "env_configs/benchmarks/strict_l1/cube_stack_capsule.yaml",
        "env_configs/benchmarks/lowlevel_primitives/cube_stack_capsule.yaml",
    ]:
        data = yaml.safe_load(Path(config_path).read_text())

        assert data["capsule_max_regions_per_group"] == 20


def test_cube_stack_prompt_matches_original_generic_experiment_prompt():
    expected = """
You are controlling a Franka Emika robot with FrankaControlApi.
Goal: pick up the red cube, place it on top of the green cube, then open the gripper.

Use non-privileged camera observations, SAM3 segmentation, Contact-GraspNet grasp
planning, Pyroki IK/planning, joint motion, and gripper commands. Do not use
privileged object-state shortcuts.

Write only executable Python code. Import numpy/scipy explicitly if needed. The
runtime-control Capsule llm_step loop may execute, inspect, and repair semantic
source groups after the initial program is generated.
"""

    assert PROMPT == expected
    for leaked_instruction in (
        "half-height",
        "reuse the grasp quaternion",
        "at least +0.2m",
        "stacking height formula",
    ):
        assert leaked_instruction not in PROMPT


def test_cube_stack_action_ablation_configs_form_controlled_a1_a4_matrix():
    base = yaml.safe_load(
        Path("env_configs/cube_stack/franka_robosuite_cube_stack.yaml").read_text()
    )
    a1 = yaml.safe_load(
        Path(
            "env_configs/cube_stack/franka_robosuite_cube_stack_a1_one_shot.yaml"
        ).read_text()
    )
    a2 = yaml.safe_load(
        Path(
            "env_configs/cube_stack/"
            "franka_robosuite_cube_stack_capsule_llm_step_a2_no_patch.yaml"
        ).read_text()
    )
    a3 = yaml.safe_load(
        Path(
            "env_configs/cube_stack/"
            "franka_robosuite_cube_stack_capsule_llm_step_a3_no_append_recovery.yaml"
        ).read_text()
    )
    a4 = yaml.safe_load(
        Path(
            "env_configs/cube_stack/"
            "franka_robosuite_cube_stack_capsule_llm_step_a4_full.yaml"
        ).read_text()
    )

    configs = (a1, a2, a3, a4)
    assert [
        (
            config["agent_mode"],
            config["capsule_llm_step_allow_patch"],
            config["capsule_llm_step_allow_append_recovery"],
        )
        for config in configs
    ] == [
        ("code", False, False),
        ("capsule", False, True),
        ("capsule", True, False),
        ("capsule", True, True),
    ]
    assert all(config["capsule_control_mode"] == "llm_step" for config in configs)
    assert "multi_turn_prompt" not in a1["env"]["cfg"]

    for config in configs:
        assert config["env"] == base["env"]
        assert config["api_servers"] == base["api_servers"]
        assert config["trials"] == base["trials"]
        assert config["num_workers"] == base["num_workers"]
        assert config["record_video"] == base["record_video"]
        assert config["use_visual_feedback"] is False
        assert config["use_img_differencing"] is False
        assert config["use_video_differencing"] is False
        assert "visual_differencing_model" not in config

    group_keys = {
        "capsule_llm_step_allow_patch",
        "capsule_llm_step_allow_append_recovery",
        "output_dir",
    }
    normalized_capsule_configs = [
        {key: value for key, value in config.items() if key not in group_keys}
        for config in (a2, a3, a4)
    ]
    assert all(
        config == normalized_capsule_configs[0]
        for config in normalized_capsule_configs[1:]
    )

    all_group_keys = group_keys | {"agent_mode"}
    normalized_configs = [
        {key: value for key, value in config.items() if key not in all_group_keys}
        for config in configs
    ]
    assert all(config == normalized_configs[0] for config in normalized_configs[1:])


def test_load_config_reads_capsule_fields():
    args = SimpleNamespace(
        config_path="env_configs/cube_stack/franka_robosuite_cube_stack_capsule_vdm.yaml",
        total_trials=None,
        num_workers=None,
        record_video=None,
        output_dir=None,
        use_oracle_code=None,
        use_visual_feedback=None,
        use_img_differencing=None,
        use_video_differencing=None,
        use_wrist_camera=None,
        use_parallel_ensemble=None,
        use_multimodel=None,
        web_ui=None,
        web_ui_port=None,
        server_url="http://127.0.0.1:8110/chat/completions",
        visual_differencing_model="google/gemini-3.1-pro-preview",
        visual_differencing_model_server_url="http://127.0.0.1:8110/chat/completions",
        visual_differencing_model_api_key=None,
    )

    _, config, _ = _load_config(args)

    assert config["agent_mode"] == "capsule"
    assert config["max_capsule_steps"] == 12
    assert config["max_regenerations"] is None
    assert config["capsule_control_mode"] == "auto_forward"
    assert config["checkpoint_policy"] == "region"
    assert config["rollback_policy"] == "none"
    assert config["capsule_execution_granularity"] == "semantic_group"
    assert config["capsule_max_regions_per_group"] == 20
    assert config["capsule_llm_step_compact_context"] is True
    assert config["capsule_llm_step_allow_patch"] is True
    assert config["capsule_llm_step_allow_append_recovery"] is True
    assert config["capsule_action_history_max_entries"] == 4
    assert config["capsule_action_trace_max_events"] == 5
    assert config["capsule_action_source_preview_chars"] == 240
    assert config["capsule_action_prompt_char_budget"] == 60000


def test_load_config_reads_max_regenerations(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
env:
  _target_: tests.fake.Env
trials: 1
max_regenerations: 5
"""
    )
    args = SimpleNamespace(
        config_path=str(config_path),
        total_trials=None,
        num_workers=None,
        record_video=None,
        output_dir=None,
        use_oracle_code=None,
        use_visual_feedback=None,
        use_img_differencing=None,
        use_video_differencing=None,
        use_wrist_camera=None,
        use_parallel_ensemble=None,
        use_multimodel=None,
        web_ui=None,
        web_ui_port=None,
        server_url="http://127.0.0.1:8110/chat/completions",
        visual_differencing_model="google/gemini-3.1-pro-preview",
        visual_differencing_model_server_url="http://127.0.0.1:8110/chat/completions",
        visual_differencing_model_api_key=None,
    )

    _, config, _ = _load_config(args)

    assert config["max_regenerations"] == 5


def test_load_config_defaults_to_no_rollback(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
env:
  _target_: tests.fake.Env
trials: 1
agent_mode: capsule
"""
    )
    args = SimpleNamespace(
        config_path=str(config_path),
        total_trials=None,
        num_workers=None,
        record_video=None,
        output_dir=None,
        use_oracle_code=None,
        use_visual_feedback=None,
        use_img_differencing=None,
        use_video_differencing=None,
        use_wrist_camera=None,
        use_parallel_ensemble=None,
        use_multimodel=None,
        web_ui=None,
        web_ui_port=None,
        server_url="http://127.0.0.1:8110/chat/completions",
        visual_differencing_model="google/gemini-3.1-pro-preview",
        visual_differencing_model_server_url="http://127.0.0.1:8110/chat/completions",
        visual_differencing_model_api_key=None,
    )

    _, config, _ = _load_config(args)

    assert config["rollback_policy"] == "none"


def test_load_config_reads_capsule_grouping_fields(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
env:
  _target_: tests.fake.Env
trials: 1
agent_mode: capsule
capsule_execution_granularity: atomic_region
capsule_max_regions_per_group: 4
"""
    )
    args = SimpleNamespace(
        config_path=str(config_path),
        total_trials=None,
        num_workers=None,
        record_video=None,
        output_dir=None,
        use_oracle_code=None,
        use_visual_feedback=None,
        use_img_differencing=None,
        use_video_differencing=None,
        use_wrist_camera=None,
        use_parallel_ensemble=None,
        use_multimodel=None,
        web_ui=None,
        web_ui_port=None,
        server_url="http://127.0.0.1:8110/chat/completions",
        visual_differencing_model="google/gemini-3.1-pro-preview",
        visual_differencing_model_server_url="http://127.0.0.1:8110/chat/completions",
        visual_differencing_model_api_key=None,
    )

    _, config, _ = _load_config(args)

    assert config["capsule_execution_granularity"] == "atomic_region"
    assert config["capsule_max_regions_per_group"] == 4


def test_load_config_reads_capsule_control_mode(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
env:
  _target_: tests.fake.Env
trials: 1
agent_mode: capsule
capsule_control_mode: auto_forward
"""
    )
    args = SimpleNamespace(
        config_path=str(config_path),
        total_trials=None,
        num_workers=None,
        record_video=None,
        output_dir=None,
        use_oracle_code=None,
        use_visual_feedback=None,
        use_img_differencing=None,
        use_video_differencing=None,
        use_wrist_camera=None,
        use_parallel_ensemble=None,
        use_multimodel=None,
        web_ui=None,
        web_ui_port=None,
        server_url="http://127.0.0.1:8110/chat/completions",
        visual_differencing_model="google/gemini-3.1-pro-preview",
        visual_differencing_model_server_url="http://127.0.0.1:8110/chat/completions",
        visual_differencing_model_api_key=None,
    )

    _, config, _ = _load_config(args)

    assert config["capsule_control_mode"] == "auto_forward"


def test_load_config_reads_compact_llm_step_prompt_fields(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
env:
  _target_: tests.fake.Env
trials: 1
agent_mode: capsule
capsule_llm_step_compact_context: false
capsule_llm_step_allow_patch: false
capsule_llm_step_allow_append_recovery: false
capsule_action_history_max_entries: 2
capsule_action_trace_max_events: 3
capsule_action_source_preview_chars: 80
capsule_action_prompt_char_budget: 12000
"""
    )
    args = SimpleNamespace(
        config_path=str(config_path),
        total_trials=None,
        num_workers=None,
        record_video=None,
        output_dir=None,
        use_oracle_code=None,
        use_visual_feedback=None,
        use_img_differencing=None,
        use_video_differencing=None,
        use_parallel_ensemble=None,
        use_wrist_camera=None,
        use_multimodel=None,
        use_multi_turn=None,
        use_runtime_control=None,
        web_ui=None,
        web_ui_port=None,
        server_url="http://127.0.0.1:8110/chat/completions",
        visual_differencing_model="google/gemini-3.1-pro-preview",
        visual_differencing_model_server_url="http://127.0.0.1:8110/chat/completions",
        visual_differencing_model_api_key=None,
    )

    _, config, _ = _load_config(args)

    assert config["capsule_llm_step_compact_context"] is False
    assert config["capsule_llm_step_allow_patch"] is False
    assert config["capsule_llm_step_allow_append_recovery"] is False
    assert config["capsule_action_history_max_entries"] == 2
    assert config["capsule_action_trace_max_events"] == 3
    assert config["capsule_action_source_preview_chars"] == 80
    assert config["capsule_action_prompt_char_budget"] == 12000
