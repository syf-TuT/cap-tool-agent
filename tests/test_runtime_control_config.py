from pathlib import Path
from types import SimpleNamespace

import yaml

from capx.envs.tasks.base import CodeExecEnvConfig
from capx.utils.launch_utils import _load_config


def _args_for_config(path):
    return SimpleNamespace(
        config_path=str(path),
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


def test_load_config_reads_libero_capsule_capabilities(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
env:
  _target_: tests.fake.Env
agent_mode: capsule
capsule_progress_mode: sparse_terminal
capsule_require_task_success_for_finish: true
capsule_validate_program_contract: true
capsule_action_visual_feedback: true
capsule_prompt_state_level: proprioceptive
capsule_diagnostic_state_level: full
"""
    )

    _, config, _ = _load_config(_args_for_config(config_path))

    assert config["capsule_progress_mode"] == "sparse_terminal"
    assert config["capsule_require_task_success_for_finish"] is True
    assert config["capsule_validate_program_contract"] is True
    assert config["capsule_action_visual_feedback"] is True
    assert config["capsule_prompt_state_level"] == "proprioceptive"
    assert config["capsule_diagnostic_state_level"] == "full"


def test_load_config_defaults_preserve_existing_capsule_behavior(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
env:
  _target_: tests.fake.Env
agent_mode: capsule
"""
    )

    _, config, _ = _load_config(_args_for_config(config_path))

    assert config["capsule_progress_mode"] == "dense"
    assert config["capsule_require_task_success_for_finish"] is False
    assert config["capsule_validate_program_contract"] is False
    assert config["capsule_action_visual_feedback"] is False
    assert config["capsule_prompt_state_level"] == "full"
    assert config["capsule_diagnostic_state_level"] == "none"


def test_libero_object_capsule_llm_step_yaml_uses_approved_capabilities():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (
        repo_root / "env_configs" / "libero" / "franka_libero_object_0_capsule_llm_step.yaml"
    )
    data = yaml.safe_load(config_path.read_text())
    cfg = data["env"]["cfg"]
    low_level = cfg["low_level"]

    assert low_level["suite_name"] == "libero_object"
    assert low_level["task_id"] == 0
    assert low_level["privileged"] is False
    assert cfg["privileged"] is False
    assert cfg["apis"] == ["FrankaLiberoApi"]
    assert cfg["molmo_base_url"] == "http://127.0.0.1:8122/v1"
    assert cfg["molmo_model_name"] == "allenai/Molmo2-8B"
    assert data["agent_mode"] == "capsule"
    assert data["capsule_control_mode"] == "llm_step"
    assert data["max_capsule_steps"] == 24
    assert data["capsule_progress_mode"] == "sparse_terminal"
    assert data["capsule_require_task_success_for_finish"] is True
    assert data["capsule_validate_program_contract"] is True
    assert data["capsule_action_visual_feedback"] is True
    assert data["capsule_prompt_state_level"] == "proprioceptive"
    assert data["capsule_diagnostic_state_level"] == "full"
    assert data["use_visual_feedback"] is True
    assert data["use_wrist_camera"] is True
    assert data["use_parallel_ensemble"] is False
    assert data["trials"] == 1
    assert data["num_workers"] == 1
    assert data["record_video"] is True

    prompt = cfg["prompt"].lower()
    assert "one complete executable python program" in prompt
    assert "strict python subset is mandatory" in prompt
    assert "no imports, classes, lambdas, try, while, or async" in prompt
    assert "dynamic or reflective calls" in prompt
    assert "callable aliases" in prompt
    assert "attribute calls" in prompt
    assert "direct calls to the public api functions" in prompt
    assert "safe builtins" in prompt
    assert "proven-pure top-level helper functions" in prompt
    assert "only statically bounded for loops" in prompt
    assert "loops must be statically bounded" not in prompt
    assert "total static computation budget" in prompt
    assert "robot side effects must be top-level" in prompt
    assert "at most one robot side-effect api call per semantic group" in prompt
    assert "import numpy" not in prompt

    docs = " ".join((repo_root / "docs" / "libero-tasks.md").read_text().lower().split())
    assert "no imports, classes, lambdas, `try`, `while`, or async constructs" in docs
    assert "only statically bounded `for` loops" in docs
    assert "all loops" not in docs
    assert "prepared wsl project at `/home/capx/code/cap-x`" in docs
    assert "do not run this command from the windows checkout" in docs
    assert "external molmo vllm service" in docs
    assert "not auto-started" in docs
    assert (
        "python -m capx.serving.vllm_server --model allenai/molmo2-8b "
        "--host 127.0.0.1 --port 8122"
    ) in docs
    assert "sam3" in docs
    assert "point-cloud" in docs


def test_code_exec_env_config_defaults_molmo_service_to_unspecified():
    cfg = CodeExecEnvConfig(low_level=object(), apis=[])

    assert cfg.molmo_base_url is None
    assert cfg.molmo_model_name is None


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
    assert config["capsule_action_history_max_entries"] == 2
    assert config["capsule_action_trace_max_events"] == 3
    assert config["capsule_action_source_preview_chars"] == 80
    assert config["capsule_action_prompt_char_budget"] == 12000
