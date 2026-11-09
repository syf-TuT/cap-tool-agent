import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from capx.envs.launch import LaunchArgs
from capx.envs.scripts.run_libero_batch import ABLATION_CONFIGS
from capx.envs.tasks.base import CodeExecEnvConfig
from capx.envs.tasks.franka.franka_pick_place import PROMPT
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


@pytest.mark.parametrize(
    "source_config",
    [
        "cube_lifting/franka_robosuite_cube_lifting.yaml",
        "cube_stack/franka_robosuite_cube_stack.yaml",
        "cube_restack/franka_robosuite_cube_restack.yaml",
        "spill_wipe/franka_robosuite_spill_wipe.yaml",
        "two_arm_lift/franka_robosuite_two_arm_lift.yaml",
        "two_arm_handover/two_arm_handover.yaml",
    ],
)
def test_capsule_benchmark_generated_config_loads(source_config, tmp_path):
    script = Path(
        "scripts/benchmarks/run_qwen7b_capsule20_six_tasks_s01_20.sh"
    ).read_text()
    generator = script.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    output = tmp_path / "capsule.yaml"
    subprocess.run(
        [sys.executable, "-c", generator, f"env_configs/{source_config}", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )

    _, config, _ = _load_config(_args_for_config(output))

    assert config["agent_mode"] == "capsule"
    assert config["max_capsule_steps"] == 20
    assert config["capsule_require_task_success_for_finish"] is True


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
    assert "capsule_control_mode" not in data
    assert data["max_capsule_steps"] == 24
    assert data["capsule_progress_mode"] == "sparse_terminal"
    assert data["capsule_require_task_success_for_finish"] is True
    assert data["capsule_validate_program_contract"] is True
    assert data["capsule_action_visual_feedback"] is False
    assert data["capsule_prompt_state_level"] == "proprioceptive"
    assert data["capsule_diagnostic_state_level"] == "full"
    assert data["use_visual_feedback"] is False
    assert data["use_wrist_camera"] is False
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
    assert "only zero-argument .copy() is allowed as an attribute call" in prompt
    assert "other attribute calls are forbidden" in prompt
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
    assert "text-only decision model" in docs
    assert "images are not attached to its initial or capsule action prompts" in docs
    assert "wrist-camera capture is disabled" in docs
    assert "molmo remains an internal perception service used by `frankaliberoapi`" in docs
    assert "no imports, classes, lambdas, `try`, `while`, or async constructs" in docs
    assert "only zero-argument `.copy()` is allowed as an attribute call" in docs
    assert "all other attribute calls are unavailable" in docs
    assert "only statically bounded `for` loops" in docs
    assert "all loops" not in docs
    assert "prepared wsl project at `/home/capx/code/cap-x`" in docs
    assert "do not run this command from the windows checkout" in docs
    assert "external molmo vllm service" in docs
    assert "not auto-started" in docs
    assert "separate molmo service environment" in docs
    assert "uv venv .venv-molmo --python 3.12" in docs
    assert "source .venv-molmo/bin/activate" in docs
    assert "uv sync --active --extra molmo" in docs
    assert (
        "python -m capx.serving.vllm_server --model allenai/molmo2-8b --host 127.0.0.1 --port 8122"
    ) in docs
    assert "sam3" in docs
    assert "point-cloud" in docs


def test_libero_object_privileged_capsule_yaml_uses_ground_truth_api_and_regular_python():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (
        repo_root
        / "env_configs"
        / "libero"
        / "franka_libero_object_0_privileged_capsule_llm_step.yaml"
    )
    data = yaml.safe_load(config_path.read_text())
    cfg = data["env"]["cfg"]
    low_level = cfg["low_level"]

    assert data["env"]["_target_"] == (
        "capx.envs.tasks.franka.franka_libero_env.FrankaLiberoCodeEnv"
    )
    assert cfg["_target_"] == "capx.envs.tasks.base.CodeExecEnvConfig"
    assert low_level["_target_"] == "capx.envs.simulators.libero.FrankaLiberoEnv"
    assert low_level["suite_name"] == "libero_object"
    assert low_level["task_id"] == 0
    assert low_level["privileged"] is True
    assert low_level["max_steps"] == 8000
    assert low_level["control_freq"] == 20
    assert cfg["privileged"] is True
    assert cfg["apis"] == ["FrankaLiberoPrivilegedApi"]
    assert "molmo_base_url" not in cfg
    assert "molmo_model_name" not in cfg
    assert data["agent_mode"] == "capsule"
    assert data["max_capsule_steps"] == 24
    assert data["capsule_execution_granularity"] == "semantic_group"
    assert data["capsule_progress_mode"] == "sparse_terminal"
    assert data["capsule_require_task_success_for_finish"] is True
    assert data["capsule_validate_program_contract"] is False
    assert data["capsule_action_visual_feedback"] is False
    assert data["capsule_prompt_state_level"] == "full"
    assert data["capsule_diagnostic_state_level"] == "full"
    assert data["use_visual_feedback"] is False
    assert data["use_wrist_camera"] is False
    assert data["use_parallel_ensemble"] is False
    assert data["record_video"] is True
    assert data["output_dir"] == (
        "./outputs/franka_libero_object_0_privileged_capsule_llm_step"
    )
    assert data["trials"] == 1
    assert data["num_workers"] == 1
    assert data["api_servers"] == [
        {
            "_target_": "capx.serving.launch_pyroki_server.main",
            "port": 8116,
            "host": "127.0.0.1",
            "robot": "panda_description",
            "target_link": "panda_hand",
        }
    ]

    prompt = cfg["prompt"].lower()
    assert "one complete executable python program" in prompt
    assert "ground-truth object poses" in prompt
    assert "functions (apis) below are already imported" in prompt
    assert "if you want to use numpy, you need to import it explicitly" in prompt
    assert "no imports" not in prompt
    assert "do not access the internal env or apis handles" not in prompt
    assert "restricted execution environment" not in prompt
    assert "safe builtins" not in prompt
    assert "you may write python code comments for reasoning" in prompt
    assert "only write the executable python code" in prompt
    assert "do not write it in code fences" in prompt


def test_code_exec_env_config_defaults_molmo_service_to_unspecified():
    cfg = CodeExecEnvConfig(low_level=object(), apis=[])

    assert cfg.molmo_base_url is None
    assert cfg.molmo_model_name is None
def test_libero_ablation_configs_control_only_agent_mode_and_recovery():
    expected = {
        "a1": ("code", False, False),
        "a2": ("capsule", False, True),
        "a3": ("capsule", True, False),
        "a4": ("capsule", True, True),
    }
    common = []
    outputs = set()
    assert ABLATION_CONFIGS.keys() == expected.keys()
    for group, filename in ABLATION_CONFIGS.items():
        path = Path("env_configs/libero") / filename
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        env, runtime, _ = _load_config(LaunchArgs(config_path=str(path)))
        assert (
            runtime["agent_mode"],
            runtime["capsule_llm_step_allow_patch"],
            runtime["capsule_llm_step_allow_append_recovery"],
        ) == expected[group]
        assert not env["cfg"].get("multi_turn_prompt")
        assert env["cfg"]["apis"] == ["FrankaLiberoApi"]
        assert env["cfg"]["privileged"] is False
        assert env["cfg"]["low_level"]["suite_name"] == "libero_spatial"
        assert env["cfg"]["low_level"]["task_id"] == 0
        assert "capsule_control_mode" not in data
        assert runtime["max_capsule_steps"] == 12
        assert runtime["capsule_execution_granularity"] == "semantic_group"
        assert runtime["checkpoint_policy"] == "region"
        assert runtime["rollback_policy"] == "none"
        for flag in (
            "use_visual_feedback", "use_img_differencing", "use_video_differencing",
            "use_parallel_ensemble", "use_multimodel",
        ):
            assert runtime[flag] is False

        outputs.add(data.pop("output_dir"))
        for key in (
            "agent_mode", "capsule_llm_step_allow_patch", "capsule_llm_step_allow_append_recovery",
        ):
            del data[key]
        common.append(data)

    assert len(outputs) == 4
    assert all(config == common[0] for config in common)


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
    assert all("capsule_control_mode" not in config for config in configs)
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


@pytest.mark.parametrize(
    "task_config",
    [
        "cube_lifting/franka_robosuite_cube_lifting",
        "cube_restack/franka_robosuite_cube_restack",
        "spill_wipe/franka_robosuite_spill_wipe",
        "two_arm_lift/franka_robosuite_two_arm_lift",
        "two_arm_handover/two_arm_handover",
    ],
)
def test_task_action_ablation_configs_form_controlled_a1_a4_matrix(task_config):
    base_path = Path(f"env_configs/{task_config}.yaml")
    base = yaml.safe_load(base_path.read_text())
    expected_env = base["env"]
    expected_env["cfg"].setdefault("privileged", False)
    groups = (
        ("a1_one_shot", "code", False, False),
        ("capsule_llm_step_a2_no_patch", "capsule", False, True),
        ("capsule_llm_step_a3_no_append_recovery", "capsule", True, False),
        ("capsule_llm_step_a4_full", "capsule", True, True),
    )
    shared_config = None
    for suffix, mode, patch, append_recovery in groups:
        path = base_path.with_name(f"{base_path.stem}_{suffix}.yaml")
        data = yaml.safe_load(path.read_text())
        env, config, servers = _load_config(LaunchArgs(config_path=str(path)))

        assert env == data["env"]
        assert env["_target_"] == expected_env["_target_"]
        cfg = dict(env["cfg"])
        prompt = cfg.pop("prompt")
        assert prompt.startswith("Goal: ")
        assert "non-privileged robot APIs" in prompt
        assert "executable Python" in prompt
        assert cfg == expected_env["cfg"]
        assert servers == base["api_servers"]

        assert config["agent_mode"] == mode
        assert config["capsule_llm_step_allow_patch"] is patch
        assert config["capsule_llm_step_allow_append_recovery"] is append_recovery
        assert config["output_dir"] == f"./outputs/{path.stem}"
        assert config["total_trials"] == base["trials"]
        assert config["num_workers"] == base["num_workers"]
        assert config["record_video"] == base["record_video"]
        assert "capsule_control_mode" not in config
        assert config["max_capsule_steps"] == 12
        assert config["capsule_llm_step_compact_context"] is True
        assert config["capsule_execution_granularity"] == "semantic_group"
        assert config["capsule_max_regions_per_group"] == 20
        assert config["checkpoint_policy"] == "region"
        assert config["rollback_policy"] == "none"
        assert config["capsule_feedback_level"] == "source_region_repair_hint"
        assert config["use_oracle_code"] is False
        assert config["use_visual_feedback"] is False
        assert config["use_img_differencing"] is False
        assert config["use_video_differencing"] is False
        for key in (
            "capsule_reward_drop_guard_min_best_reward",
            "capsule_reward_drop_guard_threshold",
        ):
            assert config.get(key) == base.get(key)

        for key in (
            "agent_mode",
            "capsule_llm_step_allow_patch",
            "capsule_llm_step_allow_append_recovery",
            "output_dir",
        ):
            del data[key]
        if shared_config is None:
            shared_config = data
        else:
            assert data == shared_config


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
    assert "capsule_control_mode" not in config
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


def test_load_config_rejects_removed_capsule_control_mode(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
env:
  _target_: tests.fake.Env
trials: 1
agent_mode: capsule
capsule_control_mode: llm_step
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

    with pytest.raises(ValueError, match="capsule_control_mode has been removed"):
        _load_config(args)


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
capsule_require_task_success_for_finish: true
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
    assert config["capsule_require_task_success_for_finish"] is True


def test_cube_stack_nonprivileged_matched_configs_share_task_contract():
    multiturn = yaml.safe_load(
        Path("env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm_matched.yaml").read_text()
    )
    assert "use_legacy_multi_turn_decision_prompt" not in multiturn
    capsule = yaml.safe_load(
        Path("env_configs/cube_stack/franka_robosuite_cube_stack_capsule_vdm_matched.yaml").read_text()
    )

    multiturn_cfg = multiturn["env"]["cfg"]
    capsule_cfg = capsule["env"]["cfg"]
    assert multiturn_cfg["privileged"] is False
    assert capsule_cfg["privileged"] is False
    assert multiturn_cfg["apis"] == ["FrankaControlApiReducedSkillLibrary"]
    assert capsule_cfg["apis"] == ["FrankaControlApiReducedSkillLibrary"]
    assert multiturn_cfg["prompt"] == capsule_cfg["prompt"]
    assert "multi_turn_prompt" in multiturn_cfg
    assert "multi_turn_prompt" not in capsule_cfg
    assert capsule["agent_mode"] == "capsule"
