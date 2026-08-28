from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from capx.rl.capsule.compat import CapsuleConfigError, validate_capsule_config
from capx.rl.capsule.task_profiles import (
    collect_environment_profile_errors,
    resolve_task_profile,
)


CONFIG_PATH = (
    Path(__file__).parents[1]
    / "env_configs"
    / "cube_stack"
    / "capsule_rl"
    / "franka_robosuite_cube_stack_capsule_critique_grpo.yaml"
)
CLEAN_REPLAY_CONFIG_PATH = CONFIG_PATH.with_name(
    "franka_robosuite_cube_stack_privileged_clean_replay.yaml"
)
LIFT_PROFILE_NAME = "robosuite_cube_lift_privileged_highlevel"
LIFT_CONFIG_PATH = (
    Path(__file__).parents[1]
    / "env_configs"
    / "cube_lifting"
    / "capsule_rl"
    / "franka_robosuite_cube_lift_capsule_smoke.yaml"
)
LIFT_CLEAN_REPLAY_CONFIG_PATH = LIFT_CONFIG_PATH.with_name(
    "franka_robosuite_cube_lift_privileged_clean_replay.yaml"
)


def valid_config() -> dict:
    return {
        "schema_version": 1,
        "trainer_factory": "capx.rl.capsule.server_factory:create_trainer",
        "runtime": {
            "verl_source_path": "capx/third_party/verl",
            "verl_pinned_sha": "d62da4950573d7a4b7ef2362337952e7ab59e78d",
            "output_dir": "outputs/cube_stack_capsule_rl",
            "dataset_path": "/path/to/capsule_dataset.parquet",
            "program_model_path": "/path/to/program_model",
            "verl_resolved_config_path": "/path/to/resolved_verl_ppo.yaml",
            "requires": {"egl": True, "pyroki": True},
        },
        "task": {
            "environment": "robosuite_cube_stack",
            "config_path": (
                "env_configs/cube_stack/capsule_rl/"
                "franka_robosuite_cube_stack_privileged_clean_replay.yaml"
            ),
            "api": "franka_control_privileged",
            "privilege": "privileged",
            "render": False,
            "record_video": False,
        },
        "program_service": {
            "mode": "actor_identity",
            "endpoint": "http://127.0.0.1:8101/v1",
            "model": "program-model",
            "api_key_env": "CAPX_PROGRAM_API_KEY",
        },
        "controller_service": {
            "endpoint": "https://coding.dashscope.aliyuncs.com/v1",
            "model": "qwen3.7-plus",
            "api_key_env": "CAPX_CONTROLLER_API_KEY",
            "frozen": True,
            "request_timeout_s": 300.0,
            "max_output_tokens": 4096,
            "stream": False,
            "enable_thinking": False,
            "temperature": 0.7,
        },
        "capsule": {
            "group_size": 8,
            "max_group_attempts": 3,
            "base_samples_before_repair": 7,
            "p0_count": 2,
            "repair_trajectories_per_p0": 2,
            "max_controller_turns": 12,
            "revision_input_max_tokens": 8192,
            "revision_response_max_tokens": 2048,
            "gamma": 0.1,
        },
        "actor_rollout_ref": {
            "model": {"external_lib": "capx.rl.capsule.verl_external"},
            "rollout": {"n": 8, "calculate_log_probs": False, "mode": "sync"},
            "actor": {
                "use_kl_loss": True,
                "kl_loss_coef": 0.001,
                "ppo_epochs": 1,
                "ppo_mini_batch_size": 8,
                "ulysses_sequence_parallel_size": 1,
                "policy_loss": {"loss_mode": "capsule_critique", "capsule_gamma": 0.1},
            },
        },
        "algorithm": {
            "adv_estimator": "grpo",
            "norm_adv_by_std_in_grpo": False,
            "rollout_is": False,
            "rollout_is_threshold": None,
            "use_kl_in_reward": False,
        },
        "reward_model": {
            "enable": False,
            "reward_manager": "typed_replay",
            "launch_reward_fn_async": False,
        },
        "server_validation": {
            "gates": [
                "preflight",
                "seed",
                "oracle_replay",
                "collector",
                "guided",
                "trainer",
                "result_audit",
            ]
        },
    }


def valid_lift_config() -> dict:
    config = deepcopy(valid_config())
    config["task"].update(
        {
            "profile": LIFT_PROFILE_NAME,
            "environment": "robosuite_cube_lift",
            "config_path": (
                "env_configs/cube_lifting/capsule_rl/"
                "franka_robosuite_cube_lift_privileged_clean_replay.yaml"
            ),
        }
    )
    return config


def test_legacy_cube_stack_config_without_profile_is_accepted() -> None:
    config = valid_config()

    assert "profile" not in config["task"]
    validate_capsule_config(config)


def test_explicit_cube_stack_profile_is_accepted() -> None:
    config = valid_config()
    config["task"]["profile"] = "robosuite_cube_stack_privileged"

    validate_capsule_config(config)


def test_explicit_privileged_highlevel_cube_lift_profile_is_accepted() -> None:
    validate_capsule_config(valid_lift_config())


def test_cube_lift_requires_an_explicit_profile() -> None:
    config = valid_lift_config()
    del config["task"]["profile"]

    with pytest.raises(CapsuleConfigError, match=r"task\.profile.*explicit"):
        validate_capsule_config(config)


def test_unknown_task_profile_is_rejected() -> None:
    config = valid_config()
    config["task"]["profile"] = "unknown_profile"

    with pytest.raises(CapsuleConfigError, match=r"task\.profile.*unknown_profile"):
        validate_capsule_config(config)


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("task", "environment"), "robosuite_cube_stack"),
        (("task", "api"), "franka_control"),
        (("task", "privilege"), "unprivileged"),
        (("task", "render"), True),
        (("task", "record_video"), True),
        (("runtime", "requires", "egl"), False),
        (("runtime", "requires", "pyroki"), False),
    ],
)
def test_selected_task_profile_rejects_contract_mismatch(path, bad_value) -> None:
    config = valid_lift_config()
    current = config
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = bad_value

    with pytest.raises(CapsuleConfigError) as caught:
        validate_capsule_config(config)

    message = str(caught.value)
    assert LIFT_PROFILE_NAME in message
    assert ".".join(path) in message


def test_task_profile_mismatches_are_aggregated() -> None:
    config = valid_lift_config()
    config["task"]["render"] = True
    config["runtime"]["requires"]["egl"] = False

    with pytest.raises(CapsuleConfigError) as caught:
        validate_capsule_config(config)

    message = str(caught.value)
    assert "task.render" in message
    assert "runtime.requires.egl" in message


def test_boolean_task_privilege_remains_invalid() -> None:
    config = valid_lift_config()
    config["task"]["privilege"] = True

    with pytest.raises(CapsuleConfigError, match=r"task\.privilege.*'privileged'"):
        validate_capsule_config(config)


@pytest.mark.parametrize(
    "field_name",
    ["max_output_tokens", "stream", "enable_thinking"],
)
def test_controller_request_contract_must_be_explicit(field_name: str) -> None:
    config = valid_config()
    del config["controller_service"][field_name]

    with pytest.raises(CapsuleConfigError, match=field_name):
        validate_capsule_config(config)


@pytest.mark.parametrize(
    ("path", "bad_value", "message"),
    [
        (("schema_version",), True, "schema"),
        (("capsule", "group_size"), True, "eight-member"),
        (("capsule", "max_group_attempts"), 2, "three"),
        (("actor_rollout_ref", "rollout", "n"), 7, "n=8"),
        (("algorithm", "adv_estimator"), "gae", "grpo"),
        (("algorithm", "norm_adv_by_std_in_grpo"), True, "std"),
        (("algorithm", "rollout_is"), True, "rollout importance"),
        (("actor_rollout_ref", "rollout", "calculate_log_probs"), True, "calculate_log_probs"),
        (("actor_rollout_ref", "rollout", "mode"), "async", "sync"),
        (("actor_rollout_ref", "actor", "use_kl_loss"), False, "use_kl_loss"),
        (("actor_rollout_ref", "actor", "kl_loss_coef"), 0.0, "kl_loss_coef"),
        (("actor_rollout_ref", "actor", "ppo_epochs"), 2, "ppo_epochs"),
        (
            ("actor_rollout_ref", "actor", "ulysses_sequence_parallel_size"),
            2,
            "sequence parallelism",
        ),
        (("actor_rollout_ref", "actor", "ppo_mini_batch_size"), 4, "ppo_mini_batch_size"),
        (("algorithm", "use_kl_in_reward"), True, "use_kl_in_reward"),
        (("reward_model", "reward_manager"), "prime", "Prime"),
        (("reward_model", "launch_reward_fn_async"), True, "async"),
        (("controller_service", "frozen"), False, "frozen"),
        (("controller_service", "temperature"), 3.0, "temperature"),
        (("controller_service", "request_timeout_s"), 0.0, "request_timeout_s"),
        (("controller_service", "max_output_tokens"), True, "max_output_tokens"),
        (("controller_service", "max_output_tokens"), 0, "max_output_tokens"),
        (("controller_service", "stream"), True, "stream"),
        (("controller_service", "stream"), "false", "stream"),
        (("controller_service", "enable_thinking"), True, "enable_thinking"),
        (("controller_service", "enable_thinking"), "false", "enable_thinking"),
        (("program_service", "mode"), "generation", "actor_identity"),
        (("task", "render"), True, "render"),
        (("capsule", "max_controller_turns"), 13, "12"),
        (("capsule", "gamma"), 0.2, "0.1"),
    ],
)
def test_unsafe_or_algorithm_changing_values_are_rejected(path, bad_value, message) -> None:
    config = deepcopy(valid_config())
    current = config
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = bad_value

    with pytest.raises(CapsuleConfigError, match=message):
        validate_capsule_config(config)


def test_program_and_controller_must_be_separate_and_use_env_var_names_only() -> None:
    config = valid_config()
    config["controller_service"]["endpoint"] = config["program_service"]["endpoint"]
    config["controller_service"]["api_key_env"] = "literal-secret-value"

    with pytest.raises(CapsuleConfigError) as caught:
        validate_capsule_config(config)

    assert "separate endpoints" in str(caught.value)
    assert "api_key_env" in str(caught.value)


@pytest.mark.parametrize("service_name", ["program_service", "controller_service"])
def test_formal_config_rejects_plaintext_http_for_non_loopback_service(
    service_name: str,
) -> None:
    config = valid_config()
    config[service_name]["endpoint"] = "http://example.com/v1"

    with pytest.raises(CapsuleConfigError, match="https.*loopback"):
        validate_capsule_config(config)


def test_repository_template_contains_all_local_and_server_contract_fields() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    validate_capsule_config(config)

    assert config["task"]["render"] is False
    assert config["task"]["record_video"] is False
    assert config["controller_service"]["api_key_env"] == "CAPX_CONTROLLER_API_KEY"
    assert config["program_service"]["mode"] == "actor_identity"
    assert config["controller_service"]["endpoint"] == (
        "https://coding.dashscope.aliyuncs.com/v1"
    )
    assert config["controller_service"]["model"] == "qwen3.7-plus"
    assert config["controller_service"]["request_timeout_s"] == 300.0
    assert config["controller_service"]["max_output_tokens"] == 4096
    assert config["controller_service"]["stream"] is False
    assert config["controller_service"]["enable_thinking"] is False
    assert "api_key" not in config["controller_service"]
    assert config["server_validation"]["gates"][:3] == [
        "preflight",
        "seed",
        "oracle_replay",
    ]
    assert config["task"]["config_path"] == str(
        CLEAN_REPLAY_CONFIG_PATH.relative_to(Path(__file__).parents[1])
    ).replace("\\", "/")


def test_clean_replay_environment_template_disables_render_and_video() -> None:
    config = yaml.safe_load(CLEAN_REPLAY_CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["env"]["cfg"]["privileged"] is True
    assert config["env"]["cfg"]["enable_render"] is False
    assert config["env"]["cfg"]["viser_debug"] is False
    assert config["record_video"] is False
    assert config["num_workers"] == 1


def test_cube_lift_repository_template_preserves_capsule_contract() -> None:
    stack_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    lift_config = yaml.safe_load(LIFT_CONFIG_PATH.read_text(encoding="utf-8"))

    validate_capsule_config(lift_config)

    assert lift_config["task"]["profile"] == LIFT_PROFILE_NAME
    assert tuple(
        lift_config["task"][field]
        for field in ("environment", "api", "privilege")
    ) == (
        "robosuite_cube_lift",
        "franka_control_privileged",
        "privileged",
    )
    assert lift_config["task"]["config_path"] == str(
        LIFT_CLEAN_REPLAY_CONFIG_PATH.relative_to(Path(__file__).parents[1])
    ).replace("\\", "/")
    assert lift_config["task"]["render"] is False
    assert lift_config["task"]["record_video"] is False

    for field in ("schema_version", "trainer_factory"):
        assert lift_config[field] == stack_config[field]
    for section in (
        "controller_service",
        "capsule",
        "actor_rollout_ref",
        "algorithm",
        "reward_model",
        "server_validation",
    ):
        assert lift_config[section] == stack_config[section]
    for field in ("run_id", "verl_source_path", "verl_pinned_sha", "requires"):
        assert lift_config["runtime"][field] == stack_config["runtime"][field]
    for field in ("mode", "endpoint", "api_key_env"):
        assert lift_config["program_service"][field] == stack_config["program_service"][field]
    assert lift_config["program_service"]["model"] == lift_config["runtime"][
        "program_model_path"
    ]

    system_prompt = lift_config["program_service"]["system_prompt"]
    assert "one complete independently executable Python robot program" in system_prompt
    assert "without Markdown code fences" in system_prompt
    assert "high-level robot functions documented in the task prompt" in system_prompt
    assert "WXYZ" in system_prompt


def test_cube_lift_clean_replay_environment_matches_selected_profile() -> None:
    lift_config = yaml.safe_load(LIFT_CONFIG_PATH.read_text(encoding="utf-8"))
    environment = yaml.safe_load(
        LIFT_CLEAN_REPLAY_CONFIG_PATH.read_text(encoding="utf-8")
    )

    profile = resolve_task_profile(lift_config)
    assert collect_environment_profile_errors(environment, profile) == ()
    assert environment == {
        "env": {
            "_target_": "capx.envs.tasks.franka.franka_lift.FrankaLiftCodeEnv",
            "cfg": {
                "_target_": "capx.envs.tasks.base.CodeExecEnvConfig",
                "low_level": "franka_robosuite_cube_lift_low_level",
                "privileged": True,
                "enable_render": False,
                "viser_debug": False,
                "apis": ["FrankaControlPrivilegedApi"],
            },
        },
        "api_servers": [
            {
                "_target_": "capx.serving.launch_pyroki_server.main",
                "port": 8116,
                "host": "127.0.0.1",
                "robot": "panda_description",
                "target_link": "panda_hand",
            }
        ],
        "record_video": False,
        "output_dir": "./outputs/cube_lift_privileged_clean_replay",
        "trials": 1,
        "num_workers": 1,
    }
