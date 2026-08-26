from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts.capsule_rl.launch_owned_services import (
    OwnedServicesConfigError,
    load_owned_services_workflow,
)


WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1]
    / "env_configs"
    / "cube_stack"
    / "capsule_rl"
    / "franka_robosuite_cube_stack_capsule_single_a800_owned_services.yaml"
)
DOC_PATH = Path(__file__).resolve().parents[1] / "docs" / "capsule_rl.md"


def test_controller_is_fixed_external_non_thinking_non_streaming_contract() -> None:
    workflow = load_owned_services_workflow(WORKFLOW_PATH)
    assert workflow["services"]["controller"] == {
        "mode": "external",
        "endpoint": "https://coding.dashscope.aliyuncs.com/v1",
        "model": "qwen3.7-plus",
        "api_key_env": "CAPX_CONTROLLER_API_KEY",
        "request_timeout_s": 300.0,
        "max_output_tokens": 4096,
        "stream": False,
        "enable_thinking": False,
        "temperature": 0.7,
    }


def test_controller_randomness_budget_is_fixed_to_three_per_oom_profile(
    tmp_path: Path,
) -> None:
    workflow = load_owned_services_workflow(WORKFLOW_PATH)
    assert workflow["runtime"]["max_controller_seed_run_ids"] == 3

    invalid = copy.deepcopy(workflow)
    invalid["runtime"]["max_controller_seed_run_ids"] = 2
    invalid_path = tmp_path / "invalid-controller-randomness-budget.yaml"
    invalid_path.write_text(
        yaml.safe_dump(invalid, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(
        OwnedServicesConfigError,
        match=r"runtime\.max_controller_seed_run_ids",
    ):
        load_owned_services_workflow(invalid_path)


def test_single_a800_docs_bind_materialized_config_and_two_phase_gate7() -> None:
    documentation = DOC_PATH.read_text(encoding="utf-8")

    assert (
        "artifacts/cube_stack_capsule_rl_seed_resolved/"
        "capsule_rl.seed_resolved.yaml"
    ) in documentation
    assert "<prepared>/capsule.yaml" not in documentation
    assert "gate07_audit.candidate.json" in documentation
    assert "--continuous-memory-artifact" in documentation
    assert "Program actor-identity" in documentation
    assert "https://coding.dashscope.aliyuncs.com/v1" in documentation
    assert "`max_tokens=4096`" in documentation
    assert "`stream=false`" in documentation
    assert "`enable_thinking=false`" in documentation
    assert "Markdown fence 本身属于 Actor 协议错误" in documentation
    assert "Controller 显式提交删除 fence open/close（以及存在时的 trailing suffix）" in documentation
    assert "whole-program cleanup" in documentation
    assert "no-op 被拒绝并进入 audit" in documentation
    assert (
        "--capsule-config /root/autodl-tmp/cap-x/artifacts/"
        "cube_stack_capsule_rl_prepare/capsule_rl.resolved.yaml"
    ) in documentation
    assert "--config artifacts/$RUN_ID/resolved/capsule.yaml" in documentation
    assert (
        "--capsule-config /root/autodl-tmp/cap-x/artifacts/"
        "cube_stack_capsule_rl_seed_resolved/capsule_rl.seed_resolved.yaml"
        not in documentation
    )
