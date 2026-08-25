from __future__ import annotations

import copy
from pathlib import Path, PurePosixPath

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


def test_controller_archive_location_matches_official_b10516_layout() -> None:
    workflow = load_owned_services_workflow(WORKFLOW_PATH)
    controller = workflow["services"]["controller"]
    archive = PurePosixPath(controller["archive_path"])
    binary = PurePosixPath(controller["binary_path"])

    assert archive.name == "llama-b10516-bin-ubuntu-x64.tar.gz"
    assert binary == archive.parent / "llama-b10516" / "llama-server"


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
    assert "不接外部模型 API" in documentation
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
