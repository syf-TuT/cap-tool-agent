from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.capsule_rl.common import (
    FINAL_RUNTIME_AUDIT_GATE_ORDER,
    GateArtifactError,
    validate_final_runtime_audit,
)


_GATE_FILES = {
    "preflight": "gate01_preflight.json",
    "seed": "gate02_seed.json",
    "oracle_replay": "gate03_oracle.json",
    "collector": "gate04_collector.json",
    "guided": "gate05_guided_group.json",
    "trainer": "gate06_trainer.json",
    "adapter_reload": "adapter_reload_smoke.json",
}


def valid_final_runtime_audit(
    *,
    run_directory: Path = Path("/capsule/run"),
    run_id: str = "capsule-smoke-001",
    config_sha256: str = "c" * 64,
    dataset_sha256: str = "9" * 64,
    resolved_environment_sha256: str = "e" * 64,
    verl_resolved_config_sha256: str = "f" * 64,
    program_model_sha256: str = "1" * 64,
    actor_binding_sha256: str = "2" * 64,
    typed_task_identities: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    run_directory = run_directory.resolve()
    chain: list[dict[str, object]] = []
    previous_sha256: str | None = None
    for index, gate in enumerate(FINAL_RUNTIME_AUDIT_GATE_ORDER, start=1):
        sha256 = f"{index:x}" * 64
        chain.append(
            {
                "gate": gate,
                "path": str(run_directory / _GATE_FILES[gate]),
                "sha256": sha256,
                "previous_sha256": previous_sha256,
            }
        )
        previous_sha256 = sha256
    return {
        "artifact_files": 7,
        "learning_groups": 1,
        "base_members": 7,
        "base_successes": 0,
        "guided_members": 1,
        "guided_successes": 1,
        "repair_attempts": 4,
        "pt_successes": 1,
        "p_hat_successes": 1,
        "retry_count": 0,
        "infra_failures": 0,
        "optimizer_steps": 1,
        "schema_version": 1,
        "artifact_type": "capsule_rl_gate07_runtime_audit",
        "runtime_verified": True,
        "run_id": run_id,
        "config_sha256": config_sha256,
        "git_sha": "d" * 40,
        "dataset_sha256": dataset_sha256,
        "resolved_environment_sha256": resolved_environment_sha256,
        "verl_resolved_config_sha256": verl_resolved_config_sha256,
        "program_model_sha256": program_model_sha256,
        "actor_binding_sha256": actor_binding_sha256,
        "typed_task_identities": typed_task_identities
        or [
            {
                "task_id": "cube-stack",
                "environment_seed": 5,
                "initial_state_sha256": "a" * 64,
            }
        ],
        "gate_statuses": {gate: "passed" for gate in FINAL_RUNTIME_AUDIT_GATE_ORDER},
        "gate_chain": chain,
        "gradient_norm": 0.25,
        "trainer_metrics": {"actor/pg_loss": 0.5},
        "adapter_reload_artifact_sha256": previous_sha256,
        "adapter_reload_max_abs_logit_diff": 1e-4,
        "adapter_model_sha256": "a" * 64,
        "adapter_config_sha256": "b" * 64,
        "gate07_candidate_sha256": "c" * 64,
        "launcher_continuous_memory_sha256": "d" * 64,
        "launcher_controller_attestation_sha256": "e" * 64,
        "launcher_owned_cleanup_sha256": "f" * 64,
        "launcher_initial_audit_sha256": "0" * 64,
        "launcher_post_controller_memory_sha256": "1" * 64,
        "continuous_memory_required_mib": 12288,
        "minimum_mem_available_mib": 19000,
        "continuous_memory_sample_count": 2,
        "continuous_memory_maximum_sample_gap_ms": 1000,
        "controller_archive_sha256": (
            "f263a91280471b4c33c4999d7c76259c0f3a0a53a0b3e692b2c0b84380137a35"
        ),
        "controller_binary_sha256": "2" * 64,
        "controller_gguf_sha256": (
            "509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c"
        ),
        "controller_runtime_tree_sha256": "3" * 64,
        "owned_service_cleanup_completed": True,
        "owned_service_cleanup_count": 3,
        "oom_profile": "base_dynamic_fp32",
        "resolved_profile_sha256": verl_resolved_config_sha256,
        "initial_hardware": {
            "gpu_name": "NVIDIA A800 80GB PCIe",
            "gpu_count": 1,
            "gpu_total_vram_mib": 81920,
            "gpu_free_vram_mib": 80000,
            "host_memory_mib": 131072,
            "shm_available_mib": 16384,
            "disk_free_mib": 100000,
            "repo_is_dirty": False,
        },
        "mem_available_after_controller_mib": 100000,
    }


def test_final_runtime_audit_accepts_the_complete_schema() -> None:
    validate_final_runtime_audit(valid_final_runtime_audit())


def test_final_runtime_audit_rejects_forged_minimal_true_flag() -> None:
    with pytest.raises(GateArtifactError, match="schema is incomplete"):
        validate_final_runtime_audit({"runtime_verified": True})


def test_final_runtime_audit_rejects_gate_chain_link_tampering() -> None:
    payload = deepcopy(valid_final_runtime_audit())
    payload["gate_chain"][4]["previous_sha256"] = "f" * 64

    with pytest.raises(GateArtifactError, match="previous_sha256"):
        validate_final_runtime_audit(payload)


def test_final_runtime_audit_rejects_resolved_profile_hash_mismatch() -> None:
    payload = deepcopy(valid_final_runtime_audit())
    payload["resolved_profile_sha256"] = "4" * 64

    with pytest.raises(GateArtifactError, match="resolved profile SHA"):
        validate_final_runtime_audit(payload)
