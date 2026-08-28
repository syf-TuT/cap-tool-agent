"""Shared validation and external gate execution for Capsule-RL server scripts."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import os
import re
import shlex
import shutil
import string
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from capx.rl.capsule.actor_identity import (
    ActorIdentityError,
    verify_actor_identity_payload,
)
from capx.rl.capsule.compat import CapsuleConfigError, validate_capsule_config
from capx.rl.capsule.controller import python_base_unit_spans
from capx.rl.capsule.lora_contract import (
    QWEN25_ALL_LINEAR_PROJECTIONS,
    QWEN25_ALL_LINEAR_TENSOR_COUNT,
    QWEN25_CODER_7B_LAYER_COUNT,
    QWEN25_PROJECTION_DIMENSIONS,
    qwen_lora_tensor_identity,
    validate_qwen_all_linear_coverage,
)
from capx.rl.capsule.provenance import runtime_dependency_hashes
from capx.rl.capsule.repair import RepairInvariantError
from capx.rl.capsule.schema import (
    LearningGroupV1,
    ProgramReplayResultV1,
    RepairTraceV1,
    ReplayOutcome,
    TaskInstanceV1,
    source_sha256,
)
from capx.rl.capsule.stable_io import (
    StablePathError,
    pin_absolute_path,
    read_stable_regular_file,
)
from capx.rl.capsule.task_profiles import (
    CapsuleTaskProfileError,
    collect_environment_profile_errors,
    resolve_task_profile,
)
from capx.rl.capsule.telemetry import summarize_replay_results
from capx.utils.program_source import normalize_program_source

SCHEMA_VERSION = 1
CANONICAL_EXECUTION_MODE = "repository_server_adapter_v1"
ADAPTER_RELOAD_PROMPT = "Return exactly one Python assignment that sets x to 1."
_ADAPTER_RELOAD_PROMPT_SHA256 = hashlib.sha256(
    ADAPTER_RELOAD_PROMPT.encode("utf-8")
).hexdigest()
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHELL_OPERATORS = frozenset({"|", "||", "&&", ";", ">", ">>", "<", "<<", "&"})
GATE_ORDER = ("preflight", "seed", "oracle_replay", "collector", "guided", "trainer")
FINAL_RUNTIME_AUDIT_GATE_ORDER = (*GATE_ORDER, "adapter_reload")
_FINAL_RUNTIME_AUDIT_GATE_FILES = {
    "preflight": "gate01_preflight.json",
    "seed": "gate02_seed.json",
    "oracle_replay": "gate03_oracle.json",
    "collector": "gate04_collector.json",
    "guided": "gate05_guided_group.json",
    "trainer": "gate06_trainer.json",
    "adapter_reload": "adapter_reload_smoke.json",
}
_FINAL_RUNTIME_AUDIT_FIELDS = frozenset(
    {
        "artifact_files",
        "learning_groups",
        "base_members",
        "base_successes",
        "guided_members",
        "guided_successes",
        "repair_attempts",
        "pt_successes",
        "p_hat_successes",
        "retry_count",
        "infra_failures",
        "optimizer_steps",
        "schema_version",
        "artifact_type",
        "runtime_verified",
        "run_id",
        "config_sha256",
        "git_sha",
        "dataset_sha256",
        "resolved_environment_sha256",
        "verl_resolved_config_sha256",
        "program_model_sha256",
        "actor_binding_sha256",
        "typed_task_identities",
        "gate_statuses",
        "gate_chain",
        "gradient_norm",
        "trainer_metrics",
        "adapter_reload_artifact_sha256",
        "adapter_reload_max_abs_logit_diff",
        "adapter_model_sha256",
        "adapter_config_sha256",
        "gate07_candidate_sha256",
        "launcher_continuous_memory_sha256",
        "launcher_controller_attestation_sha256",
        "launcher_owned_cleanup_sha256",
        "launcher_initial_audit_sha256",
        "launcher_post_controller_memory_sha256",
        "continuous_memory_required_mib",
        "minimum_mem_available_mib",
        "continuous_memory_sample_count",
        "continuous_memory_maximum_sample_gap_ms",
        "controller_mode",
        "controller_endpoint",
        "controller_model",
        "controller_binding_sha256",
        "owned_service_cleanup_completed",
        "owned_service_cleanup_count",
        "oom_profile",
        "resolved_profile_sha256",
        "initial_hardware",
        "mem_available_after_controller_mib",
    }
)
_FINAL_RUNTIME_AUDIT_SHA256_FIELDS = frozenset(
    {
        "config_sha256",
        "dataset_sha256",
        "resolved_environment_sha256",
        "verl_resolved_config_sha256",
        "program_model_sha256",
        "actor_binding_sha256",
        "adapter_reload_artifact_sha256",
        "adapter_model_sha256",
        "adapter_config_sha256",
        "gate07_candidate_sha256",
        "launcher_continuous_memory_sha256",
        "launcher_controller_attestation_sha256",
        "launcher_owned_cleanup_sha256",
        "launcher_initial_audit_sha256",
        "launcher_post_controller_memory_sha256",
        "controller_binding_sha256",
        "resolved_profile_sha256",
    }
)
_EXTERNAL_CONTROLLER_ENDPOINT = "https://coding.dashscope.aliyuncs.com/v1"
_EXTERNAL_CONTROLLER_MODEL = "qwen3.7-plus"
SINGLE_A800_OOM_PROFILE = "fsdp_base_bf16_vllm_util_045"


class ConfigValidationError(ValueError):
    """The server config cannot safely drive a Capsule-RL gate."""


class CommandValidationError(ValueError):
    """A runner command is incomplete or requires a shell."""


class GateExecutionError(RuntimeError):
    """An external gate runner failed or omitted its evidence artifact."""


class GateArtifactError(ValueError):
    """A gate artifact does not prove the required invariant."""


def _is_final_audit_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_final_audit_non_negative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _is_final_audit_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def validate_final_runtime_audit(payload: Mapping[str, Any]) -> None:
    """Validate the complete immutable Gate 7 runtime-audit consumer contract.

    A bare ``runtime_verified: true`` flag is deliberately insufficient.  Consumers accept
    only the exact schema emitted after all seven gate artifacts, Controller provenance,
    continuous host-memory monitoring, and owned-service cleanup have been bound together.
    """

    if not isinstance(payload, Mapping):
        raise GateArtifactError("final Gate 7 runtime audit must be a mapping")
    if set(payload) != _FINAL_RUNTIME_AUDIT_FIELDS:
        missing = sorted(_FINAL_RUNTIME_AUDIT_FIELDS - set(payload))
        unexpected = sorted(set(payload) - _FINAL_RUNTIME_AUDIT_FIELDS)
        raise GateArtifactError(
            "final Gate 7 runtime audit schema is incomplete or unexpected: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("artifact_type") != "capsule_rl_gate07_runtime_audit"
        or payload.get("runtime_verified") is not True
    ):
        raise GateArtifactError(
            "final Gate 7 runtime audit must use schema 1, the runtime-audit artifact type, "
            "and runtime_verified=true"
        )
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise GateArtifactError("final Gate 7 runtime audit run_id must be non-empty")
    git_sha = payload.get("git_sha")
    if not isinstance(git_sha, str) or _GIT_SHA_RE.fullmatch(git_sha) is None:
        raise GateArtifactError("final Gate 7 runtime audit git_sha must be a full Git SHA")
    for field_name in _FINAL_RUNTIME_AUDIT_SHA256_FIELDS:
        if not _is_final_audit_sha256(payload.get(field_name)):
            raise GateArtifactError(
                f"final Gate 7 runtime audit {field_name} must be lowercase SHA-256"
            )
    controller_mode = payload.get("controller_mode")
    controller_endpoint = payload.get("controller_endpoint")
    controller_model = payload.get("controller_model")
    if controller_mode not in {"local", "external"}:
        raise GateArtifactError("final Gate 7 runtime audit Controller mode is invalid")
    if not isinstance(controller_endpoint, str) or not controller_endpoint:
        raise GateArtifactError("final Gate 7 runtime audit Controller endpoint is invalid")
    if not isinstance(controller_model, str) or not controller_model:
        raise GateArtifactError("final Gate 7 runtime audit Controller model is invalid")
    if controller_mode == "external" and (
        controller_endpoint != _EXTERNAL_CONTROLLER_ENDPOINT
        or controller_model != _EXTERNAL_CONTROLLER_MODEL
    ):
        raise GateArtifactError(
            "final Gate 7 runtime audit does not bind the fixed external Controller"
        )

    counter_fields = (
        "artifact_files",
        "learning_groups",
        "base_members",
        "base_successes",
        "guided_members",
        "guided_successes",
        "repair_attempts",
        "pt_successes",
        "p_hat_successes",
        "retry_count",
        "infra_failures",
        "optimizer_steps",
    )
    for field_name in counter_fields:
        if not _is_final_audit_non_negative_int(payload.get(field_name)):
            raise GateArtifactError(
                f"final Gate 7 runtime audit {field_name} must be a non-negative integer"
            )
    if (
        payload.get("artifact_files", 0) < len(FINAL_RUNTIME_AUDIT_GATE_ORDER)
        or payload.get("learning_groups") != 1
        or payload.get("base_members") != 7
        or payload.get("base_successes") != 0
        or payload.get("guided_members") != 1
        or payload.get("guided_successes") != 1
        or payload.get("infra_failures") != 0
        or payload.get("optimizer_steps") != 1
    ):
        raise GateArtifactError(
            "final Gate 7 runtime audit does not prove the fixed 7+1 one-step contract"
        )

    identities = payload.get("typed_task_identities")
    if not isinstance(identities, list) or len(identities) != 1:
        raise GateArtifactError(
            "final Gate 7 runtime audit must bind exactly one typed seed-5 task"
        )
    identity = identities[0]
    if not isinstance(identity, Mapping) or set(identity) != {
        "task_id",
        "environment_seed",
        "initial_state_sha256",
    }:
        raise GateArtifactError("final Gate 7 typed task identity schema is invalid")
    if (
        not isinstance(identity.get("task_id"), str)
        or not identity.get("task_id")
        or identity.get("environment_seed") != 5
        or isinstance(identity.get("environment_seed"), bool)
        or not _is_final_audit_sha256(identity.get("initial_state_sha256"))
    ):
        raise GateArtifactError("final Gate 7 typed task identity is not the seed-5 task")

    expected_statuses = {gate: "passed" for gate in FINAL_RUNTIME_AUDIT_GATE_ORDER}
    if payload.get("gate_statuses") != expected_statuses:
        raise GateArtifactError("final Gate 7 runtime audit gate_statuses must all be passed")
    chain = payload.get("gate_chain")
    if not isinstance(chain, list) or len(chain) != len(FINAL_RUNTIME_AUDIT_GATE_ORDER):
        raise GateArtifactError(
            "final Gate 7 runtime audit must contain the fixed seven-item chain"
        )
    previous_sha256: str | None = None
    chain_parent: Path | None = None
    for index, (expected_gate, entry) in enumerate(
        zip(FINAL_RUNTIME_AUDIT_GATE_ORDER, chain, strict=True)
    ):
        if not isinstance(entry, Mapping) or set(entry) != {
            "gate",
            "path",
            "sha256",
            "previous_sha256",
        }:
            raise GateArtifactError(f"final Gate 7 gate_chain[{index}] schema is invalid")
        path_value = entry.get("path")
        path = Path(path_value) if isinstance(path_value, str) else None
        if (
            entry.get("gate") != expected_gate
            or path is None
            or not path.is_absolute()
            or path.name != _FINAL_RUNTIME_AUDIT_GATE_FILES[expected_gate]
            or not _is_final_audit_sha256(entry.get("sha256"))
        ):
            raise GateArtifactError(
                f"final Gate 7 gate_chain[{index}] does not bind {expected_gate}"
            )
        if chain_parent is None:
            chain_parent = path.parent
        elif path.parent != chain_parent:
            raise GateArtifactError("final Gate 7 gate_chain paths must share one run directory")
        if entry.get("previous_sha256") != previous_sha256:
            raise GateArtifactError(
                f"final Gate 7 gate_chain[{index}] previous_sha256 link is invalid"
            )
        previous_sha256 = str(entry["sha256"])
    if payload.get("adapter_reload_artifact_sha256") != previous_sha256:
        raise GateArtifactError(
            "final Gate 7 adapter reload SHA does not match the last gate-chain link"
        )

    gradient_norm = payload.get("gradient_norm")
    reload_difference = payload.get("adapter_reload_max_abs_logit_diff")
    if not _is_final_audit_finite_number(gradient_norm) or float(gradient_norm) <= 0:
        raise GateArtifactError("final Gate 7 gradient_norm must be finite and non-zero")
    if (
        not _is_final_audit_finite_number(reload_difference)
        or float(reload_difference) <= 1e-8
    ):
        raise GateArtifactError("final Gate 7 adapter reload must change finite logits")
    trainer_metrics = payload.get("trainer_metrics")
    if not isinstance(trainer_metrics, Mapping) or not trainer_metrics:
        raise GateArtifactError("final Gate 7 trainer_metrics must be non-empty")
    for name, value in trainer_metrics.items():
        if (
            not isinstance(name, str)
            or not name
            or not _is_final_audit_finite_number(value)
        ):
            raise GateArtifactError("final Gate 7 trainer_metrics must be finite numerics")

    memory_integer_fields = (
        "continuous_memory_required_mib",
        "minimum_mem_available_mib",
        "continuous_memory_sample_count",
        "continuous_memory_maximum_sample_gap_ms",
        "owned_service_cleanup_count",
        "mem_available_after_controller_mib",
    )
    for field_name in memory_integer_fields:
        if not _is_final_audit_non_negative_int(payload.get(field_name)):
            raise GateArtifactError(
                f"final Gate 7 runtime audit {field_name} must be a non-negative integer"
            )
    if (
        payload.get("continuous_memory_required_mib") != 12288
        or payload.get("minimum_mem_available_mib", 0) < 12288
        or payload.get("continuous_memory_sample_count", 0) < 2
        or payload.get("continuous_memory_maximum_sample_gap_ms", 5001) > 5000
        or payload.get("mem_available_after_controller_mib", 0) < 92160
    ):
        raise GateArtifactError("final Gate 7 runtime audit violates host-memory thresholds")
    if (
        payload.get("owned_service_cleanup_completed") is not True
        or payload.get("owned_service_cleanup_count") != 3
    ):
        raise GateArtifactError(
            "final Gate 7 runtime audit must bind all three logical services"
        )
    if payload.get("oom_profile") != SINGLE_A800_OOM_PROFILE:
        raise GateArtifactError("final Gate 7 runtime audit OOM profile is invalid")
    if payload.get("resolved_profile_sha256") != payload.get(
        "verl_resolved_config_sha256"
    ):
        raise GateArtifactError(
            "final Gate 7 resolved profile SHA must match the resolved VeRL config SHA"
        )

    hardware = payload.get("initial_hardware")
    hardware_fields = {
        "gpu_name",
        "gpu_count",
        "gpu_total_vram_mib",
        "gpu_free_vram_mib",
        "host_memory_mib",
        "shm_available_mib",
        "disk_free_mib",
        "repo_is_dirty",
    }
    if not isinstance(hardware, Mapping) or set(hardware) != hardware_fields:
        raise GateArtifactError("final Gate 7 initial_hardware schema is invalid")
    gpu_name = hardware.get("gpu_name")
    normalized_gpu_name = (
        gpu_name.upper().replace("-", "").replace(" ", "")
        if isinstance(gpu_name, str)
        else ""
    )
    if "A800" not in normalized_gpu_name or "80GB" not in normalized_gpu_name:
        raise GateArtifactError("final Gate 7 runtime audit requires an A800 80GB")
    hardware_thresholds = {
        "gpu_total_vram_mib": 81920,
        "gpu_free_vram_mib": 77824,
        "host_memory_mib": 122880,
        "shm_available_mib": 12288,
        "disk_free_mib": 81920,
    }
    if hardware.get("gpu_count") != 1 or isinstance(hardware.get("gpu_count"), bool):
        raise GateArtifactError("final Gate 7 runtime audit requires exactly one GPU")
    for field_name, minimum in hardware_thresholds.items():
        if (
            not _is_final_audit_non_negative_int(hardware.get(field_name))
            or hardware.get(field_name, 0) < minimum
        ):
            raise GateArtifactError(
                f"final Gate 7 initial_hardware {field_name} is below threshold"
            )
    if not isinstance(hardware.get("repo_is_dirty"), bool):
        raise GateArtifactError("final Gate 7 initial_hardware repo_is_dirty must be boolean")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def add_validation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config, paths, and command expansion without executing the gate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias for --validate-only; never starts a service, simulator, model, or optimizer.",
    )


def validation_requested(args: argparse.Namespace) -> bool:
    return bool(args.validate_only or args.dry_run)


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f"{field_name} must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], field_name: str, expected_type: type) -> Any:
    value = mapping.get(field_name)
    if isinstance(value, bool) and expected_type is int:
        raise ConfigValidationError(f"{field_name} must be an integer")
    if not isinstance(value, expected_type) or (isinstance(value, str) and not value.strip()):
        raise ConfigValidationError(f"{field_name} must be a non-empty {expected_type.__name__}")
    return value


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _validate_existing_path(path: Path, field_name: str, *, directory: bool) -> None:
    expected = path.is_dir() if directory else path.is_file()
    if not expected:
        kind = "directory" if directory else "file"
        raise ConfigValidationError(f"{field_name} must reference an existing {kind}: {path}")


def _validate_output_path(path: Path, field_name: str) -> None:
    if path.exists() and not path.is_dir():
        raise ConfigValidationError(f"{field_name} exists but is not a directory: {path}")
    parent = path if path.exists() else path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.is_dir():
        raise ConfigValidationError(f"{field_name} has no existing parent directory: {path}")


def artifact_file_sha256(path: str | Path) -> str:
    """Hash one persisted artifact exactly as stored on disk."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_tree_files(path: str | Path) -> tuple[Path, tuple[Path, ...]]:
    lexical_root = Path(path).expanduser()
    if lexical_root.is_symlink():
        raise GateArtifactError(f"artifact path must not be a symlink: {lexical_root}")
    root = lexical_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"artifact path does not exist: {root}")
    if root.is_file():
        return root.parent, (root,)
    files: list[Path] = []
    for entry in root.rglob("*"):
        if entry.is_symlink():
            raise GateArtifactError(f"artifact tree must not contain symlinks: {entry}")
        if entry.is_file():
            files.append(entry)
    return root, tuple(sorted(files, key=lambda item: item.relative_to(root).as_posix()))


def artifact_tree_file_count(path: str | Path) -> int:
    """Count immutable regular files in one checkpoint path."""

    _root, files = _artifact_tree_files(path)
    return len(files)


def artifact_tree_sha256(path: str | Path) -> str:
    """Hash checkpoint bytes together with normalized relative paths."""

    root, files = _artifact_tree_files(path)
    digest = hashlib.sha256()
    for file_path in files:
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(file_path.stat().st_size.to_bytes(8, "big"))
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _safetensors_tensor_metadata(payload: bytes) -> dict[str, Mapping[str, Any]]:
    if len(payload) < 9:
        raise GateArtifactError("adapter_model.safetensors has no valid header")
    header_size = int.from_bytes(payload[:8], "little")
    if header_size <= 0 or header_size > len(payload) - 8:
        raise GateArtifactError("adapter_model.safetensors header length is invalid")
    try:
        header = json.loads(payload[8 : 8 + header_size])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateArtifactError(
            f"adapter_model.safetensors header is invalid JSON: {error}"
        ) from error
    if not isinstance(header, Mapping):
        raise GateArtifactError("adapter_model.safetensors header must be a mapping")
    data_size = len(payload) - 8 - header_size
    tensors: dict[str, Mapping[str, Any]] = {}
    intervals: list[tuple[int, int]] = []
    dtype_bytes = {
        "BOOL": 1,
        "I8": 1,
        "U8": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "I16": 2,
        "U16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not name or not isinstance(tensor, Mapping):
            raise GateArtifactError("adapter_model.safetensors tensor metadata is invalid")
        offsets = tensor.get("data_offsets")
        shape = tensor.get("shape")
        dtype = tensor.get("dtype")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[0] > offsets[1]
            or offsets[1] > data_size
            or not isinstance(shape, list)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in shape
            )
            or not isinstance(dtype, str)
            or dtype not in dtype_bytes
        ):
            raise GateArtifactError("adapter_model.safetensors tensor metadata is invalid")
        element_count = math.prod(shape)
        if offsets[1] - offsets[0] != element_count * dtype_bytes[dtype]:
            raise GateArtifactError(
                "adapter_model.safetensors tensor byte length disagrees with shape/dtype"
            )
        intervals.append((offsets[0], offsets[1]))
        tensors[name] = tensor
    if not tensors:
        raise GateArtifactError("adapter_model.safetensors contains no adapter tensors")
    intervals.sort()
    if intervals[0][0] != 0 or intervals[-1][1] != data_size or any(
        previous[1] != current[0]
        for previous, current in zip(intervals, intervals[1:], strict=False)
    ):
        raise GateArtifactError(
            "adapter_model.safetensors tensor offsets do not cover the data section exactly"
        )
    return tensors


def direct_lora_adapter_evidence(checkpoint: str | Path) -> dict[str, Any]:
    """Validate VeRL v0.6.1's direct PEFT checkpoint layout and hash its bytes."""

    checkpoint_path = Path(os.path.abspath(Path(checkpoint).expanduser()))
    adapter_path = checkpoint_path / "lora_adapter"
    adapter_model_path = adapter_path / "adapter_model.safetensors"
    adapter_config_path = adapter_path / "adapter_config.json"
    try:
        with pin_absolute_path(
            checkpoint_path, label="trainer checkpoint", directory=True
        ) as pinned_checkpoint, pin_absolute_path(
            adapter_path, label="trainer lora_adapter", directory=True
        ) as pinned_adapter:
            model_snapshot = read_stable_regular_file(
                adapter_model_path, label="adapter_model.safetensors"
            )
            config_snapshot = read_stable_regular_file(
                adapter_config_path, label="adapter_config.json"
            )
            pinned_checkpoint.validate()
            pinned_adapter.validate()
    except StablePathError as error:
        raise GateArtifactError(str(error)) from error
    model_bytes = model_snapshot.raw_bytes
    config_bytes = config_snapshot.raw_bytes
    tensor_metadata = _safetensors_tensor_metadata(model_bytes)
    try:
        config = json.loads(config_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateArtifactError(f"adapter_config.json is invalid UTF-8 JSON: {error}") from error
    if not isinstance(config, Mapping):
        raise GateArtifactError("adapter_config.json root must be a mapping")
    if config.get("r") != 16 or isinstance(config.get("r"), bool):
        raise GateArtifactError("adapter_config.json must prove LoRA rank=16")
    if config.get("lora_alpha") != 32 or isinstance(config.get("lora_alpha"), bool):
        raise GateArtifactError("adapter_config.json must prove LoRA alpha=32")
    peft_type = config.get("peft_type")
    if not isinstance(peft_type, str) or peft_type.upper() != "LORA":
        raise GateArtifactError("adapter_config.json must identify PEFT type LORA")
    if config.get("bias") != "none":
        raise GateArtifactError("adapter_config.json must prove LoRA bias=none")
    if config.get("task_type") != "CAUSAL_LM":
        raise GateArtifactError("adapter_config.json must prove task_type=CAUSAL_LM")
    targets = config.get("target_modules")
    if (
        isinstance(targets, list)
        and targets
        and all(isinstance(value, str) and value.strip() for value in targets)
    ):
        if len(targets) != len(set(targets)):
            raise GateArtifactError(
                "adapter_config.json target_modules must not contain duplicates"
            )
        normalized_targets = sorted(targets)
    else:
        raise GateArtifactError(
            "adapter_config.json target_modules must expand the seven Qwen all-linear projections"
        )
    expected_targets = list(QWEN25_ALL_LINEAR_PROJECTIONS)
    if normalized_targets != expected_targets:
        raise GateArtifactError(
            "adapter_config.json must list exactly the seven Qwen all-linear projection suffixes"
        )
    try:
        coverage = validate_qwen_all_linear_coverage(tensor_metadata)
    except ValueError as error:
        raise GateArtifactError(
            f"adapter Qwen all-linear tensor coverage is invalid: {error}"
        ) from error
    for name, metadata in tensor_metadata.items():
        _layer, projection, side = qwen_lora_tensor_identity(name)
        shape = metadata["shape"]
        dtype = metadata["dtype"]
        if dtype not in {"F16", "BF16", "F32"} or len(shape) != 2:
            raise GateArtifactError(
                "adapter LoRA tensors must be rank-2 floating-point matrices"
            )
        input_dimension, output_dimension = QWEN25_PROJECTION_DIMENSIONS[projection]
        expected_shape = (
            [16, input_dimension] if side == "A" else [output_dimension, 16]
        )
        if shape != expected_shape:
            raise GateArtifactError(
                "adapter LoRA A/B tensor shapes must match Qwen2.5-Coder-7B dimensions "
                f"for {projection}: expected {expected_shape}, got {shape}"
            )
    if coverage.tensor_count != QWEN25_ALL_LINEAR_TENSOR_COUNT:
        raise GateArtifactError("adapter tensor count disagrees with Qwen all-linear coverage")
    return {
        "adapter_path": str(adapter_path),
        "adapter_model_path": str(model_snapshot.path),
        "adapter_config_path": str(config_snapshot.path),
        "adapter_model_sha256": model_snapshot.sha256,
        "adapter_config_sha256": config_snapshot.sha256,
        "adapter_tensor_count": coverage.tensor_count,
        "adapter_lora_layer_count": coverage.layer_count,
        "adapter_lora_projection_suffixes": list(coverage.projection_suffixes),
        "adapter_config": {
            "peft_type": "LORA",
            "r": 16,
            "lora_alpha": 32,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": normalized_targets,
        },
    }


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Publish a new UTF-8 file atomically without ever replacing existing evidence."""

    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with staging.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staging, destination)
    finally:
        staging.unlink(missing_ok=True)
    return destination


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    return atomic_write_text(path, serialized + "\n")


def gate_failure_artifact_path(artifact_path: str | Path) -> Path:
    """Return the independent immutable failure-evidence path for one gate artifact."""

    # Keep the lexical run-directory path.  Resolving a success symlink here would check or
    # publish the failure sidecar beside its external target instead of beside the gate name.
    artifact = Path(os.path.abspath(Path(artifact_path).expanduser()))
    return artifact.with_name(f"{artifact.name}.failure.json")


def exception_evidence(error: BaseException, *, stage: str) -> dict[str, str]:
    """Convert an exception into the stable, JSON-safe gate failure contract."""

    return {
        "type": type(error).__name__,
        "message": str(error),
        "stage": stage,
    }


def write_gate_failure_artifact(
    artifact_path: str | Path,
    *,
    gate: str,
    run_id: str,
    config_sha256: str | None,
    git_sha: str | None,
    dataset_sha256: str | None = None,
    error: BaseException,
    stage: str,
    rollback_error: BaseException | None = None,
) -> Path:
    """Publish failure evidence without occupying or replacing the success artifact."""

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "gate": gate,
        "passed": False,
        "run_id": run_id,
        "config_sha256": config_sha256,
        "git_sha": git_sha,
        "dataset_sha256": dataset_sha256,
        "exception": exception_evidence(error, stage=stage),
    }
    if rollback_error is not None:
        payload["rollback_exception"] = exception_evidence(
            rollback_error,
            stage="transaction_rollback",
        )
    return atomic_write_json(gate_failure_artifact_path(artifact_path), payload)


def _validate_endpoint(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ConfigValidationError(f"{field_name} must be a non-empty URL")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigValidationError(f"{field_name} must be an http(s) URL")
    if parsed.scheme == "http":
        hostname = parsed.hostname.lower()
        try:
            loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            loopback = hostname == "localhost"
        if not loopback:
            raise ConfigValidationError(
                f"{field_name} must use https unless it targets a loopback address"
            )


def _validate_resolved_verl_config(path: Path) -> None:
    """Validate the OmegaConf tree that ``server_factory`` mutates and reads later."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ConfigValidationError(f"cannot load resolved VeRL config {path}: {error}") from error
    root = _mapping(payload, "resolved VeRL config")
    actor_rollout_ref = _mapping(
        root.get("actor_rollout_ref"), "resolved VeRL config.actor_rollout_ref"
    )
    _mapping(
        actor_rollout_ref.get("model"),
        "resolved VeRL config.actor_rollout_ref.model",
    )
    _mapping(
        actor_rollout_ref.get("rollout"),
        "resolved VeRL config.actor_rollout_ref.rollout",
    )
    actor = _mapping(
        actor_rollout_ref.get("actor"),
        "resolved VeRL config.actor_rollout_ref.actor",
    )
    _mapping(
        actor.get("optim"),
        "resolved VeRL config.actor_rollout_ref.actor.optim",
    )
    _mapping(
        actor.get("policy_loss"),
        "resolved VeRL config.actor_rollout_ref.actor.policy_loss",
    )
    strategy = actor.get("strategy")
    if strategy not in {"fsdp", "fsdp2"}:
        raise ConfigValidationError(
            "resolved VeRL config.actor_rollout_ref.actor.strategy must select FSDP or FSDP2"
        )

    _mapping(root.get("algorithm"), "resolved VeRL config.algorithm")
    _mapping(root.get("reward_model"), "resolved VeRL config.reward_model")
    trainer = _mapping(root.get("trainer"), "resolved VeRL config.trainer")
    positive_integers: dict[str, int] = {}
    for field_name in ("total_epochs", "n_gpus_per_node", "nnodes"):
        value = trainer.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigValidationError(
                f"resolved VeRL config.trainer.{field_name} must be a positive integer"
            )
        positive_integers[field_name] = value
    world_size = positive_integers["n_gpus_per_node"] * positive_integers["nnodes"]
    if 8 % world_size != 0:
        raise ConfigValidationError(
            "Capsule group_size=8 must be divisible by the resolved VeRL FSDP "
            f"data-parallel world size; got {world_size}"
        )
    device = trainer.get("device")
    if not isinstance(device, str) or not device.strip():
        raise ConfigValidationError(
            "resolved VeRL config.trainer.device must be a non-empty string"
        )

    ray_kwargs = _mapping(root.get("ray_kwargs"), "resolved VeRL config.ray_kwargs")
    ray_init = ray_kwargs.get("ray_init", {})
    _mapping(ray_init, "resolved VeRL config.ray_kwargs.ray_init")
    _mapping(root.get("data"), "resolved VeRL config.data")


def _validate_environment_config_bytes(
    environment_bytes: bytes,
    config: Mapping[str, Any],
) -> None:
    """Parse and validate environment YAML without importing simulator modules."""

    try:
        environment_text = environment_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ConfigValidationError("task.config_path must be UTF-8 YAML") from error
    try:
        environment_payload = yaml.safe_load(environment_text)
    except yaml.YAMLError as error:
        raise ConfigValidationError(f"task.config_path is not valid YAML: {error}") from error
    if not isinstance(environment_payload, Mapping):
        raise ConfigValidationError("task.config_path YAML root must be a mapping")
    try:
        profile = resolve_task_profile(config)
    except CapsuleTaskProfileError as error:
        raise ConfigValidationError(str(error)) from error
    errors = collect_environment_profile_errors(environment_payload, profile)
    if errors:
        raise ConfigValidationError(
            "task.config_path does not match the selected Capsule task profile: "
            + "; ".join(errors)
        )


def load_and_validate_server_config(
    config_path: str | Path,
    *,
    check_runtime_paths: bool,
) -> dict[str, Any]:
    """Load the project template and enforce server-gate invariants.

    This function never imports torch, Robosuite, VeRL, or service clients.
    """

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise ConfigValidationError(f"config must be an existing YAML file: {path}")
    return load_and_validate_server_config_bytes(
        path.read_bytes(),
        check_runtime_paths=check_runtime_paths,
    )


def load_and_validate_server_config_bytes(
    config_bytes: bytes,
    *,
    check_runtime_paths: bool,
) -> dict[str, Any]:
    """Validate one immutable UTF-8 YAML byte snapshot without reopening its path."""

    if not isinstance(config_bytes, bytes):
        raise TypeError("config_bytes must be bytes")
    try:
        config_text = config_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ConfigValidationError("config must be UTF-8 YAML") from error
    payload = yaml.safe_load(config_text)
    config = dict(_mapping(payload, "config"))
    schema_version = config.get("schema_version")
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise ConfigValidationError("schema_version must be 1")
    try:
        validate_capsule_config(config)
    except CapsuleConfigError as error:
        raise ConfigValidationError(str(error)) from error

    runtime = _mapping(config.get("runtime"), "runtime")
    task = _mapping(config.get("task"), "task")
    program_service = _mapping(config.get("program_service"), "program_service")
    controller_service = _mapping(config.get("controller_service"), "controller_service")
    capsule = _mapping(config.get("capsule"), "capsule")

    project_root_value = runtime.get("project_root")
    project_root = (
        _resolve_path(project_root_value, repository_root())
        if isinstance(project_root_value, str) and project_root_value
        else repository_root()
    )
    path_fields = {
        "runtime.verl_source_path": (_required(runtime, "verl_source_path", str), True),
        "runtime.dataset_path": (_required(runtime, "dataset_path", str), False),
        "runtime.program_model_path": (_required(runtime, "program_model_path", str), True),
        "runtime.verl_resolved_config_path": (
            _required(runtime, "verl_resolved_config_path", str),
            False,
        ),
    }
    output_dir = _resolve_path(_required(runtime, "output_dir", str), project_root)
    pinned_sha = _required(runtime, "verl_pinned_sha", str)
    if _GIT_SHA_RE.fullmatch(pinned_sha) is None:
        raise ConfigValidationError("runtime.verl_pinned_sha must be a full lowercase Git SHA")
    requires = _mapping(runtime.get("requires"), "runtime.requires")
    if requires.get("egl") is not True or requires.get("pyroki") is not True:
        raise ConfigValidationError("runtime.requires must enable both egl and pyroki")

    for field_name in ("environment", "api"):
        _required(task, field_name, str)
    privilege = task.get("privilege")
    if privilege not in {"privileged", True}:
        raise ConfigValidationError("task.privilege must select privileged execution")
    if task.get("render") is not False:
        raise ConfigValidationError("task.render must be false for Capsule-RL")
    if task.get("record_video") is not False:
        raise ConfigValidationError("task.record_video must be false for Capsule-RL")

    for name, service in (
        ("program_service", program_service),
        ("controller_service", controller_service),
    ):
        _validate_endpoint(service.get("endpoint"), f"{name}.endpoint")
        _required(service, "model", str)
        key_env = _required(service, "api_key_env", str)
        if _ENV_NAME_RE.fullmatch(key_env) is None:
            raise ConfigValidationError(f"{name}.api_key_env is not a valid environment name")
    if program_service.get("mode") != "actor_identity":
        raise ConfigValidationError("program_service.mode must be actor_identity")
    if controller_service.get("frozen") is not True:
        raise ConfigValidationError("controller_service.frozen must be true")

    expected_capsule = {
        "group_size": 8,
        "base_samples_before_repair": 7,
        "p0_count": 2,
        "repair_trajectories_per_p0": 2,
        "max_controller_turns": 12,
        "revision_input_max_tokens": 8192,
        "revision_response_max_tokens": 2048,
    }
    for field_name, expected in expected_capsule.items():
        if capsule.get(field_name) != expected:
            raise ConfigValidationError(f"capsule.{field_name} must be {expected}")
    gamma = capsule.get("gamma")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or gamma != 0.1:
        raise ConfigValidationError("capsule.gamma must be 0.1")

    if check_runtime_paths:
        _validate_existing_path(project_root, "runtime.project_root", directory=True)
        for field_name, (value, directory) in path_fields.items():
            _validate_existing_path(
                _resolve_path(value, project_root), field_name, directory=directory
            )
        _validate_resolved_verl_config(
            _resolve_path(path_fields["runtime.verl_resolved_config_path"][0], project_root)
        )
        environment_config = task.get("config_path")
        if environment_config is not None:
            if not isinstance(environment_config, str) or not environment_config:
                raise ConfigValidationError("task.config_path must be a non-empty path")
            environment_path = _resolve_path(environment_config, project_root)
            _validate_existing_path(
                environment_path,
                "task.config_path",
                directory=False,
            )
            try:
                environment_bytes = environment_path.read_bytes()
            except OSError as error:
                raise ConfigValidationError(
                    f"cannot read task.config_path YAML {environment_path}: {error}"
                ) from error
            _validate_environment_config_bytes(environment_bytes, config)
        python_executable = runtime.get("python_executable", sys.executable)
        if not isinstance(python_executable, str) or not python_executable:
            raise ConfigValidationError("runtime.python_executable must be non-empty")
        if Path(python_executable).is_absolute():
            _validate_existing_path(
                Path(python_executable), "runtime.python_executable", directory=False
            )
        elif shutil.which(python_executable) is None:
            raise ConfigValidationError(
                f"runtime.python_executable was not found on PATH: {python_executable}"
            )
        _validate_output_path(output_dir, "runtime.output_dir")

    return config


def config_context(
    config: Mapping[str, Any], config_path: Path, artifact_path: Path
) -> dict[str, str]:
    runtime = _mapping(config["runtime"], "runtime")
    task = _mapping(config["task"], "task")
    program_service = _mapping(config["program_service"], "program_service")
    controller_service = _mapping(config["controller_service"], "controller_service")
    project_root_value = runtime.get("project_root")
    project_root = (
        _resolve_path(project_root_value, repository_root())
        if isinstance(project_root_value, str) and project_root_value
        else repository_root()
    )
    artifact = artifact_path.resolve()
    run_id = artifact.parent.name
    if not run_id:
        raise ConfigValidationError(
            "gate artifact must be inside a named run directory used as run_id"
        )
    return {
        "python": str(runtime.get("python_executable", sys.executable)),
        "project_root": str(project_root),
        "config": str(config_path.resolve()),
        "artifact": str(artifact),
        "run_id": run_id,
        "guided_artifact": str(artifact.parent / "gate05_guided_group.json"),
        "output_dir": str(_resolve_path(str(runtime["output_dir"]), project_root)),
        "dataset": str(_resolve_path(str(runtime["dataset_path"]), project_root)),
        "program_model": str(_resolve_path(str(runtime["program_model_path"]), project_root)),
        "verl_resolved_config": str(
            _resolve_path(str(runtime["verl_resolved_config_path"]), project_root)
        ),
        "environment": str(task["environment"]),
        "program_endpoint": str(program_service["endpoint"]),
        "program_service_model": str(program_service["model"]),
        "controller_endpoint": str(controller_service["endpoint"]),
        "controller_model": str(controller_service["model"]),
    }


def runtime_dataset_path(config: Mapping[str, Any]) -> Path:
    """Resolve ``runtime.dataset_path`` against the configured project root."""

    runtime = _mapping(config.get("runtime"), "runtime")
    project_root_value = runtime.get("project_root")
    project_root = (
        _resolve_path(project_root_value, repository_root())
        if isinstance(project_root_value, str) and project_root_value
        else repository_root()
    )
    return _resolve_path(
        _required(runtime, "dataset_path", str),
        project_root,
    )


@dataclass(frozen=True)
class ExternalGatePlan:
    gate_name: str
    config_path: str | Path
    artifact_path: str | Path
    runner_command: str
    placeholders: Mapping[str, str] = field(default_factory=dict)
    required_placeholders: frozenset[str] = frozenset({"config", "artifact"})
    direct_artifact_publish: bool = False


def _expand_runner_command(
    plan: ExternalGatePlan,
    *,
    artifact_override: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    config_path = Path(plan.config_path).expanduser().resolve()
    artifact_path = (
        artifact_override.resolve()
        if artifact_override is not None
        else Path(plan.artifact_path).expanduser().resolve()
    )
    config = load_and_validate_server_config(config_path, check_runtime_paths=True)
    context = config_context(config, config_path, artifact_path)
    context.update({key: str(value) for key, value in plan.placeholders.items()})
    fields = {
        field_name
        for _literal, field_name, _format_spec, _conversion in string.Formatter().parse(
            plan.runner_command
        )
        if field_name is not None
    }
    missing_required = sorted(plan.required_placeholders - fields)
    if missing_required:
        raise CommandValidationError(
            f"runner command is missing required placeholder(s): {', '.join(missing_required)}"
        )
    unknown = sorted(fields - context.keys())
    if unknown:
        raise CommandValidationError(
            f"runner command uses unknown placeholder(s): {', '.join(unknown)}"
        )
    try:
        expanded = plan.runner_command.format_map(context)
    except (KeyError, ValueError) as error:
        raise CommandValidationError(f"runner command expansion failed: {error}") from error
    if "\n" in expanded or "\r" in expanded:
        raise CommandValidationError("runner command must be a single line")
    argv = shlex.split(expanded, posix=True)
    if not argv:
        raise CommandValidationError("runner command is empty")
    operators = sorted(token for token in argv if token in _SHELL_OPERATORS)
    if operators:
        raise CommandValidationError(
            f"runner command contains forbidden shell operator: {operators[0]}"
        )
    executable = Path(argv[0])
    if executable.is_absolute():
        _validate_existing_path(executable, "runner executable", directory=False)
    elif shutil.which(argv[0]) is None:
        raise CommandValidationError(f"runner executable was not found on PATH: {argv[0]}")
    return argv, config


_EXTERNAL_GATE_NAMES = {
    "seed_determinism": "seed",
    "oracle_clean_replay": "oracle_replay",
    "controller_collector": "collector",
    "verified_guided_group": "guided",
    "one_step_trainer": "trainer",
}


def _canonical_external_gate_name(gate_name: str) -> str:
    return _EXTERNAL_GATE_NAMES.get(gate_name, gate_name)


def gate_log_artifact_path(artifact_path: str | Path, stream_name: str) -> Path:
    """Return an immutable companion path for captured child stdout or stderr."""

    if stream_name not in {"stdout", "stderr"}:
        raise ValueError("stream_name must be 'stdout' or 'stderr'")
    artifact = Path(artifact_path).expanduser().resolve()
    return artifact.with_name(f"{artifact.name}.{stream_name}.log")


def _available_git_sha(project_root: Path) -> str | None:
    """Read HEAD when available without turning wrapper diagnostics into a Git gate."""

    try:
        process = subprocess.Popen(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        stdout, _stderr = process.communicate(timeout=10)
        if process.returncode != 0:
            return None
        value = stdout.strip().lower()
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        return None
    except (OSError, subprocess.SubprocessError):
        return None
    return value if _GIT_SHA_RE.fullmatch(value) is not None else None


def _verify_failure_artifact(
    payload: Mapping[str, Any],
    *,
    gate: str,
    run_id: str,
    config_sha256: str | None,
    dataset_sha256: str | None = None,
) -> None:
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise GateArtifactError("failure artifact schema_version must be 1")
    if payload.get("gate") != gate or payload.get("passed") is not False:
        raise GateArtifactError("failure artifact gate/passed envelope is invalid")
    if payload.get("run_id") != run_id:
        raise GateArtifactError("failure artifact run_id does not match the requested gate")
    if payload.get("config_sha256") != config_sha256:
        raise GateArtifactError("failure artifact config_sha256 does not match the requested gate")
    actual_dataset_sha256 = payload.get("dataset_sha256")
    if dataset_sha256 is not None and actual_dataset_sha256 != dataset_sha256:
        raise GateArtifactError(
            "failure artifact dataset_sha256 does not match the requested gate"
        )
    if actual_dataset_sha256 is not None and (
        not isinstance(actual_dataset_sha256, str)
        or _SHA256_RE.fullmatch(actual_dataset_sha256) is None
    ):
        raise GateArtifactError("failure artifact dataset_sha256 must be null or SHA-256")
    git_sha = payload.get("git_sha")
    if git_sha is not None and (
        not isinstance(git_sha, str) or _GIT_SHA_RE.fullmatch(git_sha) is None
    ):
        raise GateArtifactError("failure artifact git_sha must be null or a full Git SHA")
    exception = payload.get("exception")
    if not isinstance(exception, Mapping):
        raise GateArtifactError("failure artifact exception must be a mapping")
    if not isinstance(exception.get("type"), str) or not exception["type"]:
        raise GateArtifactError("failure artifact exception.type must be non-empty")
    if not isinstance(exception.get("message"), str):
        raise GateArtifactError("failure artifact exception.message must be a string")
    if not isinstance(exception.get("stage"), str) or not exception["stage"]:
        raise GateArtifactError("failure artifact exception.stage must be non-empty")


def _completed_output(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def run_external_gate(
    plan: ExternalGatePlan,
    *,
    validate_only: bool,
    verifier: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[str]:
    """Validate or run one external server adapter without invoking a shell.

    Executions capture immutable stdout/stderr logs. Any error publishes an independent
    ``<artifact>.failure.json``; staged child failure evidence is promoted before cleanup.
    """

    artifact_path = Path(plan.artifact_path).expanduser().resolve()
    failure_path = gate_failure_artifact_path(artifact_path)
    stdout_path = gate_log_artifact_path(artifact_path, "stdout")
    stderr_path = gate_log_artifact_path(artifact_path, "stderr")
    if not validate_only:
        for claimed_path, kind in (
            (artifact_path, "artifact"),
            (failure_path, "failure artifact"),
            (stdout_path, "stdout log"),
            (stderr_path, "stderr log"),
        ):
            if claimed_path.exists() or claimed_path.is_symlink():
                raise FileExistsError(f"{kind} already exists: {claimed_path}")

    staging_path = None
    if not validate_only and not plan.direct_artifact_publish:
        staging_path = artifact_path.with_name(
            f".{artifact_path.name}.{uuid.uuid4().hex}.tmp"
        )
    staged_failure_path = (
        gate_failure_artifact_path(staging_path) if staging_path is not None else None
    )
    gate = _canonical_external_gate_name(plan.gate_name)
    run_id = artifact_path.parent.name
    config_sha256: str | None = None
    git_sha: str | None = None
    dataset_sha256: str | None = None
    stage = "runner_plan_validation" if validate_only else "config_hash"
    try:
        if not validate_only:
            config_sha256 = artifact_file_sha256(plan.config_path)
            stage = "runner_plan_validation"
        argv, config = _expand_runner_command(plan, artifact_override=staging_path)
        if not validate_only:
            stage = "dataset_hash"
            dataset_sha256 = artifact_file_sha256(runtime_dataset_path(config))
            stage = "runner_plan_validation"
        project_root_value = config["runtime"].get("project_root")
        project_root = (
            _resolve_path(project_root_value, repository_root())
            if isinstance(project_root_value, str) and project_root_value
            else repository_root()
        )
        if not validate_only:
            git_sha = _available_git_sha(project_root)
        print(
            json.dumps(
                {
                    "mode": "VALIDATION ONLY" if validate_only else "EXECUTE",
                    "gate": plan.gate_name,
                    "cwd": str(project_root),
                    "argv": argv,
                    "artifact": str(artifact_path),
                    "staging_artifact": (
                        str(staging_path) if staging_path is not None else None
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if validate_only:
            return argv

        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        stage = "runner_execution"
        completed = subprocess.run(
            argv,
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout = _completed_output(completed.stdout)
        stderr = _completed_output(completed.stderr)
        if stdout:
            sys.stdout.write(stdout)
        if stderr:
            sys.stderr.write(stderr)
        runner_error = (
            GateExecutionError(
                f"{plan.gate_name} runner exited with status {completed.returncode}"
            )
            if completed.returncode != 0
            else None
        )
        try:
            stage = "stdout_log_publish"
            atomic_write_text(stdout_path, stdout)
            stage = "stderr_log_publish"
            atomic_write_text(stderr_path, stderr)
        except BaseException as log_error:
            if runner_error is None:
                raise
            try:
                setattr(runner_error, "log_artifact_recording_error", log_error)
            except BaseException:
                pass
            stage = "runner_exit"
            raise runner_error from log_error
        if runner_error is not None:
            stage = "runner_exit"
            raise runner_error

        produced_artifact = artifact_path if plan.direct_artifact_publish else staging_path
        assert produced_artifact is not None
        stage = "runner_artifact"
        if not produced_artifact.is_file():
            raise GateExecutionError(
                f"{plan.gate_name} runner did not create artifact: {produced_artifact}"
            )
        stage = "artifact_verification"
        if verifier is not None:
            verifier(load_json_artifact(produced_artifact))
        if not plan.direct_artifact_publish:
            stage = "artifact_publish"
            try:
                os.link(produced_artifact, artifact_path)
            except FileExistsError as error:
                raise GateExecutionError(
                    f"{plan.gate_name} artifact appeared during execution: {artifact_path}"
                ) from error
        return argv
    except BaseException as error:
        if validate_only:
            raise
        recording_errors: list[BaseException] = []
        if staged_failure_path is not None and staged_failure_path.is_file():
            try:
                child_failure = load_json_artifact(staged_failure_path)
                _verify_failure_artifact(
                    child_failure,
                    gate=gate,
                    run_id=run_id,
                    config_sha256=config_sha256,
                    dataset_sha256=dataset_sha256,
                )
                os.link(staged_failure_path, failure_path)
            except BaseException as promotion_error:
                recording_errors.append(promotion_error)
        if failure_path.is_file():
            try:
                _verify_failure_artifact(
                    load_json_artifact(failure_path),
                    gate=gate,
                    run_id=run_id,
                    config_sha256=config_sha256,
                    dataset_sha256=dataset_sha256,
                )
            except BaseException as existing_failure_error:
                recording_errors.append(existing_failure_error)
        else:
            try:
                write_gate_failure_artifact(
                    artifact_path,
                    gate=gate,
                    run_id=run_id,
                    config_sha256=config_sha256,
                    git_sha=git_sha,
                    dataset_sha256=dataset_sha256,
                    error=error,
                    stage=stage,
                )
            except BaseException as recording_error:
                recording_errors.append(recording_error)
        if recording_errors:
            try:
                setattr(error, "failure_artifact_recording_errors", tuple(recording_errors))
            except BaseException:
                pass
        raise
    finally:
        if staging_path is not None:
            try:
                staging_path.unlink(missing_ok=True)
            except OSError:
                pass
        if (
            staged_failure_path is not None
            and failure_path.is_file()
            and staged_failure_path != failure_path
        ):
            try:
                staged_failure_path.unlink(missing_ok=True)
            except OSError:
                pass


def load_json_artifact(path: str | Path) -> Mapping[str, Any]:
    artifact_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateArtifactError(f"cannot read gate artifact {artifact_path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise GateArtifactError("gate artifact must be a JSON object")
    return payload


def _artifact_list(payload: Mapping[str, Any], field_name: str) -> list[Any]:
    value = payload.get(field_name)
    if not isinstance(value, list):
        raise GateArtifactError(f"artifact field {field_name} must be a list")
    return value


def verify_gate_envelope(payload: Mapping[str, Any], expected_gate: str) -> None:
    """Validate the common identity envelope shared by all six server gates."""

    if expected_gate not in GATE_ORDER:
        raise ValueError(f"unknown Capsule gate: {expected_gate}")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise GateArtifactError("gate artifact schema_version must be 1")
    if payload.get("gate") != expected_gate:
        raise GateArtifactError(f"gate artifact must identify itself as {expected_gate!r}")
    if payload.get("passed") is not True:
        raise GateArtifactError(f"gate artifact {expected_gate!r} must record passed=true")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise GateArtifactError("gate artifact run_id must be a non-empty string")
    config_sha256 = payload.get("config_sha256")
    if not isinstance(config_sha256, str) or _SHA256_RE.fullmatch(config_sha256) is None:
        raise GateArtifactError("gate artifact config_sha256 must be lowercase SHA-256")
    git_sha = payload.get("git_sha")
    if not isinstance(git_sha, str) or _GIT_SHA_RE.fullmatch(git_sha) is None:
        raise GateArtifactError("gate artifact git_sha must be a full lowercase Git SHA")
    dataset_sha256 = payload.get("dataset_sha256")
    if (
        not isinstance(dataset_sha256, str)
        or _SHA256_RE.fullmatch(dataset_sha256) is None
    ):
        raise GateArtifactError("gate artifact dataset_sha256 must be lowercase SHA-256")
    for field_name in (
        "resolved_environment_sha256",
        "verl_resolved_config_sha256",
        "program_model_sha256",
        "actor_binding_sha256",
    ):
        dependency_sha256 = payload.get(field_name)
        if (
            not isinstance(dependency_sha256, str)
            or _SHA256_RE.fullmatch(dependency_sha256) is None
        ):
            raise GateArtifactError(
                f"gate artifact {field_name} must be lowercase SHA-256"
            )


def _typed_replay(value: object, field_name: str) -> ProgramReplayResultV1:
    if not isinstance(value, Mapping):
        raise GateArtifactError(f"{field_name} must be a ProgramReplayResultV1 mapping")
    try:
        return ProgramReplayResultV1.from_dict(value)
    except (KeyError, TypeError, ValueError) as error:
        raise GateArtifactError(
            f"{field_name} is not a valid ProgramReplayResultV1: {error}"
        ) from error


def verify_replay_telemetry(
    payload: Mapping[str, Any],
    results: tuple[ProgramReplayResultV1, ...] | list[ProgramReplayResultV1],
) -> None:
    """Reject counters which cannot be derived from persisted typed replay evidence."""

    summary = summarize_replay_results(results, require_attempt_history=True)
    for field_name, expected in summary.items():
        actual = payload.get(field_name)
        if isinstance(actual, bool) or not isinstance(actual, int) or actual < 0:
            raise GateArtifactError(
                f"replay telemetry field {field_name} must be a non-negative integer"
            )
        if actual != expected:
            raise GateArtifactError(
                f"replay telemetry {field_name}={actual} disagrees with typed replay "
                f"evidence ({expected})"
            )


def _typed_trace(value: object, field_name: str) -> RepairTraceV1:
    if not isinstance(value, Mapping):
        raise GateArtifactError(f"{field_name} must be a RepairTraceV1 mapping")
    try:
        trace = RepairTraceV1.from_dict(value)
        trace.reconstruct()
        return trace
    except (KeyError, TypeError, ValueError, RepairInvariantError) as error:
        raise GateArtifactError(
            f"{field_name} is not a reconstructable RepairTraceV1: {error}"
        ) from error


def _verify_explicit_protocol_repairs(
    trace: RepairTraceV1,
    p0_result: ProgramReplayResultV1,
) -> None:
    if any(edit.before_source == edit.after_source for edit in trace.edits):
        raise GateArtifactError("collector trace must not commit no-op replacement edits")

    expected_spans = python_base_unit_spans(trace.base_source)
    protocol_unit_ids = ("fence_open", "fence_close", "protocol_suffix")
    expected_protocol_ids = [
        span.unit_id for span in expected_spans if span.unit_id in protocol_unit_ids
    ]
    if not expected_protocol_ids:
        return

    expected_units = [
        (
            f"base:{span.unit_id}",
            span.start_offset,
            span.end_offset,
            trace.base_source[span.start_offset : span.end_offset],
        )
        for span in expected_spans
    ]
    actual_units = [
        (unit.target, unit.start_offset, unit.end_offset, unit.source)
        for unit in trace.base_units
    ]
    if actual_units != expected_units:
        raise GateArtifactError(
            "fenced Actor P0 must expose exact explicit protocol repair units"
        )
    diagnostics = p0_result.diagnostics
    expected_executed_source = normalize_program_source(trace.base_source)
    if diagnostics.get("source_normalized") is not True:
        raise GateArtifactError("fenced Actor P0 must prove source normalization")
    if diagnostics.get("raw_source_sha256") != source_sha256(trace.base_source):
        raise GateArtifactError("fenced Actor P0 raw source hash is invalid")
    if diagnostics.get("executed_source_sha256") != source_sha256(
        expected_executed_source
    ):
        raise GateArtifactError("fenced Actor P0 executed source hash is invalid")

    required_targets = [f"base:{unit_id}" for unit_id in expected_protocol_ids]
    protocol_edit_turns: list[int] = []
    for target in required_targets:
        target_edits = [edit for edit in trace.edits if edit.target == target]
        if len(target_edits) != 1 or target_edits[0].after_source != "":
            raise GateArtifactError(
                "fenced Actor P0 requires one explicit protocol deletion per target"
            )
        protocol_edit_turns.append(target_edits[0].turn_index)
    semantic_turns = [
        edit.turn_index for edit in trace.edits if edit.target not in required_targets
    ]
    if semantic_turns and min(semantic_turns) < max(protocol_edit_turns):
        raise GateArtifactError(
            "explicit protocol deletions must precede semantic repair edits"
        )


def _typed_group(value: object, field_name: str = "learning_group") -> LearningGroupV1:
    if not isinstance(value, Mapping):
        raise GateArtifactError(f"{field_name} must be a LearningGroupV1 mapping")
    try:
        return LearningGroupV1.from_dict(value)
    except (KeyError, TypeError, ValueError) as error:
        raise GateArtifactError(f"{field_name} is not a valid LearningGroupV1: {error}") from error


def _typed_task(value: object, field_name: str = "task_instance") -> TaskInstanceV1:
    if not isinstance(value, Mapping):
        raise GateArtifactError(f"{field_name} must be a TaskInstanceV1 mapping")
    try:
        return TaskInstanceV1.from_dict(value)
    except (KeyError, TypeError, ValueError) as error:
        raise GateArtifactError(f"{field_name} is not a valid TaskInstanceV1: {error}") from error


def _require_semantic_failure(result: ProgramReplayResultV1, field_name: str) -> None:
    if result.outcome in {
        ReplayOutcome.SUCCESS,
        ReplayOutcome.INFRA_ERROR,
        ReplayOutcome.EVALUATOR_ERROR,
    } or result.binary_reward != 0:
        raise GateArtifactError(f"{field_name} must be a typed semantic clean-replay failure")


def _require_clean_success(result: ProgramReplayResultV1, field_name: str) -> None:
    if result.outcome is not ReplayOutcome.SUCCESS:
        raise GateArtifactError(f"{field_name} must be a typed clean-replay success")


def _validate_verified_group(group: LearningGroupV1) -> None:
    if len(group.members) != 8 or group.skip_actor_update:
        raise GateArtifactError("verified learning group must contain eight updateable members")
    prompts = {member.prompt for member in group.members}
    if len(prompts) != 1:
        raise GateArtifactError("all learning group members must use one original prompt")
    for index, member in enumerate(group.members):
        expected_type = "base" if index < 7 else "critique_guided_revision"
        expected_reward = 0.0 if index < 7 else 1.0
        if member.member_type != expected_type or member.reward != expected_reward:
            raise GateArtifactError(
                "learning group must be ordered as seven failed base plus one success"
            )
        if index < 7 and member.repair_trajectory_id is not None:
            raise GateArtifactError(
                "base learning members cannot carry repair trajectory provenance"
            )
    guided = group.members[-1]
    if not guided.repair_trajectory_id:
        raise GateArtifactError("guided learning member must retain repair trajectory provenance")


def verify_preflight_gate_artifact(payload: Mapping[str, Any]) -> None:
    verify_gate_envelope(payload, "preflight")
    checks = payload.get("checks")
    if not isinstance(checks, Mapping):
        raise GateArtifactError("preflight artifact checks must be a mapping")
    required_true = (
        "verl_sha_matches",
        "dependency_lock_present",
        "cuda_available",
        "egl_configured",
        "program_model_exists",
        "program_api_key_present",
        "controller_api_key_present",
        "program_endpoint_ready",
        "program_actor_identity_verified",
        "controller_endpoint_ready",
        "pyroki_endpoint_ready",
    )
    missing = [name for name in required_true if checks.get(name) is not True]
    if missing:
        raise GateArtifactError("preflight checks did not pass: " + ", ".join(missing))
    if checks.get("git_sha") != payload.get("git_sha"):
        raise GateArtifactError("preflight git SHA must match the common gate envelope")
    verl_source_path = checks.get("verl_source_path")
    verl_expected_sha = checks.get("verl_expected_sha")
    verl_actual_sha = checks.get("verl_actual_sha")
    if not isinstance(verl_source_path, str) or not Path(verl_source_path).is_absolute():
        raise GateArtifactError("preflight must record the absolute VeRL source path")
    if (
        not isinstance(verl_expected_sha, str)
        or _GIT_SHA_RE.fullmatch(verl_expected_sha) is None
        or verl_actual_sha != verl_expected_sha
    ):
        raise GateArtifactError("preflight must record the matching pinned VeRL Git SHA")
    environment_sha = checks.get("resolved_environment_sha256")
    if not isinstance(environment_sha, str) or _SHA256_RE.fullmatch(environment_sha) is None:
        raise GateArtifactError("preflight must record a resolved environment SHA-256")
    resolved_verl_sha = checks.get("verl_resolved_config_sha256")
    if not isinstance(resolved_verl_sha, str) or _SHA256_RE.fullmatch(resolved_verl_sha) is None:
        raise GateArtifactError("preflight must record the resolved VeRL config SHA-256")
    if environment_sha != payload.get("resolved_environment_sha256"):
        raise GateArtifactError(
            "preflight resolved environment SHA must match the common gate envelope"
        )
    if resolved_verl_sha != payload.get("verl_resolved_config_sha256"):
        raise GateArtifactError(
            "preflight resolved VeRL config SHA must match the common gate envelope"
        )
    for field_name in ("program_model_sha256", "actor_binding_sha256"):
        value = payload.get(field_name)
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise GateArtifactError(f"preflight {field_name} must be lowercase SHA-256")
        if checks.get(field_name) != value:
            raise GateArtifactError(
                f"preflight {field_name} must match the common gate identity"
            )
    model_file_count = checks.get("program_model_file_count")
    if (
        isinstance(model_file_count, bool)
        or not isinstance(model_file_count, int)
        or model_file_count < 1
    ):
        raise GateArtifactError("preflight program model file count must be positive")
    service_identity = checks.get("program_actor_identity")
    if not isinstance(service_identity, Mapping):
        raise GateArtifactError("preflight must record the Program actor identity response")
    try:
        verify_actor_identity_payload(service_identity, service_identity)
    except ActorIdentityError as error:
        raise GateArtifactError(f"preflight Program actor identity is invalid: {error}") from error
    if (
        service_identity.get("program_model_sha256") != payload["program_model_sha256"]
        or service_identity.get("actor_binding_sha256") != payload["actor_binding_sha256"]
        or service_identity.get("program_model_file_count") != model_file_count
    ):
        raise GateArtifactError(
            "preflight Program actor identity does not match its top-level binding"
        )
    dataset_path = checks.get("dataset_path")
    if not isinstance(dataset_path, str) or not Path(dataset_path).is_absolute():
        raise GateArtifactError("preflight must record the absolute runtime dataset path")
    if checks.get("dataset_sha256") != payload.get("dataset_sha256"):
        raise GateArtifactError(
            "preflight dataset SHA must match the common gate envelope"
        )
    dataset_count = checks.get("dataset_task_count")
    identities = checks.get("dataset_task_identities")
    if (
        isinstance(dataset_count, bool)
        or not isinstance(dataset_count, int)
        or dataset_count < 1
        or not isinstance(identities, list)
        or len(identities) != dataset_count
    ):
        raise GateArtifactError(
            "preflight dataset task identity summary must contain every typed task row"
        )
    observed: set[tuple[str, int]] = set()
    for identity in identities:
        if not isinstance(identity, Mapping):
            raise GateArtifactError("preflight dataset task identities must be mappings")
        task_id = identity.get("task_id")
        environment_seed = identity.get("environment_seed")
        if (
            not isinstance(task_id, str)
            or not task_id
            or isinstance(environment_seed, bool)
            or not isinstance(environment_seed, int)
            or environment_seed < 0
        ):
            raise GateArtifactError(
                "preflight dataset task identity requires typed task_id/environment_seed"
            )
        typed_identity = (task_id, environment_seed)
        if typed_identity in observed:
            raise GateArtifactError("preflight dataset task identities must be unique")
        observed.add(typed_identity)
    if payload.get("failed_checks") != []:
        raise GateArtifactError("successful preflight artifact must have no failed checks")


def verify_seed_gate_artifact(payload: Mapping[str, Any]) -> None:
    verify_gate_envelope(payload, "seed")
    seeds = _artifact_list(payload, "seeds")
    hashes = _artifact_list(payload, "initial_state_sha256")
    if seeds != [5, 6, 5] or len(hashes) != 3:
        raise GateArtifactError("seed gate must record the exact sequence 5,6,5")
    if any(not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None for value in hashes):
        raise GateArtifactError("seed gate hashes must be lowercase SHA-256 values")
    if hashes[0] != hashes[2] or hashes[0] == hashes[1]:
        raise GateArtifactError("seed gate requires hash(seed5)==hash(seed5) and !=hash(seed6)")


def verify_oracle_gate_artifact(payload: Mapping[str, Any]) -> None:
    verify_gate_envelope(payload, "oracle_replay")
    if payload.get("direct_replay") is not True or payload.get("controller_used") is not False:
        raise GateArtifactError("oracle gate must prove direct replay without a Controller")
    replays = _artifact_list(payload, "replays")
    if len(replays) != 2 or any(not isinstance(item, Mapping) for item in replays):
        raise GateArtifactError("oracle gate requires exactly two replay records")
    results: list[ProgramReplayResultV1] = []
    for item in replays:
        result = _typed_replay(item.get("result"), "oracle replay result")
        _require_clean_success(result, "oracle replay result")
        results.append(result)
        if item.get("reset_seed") != result.environment_seed or result.environment_seed != 5:
            raise GateArtifactError("both oracle replays must reset the configured seed")
        reset_info = result.diagnostics.get("reset_info")
        reset_evidence = (
            reset_info.get("capsule_reset_evidence")
            if isinstance(reset_info, Mapping)
            else None
        )
        if not isinstance(reset_evidence, Mapping):
            raise GateArtifactError(
                "oracle replay result omitted diagnostics.reset_info.capsule_reset_evidence"
            )
        reset_count = reset_evidence.get("api_reset_count")
        confirmed_count = reset_evidence.get("api_reset_confirmed_count")
        if (
            reset_evidence.get("namespace_fresh") is not True
            or reset_evidence.get("api_state_cleared") is not True
            or isinstance(reset_count, bool)
            or not isinstance(reset_count, int)
            or reset_count < 1
            or confirmed_count != reset_count
        ):
            raise GateArtifactError(
                "oracle replay typed reset evidence did not prove namespace/API cleanup"
            )
        for evidence in ("namespace_fresh", "api_state_cleared"):
            if item.get(evidence) is not reset_evidence.get(evidence):
                raise GateArtifactError(
                    f"oracle replay top-level {evidence} disagrees with typed diagnostics"
                )
        if item.get("watchdog_active") is not True:
            raise GateArtifactError("oracle replay is missing watchdog_active evidence")
    worker_ids = [item.get("worker_id") for item in replays]
    if not isinstance(worker_ids[0], str) or not worker_ids[0] or worker_ids[0] != worker_ids[1]:
        raise GateArtifactError("oracle replays must run consecutively in the same worker")
    identities = {
        (
            result.task_id,
            result.environment_seed,
            result.program_sample_id,
            result.source_sha256,
            result.initial_state_sha256,
        )
        for result in results
    }
    if len(identities) != 1:
        raise GateArtifactError(
            "oracle replays must preserve task, seed, program, and initial state"
        )
    verify_replay_telemetry(payload, results)


def verify_collector_gate_artifact(payload: Mapping[str, Any]) -> None:
    verify_gate_envelope(payload, "collector")
    traces = _artifact_list(payload, "repair_traces")
    if payload.get("controller_frozen") is not True:
        raise GateArtifactError("collector artifact must prove the Controller was frozen")
    if payload.get("intermediate_replay_count") != 0:
        raise GateArtifactError("repair collection must not perform intermediate simulator replay")
    if payload.get("p0_count") != 2 or payload.get("repair_trajectories_per_p0") != 2:
        raise GateArtifactError("collector gate must run 2 P0 x 2 trajectories")
    if len(traces) != 4 or any(not isinstance(record, Mapping) for record in traces):
        raise GateArtifactError("collector gate must persist four repair trace records")
    base_results_payload = _artifact_list(payload, "base_results")
    if len(base_results_payload) != 2:
        raise GateArtifactError("collector gate must persist two typed failed P0 replay results")
    base_results = [
        _typed_replay(value, f"collector base_results[{index}]")
        for index, value in enumerate(base_results_payload)
    ]
    for index, result in enumerate(base_results):
        _require_semantic_failure(result, f"collector base_results[{index}]")
    base_contexts = {
        (result.task_id, result.environment_seed, result.initial_state_sha256)
        for result in base_results
    }
    if len(base_contexts) != 1 or len({result.program_sample_id for result in base_results}) != 2:
        raise GateArtifactError(
            "collector P0 results must share one task/seed/state and be distinct"
        )
    observed_pairs: set[tuple[int, int]] = set()
    trajectory_ids: set[str] = set()
    p0_by_rank: dict[int, str] = {}
    context: tuple[str, int] | None = None
    for record in traces:
        rank = record.get("p0_rank")
        trajectory = record.get("trajectory_index")
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or isinstance(trajectory, bool)
            or not isinstance(trajectory, int)
            or rank not in {0, 1}
            or trajectory not in {0, 1}
        ):
            raise GateArtifactError(
                "collector repair records must cover each 2x2 pair exactly once"
            )
        pair = (rank, trajectory)
        if pair in observed_pairs:
            raise GateArtifactError(
                "collector repair records must cover each 2x2 pair exactly once"
            )
        observed_pairs.add(pair)
        trace = _typed_trace(record.get("trace"), "collector record trace")
        p0_result = base_results[rank]
        if (
            trace.task_id != p0_result.task_id
            or trace.environment_seed != p0_result.environment_seed
            or trace.program_sample_id != p0_result.program_sample_id
            or trace.base_source != p0_result.source
            or trace.base_source_sha256 != p0_result.source_sha256
        ):
            raise GateArtifactError("collector trace must be rooted in its typed failed P0 result")
        _verify_explicit_protocol_repairs(trace, p0_result)
        if trace.repair_trajectory_id in trajectory_ids:
            raise GateArtifactError("collector repair trajectory IDs must be unique")
        trajectory_ids.add(trace.repair_trajectory_id)
        previous_p0 = p0_by_rank.setdefault(rank, trace.program_sample_id)
        if previous_p0 != trace.program_sample_id:
            raise GateArtifactError(
                "both trajectories for one P0 rank must share program_sample_id"
            )
        trace_context = (trace.task_id, trace.environment_seed)
        if context is None:
            context = trace_context
        elif context != trace_context:
            raise GateArtifactError("all collector traces must share task and environment seed")
    if observed_pairs != {(0, 0), (0, 1), (1, 0), (1, 1)} or len(set(p0_by_rank.values())) != 2:
        raise GateArtifactError(
            "collector gate must contain two distinct P0 programs and two traces each"
        )

    selected_batch_index = payload.get("selected_batch_index")
    if (
        isinstance(selected_batch_index, bool)
        or not isinstance(selected_batch_index, int)
        or selected_batch_index < 0
    ):
        raise GateArtifactError("collector selected_batch_index must be a non-negative integer")
    selected_batch_payloads = _artifact_list(payload, "selected_batch_results")
    if len(selected_batch_payloads) != 7:
        raise GateArtifactError("collector selected batch must persist all seven typed results")
    selected_batch_results = [
        _typed_replay(value, f"collector selected_batch_results[{index}]")
        for index, value in enumerate(selected_batch_payloads)
    ]
    for index, result in enumerate(selected_batch_results):
        _require_semantic_failure(result, f"collector selected_batch_results[{index}]")

    replay_results: list[ProgramReplayResultV1] = []
    events_by_batch: dict[int, list[tuple[int, ProgramReplayResultV1]]] = {}
    for event_index, event in enumerate(_artifact_list(payload, "replay_events")):
        if not isinstance(event, Mapping):
            raise GateArtifactError("collector replay_events must contain mappings")
        batch_index = event.get("batch_index")
        base_index = event.get("base_index")
        selected = event.get("selected_batch")
        if (
            isinstance(batch_index, bool)
            or not isinstance(batch_index, int)
            or batch_index < 0
            or isinstance(base_index, bool)
            or not isinstance(base_index, int)
            or base_index not in range(7)
            or not isinstance(selected, bool)
        ):
            raise GateArtifactError("collector replay event identity is invalid")
        if selected is not (batch_index == selected_batch_index):
            raise GateArtifactError("collector replay event selected_batch flag is inconsistent")
        result = _typed_replay(event.get("result"), f"collector replay_events[{event_index}]")
        replay_results.append(result)
        events_by_batch.setdefault(batch_index, []).append((base_index, result))
    if set(events_by_batch) != set(range(selected_batch_index + 1)):
        raise GateArtifactError("collector replay events must cover every attempted batch in order")
    for batch_index, entries in events_by_batch.items():
        if sorted(index for index, _ in entries) != list(range(7)):
            raise GateArtifactError("collector replay events must cover seven unique base indices")
        entries.sort(key=lambda item: item[0])
        if batch_index == selected_batch_index and [result.to_dict() for _, result in entries] != [
            result.to_dict() for result in selected_batch_results
        ]:
            raise GateArtifactError("collector selected batch disagrees with replay events")

    discarded_payloads = _artifact_list(payload, "discarded_batches")
    if len(discarded_payloads) != selected_batch_index:
        raise GateArtifactError("collector must persist every discarded batch before selection")
    for expected_index, discarded in enumerate(discarded_payloads):
        if not isinstance(discarded, Mapping) or discarded.get("batch_index") != expected_index:
            raise GateArtifactError("collector discarded batches must be ordered and complete")
        reason = discarded.get("reason")
        if not isinstance(reason, str) or not reason:
            raise GateArtifactError("collector discarded batch requires a typed reason")
        discarded_results = [
            _typed_replay(value, f"collector discarded batch {expected_index}")
            for value in _artifact_list(discarded, "results")
        ]
        expected_results = [
            result
            for _, result in sorted(events_by_batch[expected_index], key=lambda item: item[0])
        ]
        if [result.to_dict() for result in discarded_results] != [
            result.to_dict() for result in expected_results
        ]:
            raise GateArtifactError("collector discarded batch disagrees with replay events")

    from capx.rl.capsule.group import CapsuleGroupAssembler, ProgramCandidate

    selected_candidates = [
        ProgramCandidate(result.program_sample_id, result.source)
        for result in selected_batch_results
    ]
    expected_p0_indices = CapsuleGroupAssembler._select_p0_indices(
        selected_candidates, selected_batch_results
    )
    if [result.to_dict() for result in base_results] != [
        selected_batch_results[index].to_dict() for index in expected_p0_indices
    ]:
        raise GateArtifactError(
            "collector P0 results violate deterministic rank/distance selection"
        )
    verify_replay_telemetry(payload, replay_results)


def verify_guided_gate_artifact(payload: Mapping[str, Any]) -> None:
    verify_gate_envelope(payload, "guided")
    task = _typed_task(payload.get("task_instance"))
    group = _typed_group(payload.get("learning_group"))
    _validate_verified_group(group)
    members = group.members
    original_prompt = payload.get("original_prompt")
    if not isinstance(original_prompt, str) or not original_prompt:
        raise GateArtifactError("guided gate must record the original prompt")
    if original_prompt != task.prompt:
        raise GateArtifactError("guided original_prompt must match its typed task_instance")
    if payload.get("training_input_contains_critique") is not False:
        raise GateArtifactError("guided training input must exclude P0, rho, and critique text")
    for member in members:
        if member.prompt != original_prompt:
            raise GateArtifactError("every guided-gate member must use the original prompt")
    guided = members[-1]
    base_payloads = _artifact_list(payload, "base_results")
    if len(base_payloads) != 7:
        raise GateArtifactError("guided gate must persist seven typed base replay results")
    base_results = [
        _typed_replay(value, f"guided base_results[{index}]")
        for index, value in enumerate(base_payloads)
    ]
    for index, (member, result) in enumerate(zip(members[:7], base_results, strict=True)):
        _require_semantic_failure(result, f"guided base_results[{index}]")
        if (
            result.task_id != group.task_id
            or result.environment_seed != group.environment_seed
            or result.initial_state_sha256 != group.initial_state_sha256
            or result.program_sample_id != member.program_sample_id
            or result.source != member.response
        ):
            raise GateArtifactError("guided base replay result identity does not match its member")
    repair_attempt_payloads = _artifact_list(payload, "repair_attempts")
    try:
        from capx.rl.capsule.group import GroupAssemblyResult, RepairAttempt
        from capx.rl.capsule.trainer import validate_group_provenance

        assembly = GroupAssemblyResult.from_dict(
            {
                "group": payload["learning_group"],
                "base_results": base_payloads,
                "repair_attempts": repair_attempt_payloads,
            }
        )
        validate_group_provenance(task, assembly)
    except (KeyError, TypeError, ValueError, RepairInvariantError) as error:
        raise GateArtifactError(f"guided repair_attempts provenance is invalid: {error}") from error

    selected_attempts = [attempt for attempt in assembly.repair_attempts if attempt.selected]
    if len(selected_attempts) != 1:
        raise GateArtifactError("guided gate must contain exactly one selected repair attempt")
    selected_attempt = selected_attempts[0]
    selected = payload.get("selected_repair")
    if not isinstance(selected, Mapping):
        raise GateArtifactError("guided gate must identify the selected repair")
    trace = _typed_trace(selected.get("trace"), "selected repair trace")
    p0_result = _typed_replay(selected.get("p0_result"), "selected repair P0 result")
    pt_result = _typed_replay(selected.get("pt_result"), "selected repair PT result")
    p_hat_result = _typed_replay(selected.get("p_hat_result"), "selected repair P_hat result")
    _require_semantic_failure(p0_result, "selected repair P0 result")
    _require_clean_success(pt_result, "selected repair PT result")
    _require_clean_success(p_hat_result, "selected repair P_hat result")
    common_context = (group.task_id, group.environment_seed, group.initial_state_sha256)
    for name, result in (
        ("P0", p0_result),
        ("PT", pt_result),
        ("P_hat", p_hat_result),
    ):
        if (result.task_id, result.environment_seed, result.initial_state_sha256) != common_context:
            raise GateArtifactError(f"selected repair {name} result must share group identity")
    if not any(
        result.program_sample_id == p0_result.program_sample_id
        and result.source_sha256 == p0_result.source_sha256
        for result in base_results
    ):
        raise GateArtifactError("selected P0 must be one of the seven failed base results")
    selected_p0_result = next(
        result
        for result in base_results
        if result.program_sample_id == selected_attempt.p0_program_sample_id
    )
    selected_p0_rank = selected.get("p0_rank")
    selected_trajectory_index = selected.get("trajectory_index")
    if (
        isinstance(selected_p0_rank, bool)
        or not isinstance(selected_p0_rank, int)
        or selected_p0_rank != selected_attempt.p0_rank
        or isinstance(selected_trajectory_index, bool)
        or not isinstance(selected_trajectory_index, int)
        or selected_trajectory_index != selected_attempt.trajectory_index
        or selected_attempt.trace is None
        or selected_attempt.pt_result is None
        or selected_attempt.revision_result is None
        or trace.to_dict() != selected_attempt.trace.to_dict()
        or p0_result.to_dict() != selected_p0_result.to_dict()
        or pt_result.to_dict() != selected_attempt.pt_result.to_dict()
        or p_hat_result.to_dict() != selected_attempt.revision_result.to_dict()
    ):
        raise GateArtifactError(
            "selected repair summary must exactly match the selected typed repair_attempt"
        )
    if (
        trace.task_id != p0_result.task_id
        or trace.environment_seed != p0_result.environment_seed
        or trace.program_sample_id != p0_result.program_sample_id
        or trace.base_source_sha256 != p0_result.source_sha256
        or trace.final_source != pt_result.source
        or guided.repair_trajectory_id != trace.repair_trajectory_id
        or guided.program_sample_id != p_hat_result.program_sample_id
        or guided.response != p_hat_result.source
    ):
        raise GateArtifactError(
            "selected repair lineage does not match P0, PT, P_hat, and guided member"
        )

    selected_group_attempt_index = payload.get("selected_group_attempt_index")
    if (
        isinstance(selected_group_attempt_index, bool)
        or not isinstance(selected_group_attempt_index, int)
        or selected_group_attempt_index < 0
    ):
        raise GateArtifactError(
            "guided selected_group_attempt_index must be a non-negative integer"
        )
    replay_results: list[ProgramReplayResultV1] = []
    events_by_attempt: dict[int, list[tuple[int, bool, ProgramReplayResultV1]]] = {}
    for event_index, event in enumerate(_artifact_list(payload, "replay_events")):
        if not isinstance(event, Mapping):
            raise GateArtifactError("guided replay_events must contain mappings")
        attempt_index = event.get("group_attempt_index")
        result_index = event.get("result_index")
        selected_group = event.get("selected_group")
        if (
            isinstance(attempt_index, bool)
            or not isinstance(attempt_index, int)
            or attempt_index < 0
            or isinstance(result_index, bool)
            or not isinstance(result_index, int)
            or result_index < 0
            or not isinstance(selected_group, bool)
        ):
            raise GateArtifactError("guided replay event identity is invalid")
        if selected_group is not (attempt_index == selected_group_attempt_index):
            raise GateArtifactError("guided replay event selected_group flag is inconsistent")
        result = _typed_replay(event.get("result"), f"guided replay_events[{event_index}]")
        if (
            result.task_id,
            result.environment_seed,
            result.initial_state_sha256,
        ) != common_context:
            raise GateArtifactError("guided replay event does not share the group identity")
        replay_results.append(result)
        events_by_attempt.setdefault(attempt_index, []).append(
            (result_index, selected_group, result)
        )
    if set(events_by_attempt) != set(range(selected_group_attempt_index + 1)):
        raise GateArtifactError("guided replay events must cover every group attempt in order")
    for entries in events_by_attempt.values():
        if sorted(index for index, _, _ in entries) != list(range(len(entries))):
            raise GateArtifactError("guided replay result indices must be unique and contiguous")

    selected_event_results = [
        result
        for _, _, result in sorted(
            events_by_attempt[selected_group_attempt_index], key=lambda item: item[0]
        )
    ]
    expected_selected_results = list(assembly.base_results)
    for attempt in assembly.repair_attempts:
        if attempt.pt_result is not None:
            expected_selected_results.append(attempt.pt_result)
        if attempt.revision_result is not None:
            expected_selected_results.append(attempt.revision_result)
    if [result.to_dict() for result in selected_event_results] != [
        result.to_dict() for result in expected_selected_results
    ]:
        raise GateArtifactError("guided selected replay events disagree with group provenance")

    discarded_payloads = _artifact_list(payload, "discarded_group_attempts")
    if len(discarded_payloads) != selected_group_attempt_index:
        raise GateArtifactError("guided gate must persist every discarded group attempt")
    for expected_index, discarded in enumerate(discarded_payloads):
        if (
            not isinstance(discarded, Mapping)
            or discarded.get("group_attempt_index") != expected_index
        ):
            raise GateArtifactError("guided discarded attempts must be ordered and complete")
        if not isinstance(discarded.get("reason"), str) or not discarded.get("reason"):
            raise GateArtifactError("guided discarded attempt requires a reason")
        if not isinstance(discarded.get("message"), str) or not discarded.get("message"):
            raise GateArtifactError("guided discarded attempt requires a message")
        persisted_results = [
            _typed_replay(value, f"guided discarded attempt {expected_index}")
            for value in _artifact_list(discarded, "replay_results")
        ]
        event_results = [
            result
            for _, _, result in sorted(
                events_by_attempt[expected_index], key=lambda item: item[0]
            )
        ]
        if [result.to_dict() for result in persisted_results] != [
            result.to_dict() for result in event_results
        ]:
            raise GateArtifactError("guided discarded attempt disagrees with replay events")
        partial_payloads = _artifact_list(discarded, "partial_repair_attempts")
        try:
            partial_attempts = tuple(
                RepairAttempt.from_dict(_mapping(value, "partial repair attempt"))
                for value in partial_payloads
            )
            for partial_attempt in partial_attempts:
                if partial_attempt.trace is not None:
                    partial_attempt.trace.reconstruct()
        except (KeyError, TypeError, ValueError, RepairInvariantError) as error:
            raise GateArtifactError(
                f"guided discarded partial repair lineage is invalid: {error}"
            ) from error
        assembly_payload = discarded.get("assembly")
        if assembly_payload is not None:
            if partial_attempts:
                raise GateArtifactError(
                    "guided discarded fallback must use its full assembly, not partial attempts"
                )
            try:
                discarded_assembly = GroupAssemblyResult.from_dict(
                    _mapping(assembly_payload, "discarded group assembly")
                )
                validate_group_provenance(task, discarded_assembly)
            except (KeyError, TypeError, ValueError, RepairInvariantError) as error:
                raise GateArtifactError(
                    f"guided discarded group assembly is invalid: {error}"
                ) from error
            if discarded_assembly.group.metadata.get("guided_member_selected") is True:
                raise GateArtifactError("discarded fallback assembly unexpectedly selected guided")
            # The assembler evaluates the seven repair-triggering base programs first,
            # then PT/P_hat candidates, and only then samples the eighth fallback base
            # when no guided revision succeeds.  Preserve that execution order here;
            # ``base_results`` stores all eight bases together for group provenance.
            assembly_replays = list(discarded_assembly.base_results[:7])
            for repair_attempt in discarded_assembly.repair_attempts:
                if repair_attempt.pt_result is not None:
                    assembly_replays.append(repair_attempt.pt_result)
                if repair_attempt.revision_result is not None:
                    assembly_replays.append(repair_attempt.revision_result)
            assembly_replays.extend(discarded_assembly.base_results[7:])
            if [result.to_dict() for result in event_results] != [
                result.to_dict() for result in assembly_replays
            ]:
                raise GateArtifactError(
                    "guided discarded fallback replay events disagree with its assembly"
                )
        else:
            lineage_replays: list[ProgramReplayResultV1] = []
            for partial_attempt in partial_attempts:
                if partial_attempt.pt_result is not None:
                    lineage_replays.append(partial_attempt.pt_result)
                if partial_attempt.revision_result is not None:
                    lineage_replays.append(partial_attempt.revision_result)
            event_identities = {
                (result.program_sample_id, result.source_sha256) for result in event_results
            }
            if any(
                (result.program_sample_id, result.source_sha256) not in event_identities
                for result in lineage_replays
            ):
                raise GateArtifactError(
                    "guided discarded partial repair lineage is missing typed replay evidence"
                )
    verify_replay_telemetry(payload, replay_results)


def _validate_verl_provenance(value: object, world_size: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GateArtifactError("trainer gate must record VeRL provenance mappings")
    source_path = value.get("source_path")
    expected_sha = value.get("expected_sha")
    actual_sha = value.get("actual_sha")
    clean = value.get("clean")
    worker_count = value.get("worker_count")
    ranks = value.get("worker_ranks")
    module_paths = value.get("worker_module_paths")
    if not isinstance(source_path, str) or not Path(source_path).is_absolute():
        raise GateArtifactError("VeRL provenance source_path must be absolute")
    if (
        not isinstance(expected_sha, str)
        or _GIT_SHA_RE.fullmatch(expected_sha) is None
        or actual_sha != expected_sha
        or clean is not True
    ):
        raise GateArtifactError("VeRL provenance must prove the expected clean Git SHA")
    if worker_count != world_size or isinstance(worker_count, bool):
        raise GateArtifactError("VeRL provenance worker_count must match data parallelism")
    if (
        not isinstance(ranks, list)
        or len(ranks) != worker_count
        or any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 for rank in ranks)
        or len(set(ranks)) != worker_count
    ):
        raise GateArtifactError("VeRL provenance must record unique non-negative worker ranks")
    package_root = (Path(source_path) / "verl").resolve()
    if not isinstance(module_paths, list) or len(module_paths) != worker_count:
        raise GateArtifactError("VeRL provenance must record every worker module path")
    for module_path in module_paths:
        if not isinstance(module_path, str) or not Path(module_path).is_absolute():
            raise GateArtifactError("VeRL worker module paths must be absolute")
        if not Path(module_path).resolve().is_relative_to(package_root):
            raise GateArtifactError("VeRL worker module path is outside the pinned checkout")
    return dict(value)


def _validate_lora_runtime_evidence(value: object, world_size: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GateArtifactError("trainer gate must record LoRA runtime evidence")
    if (
        value.get("lora_rank") != 16
        or isinstance(value.get("lora_rank"), bool)
        or value.get("lora_alpha") != 32
        or isinstance(value.get("lora_alpha"), bool)
        or value.get("lora_target_modules") != ["all-linear"]
    ):
        raise GateArtifactError("trainer LoRA runtime must be rank=16, alpha=32, all-linear")
    worker_count = value.get("worker_count")
    ranks = value.get("worker_ranks")
    if (
        isinstance(worker_count, bool)
        or worker_count != world_size
        or ranks != list(range(world_size))
    ):
        raise GateArtifactError("trainer LoRA runtime must cover every actor rank")
    total = value.get("total_parameter_count")
    trainable = value.get("trainable_parameter_count")
    non_lora = value.get("non_lora_trainable_parameter_count")
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or total <= 0
        or isinstance(trainable, bool)
        or not isinstance(trainable, int)
        or trainable <= 0
        or trainable > total
        or non_lora != 0
        or isinstance(non_lora, bool)
        or value.get("only_lora_trainable") is not True
    ):
        raise GateArtifactError("trainer gate must prove that only LoRA parameters are trainable")
    if (
        value.get("lora_layer_count") != QWEN25_CODER_7B_LAYER_COUNT
        or value.get("lora_projection_suffixes")
        != list(QWEN25_ALL_LINEAR_PROJECTIONS)
        or value.get("lora_tensor_count_per_worker")
        != QWEN25_ALL_LINEAR_TENSOR_COUNT
    ):
        raise GateArtifactError(
            "trainer LoRA runtime must prove complete Qwen all-linear A/B coverage"
        )
    cuda_peak = value.get("cuda_peak_reserved_bytes")
    worker_host_min = value.get("host_mem_available_min_bytes")
    if (
        isinstance(cuda_peak, bool)
        or not isinstance(cuda_peak, int)
        or cuda_peak < 0
        or cuda_peak > 70 * 1024**3
    ):
        raise GateArtifactError("trainer LoRA CUDA peak reserved memory must not exceed 70 GiB")
    if (
        isinstance(worker_host_min, bool)
        or not isinstance(worker_host_min, int)
        or worker_host_min <= 0
    ):
        raise GateArtifactError("trainer LoRA worker MemAvailable evidence is invalid")
    names_sha256s = value.get("trainable_parameter_name_sha256s")
    workers = value.get("workers")
    if (
        not isinstance(names_sha256s, list)
        or len(names_sha256s) != world_size
        or any(
            not isinstance(item, str) or _SHA256_RE.fullmatch(item) is None
            for item in names_sha256s
        )
        or not isinstance(workers, list)
        or len(workers) != world_size
    ):
        raise GateArtifactError("trainer LoRA evidence must bind every rank's trainable names")
    normalized_workers: list[dict[str, Any]] = []
    for expected_rank, (record, expected_names_sha) in enumerate(
        zip(workers, names_sha256s, strict=True)
    ):
        if not isinstance(record, Mapping):
            raise GateArtifactError("trainer LoRA worker evidence must be mappings")
        if (
            record.get("rank") != expected_rank
            or record.get("only_lora_trainable") is not True
            or record.get("non_lora_trainable_parameter_count") != 0
            or record.get("trainable_parameter_names_sha256") != expected_names_sha
            or record.get("trainable_tensor_count") != QWEN25_ALL_LINEAR_TENSOR_COUNT
            or record.get("lora_layer_count") != QWEN25_CODER_7B_LAYER_COUNT
            or record.get("lora_projection_suffixes")
            != list(QWEN25_ALL_LINEAR_PROJECTIONS)
        ):
            raise GateArtifactError("trainer LoRA worker found non-LoRA trainable parameters")
        for name in (
            "total_parameter_count",
            "trainable_parameter_count",
            "trainable_tensor_count",
            "cuda_peak_reserved_bytes",
            "host_mem_available_bytes",
        ):
            count = record.get(name)
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise GateArtifactError("trainer LoRA worker counts must be positive integers")
        if record["cuda_peak_reserved_bytes"] > 70 * 1024**3:
            raise GateArtifactError("trainer LoRA worker CUDA peak exceeds 70 GiB")
        normalized_workers.append(dict(record))
    if sum(record["total_parameter_count"] for record in normalized_workers) != total:
        raise GateArtifactError("trainer LoRA total parameter count disagrees across evidence")
    if sum(record["trainable_parameter_count"] for record in normalized_workers) != trainable:
        raise GateArtifactError("trainer LoRA trainable parameter count disagrees across evidence")
    return dict(value)


def verify_trainer_gate_artifact(payload: Mapping[str, Any]) -> None:
    verify_gate_envelope(payload, "trainer")
    group = _typed_group(payload.get("learning_group"))
    _validate_verified_group(group)
    if payload.get("loss_mode") != "capsule_critique":
        raise GateArtifactError("trainer gate must use loss_mode='capsule_critique'")
    capsule_gamma = payload.get("capsule_gamma")
    if (
        isinstance(capsule_gamma, bool)
        or not isinstance(capsule_gamma, (int, float))
        or float(capsule_gamma) != 0.1
    ):
        raise GateArtifactError("trainer gate must use capsule_gamma=0.1")
    if payload.get("reference_kl_enabled") is not True:
        raise GateArtifactError("trainer gate must keep reference KL enabled")
    if payload.get("reference_policy_mode") != "actor_base_adapter_disabled":
        raise GateArtifactError(
            "trainer gate requires reference_policy_mode=actor_base_adapter_disabled"
        )
    reference_kl_coef = payload.get("reference_kl_coef")
    if (
        isinstance(reference_kl_coef, bool)
        or not isinstance(reference_kl_coef, (int, float))
        or not math.isfinite(float(reference_kl_coef))
        or reference_kl_coef <= 0
    ):
        raise GateArtifactError("trainer reference_kl_coef must be positive and finite")
    if payload.get("rollout_mode") != "sync":
        raise GateArtifactError("trainer gate requires synchronous rollout mode")
    ppo_epochs = payload.get("ppo_epochs")
    if isinstance(ppo_epochs, bool) or ppo_epochs != 1:
        raise GateArtifactError("trainer gate requires ppo_epochs=1")
    ppo_mini_batch_size = payload.get("ppo_mini_batch_size")
    if isinstance(ppo_mini_batch_size, bool) or ppo_mini_batch_size != 8:
        raise GateArtifactError("trainer gate requires ppo_mini_batch_size=8")
    data_parallel_world_size = payload.get("data_parallel_world_size")
    if (
        isinstance(data_parallel_world_size, bool)
        or not isinstance(data_parallel_world_size, int)
        or data_parallel_world_size < 1
        or 8 % data_parallel_world_size != 0
    ):
        raise GateArtifactError(
            "trainer data_parallel_world_size must be a positive divisor of 8"
        )
    sequence_parallel_size = payload.get("sequence_parallel_size")
    if isinstance(sequence_parallel_size, bool) or sequence_parallel_size != 1:
        raise GateArtifactError("trainer gate requires sequence_parallel_size=1")
    provenance_before = _validate_verl_provenance(
        payload.get("verl_provenance_before"), data_parallel_world_size
    )
    provenance_after = _validate_verl_provenance(
        payload.get("verl_provenance_after"), data_parallel_world_size
    )
    if provenance_after != provenance_before:
        raise GateArtifactError("VeRL provenance changed during the trainer gate")
    lora_before = _validate_lora_runtime_evidence(
        payload.get("lora_runtime_before"), data_parallel_world_size
    )
    lora_after = _validate_lora_runtime_evidence(
        payload.get("lora_runtime_after"), data_parallel_world_size
    )
    stable_lora_fields = (
        "lora_rank",
        "lora_alpha",
        "lora_target_modules",
        "worker_count",
        "worker_ranks",
        "trainable_parameter_name_sha256s",
        "total_parameter_count",
        "trainable_parameter_count",
        "non_lora_trainable_parameter_count",
        "only_lora_trainable",
        "lora_layer_count",
        "lora_projection_suffixes",
        "lora_tensor_count_per_worker",
    )
    if any(lora_after.get(field) != lora_before.get(field) for field in stable_lora_fields):
        raise GateArtifactError("trainer LoRA trainability evidence changed during Gate 6")
    cuda_peak_reserved = payload.get("cuda_peak_reserved_bytes")
    expected_cuda_peak = max(
        lora_before["cuda_peak_reserved_bytes"], lora_after["cuda_peak_reserved_bytes"]
    )
    if (
        isinstance(cuda_peak_reserved, bool)
        or not isinstance(cuda_peak_reserved, int)
        or cuda_peak_reserved != expected_cuda_peak
        or cuda_peak_reserved > 70 * 1024**3
    ):
        raise GateArtifactError("trainer CUDA peak reserved memory must not exceed 70 GiB")
    host_memory = payload.get("host_memory")
    if not isinstance(host_memory, Mapping):
        raise GateArtifactError("trainer gate must record host memory monitoring evidence")
    sample_count = host_memory.get("sample_count")
    minimum_host_available = host_memory.get("minimum_mem_available_bytes")
    poll_interval_s = host_memory.get("poll_interval_s")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 5
        or isinstance(minimum_host_available, bool)
        or not isinstance(minimum_host_available, int)
        or minimum_host_available < 12 * 1024**3
        or isinstance(poll_interval_s, bool)
        or not isinstance(poll_interval_s, (int, float))
        or float(poll_interval_s) != 0.25
        or host_memory.get("monitor_scope")
        != "before_worker_start_through_after_ray_shutdown"
    ):
        raise GateArtifactError(
            "trainer host monitor must sample the lifecycle and stay above 12 GiB"
        )
    ray_release = payload.get("ray_release")
    if not isinstance(ray_release, Mapping) or ray_release != {
        "worker_close_calls": 1,
        "ray_shutdown_calls": 1,
        "ray_shutdown_complete": True,
    }:
        raise GateArtifactError("trainer must close workers and shut Ray down exactly once")
    optimizer_steps = payload.get("optimizer_steps")
    if isinstance(optimizer_steps, bool) or optimizer_steps != 1:
        raise GateArtifactError("trainer gate must perform exactly one optimizer step")
    actor_update_rpcs = payload.get("actor_update_rpcs")
    if isinstance(actor_update_rpcs, bool) or actor_update_rpcs != 1:
        raise GateArtifactError("trainer gate must perform exactly one actor update RPC")
    optimizer_step_before = payload.get("optimizer_step_before")
    optimizer_step_after = payload.get("optimizer_step_after")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (optimizer_step_before, optimizer_step_after)
    ) or optimizer_step_after - optimizer_step_before != 1:
        raise GateArtifactError(
            "trainer optimizer before/after evidence must prove an exact step delta of one"
        )
    gradient_norm = payload.get("gradient_norm")
    if (
        isinstance(gradient_norm, bool)
        or not isinstance(gradient_norm, (int, float))
        or not math.isfinite(float(gradient_norm))
        or gradient_norm <= 0
    ):
        raise GateArtifactError("trainer gradient norm must be finite and non-zero")
    rewards = [member.reward for member in group.members]
    if rewards != [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]:
        raise GateArtifactError("trainer gate input must be the verified 7+1 reward group")
    if payload.get("group_rewards") != [0, 0, 0, 0, 0, 0, 0, 1]:
        raise GateArtifactError(
            "trainer artifact reward summary must match its typed learning group"
        )
    if payload.get("guided_token_mask_present") is not True:
        raise GateArtifactError("trainer gate must prove the guided token mask reached the actor")
    guided_token_count = payload.get("guided_token_count")
    if (
        isinstance(guided_token_count, bool)
        or not isinstance(guided_token_count, int)
        or guided_token_count < 1
    ):
        raise GateArtifactError("trainer gate must record a positive guided token count")
    if payload.get("guided_mask_response_only") is not True:
        raise GateArtifactError("trainer guided token mask must be limited to response tokens")
    guided_shape = payload.get("guided_token_mask_shape")
    reference_shape = payload.get("reference_log_prob_shape")
    if (
        not isinstance(guided_shape, list)
        or len(guided_shape) != 2
        or guided_shape[0] != 8
        or isinstance(guided_shape[1], bool)
        or not isinstance(guided_shape[1], int)
        or guided_shape[1] < 1
        or reference_shape != guided_shape
    ):
        raise GateArtifactError(
            "trainer guided mask and reference log-prob shapes must agree for all 8 responses"
        )
    expected_guided_rows = [
        index
        for index, member in enumerate(group.members)
        if member.member_type == "critique_guided_revision"
    ]
    if (
        payload.get("guided_row_indices") != expected_guided_rows
        or payload.get("rollout_mask_matches_guided") is not True
        or payload.get("old_log_probs_finite") is not True
        or payload.get("reference_log_probs_finite") is not True
        or payload.get("training_call_trace")
        != ["old_logprob", "reference_logprob", "update"]
    ):
        raise GateArtifactError(
            "trainer must derive guided-mask/reference-KL evidence from the exact update batch"
        )
    response_token_counts = payload.get("reference_log_prob_response_token_counts")
    if (
        not isinstance(response_token_counts, list)
        or len(response_token_counts) != 8
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 1
            for count in response_token_counts
        )
        or any(count > guided_shape[1] for count in response_token_counts)
    ):
        raise GateArtifactError(
            "trainer reference log-prob response mask must cover each of the 8 responses"
        )
    if payload.get("rollout_is") is not False:
        raise GateArtifactError("trainer gate must keep standard rollout importance sampling off")
    if payload.get("norm_adv_by_std_in_grpo") is not False:
        raise GateArtifactError("trainer gate must keep GRPO std normalization off")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        raise GateArtifactError("trainer gate must record non-empty metrics")
    for name, value in metrics.items():
        if (
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise GateArtifactError("trainer metrics must be named finite numeric values")
    if payload.get("actor_update_skipped") is not False:
        raise GateArtifactError("trainer gate must prove the actor update was not skipped")
    checkpoint = payload.get("checkpoint")
    checkpoint_path = Path(checkpoint) if isinstance(checkpoint, str) and checkpoint else None
    if checkpoint_path is None or not checkpoint_path.is_absolute() or not checkpoint_path.exists():
        raise GateArtifactError("trainer gate must record an existing checkpoint path")
    checkpoint_file_count = payload.get("checkpoint_file_count")
    if (
        isinstance(checkpoint_file_count, bool)
        or not isinstance(checkpoint_file_count, int)
        or checkpoint_file_count < 1
        or checkpoint_file_count != artifact_tree_file_count(checkpoint_path)
    ):
        raise GateArtifactError("trainer checkpoint_file_count does not match checkpoint contents")
    checkpoint_sha256 = payload.get("checkpoint_sha256")
    if (
        not isinstance(checkpoint_sha256, str)
        or _SHA256_RE.fullmatch(checkpoint_sha256) is None
        or checkpoint_sha256 != artifact_tree_sha256(checkpoint_path)
    ):
        raise GateArtifactError("trainer checkpoint_sha256 does not match checkpoint contents")
    guided_artifact_sha256 = payload.get("guided_artifact_sha256")
    if (
        not isinstance(guided_artifact_sha256, str)
        or _SHA256_RE.fullmatch(guided_artifact_sha256) is None
    ):
        raise GateArtifactError("trainer gate must record the guided artifact SHA-256")
    checkpoint_manifest = payload.get("checkpoint_manifest")
    manifest_path = (
        Path(checkpoint_manifest)
        if isinstance(checkpoint_manifest, str) and checkpoint_manifest
        else None
    )
    if manifest_path is None or not manifest_path.is_absolute() or not manifest_path.is_file():
        raise GateArtifactError("trainer gate must record an existing checkpoint manifest")
    manifest = load_json_artifact(manifest_path)
    expected_manifest_fields = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_file_count": checkpoint_file_count,
        "checkpoint_sha256": checkpoint_sha256,
        "optimizer_step_before": optimizer_step_before,
        "optimizer_step_after": optimizer_step_after,
        "optimizer_step_delta": 1,
    }
    if any(manifest.get(key) != value for key, value in expected_manifest_fields.items()):
        raise GateArtifactError("checkpoint manifest does not match trainer gate evidence")
    adapter_evidence = direct_lora_adapter_evidence(checkpoint_path)
    if any(payload.get(key) != value for key, value in adapter_evidence.items()):
        raise GateArtifactError("trainer direct LoRA adapter evidence does not match checkpoint")


def verify_adapter_reload_artifact(payload: Mapping[str, Any]) -> None:
    """Validate the post-Ray, fresh-FP32 PEFT adapter reload smoke artifact."""

    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("gate") != "adapter_reload"
        or payload.get("passed") is not True
        or payload.get("execution_mode") != CANONICAL_EXECUTION_MODE
    ):
        raise GateArtifactError("adapter reload artifact envelope is invalid")
    gate6_value = payload.get("gate06_artifact")
    gate6_path = Path(gate6_value) if isinstance(gate6_value, str) else None
    if gate6_path is None or not gate6_path.is_absolute():
        raise GateArtifactError("adapter reload must record an absolute Gate 6 artifact path")
    try:
        gate6_snapshot = read_stable_regular_file(
            gate6_path, label="Gate 6 trainer artifact"
        )
        gate6_payload = json.loads(gate6_snapshot.raw_bytes)
    except (StablePathError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateArtifactError(f"cannot read bound Gate 6 trainer artifact: {error}") from error
    if not isinstance(gate6_payload, Mapping):
        raise GateArtifactError("bound Gate 6 trainer artifact must be a JSON mapping")
    verify_trainer_gate_artifact(gate6_payload)
    if payload.get("gate06_artifact_sha256") != gate6_snapshot.sha256:
        raise GateArtifactError("adapter reload Gate 6 artifact SHA-256 does not match")
    identity_fields = (
        "run_id",
        "config_sha256",
        "git_sha",
        "dataset_sha256",
        "resolved_environment_sha256",
        "verl_resolved_config_sha256",
        "program_model_sha256",
        "actor_binding_sha256",
    )
    if any(payload.get(field) != gate6_payload.get(field) for field in identity_fields):
        raise GateArtifactError("adapter reload identity does not match Gate 6")
    if payload.get("ray_release") != gate6_payload.get("ray_release") or payload.get(
        "ray_release"
    ) != {
        "worker_close_calls": 1,
        "ray_shutdown_calls": 1,
        "ray_shutdown_complete": True,
    }:
        raise GateArtifactError("adapter reload requires proof that Ray was released exactly once")
    for field in ("adapter_path", "adapter_model_sha256", "adapter_config_sha256"):
        if payload.get(field) != gate6_payload.get(field):
            raise GateArtifactError(f"adapter reload {field} does not match Gate 6")
    base_model_path = payload.get("base_model_path")
    if (
        not isinstance(base_model_path, str)
        or not base_model_path
        or not Path(base_model_path).is_absolute()
        or payload.get("base_model_dtype") != "float32"
        or payload.get("device") != "cuda:0"
    ):
        raise GateArtifactError("adapter reload must use an absolute FP32 base on cuda:0")
    prompt_sha256 = payload.get("prompt_sha256")
    input_token_count = payload.get("input_token_count")
    if (
        not isinstance(prompt_sha256, str)
        or prompt_sha256 != _ADAPTER_RELOAD_PROMPT_SHA256
        or isinstance(input_token_count, bool)
        or not isinstance(input_token_count, int)
        or input_token_count < 1
    ):
        raise GateArtifactError("adapter reload fixed prompt evidence is invalid")
    if (
        payload.get("adapter_disabled_logits_finite") is not True
        or payload.get("adapter_enabled_logits_finite") is not True
    ):
        raise GateArtifactError("adapter reload logits must both be finite")
    max_difference = payload.get("max_abs_logit_diff")
    if (
        isinstance(max_difference, bool)
        or not isinstance(max_difference, (int, float))
        or not math.isfinite(float(max_difference))
        or max_difference <= 1e-8
    ):
        raise GateArtifactError("adapter reload must change logits by more than 1e-8")
    cuda_peak = payload.get("cuda_peak_reserved_bytes")
    if (
        isinstance(cuda_peak, bool)
        or not isinstance(cuda_peak, int)
        or cuda_peak <= 0
        or cuda_peak > 70 * 1024**3
    ):
        raise GateArtifactError("adapter reload CUDA peak reserved memory must stay below 70 GiB")
    host_before = payload.get("host_mem_available_before_bytes")
    host_after = payload.get("host_mem_available_after_bytes")
    host_minimum = payload.get("host_mem_available_min_bytes")
    reload_host_memory = payload.get("host_memory")
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (host_before, host_after, host_minimum)
        )
        or host_minimum > min(host_before, host_after)
        or host_minimum < 12 * 1024**3
    ):
        raise GateArtifactError("adapter reload host MemAvailable must stay above 12 GiB")
    reload_sample_count = (
        reload_host_memory.get("sample_count")
        if isinstance(reload_host_memory, Mapping)
        else None
    )
    if (
        not isinstance(reload_host_memory, Mapping)
        or isinstance(reload_sample_count, bool)
        or not isinstance(reload_sample_count, int)
        or reload_sample_count < 6
        or reload_host_memory.get("minimum_mem_available_bytes") != host_minimum
        or reload_host_memory.get("poll_interval_s") != 0.25
        or reload_host_memory.get("monitor_scope")
        != "adapter_reload_model_load_through_cuda_release"
    ):
        raise GateArtifactError(
            "adapter reload must poll host MemAvailable across model load and CUDA release"
        )


def external_gate_main(
    *,
    gate_name: str,
    description: str,
    placeholders: Mapping[str, str],
    required_placeholders: frozenset[str],
    verifier: Callable[[Mapping[str, Any]], None],
    default_runner_command: str | None = None,
    direct_artifact_publish: bool = False,
    lock_runner_command: bool = False,
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    if lock_runner_command:
        if default_runner_command is None:
            raise ValueError("a locked runner command requires a repository default")
    else:
        parser.add_argument(
            "--runner-command",
            required=default_runner_command is None,
            default=default_runner_command,
            help=(
                "Optional quoted command template override; executed as argv without a shell. "
                "The repository server adapter is used by default."
            ),
        )
    add_validation_arguments(parser)
    args = parser.parse_args(argv)
    validate_only = validation_requested(args)
    plan = ExternalGatePlan(
        gate_name=gate_name,
        config_path=args.config,
        artifact_path=args.artifact,
        runner_command=(
            default_runner_command if lock_runner_command else args.runner_command
        ),
        placeholders=placeholders,
        required_placeholders=required_placeholders,
        direct_artifact_publish=direct_artifact_publish,
    )
    expanded_argv = run_external_gate(
        plan,
        validate_only=validate_only,
        verifier=None if validate_only else verifier,
    )
    uses_repository_adapter = default_runner_command is not None and (
        lock_runner_command or args.runner_command == default_runner_command
    )
    if validate_only and uses_repository_adapter:
        if expanded_argv[1:3] != ["-m", "scripts.capsule_rl.server_adapter"]:
            raise CommandValidationError(
                "repository gate wrapper default must invoke scripts.capsule_rl.server_adapter"
            )
        from .server_adapter import validate_cli_request

        validate_cli_request(expanded_argv[3:])
    if not validate_only:
        print(f"{gate_name}: PASS ({args.artifact.resolve()})")
    return 0


__all__ = [
    "ADAPTER_RELOAD_PROMPT",
    "CANONICAL_EXECUTION_MODE",
    "CommandValidationError",
    "ConfigValidationError",
    "ExternalGatePlan",
    "FINAL_RUNTIME_AUDIT_GATE_ORDER",
    "GATE_ORDER",
    "GateArtifactError",
    "GateExecutionError",
    "add_validation_arguments",
    "artifact_file_sha256",
    "artifact_tree_file_count",
    "artifact_tree_sha256",
    "atomic_write_json",
    "atomic_write_text",
    "direct_lora_adapter_evidence",
    "external_gate_main",
    "exception_evidence",
    "gate_failure_artifact_path",
    "gate_log_artifact_path",
    "load_and_validate_server_config",
    "load_and_validate_server_config_bytes",
    "load_json_artifact",
    "repository_root",
    "runtime_dataset_path",
    "runtime_dependency_hashes",
    "run_external_gate",
    "validation_requested",
    "validate_final_runtime_audit",
    "verify_gate_envelope",
    "verify_adapter_reload_artifact",
    "verify_collector_gate_artifact",
    "verify_guided_gate_artifact",
    "verify_oracle_gate_artifact",
    "verify_preflight_gate_artifact",
    "verify_replay_telemetry",
    "verify_seed_gate_artifact",
    "verify_trainer_gate_artifact",
    "write_gate_failure_artifact",
]
