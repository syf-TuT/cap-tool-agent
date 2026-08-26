"""Supervise the immutable one-A800 Capsule-RL Gate 1--7 workflow."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from scripts.capsule_rl.llama_attestation import (
    ATTESTATION_ARTIFACT_TYPE,
    LlamaRuntimeAttestationError,
    attest_llama_cpp_runtime,
    llama_build_number,
    sanitize_dynamic_loader_environment,
)


OOM_LADDER = (
    "base_dynamic_fp32",
    "vllm_util_026",
    "fixed_microbatch_1",
    "fsdp_base_bf16",
    "fsdp_base_bf16_vllm_util_045",
)
GATE_ORDER = (
    "gate01_preflight",
    "gate02_seed",
    "gate03_oracle_replay",
    "gate04_collector",
    "gate05_guided",
    "gate06_trainer",
    "adapter_reload_smoke",
    "gate07_audit",
)
GPU_OOM_GATES = frozenset(GATE_ORDER[1:6])
_GPU_OOM_LOG_MARKERS = (
    b"out of memory",
    b"cuda oom",
    b"cuda error: out of memory",
    b"no available memory for the cache blocks",
)
VERL_V061_SHA = "d62da4950573d7a4b7ef2362337952e7ab59e78d"
CONTROLLER_GGUF_SHA256 = (
    "509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c"
)
LLAMA_ARCHIVE_SHA256 = (
    "f263a91280471b4c33c4999d7c76259c0f3a0a53a0b3e692b2c0b84380137a35"
)
MAX_CONTROLLER_SEED_RUN_IDS = 3
EXTERNAL_CONTROLLER_ENDPOINT = "https://coding.dashscope.aliyuncs.com/v1"
EXTERNAL_CONTROLLER_MODEL = "qwen3.7-plus"
EXTERNAL_CONTROLLER_ATTESTATION_TYPE = "external_controller_runtime_attestation"
_CAPSULE_CREDENTIAL_ENV_NAMES = frozenset(
    {"CAPX_PROGRAM_API_KEY", "CAPX_CONTROLLER_API_KEY"}
)


class OwnedServicesConfigError(ValueError):
    """The workflow, profile, or audited host violates the fixed contract."""


class FakeFailure(RuntimeError):
    """Compatibility exception retained for focused fake-runtime tests."""


class GateCommandError(RuntimeError):
    """One gate failed; only an identified OOM may advance the retry ladder."""

    def __init__(
        self,
        gate_name: str,
        returncode: int,
        *,
        oom: bool,
        guided_retry: bool = False,
    ) -> None:
        super().__init__(f"{gate_name} exited with status {returncode}")
        self.gate_name = gate_name
        self.returncode = returncode
        self.oom = oom
        self.guided_retry = guided_retry


def _gate_log_indicates_gpu_oom(log_tail: bytes) -> bool:
    """Recognize allocator and vLLM KV-cache capacity failures."""

    normalized = log_tail.lower()
    return any(marker in normalized for marker in _GPU_OOM_LOG_MARKERS)


@dataclass(frozen=True)
class AuditSnapshot:
    gpu_name: str
    gpu_count: int
    gpu_total_vram_mib: int
    gpu_free_vram_mib: int
    other_gpu_processes_mib: list[int]
    host_memory_mib: int
    mem_available_before_controller_mib: int
    mem_available_after_controller_mib: int
    mem_available_during_run_mib: int
    shm_available_mib: int
    disk_free_mib: int
    cuda_version: str
    nvidia_driver: str
    repo_head: str
    repo_is_dirty: bool
    system_version: str = ""


@dataclass(frozen=True)
class ProcessIdentity:
    name: str
    pid: int
    starttime_ticks: int


@dataclass(frozen=True)
class RenderedCommand:
    argv: list[str]
    env: dict[str, str]


@dataclass(frozen=True)
class RuntimeContext:
    workflow_path: Path
    profile_path: Path
    capsule_config_path: Path
    repo_root: Path
    output_root: Path
    artifact_root: Path
    checkpoint_root: Path
    retry_name: str


@dataclass(frozen=True)
class AttemptResult:
    retry_name: str
    run_id: str
    profile_sha256: str
    profile_path: Path
    capsule_config_path: Path
    output_dir: Path
    artifact_dir: Path


@dataclass(frozen=True)
class WorkflowResult:
    run_id: str
    output_dir: Path
    artifact_dir: Path
    capsule_config_path: Path
    rendered_services: dict[str, RenderedCommand]
    audit: AuditSnapshot
    controller_seed_run_ids: list[str]
    attempts: tuple[AttemptResult, ...] = ()


class _ContinuousMemoryMonitor:
    """Sample MemAvailable throughout the owned service and Gate lifetime."""

    def __init__(
        self,
        *,
        loader: Any,
        required_mib: int,
        evidence_path: Path,
        interval_s: float,
    ) -> None:
        self._loader = loader
        self._required_mib = required_mib
        self._evidence_path = evidence_path
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_ns = 0
        self._samples: list[dict[str, int]] = []
        self._error: BaseException | None = None

    def _capture_sample(self) -> None:
        if self._error is not None:
            return
        try:
            available = self._loader()
            if isinstance(available, bool) or not isinstance(available, int) or available < 0:
                raise RuntimeError("continuous MemAvailable probe returned an invalid value")
            elapsed_ms = (time.monotonic_ns() - self._started_ns) // 1_000_000
            self._samples.append(
                {"elapsed_ms": int(elapsed_ms), "available_mib": available}
            )
        except BaseException as error:
            self._error = error

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._capture_sample()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("continuous memory monitor was already started")
        self._started_ns = time.monotonic_ns()
        self._capture_sample()
        self._thread = threading.Thread(
            target=self._run,
            name="capsule-memory-monitor",
            daemon=True,
        )
        self._thread.start()

    def raise_if_breached(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"continuous MemAvailable probe failed: {self._error}")
        minimum = min(
            (sample["available_mib"] for sample in self._samples), default=None
        )
        if minimum is not None and minimum < self._required_mib:
            raise OwnedServicesConfigError(
                f"runtime minimum MemAvailable {minimum}MiB is below "
                f"{self._required_mib}MiB"
            )

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(5.0, self._interval_s * 2))
        if self._thread.is_alive() and self._error is None:
            self._error = RuntimeError("continuous memory monitor did not stop")
        self._capture_sample()
        minimum = min(
            (sample["available_mib"] for sample in self._samples), default=None
        )
        maximum_gap_ms = max(
            (
                later["elapsed_ms"] - earlier["elapsed_ms"]
                for earlier, later in zip(self._samples, self._samples[1:])
            ),
            default=0,
        )
        maximum_allowed_gap_ms = int(max(5.0, self._interval_s * 5) * 1000)
        passed = (
            self._error is None
            and minimum is not None
            and minimum >= self._required_mib
            and maximum_gap_ms <= maximum_allowed_gap_ms
        )
        _write_json_exclusive(
            self._evidence_path,
            {
                "schema_version": 1,
                "artifact_type": "single_a800_continuous_memory",
                "interval_s": self._interval_s,
                "required_mib": self._required_mib,
                "sample_count": len(self._samples),
                "minimum_available_mib": minimum,
                "maximum_sample_gap_ms": maximum_gap_ms,
                "maximum_allowed_gap_ms": maximum_allowed_gap_ms,
                "passed": passed,
                "probe_error": None if self._error is None else str(self._error),
                "samples": self._samples,
            },
        )
        if self._error is not None:
            raise RuntimeError(f"continuous MemAvailable probe failed: {self._error}")
        if maximum_gap_ms > maximum_allowed_gap_ms:
            raise RuntimeError(
                f"continuous MemAvailable sampling gap {maximum_gap_ms}ms exceeds "
                f"{maximum_allowed_gap_ms}ms"
            )
        if not passed:
            raise OwnedServicesConfigError(
                f"runtime minimum MemAvailable {minimum}MiB is below "
                f"{self._required_mib}MiB"
            )


def _llama_build_number(version_text: str) -> int | None:
    return llama_build_number(version_text)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise OwnedServicesConfigError(f"expected YAML file: {resolved}")
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise OwnedServicesConfigError(f"cannot load YAML {resolved}: {error}") from error
    if not isinstance(payload, dict):
        raise OwnedServicesConfigError(f"top-level YAML must be a mapping: {resolved}")
    return payload


def _mapping(payload: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    value = payload.get(field_name)
    if not isinstance(value, Mapping):
        raise OwnedServicesConfigError(f"{field_name} must be a mapping")
    return dict(value)


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OwnedServicesConfigError(f"{field_name} must be a non-empty string")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise OwnedServicesConfigError(f"{field_name} must be a positive integer")
    return value


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise OwnedServicesConfigError(f"{field_name} must be a boolean")
    return value


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise OwnedServicesConfigError(f"{field_name} must be a non-empty list")
    return [_string(item, f"{field_name}[{index}]") for index, item in enumerate(value)]


def _require_equal(actual: Any, expected: Any, field_name: str) -> None:
    if actual != expected:
        raise OwnedServicesConfigError(f"{field_name} must be {expected!r}")


def _external_controller_config(controller: Mapping[str, Any]) -> dict[str, Any]:
    expected_fields = {
        "mode",
        "endpoint",
        "model",
        "api_key_env",
        "request_timeout_s",
        "max_output_tokens",
        "stream",
        "enable_thinking",
        "temperature",
    }
    if set(controller) != expected_fields:
        raise OwnedServicesConfigError(
            "external Controller fields are incomplete or unexpected"
        )
    expected = {
        "mode": "external",
        "endpoint": EXTERNAL_CONTROLLER_ENDPOINT,
        "model": EXTERNAL_CONTROLLER_MODEL,
        "api_key_env": "CAPX_CONTROLLER_API_KEY",
        "request_timeout_s": 300.0,
        "max_output_tokens": 4096,
        "stream": False,
        "enable_thinking": False,
        "temperature": 0.7,
    }
    for field_name, expected_value in expected.items():
        _require_equal(
            controller.get(field_name),
            expected_value,
            f"controller.{field_name}",
        )
    return dict(controller)


def _external_controller_binding_sha256(controller: Mapping[str, Any]) -> str:
    config = _external_controller_config(controller)
    encoded = json.dumps(
        config, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _external_controller_attestation(
    controller: Mapping[str, Any], *, credential_present: bool
) -> dict[str, Any]:
    config = _external_controller_config(controller)
    if credential_present is not True:
        raise OwnedServicesConfigError(
            "external Controller credential environment is unset"
        )
    return {
        "schema_version": 1,
        "artifact_type": EXTERNAL_CONTROLLER_ATTESTATION_TYPE,
        "ownership": "external",
        **config,
        "credential_present": True,
        "controller_binding_sha256": _external_controller_binding_sha256(config),
    }


def _require_path(payload: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for field in dotted_path.split("."):
        if not isinstance(current, Mapping) or field not in current:
            raise OwnedServicesConfigError(f"{dotted_path} is required")
        current = current[field]
    return current


def _require_gate_credentials(
    env: Mapping[str, Any], *, expected: frozenset[str], field_name: str
) -> None:
    for secret_name in _CAPSULE_CREDENTIAL_ENV_NAMES:
        placeholder = f"{{env:{secret_name}}}"
        occurrences = [
            (str(name), str(value))
            for name, value in env.items()
            if name == secret_name or value == placeholder
        ]
        required = [(secret_name, placeholder)] if secret_name in expected else []
        if occurrences != required:
            raise OwnedServicesConfigError(
                f"{field_name} credential forwarding for {secret_name} must be {required!r}"
            )


def load_single_a800_resolved_profile(path: str | Path) -> dict[str, Any]:
    """Validate the v0.6.1 one-A800 LoRA base profile."""

    payload = _load_yaml(path)
    actor_rollout_ref = _mapping(payload, "actor_rollout_ref")
    model = _mapping(actor_rollout_ref, "model")
    actor = _mapping(actor_rollout_ref, "actor")
    rollout = _mapping(actor_rollout_ref, "rollout")
    ref = _mapping(actor_rollout_ref, "ref")
    fsdp = _mapping(actor, "fsdp_config")
    ref_fsdp = _mapping(ref, "fsdp_config")
    trainer = _mapping(payload, "trainer")
    data = _mapping(payload, "data")
    ray_kwargs = _mapping(payload, "ray_kwargs")
    capsule_runtime = _mapping(payload, "capsule_runtime")

    required_worker_surfaces = {
        "actor_rollout_ref.model._target_": "verl.workers.config.HFModelConfig",
        "actor_rollout_ref.model.exclude_modules": None,
        "actor_rollout_ref.model.lora_adapter_path": None,
        "actor_rollout_ref.actor._target_": "verl.workers.config.FSDPActorConfig",
        "actor_rollout_ref.actor.optim._target_": (
            "verl.workers.config.FSDPOptimizerConfig"
        ),
        "actor_rollout_ref.actor.fsdp_config._target_": (
            "verl.workers.config.FSDPEngineConfig"
        ),
        "actor_rollout_ref.actor.policy_loss._target_": (
            "capx.rl.capsule.verl_config.CapsulePolicyLossConfig"
        ),
        "actor_rollout_ref.actor.policy_loss.capsule_gamma": 0.1,
        "actor_rollout_ref.actor.ppo_micro_batch_size": None,
        "actor_rollout_ref.rollout._target_": "verl.workers.config.RolloutConfig",
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size": None,
        "actor_rollout_ref.ref.log_prob_micro_batch_size": None,
        "actor_rollout_ref.ref.fsdp_config._target_": (
            "verl.workers.config.FSDPEngineConfig"
        ),
        "trainer.device": "cuda",
        "data.trust_remote_code": False,
    }
    for dotted_path, expected in required_worker_surfaces.items():
        _require_equal(_require_path(payload, dotted_path), expected, dotted_path)
    _require_path(payload, "ray_kwargs.ray_init")
    _mapping(ray_kwargs, "ray_init")

    _string(model.get("path"), "model.path")
    for field, expected in (
        ("lora_rank", 16),
        ("lora_alpha", 32),
        ("target_modules", "all-linear"),
        ("enable_gradient_checkpointing", True),
        ("use_remove_padding", True),
        ("enable_activation_offload", False),
        ("use_shm", False),
    ):
        _require_equal(model.get(field), expected, f"model.{field}")
    _require_equal(actor.get("strategy"), "fsdp", "actor.strategy")
    _require_equal(actor.get("use_dynamic_bsz"), True, "actor.use_dynamic_bsz")
    _require_equal(actor.get("ppo_max_token_len_per_gpu"), 10240, "actor token cap")
    for field, expected in (
        ("fsdp_size", 1),
        ("param_offload", True),
        ("optimizer_offload", False),
        ("offload_policy", False),
        ("model_dtype", "fp32"),
        ("dtype", "bfloat16"),
    ):
        _require_equal(fsdp.get(field), expected, f"actor.fsdp_config.{field}")
    _require_equal(rollout.get("name"), "vllm", "rollout.name")
    _require_equal(rollout.get("dtype"), "bfloat16", "rollout.dtype")
    for field in (
        "tensor_model_parallel_size",
        "data_parallel_size",
        "pipeline_model_parallel_size",
    ):
        _require_equal(_positive_int(rollout.get(field), f"rollout.{field}"), 1, field)
    _require_equal(float(rollout.get("gpu_memory_utilization")), 0.30, "vLLM util")
    for field in (
        "free_cache_engine",
        "enforce_eager",
        "enable_chunked_prefill",
        "log_prob_use_dynamic_bsz",
    ):
        _require_equal(_bool(rollout.get(field), f"rollout.{field}"), True, field)
    for field in (
        "max_num_batched_tokens",
        "max_model_len",
        "log_prob_max_token_len_per_gpu",
    ):
        _require_equal(_positive_int(rollout.get(field), f"rollout.{field}"), 10240, field)
    _require_equal(rollout.get("max_num_seqs"), 1, "rollout.max_num_seqs")
    _require_equal(rollout.get("load_format"), "safetensors", "rollout.load_format")
    _require_equal(ref.get("log_prob_use_dynamic_bsz"), True, "ref dynamic batch")
    _require_equal(ref.get("log_prob_max_token_len_per_gpu"), 10240, "ref token cap")
    for field, expected in (
        ("fsdp_size", 1),
        ("param_offload", True),
        ("optimizer_offload", False),
        ("offload_policy", False),
        ("model_dtype", "fp32"),
        ("dtype", "bfloat16"),
    ):
        _require_equal(ref_fsdp.get(field), expected, f"ref.fsdp_config.{field}")
    _require_equal(trainer.get("n_gpus_per_node"), 1, "trainer.n_gpus_per_node")
    _require_equal(trainer.get("nnodes"), 1, "trainer.nnodes")
    _require_equal(trainer.get("total_epochs"), 1, "trainer.total_epochs")
    _require_equal(data.get("train_batch_size"), 1, "data.train_batch_size")
    _require_equal(
        capsule_runtime.get("reference_policy_mode"),
        "actor_base_adapter_disabled",
        "capsule_runtime.reference_policy_mode",
    )
    return payload


def load_owned_services_workflow(path: str | Path) -> dict[str, Any]:
    """Validate fixed service, hardware, provenance, and retry contracts."""

    payload = _load_yaml(path)
    runtime = _mapping(payload, "runtime")
    hardware = _mapping(payload, "hardware")
    paths = _mapping(payload, "paths")
    services = _mapping(payload, "services")
    _require_equal(runtime.get("repo_root"), "/root/autodl-tmp/cap-x", "runtime.repo_root")
    _require_equal(runtime.get("verl_pinned_sha"), VERL_V061_SHA, "runtime.verl_pinned_sha")
    _string(runtime.get("python_executable"), "runtime.python_executable")
    _require_equal(
        _positive_int(
            runtime.get("max_controller_seed_run_ids"),
            "runtime.max_controller_seed_run_ids",
        ),
        MAX_CONTROLLER_SEED_RUN_IDS,
        "runtime.max_controller_seed_run_ids",
    )
    required_hardware = {
        "gpu_count": 1,
        "gpu_total_vram_required_mib": 81920,
        "gpu_free_vram_required_mib": 77824,
        "max_other_process_vram_mib": 512,
        "host_memory_required_mib": 122880,
        "mem_available_after_controller_required_mib": 92160,
        "mem_available_during_run_required_mib": 12288,
        "shm_required_mib": 12288,
        "disk_free_required_mib": 81920,
    }
    _require_equal(hardware.get("gpu_model"), "A800", "hardware.gpu_model")
    for field, expected in required_hardware.items():
        _require_equal(_positive_int(hardware.get(field), f"hardware.{field}"), expected, field)
    _require_equal(paths.get("forbid_overwrite"), True, "paths.forbid_overwrite")
    _require_equal(list(payload.get("oom_ladder", ())), list(OOM_LADDER), "oom_ladder")
    _require_equal(list(payload.get("gates", ())), list(GATE_ORDER), "gates")
    _require_equal(
        set(_mapping(payload, "gate_commands")),
        {*GATE_ORDER, "gate07_finalize"},
        "gate_commands",
    )
    _require_equal(set(services), {"controller", "program", "pyroki"}, "services")

    controller = _mapping(services, "controller")
    controller_mode = controller.get("mode", "local")
    if controller_mode == "external":
        _external_controller_config(controller)
    elif controller_mode == "local":
        _require_equal(controller.get("version_tag"), "b10516", "controller.version_tag")
        _require_equal(
            controller.get("model_sha256"), CONTROLLER_GGUF_SHA256, "controller GGUF SHA"
        )
        _require_equal(
            controller.get("archive_sha256"), LLAMA_ARCHIVE_SHA256, "llama archive SHA"
        )
        controller_argv = _string_list(
            controller.get("argv_template"), "controller.argv_template"
        )
        for required in ("--model", "--alias", "--n-gpu-layers", "--parallel", "--ctx-size"):
            if required not in controller_argv:
                raise OwnedServicesConfigError(f"controller command is missing {required}")
        _require_equal(
            _mapping(controller, "env"),
            {
                "CUDA_VISIBLE_DEVICES": "",
                "LLAMA_API_KEY": "{env:CAPX_CONTROLLER_API_KEY}",
            },
            "Controller environment",
        )
    else:
        raise OwnedServicesConfigError("controller.mode must be local or external")
    program = _mapping(services, "program")
    program_argv = _string_list(
        program.get("argv_template"), "program.argv_template"
    )
    _require_equal(program_argv[1:3], ["-m", "capx.rl.capsule.actor_identity"], "program module")
    _require_equal(
        _mapping(program, "env"),
        {
            "CUDA_VISIBLE_DEVICES": "",
            "CAPX_PROGRAM_API_KEY": "{env:CAPX_PROGRAM_API_KEY}",
        },
        "Program identity environment",
    )
    _require_equal(
        _mapping(_mapping(services, "pyroki"), "env"),
        {"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"},
        "PyRoKi CPU environment",
    )
    gate_credentials = {
        "gate01_preflight": frozenset(_CAPSULE_CREDENTIAL_ENV_NAMES),
        "gate04_collector": frozenset({"CAPX_CONTROLLER_API_KEY"}),
        "gate05_guided": frozenset({"CAPX_CONTROLLER_API_KEY"}),
    }
    gate_commands = _mapping(payload, "gate_commands")
    for name in gate_commands:
        _require_gate_credentials(
            _mapping(_mapping(gate_commands, str(name)), "env"),
            expected=gate_credentials.get(str(name), frozenset()),
            field_name=f"gate_commands.{name}.env",
        )
    return payload


def build_controller_seed_run_ids(
    run_id: str, max_run_ids: int = MAX_CONTROLLER_SEED_RUN_IDS
) -> list[str]:
    if max_run_ids != MAX_CONTROLLER_SEED_RUN_IDS:
        raise OwnedServicesConfigError(
            f"max_controller_seed_run_ids must be {MAX_CONTROLLER_SEED_RUN_IDS}"
        )
    return [
        f"{run_id}-controller-seed-{index}"
        for index in range(1, max_run_ids + 1)
    ]


def _canonical_yaml_bytes(payload: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True).encode()


def _file_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_controller_attestation(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("artifact_type") == EXTERNAL_CONTROLLER_ATTESTATION_TYPE:
        config_fields = {
            "mode",
            "endpoint",
            "model",
            "api_key_env",
            "request_timeout_s",
            "max_output_tokens",
            "stream",
            "enable_thinking",
            "temperature",
        }
        required_fields = {
            "schema_version",
            "artifact_type",
            "ownership",
            *config_fields,
            "credential_present",
            "controller_binding_sha256",
        }
        if set(payload) != required_fields:
            raise OwnedServicesConfigError(
                "external Controller attestation fields do not match schema version 1"
            )
        config = {field_name: payload[field_name] for field_name in config_fields}
        expected = _external_controller_attestation(
            config, credential_present=payload.get("credential_present") is True
        )
        if dict(payload) != expected or payload.get("schema_version") != 1:
            raise OwnedServicesConfigError(
                "external Controller attestation does not match the fixed runtime contract"
            )
        return expected

    required_fields = {
        "schema_version",
        "artifact_type",
        "version_tag",
        "archive_path",
        "archive_sha256",
        "binary_path",
        "binary_archive_member",
        "binary_sha256",
        "gguf_path",
        "gguf_sha256",
        "build_number",
        "runtime_tree_sha256",
        "regular_file_count",
        "symlink_count",
    }
    if set(payload) != required_fields:
        raise OwnedServicesConfigError(
            "controller runtime attestation fields do not match schema version 1"
        )
    result = dict(payload)
    _require_equal(result["schema_version"], 1, "controller attestation schema_version")
    _require_equal(
        result["artifact_type"],
        ATTESTATION_ARTIFACT_TYPE,
        "controller attestation artifact_type",
    )
    _require_equal(result["version_tag"], "b10516", "controller attestation version_tag")
    _require_equal(
        result["archive_sha256"],
        LLAMA_ARCHIVE_SHA256,
        "controller attestation archive_sha256",
    )
    _require_equal(
        result["gguf_sha256"],
        CONTROLLER_GGUF_SHA256,
        "controller attestation gguf_sha256",
    )
    _require_equal(result["build_number"], 10516, "controller attestation build_number")
    for field in ("archive_path", "binary_path", "binary_archive_member", "gguf_path"):
        _string(result[field], f"controller attestation {field}")
    for field in ("binary_sha256", "runtime_tree_sha256"):
        value = _string(result[field], f"controller attestation {field}")
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise OwnedServicesConfigError(
                f"controller attestation {field} must be a lowercase SHA-256"
            )
    _positive_int(
        result["regular_file_count"], "controller attestation regular_file_count"
    )
    symlink_count = result["symlink_count"]
    if (
        isinstance(symlink_count, bool)
        or not isinstance(symlink_count, int)
        or symlink_count < 0
    ):
        raise OwnedServicesConfigError(
            "controller attestation symlink_count must be a non-negative integer"
        )
    return result


def _retry_profile(base_profile: Mapping[str, Any], retry_name: str) -> dict[str, Any]:
    profile = copy.deepcopy(dict(base_profile))
    root = profile["actor_rollout_ref"]
    actor, rollout, ref = root["actor"], root["rollout"], root["ref"]
    if retry_name not in OOM_LADDER:
        raise OwnedServicesConfigError(f"unknown retry profile: {retry_name}")
    if retry_name in OOM_LADDER[1:]:
        rollout["gpu_memory_utilization"] = 0.26
    if retry_name in OOM_LADDER[2:]:
        actor["use_dynamic_bsz"] = False
        actor["ppo_micro_batch_size_per_gpu"] = 1
        rollout["log_prob_use_dynamic_bsz"] = False
        rollout["log_prob_micro_batch_size_per_gpu"] = 1
        ref["log_prob_use_dynamic_bsz"] = False
        ref["log_prob_micro_batch_size_per_gpu"] = 1
    if retry_name in OOM_LADDER[3:]:
        actor["fsdp_config"]["model_dtype"] = "bf16"
        ref["fsdp_config"]["model_dtype"] = "bf16"
    if retry_name == OOM_LADDER[4]:
        rollout["gpu_memory_utilization"] = 0.45
    profile["capsule_runtime"]["oom_profile"] = retry_name
    return profile


def materialize_retry_profile(
    *, base_profile: Mapping[str, Any], retry_name: str, destination: Path
) -> Path:
    profile = _retry_profile(base_profile, retry_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(profile, stream, sort_keys=False, allow_unicode=True)
    return destination


def _attempt_run_id(
    capsule_config_bytes: bytes, profile: Mapping[str, Any], retry_name: str
) -> tuple[str, str]:
    profile_sha = hashlib.sha256(_canonical_yaml_bytes(profile)).hexdigest()
    digest = hashlib.sha256(
        capsule_config_bytes
        + b"\0"
        + retry_name.encode("ascii")
        + b"\0"
        + profile_sha.encode("ascii")
    ).hexdigest()
    return f"{retry_name}-{digest[:12]}", profile_sha


def _render(tokens: Sequence[str], replacements: Mapping[str, str]) -> list[str]:
    result: list[str] = []
    for token in tokens:
        current = token
        for key, value in replacements.items():
            current = current.replace("{" + key + "}", value)
        if "{" in current or "}" in current:
            raise OwnedServicesConfigError(f"unresolved command placeholder: {current!r}")
        result.append(current)
    return result


def _render_services(
    workflow: Mapping[str, Any], *, capsule_config_path: Path
) -> dict[str, RenderedCommand]:
    runtime = _mapping(workflow, "runtime")
    services = _mapping(workflow, "services")
    controller = _mapping(services, "controller")
    replacements = {
        "python": _string(runtime.get("python_executable"), "runtime.python_executable"),
        "capsule_config": str(capsule_config_path),
    }
    service_names = ["program", "pyroki"]
    if controller.get("mode", "local") == "local":
        replacements.update(
            {
                "controller_binary": _string(
                    controller.get("binary_path"), "controller.binary_path"
                ),
                "controller_gguf": _string(
                    controller.get("gguf_path"), "controller.gguf_path"
                ),
                "controller_alias": _string(
                    controller.get("model_alias"), "controller.model_alias"
                ),
            }
        )
        service_names.insert(0, "controller")
    result: dict[str, RenderedCommand] = {}
    for name in service_names:
        service = _mapping(services, name)
        result[name] = RenderedCommand(
            argv=_render(
                _string_list(service.get("argv_template"), f"services.{name}.argv_template"),
                replacements,
            ),
            env={str(key): str(value) for key, value in _mapping(service, "env").items()},
        )
    return result


def _gate_artifacts(artifact_dir: Path) -> dict[str, Path]:
    return {
        "gate01_preflight": artifact_dir / "gate01_preflight.json",
        "gate02_seed": artifact_dir / "gate02_seed.json",
        "gate03_oracle_replay": artifact_dir / "gate03_oracle.json",
        "gate04_collector": artifact_dir / "gate04_collector.json",
        "gate05_guided": artifact_dir / "gate05_guided_group.json",
        "gate06_trainer": artifact_dir / "gate06_trainer.json",
        "adapter_reload_smoke": artifact_dir / "adapter_reload_smoke.json",
        "gate07_audit_candidate_json": artifact_dir / "gate07_audit.candidate.json",
        "gate07_audit_candidate_report": artifact_dir / "gate07_audit.candidate.md",
        "gate07_audit_json": artifact_dir / "gate07_audit.json",
        "gate07_audit_report": artifact_dir / "gate07_audit.md",
        "launcher_continuous_memory": artifact_dir / "launcher_continuous_memory.json",
        "launcher_controller_attestation": (
            artifact_dir / "launcher_controller_attestation.json"
        ),
        "launcher_owned_cleanup": artifact_dir / "launcher_owned_cleanup.json",
    }


def _render_gate_commands(
    workflow: Mapping[str, Any],
    *,
    capsule_config_path: Path,
    run_id: str,
    artifact_dir: Path,
) -> dict[str, RenderedCommand]:
    runtime = _mapping(workflow, "runtime")
    templates = _mapping(workflow, "gate_commands")
    replacements = {
        "python": _string(runtime.get("python_executable"), "runtime.python_executable"),
        "capsule_config": str(capsule_config_path),
        "run_id": run_id,
        "input_dir": str(artifact_dir),
        **{name: str(path) for name, path in _gate_artifacts(artifact_dir).items()},
    }
    result: dict[str, RenderedCommand] = {}
    for name in (*GATE_ORDER, "gate07_finalize"):
        gate = _mapping(templates, name)
        result[name] = RenderedCommand(
            argv=_render(
                _string_list(gate.get("argv_template"), f"gate_commands.{name}.argv_template"),
                replacements,
            ),
            env={str(key): str(value) for key, value in _mapping(gate, "env").items()},
        )
    return result


def _validate_audit(snapshot: AuditSnapshot, workflow: Mapping[str, Any]) -> None:
    hardware = _mapping(workflow, "hardware")
    normalized = snapshot.gpu_name.upper().replace("-", "").replace(" ", "")
    if "A800" not in normalized or "80GB" not in normalized:
        raise OwnedServicesConfigError("hardware audit failed: GPU must be an A800 80GB")
    if snapshot.gpu_count != hardware["gpu_count"]:
        raise OwnedServicesConfigError("hardware audit failed: exactly one A800 is required")
    if snapshot.gpu_total_vram_mib < hardware["gpu_total_vram_required_mib"]:
        raise OwnedServicesConfigError("hardware audit failed: A800 does not report 80GB VRAM")
    if snapshot.gpu_free_vram_mib < hardware["gpu_free_vram_required_mib"]:
        raise OwnedServicesConfigError("hardware audit failed: free VRAM is below 76GiB")
    if any(
        used > hardware["max_other_process_vram_mib"]
        for used in snapshot.other_gpu_processes_mib
    ):
        raise OwnedServicesConfigError("hardware audit failed: another GPU process exceeds 512MiB")
    if snapshot.host_memory_mib < hardware["host_memory_required_mib"]:
        raise OwnedServicesConfigError("hardware audit failed: host RAM is below 120GiB")
    if snapshot.shm_available_mib < hardware["shm_required_mib"]:
        raise OwnedServicesConfigError("hardware audit failed: /dev/shm is below 12GiB")
    if snapshot.disk_free_mib < hardware["disk_free_required_mib"]:
        raise OwnedServicesConfigError("hardware audit failed: experiment disk is below 80GiB free")


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(payload), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _check_memory(
    runtime: Any,
    workflow: Mapping[str, Any],
    *,
    after_controller: bool,
    stage: str,
    evidence_path: Path,
) -> None:
    hardware = _mapping(workflow, "hardware")
    field = (
        "mem_available_after_controller_required_mib"
        if after_controller
        else "mem_available_during_run_required_mib"
    )
    available = runtime.mem_available_mib()
    required = hardware[field]
    _write_json_exclusive(
        evidence_path,
        {
            "schema_version": 1,
            "artifact_type": "single_a800_memory_check",
            "stage": stage,
            "available_mib": available,
            "required_mib": required,
            "passed": available >= required,
        },
    )
    if available < required:
        stage = "post-controller" if after_controller else "runtime"
        raise OwnedServicesConfigError(
            f"{stage} MemAvailable {available}MiB is below {required}MiB"
        )


def _materialize_capsule_config(
    *,
    source_path: Path,
    destination: Path,
    workflow: Mapping[str, Any],
    profile: Mapping[str, Any],
    resolved_profile_path: Path,
    run_id: str,
    output_dir: Path,
) -> None:
    config = _load_yaml(source_path)
    runtime = _mapping(config, "runtime")
    workflow_runtime = _mapping(workflow, "runtime")
    controller = _mapping(_mapping(workflow, "services"), "controller")
    model = _mapping(_mapping(profile, "actor_rollout_ref"), "model")
    runtime.update(
        {
            "project_root": workflow_runtime["repo_root"],
            "run_id": run_id,
            "verl_source_path": workflow_runtime["verl_source_path"],
            "verl_pinned_sha": VERL_V061_SHA,
            "verl_resolved_config_path": str(resolved_profile_path),
            "program_model_path": model["path"],
            "output_dir": str(output_dir),
            "python_executable": workflow_runtime["python_executable"],
        }
    )
    config["runtime"] = runtime
    program_service = _mapping(config, "program_service")
    program_service.update({"mode": "actor_identity", "model": model["path"]})
    config["program_service"] = program_service
    controller_service = _mapping(config, "controller_service")
    if controller.get("mode", "local") == "external":
        controller_service.update(
            {
                field_name: controller[field_name]
                for field_name in (
                    "endpoint",
                    "model",
                    "api_key_env",
                    "request_timeout_s",
                    "max_output_tokens",
                    "stream",
                    "enable_thinking",
                    "temperature",
                )
            }
        )
    else:
        controller_service.update(
            {
                "model": controller["model_alias"],
                "request_timeout_s": 300.0,
                "max_output_tokens": 4096,
                "stream": False,
                "enable_thinking": False,
            }
        )
    config["controller_service"] = controller_service
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)


def _write_failure(path: Path, error: BaseException, attempt: AttemptResult) -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "single_a800_launcher_failure",
        "run_id": attempt.run_id,
        "retry_name": attempt.retry_name,
        "error_type": type(error).__name__,
        "error": str(error),
        "oom": bool(isinstance(error, GateCommandError) and error.oom),
    }
    _write_json_exclusive(path, payload)


def _write_initial_audit(
    path: Path, snapshot: AuditSnapshot, attempt: AttemptResult
) -> None:
    _write_json_exclusive(
        path,
        {
            "schema_version": 1,
            "artifact_type": "single_a800_initial_audit",
            "run_id": attempt.run_id,
            "retry_name": attempt.retry_name,
            "profile_sha256": attempt.profile_sha256,
            "snapshot": asdict(snapshot),
        },
    )


def _write_owned_process(
    path: Path,
    identity: ProcessIdentity,
    command: RenderedCommand,
) -> None:
    argv_bytes = json.dumps(
        command.argv, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    credential_markers = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    credential_keys = sorted(
        name
        for name, value in command.env.items()
        if any(marker in name.upper() for marker in credential_markers)
        or value.startswith("{env:")
    )
    _write_json_exclusive(
        path,
        {
            "schema_version": 1,
            "artifact_type": "single_a800_owned_process",
            "name": identity.name,
            "pid": identity.pid,
            "starttime_ticks": identity.starttime_ticks,
            "argv_sha256": hashlib.sha256(argv_bytes).hexdigest(),
            "env_keys": sorted(command.env),
            "credential_env_keys": credential_keys,
        },
    )


def _write_owned_cleanup(
    path: Path,
    *,
    run_id: str,
    identities: Sequence[ProcessIdentity],
    controller_mode: str,
) -> None:
    expected_names = (
        ("controller", "program", "pyroki")
        if controller_mode == "local"
        else ("program", "pyroki")
    )
    if tuple(identity.name for identity in identities) != expected_names:
        raise RuntimeError(
            "owned-service cleanup evidence does not match Controller ownership"
        )
    process_identities = {
        (identity.pid, identity.starttime_ticks) for identity in identities
    }
    if len(process_identities) != len(expected_names):
        raise RuntimeError("owned-service cleanup process identities must be unique")
    _write_json_exclusive(
        path,
        {
            "schema_version": 1,
            "artifact_type": "single_a800_owned_service_cleanup",
            "run_id": run_id,
            "cleanup_completed": True,
            "services": (
                []
                if controller_mode == "local"
                else [
                    {
                        "name": "controller",
                        "ownership": "external",
                        "termination_confirmed": None,
                    }
                ]
            )
            + [
                {
                    "name": identity.name,
                    "ownership": "owned",
                    "pid": identity.pid,
                    "starttime_ticks": identity.starttime_ticks,
                    "termination_confirmed": True,
                }
                for identity in identities
            ],
        },
    )


def _attempt_paths(context: RuntimeContext, run_id: str) -> tuple[Path, Path, Path]:
    return (
        context.output_root / run_id,
        context.artifact_root / run_id,
        context.checkpoint_root / run_id,
    )


def _hypothetical_attempt(
    context: RuntimeContext,
    capsule_bytes: bytes,
    base_profile: Mapping[str, Any],
    retry_name: str,
) -> AttemptResult:
    profile = _retry_profile(base_profile, retry_name)
    run_id, profile_sha = _attempt_run_id(capsule_bytes, profile, retry_name)
    output_dir, artifact_dir, _ = _attempt_paths(context, run_id)
    return AttemptResult(
        retry_name=retry_name,
        run_id=run_id,
        profile_sha256=profile_sha,
        profile_path=artifact_dir / "resolved" / "verl.yaml",
        capsule_config_path=artifact_dir / "resolved" / "capsule.yaml",
        output_dir=output_dir,
        artifact_dir=artifact_dir,
    )


def _controller_seed_attempt(base: AttemptResult, seed_index: int) -> AttemptResult:
    run_id = build_controller_seed_run_ids(base.run_id)[seed_index - 1]
    return AttemptResult(
        retry_name=base.retry_name,
        run_id=run_id,
        profile_sha256=base.profile_sha256,
        profile_path=base.artifact_dir.parent / run_id / "resolved" / "verl.yaml",
        capsule_config_path=base.artifact_dir.parent / run_id / "resolved" / "capsule.yaml",
        output_dir=base.output_dir.parent / run_id,
        artifact_dir=base.artifact_dir.parent / run_id,
    )


def _run_attempt(
    *,
    workflow: Mapping[str, Any],
    context: RuntimeContext,
    attempt: AttemptResult,
    profile: Mapping[str, Any],
    audit: AuditSnapshot,
    controller_attestation: Mapping[str, Any],
    runtime: Any,
) -> dict[str, RenderedCommand]:
    output_dir, artifact_dir, checkpoint_dir = _attempt_paths(context, attempt.run_id)
    if any(path.exists() for path in (output_dir, artifact_dir, checkpoint_dir)):
        raise FileExistsError(f"existing run path blocks launch: {attempt.run_id}")
    output_dir.mkdir(parents=True, exist_ok=False)
    artifact_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    resolved_dir = artifact_dir / "resolved"
    resolved_dir.mkdir(exist_ok=False)
    _write_initial_audit(artifact_dir / "launcher_initial_audit.json", audit, attempt)
    with attempt.profile_path.open("x", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(profile, stream, sort_keys=False, allow_unicode=True)
    _materialize_capsule_config(
        source_path=context.capsule_config_path,
        destination=attempt.capsule_config_path,
        workflow=workflow,
        profile=profile,
        resolved_profile_path=attempt.profile_path,
        run_id=attempt.run_id,
        output_dir=attempt.output_dir,
    )
    services = _render_services(workflow, capsule_config_path=attempt.capsule_config_path)
    gates = _render_gate_commands(
        workflow,
        capsule_config_path=attempt.capsule_config_path,
        run_id=attempt.run_id,
        artifact_dir=attempt.artifact_dir,
    )
    if hasattr(runtime, "configure_attempt"):
        runtime.configure_attempt(attempt, workflow)
    if hasattr(runtime, "verify_controller_runtime"):
        controller_attestation = runtime.verify_controller_runtime(context, workflow)
    _write_json_exclusive(
        artifact_dir / "launcher_controller_attestation.json",
        _validated_controller_attestation(controller_attestation),
    )
    owned: list[ProcessIdentity] = []
    memory_monitor: _ContinuousMemoryMonitor | None = None
    primary_error: BaseException | None = None

    def start_memory_monitor() -> _ContinuousMemoryMonitor:
        _check_memory(
            runtime,
            workflow,
            after_controller=True,
            stage="post-controller",
            evidence_path=artifact_dir / "launcher_memory_00_post-controller.json",
        )
        continuous_loader = getattr(
            runtime,
            "continuous_mem_available_mib",
            runtime.mem_available_mib,
        )
        monitor = _ContinuousMemoryMonitor(
            loader=continuous_loader,
            required_mib=_mapping(workflow, "hardware")[
                "mem_available_during_run_required_mib"
            ],
            evidence_path=artifact_dir / "launcher_continuous_memory.json",
            interval_s=float(getattr(runtime, "memory_monitor_interval_s", 1.0)),
        )
        monitor.start()
        return monitor

    controller_mode = _mapping(_mapping(workflow, "services"), "controller").get(
        "mode", "local"
    )
    try:
        if controller_mode == "external":
            memory_monitor = start_memory_monitor()
            memory_monitor.raise_if_breached()
        for service_name, command in services.items():
            if memory_monitor is not None:
                memory_monitor.raise_if_breached()
            identity = runtime.spawn(service_name, command.argv, command.env)
            owned.append(identity)
            _write_owned_process(
                artifact_dir / f"launcher_owned_process_{service_name}.json",
                identity,
                command,
            )
            runtime.wait_ready(identity)
            if memory_monitor is not None:
                memory_monitor.raise_if_breached()
            if service_name == "controller":
                memory_monitor = start_memory_monitor()
                memory_monitor.raise_if_breached()
        for gate_index, gate_name in enumerate(GATE_ORDER, start=1):
            if memory_monitor is not None:
                memory_monitor.raise_if_breached()
            command = gates[gate_name]
            runtime.run_gate(gate_name, command.argv, command.env)
            if memory_monitor is not None:
                memory_monitor.raise_if_breached()
            _check_memory(
                runtime,
                workflow,
                after_controller=False,
                stage=f"after-{gate_name}",
                evidence_path=(
                    artifact_dir
                    / f"launcher_memory_{gate_index:02d}_after-{gate_name}.json"
                ),
            )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        monitor_errors: list[BaseException] = []
        if memory_monitor is not None:
            try:
                memory_monitor.stop()
            except BaseException as monitor_error:
                monitor_errors.append(monitor_error)
        cleanup_errors: list[BaseException] = []
        for identity in reversed(owned):
            try:
                runtime.terminate(identity)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        for identity in owned:
            try:
                if not runtime.confirm_terminated(identity):
                    raise RuntimeError(
                        f"{identity.name} termination was not confirmed for "
                        f"pid={identity.pid}, starttime_ticks={identity.starttime_ticks}"
                    )
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        finalization_errors = monitor_errors + cleanup_errors
        if finalization_errors:
            if primary_error is not None:
                if monitor_errors:
                    setattr(primary_error, "continuous_memory_errors", monitor_errors)
                if cleanup_errors:
                    setattr(primary_error, "owned_cleanup_errors", cleanup_errors)
            else:
                raise finalization_errors[0]
    _write_owned_cleanup(
        artifact_dir / "launcher_owned_cleanup.json",
        run_id=attempt.run_id,
        identities=owned,
        controller_mode=str(controller_mode),
    )
    finalizer = gates["gate07_finalize"]
    runtime.run_gate("gate07_finalize", finalizer.argv, finalizer.env)
    return services


def execute_owned_service_workflow(
    *,
    workflow_path: str | Path,
    profile_path: str | Path,
    capsule_config_path: Path,
    runtime: Any,
    dry_run: bool,
) -> WorkflowResult:
    workflow = load_owned_services_workflow(workflow_path)
    base_profile = load_single_a800_resolved_profile(profile_path)
    capsule_path = Path(capsule_config_path).expanduser().resolve()
    if capsule_path.is_file():
        capsule_bytes = capsule_path.read_bytes()
    elif dry_run and hasattr(runtime, "tmp_path"):
        capsule_bytes = str(capsule_path).encode()
    else:
        raise OwnedServicesConfigError(f"Capsule config does not exist: {capsule_path}")
    paths = _mapping(workflow, "paths")
    roots = (
        Path(paths["output_root"]),
        Path(paths["artifact_root"]),
        Path(paths["checkpoint_root"]),
    )
    if hasattr(runtime, "tmp_path"):
        root = Path(runtime.tmp_path)
        roots = (root / "outputs", root / "artifacts", root / "checkpoints")
    context = RuntimeContext(
        workflow_path=Path(workflow_path).expanduser().resolve(),
        profile_path=Path(profile_path).expanduser().resolve(),
        capsule_config_path=capsule_path,
        repo_root=Path(_mapping(workflow, "runtime")["repo_root"]),
        output_root=roots[0],
        artifact_root=roots[1],
        checkpoint_root=roots[2],
        retry_name=OOM_LADDER[0],
    )
    audit = runtime.collect_audit_snapshot(context)
    _validate_audit(audit, workflow)
    controller_attestation = _validated_controller_attestation(
        runtime.verify_runtime_inputs(context, workflow)
    )
    first_base = _hypothetical_attempt(context, capsule_bytes, base_profile, OOM_LADDER[0])
    first = _controller_seed_attempt(first_base, 1)
    first_seed_run_ids = build_controller_seed_run_ids(first_base.run_id)
    first_services = _render_services(workflow, capsule_config_path=first.capsule_config_path)
    if dry_run:
        return WorkflowResult(
            run_id=first.run_id,
            output_dir=first.output_dir,
            artifact_dir=first.artifact_dir,
            capsule_config_path=first.capsule_config_path,
            rendered_services=first_services,
            audit=audit,
            controller_seed_run_ids=first_seed_run_ids,
            attempts=(first,),
        )
    attempted: list[AttemptResult] = []
    max_controller_seed_run_ids = _mapping(workflow, "runtime")[
        "max_controller_seed_run_ids"
    ]
    for index, retry_name in enumerate(OOM_LADDER):
        profile = _retry_profile(base_profile, retry_name)
        base_attempt = _hypothetical_attempt(
            context, capsule_bytes, base_profile, retry_name
        )
        seed_run_ids = build_controller_seed_run_ids(
            base_attempt.run_id, max_controller_seed_run_ids
        )
        advance_oom_ladder = False
        for seed_index in range(1, max_controller_seed_run_ids + 1):
            attempt = _controller_seed_attempt(base_attempt, seed_index)
            attempted.append(attempt)
            try:
                services = _run_attempt(
                    workflow=workflow,
                    context=context,
                    attempt=attempt,
                    profile=profile,
                    audit=audit,
                    controller_attestation=controller_attestation,
                    runtime=runtime,
                )
            except BaseException as error:
                if attempt.artifact_dir.is_dir():
                    failure = attempt.artifact_dir / "launcher_failure.json"
                    if not failure.exists():
                        _write_failure(failure, error, attempt)
                if isinstance(error, GateCommandError) and error.guided_retry:
                    if seed_index < max_controller_seed_run_ids:
                        continue
                    raise
                if (
                    isinstance(error, GateCommandError)
                    and error.oom
                    and error.gate_name in GPU_OOM_GATES
                ):
                    if index < len(OOM_LADDER) - 1:
                        advance_oom_ladder = True
                        break
                raise
            return WorkflowResult(
                run_id=attempt.run_id,
                output_dir=attempt.output_dir,
                artifact_dir=attempt.artifact_dir,
                capsule_config_path=attempt.capsule_config_path,
                rendered_services=services,
                audit=audit,
                controller_seed_run_ids=seed_run_ids,
                attempts=tuple(attempted),
            )
        if advance_oom_ladder:
            continue
    raise AssertionError("OOM ladder exhausted without a terminal error")


class LinuxRuntime:
    """Concrete Linux audit, subprocess, readiness, and owned-process runtime."""

    def __init__(self) -> None:
        self.memory_monitor_interval_s = 1.0
        self._attempt: AttemptResult | None = None
        self._workflow: Mapping[str, Any] | None = None
        self._controller_attestation: dict[str, Any] | None = None
        self._processes: dict[tuple[int, int], subprocess.Popen[bytes]] = {}

    @staticmethod
    def _run_text(argv: Sequence[str], *, timeout: float = 20) -> str:
        environment = (
            {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
            if argv and argv[0] == "git"
            else None
        )
        try:
            result = subprocess.run(
                list(argv),
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=environment,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise OwnedServicesConfigError(f"host probe failed: {argv[0]}: {error}") from error
        return result.stdout.strip()

    @staticmethod
    def _meminfo() -> dict[str, int]:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            name, raw = line.split(":", 1)
            values[name] = int(raw.strip().split()[0]) // 1024
        return values

    def mem_available_mib(self) -> int:
        return self._meminfo()["MemAvailable"]

    def continuous_mem_available_mib(self) -> int:
        return self._meminfo()["MemAvailable"]

    @staticmethod
    def _free_mib(path: Path) -> int:
        stats = os.statvfs(path)
        return int(stats.f_bavail * stats.f_frsize // (1024 * 1024))

    def collect_audit_snapshot(self, context: RuntimeContext) -> AuditSnapshot:
        rows = self._run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines()
        gpu_name, total, free, driver = "", 0, 0, ""
        if len(rows) == 1:
            parts = [part.strip() for part in rows[0].split(",")]
            if len(parts) == 4:
                gpu_name, total_raw, free_raw, driver = parts
                total, free = int(total_raw), int(free_raw)
        else:
            gpu_name = ", ".join(row.split(",", 1)[0].strip() for row in rows)
        process_text = self._run_text(
            [
                "nvidia-smi",
                "--query-compute-apps=used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        process_memory = [int(line) for line in process_text.splitlines() if line.strip().isdigit()]
        meminfo = self._meminfo()
        repo_head = self._run_text(["git", "-C", str(context.repo_root), "rev-parse", "HEAD"])
        dirty = bool(
            self._run_text(
                [
                    "git",
                    "-C",
                    str(context.repo_root),
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                ]
            )
        )
        available = meminfo["MemAvailable"]
        smi_banner = self._run_text(["nvidia-smi"])
        cuda_version = "unknown"
        if "CUDA Version:" in smi_banner:
            cuda_version = smi_banner.split("CUDA Version:", 1)[1].split()[0]
        return AuditSnapshot(
            gpu_name=gpu_name,
            gpu_count=len(rows),
            gpu_total_vram_mib=total,
            gpu_free_vram_mib=free,
            other_gpu_processes_mib=process_memory,
            host_memory_mib=meminfo["MemTotal"],
            mem_available_before_controller_mib=available,
            mem_available_after_controller_mib=available,
            mem_available_during_run_mib=available,
            shm_available_mib=self._free_mib(Path("/dev/shm")),
            disk_free_mib=self._free_mib(context.repo_root),
            cuda_version=cuda_version,
            nvidia_driver=driver,
            repo_head=repo_head,
            repo_is_dirty=dirty,
            system_version=platform.platform(),
        )

    def verify_runtime_inputs(
        self, context: RuntimeContext, workflow: Mapping[str, Any]
    ) -> dict[str, Any]:
        controller_attestation = self.verify_controller_runtime(context, workflow)
        model = _mapping(
            _mapping(_load_yaml(context.profile_path), "actor_rollout_ref"), "model"
        )
        model_path = Path(model["path"])
        if not model_path.is_dir() or model_path.is_symlink():
            raise OwnedServicesConfigError(f"Qwen model must be materialized: {model_path}")
        return controller_attestation

    def verify_controller_runtime(
        self, _context: RuntimeContext, workflow: Mapping[str, Any]
    ) -> dict[str, Any]:
        controller = _mapping(_mapping(workflow, "services"), "controller")
        if controller.get("mode", "local") == "external":
            api_key_env = _string(
                controller.get("api_key_env"), "controller.api_key_env"
            )
            attestation = _external_controller_attestation(
                controller,
                credential_present=bool(os.environ.get(api_key_env)),
            )
            self._controller_attestation = attestation
            return attestation
        try:
            attestation = _validated_controller_attestation(
                attest_llama_cpp_runtime(
                    archive_path=_string(
                        controller.get("archive_path"), "controller.archive_path"
                    ),
                    expected_archive_sha256=LLAMA_ARCHIVE_SHA256,
                    binary_path=_string(
                        controller.get("binary_path"), "controller.binary_path"
                    ),
                    gguf_path=_string(controller.get("gguf_path"), "controller.gguf_path"),
                    expected_gguf_sha256=CONTROLLER_GGUF_SHA256,
                    expected_build_number=10516,
                    version_tag="b10516",
                )
            )
        except LlamaRuntimeAttestationError as error:
            raise OwnedServicesConfigError(
                f"llama.cpp b10516 runtime attestation failed: {error}"
            ) from error
        self._controller_attestation = attestation
        return attestation

    def configure_attempt(
        self, attempt: AttemptResult, workflow: Mapping[str, Any]
    ) -> None:
        self._attempt, self._workflow = attempt, workflow

    @staticmethod
    def _proc_identity(pid: int) -> tuple[int, int] | None:
        stat_path = Path(f"/proc/{pid}/stat")
        try:
            text = stat_path.read_text(encoding="ascii")
        except FileNotFoundError:
            return None
        except OSError as error:
            raise RuntimeError(f"cannot read {stat_path}: {error}") from error
        closing = text.rfind(")")
        if closing < 0:
            raise RuntimeError(f"cannot parse /proc/{pid}/stat")
        fields = text[closing + 2 :].split()
        try:
            process_group_id = int(fields[2])
            starttime_ticks = int(fields[19])
        except (IndexError, ValueError) as error:
            raise RuntimeError(f"cannot parse /proc/{pid}/stat") from error
        return starttime_ticks, process_group_id

    @classmethod
    def _starttime(cls, pid: int) -> int:
        identity = cls._proc_identity(pid)
        if identity is None:
            raise FileNotFoundError(f"/proc/{pid}/stat")
        return identity[0]

    @staticmethod
    def _resolve_env(env: Mapping[str, str]) -> dict[str, str]:
        result = {
            name: value
            for name, value in os.environ.items()
            if name not in _CAPSULE_CREDENTIAL_ENV_NAMES
        }
        for name, value in env.items():
            if value.startswith("{env:") and value.endswith("}"):
                source = value[5:-1]
                secret = os.environ.get(source)
                if not secret:
                    raise OwnedServicesConfigError(
                        f"required credential environment {source!r} is unset"
                    )
                result[name] = secret
            else:
                result[name] = value
        return result

    def spawn(self, name: str, argv: list[str], env: dict[str, str]) -> ProcessIdentity:
        if self._attempt is None or self._workflow is None:
            raise RuntimeError("runtime attempt was not configured")
        log_path = self._attempt.artifact_dir / f"launcher_service_{name}.log"
        resolved_env = self._resolve_env(env)
        if name == "controller":
            resolved_env = sanitize_dynamic_loader_environment(resolved_env)
        with log_path.open("xb") as log_stream:
            process = subprocess.Popen(
                argv,
                cwd=str(self._workflow["runtime"]["repo_root"]),
                env=resolved_env,
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            starttime = self._starttime(process.pid)
            if name == "controller" and self._controller_attestation is not None:
                running_sha256 = _file_content_sha256(Path(f"/proc/{process.pid}/exe"))
                if running_sha256 != self._controller_attestation["binary_sha256"]:
                    raise OwnedServicesConfigError(
                        "running Controller executable does not match the attested llama-server"
                    )
        except BaseException:
            process.terminate()
            process.wait(timeout=10)
            raise
        identity = ProcessIdentity(name=name, pid=process.pid, starttime_ticks=starttime)
        self._processes[(identity.pid, identity.starttime_ticks)] = process
        return identity

    def wait_ready(self, identity: ProcessIdentity) -> None:
        if self._workflow is None:
            raise RuntimeError("runtime attempt was not configured")
        readiness = _mapping(
            _mapping(_mapping(self._workflow, "services"), identity.name), "readiness"
        )
        deadline = time.monotonic() + float(readiness.get("timeout_s", 300))
        headers: dict[str, str] = {}
        token_env = readiness.get("bearer_token_env")
        if token_env is not None:
            token = os.environ.get(_string(token_env, "bearer_token_env"))
            if not token:
                raise OwnedServicesConfigError(f"readiness credential {token_env!r} is unset")
            headers["Authorization"] = f"Bearer {token}"
        process = self._processes[(identity.pid, identity.starttime_ticks)]
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"{identity.name} exited before readiness")
            try:
                request = urllib.request.Request(readiness["url"], headers=headers, method="GET")
                with urllib.request.urlopen(request, timeout=3) as response:
                    if 200 <= response.status < 300:
                        response.read(65536)
                        return
            except (OSError, urllib.error.URLError) as error:
                last_error = error
            time.sleep(1)
        raise TimeoutError(f"{identity.name} readiness timed out: {last_error}")

    def run_gate(self, gate_name: str, argv: list[str], env: dict[str, str]) -> None:
        if self._attempt is None or self._workflow is None:
            raise RuntimeError("runtime attempt was not configured")
        log_path = self._attempt.artifact_dir / f"launcher_{gate_name}.log"
        with log_path.open("xb") as log_stream:
            result = subprocess.run(
                argv,
                cwd=str(self._workflow["runtime"]["repo_root"]),
                env=self._resolve_env(env),
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            tail = log_path.read_bytes()[-256 * 1024 :].lower()
            guided_retry = (
                gate_name == "gate05_guided"
                and b"no pt/p_hat double-success group after" in tail
            )
            raise GateCommandError(
                gate_name,
                result.returncode,
                oom=(
                    gate_name in GPU_OOM_GATES
                    and _gate_log_indicates_gpu_oom(tail)
                ),
                guided_retry=guided_retry,
            )

    def terminate(self, identity: ProcessIdentity) -> None:
        process = self._processes.pop((identity.pid, identity.starttime_ticks), None)
        if process is None or process.poll() is not None:
            return
        try:
            if self._starttime(identity.pid) != identity.starttime_ticks:
                return
        except FileNotFoundError:
            return
        try:
            os.killpg(identity.pid, signal.SIGTERM)
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            if self._starttime(identity.pid) == identity.starttime_ticks:
                os.killpg(identity.pid, signal.SIGKILL)
            process.wait(timeout=10)

    def confirm_terminated(self, identity: ProcessIdentity) -> bool:
        """Confirm both the recorded PID identity and its owned process group are gone."""

        running_identity = self._proc_identity(identity.pid)
        if running_identity is not None and running_identity[0] == identity.starttime_ticks:
            return False
        try:
            os.killpg(identity.pid, 0)
        except ProcessLookupError:
            return True
        except OSError as error:
            raise RuntimeError(
                f"cannot probe owned process group {identity.pid}: {error}"
            ) from error
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the one-A800 Capsule-RL workflow.")
    parser.add_argument("--workflow-config", type=Path, required=True)
    parser.add_argument("--profile-config", type=Path, required=True)
    parser.add_argument("--capsule-config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = execute_owned_service_workflow(
        workflow_path=args.workflow_config,
        profile_path=args.profile_config,
        capsule_config_path=args.capsule_config,
        runtime=LinuxRuntime(),
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "mode": "DRY RUN" if args.dry_run else "EXECUTE",
                "run_id": result.run_id,
                "output_dir": str(result.output_dir),
                "artifact_dir": str(result.artifact_dir),
                "capsule_config_path": str(result.capsule_config_path),
                "gate7_audit_path": str(result.artifact_dir / "gate07_audit.json"),
                "controller_seed_run_ids": result.controller_seed_run_ids,
                "audit": asdict(result.audit),
                "services": {
                    name: {"argv": command.argv, "env": command.env}
                    for name, command in result.rendered_services.items()
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
