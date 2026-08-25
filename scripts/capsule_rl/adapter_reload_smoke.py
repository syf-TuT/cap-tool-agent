"""Reload a Gate 6 PEFT adapter on a fresh FP32 base after the Ray process exits."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from capx.rl.capsule.actor_identity import ActorIdentityError, build_actor_identity
from capx.rl.capsule.stable_io import (
    MutationWatch,
    PathMutationGuard,
    StablePathError,
    read_stable_regular_file,
)

from .common import (
    ADAPTER_RELOAD_PROMPT,
    CANONICAL_EXECUTION_MODE,
    GateArtifactError,
    atomic_write_json,
    direct_lora_adapter_evidence,
    load_and_validate_server_config_bytes,
    verify_adapter_reload_artifact,
    verify_trainer_gate_artifact,
)


RELOAD_PROMPT = ADAPTER_RELOAD_PROMPT


class AdapterReloadError(RuntimeError):
    """The independent adapter reload smoke cannot produce valid evidence."""


@contextmanager
def _guard_reload_inputs(
    *,
    config_path: Path,
    gate6_path: Path,
    base_model_path: Path,
    adapter_path: Path,
    resolved_verl_config_path: Path,
):
    watches = (
        MutationWatch(config_path, "resolved Capsule config"),
        MutationWatch(gate6_path, "Gate 6 artifact"),
        MutationWatch(base_model_path, "base model tree", recursive=True),
        MutationWatch(adapter_path, "LoRA adapter tree", recursive=True),
        MutationWatch(resolved_verl_config_path, "resolved VeRL config"),
    )
    try:
        guard = PathMutationGuard.open(watches)
    except StablePathError as error:
        raise AdapterReloadError(f"cannot guard adapter reload inputs: {error}") from error
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    primary_error: BaseException | None = None
    try:
        yield guard
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            guard.assert_unchanged(context="during the FP32 adapter reload")
        except StablePathError as error:
            translated = AdapterReloadError(str(error))
            if primary_error is None:
                raise translated from error
            add_note = getattr(primary_error, "add_note", None)
            if callable(add_note):
                add_note(f"adapter reload mutation guard also failed: {error}")
        finally:
            sys.dont_write_bytecode = previous_dont_write_bytecode
            guard.close()


def _host_mem_available_bytes(path: Path = Path("/proc/meminfo")) -> int:
    try:
        for line in path.read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB":
                    value = int(fields[1]) * 1024
                    if value > 0:
                        return value
    except (OSError, UnicodeError, ValueError) as error:
        raise AdapterReloadError(f"cannot read host MemAvailable: {error}") from error
    raise AdapterReloadError("/proc/meminfo omitted a positive MemAvailable value")


def run_fp32_adapter_reload(
    *,
    base_model_path: Path,
    adapter_path: Path,
) -> dict[str, Any]:
    """Load fresh local model bytes, compare adapter-off/on logits, and release CUDA."""

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise AdapterReloadError("adapter reload requires exactly one visible CUDA GPU")
    from .server_adapter import _HostMemoryMonitor

    device = "cuda:0"
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    tokenizer: Any = None
    base_model: Any = None
    model: Any = None
    memory_monitor = _HostMemoryMonitor(
        loader=_host_mem_available_bytes,
        monitor_scope="adapter_reload_model_load_through_cuda_release",
    )
    with memory_monitor:
        host_before = memory_monitor.sample()
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(base_model_path),
                local_files_only=True,
                trust_remote_code=False,
            )
            memory_monitor.sample()
            base_model = AutoModelForCausalLM.from_pretrained(
                str(base_model_path),
                torch_dtype=torch.float32,
                device_map={"": device},
                local_files_only=True,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
            )
            memory_monitor.sample()
            non_fp32 = [
                name
                for name, parameter in base_model.named_parameters()
                if parameter.is_floating_point() and parameter.dtype != torch.float32
            ]
            if non_fp32:
                raise AdapterReloadError(
                    f"fresh base model contains non-FP32 parameters: {non_fp32[:3]!r}"
                )
            non_cuda = [
                name
                for name, parameter in base_model.named_parameters()
                if parameter.device.type != "cuda"
            ]
            if non_cuda:
                raise AdapterReloadError(
                    f"fresh base model contains parameters outside CUDA: {non_cuda[:3]!r}"
                )
            model = PeftModel.from_pretrained(
                base_model,
                str(adapter_path),
                is_trainable=False,
                local_files_only=True,
            )
            memory_monitor.sample()
            adapter_non_cuda = [
                name
                for name, parameter in model.named_parameters()
                if ".lora_" in name.lower() and parameter.device.type != "cuda"
            ]
            if adapter_non_cuda:
                raise AdapterReloadError(
                    "reloaded adapter contains parameters outside CUDA: "
                    f"{adapter_non_cuda[:3]!r}"
                )
            model.eval()
            encoded = tokenizer(RELOAD_PROMPT, return_tensors="pt", add_special_tokens=True)
            input_ids = encoded["input_ids"].to(device)
            input_token_count = int(input_ids.numel())
            if input_token_count < 1:
                raise AdapterReloadError("fixed reload prompt tokenized to an empty input")
            with torch.inference_mode():
                with model.disable_adapter():
                    disabled_logits = model(input_ids=input_ids).logits[..., -1, :].float()
                enabled_logits = model(input_ids=input_ids).logits[..., -1, :].float()
                disabled_finite = bool(torch.isfinite(disabled_logits).all().item())
                enabled_finite = bool(torch.isfinite(enabled_logits).all().item())
                maximum_difference = float(
                    torch.max(torch.abs(enabled_logits - disabled_logits)).item()
                )
            memory_monitor.sample()
            cuda_peak = int(torch.cuda.max_memory_reserved(0))
            if not disabled_finite or not enabled_finite:
                raise AdapterReloadError("adapter-off/on logits must both be finite")
            if not math.isfinite(maximum_difference) or maximum_difference <= 1e-8:
                raise AdapterReloadError(
                    "loaded adapter did not change logits by more than 1e-8"
                )
            if cuda_peak > 70 * 1024**3:
                raise AdapterReloadError(
                    "adapter reload CUDA peak reserved memory exceeded 70 GiB"
                )
        finally:
            del model, base_model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            host_after = memory_monitor.sample()
    host_memory = memory_monitor.evidence()
    host_minimum = int(host_memory["minimum_mem_available_bytes"])
    if host_minimum < 12 * 1024**3:
        raise AdapterReloadError("adapter reload host MemAvailable fell below 12 GiB")
    return {
        "base_model_path": str(base_model_path),
        "base_model_dtype": "float32",
        "device": device,
        "prompt_sha256": hashlib.sha256(RELOAD_PROMPT.encode("utf-8")).hexdigest(),
        "input_token_count": input_token_count,
        "adapter_disabled_logits_finite": True,
        "adapter_enabled_logits_finite": True,
        "max_abs_logit_diff": maximum_difference,
        "cuda_peak_reserved_bytes": cuda_peak,
        "host_mem_available_before_bytes": host_before,
        "host_mem_available_after_bytes": host_after,
        "host_mem_available_min_bytes": host_minimum,
        "host_memory": host_memory,
    }


def _project_path(config: Mapping[str, Any], value: object, field_name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AdapterReloadError(f"{field_name} must be a non-empty path")
    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping):
        raise AdapterReloadError("runtime must be a mapping")
    configured_root = runtime.get("project_root")
    project_root = (
        Path(configured_root).expanduser().resolve()
        if isinstance(configured_root, str) and configured_root
        else Path(__file__).resolve().parents[2]
    )
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate6-artifact", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    smoke_runner: Callable[..., Mapping[str, Any]] = run_fp32_adapter_reload,
) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(os.path.abspath(args.config.expanduser()))
    gate6_path = Path(os.path.abspath(args.gate6_artifact.expanduser()))
    artifact_path = Path(os.path.abspath(args.artifact.expanduser()))
    if artifact_path.exists():
        raise FileExistsError(f"adapter reload artifact already exists: {artifact_path}")
    try:
        config_snapshot = read_stable_regular_file(config_path, label="resolved config")
        gate6_snapshot = read_stable_regular_file(gate6_path, label="Gate 6 artifact")
        config = load_and_validate_server_config_bytes(
            config_snapshot.raw_bytes, check_runtime_paths=True
        )
        gate6 = json.loads(gate6_snapshot.raw_bytes)
    except (StablePathError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterReloadError(f"cannot load adapter reload inputs: {error}") from error
    if not isinstance(gate6, Mapping):
        raise AdapterReloadError("Gate 6 artifact root must be a JSON mapping")
    verify_trainer_gate_artifact(gate6)
    if gate6.get("run_id") != args.run_id:
        raise AdapterReloadError("Gate 6 run_id does not match --run-id")
    if gate6.get("config_sha256") != config_snapshot.sha256:
        raise AdapterReloadError("Gate 6 config SHA-256 does not match --config")
    ray_release = gate6.get("ray_release")
    if ray_release != {
        "worker_close_calls": 1,
        "ray_shutdown_calls": 1,
        "ray_shutdown_complete": True,
    }:
        raise AdapterReloadError("Gate 6 did not prove complete Ray release")
    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping):
        raise AdapterReloadError("runtime must be a mapping")
    base_model_path = _project_path(
        config, runtime.get("program_model_path"), "runtime.program_model_path"
    )
    adapter_path = Path(str(gate6["adapter_path"]))
    if not adapter_path.is_absolute():
        raise AdapterReloadError("Gate 6 adapter_path must be absolute")
    resolved_verl_config_path = _project_path(
        config,
        runtime.get("verl_resolved_config_path"),
        "runtime.verl_resolved_config_path",
    )
    try:
        actor_identity_before = build_actor_identity(config)
    except ActorIdentityError as error:
        raise AdapterReloadError(f"cannot verify actor identity before reload: {error}") from error
    if (
        actor_identity_before["program_model_sha256"]
        != gate6.get("program_model_sha256")
        or actor_identity_before["actor_binding_sha256"]
        != gate6.get("actor_binding_sha256")
        or actor_identity_before["verl_resolved_config_sha256"]
        != gate6.get("verl_resolved_config_sha256")
        or Path(str(actor_identity_before["program_model_path"])) != base_model_path
    ):
        raise AdapterReloadError("current actor identity does not match Gate 6")
    plan = {
        "mode": "VALIDATION ONLY" if args.validate_only or args.dry_run else "EXECUTE",
        "run_id": args.run_id,
        "base_model_path": str(base_model_path),
        "adapter_path": str(adapter_path),
        "gate06_artifact_sha256": gate6_snapshot.sha256,
        "artifact": str(artifact_path),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    if args.validate_only or args.dry_run:
        return 0
    with _guard_reload_inputs(
        config_path=config_path,
        gate6_path=gate6_path,
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        resolved_verl_config_path=resolved_verl_config_path,
    ):
        try:
            if (
                read_stable_regular_file(config_path, label="resolved config").sha256
                != config_snapshot.sha256
                or read_stable_regular_file(gate6_path, label="Gate 6 artifact").sha256
                != gate6_snapshot.sha256
            ):
                raise AdapterReloadError(
                    "adapter reload config/Gate 6 inputs changed before guarded loading"
                )
        except StablePathError as error:
            raise AdapterReloadError(f"cannot revalidate guarded reload inputs: {error}") from error
        try:
            guarded_actor_identity = build_actor_identity(config)
        except ActorIdentityError as error:
            raise AdapterReloadError(
                f"cannot verify guarded actor identity before reload: {error}"
            ) from error
        if guarded_actor_identity != actor_identity_before:
            raise AdapterReloadError("actor identity changed before guarded adapter loading")
        try:
            adapter_before = direct_lora_adapter_evidence(adapter_path.parent)
        except GateArtifactError as error:
            raise AdapterReloadError(
                f"cannot verify guarded adapter before reload: {error}"
            ) from error
        if any(gate6.get(field) != value for field, value in adapter_before.items()):
            raise AdapterReloadError("current adapter identity does not match Gate 6")

        smoke = dict(
            smoke_runner(base_model_path=base_model_path, adapter_path=adapter_path)
        )
        expected_smoke_fields = {
            "base_model_path",
            "base_model_dtype",
            "device",
            "prompt_sha256",
            "input_token_count",
            "adapter_disabled_logits_finite",
            "adapter_enabled_logits_finite",
            "max_abs_logit_diff",
            "cuda_peak_reserved_bytes",
            "host_mem_available_before_bytes",
            "host_mem_available_after_bytes",
            "host_mem_available_min_bytes",
            "host_memory",
        }
        if set(smoke) != expected_smoke_fields:
            raise AdapterReloadError(
                "adapter reload runner returned an incomplete or unexpected evidence schema"
            )
        try:
            actor_identity_after = build_actor_identity(config)
        except ActorIdentityError as error:
            raise AdapterReloadError(
                f"cannot verify actor identity after reload: {error}"
            ) from error
        if actor_identity_after != actor_identity_before:
            raise AdapterReloadError("actor identity changed during adapter reload smoke")
        try:
            adapter_after = direct_lora_adapter_evidence(adapter_path.parent)
        except GateArtifactError as error:
            raise AdapterReloadError(f"adapter changed during reload smoke: {error}") from error
        if any(gate6.get(field) != value for field, value in adapter_after.items()):
            raise AdapterReloadError("adapter identity changed during reload smoke")
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
    payload: dict[str, Any] = {
        "schema_version": 1,
        "gate": "adapter_reload",
        "passed": True,
        "execution_mode": CANONICAL_EXECUTION_MODE,
        **{field: gate6[field] for field in identity_fields},
        "gate06_artifact": str(gate6_path),
        "gate06_artifact_sha256": gate6_snapshot.sha256,
        "ray_release": dict(ray_release),
        "adapter_path": gate6["adapter_path"],
        "adapter_model_sha256": gate6["adapter_model_sha256"],
        "adapter_config_sha256": gate6["adapter_config_sha256"],
        **smoke,
    }
    verify_adapter_reload_artifact(payload)
    atomic_write_json(artifact_path, payload)
    print(f"adapter_reload: PASS ({artifact_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AdapterReloadError",
    "RELOAD_PROMPT",
    "build_parser",
    "main",
    "run_fp32_adapter_reload",
]
