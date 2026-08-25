from __future__ import annotations

import copy
import hashlib
import importlib
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from capx.rl.capsule import server_factory

from scripts.capsule_rl import (
    adapter_reload_smoke,
    analyze_artifacts,
    server_adapter,
    server_preflight,
)

from scripts.capsule_rl.launch_owned_services import (
    AuditSnapshot,
    FakeFailure,
    GateCommandError,
    LinuxRuntime,
    OwnedServicesConfigError,
    ProcessIdentity,
    RuntimeContext,
    build_controller_seed_run_ids,
    execute_owned_service_workflow,
    main,
    load_owned_services_workflow,
    load_single_a800_resolved_profile,
    materialize_retry_profile,
    _gate_artifacts,
    _llama_build_number,
    _render_gate_commands,
    _render_services,
)


PROFILE_PATH = (
    Path(__file__).parents[1]
    / "env_configs"
    / "cube_stack"
    / "capsule_rl"
    / "franka_robosuite_cube_stack_capsule_single_a800_verl.yaml"
)

WORKFLOW_PATH = (
    Path(__file__).parents[1]
    / "env_configs"
    / "cube_stack"
    / "capsule_rl"
    / "franka_robosuite_cube_stack_capsule_single_a800_owned_services.yaml"
)

CAPSULE_PATH = (
    Path(__file__).parents[1]
    / "env_configs"
    / "cube_stack"
    / "capsule_rl"
    / "franka_robosuite_cube_stack_capsule_critique_grpo.yaml"
)


@dataclass
class FakeProcess:
    pid: int
    starttime_ticks: int


class FakeRuntime:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.spawned: list[tuple[str, list[str], dict[str, str]]] = []
        self.gates: list[str] = []
        self.terminated: list[tuple[int, int]] = []
        self.active_identities: set[tuple[int, int]] = set()
        self.ready_fail_after: set[str] = set()
        self.gate_failure: str | None = None
        self.terminate_failure_name: str | None = None
        self.noop_terminate_names: set[str] = set()
        self.unconfirmed_termination_names: set[str] = set()
        self.confirmed: list[tuple[int, int]] = []
        self.next_pid = 4100
        self.events: list[str] = []
        self.mem_available_values = [100000] * 64
        self.continuous_mem_available_value = 20000
        self.memory_monitor_interval_s = 0.001
        self.verified_inputs = 0
        self.oom_failures_remaining = 0
        self.oom_failure_gate = "gate06_trainer"
        self.guided_failures_remaining = 0
        self.breach_during_gate7 = False
        self.configured_attempts = []

    def collect_audit_snapshot(self, _context: RuntimeContext) -> AuditSnapshot:
        self.events.append("audit")
        return AuditSnapshot(
            gpu_name="NVIDIA A800 80GB PCIe",
            gpu_count=1,
            gpu_total_vram_mib=81920,
            gpu_free_vram_mib=80000,
            other_gpu_processes_mib=[128, 256],
            host_memory_mib=131072,
            mem_available_before_controller_mib=110000,
            mem_available_after_controller_mib=98000,
            mem_available_during_run_mib=20000,
            shm_available_mib=16384,
            disk_free_mib=100000,
            cuda_version="12.8",
            nvidia_driver="570.00",
            repo_head="deadbeef",
            repo_is_dirty=True,
        )

    def verify_runtime_inputs(self, _context: RuntimeContext, _workflow: dict) -> dict:
        self.verified_inputs += 1
        self.events.append("verify-inputs")
        return {
            "schema_version": 1,
            "artifact_type": "llama_cpp_b10516_runtime_attestation",
            "version_tag": "b10516",
            "archive_path": "/fake/llama.tar.gz",
            "archive_sha256": (
                "f263a91280471b4c33c4999d7c76259c0f3a0a53a0b3e692b2c0b84380137a35"
            ),
            "binary_path": "/fake/llama-server",
            "binary_archive_member": "llama-server",
            "binary_sha256": "b" * 64,
            "gguf_path": "/fake/controller.gguf",
            "gguf_sha256": (
                "509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c"
            ),
            "build_number": 10516,
            "runtime_tree_sha256": "d" * 64,
            "regular_file_count": 2,
            "symlink_count": 1,
        }

    def mem_available_mib(self) -> int:
        value = self.mem_available_values.pop(0)
        self.events.append(f"memory:{value}")
        return value

    def continuous_mem_available_mib(self) -> int:
        return self.continuous_mem_available_value

    def configure_attempt(self, attempt, _workflow) -> None:
        self.configured_attempts.append(attempt)

    def spawn(self, name: str, argv: list[str], env: dict[str, str]) -> ProcessIdentity:
        self.next_pid += 1
        process = FakeProcess(pid=self.next_pid, starttime_ticks=self.next_pid * 10)
        self.spawned.append((name, argv, env))
        self.active_identities.add((process.pid, process.starttime_ticks))
        self.events.append(f"spawn:{name}")
        return ProcessIdentity(
            name=name,
            pid=process.pid,
            starttime_ticks=process.starttime_ticks,
        )

    def wait_ready(self, identity: ProcessIdentity) -> None:
        self.events.append(f"ready:{identity.name}")
        if identity.name in self.ready_fail_after:
            raise RuntimeError(f"{identity.name} readiness failed")

    def run_gate(self, gate_name: str, argv: list[str], env: dict[str, str]) -> None:
        self.gates.append(gate_name)
        self.events.append(f"gate:{gate_name}")
        if gate_name == "gate07_audit" and self.breach_during_gate7:
            self.continuous_mem_available_value = 12000
            time.sleep(0.01)
        if gate_name == self.oom_failure_gate and self.oom_failures_remaining:
            self.oom_failures_remaining -= 1
            raise GateCommandError(gate_name, 1, oom=True)
        if gate_name == "gate05_guided" and self.guided_failures_remaining:
            self.guided_failures_remaining -= 1
            raise GateCommandError(gate_name, 1, oom=False, guided_retry=True)
        if self.gate_failure == gate_name:
            raise FakeFailure(f"{gate_name} failed")

    def terminate(self, identity: ProcessIdentity) -> None:
        if identity.name in self.noop_terminate_names:
            self.events.append(f"terminate-noop:{identity.name}")
            return
        self.terminated.append((identity.pid, identity.starttime_ticks))
        self.active_identities.discard((identity.pid, identity.starttime_ticks))
        self.events.append(f"terminate:{identity.name}")
        if identity.name == self.terminate_failure_name:
            raise RuntimeError(f"{identity.name} terminate failed")

    def confirm_terminated(self, identity: ProcessIdentity) -> bool:
        self.confirmed.append((identity.pid, identity.starttime_ticks))
        self.events.append(f"confirm-terminated:{identity.name}")
        process_identity = (identity.pid, identity.starttime_ticks)
        return (
            process_identity not in self.active_identities
            and identity.name not in self.unconfirmed_termination_names
        )


def test_repository_single_a800_profile_matches_exact_contract() -> None:
    profile = load_single_a800_resolved_profile(PROFILE_PATH)
    actor_rollout_ref = profile["actor_rollout_ref"]
    model = actor_rollout_ref["model"]
    actor = actor_rollout_ref["actor"]
    rollout = actor_rollout_ref["rollout"]
    ref = actor_rollout_ref["ref"]

    assert model["path"] == (
        "/root/autodl-tmp/cap-x/.codex-downloads/models/"
        "Qwen2.5-Coder-7B-Instruct"
    )
    assert model["lora_rank"] == 16
    assert model["lora_alpha"] == 32
    assert model["target_modules"] == "all-linear"
    assert model["enable_gradient_checkpointing"] is True
    assert model["use_remove_padding"] is True
    assert model["use_shm"] is False
    assert actor["strategy"] == "fsdp"
    assert actor["use_dynamic_bsz"] is True
    assert actor["ppo_max_token_len_per_gpu"] == 10240
    assert actor["fsdp_config"]["param_offload"] is True
    assert actor["fsdp_config"]["optimizer_offload"] is False
    assert model["enable_activation_offload"] is False
    assert rollout["dtype"] == "bfloat16"
    assert rollout["tensor_model_parallel_size"] == 1
    assert rollout["data_parallel_size"] == 1
    assert rollout["pipeline_model_parallel_size"] == 1
    assert rollout["gpu_memory_utilization"] == pytest.approx(0.30)
    assert rollout["free_cache_engine"] is True
    assert rollout["enforce_eager"] is True
    assert rollout["max_num_batched_tokens"] == 10240
    assert rollout["max_model_len"] == 10240
    assert rollout["enable_chunked_prefill"] is True
    assert rollout["max_num_seqs"] == 1
    assert rollout["load_format"] == "safetensors"
    assert rollout["log_prob_use_dynamic_bsz"] is True
    assert rollout["log_prob_max_token_len_per_gpu"] == 10240
    assert actor["fsdp_config"]["model_dtype"] == "fp32"
    assert ref["log_prob_use_dynamic_bsz"] is True
    assert ref["log_prob_max_token_len_per_gpu"] == 10240
    assert ref["fsdp_config"]["param_offload"] is True
    assert profile["trainer"]["n_gpus_per_node"] == 1
    assert profile["trainer"]["nnodes"] == 1
    assert profile["trainer"]["total_epochs"] == 1


def test_single_a800_profile_has_every_pre_ray_worker_access_surface() -> None:
    profile = load_single_a800_resolved_profile(PROFILE_PATH)
    root = profile["actor_rollout_ref"]

    assert root["model"]["_target_"] == "verl.workers.config.HFModelConfig"
    assert root["model"]["exclude_modules"] is None
    assert root["model"]["lora_adapter_path"] is None
    assert root["actor"]["_target_"] == "verl.workers.config.FSDPActorConfig"
    assert root["actor"]["optim"]["_target_"] == (
        "verl.workers.config.FSDPOptimizerConfig"
    )
    assert root["actor"]["fsdp_config"]["_target_"] == (
        "verl.workers.config.FSDPEngineConfig"
    )
    assert root["actor"]["policy_loss"]["_target_"] == (
        "capx.rl.capsule.verl_config.CapsulePolicyLossConfig"
    )
    assert root["actor"]["policy_loss"]["capsule_gamma"] == pytest.approx(0.1)
    assert root["actor"]["ppo_micro_batch_size"] is None
    assert root["rollout"]["_target_"] == "verl.workers.config.RolloutConfig"
    assert root["rollout"]["temperature"] == pytest.approx(1.0)
    assert root["rollout"]["log_prob_micro_batch_size"] is None
    assert root["ref"]["log_prob_micro_batch_size"] is None
    assert root["ref"]["fsdp_config"]["_target_"] == (
        "verl.workers.config.FSDPEngineConfig"
    )
    assert profile["trainer"]["device"] == "cuda"
    assert profile["data"]["trust_remote_code"] is False
    assert profile["ray_kwargs"]["ray_init"] == {"num_cpus": None}


@pytest.mark.parametrize(
    "dotted_path",
    [
        "actor_rollout_ref.model._target_",
        "actor_rollout_ref.model.exclude_modules",
        "actor_rollout_ref.model.lora_adapter_path",
        "actor_rollout_ref.actor._target_",
        "actor_rollout_ref.actor.optim._target_",
        "actor_rollout_ref.actor.fsdp_config._target_",
        "actor_rollout_ref.actor.policy_loss._target_",
        "actor_rollout_ref.actor.policy_loss.capsule_gamma",
        "actor_rollout_ref.actor.ppo_micro_batch_size",
        "actor_rollout_ref.rollout._target_",
        "actor_rollout_ref.rollout.temperature",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size",
        "actor_rollout_ref.ref.log_prob_micro_batch_size",
        "actor_rollout_ref.ref.fsdp_config._target_",
        "trainer.device",
        "data.trust_remote_code",
        "ray_kwargs.ray_init",
    ],
)
def test_profile_loader_fails_fast_on_missing_worker_surface(
    dotted_path: str, tmp_path: Path
) -> None:
    profile = copy.deepcopy(yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8")))
    node = profile
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        node = node[part]
    del node[parts[-1]]
    broken = tmp_path / "missing-worker-surface.yaml"
    broken.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")

    with pytest.raises(OwnedServicesConfigError, match=dotted_path.replace(".", r"\.")):
        load_single_a800_resolved_profile(broken)


def test_profile_round_trips_through_capsule_factory_before_ray(tmp_path: Path) -> None:
    config = yaml.safe_load(CAPSULE_PATH.read_text(encoding="utf-8"))
    config["runtime"].update(
        {
            "verl_resolved_config_path": str(PROFILE_PATH),
            "program_model_path": str(tmp_path / "materialized-base-model"),
            "output_dir": str(tmp_path / "outputs"),
        }
    )

    resolved = server_factory._load_resolved_verl_config(
        config, Path(__file__).parents[1]
    )

    assert resolved.ray_kwargs.ray_init.num_cpus is None
    assert resolved.trainer.device == "cuda"
    assert resolved.data.get("trust_remote_code") is False
    assert resolved.actor_rollout_ref.actor.ppo_micro_batch_size is None
    assert resolved.actor_rollout_ref.rollout.log_prob_micro_batch_size is None
    assert resolved.actor_rollout_ref.ref.log_prob_micro_batch_size is None


def test_capsule_policy_loss_target_accepts_and_preserves_gamma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import dataclass

    @dataclass
    class FakePolicyLossConfig:
        loss_mode: str = "vanilla"

    fake_verl = types.ModuleType("verl")
    fake_workers = types.ModuleType("verl.workers")
    fake_config = types.ModuleType("verl.workers.config")
    fake_config.PolicyLossConfig = FakePolicyLossConfig
    fake_workers.config = fake_config
    fake_verl.workers = fake_workers
    monkeypatch.setitem(sys.modules, "verl", fake_verl)
    monkeypatch.setitem(sys.modules, "verl.workers", fake_workers)
    monkeypatch.setitem(sys.modules, "verl.workers.config", fake_config)
    monkeypatch.delitem(sys.modules, "capx.rl.capsule.verl_config", raising=False)
    capsule_package = sys.modules["capx.rl.capsule"]
    if hasattr(capsule_package, "verl_config"):
        delattr(capsule_package, "verl_config")

    module = importlib.import_module("capx.rl.capsule.verl_config")
    converted = module.CapsulePolicyLossConfig(
        loss_mode="capsule_critique", capsule_gamma=0.1
    )

    assert isinstance(converted, FakePolicyLossConfig)
    assert converted.loss_mode == "capsule_critique"
    assert converted.capsule_gamma == pytest.approx(0.1)
    with pytest.raises(ValueError, match="capsule_gamma must be positive"):
        module.CapsulePolicyLossConfig(capsule_gamma=0.0)
    sys.modules.pop("capx.rl.capsule.verl_config", None)
    delattr(capsule_package, "verl_config")


def test_capsule_policy_loss_survives_recursive_hydra_actor_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import dataclass

    hydra_utils = pytest.importorskip("hydra.utils")
    omegaconf = pytest.importorskip("omegaconf")

    @dataclass
    class FakePolicyLossConfig:
        loss_mode: str = "vanilla"

    @dataclass
    class FakeFSDPActorConfig:
        policy_loss: object

    fake_verl = types.ModuleType("verl")
    fake_workers = types.ModuleType("verl.workers")
    fake_config = types.ModuleType("verl.workers.config")
    fake_config.PolicyLossConfig = FakePolicyLossConfig
    fake_config.FSDPActorConfig = FakeFSDPActorConfig
    fake_workers.config = fake_config
    fake_verl.workers = fake_workers
    monkeypatch.setitem(sys.modules, "verl", fake_verl)
    monkeypatch.setitem(sys.modules, "verl.workers", fake_workers)
    monkeypatch.setitem(sys.modules, "verl.workers.config", fake_config)
    monkeypatch.delitem(sys.modules, "capx.rl.capsule.verl_config", raising=False)
    capsule_package = sys.modules["capx.rl.capsule"]
    if hasattr(capsule_package, "verl_config"):
        delattr(capsule_package, "verl_config")

    converted = hydra_utils.instantiate(
        omegaconf.OmegaConf.create(
            {
                "_target_": "verl.workers.config.FSDPActorConfig",
                "policy_loss": {
                    "_target_": (
                        "capx.rl.capsule.verl_config.CapsulePolicyLossConfig"
                    ),
                    "loss_mode": "capsule_critique",
                    "capsule_gamma": 0.1,
                },
            }
        ),
        _convert_="partial",
    )

    assert isinstance(converted, FakeFSDPActorConfig)
    assert isinstance(converted.policy_loss, FakePolicyLossConfig)
    assert converted.policy_loss.capsule_gamma == pytest.approx(0.1)
    sys.modules.pop("capx.rl.capsule.verl_config", None)
    delattr(capsule_package, "verl_config")


def test_llama_build_number_parses_only_bounded_numeric_release() -> None:
    assert _llama_build_number("version: 10516 (abcdef)\nbuilt with cc") == 10516
    assert _llama_build_number("build: 10516, commit: abcdef") == 10516
    assert _llama_build_number("version: 10515 (abcdef)") == 10515
    assert _llama_build_number("version: 105160 (abcdef)") == 105160
    assert _llama_build_number("version: b10516") is None


def test_launcher_gate_paths_match_gate7_required_filenames(tmp_path: Path) -> None:
    launcher_paths = _gate_artifacts(tmp_path)

    assert (
        launcher_paths["gate02_seed"].name
        == analyze_artifacts.REQUIRED_GATE_FILES["seed"]
    )
    assert (
        launcher_paths["gate03_oracle_replay"].name
        == analyze_artifacts.REQUIRED_GATE_FILES["oracle_replay"]
    )


def test_linux_runtime_git_probe_disables_optional_locks(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> types.SimpleNamespace:
        captured.update(kwargs)
        return types.SimpleNamespace(stdout="head\n")

    monkeypatch.setattr(
        "scripts.capsule_rl.launch_owned_services.subprocess.run", fake_run
    )

    assert LinuxRuntime._run_text(["git", "rev-parse", "HEAD"]) == "head"
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["GIT_OPTIONAL_LOCKS"] == "0"


def test_repository_owned_workflow_matches_exact_service_and_audit_contract() -> None:
    workflow = load_owned_services_workflow(WORKFLOW_PATH)
    assert workflow["runtime"]["repo_root"] == "/root/autodl-tmp/cap-x"
    assert workflow["hardware"]["gpu_model"] == "A800"
    assert workflow["hardware"]["gpu_total_vram_required_mib"] == 81920
    assert workflow["hardware"]["gpu_count"] == 1
    assert workflow["hardware"]["gpu_free_vram_required_mib"] == 77824
    assert workflow["hardware"]["max_other_process_vram_mib"] == 512
    assert workflow["hardware"]["host_memory_required_mib"] == 122880
    assert workflow["hardware"]["mem_available_after_controller_required_mib"] == 92160
    assert workflow["hardware"]["mem_available_during_run_required_mib"] == 12288
    assert workflow["hardware"]["shm_required_mib"] == 12288
    assert workflow["hardware"]["disk_free_required_mib"] == 81920
    assert workflow["services"]["program"]["argv_template"] == [
        "{python}",
        "-m",
        "capx.rl.capsule.actor_identity",
        "--config",
        "{capsule_config}",
        "--host",
        "127.0.0.1",
        "--port",
        "8101",
    ]
    assert workflow["services"]["controller"]["argv_template"] == [
        "{controller_binary}",
        "--model",
        "{controller_gguf}",
        "--alias",
        "{controller_alias}",
        "--host",
        "127.0.0.1",
        "--port",
        "8102",
        "--n-gpu-layers",
        "0",
        "--parallel",
        "1",
        "--ctx-size",
        "16384",
    ]
    assert workflow["services"]["controller"]["version_tag"] == "b10516"
    assert (
        workflow["services"]["controller"]["model_sha256"]
        == "509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c"
    )
    assert workflow["services"]["controller"]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert workflow["services"]["program"]["env"] == {
        "CUDA_VISIBLE_DEVICES": "",
        "CAPX_PROGRAM_API_KEY": "{env:CAPX_PROGRAM_API_KEY}",
    }
    assert workflow["services"]["pyroki"]["env"] == {
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
    }
    assert workflow["oom_ladder"] == [
        "base_dynamic_fp32",
        "vllm_util_026",
        "fixed_microbatch_1",
        "fsdp_base_bf16",
    ]
    assert workflow["runtime"]["verl_pinned_sha"] == "d62da4950573d7a4b7ef2362337952e7ab59e78d"
    assert workflow["runtime"]["max_controller_seed_run_ids"] == 3


@pytest.mark.parametrize("invalid", [None, 0, 2, 4, True, "3"])
def test_workflow_rejects_non_fixed_controller_seed_run_limit(
    invalid: object, tmp_path: Path
) -> None:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    workflow["runtime"]["max_controller_seed_run_ids"] = invalid
    path = tmp_path / "bad-seed-limit.yaml"
    path.write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")

    with pytest.raises(OwnedServicesConfigError, match="max_controller_seed_run_ids"):
        load_owned_services_workflow(path)


def test_resolved_service_and_gate_environments_isolate_capsule_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CAPX_PROGRAM_API_KEY", "program-secret")
    monkeypatch.setenv("CAPX_CONTROLLER_API_KEY", "controller-secret")
    monkeypatch.setenv("UNRELATED_PARENT_VALUE", "retained")
    workflow = load_owned_services_workflow(WORKFLOW_PATH)
    services = _render_services(workflow, capsule_config_path=CAPSULE_PATH)
    gates = _render_gate_commands(
        workflow,
        capsule_config_path=CAPSULE_PATH,
        run_id="environment-isolation",
        artifact_dir=tmp_path,
    )

    resolved_services = {
        name: LinuxRuntime._resolve_env(command.env)
        for name, command in services.items()
    }
    assert resolved_services["controller"]["LLAMA_API_KEY"] == "controller-secret"
    assert "CAPX_CONTROLLER_API_KEY" not in resolved_services["controller"]
    assert "CAPX_PROGRAM_API_KEY" not in resolved_services["controller"]
    assert resolved_services["program"]["CAPX_PROGRAM_API_KEY"] == "program-secret"
    assert "CAPX_CONTROLLER_API_KEY" not in resolved_services["program"]
    assert "CAPX_PROGRAM_API_KEY" not in resolved_services["pyroki"]
    assert "CAPX_CONTROLLER_API_KEY" not in resolved_services["pyroki"]
    assert resolved_services["pyroki"]["UNRELATED_PARENT_VALUE"] == "retained"

    resolved_gates = {
        name: LinuxRuntime._resolve_env(command.env) for name, command in gates.items()
    }
    assert resolved_gates["gate01_preflight"]["CAPX_PROGRAM_API_KEY"] == (
        "program-secret"
    )
    assert resolved_gates["gate01_preflight"]["CAPX_CONTROLLER_API_KEY"] == (
        "controller-secret"
    )
    for name in ("gate04_collector", "gate05_guided"):
        assert resolved_gates[name]["CAPX_CONTROLLER_API_KEY"] == "controller-secret"
        assert "CAPX_PROGRAM_API_KEY" not in resolved_gates[name]
    for name in (
        "gate02_seed",
        "gate03_oracle_replay",
        "gate06_trainer",
        "adapter_reload_smoke",
        "gate07_audit",
        "gate07_finalize",
    ):
        assert "CAPX_PROGRAM_API_KEY" not in resolved_gates[name]
        assert "CAPX_CONTROLLER_API_KEY" not in resolved_gates[name]


def test_dry_run_renders_exact_commands_and_does_not_create_outputs(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    result = execute_owned_service_workflow(
        workflow_path=WORKFLOW_PATH,
        profile_path=PROFILE_PATH,
        capsule_config_path=CAPSULE_PATH,
        runtime=runtime,
        dry_run=True,
    )

    assert result.run_id.startswith("base_dynamic_fp32-")
    assert result.output_dir.exists() is False
    assert result.artifact_dir.exists() is False
    assert result.capsule_config_path == result.attempts[0].capsule_config_path
    assert runtime.spawned == []
    assert result.rendered_services["program"].argv == [
        "/root/autodl-tmp/cap-x/.venv/bin/python",
        "-m",
        "capx.rl.capsule.actor_identity",
        "--config",
        str(result.attempts[0].capsule_config_path),
        "--host",
        "127.0.0.1",
        "--port",
        "8101",
    ]
    assert result.rendered_services["controller"].env["CUDA_VISIBLE_DEVICES"] == ""
    assert result.rendered_services["pyroki"].env["JAX_PLATFORMS"] == "cpu"


def test_supervisor_runs_services_then_gates_then_cleans_up(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    result = execute_owned_service_workflow(
        workflow_path=WORKFLOW_PATH,
        profile_path=PROFILE_PATH,
        capsule_config_path=CAPSULE_PATH,
        runtime=runtime,
        dry_run=False,
    )

    assert [item[0] for item in runtime.spawned] == ["controller", "program", "pyroki"]
    assert runtime.gates == [
        "gate01_preflight",
        "gate02_seed",
        "gate03_oracle_replay",
        "gate04_collector",
        "gate05_guided",
        "gate06_trainer",
        "adapter_reload_smoke",
        "gate07_audit",
        "gate07_finalize",
    ]
    assert len(runtime.terminated) == 3
    assert len(runtime.confirmed) == 3
    assert result.audit.gpu_free_vram_mib == 80000
    assert result.run_id == result.controller_seed_run_ids[0]
    assert result.capsule_config_path == result.attempts[-1].capsule_config_path
    assert len(result.controller_seed_run_ids) == 3
    assert runtime.verified_inputs == 1
    attestation_path = result.artifact_dir / "launcher_controller_attestation.json"
    assert attestation_path.is_file()
    attestation = yaml.safe_load(attestation_path.read_text(encoding="utf-8"))
    assert attestation["artifact_type"] == "llama_cpp_b10516_runtime_attestation"
    cleanup_path = result.artifact_dir / "launcher_owned_cleanup.json"
    cleanup = yaml.safe_load(cleanup_path.read_text(encoding="utf-8"))
    assert cleanup["run_id"] == result.run_id
    assert cleanup["cleanup_completed"] is True
    assert [service["name"] for service in cleanup["services"]] == [
        "controller",
        "program",
        "pyroki",
    ]
    assert runtime.events.index("verify-inputs") < runtime.events.index(
        "spawn:controller"
    )
    controller_ready = runtime.events.index("ready:controller")
    first_memory_after_controller = next(
        index
        for index, event in enumerate(runtime.events)
        if index > controller_ready and event.startswith("memory:")
    )
    assert first_memory_after_controller < runtime.events.index("spawn:program")
    for gate in runtime.gates[:-1]:
        gate_index = runtime.events.index(f"gate:{gate}")
        assert runtime.events[gate_index + 1].startswith("memory:")
    assert runtime.events.index("terminate:controller") < runtime.events.index(
        "gate:gate07_finalize"
    )
    assert runtime.events.index("confirm-terminated:controller") < runtime.events.index(
        "gate:gate07_finalize"
    )


def test_noop_terminate_without_probe_confirmation_blocks_cleanup_and_finalizer(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(tmp_path)
    runtime.noop_terminate_names.add("program")

    with pytest.raises(RuntimeError, match="program termination was not confirmed"):
        execute_owned_service_workflow(
            workflow_path=WORKFLOW_PATH,
            profile_path=PROFILE_PATH,
            capsule_config_path=CAPSULE_PATH,
            runtime=runtime,
            dry_run=False,
        )

    attempt = runtime.configured_attempts[0]
    assert len(runtime.confirmed) == 3
    assert not (attempt.artifact_dir / "launcher_owned_cleanup.json").exists()
    assert "gate07_finalize" not in runtime.gates


def test_readiness_or_gate_failure_cleans_up_earlier_owned_processes(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    runtime.ready_fail_after.add("controller")
    with pytest.raises(RuntimeError, match="controller readiness failed"):
        execute_owned_service_workflow(
            workflow_path=WORKFLOW_PATH,
            profile_path=PROFILE_PATH,
            capsule_config_path=CAPSULE_PATH,
            runtime=runtime,
            dry_run=False,
        )
    assert [item[0] for item in runtime.spawned] == ["controller"]
    assert len(runtime.terminated) == 1

    runtime = FakeRuntime(tmp_path / "gate-failure")
    runtime.gate_failure = "gate04_collector"
    with pytest.raises(FakeFailure, match="gate04_collector failed"):
        execute_owned_service_workflow(
            workflow_path=WORKFLOW_PATH,
            profile_path=PROFILE_PATH,
            capsule_config_path=CAPSULE_PATH,
            runtime=runtime,
            dry_run=False,
        )
    assert len(runtime.terminated) == 3
    assert runtime.gates[:4] == [
        "gate01_preflight",
        "gate02_seed",
        "gate03_oracle_replay",
        "gate04_collector",
    ]


def test_retry_profile_materialization_is_exact_and_semantics_preserving(tmp_path: Path) -> None:
    base_profile = load_single_a800_resolved_profile(PROFILE_PATH)
    retry_one = materialize_retry_profile(
        base_profile=base_profile,
        retry_name="vllm_util_026",
        destination=tmp_path / "retry-01.yaml",
    )
    retry_two = materialize_retry_profile(
        base_profile=base_profile,
        retry_name="fixed_microbatch_1",
        destination=tmp_path / "retry-02.yaml",
    )
    retry_three = materialize_retry_profile(
        base_profile=base_profile,
        retry_name="fsdp_base_bf16",
        destination=tmp_path / "retry-03.yaml",
    )

    payload_one = yaml.safe_load(retry_one.read_text(encoding="utf-8"))
    payload_two = yaml.safe_load(retry_two.read_text(encoding="utf-8"))
    payload_three = yaml.safe_load(retry_three.read_text(encoding="utf-8"))

    assert payload_one["actor_rollout_ref"]["rollout"][
        "gpu_memory_utilization"
    ] == pytest.approx(0.26)
    assert payload_two["actor_rollout_ref"]["rollout"][
        "gpu_memory_utilization"
    ] == pytest.approx(0.26)
    assert payload_two["actor_rollout_ref"]["actor"]["use_dynamic_bsz"] is False
    assert payload_two["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] == 1
    assert payload_two["actor_rollout_ref"]["rollout"]["log_prob_use_dynamic_bsz"] is False
    assert payload_two["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"] == 1
    assert payload_two["actor_rollout_ref"]["ref"]["log_prob_use_dynamic_bsz"] is False
    assert payload_two["actor_rollout_ref"]["ref"]["log_prob_micro_batch_size_per_gpu"] == 1
    assert payload_three["actor_rollout_ref"]["rollout"][
        "gpu_memory_utilization"
    ] == pytest.approx(0.26)
    assert payload_three["actor_rollout_ref"]["actor"]["use_dynamic_bsz"] is False
    assert payload_three["actor_rollout_ref"]["ref"]["log_prob_use_dynamic_bsz"] is False
    assert payload_three["actor_rollout_ref"]["actor"]["fsdp_config"]["model_dtype"] == "bf16"
    assert payload_three["actor_rollout_ref"]["ref"]["fsdp_config"]["model_dtype"] == "bf16"
    for payload in (payload_one, payload_two, payload_three):
        assert payload["actor_rollout_ref"]["actor"]["ppo_max_token_len_per_gpu"] == 10240
        assert payload["actor_rollout_ref"]["rollout"]["max_model_len"] == 10240
        assert (
            payload["actor_rollout_ref"]["ref"].get(
                "log_prob_max_token_len_per_gpu", 10240
            )
            == 10240
        )


def test_existing_run_artifact_or_checkpoint_is_refused(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    output_root = tmp_path / "outputs"
    artifact_root = tmp_path / "artifacts"
    checkpoint_root = tmp_path / "checkpoints"
    output_root.mkdir()
    artifact_root.mkdir()
    checkpoint_root.mkdir()

    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    workflow["paths"]["output_root"] = str(output_root)
    workflow["paths"]["artifact_root"] = str(artifact_root)
    workflow["paths"]["checkpoint_root"] = str(checkpoint_root)
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")

    run_id = execute_owned_service_workflow(
        workflow_path=workflow_path,
        profile_path=PROFILE_PATH,
        capsule_config_path=CAPSULE_PATH,
        runtime=runtime,
        dry_run=True,
    ).run_id

    (output_root / run_id).mkdir()
    with pytest.raises(FileExistsError, match="existing run path"):
        execute_owned_service_workflow(
            workflow_path=workflow_path,
            profile_path=PROFILE_PATH,
            capsule_config_path=CAPSULE_PATH,
            runtime=runtime,
            dry_run=False,
        )


def test_gate_templates_parse_with_real_repository_clis(tmp_path: Path) -> None:
    workflow = load_owned_services_workflow(WORKFLOW_PATH)
    commands = _render_gate_commands(
        workflow,
        capsule_config_path=CAPSULE_PATH,
        run_id="capsule-a800-parser-check",
        artifact_dir=tmp_path,
    )

    server_preflight.build_parser().parse_args(commands["gate01_preflight"].argv[3:])
    for gate_name in (
        "gate02_seed",
        "gate03_oracle_replay",
        "gate04_collector",
        "gate05_guided",
        "gate06_trainer",
    ):
        parsed = server_adapter.build_parser().parse_args(commands[gate_name].argv[3:])
        assert parsed.run_id == "capsule-a800-parser-check"
    analyze_artifacts.build_parser().parse_args(commands["gate07_audit"].argv[3:])
    analyze_artifacts.build_parser().parse_args(commands["gate07_finalize"].argv[3:])
    reload_argv = commands["adapter_reload_smoke"].argv
    assert reload_argv[1:3] == ["-m", "scripts.capsule_rl.adapter_reload_smoke"]
    assert "--gate6-artifact" in reload_argv
    adapter_reload_smoke.build_parser().parse_args(reload_argv[3:])


def test_oom_retries_are_cumulative_new_profiles_and_retain_evidence(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(tmp_path)
    runtime.oom_failures_remaining = 3

    result = execute_owned_service_workflow(
        workflow_path=WORKFLOW_PATH,
        profile_path=PROFILE_PATH,
        capsule_config_path=CAPSULE_PATH,
        runtime=runtime,
        dry_run=False,
    )

    assert [attempt.retry_name for attempt in result.attempts] == [
        "base_dynamic_fp32",
        "vllm_util_026",
        "fixed_microbatch_1",
        "fsdp_base_bf16",
    ]
    assert len({attempt.run_id for attempt in result.attempts}) == 4
    assert len({attempt.profile_sha256 for attempt in result.attempts}) == 4
    for failed in result.attempts[:3]:
        assert (failed.artifact_dir / "launcher_failure.json").is_file()
    for attempt in result.attempts:
        assert hashlib.sha256(attempt.profile_path.read_bytes()).hexdigest() == (
            attempt.profile_sha256
        )
    final = yaml.safe_load(result.attempts[-1].profile_path.read_text(encoding="utf-8"))
    root = final["actor_rollout_ref"]
    assert root["rollout"]["gpu_memory_utilization"] == pytest.approx(0.26)
    assert root["actor"]["use_dynamic_bsz"] is False
    assert root["ref"]["log_prob_use_dynamic_bsz"] is False
    assert root["actor"]["fsdp_config"]["model_dtype"] == "bf16"
    assert root["rollout"]["max_model_len"] == 10240


def test_gate1_oom_stops_without_changing_profile(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    runtime.oom_failures_remaining = 1
    runtime.oom_failure_gate = "gate01_preflight"

    with pytest.raises(GateCommandError) as failure:
        execute_owned_service_workflow(
            workflow_path=WORKFLOW_PATH,
            profile_path=PROFILE_PATH,
            capsule_config_path=CAPSULE_PATH,
            runtime=runtime,
            dry_run=False,
        )

    assert failure.value.gate_name == "gate01_preflight"
    assert [attempt.retry_name for attempt in runtime.configured_attempts] == [
        "base_dynamic_fp32"
    ]


def test_gate2_vllm_oom_advances_to_utilization_026(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    runtime.oom_failures_remaining = 1
    runtime.oom_failure_gate = "gate02_seed"

    result = execute_owned_service_workflow(
        workflow_path=WORKFLOW_PATH,
        profile_path=PROFILE_PATH,
        capsule_config_path=CAPSULE_PATH,
        runtime=runtime,
        dry_run=False,
    )

    assert [attempt.retry_name for attempt in result.attempts] == [
        "base_dynamic_fp32",
        "vllm_util_026",
    ]
    resolved = yaml.safe_load(result.attempts[-1].profile_path.read_text(encoding="utf-8"))
    assert resolved["actor_rollout_ref"]["rollout"][
        "gpu_memory_utilization"
    ] == pytest.approx(0.26)


def test_guided_randomness_uses_at_most_three_new_run_ids(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    runtime.guided_failures_remaining = 2

    result = execute_owned_service_workflow(
        workflow_path=WORKFLOW_PATH,
        profile_path=PROFILE_PATH,
        capsule_config_path=CAPSULE_PATH,
        runtime=runtime,
        dry_run=False,
    )

    assert len(result.attempts) == 3
    assert [attempt.run_id for attempt in result.attempts] == result.controller_seed_run_ids
    assert result.attempts[0].run_id.endswith("controller-seed-1")
    assert result.attempts[1].run_id.endswith("controller-seed-2")
    assert result.attempts[2].run_id.endswith("controller-seed-3")


def test_cleanup_preserves_primary_failure_and_attempts_every_owned_process(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(tmp_path)
    runtime.gate_failure = "gate04_collector"
    runtime.terminate_failure_name = "program"

    with pytest.raises(FakeFailure, match="gate04_collector failed"):
        execute_owned_service_workflow(
            workflow_path=WORKFLOW_PATH,
            profile_path=PROFILE_PATH,
            capsule_config_path=CAPSULE_PATH,
            runtime=runtime,
            dry_run=False,
        )

    assert len(runtime.terminated) == 3
    attempt = runtime.configured_attempts[0]
    assert not (attempt.artifact_dir / "launcher_owned_cleanup.json").exists()


def test_failure_retains_audit_owned_pid_and_memory_evidence(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    runtime.gate_failure = "gate04_collector"

    with pytest.raises(FakeFailure, match="gate04_collector failed"):
        execute_owned_service_workflow(
            workflow_path=WORKFLOW_PATH,
            profile_path=PROFILE_PATH,
            capsule_config_path=CAPSULE_PATH,
            runtime=runtime,
            dry_run=False,
        )

    attempt = runtime.configured_attempts[0]
    audit_path = attempt.artifact_dir / "launcher_initial_audit.json"
    assert audit_path.is_file()
    audit = yaml.safe_load(audit_path.read_text(encoding="utf-8"))
    assert audit["run_id"] == attempt.run_id
    assert audit["snapshot"]["gpu_name"] == "NVIDIA A800 80GB PCIe"

    process_records = sorted(attempt.artifact_dir.glob("launcher_owned_process_*.json"))
    assert len(process_records) == 3
    recorded_identities = []
    for record_path in process_records:
        raw = record_path.read_text(encoding="utf-8")
        record = yaml.safe_load(raw)
        recorded_identities.append((record["pid"], record["starttime_ticks"]))
        assert len(record["argv_sha256"]) == 64
        assert "env_values" not in record
        assert "{env:" not in raw
    assert sorted(recorded_identities) == sorted(runtime.terminated)

    memory_records = sorted(attempt.artifact_dir.glob("launcher_memory_*.json"))
    assert len(memory_records) == 4
    assert yaml.safe_load(memory_records[0].read_text(encoding="utf-8"))["stage"] == (
        "post-controller"
    )
    continuous_path = attempt.artifact_dir / "launcher_continuous_memory.json"
    continuous = yaml.safe_load(continuous_path.read_text(encoding="utf-8"))
    assert continuous["passed"] is True
    assert continuous["minimum_available_mib"] == 20000
    assert continuous["sample_count"] >= 2
    assert continuous["maximum_sample_gap_ms"] <= continuous["maximum_allowed_gap_ms"]


def test_continuous_memory_breach_stops_services_and_is_immutable(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(tmp_path)
    runtime.continuous_mem_available_value = 12000

    with pytest.raises(
        OwnedServicesConfigError,
        match=r"runtime minimum MemAvailable 12000MiB is below 12288MiB",
    ):
        execute_owned_service_workflow(
            workflow_path=WORKFLOW_PATH,
            profile_path=PROFILE_PATH,
            capsule_config_path=CAPSULE_PATH,
            runtime=runtime,
            dry_run=False,
        )

    attempt = runtime.configured_attempts[0]
    evidence = yaml.safe_load(
        (attempt.artifact_dir / "launcher_continuous_memory.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["passed"] is False
    assert evidence["minimum_available_mib"] == 12000
    assert evidence["required_mib"] == 12288
    assert [item[0] for item in runtime.spawned] == ["controller"]
    assert len(runtime.terminated) == 1
    assert (attempt.artifact_dir / "launcher_failure.json").is_file()


def test_gate7_memory_breach_prevents_runtime_verification_finalizer(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(tmp_path)
    runtime.breach_during_gate7 = True

    with pytest.raises(
        OwnedServicesConfigError,
        match=r"runtime minimum MemAvailable 12000MiB is below 12288MiB",
    ):
        execute_owned_service_workflow(
            workflow_path=WORKFLOW_PATH,
            profile_path=PROFILE_PATH,
            capsule_config_path=CAPSULE_PATH,
            runtime=runtime,
            dry_run=False,
        )

    attempt = runtime.configured_attempts[0]
    assert runtime.gates[-1] == "gate07_audit"
    assert "gate07_finalize" not in runtime.gates
    assert not (attempt.artifact_dir / "gate07_audit.json").exists()
    assert (attempt.artifact_dir / "launcher_failure.json").is_file()


def test_post_controller_90gib_check_precedes_continuous_monitor(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(tmp_path)
    runtime.mem_available_values[0] = 90000

    with pytest.raises(
        OwnedServicesConfigError,
        match=r"post-controller MemAvailable 90000MiB is below 92160MiB",
    ):
        execute_owned_service_workflow(
            workflow_path=WORKFLOW_PATH,
            profile_path=PROFILE_PATH,
            capsule_config_path=CAPSULE_PATH,
            runtime=runtime,
            dry_run=False,
        )

    attempt = runtime.configured_attempts[0]
    point = yaml.safe_load(
        (attempt.artifact_dir / "launcher_memory_00_post-controller.json").read_text(
            encoding="utf-8"
        )
    )
    assert point["required_mib"] == 92160
    assert point["passed"] is False
    assert not (attempt.artifact_dir / "launcher_continuous_memory.json").exists()
    assert [item[0] for item in runtime.spawned] == ["controller"]
    assert len(runtime.terminated) == 1


def test_linux_spawn_closes_parent_service_log_handle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class PopenStub:
        pid = 4312

    def popen_stub(argv: list[str], **kwargs: object) -> PopenStub:
        captured["argv"] = argv
        captured["stdout"] = kwargs["stdout"]
        return PopenStub()

    runtime = LinuxRuntime()
    runtime._attempt = types.SimpleNamespace(artifact_dir=tmp_path)
    runtime._workflow = {"runtime": {"repo_root": str(tmp_path)}}
    monkeypatch.setattr(
        "scripts.capsule_rl.launch_owned_services.subprocess.Popen", popen_stub
    )
    monkeypatch.setattr(runtime, "_starttime", lambda _pid: 9001)

    identity = runtime.spawn(
        "controller",
        ["llama-server", "--version"],
        {"CUDA_VISIBLE_DEVICES": ""},
    )

    assert identity == ProcessIdentity("controller", 4312, 9001)
    assert captured["stdout"].closed is True


def test_linux_spawn_sanitizes_loader_environment_only_for_controller(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_environments: dict[str, dict[str, str]] = {}
    next_pid = iter((4320, 4321))

    class PopenStub:
        def __init__(self, pid: int) -> None:
            self.pid = pid

    def popen_stub(argv: list[str], **kwargs: object) -> PopenStub:
        captured_environments[argv[0]] = kwargs["env"]  # type: ignore[assignment]
        return PopenStub(next(next_pid))

    for name in (
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "LD_AUDIT",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
    ):
        monkeypatch.setenv(name, f"hostile-{name}")
    monkeypatch.setenv("CAPX_CONTROLLER_API_KEY", "controller-secret")
    monkeypatch.setenv("CAPX_PROGRAM_API_KEY", "program-secret")
    monkeypatch.setenv("UNRELATED_RUNTIME_VALUE", "retained")
    runtime = LinuxRuntime()
    runtime._attempt = types.SimpleNamespace(artifact_dir=tmp_path)
    runtime._workflow = {"runtime": {"repo_root": str(tmp_path)}}
    monkeypatch.setattr(
        "scripts.capsule_rl.launch_owned_services.subprocess.Popen", popen_stub
    )
    monkeypatch.setattr(runtime, "_starttime", lambda pid: pid * 10)

    runtime.spawn(
        "controller",
        ["controller-bin"],
        {"LLAMA_API_KEY": "{env:CAPX_CONTROLLER_API_KEY}"},
    )
    runtime.spawn(
        "program",
        ["program-bin"],
        {"CAPX_PROGRAM_API_KEY": "{env:CAPX_PROGRAM_API_KEY}"},
    )

    controller_env = captured_environments["controller-bin"]
    for name in (
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "LD_AUDIT",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
    ):
        assert name not in controller_env
        assert name in captured_environments["program-bin"]
    assert controller_env["LLAMA_API_KEY"] == "controller-secret"
    assert "CAPX_CONTROLLER_API_KEY" not in controller_env
    assert "CAPX_PROGRAM_API_KEY" not in controller_env
    assert captured_environments["program-bin"]["CAPX_PROGRAM_API_KEY"] == (
        "program-secret"
    )
    assert "CAPX_CONTROLLER_API_KEY" not in captured_environments["program-bin"]
    assert controller_env["UNRELATED_RUNTIME_VALUE"] == "retained"


def test_linux_controller_spawn_verifies_proc_executable_against_attestation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class PopenStub:
        pid = 4313

        def __init__(self) -> None:
            self.terminated = False
            self.waited = False

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, *, timeout: int) -> None:
            assert timeout == 10
            self.waited = True

    process = PopenStub()
    runtime = LinuxRuntime()
    runtime._attempt = types.SimpleNamespace(artifact_dir=tmp_path)
    runtime._workflow = {"runtime": {"repo_root": str(tmp_path)}}
    runtime._controller_attestation = {"binary_sha256": "a" * 64}
    monkeypatch.setattr(
        "scripts.capsule_rl.launch_owned_services.subprocess.Popen",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(runtime, "_starttime", lambda _pid: 9002)
    monkeypatch.setattr(
        "scripts.capsule_rl.launch_owned_services._file_content_sha256",
        lambda _path: "b" * 64,
    )

    with pytest.raises(
        OwnedServicesConfigError, match="running Controller executable"
    ):
        runtime.spawn("controller", ["llama-server"], {"CUDA_VISIBLE_DEVICES": ""})

    assert process.terminated is True
    assert process.waited is True


def test_linux_termination_confirmation_requires_pid_identity_and_empty_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = LinuxRuntime()
    identity = ProcessIdentity("controller", 4314, 9003)

    monkeypatch.setattr(runtime, "_proc_identity", lambda _pid: None)
    monkeypatch.setattr(
        "scripts.capsule_rl.launch_owned_services.os.killpg",
        lambda _pgid, _signal: (_ for _ in ()).throw(ProcessLookupError()),
    )
    assert runtime.confirm_terminated(identity) is True

    monkeypatch.setattr(runtime, "_proc_identity", lambda _pid: (9003, 4314))
    assert runtime.confirm_terminated(identity) is False

    monkeypatch.setattr(runtime, "_proc_identity", lambda _pid: None)
    monkeypatch.setattr(
        "scripts.capsule_rl.launch_owned_services.os.killpg",
        lambda _pgid, _signal: None,
    )
    assert runtime.confirm_terminated(identity) is False


def test_linux_termination_confirmation_fails_closed_on_proc_read_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = LinuxRuntime()
    identity = ProcessIdentity("controller", 4315, 9004)
    monkeypatch.setattr(
        "scripts.capsule_rl.launch_owned_services.Path.read_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )

    with pytest.raises(RuntimeError, match="cannot read /proc/4315/stat"):
        runtime.confirm_terminated(identity)


def test_cli_uses_concrete_runtime_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    monkeypatch.setattr(
        "scripts.capsule_rl.launch_owned_services.LinuxRuntime",
        lambda: runtime,
    )

    assert main(
        [
            "--workflow-config",
            str(WORKFLOW_PATH),
            "--profile-config",
            str(PROFILE_PATH),
            "--capsule-config",
            str(CAPSULE_PATH),
            "--dry-run",
        ]
    ) == 0
    assert runtime.events[:2] == ["audit", "verify-inputs"]
