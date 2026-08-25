from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import capx.rl.capsule.server_factory as server_factory
from capx.rl.capsule.group import CandidateCollectionError
from capx.rl.capsule.schema import source_sha256
from capx.rl.capsule.server_factory import (
    CapsuleServerRuntime,
    ServerFactoryError,
    VeRLGroupEncoder,
    VeRLProgramGenerator,
    YamlEnvironmentFactory,
    create_trainer,
    load_task_instances,
    resolve_task_instances,
)
from capx.rl.capsule.trainer import DiscardedGroupRecord


CONFIG_PATH = (
    Path(__file__).parents[1]
    / "env_configs"
    / "cube_stack"
    / "capsule_rl"
    / "franka_robosuite_cube_stack_capsule_critique_grpo.yaml"
)


def _config(tmp_path: Path, dataset_path: Path) -> dict:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["runtime"]["project_root"] = str(tmp_path)
    config["runtime"]["dataset_path"] = str(dataset_path)
    config["runtime"]["output_dir"] = str(tmp_path / "outputs")
    return config


def _task_record() -> dict:
    return {
        "schema_version": 1,
        "task_id": "cube-stack",
        "environment_seed": 5,
        "prompt": "stack the red cube on the green cube",
        "environment": "robosuite_cube_stack",
        "api": "franka_control_privileged",
        "privilege": "privileged",
        "initial_state_sha256": source_sha256("seed-five-state"),
        "metadata": {"split": "train"},
    }


def _verl_worker_config(
    *,
    lora_rank: int = 0,
    lora_alpha: int | None = None,
    target_modules: str | list[str] = "all-linear",
    rollout_name: str = "vllm",
    load_format: str = "safetensors",
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            model=SimpleNamespace(
                lora_rank=lora_rank,
                lora_alpha=(32 if lora_rank > 0 else 0)
                if lora_alpha is None
                else lora_alpha,
                target_modules=target_modules,
            ),
            actor=SimpleNamespace(strategy="fsdp"),
            rollout=SimpleNamespace(
                name=rollout_name,
                load_format=load_format,
                tensor_model_parallel_size=tensor_parallel_size,
                data_parallel_size=data_parallel_size,
                pipeline_model_parallel_size=pipeline_parallel_size,
            ),
        ),
        trainer=SimpleNamespace(n_gpus_per_node=1, nnodes=1),
    )


class _InitProbe:
    def __init__(self) -> None:
        self.init_calls = 0

    def init_model(self) -> None:
        self.init_calls += 1


def test_lora_worker_topology_reuses_one_actor_for_frozen_base_reference() -> None:
    topology = server_factory._resolve_verl_worker_topology(
        _verl_worker_config(lora_rank=16, lora_alpha=32)
    )
    actor = _InitProbe()

    actor_wg, ref_wg = server_factory._initialize_verl_worker_groups(
        {"actor_rollout": actor}, topology
    )

    assert topology.worker_roles == ("actor_rollout",)
    assert topology.reference_policy_mode == "actor_base_adapter_disabled"
    assert topology.lora_rank == 16
    assert topology.lora_alpha == 32
    assert topology.lora_target_modules == ("all-linear",)
    assert actor_wg is actor
    assert ref_wg is actor
    assert actor.init_calls == 1


def test_non_lora_worker_topology_keeps_distinct_reference_worker() -> None:
    topology = server_factory._resolve_verl_worker_topology(_verl_worker_config())
    actor = _InitProbe()
    reference = _InitProbe()

    actor_wg, ref_wg = server_factory._initialize_verl_worker_groups(
        {"actor_rollout": actor, "ref": reference}, topology
    )

    assert topology.worker_roles == ("actor_rollout", "ref")
    assert topology.reference_policy_mode == "standalone"
    assert actor_wg is actor
    assert ref_wg is reference
    assert actor.init_calls == 1
    assert reference.init_calls == 1


def test_lora_reference_source_contract_requires_adapter_disable_path(tmp_path: Path) -> None:
    worker_source = tmp_path / "verl" / "workers" / "fsdp_workers.py"
    worker_source.parent.mkdir(parents=True)
    worker_source.write_text(
        "from contextlib import nullcontext\n"
        "def compute_log_prob(self, data):\n"
        "    is_lora = data.meta_info.pop('is_lora', False)\n"
        "    adapter_ctx = (self.actor.actor_module.disable_adapter()\n"
        "                   if is_lora else nullcontext())\n"
        "    with adapter_ctx:\n"
        "        return None\n",
        encoding="utf-8",
    )

    server_factory._verify_lora_reference_source_contract(tmp_path)

    worker_source.write_text(
        "from contextlib import nullcontext\n"
        "def compute_log_prob(self, data):\n"
        "    is_lora = data.meta_info.pop('is_lora', False)\n"
        "    adapter_ctx = self.actor.actor_module.disable_adapter()\n"
        "    with adapter_ctx:\n"
        "        return is_lora\n",
        encoding="utf-8",
    )
    with pytest.raises(ServerFactoryError, match="disable_adapter"):
        server_factory._verify_lora_reference_source_contract(tmp_path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"lora_rank": True}, "lora_rank"),
        ({"lora_rank": -1}, "lora_rank"),
        ({"lora_rank": 16, "lora_alpha": 0}, "lora_alpha"),
        ({"lora_rank": 16, "target_modules": []}, "target_modules"),
        ({"lora_rank": 16, "rollout_name": "hf"}, "vLLM"),
        ({"lora_rank": 16, "load_format": "auto"}, "safetensors"),
        ({"lora_rank": 16, "tensor_parallel_size": 2}, "parallel"),
        ({"lora_rank": 16, "data_parallel_size": 2}, "parallel"),
        ({"lora_rank": 16, "pipeline_parallel_size": 2}, "parallel"),
    ],
)
def test_lora_worker_topology_fails_fast_on_unsupported_config(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ServerFactoryError, match=message):
        server_factory._resolve_verl_worker_topology(_verl_worker_config(**overrides))


def test_repository_config_names_concrete_project_factory() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["trainer_factory"] == "capx.rl.capsule.server_factory:create_trainer"
    assert config["runtime"]["verl_resolved_config_path"].endswith(".yaml")


def test_create_trainer_is_side_effect_free_until_fit(tmp_path: Path) -> None:
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(json.dumps(_task_record()) + "\n", encoding="utf-8")

    trainer = create_trainer(_config(tmp_path, dataset))

    assert isinstance(trainer, CapsuleServerRuntime)
    assert callable(trainer.fit)


def test_load_task_instances_round_trips_seed_resolved_jsonl(tmp_path: Path) -> None:
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(json.dumps(_task_record()) + "\n", encoding="utf-8")

    tasks = load_task_instances(_config(tmp_path, dataset))

    assert len(tasks) == 1
    assert tasks[0].environment_seed == 5
    assert tasks[0].initial_state_sha256 == _task_record()["initial_state_sha256"]


def test_load_task_instances_rejects_unresolved_initial_state(tmp_path: Path) -> None:
    record = _task_record()
    del record["initial_state_sha256"]
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ServerFactoryError, match="seed-resolution"):
        load_task_instances(_config(tmp_path, dataset))


def test_server_resolver_binds_missing_hash_from_real_reset_contract(tmp_path: Path) -> None:
    record = _task_record()
    del record["initial_state_sha256"]
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")
    events: list[object] = []

    class _Environment:
        def reset(self, *, seed, options):
            events.append((seed, options))
            return {}, {"initial_state_sha256": source_sha256(f"state-{seed}")}

        def close(self):
            events.append("close")

    tasks = resolve_task_instances(
        _config(tmp_path, dataset), environment_factory=lambda _task: _Environment()
    )

    assert tasks[0].initial_state_sha256 == source_sha256("state-5")
    assert events == [(5, {"capsule_task_state_resolution": True}), "close"]


def test_server_resolver_overrides_untrusted_existing_hash_with_real_reset(
    tmp_path: Path,
) -> None:
    record = _task_record()
    record["initial_state_sha256"] = "f" * 64
    dataset = tmp_path / "tasks-with-fake-hash.jsonl"
    dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")
    events: list[object] = []

    class _Environment:
        def reset(self, *, seed, options):
            events.append((seed, options))
            return {}, {"initial_state_sha256": source_sha256(f"real-state-{seed}")}

        def close(self):
            events.append("close")

    tasks = resolve_task_instances(
        _config(tmp_path, dataset), environment_factory=lambda _task: _Environment()
    )

    assert tasks[0].initial_state_sha256 == source_sha256("real-state-5")
    assert events == [(5, {"capsule_task_state_resolution": True}), "close"]


class _Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        assert messages[-1]["content"] == "task prompt"
        return [11, 12, 13]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if text == "print('ok')":
            return [21, 22]
        return [ord(character) for character in text]

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return "print('ok')" if token_ids == [21, 22] else ""


class _DataProto:
    def __init__(self, *, tensors, non_tensors=None, meta_info=None):
        self.batch = tensors
        self.non_tensor_batch = non_tensors or {}
        self.meta_info = meta_info or {}


class _Actor:
    def __init__(self) -> None:
        self.request = None

    def generate_sequences(self, request):
        self.request = request
        return _DataProto(
            tensors={
                "responses": torch.tensor([[21, 22, 2, 0]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool),
            }
        )


def test_verl_program_generator_builds_typed_candidate_without_http() -> None:
    actor = _Actor()
    generator = VeRLProgramGenerator(
        actor_rollout_wg=actor,
        tokenizer=_Tokenizer(),
        data_proto_factory=_DataProto,
        prompt_token_limit=8,
        response_token_limit=4,
        system_prompt="system",
    )

    candidate = generator.generate("task prompt", "sample-0")

    assert candidate.program_sample_id == "sample-0"
    assert candidate.source == "print('ok')"
    assert candidate.finish_reason == "stop"
    assert candidate.truncated is False
    assert actor.request.batch["input_ids"].tolist() == [[11, 12, 13]]
    assert actor.request.non_tensor_batch["raw_prompt_ids"].dtype == object
    assert actor.request.meta_info == {"do_sample": True, "validate": False}


def test_program_generator_rejects_decode_retokenize_identity_drift() -> None:
    class _NonRoundTripTokenizer(_Tokenizer):
        def encode(self, text, *, add_special_tokens):
            token_ids = super().encode(text, add_special_tokens=add_special_tokens)
            return [31, 32] if text == "print('ok')" else token_ids

    generator = VeRLProgramGenerator(
        actor_rollout_wg=_Actor(),
        tokenizer=_NonRoundTripTokenizer(),
        data_proto_factory=_DataProto,
        prompt_token_limit=8,
        response_token_limit=4,
        system_prompt="system",
    )

    with pytest.raises(CandidateCollectionError, match="round-trip"):
        generator.generate("task prompt", "sample-0")


def test_program_generator_rejects_response_without_eos() -> None:
    class _NoEosActor(_Actor):
        def generate_sequences(self, request):
            self.request = request
            return _DataProto(
                tensors={
                    "responses": torch.tensor([[21, 22, 0, 0]], dtype=torch.long),
                    "attention_mask": torch.tensor(
                        [[1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool
                    ),
                }
            )

    generator = VeRLProgramGenerator(
        actor_rollout_wg=_NoEosActor(),
        tokenizer=_Tokenizer(),
        data_proto_factory=_DataProto,
        prompt_token_limit=8,
        response_token_limit=4,
        system_prompt="system",
    )

    with pytest.raises(CandidateCollectionError, match="EOS"):
        generator.generate("task prompt", "sample-0")


def test_program_generator_has_distinct_prompt_and_raw_response_counters() -> None:
    generator = VeRLProgramGenerator(
        actor_rollout_wg=_Actor(),
        tokenizer=_Tokenizer(),
        data_proto_factory=_DataProto,
        prompt_token_limit=8,
        response_token_limit=4,
        system_prompt="system",
    )

    assert generator.count_prompt_tokens("task prompt") == 3
    assert generator.count_raw_response_tokens("print('ok')") == 2


def test_training_encoder_uses_same_chat_template_as_rollout() -> None:
    encoder = VeRLGroupEncoder(
        tokenizer=_Tokenizer(),
        data_proto_factory=_DataProto,
        prompt_token_limit=5,
        response_token_limit=3,
        system_prompt="system",
    )

    batch = encoder.encode(("task prompt",) * 8, ("x",) * 8)

    assert batch.batch["prompts"][0].tolist() == [0, 0, 11, 12, 13]
    assert batch.batch["responses"][0].tolist() == [ord("x"), 2, 0]
    assert batch.batch["response_mask"].dtype == torch.bool


def test_training_encoder_preserves_generator_rollout_action_token_identity() -> None:
    actor = _Actor()
    tokenizer = _Tokenizer()
    generator = VeRLProgramGenerator(
        actor_rollout_wg=actor,
        tokenizer=tokenizer,
        data_proto_factory=_DataProto,
        prompt_token_limit=8,
        response_token_limit=4,
        system_prompt="system",
    )
    candidate = generator.generate("task prompt", "sample-0")
    encoder = VeRLGroupEncoder(
        tokenizer=tokenizer,
        data_proto_factory=_DataProto,
        prompt_token_limit=8,
        response_token_limit=4,
        system_prompt="system",
    )

    batch = encoder.encode(("task prompt",), (candidate.source,))

    assert batch.batch["responses"][0].tolist() == [21, 22, 2, 0]


def test_yaml_environment_factory_is_lazy_and_pickle_safe(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "environment.yaml"
    config_path.write_text("env: {_target_: fake.Environment}\n", encoding="utf-8")
    marker = object()
    calls: list[object] = []
    monkeypatch.setattr(
        "capx.rl.capsule.server_factory.DictLoader.load",
        lambda path: {"env": {"_target_": "fake.Environment", "path": path}},
    )
    monkeypatch.setattr(
        "capx.rl.capsule.server_factory.instantiate",
        lambda value: calls.append(value) or marker,
    )
    factory = YamlEnvironmentFactory(str(config_path))

    assert calls == []
    task = SimpleNamespace()
    assert factory(task) is marker
    assert calls[0]["path"] == str(config_path)


def test_program_generator_rejects_prompt_overflow() -> None:
    generator = VeRLProgramGenerator(
        actor_rollout_wg=SimpleNamespace(),
        tokenizer=_Tokenizer(),
        data_proto_factory=_DataProto,
        prompt_token_limit=2,
        response_token_limit=4,
        system_prompt="system",
    )

    with pytest.raises(CandidateCollectionError, match="not truncated"):
        generator.generate("task prompt", "sample-0")


def test_server_runtime_binds_components_without_starting_external_services(
    monkeypatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(json.dumps(_task_record()) + "\n", encoding="utf-8")
    config = _config(tmp_path, dataset)
    task = load_task_instances(config)[0]
    events: list[str] = []

    class _Workers:
        actor_rollout_wg = SimpleNamespace()
        ref_policy_wg = SimpleNamespace()
        tokenizer = _Tokenizer()
        data_proto_factory = _DataProto

        def __init__(self):
            self._optimizer_steps = iter((0, 1))

        def optimizer_step(self):
            return next(self._optimizer_steps)

        def verl_provenance(self):
            return {
                "source_path": str((tmp_path / "verl").resolve()),
                "expected_sha": "e" * 40,
                "actual_sha": "e" * 40,
                "clean": True,
                "worker_count": 1,
                "worker_ranks": [0],
                "worker_module_paths": ["/pinned/verl/__init__.py"],
            }

        def save_checkpoint(self, path, step):
            events.append(f"checkpoint:{step}")
            path.mkdir(parents=True)
            (path / "state.bin").write_bytes(b"checkpoint")

        def close(self):
            events.append("workers:close")

    class _Evaluator:
        def __init__(self, backend):
            events.append(f"evaluator:{type(backend).__name__}")

        def close(self):
            events.append("evaluator:close")

    class _Trainer:
        def __init__(self, **kwargs):
            assert kwargs["ref_policy_wg"] is not None
            events.append("trainer:init")

        def fit(self, tasks):
            assert len(tasks) == 1
            scheduled_task = tasks[0]
            assert scheduled_task.task_id == task.task_id
            assert scheduled_task.environment_seed == task.environment_seed
            assert scheduled_task.initial_state_sha256 == task.initial_state_sha256
            assert scheduled_task.metadata["split"] == "train"
            assert scheduled_task.metadata["capsule_collection_id"] == (
                "epoch-00000000:task-00000000"
            )
            events.append("trainer:fit")
            return (SimpleNamespace(skipped_actor_update=False),)

    monkeypatch.setattr(
        "capx.rl.capsule.server_factory.PersistentProcessReplayBackend",
        lambda environment_factory: SimpleNamespace(environment_factory=environment_factory),
    )
    monkeypatch.setattr("capx.rl.capsule.server_factory.CleanReplayEvaluator", _Evaluator)
    monkeypatch.setattr(
        "capx.rl.capsule.server_factory.CandidateCleanReplayAdapter",
        lambda evaluator: SimpleNamespace(evaluator=evaluator),
    )
    monkeypatch.setattr(
        "capx.rl.capsule.server_factory.CapsuleCritiqueRayTrainer", _Trainer
    )

    result = CapsuleServerRuntime(
        config,
        worker_starter=lambda _config: _Workers(),
        task_loader=lambda _config: (task,),
    ).fit()

    assert result["step_count"] == 1
    assert result["actor_updates"] == 1
    assert result["optimizer_step_delta"] == 1
    assert result["status"] == "completed"
    assert result["verl_provenance_before"] == result["verl_provenance_after"]
    assert events[-3:] == ["checkpoint:1", "evaluator:close", "workers:close"]


def test_server_runtime_persists_full_discard_audit_and_refuses_empty_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(json.dumps(_task_record()) + "\n", encoding="utf-8")
    config = _config(tmp_path, dataset)
    task = load_task_instances(config)[0]
    events: list[str] = []

    class _Workers:
        actor_rollout_wg = SimpleNamespace()
        ref_policy_wg = SimpleNamespace()
        tokenizer = _Tokenizer()
        data_proto_factory = _DataProto
        total_epochs = 1

        def optimizer_step(self):
            return 0

        def verl_provenance(self):
            return {
                "source_path": str((tmp_path / "verl").resolve()),
                "expected_sha": "e" * 40,
                "actual_sha": "e" * 40,
                "clean": True,
                "worker_count": 1,
                "worker_ranks": [0],
                "worker_module_paths": ["/pinned/verl/__init__.py"],
            }

        def save_checkpoint(self, _path, _step):
            raise AssertionError("an all-discard run must not write a checkpoint")

        def close(self):
            events.append("workers:close")

    class _Evaluator:
        def __init__(self, _backend):
            pass

        def close(self):
            events.append("evaluator:close")

    class _Trainer:
        def __init__(self, **_kwargs):
            self.discarded_groups = (
                DiscardedGroupRecord(
                    task_index=0,
                    task_id=task.task_id,
                    environment_seed=task.environment_seed,
                    initial_state_sha256=task.initial_state_sha256,
                    reason="unknown_replay_reward",
                    message="typed evaluator returned infra_error",
                ),
            )

        def fit(self, _tasks):
            return ()

    monkeypatch.setattr(
        "capx.rl.capsule.server_factory.PersistentProcessReplayBackend",
        lambda environment_factory: SimpleNamespace(environment_factory=environment_factory),
    )
    monkeypatch.setattr("capx.rl.capsule.server_factory.CleanReplayEvaluator", _Evaluator)
    monkeypatch.setattr(
        "capx.rl.capsule.server_factory.CandidateCleanReplayAdapter",
        lambda evaluator: SimpleNamespace(evaluator=evaluator),
    )
    monkeypatch.setattr("capx.rl.capsule.server_factory.CapsuleCritiqueRayTrainer", _Trainer)

    runtime = CapsuleServerRuntime(
        config,
        worker_starter=lambda _config: _Workers(),
        task_loader=lambda _config: (task,),
    )
    with pytest.raises(ServerFactoryError, match="all scheduled groups were discarded"):
        runtime.fit()

    audit_paths = list((tmp_path / "outputs" / "discarded_groups").glob("*.json"))
    assert len(audit_paths) == 1
    audit_path = audit_paths[0]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["discarded_groups"][0]["reason"] == "unknown_replay_reward"
    assert audit["discarded_groups"][0]["message"].endswith("infra_error")
    assert events == ["evaluator:close", "workers:close"]

    def forbidden_worker_starter(_config):
        raise AssertionError("an existing run audit must be rejected before workers start")

    retry = CapsuleServerRuntime(
        config,
        worker_starter=forbidden_worker_starter,
        task_loader=lambda _config: (task,),
    )
    with pytest.raises(FileExistsError, match="discard audit already exists"):
        retry.fit()
