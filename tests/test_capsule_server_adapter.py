from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from capx.rl.capsule.actor_identity import build_actor_identity
from capx.rl.capsule.checkpoint import AtomicCheckpointClaim
from capx.rl.capsule.group import RepairAttempt, deterministic_group_uid
from capx.rl.capsule.repair import BaseUnitSpan, RepairDraft
from capx.rl.capsule.schema import (
    LearningGroupV1,
    LearningMemberV1,
    ProgramReplayResultV1,
    ReplayOutcome,
    TaskInstanceV1,
)
from scripts.capsule_rl import server_adapter
from scripts.capsule_rl.common import artifact_file_sha256


class _FakeRuntime:
    def __init__(self, evidence=None) -> None:
        self.evidence = evidence or {}
        self.calls: list[tuple] = []

    def seed(self, seeds, *, run_id):
        self.calls.append(("seed", seeds, run_id))
        return self.evidence

    def oracle(self, seed, replay_count, *, run_id):
        self.calls.append(("oracle", seed, replay_count, run_id))
        return self.evidence

    def collector(self, p0_count, trajectories_per_p0, max_turns, *, run_id):
        self.calls.append(
            ("collector", p0_count, trajectories_per_p0, max_turns, run_id)
        )
        return self.evidence

    def guided(
        self, group_size, base_count, guided_count, max_group_attempts, *, run_id
    ):
        self.calls.append(
            (
                "guided",
                group_size,
                base_count,
                guided_count,
                max_group_attempts,
                run_id,
            )
        )
        return self.evidence

    def trainer(
        self, optimizer_steps, group_rewards, guided_artifact, *, run_id
    ):
        self.calls.append(
            ("trainer", optimizer_steps, group_rewards, guided_artifact, run_id)
        )
        return self.evidence


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_id="run-01",
        seeds=(5, 6, 5),
        seed=5,
        replays=2,
        p0_count=2,
        trajectories=2,
        max_turns=12,
        group_size=8,
        base_count=7,
        guided_count=1,
        max_group_attempts=20,
        optimizer_steps=1,
        group_rewards=(0, 0, 0, 0, 0, 0, 0, 1),
        guided_artifact=tmp_path / "guided.json",
    )


def _runtime_config(tmp_path: Path) -> dict[str, object]:
    dataset_path = tmp_path / "dataset.jsonl"
    if not dataset_path.exists():
        dataset_path.write_text('{"task_id":"cube-stack-5"}\n', encoding="utf-8")
    environment_config_path = tmp_path / "environment.yaml"
    if not environment_config_path.exists():
        environment_config_path.write_text("task: CubeStack\n", encoding="utf-8")
    resolved_verl_config_path = tmp_path / "resolved_verl.yaml"
    if not resolved_verl_config_path.exists():
        resolved_verl_config_path.write_text(
            "actor_rollout_ref:\n"
            "  model:\n"
            "    lora_rank: 16\n"
            "    lora_alpha: 32\n"
            "    target_modules: all-linear\n"
            "trainer: {}\n",
            encoding="utf-8",
        )
    model_path = tmp_path / "program-model"
    model_path.mkdir(exist_ok=True)
    model_config = model_path / "config.json"
    if not model_config.exists():
        model_config.write_text('{"model_type":"qwen2"}\n', encoding="utf-8")
    verl_source_path = tmp_path / "verl-source"
    verl_source_path.mkdir(exist_ok=True)
    return {
        "runtime": {
            "project_root": str(tmp_path),
            "dataset_path": str(dataset_path),
            "program_model_path": str(model_path),
            "verl_source_path": str(verl_source_path),
            "verl_pinned_sha": "a" * 40,
            "verl_resolved_config_path": str(resolved_verl_config_path),
        },
        "program_service": {
            "mode": "actor_identity",
            "model": str(model_path),
        },
        "task": {
            "environment": "robosuite_cube_stack",
            "api": "franka_control_privileged",
            "privilege": "privileged",
            "config_path": str(environment_config_path),
        }
    }


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("seed", ("seed", (5, 6, 5), "run-01")),
        ("oracle", ("oracle", 5, 2, "run-01")),
        ("collector", ("collector", 2, 2, 12, "run-01")),
        ("guided", ("guided", 8, 7, 1, 20, "run-01")),
    ],
)
def test_dispatches_each_collection_gate_without_runtime_imports(
    command: str, expected: tuple, tmp_path: Path
) -> None:
    runtime = _FakeRuntime()

    assert server_adapter._dispatch(runtime, command, _args(tmp_path)) == {}
    assert runtime.calls == [expected]


def test_dispatches_trainer_with_verified_group_artifact_path(tmp_path: Path) -> None:
    runtime = _FakeRuntime()

    server_adapter._dispatch(runtime, "trainer", _args(tmp_path))

    assert runtime.calls == [
        (
            "trainer",
            1,
            (0, 0, 0, 0, 0, 0, 0, 1),
            (tmp_path / "guided.json").resolve(),
            "run-01",
        )
    ]


def test_execute_seed_gate_verifies_and_atomically_writes_envelope(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    artifact_path = tmp_path / "gate02_seed.json"
    runtime = _FakeRuntime(
        {
            "seeds": [5, 6, 5],
            "initial_state_sha256": ["a" * 64, "b" * 64, "a" * 64],
        }
    )

    payload = server_adapter.execute_gate(
        config_path=config_path,
        artifact_path=artifact_path,
        command="seed",
        run_id="run-01",
        runtime=runtime,
        args=_args(tmp_path),
        config_loader=lambda _path, *, check_runtime_paths: _runtime_config(tmp_path),
        git_sha_loader=lambda _root: "c" * 40,
    )

    assert payload["gate"] == "seed"
    assert payload["passed"] is True
    assert payload["execution_mode"] == server_adapter.CANONICAL_EXECUTION_MODE
    assert payload["run_id"] == "run-01"
    assert payload["dataset_sha256"] == artifact_file_sha256(
        tmp_path / "dataset.jsonl"
    )
    dependencies = server_adapter.runtime_dependency_hashes(_runtime_config(tmp_path))
    assert payload["resolved_environment_sha256"] == dependencies[
        "resolved_environment_sha256"
    ]
    assert payload["verl_resolved_config_sha256"] == dependencies[
        "verl_resolved_config_sha256"
    ]
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == payload
    assert not list(tmp_path.glob("*.tmp"))


def test_execute_gate_hashes_the_exact_bytes_given_to_the_config_loader(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_bytes = b"schema_version: 1\n"
    config_path.write_bytes(config_bytes)
    observed: dict[str, bytes] = {}

    def recording_loader(raw: bytes, *, check_runtime_paths: bool) -> dict[str, object]:
        assert check_runtime_paths is True
        observed["raw"] = raw
        return _runtime_config(tmp_path)

    payload = server_adapter.execute_gate(
        config_path=config_path,
        artifact_path=tmp_path / "gate02_seed.json",
        command="seed",
        run_id="run-01",
        runtime=_FakeRuntime(
            {
                "seeds": [5, 6, 5],
                "initial_state_sha256": ["a" * 64, "b" * 64, "a" * 64],
            }
        ),
        args=_args(tmp_path),
        config_loader=recording_loader,
        git_sha_loader=lambda _root: "c" * 40,
    )

    assert observed["raw"] == config_bytes
    assert payload["config_sha256"] == hashlib.sha256(observed["raw"]).hexdigest()


def test_execute_gate_rechecks_config_after_typed_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    artifact = tmp_path / "gate02_seed.json"
    real_verifier = server_adapter._VERIFIERS["seed"]

    def mutate_after_verification(payload: object) -> None:
        real_verifier(payload)
        config_path.write_text("schema_version: 2\n", encoding="utf-8")

    monkeypatch.setitem(server_adapter._VERIFIERS, "seed", mutate_after_verification)

    with pytest.raises(server_adapter.ServerAdapterError, match="config.*changed"):
        server_adapter.execute_gate(
            config_path=config_path,
            artifact_path=artifact,
            command="seed",
            run_id="run-01",
            runtime=_FakeRuntime(
                {
                    "seeds": [5, 6, 5],
                    "initial_state_sha256": ["a" * 64, "b" * 64, "a" * 64],
                }
            ),
            args=_args(tmp_path),
            config_loader=lambda _raw, *, check_runtime_paths: _runtime_config(tmp_path),
            git_sha_loader=lambda _root: "c" * 40,
        )

    assert not artifact.exists()
    failure = json.loads(
        (tmp_path / "gate02_seed.json.failure.json").read_text(encoding="utf-8")
    )
    assert failure["exception"]["stage"] == "post_config"


def test_main_constructs_runtime_from_the_authoritative_config_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    preflight_config = _runtime_config(tmp_path)
    preflight_config["snapshot_marker"] = "pre-execute"
    authoritative_config = _runtime_config(tmp_path)
    authoritative_config["snapshot_marker"] = "execute"
    constructed_from: list[str] = []

    monkeypatch.setattr(
        server_adapter,
        "load_and_validate_server_config",
        lambda _path, *, check_runtime_paths: preflight_config,
    )
    monkeypatch.setattr(
        server_adapter,
        "load_and_validate_server_config_bytes",
        lambda _raw, *, check_runtime_paths: authoritative_config,
    )
    monkeypatch.setattr(server_adapter, "_validate_gate_request", lambda **_kwargs: {})

    def runtime_factory(config: dict[str, object]) -> _FakeRuntime:
        constructed_from.append(str(config["snapshot_marker"]))
        return _FakeRuntime(
            {
                "seeds": [5, 6, 5],
                "initial_state_sha256": ["a" * 64, "b" * 64, "a" * 64],
            }
        )

    assert (
        server_adapter.main(
            [
                "--config",
                str(config_path),
                "--artifact",
                str(tmp_path / "gate02_seed.json"),
                "--run-id",
                "run-01",
                "seed",
            ],
            runtime_factory=runtime_factory,
            git_sha_loader=lambda _root: "c" * 40,
        )
        == 0
    )

    assert constructed_from == ["execute"]


def test_execute_gate_refuses_runtime_envelope_spoofing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    runtime = _FakeRuntime({"passed": True})

    with pytest.raises(server_adapter.ServerAdapterError, match="cannot override"):
        server_adapter.execute_gate(
            config_path=config_path,
            artifact_path=tmp_path / "artifact.json",
            command="seed",
            run_id="run-01",
            runtime=runtime,
            args=_args(tmp_path),
            config_loader=lambda _path, *, check_runtime_paths: _runtime_config(tmp_path),
            git_sha_loader=lambda _root: "c" * 40,
        )

    failure = json.loads(
        (tmp_path / "artifact.json.failure.json").read_text(encoding="utf-8")
    )
    assert failure["exception"] == {
        "type": "ServerAdapterError",
        "message": "gate runtime cannot override envelope field(s): passed",
        "stage": "evidence_validation",
    }


def test_execute_gate_persists_runtime_dispatch_failure_without_success_artifact(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    artifact = tmp_path / "gate02_seed.json"

    class _FailingRuntime(_FakeRuntime):
        def seed(self, seeds, *, run_id):
            raise RuntimeError("reset exploded")

    with pytest.raises(RuntimeError, match="reset exploded"):
        server_adapter.execute_gate(
            config_path=config_path,
            artifact_path=artifact,
            command="seed",
            run_id="run-01",
            runtime=_FailingRuntime(),
            args=_args(tmp_path),
            config_loader=lambda _path, *, check_runtime_paths: _runtime_config(tmp_path),
            git_sha_loader=lambda _root: "c" * 40,
        )

    failure = json.loads(
        (tmp_path / "gate02_seed.json.failure.json").read_text(encoding="utf-8")
    )
    assert not artifact.exists()
    assert failure == {
        "schema_version": 1,
        "gate": "seed",
        "passed": False,
        "run_id": "run-01",
        "config_sha256": artifact_file_sha256(config_path),
        "git_sha": "c" * 40,
        "dataset_sha256": artifact_file_sha256(tmp_path / "dataset.jsonl"),
        "exception": {
            "type": "RuntimeError",
            "message": "reset exploded",
            "stage": "runtime_dispatch",
        },
    }


def test_execute_gate_refuses_to_overwrite_existing_failure_evidence(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    artifact = tmp_path / "gate02_seed.json"
    failure = tmp_path / "gate02_seed.json.failure.json"
    failure.write_bytes(b"original immutable failure\n")
    runtime = _FakeRuntime()

    with pytest.raises(FileExistsError, match="failure artifact already exists"):
        server_adapter.execute_gate(
            config_path=config_path,
            artifact_path=artifact,
            command="seed",
            run_id="run-01",
            runtime=runtime,
            args=_args(tmp_path),
            config_loader=lambda _path, *, check_runtime_paths: _runtime_config(tmp_path),
            git_sha_loader=lambda _root: "c" * 40,
        )

    assert failure.read_bytes() == b"original immutable failure\n"
    assert runtime.calls == []


def test_execute_gate_rejects_project_sha_change_before_publishing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    runtime = _FakeRuntime(
        {
            "seeds": [5, 6, 5],
            "initial_state_sha256": ["a" * 64, "b" * 64, "a" * 64],
        }
    )
    shas = iter(("c" * 40, "d" * 40))
    artifact = tmp_path / "seed.json"

    with pytest.raises(server_adapter.ServerAdapterError, match="changed"):
        server_adapter.execute_gate(
            config_path=config_path,
            artifact_path=artifact,
            command="seed",
            run_id="run-01",
            runtime=runtime,
            args=_args(tmp_path),
            config_loader=lambda _path, *, check_runtime_paths: _runtime_config(tmp_path),
            git_sha_loader=lambda _root: next(shas),
        )

    assert not artifact.exists()
    failure = json.loads(
        (tmp_path / "seed.json.failure.json").read_text(encoding="utf-8")
    )
    assert failure["exception"]["stage"] == "post_git"
    assert failure["git_sha"] == "c" * 40


def test_execute_gate_rejects_dataset_change_before_publishing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    config = _runtime_config(tmp_path)
    dataset_path = Path(config["runtime"]["dataset_path"])
    artifact = tmp_path / "seed.json"

    class _MutatingRuntime(_FakeRuntime):
        def seed(self, seeds, *, run_id):
            dataset_path.write_text("changed-dataset\n", encoding="utf-8")
            return {
                "seeds": [5, 6, 5],
                "initial_state_sha256": ["a" * 64, "b" * 64, "a" * 64],
            }

    with pytest.raises(server_adapter.ServerAdapterError, match="dataset.*changed"):
        server_adapter.execute_gate(
            config_path=config_path,
            artifact_path=artifact,
            command="seed",
            run_id="run-01",
            runtime=_MutatingRuntime(),
            args=_args(tmp_path),
            config_loader=lambda _path, *, check_runtime_paths: config,
            git_sha_loader=lambda _root: "c" * 40,
        )

    assert not artifact.exists()
    failure = json.loads(
        (tmp_path / "seed.json.failure.json").read_text(encoding="utf-8")
    )
    assert failure["exception"]["stage"] == "post_dataset"


@pytest.mark.parametrize(
    ("path_field", "message"),
    [
        (("task", "config_path"), "resolved environment.*changed"),
        (("runtime", "verl_resolved_config_path"), "resolved VeRL config.*changed"),
    ],
)
def test_execute_gate_rejects_runtime_dependency_change_before_publishing(
    path_field: tuple[str, str], message: str, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    config = _runtime_config(tmp_path)
    dependency_path = Path(config[path_field[0]][path_field[1]])
    artifact = tmp_path / f"{path_field[1]}.seed.json"

    class _MutatingRuntime(_FakeRuntime):
        def seed(self, seeds, *, run_id):
            dependency_path.write_text("mutated: true\n", encoding="utf-8")
            return {
                "seeds": [5, 6, 5],
                "initial_state_sha256": ["a" * 64, "b" * 64, "a" * 64],
            }

    with pytest.raises(server_adapter.ServerAdapterError, match=message):
        server_adapter.execute_gate(
            config_path=config_path,
            artifact_path=artifact,
            command="seed",
            run_id="run-01",
            runtime=_MutatingRuntime(),
            args=_args(tmp_path),
            config_loader=lambda _path, *, check_runtime_paths: config,
            git_sha_loader=lambda _root: "c" * 40,
        )

    failure = json.loads(
        artifact.with_name(f"{artifact.name}.failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["exception"]["stage"] == "post_runtime_dependencies"


def test_execute_gate_rolls_back_transaction_when_post_checks_fail(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    calls: list[str] = []
    transaction = server_adapter.GateTransaction(
        evidence={
            "seeds": [5, 6, 5],
            "initial_state_sha256": ["a" * 64, "b" * 64, "a" * 64],
        },
        commit_callback=lambda: calls.append("commit"),
        rollback_callback=lambda: calls.append("rollback"),
    )
    runtime = _FakeRuntime(transaction)
    shas = iter(("c" * 40, "d" * 40))

    with pytest.raises(server_adapter.ServerAdapterError, match="changed"):
        server_adapter.execute_gate(
            config_path=config_path,
            artifact_path=tmp_path / "seed.json",
            command="seed",
            run_id="run-01",
            runtime=runtime,
            args=_args(tmp_path),
            config_loader=lambda _path, *, check_runtime_paths: _runtime_config(tmp_path),
            git_sha_loader=lambda _root: next(shas),
        )

    assert calls == ["rollback"]


def test_execute_gate_keeps_original_failure_primary_when_rollback_fails(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    artifact = tmp_path / "seed.json"
    transaction = server_adapter.GateTransaction(
        evidence={
            "seeds": [5, 6, 5],
            "initial_state_sha256": ["a" * 64, "b" * 64, "a" * 64],
        },
        commit_callback=lambda: None,
        rollback_callback=lambda: (_ for _ in ()).throw(RuntimeError("rollback broke")),
    )
    shas = iter(("c" * 40, "d" * 40))

    with pytest.raises(server_adapter.ServerAdapterError, match="changed"):
        server_adapter.execute_gate(
            config_path=config_path,
            artifact_path=artifact,
            command="seed",
            run_id="run-01",
            runtime=_FakeRuntime(transaction),
            args=_args(tmp_path),
            config_loader=lambda _path, *, check_runtime_paths: _runtime_config(tmp_path),
            git_sha_loader=lambda _root: next(shas),
        )

    failure = json.loads(
        (tmp_path / "seed.json.failure.json").read_text(encoding="utf-8")
    )
    assert not artifact.exists()
    assert failure["exception"]["type"] == "ServerAdapterError"
    assert failure["exception"]["stage"] == "post_git"
    assert failure["rollback_exception"] == {
        "type": "RuntimeError",
        "message": "rollback broke",
        "stage": "transaction_rollback",
    }


def test_gate6_verifier_failure_aborts_published_checkpoint_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    config = _runtime_config(tmp_path)
    dataset_sha256 = artifact_file_sha256(config["runtime"]["dataset_path"])
    dependency_hashes = server_adapter.runtime_dependency_hashes(config)
    actor_identity = build_actor_identity(config)
    args = _args(tmp_path)
    args.guided_artifact.write_text(
        json.dumps(
            {
                "run_id": "run-01",
                "config_sha256": artifact_file_sha256(config_path),
                "git_sha": "c" * 40,
                    "dataset_sha256": dataset_sha256,
                    **dependency_hashes,
                    "program_model_sha256": actor_identity["program_model_sha256"],
                    "actor_binding_sha256": actor_identity["actor_binding_sha256"],
            }
        ),
        encoding="utf-8",
    )
    claim_root = tmp_path / "gate06-checkpoint"
    checkpoint = claim_root / "global_step_1" / "actor"
    claim = AtomicCheckpointClaim(checkpoint, claim_root=claim_root)
    claim.__enter__()

    def save_checkpoint(staging: Path) -> None:
        staging.mkdir(parents=True)
        (staging / "state.bin").write_bytes(b"checkpoint")

    claim.publish(
        save_checkpoint,
        optimizer_step_before=0,
        optimizer_step_after=1,
    )
    transaction = server_adapter.GateTransaction(
        evidence={"optimizer_steps": 1},
        commit_callback=claim.commit,
        rollback_callback=claim.abort,
    )
    monkeypatch.setattr(server_adapter, "_validate_gate_request", lambda **_kwargs: {})

    def reject_trainer(_payload: object) -> None:
        raise server_adapter.GateArtifactError("trainer evidence rejected")

    monkeypatch.setitem(server_adapter._VERIFIERS, "trainer", reject_trainer)
    try:
        with pytest.raises(server_adapter.GateArtifactError, match="rejected"):
            server_adapter.execute_gate(
                config_path=config_path,
                artifact_path=tmp_path / "gate06_trainer.json",
                command="trainer",
                run_id="run-01",
                runtime=_FakeRuntime(transaction),
                args=args,
                config_loader=lambda _path, *, check_runtime_paths: config,
                git_sha_loader=lambda _root: "c" * 40,
            )
    finally:
        if claim_root.exists():
            claim.abort()

    assert not claim_root.exists()
    failure = json.loads(
        (tmp_path / "gate06_trainer.json.failure.json").read_text(encoding="utf-8")
    )
    assert failure["gate"] == "trainer"
    assert failure["exception"]["stage"] == "artifact_verification"
    assert "rollback_exception" not in failure


def test_execute_gate_persists_typed_verifier_failure(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    artifact = tmp_path / "seed.json"
    runtime = _FakeRuntime(
        {
            "seeds": [5, 5, 5],
            "initial_state_sha256": ["a" * 64, "a" * 64, "a" * 64],
        }
    )

    with pytest.raises(Exception, match="seed"):
        server_adapter.execute_gate(
            config_path=config_path,
            artifact_path=artifact,
            command="seed",
            run_id="run-01",
            runtime=runtime,
            args=_args(tmp_path),
            config_loader=lambda _path, *, check_runtime_paths: _runtime_config(tmp_path),
            git_sha_loader=lambda _root: "c" * 40,
        )

    failure = json.loads(
        (tmp_path / "seed.json.failure.json").read_text(encoding="utf-8")
    )
    assert not artifact.exists()
    assert failure["exception"]["type"] == "GateArtifactError"
    assert failure["exception"]["stage"] == "artifact_verification"
    assert failure["dataset_sha256"] == artifact_file_sha256(
        _runtime_config(tmp_path)["runtime"]["dataset_path"]
    )


def test_execute_gate_commits_transaction_only_after_artifact_publish(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    artifact = tmp_path / "seed.json"
    calls: list[str] = []
    transaction = server_adapter.GateTransaction(
        evidence={
            "seeds": [5, 6, 5],
            "initial_state_sha256": ["a" * 64, "b" * 64, "a" * 64],
        },
        commit_callback=lambda: calls.append(
            "commit_after_artifact" if artifact.is_file() else "commit_before_artifact"
        ),
        rollback_callback=lambda: calls.append("rollback"),
    )

    server_adapter.execute_gate(
        config_path=config_path,
        artifact_path=artifact,
        command="seed",
        run_id="run-01",
        runtime=_FakeRuntime(transaction),
        args=_args(tmp_path),
        config_loader=lambda _path, *, check_runtime_paths: _runtime_config(tmp_path),
        git_sha_loader=lambda _root: "c" * 40,
    )

    assert calls == ["commit_after_artifact"]


def test_execute_gate_commit_failure_removes_owned_artifact_and_rolls_back(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    artifact = tmp_path / "seed.json"
    calls: list[str] = []

    def fail_commit() -> None:
        calls.append("commit")
        raise RuntimeError("commit failed")

    transaction = server_adapter.GateTransaction(
        evidence={
            "seeds": [5, 6, 5],
            "initial_state_sha256": ["a" * 64, "b" * 64, "a" * 64],
        },
        commit_callback=fail_commit,
        rollback_callback=lambda: calls.append("rollback"),
    )

    with pytest.raises(RuntimeError, match="commit failed"):
        server_adapter.execute_gate(
            config_path=config_path,
            artifact_path=artifact,
            command="seed",
            run_id="run-01",
            runtime=_FakeRuntime(transaction),
            args=_args(tmp_path),
            config_loader=lambda _path, *, check_runtime_paths: _runtime_config(tmp_path),
            git_sha_loader=lambda _root: "c" * 40,
        )

    assert calls == ["commit", "rollback"]
    assert not artifact.exists()
    failure = json.loads(
        (tmp_path / "seed.json.failure.json").read_text(encoding="utf-8")
    )
    assert failure["exception"] == {
        "type": "RuntimeError",
        "message": "commit failed",
        "stage": "transaction_commit",
    }


@pytest.mark.parametrize("replacement_bytes", [b"foreign", None], ids=["different", "identical"])
def test_execute_gate_commit_failure_never_deletes_replaced_artifact(
    tmp_path: Path,
    replacement_bytes: bytes | None,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    artifact = tmp_path / "seed.json"
    calls: list[str] = []
    replacement_identity: tuple[int, int] | None = None
    expected_replacement: bytes | None = None

    def replace_then_fail_commit() -> None:
        nonlocal replacement_identity, expected_replacement
        published_bytes = artifact.read_bytes()
        expected_replacement = published_bytes if replacement_bytes is None else replacement_bytes
        artifact.unlink()
        artifact.write_bytes(expected_replacement)
        replacement_stat = artifact.stat()
        replacement_identity = (replacement_stat.st_dev, replacement_stat.st_ino)
        calls.append("commit")
        raise RuntimeError("commit failed after replacement")

    transaction = server_adapter.GateTransaction(
        evidence={
            "seeds": [5, 6, 5],
            "initial_state_sha256": ["a" * 64, "b" * 64, "a" * 64],
        },
        commit_callback=replace_then_fail_commit,
        rollback_callback=lambda: calls.append("rollback"),
    )

    with pytest.raises(RuntimeError, match="commit failed after replacement"):
        server_adapter.execute_gate(
            config_path=config_path,
            artifact_path=artifact,
            command="seed",
            run_id="run-01",
            runtime=_FakeRuntime(transaction),
            args=_args(tmp_path),
            config_loader=lambda _path, *, check_runtime_paths: _runtime_config(tmp_path),
            git_sha_loader=lambda _root: "c" * 40,
        )

    assert calls == ["commit", "rollback"]
    assert artifact.read_bytes() == expected_replacement
    artifact_stat = artifact.stat()
    assert (artifact_stat.st_dev, artifact_stat.st_ino) == replacement_identity
    assert (tmp_path / "seed.json.failure.json").is_file()


def test_execute_gate_commit_failure_never_deletes_owned_artifact_with_changed_bytes(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    artifact = tmp_path / "seed.json"
    calls: list[str] = []

    def mutate_then_fail_commit() -> None:
        original_stat = artifact.stat()
        artifact.write_bytes(b"concurrent in-place mutation")
        mutated_stat = artifact.stat()
        assert (mutated_stat.st_dev, mutated_stat.st_ino) == (
            original_stat.st_dev,
            original_stat.st_ino,
        )
        calls.append("commit")
        raise RuntimeError("commit failed after mutation")

    transaction = server_adapter.GateTransaction(
        evidence={
            "seeds": [5, 6, 5],
            "initial_state_sha256": ["a" * 64, "b" * 64, "a" * 64],
        },
        commit_callback=mutate_then_fail_commit,
        rollback_callback=lambda: calls.append("rollback"),
    )

    with pytest.raises(RuntimeError, match="commit failed after mutation"):
        server_adapter.execute_gate(
            config_path=config_path,
            artifact_path=artifact,
            command="seed",
            run_id="run-01",
            runtime=_FakeRuntime(transaction),
            args=_args(tmp_path),
            config_loader=lambda _path, *, check_runtime_paths: _runtime_config(tmp_path),
            git_sha_loader=lambda _root: "c" * 40,
        )

    assert calls == ["commit", "rollback"]
    assert artifact.read_bytes() == b"concurrent in-place mutation"
    assert (tmp_path / "seed.json.failure.json").is_file()


def test_trainer_gate_rejects_guided_dataset_identity_before_dispatch(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    config = _runtime_config(tmp_path)
    args = _args(tmp_path)
    guided_payload = _guided_payload("run-01")
    guided_payload["config_sha256"] = artifact_file_sha256(config_path)
    guided_payload["git_sha"] = "c" * 40
    guided_payload["dataset_sha256"] = "0" * 64
    args.guided_artifact.write_text(json.dumps(guided_payload), encoding="utf-8")
    runtime = _FakeRuntime()

    with pytest.raises(server_adapter.ServerAdapterError, match="dataset_sha256"):
        server_adapter.execute_gate(
            config_path=config_path,
            artifact_path=tmp_path / "trainer.json",
            command="trainer",
            run_id="run-01",
            runtime=runtime,
            args=args,
            config_loader=lambda _path, *, check_runtime_paths: config,
            git_sha_loader=lambda _root: "c" * 40,
        )

    assert runtime.calls == []


def test_adapter_validate_only_never_constructs_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    monkeypatch.setattr(
        server_adapter,
        "load_and_validate_server_config",
        lambda _path, *, check_runtime_paths: {"runtime": {"project_root": str(tmp_path)}},
    )

    def forbidden_runtime(_config):
        raise AssertionError("validate-only must not build a runtime")

    exit_code = server_adapter.main(
        [
            "--config",
            str(config_path),
            "--artifact",
            str(tmp_path / "gate02.json"),
            "--run-id",
            "run-01",
            "--validate-only",
            "seed",
            "--seeds",
            "5,6,5",
        ],
        runtime_factory=forbidden_runtime,
    )

    assert exit_code == 0
    assert not (tmp_path / "gate02.json").exists()
    assert json.loads(capsys.readouterr().out)["request"] == {
        "seed_sequence": [5, 6, 5]
    }


def test_adapter_validate_only_rejects_gate_specific_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    monkeypatch.setattr(
        server_adapter,
        "load_and_validate_server_config",
        lambda _path, *, check_runtime_paths: {"runtime": {"project_root": str(tmp_path)}},
    )

    with pytest.raises(server_adapter.ServerAdapterError, match="exact 5,6,5"):
        server_adapter.main(
            [
                "--config",
                str(config_path),
                "--artifact",
                str(tmp_path / "gate02.json"),
                "--run-id",
                "run-01",
                "--validate-only",
                "seed",
                "--seeds",
                "5,5,5",
            ]
        )


def test_adapter_trainer_validate_only_requires_and_verifies_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    monkeypatch.setattr(
        server_adapter,
        "load_and_validate_server_config",
        lambda _path, *, check_runtime_paths: {"runtime": {"project_root": str(tmp_path)}},
    )

    with pytest.raises(FileNotFoundError, match="guided artifact"):
        server_adapter.main(
            [
                "--config",
                str(config_path),
                "--artifact",
                str(tmp_path / "gate06.json"),
                "--run-id",
                "run-01",
                "--validate-only",
                "trainer",
                "--guided-artifact",
                str(tmp_path / "missing-gate05.json"),
            ]
        )


@pytest.mark.parametrize(
    ("identity_field", "bad_value"),
    [
        ("run_id", "different-run"),
        ("config_sha256", "e" * 64),
        ("git_sha", "e" * 40),
    ],
)
def test_adapter_trainer_validate_only_rejects_any_guided_identity_mismatch(
    identity_field: str,
    bad_value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    guided_artifact = tmp_path / "gate05.json"
    guided_payload = _guided_payload("run-01")
    guided_payload["config_sha256"] = artifact_file_sha256(config_path)
    guided_payload["git_sha"] = "d" * 40
    guided_payload[identity_field] = bad_value
    guided_artifact.write_text(json.dumps(guided_payload), encoding="utf-8")
    monkeypatch.setattr(
        server_adapter,
        "load_and_validate_server_config",
        lambda _path, *, check_runtime_paths: {
            "runtime": {"project_root": str(project_root)}
        },
    )
    observed_roots: list[Path] = []

    def clean_git_sha(root: Path) -> str:
        observed_roots.append(root)
        return "d" * 40

    with pytest.raises(
        server_adapter.ServerAdapterError,
        match="run_id/config_sha256/git_sha",
    ):
        server_adapter.main(
            [
                "--config",
                str(config_path),
                "--artifact",
                str(tmp_path / "gate06.json"),
                "--run-id",
                "run-01",
                "--validate-only",
                "trainer",
                "--guided-artifact",
                str(guided_artifact),
            ],
            runtime_factory=lambda _config: (_ for _ in ()).throw(
                AssertionError("validate-only must not build a runtime")
            ),
            git_sha_loader=clean_git_sha,
        )

    assert observed_roots == [project_root.resolve()]


def test_validate_cli_request_accepts_matching_trainer_identity_from_clean_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    guided_artifact = tmp_path / "gate05.json"
    guided_payload = _guided_payload("run-01")
    guided_payload["config_sha256"] = artifact_file_sha256(config_path)
    guided_payload["git_sha"] = "d" * 40
    guided_artifact.write_text(json.dumps(guided_payload), encoding="utf-8")
    monkeypatch.setattr(
        server_adapter,
        "load_and_validate_server_config",
        lambda _path, *, check_runtime_paths: {
            "runtime": {"project_root": str(project_root)}
        },
    )
    observed_roots: list[Path] = []

    result = server_adapter.validate_cli_request(
        [
            "--config",
            str(config_path),
            "--artifact",
            str(tmp_path / "gate06.json"),
            "--run-id",
            "run-01",
            "--validate-only",
            "trainer",
            "--guided-artifact",
            str(guided_artifact),
        ],
        git_sha_loader=lambda root: observed_roots.append(root) or "d" * 40,
    )

    assert result["gate"] == "trainer"
    assert observed_roots == [project_root.resolve()]


def test_actor_metric_extraction_requires_finite_numeric_evidence() -> None:
    actor_output = type(
        "ActorOutput",
        (),
        {
            "meta_info": {
                "metrics": {"actor/grad_norm": [1.0, 2.0], "ignored": "text"}
            }
        },
    )()

    assert server_adapter.ConcreteGateRuntime._actor_metrics(actor_output) == {
        "actor/grad_norm": 1.5
    }

    with pytest.raises(server_adapter.ServerAdapterError, match="no finite"):
        server_adapter.ConcreteGateRuntime._actor_metrics({"metrics": {"loss": float("nan")}})


def test_collection_cleanup_attempts_collector_evaluator_and_workers() -> None:
    events: list[str] = []

    class _Resource:
        def __init__(self, name: str, *, fails: bool = False) -> None:
            self.name = name
            self.fails = fails

        def close(self) -> None:
            events.append(self.name)
            if self.fails:
                raise RuntimeError(f"{self.name} close failed")

    with pytest.raises(RuntimeError, match="collector close failed"):
        server_adapter._close_collection_resources(
            _Resource("collector", fails=True),
            _Resource("evaluator", fails=True),
            _Resource("workers"),
        )

    assert events == ["collector", "evaluator", "workers"]


def test_collector_close_failure_does_not_mask_primary_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_error = ValueError("collector body failed")

    class _Generator:
        def generate(self, *_args: object) -> object:
            raise primary_error

    class _Session:
        generator = _Generator()

        def close(self) -> None:
            raise RuntimeError("collector session close failed")

    runtime = server_adapter.ConcreteGateRuntime({})
    monkeypatch.setattr(runtime, "_task_for_seed", lambda _seed: SimpleNamespace(prompt="task"))
    monkeypatch.setattr(runtime, "_open_collection_session", lambda _task: _Session())

    with pytest.raises(ValueError, match="collector body failed") as caught:
        runtime.collector(2, 2, 12, run_id="run-01")

    assert caught.value is primary_error
    assert isinstance(caught.value.cleanup_error, RuntimeError)
    assert str(caught.value.cleanup_error) == "collector session close failed"


def test_guided_close_failure_does_not_mask_primary_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_error = ValueError("guided body failed")

    class _CleanEvaluator:
        def drain_history(self) -> tuple[object, ...]:
            return ()

    class _Assembler:
        def __init__(self, clean_evaluator: object) -> None:
            self.clean_evaluator = clean_evaluator

        def assemble(self, _task: object) -> object:
            raise primary_error

    class _Session:
        def __init__(self) -> None:
            self.clean_evaluator = _CleanEvaluator()
            self.assembler = _Assembler(self.clean_evaluator)

        def close(self) -> None:
            raise RuntimeError("guided session close failed")

    runtime = server_adapter.ConcreteGateRuntime({})
    monkeypatch.setattr(runtime, "_task_for_seed", lambda _seed: object())
    monkeypatch.setattr(runtime, "_open_collection_session", lambda _task: _Session())

    with pytest.raises(ValueError, match="guided body failed") as caught:
        runtime.guided(8, 7, 1, 1, run_id="run-01")

    assert caught.value is primary_error
    assert isinstance(caught.value.cleanup_error, RuntimeError)
    assert str(caught.value.cleanup_error) == "guided session close failed"


def test_concrete_runtime_uses_server_seed_state_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from capx.rl.capsule import server_factory

    sentinel = (object(),)
    monkeypatch.setattr(
        server_factory,
        "load_task_instances",
        lambda _config: (_ for _ in ()).throw(AssertionError("strict loader was used")),
    )
    monkeypatch.setattr(server_factory, "resolve_task_instances", lambda _config: sentinel)

    assert server_adapter.ConcreteGateRuntime({})._tasks() is sentinel


def test_oracle_reset_evidence_is_read_from_typed_replay_diagnostics() -> None:
    result = SimpleNamespace(
        diagnostics={
            "reset_info": {
                "capsule_reset_evidence": {
                    "namespace_fresh": True,
                    "api_state_cleared": True,
                    "api_reset_count": 1,
                    "api_reset_confirmed_count": 1,
                }
            }
        }
    )

    assert server_adapter.ConcreteGateRuntime._reset_evidence(result) == (True, True)

    with pytest.raises(server_adapter.ServerAdapterError, match="capsule_reset_evidence"):
        server_adapter.ConcreteGateRuntime._reset_evidence(SimpleNamespace(diagnostics={}))


def _guided_payload(run_id: str) -> dict[str, object]:
    task_id = "cube-stack-5"
    prompt = "stack the cubes"
    state_hash = "a" * 64
    task = TaskInstanceV1(
        task_id=task_id,
        environment_seed=5,
        prompt=prompt,
        environment="robosuite_cube_stack",
        api="franka_control_privileged",
        privilege="privileged",
        initial_state_sha256=state_hash,
    )
    group_uid = deterministic_group_uid(task)
    base_sources = [f"program_{index}()" for index in range(7)]

    def replay_diagnostics(outcome: str) -> dict[str, object]:
        return {
            "evaluator_attempt_history": [
                {
                    "attempt": 1,
                    "outcome": outcome,
                    "worker_replaced": False,
                    "retry_scheduled": False,
                    "error_type": None,
                    "error_message": None,
                }
            ]
        }

    base_results = [
        ProgramReplayResultV1(
            task_id=task_id,
            environment_seed=5,
            program_sample_id=f"sample-{index}",
            source=source,
            initial_state_sha256=state_hash,
            outcome=ReplayOutcome.TASK_FAILURE,
            raw_reward=0.9 if index == 0 else 0.8 if index == 1 else 0.0,
            binary_reward=0.0,
            task_completed=False,
            diagnostics=replay_diagnostics("task_failure"),
        )
        for index, source in enumerate(base_sources)
    ]
    trajectory_id = f"{group_uid}:p0-0:trajectory-0"
    draft = RepairDraft(
        task_id=task_id,
        environment_seed=5,
        program_sample_id="sample-0",
        repair_trajectory_id=trajectory_id,
        base_source=base_sources[0],
        base_units=(
            BaseUnitSpan("program", 0, len(base_sources[0]), base_sources[0]),
        ),
    )
    draft.submit(
        {
            "action": "replace",
            "target": "base:program",
            "source": "pt_program()",
            "rationale": "repair",
        }
    )
    trace = draft.to_trace()
    pt_result = ProgramReplayResultV1(
        task_id=task_id,
        environment_seed=5,
        program_sample_id=f"{trajectory_id}:pt",
        source=trace.final_source,
        initial_state_sha256=state_hash,
        outcome=ReplayOutcome.SUCCESS,
        raw_reward=1.0,
        binary_reward=1.0,
        task_completed=True,
        diagnostics=replay_diagnostics("success"),
    )
    p_hat_result = ProgramReplayResultV1(
        task_id=task_id,
        environment_seed=5,
        program_sample_id="sample-7",
        source="success_program()",
        initial_state_sha256=state_hash,
        outcome=ReplayOutcome.SUCCESS,
        raw_reward=1.0,
        binary_reward=1.0,
        task_completed=True,
        diagnostics=replay_diagnostics("success"),
    )
    members = tuple(
        [
            LearningMemberV1(
                member_type="base",
                program_sample_id=result.program_sample_id,
                prompt=prompt,
                response=result.source,
                reward=0.0,
            )
            for result in base_results
        ]
        + [
            LearningMemberV1(
                member_type="critique_guided_revision",
                program_sample_id=p_hat_result.program_sample_id,
                repair_trajectory_id=trajectory_id,
                prompt=prompt,
                response=p_hat_result.source,
                reward=1.0,
            )
        ]
    )
    group = LearningGroupV1(
        task_id=task_id,
        environment_seed=5,
        group_uid=group_uid,
        initial_state_sha256=state_hash,
        members=members,
    )
    attempts: list[RepairAttempt] = []
    for p0_rank, p0_sample_id in enumerate(("sample-0", "sample-1")):
        for trajectory_index in range(2):
            attempt_trajectory_id = (
                f"{group_uid}:p0-{p0_rank}:trajectory-{trajectory_index}"
            )
            if (p0_rank, trajectory_index) == (0, 0):
                attempts.append(
                    RepairAttempt(
                        p0_rank=0,
                        trajectory_index=0,
                        p0_program_sample_id="sample-0",
                        repair_trajectory_id=trajectory_id,
                        status="guided_success",
                        trace=trace,
                        pt_result=pt_result,
                        revision_program_sample_id=p_hat_result.program_sample_id,
                        revision_source=p_hat_result.source,
                        revision_result=p_hat_result,
                        selected=True,
                    )
                )
            else:
                attempts.append(
                    RepairAttempt(
                        p0_rank=p0_rank,
                        trajectory_index=trajectory_index,
                        p0_program_sample_id=p0_sample_id,
                        repair_trajectory_id=attempt_trajectory_id,
                        status="rejected",
                        rejection_reason="collector_error",
                        rejection_message="mocked rejection",
                    )
                )
    replay_results = [*base_results, pt_result, p_hat_result]
    return {
        "schema_version": 1,
        "gate": "guided",
        "passed": True,
        "run_id": run_id,
        "config_sha256": "c" * 64,
        "git_sha": "d" * 40,
        "dataset_sha256": "9" * 64,
        "resolved_environment_sha256": "e" * 64,
        "verl_resolved_config_sha256": "f" * 64,
        "program_model_sha256": "1" * 64,
        "actor_binding_sha256": "2" * 64,
        "task_instance": task.to_dict(),
        "original_prompt": prompt,
        "training_input_contains_critique": False,
        "learning_group": group.to_dict(),
        "base_results": [result.to_dict() for result in base_results],
        "repair_attempts": [attempt.to_dict() for attempt in attempts],
        "selected_repair": {
            "p0_rank": 0,
            "trajectory_index": 0,
            "trace": trace.to_dict(),
            "p0_result": base_results[0].to_dict(),
            "pt_result": pt_result.to_dict(),
            "p_hat_result": p_hat_result.to_dict(),
        },
        "selected_group_attempt_index": 0,
        "discarded_group_attempts": [],
        "replay_events": [
            {
                "group_attempt_index": 0,
                "result_index": index,
                "selected_group": True,
                "result": result.to_dict(),
            }
            for index, result in enumerate(replay_results)
        ],
        "replay_event_count": len(replay_results),
        "attempt_event_count": len(replay_results),
        "retry_count": 0,
        "infra_failures": 0,
        "evaluator_failures": 0,
        "worker_replacements": 0,
    }


def test_concrete_trainer_rebuilds_full_repair_audit_and_hashes_input_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capx.rl.capsule import server_factory, trainer as trainer_module

    run_id = "run/../unsafe"
    guided_artifact = tmp_path / "gate05_guided_group.json"
    guided_payload = _guided_payload(run_id)
    guided_artifact.write_text(json.dumps(guided_payload), encoding="utf-8")
    guided_task = TaskInstanceV1.from_dict(guided_payload["task_instance"])
    observed: dict[str, object] = {}

    class _Workers:
        actor_rollout_wg = object()
        ref_policy_wg = object()
        tokenizer = object()
        data_proto_factory = object()
        rollout_mode = "sync"
        ppo_epochs = 1
        ppo_mini_batch_size = 8
        kl_loss_coef = 0.001
        data_parallel_world_size = 1
        sequence_parallel_size = 1
        _verl_provenance = {
            "source_path": str((tmp_path / "verl").resolve()),
            "expected_sha": "e" * 40,
            "actual_sha": "e" * 40,
            "clean": True,
            "worker_count": 1,
            "worker_ranks": [0],
            "worker_module_paths": [
                str((tmp_path / "verl" / "verl" / "__init__.py").resolve())
            ],
        }

        def __init__(self) -> None:
            self._optimizer_steps = iter((0, 1))

        def optimizer_step(self) -> int:
            return next(self._optimizer_steps)

        def save_checkpoint(self, path: Path, step: int) -> None:
            observed["checkpoint_step"] = step
            path.mkdir(parents=True)
            (path / "state.bin").write_bytes(b"checkpoint")

        def verl_provenance(self) -> dict[str, object]:
            observed["verl_provenance_calls"] = int(
                observed.get("verl_provenance_calls", 0)
            ) + 1
            return dict(self._verl_provenance)

        def close(self) -> None:
            observed["workers_closed"] = True

    class _Encoder:
        def __init__(self, **_kwargs: object) -> None:
            pass

    class _Trainer:
        def __init__(self, **kwargs: object) -> None:
            self.assembler = kwargs["assembler"]
            self.actor_updates_completed = 0

        def run_step(self, task):
            assembly = self.assembler.assemble(task)
            observed["repair_attempt_count"] = len(assembly.repair_attempts)
            observed["selected_attempt_count"] = sum(
                attempt.selected for attempt in assembly.repair_attempts
            )
            self.actor_updates_completed = 1
            return SimpleNamespace(
                actor_output={"metrics": {"actor/grad_norm": 2.0}},
                artifact=SimpleNamespace(guided_token_mask=((0,), (1,))),
                skipped_actor_update=False,
            )

    workers = _Workers()
    monkeypatch.setattr(server_factory, "start_verl_workers", lambda _config: workers)
    monkeypatch.setattr(server_factory, "VeRLGroupEncoder", _Encoder)
    monkeypatch.setattr(trainer_module, "CapsuleCritiqueRayTrainer", _Trainer)
    monkeypatch.setattr(trainer_module, "MemoryArtifactSink", lambda: object())
    output_dir = tmp_path / "outputs"
    runtime = server_adapter.ConcreteGateRuntime(
        {
            "task": {
                "environment": "robosuite_cube_stack",
                "api": "franka_control_privileged",
                "privilege": "privileged",
            },
            "capsule": {
                "revision_input_max_tokens": 8192,
                "revision_response_max_tokens": 2048,
            },
            "program_service": {},
            "runtime": {
                "project_root": str(tmp_path),
                "output_dir": str(output_dir),
            },
        }
    )
    monkeypatch.setattr(runtime, "_tasks", lambda: (guided_task,))

    evidence = runtime.trainer(
        1,
        (0, 0, 0, 0, 0, 0, 0, 1),
        guided_artifact,
        run_id=run_id,
    )

    assert isinstance(evidence, server_adapter.GateTransaction)

    assert observed["repair_attempt_count"] == 4
    assert observed["selected_attempt_count"] == 1
    checkpoint = Path(str(evidence["checkpoint"]))
    assert checkpoint.is_relative_to(output_dir / "gate06")
    assert ".." not in checkpoint.parts
    assert evidence["guided_artifact_sha256"] == artifact_file_sha256(guided_artifact)
    assert evidence["rollout_mode"] == "sync"
    assert evidence["ppo_epochs"] == 1
    assert evidence["ppo_mini_batch_size"] == 8
    assert evidence["data_parallel_world_size"] == 1
    assert evidence["sequence_parallel_size"] == 1
    assert evidence["reference_kl_coef"] == 0.001
    assert evidence["actor_update_rpcs"] == 1
    assert evidence["optimizer_step_before"] == 0
    assert evidence["optimizer_step_after"] == 1
    assert observed["checkpoint_step"] == 1
    assert evidence["checkpoint_file_count"] == 1
    assert len(evidence["checkpoint_sha256"]) == 64
    assert evidence["verl_provenance_before"] == evidence["verl_provenance_after"]
    assert evidence["verl_provenance_after"]["actual_sha"] == "e" * 40
    assert observed["verl_provenance_calls"] == 2
    assert observed["workers_closed"] is True
    evidence.commit()


def test_concrete_trainer_worker_close_failure_does_not_mask_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from capx.rl.capsule import server_factory

    run_id = "run-01"
    guided_artifact = tmp_path / "gate05_guided_group.json"
    guided_payload = _guided_payload(run_id)
    guided_artifact.write_text(json.dumps(guided_payload), encoding="utf-8")
    task = TaskInstanceV1.from_dict(guided_payload["task_instance"])
    primary_error = ValueError("trainer body failed")
    observed = {"closed": False}

    class _Workers:
        def verl_provenance(self) -> dict[str, object]:
            raise primary_error

        def close(self) -> None:
            observed["closed"] = True
            raise RuntimeError("trainer workers close failed")

    monkeypatch.setattr(server_factory, "start_verl_workers", lambda _config: _Workers())
    output_dir = tmp_path / "outputs"
    runtime = server_adapter.ConcreteGateRuntime(
        {
            "runtime": {
                "project_root": str(tmp_path),
                "output_dir": str(output_dir),
            }
        }
    )
    monkeypatch.setattr(runtime, "_task_for_seed", lambda _seed: task)

    with pytest.raises(ValueError, match="trainer body failed") as caught:
        runtime.trainer(
            1,
            (0, 0, 0, 0, 0, 0, 0, 1),
            guided_artifact,
            run_id=run_id,
        )

    assert caught.value is primary_error
    assert isinstance(caught.value.cleanup_error, RuntimeError)
    assert str(caught.value.cleanup_error) == "trainer workers close failed"
    assert observed["closed"] is True
    claim_root = output_dir / "gate06" / server_adapter._checkpoint_run_slug(run_id)
    assert not claim_root.exists()


def test_concrete_trainer_refuses_existing_checkpoint_before_starting_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capx.rl.capsule import server_factory

    run_id = "same-run"
    payload = _guided_payload(run_id)
    guided_artifact = tmp_path / "gate05_guided_group.json"
    guided_artifact.write_text(json.dumps(payload), encoding="utf-8")
    task = TaskInstanceV1.from_dict(payload["task_instance"])
    output_dir = tmp_path / "outputs"
    checkpoint = (
        output_dir
        / "gate06"
        / server_adapter._checkpoint_run_slug(run_id)
        / "global_step_1"
        / "actor"
    )
    checkpoint.mkdir(parents=True)
    runtime = server_adapter.ConcreteGateRuntime(
        {
            "task": {
                "environment": task.environment,
                "api": task.api,
                "privilege": task.privilege,
            },
            "runtime": {"project_root": str(tmp_path), "output_dir": str(output_dir)},
        }
    )
    monkeypatch.setattr(runtime, "_tasks", lambda: (task,))
    monkeypatch.setattr(
        server_factory,
        "start_verl_workers",
        lambda _config: (_ for _ in ()).throw(
            AssertionError("workers must not start for an existing checkpoint")
        ),
    )

    with pytest.raises(FileExistsError, match="exist"):
        runtime.trainer(
            1,
            (0, 0, 0, 0, 0, 0, 0, 1),
            guided_artifact,
            run_id=run_id,
        )
