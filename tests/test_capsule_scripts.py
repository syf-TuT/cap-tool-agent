from __future__ import annotations

import json
import builtins
import hashlib
from copy import deepcopy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from capx.rl.capsule.group import RepairAttempt, deterministic_group_uid
from capx.rl.capsule import main_ppo
from capx.rl.capsule.repair import BaseUnitSpan, RepairDraft
from capx.rl.capsule.schema import (
    LearningGroupV1,
    LearningMemberV1,
    ProgramReplayResultV1,
    ReplayOutcome,
    TaskInstanceV1,
)
from scripts.capsule_rl import (
    analyze_artifacts,
    build_verified_group,
    check_seed_determinism,
    common,
    controller_collector_smoke,
    one_step_trainer_smoke,
    oracle_clean_replay,
    materialize_resolved_dataset,
    prepare_dataset_config,
    server_adapter,
    server_preflight,
)


ENTRYPOINTS = (
    "prepare_dataset_config.py",
    "materialize_resolved_dataset.py",
    "server_preflight.py",
    "check_seed_determinism.py",
    "oracle_clean_replay.py",
    "controller_collector_smoke.py",
    "build_verified_group.py",
    "one_step_trainer_smoke.py",
    "server_adapter.py",
    "analyze_artifacts.py",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_TEMPLATE = (
    REPOSITORY_ROOT
    / "env_configs"
    / "cube_stack"
    / "capsule_rl"
    / "franka_robosuite_cube_stack_capsule_critique_grpo.yaml"
)
CLEAN_REPLAY_CONFIG = (
    REPOSITORY_ROOT
    / "env_configs"
    / "cube_stack"
    / "capsule_rl"
    / "franka_robosuite_cube_stack_privileged_clean_replay.yaml"
)


def _minimal_resolved_verl_config() -> dict[str, object]:
    return {
        "actor_rollout_ref": {
            "model": {},
            "rollout": {},
            "actor": {
                "strategy": "fsdp",
                "optim": {},
                "policy_loss": {},
            },
        },
        "algorithm": {},
        "reward_model": {},
        "trainer": {
            "total_epochs": 1,
            "n_gpus_per_node": 1,
            "nnodes": 1,
            "device": "cuda",
        },
        "ray_kwargs": {"ray_init": {}},
        "data": {},
    }


def _server_config(tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    project_root.mkdir()
    verl = project_root / "verl"
    verl.mkdir()
    model = project_root / "program-model"
    model.mkdir()
    dataset = project_root / "dataset.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": "cube-stack",
                "task_instance_id": "cube-stack:seed-5",
                "environment_seed": 5,
                "prompt": "stack",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    resolved_verl_config = project_root / "resolved_verl_ppo.yaml"
    resolved_verl_config.write_text(
        yaml.safe_dump(_minimal_resolved_verl_config(), sort_keys=False),
        encoding="utf-8",
    )
    output = project_root / "outputs" / "capsule"
    config = yaml.safe_load(CONFIG_TEMPLATE.read_text(encoding="utf-8"))
    config["runtime"].update(
        {
            "project_root": str(project_root),
            "python_executable": sys.executable,
            "verl_source_path": str(verl),
            "output_dir": str(output),
            "dataset_path": str(dataset),
            "program_model_path": str(model),
            "verl_resolved_config_path": str(resolved_verl_config),
        }
    )
    config["task"]["config_path"] = str(CLEAN_REPLAY_CONFIG)
    config["program_service"].update(
        {
            "endpoint": "http://127.0.0.1:8000/v1",
            "model": "program-actor",
            "api_key_env": "PROGRAM_API_KEY",
        }
    )
    config["controller_service"].update(
        {
            "endpoint": "http://127.0.0.1:8001/v1",
            "model": "frozen-controller",
            "api_key_env": "CONTROLLER_API_KEY",
        }
    )
    path = tmp_path / "capsule.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_all_server_entrypoints_exist_and_advertise_safe_validation_mode() -> None:
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts" / "capsule_rl"

    for filename in ENTRYPOINTS:
        source = (scripts_dir / filename).read_text(encoding="utf-8")
        assert "--validate-only" in source or "--dry-run" in source
        assert "if __name__ == \"__main__\"" in source


def test_server_config_validation_checks_algorithm_and_runtime_invariants(tmp_path: Path) -> None:
    config_path = _server_config(tmp_path)

    loaded = common.load_and_validate_server_config(config_path, check_runtime_paths=True)

    assert loaded["capsule"]["group_size"] == 8
    assert loaded["controller_service"]["frozen"] is True

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["task"]["render"] = True
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(common.ConfigValidationError, match="render"):
        common.load_and_validate_server_config(config_path, check_runtime_paths=True)


def test_server_config_bytes_loader_validates_the_supplied_snapshot(tmp_path: Path) -> None:
    config_path = _server_config(tmp_path)
    original_bytes = config_path.read_bytes()
    config_path.write_bytes(b"schema_version: 999\n")

    loaded = common.load_and_validate_server_config_bytes(
        original_bytes,
        check_runtime_paths=True,
    )

    assert loaded["schema_version"] == 1
    assert hashlib.sha256(original_bytes).hexdigest() != common.artifact_file_sha256(
        config_path
    )


def test_server_config_requires_existing_resolved_verl_config(tmp_path: Path) -> None:
    config_path = _server_config(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["runtime"].pop("verl_resolved_config_path")
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(common.ConfigValidationError, match="verl_resolved_config_path"):
        common.load_and_validate_server_config(config_path, check_runtime_paths=True)


@pytest.mark.parametrize(
    "missing_path",
    [
        ("actor_rollout_ref",),
        ("actor_rollout_ref", "model"),
        ("actor_rollout_ref", "rollout"),
        ("actor_rollout_ref", "actor"),
        ("actor_rollout_ref", "actor", "optim"),
        ("actor_rollout_ref", "actor", "policy_loss"),
        ("algorithm",),
        ("reward_model",),
        ("trainer",),
        ("ray_kwargs",),
        ("data",),
    ],
)
def test_server_config_rejects_resolved_verl_missing_required_tree(
    missing_path: tuple[str, ...], tmp_path: Path
) -> None:
    config_path = _server_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    resolved_path = Path(config["runtime"]["verl_resolved_config_path"])
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    parent = resolved
    for key in missing_path[:-1]:
        parent = parent[key]
    parent.pop(missing_path[-1])
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")

    with pytest.raises(common.ConfigValidationError, match="\\.".join(missing_path)):
        common.load_and_validate_server_config(config_path, check_runtime_paths=True)


@pytest.mark.parametrize(
    ("field_path", "bad_value", "message"),
    [
        (("actor_rollout_ref", "model"), [], "mapping"),
        (("actor_rollout_ref", "rollout"), None, "mapping"),
        (("actor_rollout_ref", "actor", "optim"), 1, "mapping"),
        (("actor_rollout_ref", "actor", "policy_loss"), "loss", "mapping"),
        (("actor_rollout_ref", "actor", "strategy"), "megatron", "FSDP"),
        (("algorithm",), [], "mapping"),
        (("reward_model",), None, "mapping"),
        (("trainer",), [], "mapping"),
        (("trainer", "total_epochs"), 0, "positive integer"),
        (("trainer", "n_gpus_per_node"), True, "positive integer"),
        (("trainer", "nnodes"), -1, "positive integer"),
        (("trainer", "device"), "", "non-empty string"),
        (("trainer", "n_gpus_per_node"), 3, "divisible"),
        (("ray_kwargs",), [], "mapping"),
        (("ray_kwargs", "ray_init"), [], "mapping"),
        (("data",), [], "mapping"),
    ],
)
def test_server_config_rejects_invalid_resolved_verl_scalars_and_shapes(
    field_path: tuple[str, ...],
    bad_value: object,
    message: str,
    tmp_path: Path,
) -> None:
    config_path = _server_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    resolved_path = Path(config["runtime"]["verl_resolved_config_path"])
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    parent = resolved
    for key in field_path[:-1]:
        parent = parent[key]
    parent[field_path[-1]] = bad_value
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")

    with pytest.raises(common.ConfigValidationError, match=message):
        common.load_and_validate_server_config(config_path, check_runtime_paths=True)


def test_server_config_validates_resolved_verl_without_torch_or_omegaconf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _server_config(tmp_path)
    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("static VeRL config validation must not import torch")
        if name == "omegaconf" or name.startswith("omegaconf."):
            raise AssertionError("static VeRL config validation must not import OmegaConf")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    common.load_and_validate_server_config(config_path, check_runtime_paths=True)


def test_preflight_hashes_resolved_verl_config_without_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _server_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project_root = Path(config["runtime"]["project_root"])
    (project_root / "uv.lock").write_text("lock", encoding="utf-8")
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PROGRAM_API_KEY", "present")
    monkeypatch.setenv("CONTROLLER_API_KEY", "present")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setattr(server_preflight, "_endpoint_open", lambda _endpoint: True)

    def fake_git_sha(path: Path) -> str:
        if path == Path(config["runtime"]["verl_source_path"]):
            return config["runtime"]["verl_pinned_sha"]
        return "d" * 40

    monkeypatch.setattr(server_preflight, "_git_sha", fake_git_sha)
    artifact = tmp_path / "gate01_preflight.json"

    payload = server_preflight.run_preflight(
        config_path, artifact, run_id="capsule-smoke-001"
    )

    assert payload["checks"]["verl_resolved_config_sha256"] == common.artifact_file_sha256(
        config["runtime"]["verl_resolved_config_path"]
    )
    assert payload["verl_resolved_config_sha256"] == payload["checks"][
        "verl_resolved_config_sha256"
    ]
    assert payload["resolved_environment_sha256"] == payload["checks"][
        "resolved_environment_sha256"
    ]
    dataset_path = Path(config["runtime"]["dataset_path"])
    assert payload["dataset_sha256"] == common.artifact_file_sha256(dataset_path)
    assert payload["checks"]["dataset_sha256"] == payload["dataset_sha256"]
    assert payload["checks"]["dataset_path"] == str(dataset_path.resolve())
    assert payload["checks"]["dataset_task_count"] == 1
    assert payload["checks"]["dataset_task_identities"] == [
        {"task_id": "cube-stack", "environment_seed": 5}
    ]
    assert payload["execution_mode"] == common.CANONICAL_EXECUTION_MODE
    common.verify_preflight_gate_artifact(payload)
    with pytest.raises(FileExistsError, match="artifact already exists"):
        server_preflight.run_preflight(config_path, artifact, run_id="capsule-smoke-001")


def test_preflight_failure_is_published_separately_without_claiming_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _server_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr(server_preflight, "_endpoint_open", lambda _endpoint: True)
    monkeypatch.setattr(
        server_preflight,
        "_git_sha",
        lambda path: (
            config["runtime"]["verl_pinned_sha"]
            if path == Path(config["runtime"]["verl_source_path"])
            else "d" * 40
        ),
    )
    artifact = tmp_path / "gate01_preflight.json"

    with pytest.raises(common.GateArtifactError, match="preflight failed"):
        server_preflight.run_preflight(
            config_path, artifact, run_id="capsule-smoke-001"
        )

    assert not artifact.exists()
    failure_path = common.gate_failure_artifact_path(artifact)
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["gate"] == "preflight"
    assert failure["passed"] is False
    assert failure["exception"]["stage"] == "required_checks"
    assert failure["config_sha256"] == common.artifact_file_sha256(config_path)
    assert failure["dataset_sha256"] == common.artifact_file_sha256(
        config["runtime"]["dataset_path"]
    )
    original = failure_path.read_bytes()
    with pytest.raises(FileExistsError, match="failure artifact already exists"):
        server_preflight.run_preflight(
            config_path, artifact, run_id="capsule-smoke-001"
        )
    assert failure_path.read_bytes() == original


def test_preflight_typed_verification_precedes_success_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _server_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project_root = Path(config["runtime"]["project_root"])
    (project_root / "uv.lock").write_text("lock", encoding="utf-8")
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PROGRAM_API_KEY", "present")
    monkeypatch.setenv("CONTROLLER_API_KEY", "present")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setattr(server_preflight, "_endpoint_open", lambda _endpoint: True)
    monkeypatch.setattr(
        server_preflight,
        "_git_sha",
        lambda path: (
            config["runtime"]["verl_pinned_sha"]
            if path == Path(config["runtime"]["verl_source_path"])
            else "d" * 40
        ),
    )
    def reject_typed_evidence(_payload: object) -> None:
        raise common.GateArtifactError("typed evidence rejected")

    monkeypatch.setattr(
        server_preflight,
        "verify_preflight_gate_artifact",
        reject_typed_evidence,
    )
    artifact = tmp_path / "gate01_preflight.json"

    with pytest.raises(common.GateArtifactError, match="typed evidence rejected"):
        server_preflight.run_preflight(
            config_path, artifact, run_id="capsule-smoke-001"
        )

    assert not artifact.exists()
    failure = json.loads(
        common.gate_failure_artifact_path(artifact).read_text(encoding="utf-8")
    )
    assert failure["exception"]["stage"] == "artifact_verification"


def _prepare_successful_preflight(
    config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project_root = Path(config["runtime"]["project_root"])
    (project_root / "uv.lock").write_text("lock", encoding="utf-8")
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PROGRAM_API_KEY", "present")
    monkeypatch.setenv("CONTROLLER_API_KEY", "present")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setattr(server_preflight, "_endpoint_open", lambda _endpoint: True)
    return config


@pytest.mark.parametrize(
    ("repository", "mutation", "expected_stage"),
    [
        ("project", "sha", "post_project_git"),
        ("project", "dirty", "post_project_git"),
        ("verl", "sha", "post_verl_git"),
        ("verl", "dirty", "post_verl_git"),
    ],
)
def test_preflight_rechecks_git_sha_and_cleanliness_before_publication(
    repository: str,
    mutation: str,
    expected_stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _server_config(tmp_path)
    config = _prepare_successful_preflight(config_path, monkeypatch)
    project_root = Path(config["runtime"]["project_root"]).resolve()
    verl_root = Path(config["runtime"]["verl_source_path"]).resolve()
    calls = {"project": 0, "verl": 0}

    def changing_git_sha(path: Path) -> str:
        resolved = path.resolve()
        name = "verl" if resolved == verl_root else "project"
        assert resolved in {project_root, verl_root}
        calls[name] += 1
        baseline = (
            str(config["runtime"]["verl_pinned_sha"])
            if name == "verl"
            else "d" * 40
        )
        if name == repository and calls[name] == 2:
            if mutation == "dirty":
                raise common.GateArtifactError(f"Git checkout became dirty: {path}")
            return "e" * 40
        return baseline

    monkeypatch.setattr(server_preflight, "_git_sha", changing_git_sha)
    artifact = tmp_path / f"gate01_{repository}_{mutation}.json"

    with pytest.raises(common.GateArtifactError, match="Git"):
        server_preflight.run_preflight(config_path, artifact, run_id="capsule-smoke-001")

    assert not artifact.exists()
    failure = json.loads(
        common.gate_failure_artifact_path(artifact).read_text(encoding="utf-8")
    )
    assert failure["exception"]["stage"] == expected_stage
    assert calls[repository] == 2


@pytest.mark.parametrize(
    ("mutable_input", "expected_stage", "message"),
    [
        ("config", "post_config", "config"),
        ("dataset", "post_dataset", "dataset"),
        ("environment", "post_runtime_dependencies", "environment"),
        ("verl_config", "post_runtime_dependencies", "VeRL"),
    ],
)
def test_preflight_rechecks_mutable_inputs_after_typed_verification(
    mutable_input: str,
    expected_stage: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _server_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    environment_path = tmp_path / "environment.yaml"
    environment_path.write_text("task: CubeStack\n", encoding="utf-8")
    config["task"]["config_path"] = str(environment_path)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    config = _prepare_successful_preflight(config_path, monkeypatch)
    project_root = Path(config["runtime"]["project_root"]).resolve()
    verl_root = Path(config["runtime"]["verl_source_path"]).resolve()
    monkeypatch.setattr(
        server_preflight,
        "_git_sha",
        lambda path: (
            str(config["runtime"]["verl_pinned_sha"])
            if path.resolve() == verl_root
            else "d" * 40
        ),
    )
    assert project_root != verl_root
    targets = {
        "config": config_path,
        "dataset": Path(config["runtime"]["dataset_path"]),
        "environment": environment_path,
        "verl_config": Path(config["runtime"]["verl_resolved_config_path"]),
    }
    real_verifier = server_preflight.verify_preflight_gate_artifact

    def mutate_after_verification(payload: object) -> None:
        real_verifier(payload)
        with targets[mutable_input].open("ab") as stream:
            stream.write(b"\n# changed after typed verification\n")

    monkeypatch.setattr(
        server_preflight,
        "verify_preflight_gate_artifact",
        mutate_after_verification,
    )
    artifact = tmp_path / f"gate01_{mutable_input}.json"

    with pytest.raises(common.GateArtifactError, match=message):
        server_preflight.run_preflight(config_path, artifact, run_id="capsule-smoke-001")

    assert not artifact.exists()
    failure = json.loads(
        common.gate_failure_artifact_path(artifact).read_text(encoding="utf-8")
    )
    assert failure["exception"]["stage"] == expected_stage


def test_preflight_hashes_the_exact_bytes_given_to_the_config_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _server_config(tmp_path)
    config = _prepare_successful_preflight(config_path, monkeypatch)
    initial_bytes = config_path.read_bytes()
    observed: dict[str, bytes] = {}
    real_loader = common.load_and_validate_server_config_bytes

    def recording_loader(raw: bytes, *, check_runtime_paths: bool) -> dict[str, object]:
        observed["raw"] = raw
        return real_loader(raw, check_runtime_paths=check_runtime_paths)

    monkeypatch.setattr(
        server_preflight,
        "load_and_validate_server_config_bytes",
        recording_loader,
    )
    monkeypatch.setattr(
        server_preflight,
        "_git_sha",
        lambda path: (
            str(config["runtime"]["verl_pinned_sha"])
            if path.resolve() == Path(config["runtime"]["verl_source_path"]).resolve()
            else "d" * 40
        ),
    )

    payload = server_preflight.run_preflight(
        config_path,
        tmp_path / "gate01_snapshot.json",
        run_id="capsule-smoke-001",
    )

    assert observed["raw"] == initial_bytes
    assert payload["config_sha256"] == hashlib.sha256(observed["raw"]).hexdigest()


def test_preflight_validate_only_rejects_blank_run_id(tmp_path: Path) -> None:
    config_path = _server_config(tmp_path)

    with pytest.raises(ValueError, match="run_id must be non-empty"):
        server_preflight.main(
            [
                "--config",
                str(config_path),
                "--artifact",
                str(tmp_path / "preflight.json"),
                "--run-id",
                "   ",
                "--validate-only",
            ]
        )


def test_preflight_execute_config_failure_still_publishes_failure_evidence(
    tmp_path: Path,
) -> None:
    config_path = _server_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["runtime"].pop("dataset_path")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    artifact = tmp_path / "gate01_preflight.json"

    with pytest.raises(common.ConfigValidationError, match="dataset_path"):
        server_preflight.main(
            [
                "--config",
                str(config_path),
                "--artifact",
                str(artifact),
                "--run-id",
                "capsule-smoke-001",
            ]
        )

    assert not artifact.exists()
    failure = json.loads(
        common.gate_failure_artifact_path(artifact).read_text(encoding="utf-8")
    )
    assert failure["exception"]["stage"] == "config_load"
    assert failure["dataset_sha256"] is None


def test_external_gate_validate_only_expands_placeholders_without_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _server_config(tmp_path)
    artifact = tmp_path / "seed_gate.json"

    def forbidden_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("validate-only must not execute a subprocess")

    monkeypatch.setattr(common.subprocess, "run", forbidden_run)
    plan = common.ExternalGatePlan(
        gate_name="seed_determinism",
        config_path=config_path,
        artifact_path=artifact,
        runner_command=(
            f"{sys.executable} fake_runner.py --config {{config}} "
            "--seeds {seed_sequence} --output {artifact}"
        ),
        placeholders={"seed_sequence": "5,6,5"},
        required_placeholders=frozenset({"config", "seed_sequence", "artifact"}),
    )

    argv = common.run_external_gate(plan, validate_only=True)

    assert "5,6,5" in argv
    assert str(config_path.resolve()) in argv
    assert str(artifact.resolve()) in argv
    assert "VALIDATION ONLY" in capsys.readouterr().out
    assert not artifact.exists()


@pytest.mark.parametrize(
    ("entrypoint", "artifact_name", "subcommand", "expected_tail"),
    [
        (
            check_seed_determinism.main,
            "gate02_seed.json",
            "seed",
            ("--seeds", "5,6,5"),
        ),
        (
            oracle_clean_replay.main,
            "gate03_oracle.json",
            "oracle",
            ("--seed", "5", "--replays", "2"),
        ),
        (
            controller_collector_smoke.main,
            "gate04_collector.json",
            "collector",
            (
                "--p0-count",
                "2",
                "--trajectories",
                "2",
                "--max-turns",
                "12",
            ),
        ),
        (
            build_verified_group.main,
            "gate05_guided_group.json",
            "guided",
            (
                "--group-size",
                "8",
                "--base-count",
                "7",
                "--guided-count",
                "1",
                "--max-group-attempts",
                "20",
            ),
        ),
        (
            one_step_trainer_smoke.main,
            "gate06_trainer.json",
            "trainer",
            (
                "--optimizer-steps",
                "1",
                "--group-rewards",
                "0,0,0,0,0,0,0,1",
                "--guided-artifact",
                "{guided_artifact}",
            ),
        ),
    ],
)
def test_gate_wrapper_validate_only_expands_repository_server_adapter(
    entrypoint,
    artifact_name: str,
    subcommand: str,
    expected_tail: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _server_config(tmp_path)
    run_dir = tmp_path / "capsule-smoke-001"
    run_dir.mkdir()
    artifact = run_dir / artifact_name
    if entrypoint is one_step_trainer_smoke.main:
        guided_payload = _guided_gate_payload()
        guided_payload["run_id"] = run_dir.name
        guided_payload["config_sha256"] = common.artifact_file_sha256(config_path)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        project_root = Path(config["runtime"]["project_root"]).resolve()

        def clean_git_sha(root: Path) -> str:
            assert root == project_root
            return str(guided_payload["git_sha"])

        monkeypatch.setattr(
            server_adapter,
            "_git_sha",
            clean_git_sha,
        )
        (run_dir / "gate05_guided_group.json").write_text(
            json.dumps(guided_payload), encoding="utf-8"
        )

    def forbidden_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("wrapper validate-only must not execute the server adapter")

    monkeypatch.setattr(common.subprocess, "run", forbidden_run)

    assert (
        entrypoint(
            [
                "--config",
                str(config_path),
                "--artifact",
                str(artifact),
                "--validate-only",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    argv = payload["argv"]
    assert argv[:3] == [sys.executable, "-m", "scripts.capsule_rl.server_adapter"]
    assert argv[3:9] == [
        "--config",
        str(config_path.resolve()),
        "--artifact",
        str(artifact.resolve()),
        "--run-id",
        run_dir.name,
    ]
    subcommand_index = argv.index(subcommand)
    actual_tail = argv[subcommand_index + 1 :]
    resolved_expected_tail = [
        str(run_dir / "gate05_guided_group.json")
        if value == "{guided_artifact}"
        else value
        for value in expected_tail
    ]
    assert actual_tail == resolved_expected_tail
    assert not artifact.exists()


def test_trainer_wrapper_validate_only_rejects_missing_guided_dependency(
    tmp_path: Path,
) -> None:
    config_path = _server_config(tmp_path)
    run_dir = tmp_path / "capsule-smoke-missing-guided"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="guided artifact"):
        one_step_trainer_smoke.main(
            [
                "--config",
                str(config_path),
                "--artifact",
                str(run_dir / "gate06_trainer.json"),
                "--validate-only",
            ]
        )


def test_external_gate_rejects_missing_placeholder_and_shell_operators(tmp_path: Path) -> None:
    config_path = _server_config(tmp_path)
    base = dict(
        gate_name="collector",
        config_path=config_path,
        artifact_path=tmp_path / "collector.json",
        placeholders={},
        required_placeholders=frozenset({"config", "artifact"}),
    )
    with pytest.raises(common.CommandValidationError, match="placeholder"):
        common.run_external_gate(
            common.ExternalGatePlan(runner_command=f"{sys.executable} runner.py", **base),
            validate_only=True,
        )
    with pytest.raises(common.CommandValidationError, match="shell operator"):
        common.run_external_gate(
            common.ExternalGatePlan(
                runner_command=(
                    f"{sys.executable} runner.py --config {{config}} "
                    "--output {artifact} && echo unsafe"
                ),
                **base,
            ),
            validate_only=True,
        )


def test_gate_artifact_verifiers_enforce_seed_oracle_group_and_trainer_gates(
    tmp_path: Path,
) -> None:
    common.verify_seed_gate_artifact(
        {
            **_gate_envelope("seed"),
            "seeds": [5, 6, 5],
            "initial_state_sha256": ["a" * 64, "b" * 64, "a" * 64],
        }
    )
    oracle_result = _replay_result(
        program_sample_id="oracle-0", source="oracle = True\n", success=True
    )
    common.verify_oracle_gate_artifact(
        {
            **_gate_envelope("oracle_replay"),
            "direct_replay": True,
            "controller_used": False,
            "replays": [
                {
                    "result": oracle_result.to_dict(),
                    "worker_id": "worker-1",
                    "reset_seed": 5,
                    "namespace_fresh": True,
                    "api_state_cleared": True,
                    "watchdog_active": True,
                },
                {
                    "result": oracle_result.to_dict(),
                    "worker_id": "worker-1",
                    "reset_seed": 5,
                    "namespace_fresh": True,
                    "api_state_cleared": True,
                    "watchdog_active": True,
                },
            ],
            "replay_event_count": 2,
            "attempt_event_count": 2,
            "retry_count": 0,
            "infra_failures": 0,
            "evaluator_failures": 0,
            "worker_replacements": 0,
        }
    )
    repair_records = []
    collector_selected_results = [
        _replay_result(
            program_sample_id=f"base-{index}",
            source=f"failed_{index} = True\n",
            success=False,
        )
        for index in range(7)
    ]
    collector_base_results = [result.to_dict() for result in collector_selected_results[:2]]
    for p0_rank in range(2):
        source = f"failed_{p0_rank} = True\n"
        for trajectory_index in range(2):
            draft = RepairDraft(
                task_id="cube-stack-5",
                environment_seed=5,
                program_sample_id=f"base-{p0_rank}",
                repair_trajectory_id=f"repair-{p0_rank}-{trajectory_index}",
                base_source=source,
                base_units=[BaseUnitSpan("whole", 0, len(source), source)],
            )
            draft.submit({"action": "finish", "rationale": "complete"})
            repair_records.append(
                {
                    "p0_rank": p0_rank,
                    "trajectory_index": trajectory_index,
                    "trace": draft.to_trace().to_dict(),
                }
            )
    common.verify_collector_gate_artifact(
        {
            **_gate_envelope("collector"),
            "controller_frozen": True,
            "intermediate_replay_count": 0,
            "p0_count": 2,
            "repair_trajectories_per_p0": 2,
            "base_results": collector_base_results,
            "selected_batch_index": 0,
            "selected_batch_results": [
                result.to_dict() for result in collector_selected_results
            ],
            "discarded_batches": [],
            "replay_events": [
                {
                    "batch_index": 0,
                    "base_index": index,
                    "selected_batch": True,
                    "result": result.to_dict(),
                }
                for index, result in enumerate(collector_selected_results)
            ],
            "replay_event_count": 7,
            "attempt_event_count": 7,
            "retry_count": 0,
            "infra_failures": 0,
            "evaluator_failures": 0,
            "worker_replacements": 0,
            "repair_traces": repair_records,
        }
    )
    group = _verified_group()
    guided_provenance = _guided_provenance(group)
    common.verify_guided_gate_artifact(
        {
            **_gate_envelope("guided"),
            "task_instance": _verified_task().to_dict(),
            "original_prompt": "stack the cubes",
            "training_input_contains_critique": False,
            "learning_group": group.to_dict(),
            **guided_provenance,
        }
    )
    checkpoint = tmp_path / "gate06" / "run" / "global_step_1" / "actor"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.bin").write_bytes(b"checkpoint")
    checkpoint_contract = _checkpoint_contract(checkpoint)
    common.verify_trainer_gate_artifact(
        {
            **_gate_envelope("trainer"),
            "learning_group": group.to_dict(),
            "actor_update_rpcs": 1,
            "optimizer_steps": 1,
            "optimizer_step_before": 0,
            "optimizer_step_after": 1,
            "gradient_norm": 0.25,
            "checkpoint": str(checkpoint),
            "group_rewards": [0, 0, 0, 0, 0, 0, 0, 1],
            "guided_token_mask_present": True,
            "guided_token_count": 4,
            "guided_mask_response_only": True,
            "rollout_is": False,
            "norm_adv_by_std_in_grpo": False,
            "loss_mode": "capsule_critique",
            "capsule_gamma": 0.1,
            "reference_kl_enabled": True,
            "reference_kl_coef": 0.001,
            "rollout_mode": "sync",
            "ppo_epochs": 1,
            "ppo_mini_batch_size": 8,
            "data_parallel_world_size": 1,
            "sequence_parallel_size": 1,
            "verl_provenance_before": _verl_provenance(),
            "verl_provenance_after": _verl_provenance(),
            "actor_update_skipped": False,
            "metrics": {"actor/pg_loss": 0.5},
            **checkpoint_contract,
            "guided_artifact_sha256": "e" * 64,
        }
    )

    with pytest.raises(common.GateArtifactError):
        common.verify_seed_gate_artifact(
            {
                **_gate_envelope("seed"),
                "seeds": [5, 6, 5],
                "initial_state_sha256": ["a" * 64] * 3,
            }
        )
    with pytest.raises(common.GateArtifactError):
        invalid_group = group.to_dict()
        invalid_group["group_uid"] = "invalid"
        common.verify_guided_gate_artifact(
            {
                **_gate_envelope("guided"),
                "learning_group": invalid_group,
            }
        )


def test_collector_verifier_rejects_empty_trace_records() -> None:
    with pytest.raises(common.GateArtifactError, match="trace"):
        common.verify_collector_gate_artifact(
            {
                **_gate_envelope("collector"),
                "controller_frozen": True,
                "intermediate_replay_count": 0,
                "p0_count": 2,
                "repair_trajectories_per_p0": 2,
                "base_results": [
                    _replay_result(
                        program_sample_id=f"base-{rank}",
                        source=f"failed_{rank} = True\n",
                        success=False,
                    ).to_dict()
                    for rank in range(2)
                ],
                "repair_traces": [
                    {"p0_rank": rank, "trajectory_index": trajectory}
                    for rank in range(2)
                    for trajectory in range(2)
                ],
            }
        )


def test_prepare_validate_only_does_not_write_dataset_or_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _server_config(tmp_path)
    source = tmp_path / "source.jsonl"
    source.write_text('{"task_id":"cube-stack","prompt":"stack cubes"}\n', encoding="utf-8")
    destination = tmp_path / "prepared"

    result = prepare_dataset_config.prepare(
        config_path=config_path,
        source_dataset=source,
        output_dir=destination,
        seeds=(5, 6),
        validate_only=True,
    )

    assert result.record_count == 2
    assert not destination.exists()
    assert "VALIDATION ONLY" in capsys.readouterr().out


def test_prepare_rejects_negative_seed_without_writing(tmp_path: Path) -> None:
    config_path = _server_config(tmp_path)
    source = tmp_path / "negative-seed-source.jsonl"
    source.write_text('{"task_id":"cube-stack","prompt":"stack cubes"}\n', encoding="utf-8")
    destination = tmp_path / "negative-seed-output"

    with pytest.raises(common.ConfigValidationError, match="non-negative"):
        prepare_dataset_config.prepare(
            config_path=config_path,
            source_dataset=source,
            output_dir=destination,
            seeds=(-1,),
            validate_only=True,
        )

    assert not destination.exists()


def test_prepare_rejects_duplicate_seeds_without_writing(tmp_path: Path) -> None:
    config_path = _server_config(tmp_path)
    source = tmp_path / "duplicate-seed-source.jsonl"
    source.write_text('{"task_id":"cube-stack","prompt":"stack cubes"}\n', encoding="utf-8")
    destination = tmp_path / "duplicate-seed-output"

    with pytest.raises(common.ConfigValidationError, match="duplicate"):
        prepare_dataset_config.prepare(
            config_path=config_path,
            source_dataset=source,
            output_dir=destination,
            seeds=(5, 5),
            validate_only=True,
        )

    assert not destination.exists()


def test_prepare_strips_untrusted_source_initial_state_hash(tmp_path: Path) -> None:
    config_path = _server_config(tmp_path)
    source = tmp_path / "source-with-fake-hash.jsonl"
    source.write_text(
        json.dumps(
            {
                "task_id": "cube-stack",
                "prompt": "stack cubes",
                "initial_state_sha256": "f" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "prepared-no-untrusted-hash"

    result = prepare_dataset_config.prepare(
        config_path=config_path,
        source_dataset=source,
        output_dir=destination,
        seeds=(5,),
        validate_only=False,
    )

    record = json.loads(result.dataset_path.read_text(encoding="utf-8"))
    assert "initial_state_sha256" not in record


def test_prepare_failure_never_publishes_a_partial_two_file_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _server_config(tmp_path)
    source = tmp_path / "atomic-source.jsonl"
    source.write_text('{"task_id":"cube-stack","prompt":"stack cubes"}\n', encoding="utf-8")
    destination = tmp_path / "atomic-prepared"
    original_write_text = Path.write_text

    def interrupt_resolved_config(self: Path, *args: object, **kwargs: object) -> int:
        if self.name == "capsule_rl.resolved.yaml":
            raise KeyboardInterrupt("interrupted between bundle writes")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", interrupt_resolved_config)

    with pytest.raises(KeyboardInterrupt, match="between bundle writes"):
        prepare_dataset_config.prepare(
            config_path=config_path,
            source_dataset=source,
            output_dir=destination,
            seeds=(5,),
            validate_only=False,
        )

    assert not destination.exists()
    assert list(tmp_path.glob(".atomic-prepared.staging-*")) == []


def _resolved_task() -> TaskInstanceV1:
    return TaskInstanceV1(
        task_id="cube-stack",
        environment_seed=5,
        prompt="stack",
        environment="robosuite_cube_stack",
        api="franka_control_privileged",
        privilege="privileged",
        initial_state_sha256="a" * 64,
    )


def _resolved_task_variant(**changes: object) -> TaskInstanceV1:
    payload = _resolved_task().to_dict()
    payload.update(changes)
    return TaskInstanceV1.from_dict(payload)


def _materialization_gate7_audit(
    config_path: Path, *, initial_state_sha256: str = "a" * 64
) -> Path:
    config = common.load_and_validate_server_config(
        config_path, check_runtime_paths=True
    )
    dependencies = common.runtime_dependency_hashes(config)
    audit_path = config_path.parent / f"{config_path.stem}.gate07_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "runtime_verified": True,
                "run_id": "capsule-smoke-001",
                "config_sha256": common.artifact_file_sha256(config_path),
                "dataset_sha256": common.artifact_file_sha256(
                    common.runtime_dataset_path(config)
                ),
                **dependencies,
                "typed_task_identities": [
                    {
                        "task_id": "cube-stack",
                        "environment_seed": 5,
                        "initial_state_sha256": initial_state_sha256,
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return audit_path


def test_materialize_validate_only_never_calls_state_resolver_or_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _server_config(tmp_path)
    destination = tmp_path / "seed-resolved"

    def forbidden_resolver(_config):
        raise AssertionError("validate-only must not resolve task state")

    result = materialize_resolved_dataset.materialize(
        config_path=config_path,
        gate7_audit_path=_materialization_gate7_audit(config_path),
        output_dir=destination,
        validate_only=True,
        task_resolver=forbidden_resolver,
    )

    assert result.dataset_path == destination / "capsule_rl.seed_resolved.dataset.jsonl"
    assert result.config_path == destination / "capsule_rl.seed_resolved.yaml"
    assert not destination.exists()
    assert "VALIDATION ONLY" in capsys.readouterr().out


def _overwrite_config_dataset(config_path: Path, rows: list[object]) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_path = Path(config["runtime"]["dataset_path"])
    dataset_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


def _unresolved_task_row(*, task_id: str = "cube-stack", seed: int = 5) -> dict[str, object]:
    return {
        "schema_version": 1,
        "task_id": task_id,
        "task_instance_id": f"{task_id}:seed-{seed}",
        "environment_seed": seed,
        "prompt": "stack",
    }


def test_materialize_validate_only_rejects_malformed_json_without_resolving(
    tmp_path: Path,
) -> None:
    config_path = _server_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    Path(config["runtime"]["dataset_path"]).write_text("{not-json\n", encoding="utf-8")
    destination = tmp_path / "malformed-json-output"

    def forbidden_resolver(_config):
        raise AssertionError("invalid validate-only input must not resolve task state")

    with pytest.raises(common.ConfigValidationError, match="line 1.*invalid JSON"):
        materialize_resolved_dataset.materialize(
            config_path=config_path,
            gate7_audit_path=_materialization_gate7_audit(config_path),
            output_dir=destination,
            validate_only=True,
            task_resolver=forbidden_resolver,
        )

    assert not destination.exists()


def test_materialize_validate_only_rejects_malformed_task_row_without_resolving(
    tmp_path: Path,
) -> None:
    config_path = _server_config(tmp_path)
    row = _unresolved_task_row()
    row["prompt"] = 123
    _overwrite_config_dataset(config_path, [row])
    destination = tmp_path / "malformed-task-output"

    def forbidden_resolver(_config):
        raise AssertionError("invalid validate-only input must not resolve task state")

    with pytest.raises(common.ConfigValidationError, match="line 1.*TaskInstanceV1"):
        materialize_resolved_dataset.materialize(
            config_path=config_path,
            gate7_audit_path=_materialization_gate7_audit(config_path),
            output_dir=destination,
            validate_only=True,
            task_resolver=forbidden_resolver,
        )

    assert not destination.exists()


def test_materialize_validate_only_rejects_negative_seed_without_resolving(
    tmp_path: Path,
) -> None:
    config_path = _server_config(tmp_path)
    _overwrite_config_dataset(config_path, [_unresolved_task_row(seed=-1)])
    destination = tmp_path / "negative-materialize-seed-output"

    def forbidden_resolver(_config):
        raise AssertionError("invalid validate-only input must not resolve task state")

    with pytest.raises(common.ConfigValidationError, match="environment_seed must be non-negative"):
        materialize_resolved_dataset.materialize(
            config_path=config_path,
            gate7_audit_path=_materialization_gate7_audit(config_path),
            output_dir=destination,
            validate_only=True,
            task_resolver=forbidden_resolver,
        )

    assert not destination.exists()


def test_materialize_validate_only_rejects_duplicate_task_identity_without_resolving(
    tmp_path: Path,
) -> None:
    config_path = _server_config(tmp_path)
    _overwrite_config_dataset(config_path, [_unresolved_task_row(), _unresolved_task_row()])
    destination = tmp_path / "duplicate-materialize-identity-output"

    def forbidden_resolver(_config):
        raise AssertionError("invalid validate-only input must not resolve task state")

    with pytest.raises(common.ConfigValidationError, match="duplicate task identity"):
        materialize_resolved_dataset.materialize(
            config_path=config_path,
            gate7_audit_path=_materialization_gate7_audit(config_path),
            output_dir=destination,
            validate_only=True,
            task_resolver=forbidden_resolver,
        )

    assert not destination.exists()


def test_materialize_publishes_typed_dataset_and_updated_config(tmp_path: Path) -> None:
    config_path = _server_config(tmp_path)
    destination = tmp_path / "seed-resolved"

    result = materialize_resolved_dataset.materialize(
        config_path=config_path,
        gate7_audit_path=_materialization_gate7_audit(config_path),
        output_dir=destination,
        validate_only=False,
        task_resolver=lambda _config: (_resolved_task(),),
    )

    records = [
        TaskInstanceV1.from_json(line)
        for line in result.dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records == [_resolved_task()]
    resolved_config = yaml.safe_load(result.config_path.read_text(encoding="utf-8"))
    assert resolved_config["runtime"]["dataset_path"] == str(result.dataset_path)
    assert resolved_config["runtime"]["bundle_manifest_path"] == str(
        result.manifest_path
    )
    assert resolved_config["runtime"]["gate7_audit_path"] == str(
        destination / "gate07_audit.json"
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["record_count"] == 1
    assert manifest["dataset_sha256"] == common.artifact_file_sha256(result.dataset_path)
    assert manifest["source_config_sha256"] == common.artifact_file_sha256(config_path)
    source_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert manifest["source_dataset_sha256"] == common.artifact_file_sha256(
        source_config["runtime"]["dataset_path"]
    )
    assert manifest["gate7_run_id"] == "capsule-smoke-001"
    assert manifest["gate7_audit_sha256"] == common.artifact_file_sha256(
        manifest["gate7_audit_path"]
    )
    assert manifest["gate7_typed_task_identities"] == [
        {
            "task_id": "cube-stack",
            "environment_seed": 5,
            "initial_state_sha256": "a" * 64,
        }
    ]
    assert manifest["output_dataset_sha256"] == common.artifact_file_sha256(
        result.dataset_path
    )
    assert manifest["output_config_sha256"] == common.artifact_file_sha256(
        result.config_path
    )
    formal_config, formal_config_path = main_ppo.load_and_validate_config(
        result.config_path
    )
    verified_bundle = main_ppo.verify_bundle_provenance(
        formal_config, formal_config_path
    )
    assert verified_bundle["gate7_run_id"] == "capsule-smoke-001"


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("task_id", "replacement-task"),
        ("environment_seed", 6),
        ("prompt", "replacement prompt"),
        ("environment", "replacement_environment"),
        ("api", "replacement_api"),
        ("privilege", "replacement_privilege"),
        ("metadata", {"split": "replacement"}),
    ],
)
def test_materialize_rejects_same_count_resolver_immutable_field_substitution(
    field_name: str, changed_value: object, tmp_path: Path
) -> None:
    config_path = _server_config(tmp_path)

    with pytest.raises(
        common.ConfigValidationError,
        match=rf"resolved task 0.*immutable source field {field_name}",
    ):
        materialize_resolved_dataset.materialize(
            config_path=config_path,
            gate7_audit_path=_materialization_gate7_audit(config_path),
            output_dir=tmp_path / f"substituted-{field_name}",
            validate_only=False,
            task_resolver=lambda _config: (
                _resolved_task_variant(**{field_name: changed_value}),
            ),
        )


def test_materialize_rejects_same_count_resolver_row_reordering(tmp_path: Path) -> None:
    config_path = _server_config(tmp_path)
    _overwrite_config_dataset(
        config_path,
        [
            _unresolved_task_row(task_id="cube-stack", seed=5),
            {
                **_unresolved_task_row(task_id="cube-stack-second", seed=6),
                "prompt": "stack second",
                "metadata": {"split": "second"},
            },
        ],
    )
    first = _resolved_task()
    second = _resolved_task_variant(
        task_id="cube-stack-second",
        environment_seed=6,
        prompt="stack second",
        initial_state_sha256="b" * 64,
        metadata={"split": "second"},
    )

    with pytest.raises(
        common.ConfigValidationError, match="resolved task 0.*immutable source field"
    ):
        materialize_resolved_dataset.materialize(
            config_path=config_path,
            gate7_audit_path=_materialization_gate7_audit(config_path),
            output_dir=tmp_path / "reordered-rows",
            validate_only=False,
            task_resolver=lambda _config: (second, first),
        )


def test_materialize_preserves_source_row_order_and_immutable_fields(tmp_path: Path) -> None:
    config_path = _server_config(tmp_path)
    _overwrite_config_dataset(
        config_path,
        [
            {**_unresolved_task_row(), "metadata": {"split": "first"}},
            {
                **_unresolved_task_row(task_id="cube-stack-second", seed=6),
                "prompt": "stack second",
                "metadata": {"split": "second"},
            },
        ],
    )
    first = _resolved_task_variant(metadata={"split": "first"})
    second = _resolved_task_variant(
        task_id="cube-stack-second",
        environment_seed=6,
        prompt="stack second",
        initial_state_sha256="b" * 64,
        metadata={"split": "second"},
    )

    result = materialize_resolved_dataset.materialize(
        config_path=config_path,
        gate7_audit_path=_materialization_gate7_audit(config_path),
        output_dir=tmp_path / "ordered-rows",
        validate_only=False,
        task_resolver=lambda _config: (first, second),
    )

    assert [
        TaskInstanceV1.from_json(line)
        for line in result.dataset_path.read_text(encoding="utf-8").splitlines()
    ] == [first, second]


def test_materialize_rejects_change_to_source_real_initial_state_hash(
    tmp_path: Path,
) -> None:
    config_path = _server_config(tmp_path)
    source_row = _unresolved_task_row()
    source_row["initial_state_sha256"] = "c" * 64
    _overwrite_config_dataset(config_path, [source_row])

    with pytest.raises(
        common.ConfigValidationError, match="initial_state_sha256.*immutable source"
    ):
        materialize_resolved_dataset.materialize(
            config_path=config_path,
            gate7_audit_path=_materialization_gate7_audit(
                config_path, initial_state_sha256="c" * 64
            ),
            output_dir=tmp_path / "changed-real-state",
            validate_only=False,
            task_resolver=lambda _config: (_resolved_task(),),
        )


def test_materialize_accepts_unchanged_source_real_initial_state_hash(
    tmp_path: Path,
) -> None:
    config_path = _server_config(tmp_path)
    source_row = _unresolved_task_row()
    source_row["initial_state_sha256"] = "c" * 64
    _overwrite_config_dataset(config_path, [source_row])

    result = materialize_resolved_dataset.materialize(
        config_path=config_path,
        gate7_audit_path=_materialization_gate7_audit(
            config_path, initial_state_sha256="c" * 64
        ),
        output_dir=tmp_path / "preserved-real-state",
        validate_only=False,
        task_resolver=lambda _config: (
            _resolved_task_variant(initial_state_sha256="c" * 64),
        ),
    )

    assert TaskInstanceV1.from_json(
        result.dataset_path.read_text(encoding="utf-8").strip()
    ).initial_state_sha256 == "c" * 64


def test_materialize_allows_explicit_initial_state_placeholder_to_resolve(
    tmp_path: Path,
) -> None:
    config_path = _server_config(tmp_path)
    source_row = _unresolved_task_row()
    source_row["initial_state_sha256"] = "0" * 64
    _overwrite_config_dataset(config_path, [source_row])

    result = materialize_resolved_dataset.materialize(
        config_path=config_path,
        gate7_audit_path=_materialization_gate7_audit(config_path),
        output_dir=tmp_path / "resolved-explicit-placeholder",
        validate_only=False,
        task_resolver=lambda _config: (_resolved_task(),),
    )

    assert TaskInstanceV1.from_json(
        result.dataset_path.read_text(encoding="utf-8").strip()
    ).initial_state_sha256 == "a" * 64


def test_materialize_rejects_unresolved_initial_state_placeholder(
    tmp_path: Path,
) -> None:
    config_path = _server_config(tmp_path)

    with pytest.raises(
        common.ConfigValidationError, match="did not replace.*initial-state placeholder"
    ):
        materialize_resolved_dataset.materialize(
            config_path=config_path,
            gate7_audit_path=_materialization_gate7_audit(config_path),
            output_dir=tmp_path / "unresolved-placeholder",
            validate_only=False,
            task_resolver=lambda _config: (
                _resolved_task_variant(initial_state_sha256="0" * 64),
            ),
        )


def test_materialize_refuses_overwrite_before_resolving(tmp_path: Path) -> None:
    config_path = _server_config(tmp_path)
    destination = tmp_path / "seed-resolved"
    destination.mkdir()

    def forbidden_resolver(_config):
        raise AssertionError("existing output must fail before state resolution")

    with pytest.raises(FileExistsError, match="already exists"):
        materialize_resolved_dataset.materialize(
            config_path=config_path,
            gate7_audit_path=_materialization_gate7_audit(config_path),
            output_dir=destination,
            validate_only=False,
            task_resolver=forbidden_resolver,
        )


def test_materialize_resolver_failure_leaves_no_bundle(tmp_path: Path) -> None:
    config_path = _server_config(tmp_path)
    destination = tmp_path / "seed-resolved"

    def failed_resolver(_config):
        raise RuntimeError("reset failed")

    with pytest.raises(RuntimeError, match="reset failed"):
        materialize_resolved_dataset.materialize(
            config_path=config_path,
            gate7_audit_path=_materialization_gate7_audit(config_path),
            output_dir=destination,
            validate_only=False,
            task_resolver=failed_resolver,
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".seed-resolved.*.tmp"))


def test_materialize_rejects_gate7_audit_dataset_mismatch_before_resolving(
    tmp_path: Path,
) -> None:
    config_path = _server_config(tmp_path)
    audit_path = _materialization_gate7_audit(config_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["dataset_sha256"] = "0" * 64
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    def forbidden_resolver(_config):
        raise AssertionError("mismatched Gate7 audit must fail before resolution")

    with pytest.raises(common.ConfigValidationError, match="Gate7.*dataset SHA"):
        materialize_resolved_dataset.materialize(
            config_path=config_path,
            gate7_audit_path=audit_path,
            output_dir=tmp_path / "mismatched-audit",
            validate_only=False,
            task_resolver=forbidden_resolver,
        )


def test_materialize_base_exception_removes_only_owned_partial_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _server_config(tmp_path)
    destination = tmp_path / "interrupted-resolved"
    real_replace = materialize_resolved_dataset.os.replace
    replace_count = 0

    def interrupt_second_publish(source: object, target: object) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise KeyboardInterrupt("interrupted during bundle publication")
        real_replace(source, target)

    monkeypatch.setattr(
        materialize_resolved_dataset.os, "replace", interrupt_second_publish
    )

    with pytest.raises(KeyboardInterrupt, match="bundle publication"):
        materialize_resolved_dataset.materialize(
            config_path=config_path,
            gate7_audit_path=_materialization_gate7_audit(config_path),
            output_dir=destination,
            validate_only=False,
            task_resolver=lambda _config: (_resolved_task(),),
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".interrupted-resolved.*.tmp"))


def test_materialize_cleanup_refuses_concurrently_replaced_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _server_config(tmp_path)
    destination = tmp_path / "replaced-resolved"
    displaced_owner = tmp_path / "displaced-owned-directory"

    def replace_destination_then_interrupt(source: object, target: object) -> None:
        del source, target
        destination.rename(displaced_owner)
        destination.mkdir()
        (destination / "concurrent-owner.txt").write_text(
            "must survive", encoding="utf-8"
        )
        raise KeyboardInterrupt("destination replaced concurrently")

    monkeypatch.setattr(
        materialize_resolved_dataset.os,
        "replace",
        replace_destination_then_interrupt,
    )

    with pytest.raises(KeyboardInterrupt, match="replaced concurrently"):
        materialize_resolved_dataset.materialize(
            config_path=config_path,
            gate7_audit_path=_materialization_gate7_audit(config_path),
            output_dir=destination,
            validate_only=False,
            task_resolver=lambda _config: (_resolved_task(),),
        )

    assert (destination / "concurrent-owner.txt").read_text(encoding="utf-8") == (
        "must survive"
    )
    assert displaced_owner.is_dir()
    assert not list(tmp_path.glob(".replaced-resolved.*.tmp"))


def test_materialize_completed_publication_survives_temp_cleanup_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _server_config(tmp_path)
    destination = tmp_path / "cleanup-warning-resolved"
    real_rmtree = materialize_resolved_dataset.shutil.rmtree

    def fail_only_staging_cleanup(path: object, *args: object, **kwargs: object) -> None:
        if Path(path).name.startswith(".cleanup-warning-resolved."):
            raise OSError("temporary cleanup failed after publication")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(
        materialize_resolved_dataset.shutil, "rmtree", fail_only_staging_cleanup
    )

    result = materialize_resolved_dataset.materialize(
        config_path=config_path,
        gate7_audit_path=_materialization_gate7_audit(config_path),
        output_dir=destination,
        validate_only=False,
        task_resolver=lambda _config: (_resolved_task(),),
    )

    assert result.manifest_path.is_file()
    assert result.dataset_path.is_file()
    assert result.config_path.is_file()


def test_artifact_analyzer_summarizes_without_importing_runtime(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "group.json").write_text(
        json.dumps(
            {
                "artifact_type": "learning_group",
                "members": [
                    *[{"member_type": "base", "reward": 0.0} for _ in range(7)],
                    {"member_type": "critique_guided_revision", "reward": 1.0},
                ],
                "repair_attempts": [
                    {"pt_outcome": "success", "p_hat_outcome": "task_failure"},
                    {"pt_outcome": "success", "p_hat_outcome": "success"},
                ],
                "retry_count": 2,
                "infra_failures": 1,
            }
        ),
        encoding="utf-8",
    )

    summary = analyze_artifacts.analyze_directory(artifacts)

    assert summary["learning_groups"] == 1
    assert summary["base_members"] == 7
    assert summary["guided_members"] == 1
    assert summary["pt_successes"] == 2
    assert summary["p_hat_successes"] == 1
    assert summary["retry_count"] == 2
    assert summary["infra_failures"] == 1


def test_artifact_analyzer_counts_the_same_typed_group_only_once(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    learning_group = {
        "group_uid": "cube-stack-5:group-0",
        "members": [
            {"member_type": "base", "reward": 0.0},
            {"member_type": "critique_guided_revision", "reward": 1.0},
        ],
    }
    for name in ("gate05_guided_group.json", "gate06_trainer.json"):
        (artifacts / name).write_text(
            json.dumps({"learning_group": learning_group}),
            encoding="utf-8",
        )

    summary = analyze_artifacts.analyze_directory(artifacts)

    assert summary["learning_groups"] == 1
    assert summary["base_members"] == 1
    assert summary["guided_members"] == 1


def test_artifact_tree_rejects_a_symlink_used_as_the_root(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "state.bin").write_bytes(b"checkpoint")
    checkpoint_link = tmp_path / "checkpoint-link"
    checkpoint_link.symlink_to(checkpoint, target_is_directory=True)

    with pytest.raises(common.GateArtifactError, match="artifact path must not be a symlink"):
        common.artifact_tree_sha256(checkpoint_link)


def test_documentation_records_all_gates_and_runtime_not_verified_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs" / "capsule_rl.md").read_text(encoding="utf-8")

    for gate in (
        "Preflight",
        "Seed gate",
        "Oracle replay gate",
        "Collector gate",
        "Guided gate",
        "Trainer gate",
        "Result audit",
    ):
        assert gate in text
    for requirement in (
        "CONTROLLER_API_KEY",
        "Controller endpoint",
        "verl_resolved_config_path",
        "PyRoKi",
        "MUJOCO_GL=egl",
        "ProgramReplayResultV1",
        "LearningGroupV1",
        "config_sha256",
        "--run-id",
        "runtime_verified",
        "outputs/",
        "artifacts/",
        "runtime verified",
    ):
        assert requirement in text


def _gate_envelope(gate: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "gate": gate,
        "passed": True,
        "execution_mode": common.CANONICAL_EXECUTION_MODE,
        "run_id": "capsule-smoke-001",
        "config_sha256": "c" * 64,
        "git_sha": "d" * 40,
        "dataset_sha256": "9" * 64,
        "resolved_environment_sha256": "e" * 64,
        "verl_resolved_config_sha256": "f" * 64,
    }


def test_gate_envelope_requires_dataset_sha256() -> None:
    payload = {
        **_gate_envelope("seed"),
        "seeds": [5, 6, 5],
        "initial_state_sha256": ["a" * 64, "b" * 64, "a" * 64],
    }
    del payload["dataset_sha256"]

    with pytest.raises(common.GateArtifactError, match="dataset_sha256"):
        common.verify_seed_gate_artifact(payload)


@pytest.mark.parametrize(
    "field_name", ["resolved_environment_sha256", "verl_resolved_config_sha256"]
)
def test_gate_envelope_requires_runtime_dependency_sha256(field_name: str) -> None:
    payload = {
        **_gate_envelope("seed"),
        "seeds": [5, 6, 5],
        "initial_state_sha256": ["a" * 64, "b" * 64, "a" * 64],
    }
    del payload[field_name]

    with pytest.raises(common.GateArtifactError, match=field_name):
        common.verify_seed_gate_artifact(payload)


@pytest.mark.parametrize("invalid_version", [True, 1.0])
def test_gate_envelope_requires_exact_integer_schema_version(
    invalid_version: object,
) -> None:
    payload = {
        **_gate_envelope("seed"),
        "schema_version": invalid_version,
        "seeds": [5, 6, 5],
        "initial_state_sha256": ["a" * 64, "b" * 64, "a" * 64],
    }

    with pytest.raises(common.GateArtifactError, match="schema_version"):
        common.verify_seed_gate_artifact(payload)


def _replay_result(
    *,
    program_sample_id: str,
    source: str,
    success: bool,
    initial_state_sha256: str = "a" * 64,
) -> ProgramReplayResultV1:
    return ProgramReplayResultV1(
        task_id="cube-stack-5",
        environment_seed=5,
        program_sample_id=program_sample_id,
        source=source,
        initial_state_sha256=initial_state_sha256,
        outcome=ReplayOutcome.SUCCESS if success else ReplayOutcome.TASK_FAILURE,
        raw_reward=1.0 if success else 0.25,
        binary_reward=1.0 if success else 0.0,
        task_completed=success,
        diagnostics={
            "evaluator_attempt_history": [
                {
                    "attempt": 1,
                    "outcome": "success" if success else "task_failure",
                    "worker_replaced": False,
                    "retry_scheduled": False,
                    "error_type": None,
                    "error_message": None,
                }
            ],
            "reset_info": {
                "capsule_reset_evidence": {
                    "namespace_fresh": True,
                    "api_state_cleared": True,
                    "api_reset_count": 1,
                    "api_reset_confirmed_count": 1,
                }
            }
        },
    )


def test_replay_telemetry_is_derived_from_typed_results_and_rejects_forged_counts() -> None:
    recovered = ProgramReplayResultV1(
        task_id="cube-stack-5",
        environment_seed=5,
        program_sample_id="recovered",
        source="pass\n",
        initial_state_sha256="a" * 64,
        outcome=ReplayOutcome.TASK_FAILURE,
        raw_reward=0.2,
        binary_reward=0.0,
        task_completed=False,
        attempts=2,
        diagnostics={
            "evaluator_attempt_history": [
                {
                    "attempt": 1,
                    "outcome": "evaluator_error",
                    "worker_replaced": True,
                    "retry_scheduled": True,
                    "error_type": "MalformedPayloadError",
                    "error_message": "bad payload",
                },
                {
                    "attempt": 2,
                    "outcome": "task_failure",
                    "worker_replaced": False,
                    "retry_scheduled": False,
                    "error_type": None,
                    "error_message": None,
                },
            ]
        },
    )
    exhausted = ProgramReplayResultV1(
        task_id="cube-stack-5",
        environment_seed=5,
        program_sample_id="exhausted",
        source="pass\n",
        initial_state_sha256="a" * 64,
        outcome=ReplayOutcome.INFRA_ERROR,
        raw_reward=None,
        binary_reward=None,
        task_completed=False,
        attempts=3,
        error_type="WorkerCrashedError",
        error_message="worker remained poisoned",
        diagnostics={
            "evaluator_attempt_history": [
                {
                    "attempt": attempt,
                    "outcome": "infra_error",
                    "worker_replaced": True,
                    "retry_scheduled": attempt < 3,
                    "error_type": "WorkerCrashedError",
                    "error_message": "worker remained poisoned",
                }
                for attempt in range(1, 4)
            ]
        },
    )
    payload = {
        "replay_event_count": 2,
        "attempt_event_count": 5,
        "retry_count": 3,
        "infra_failures": 3,
        "evaluator_failures": 1,
        "worker_replacements": 4,
    }

    common.verify_replay_telemetry(payload, (recovered, exhausted))

    payload["infra_failures"] = 0
    with pytest.raises(common.GateArtifactError, match="telemetry"):
        common.verify_replay_telemetry(payload, (recovered, exhausted))


def test_guided_artifact_verifier_does_not_import_torch_or_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _guided_gate_payload()
    import capx.rl.capsule as capsule_package

    original_module = sys.modules.get("capx.rl.capsule.trainer")
    original_attribute = getattr(capsule_package, "trainer", None)
    sys.modules.pop("capx.rl.capsule.trainer", None)
    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object):
        if name.split(".", 1)[0] in {"torch", "numpy"}:
            raise AssertionError(f"static artifact verification imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    try:
        common.verify_guided_gate_artifact(payload)
    finally:
        if original_module is None:
            sys.modules.pop("capx.rl.capsule.trainer", None)
        else:
            sys.modules["capx.rl.capsule.trainer"] = original_module
        if original_attribute is not None:
            setattr(capsule_package, "trainer", original_attribute)


def _verified_task() -> TaskInstanceV1:
    return TaskInstanceV1(
        task_id="cube-stack-5",
        environment_seed=5,
        prompt="stack the cubes",
        environment="robosuite_cube_stack",
        api="franka_control_privileged",
        privilege="privileged",
        initial_state_sha256="a" * 64,
    )


def _verified_group() -> LearningGroupV1:
    task = _verified_task()
    prompt = task.prompt
    group_uid = deterministic_group_uid(task)
    members = tuple(
        [
            LearningMemberV1(
                member_type="base",
                program_sample_id=f"base-{index}",
                prompt=prompt,
                response=f"failed_{index} = True\n",
                reward=0.0,
            )
            for index in range(7)
        ]
        + [
            LearningMemberV1(
                member_type="critique_guided_revision",
                program_sample_id="guided-0",
                repair_trajectory_id=f"{group_uid}:p0-0:trajectory-0",
                prompt=prompt,
                response="success = True\n",
                reward=1.0,
            )
        ]
    )
    return LearningGroupV1(
        task_id=task.task_id,
        environment_seed=task.environment_seed,
        group_uid=group_uid,
        initial_state_sha256=task.initial_state_sha256,
        members=members,
    )


def _guided_provenance(group: LearningGroupV1) -> dict[str, object]:
    base_results = [
        _replay_result(
            program_sample_id=member.program_sample_id,
            source=member.response,
            success=False,
        )
        for member in group.members[:7]
    ]
    selected_trajectory_id = str(group.members[-1].repair_trajectory_id)
    draft = RepairDraft(
        task_id=group.task_id,
        environment_seed=group.environment_seed,
        program_sample_id=base_results[0].program_sample_id,
        repair_trajectory_id=selected_trajectory_id,
        base_source=base_results[0].source,
        base_units=[
            BaseUnitSpan(
                "whole",
                0,
                len(base_results[0].source),
                base_results[0].source,
            )
        ],
    )
    draft.submit(
        {
            "action": "append",
            "generation_id": "recovery-1",
            "unit_id": "whole",
            "source": "recovered = True\n",
            "rationale": "repair",
        }
    )
    draft.submit({"action": "finish", "rationale": "complete"})
    trace = draft.to_trace()
    pt_result = _replay_result(
        program_sample_id=f"{selected_trajectory_id}:pt",
        source=trace.final_source,
        success=True,
    )
    p_hat_result = _replay_result(
        program_sample_id=group.members[-1].program_sample_id,
        source=group.members[-1].response,
        success=True,
    )
    selected_replay_results = [*base_results, pt_result, p_hat_result]
    attempts: list[RepairAttempt] = []
    for p0_rank, p0_sample_id in enumerate(
        (base_results[0].program_sample_id, base_results[1].program_sample_id)
    ):
        for trajectory_index in range(2):
            trajectory_id = (
                f"{group.group_uid}:p0-{p0_rank}:trajectory-{trajectory_index}"
            )
            if (p0_rank, trajectory_index) == (0, 0):
                attempts.append(
                    RepairAttempt(
                        p0_rank=p0_rank,
                        trajectory_index=trajectory_index,
                        p0_program_sample_id=p0_sample_id,
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
                        repair_trajectory_id=trajectory_id,
                        status="rejected",
                        rejection_reason="collector_error",
                        rejection_message="mocked rejection",
                    )
                )
    return {
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
            for index, result in enumerate(selected_replay_results)
        ],
        "replay_event_count": len(selected_replay_results),
        "attempt_event_count": len(selected_replay_results),
        "retry_count": 0,
        "infra_failures": 0,
        "evaluator_failures": 0,
        "worker_replacements": 0,
    }


def _guided_gate_payload() -> dict[str, object]:
    group = _verified_group()
    return {
        **_gate_envelope("guided"),
        "task_instance": _verified_task().to_dict(),
        "original_prompt": "stack the cubes",
        "training_input_contains_critique": False,
        "learning_group": group.to_dict(),
        **_guided_provenance(group),
    }


def test_external_gate_execute_rejects_existing_artifact_before_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _server_config(tmp_path)
    artifact = tmp_path / "seed_gate.json"
    artifact.write_text('{"old":true}\n', encoding="utf-8")

    def forbidden_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("an existing artifact must fail before the runner starts")

    monkeypatch.setattr(common.subprocess, "run", forbidden_run)
    plan = common.ExternalGatePlan(
        gate_name="seed",
        config_path=config_path,
        artifact_path=artifact,
        runner_command=(
            f"{sys.executable} fake_runner.py --config {{config}} "
            "--seeds {seed_sequence} --output {artifact}"
        ),
        placeholders={"seed_sequence": "5,6,5"},
        required_placeholders=frozenset({"config", "seed_sequence", "artifact"}),
    )

    with pytest.raises(FileExistsError, match="artifact already exists"):
        common.run_external_gate(plan, validate_only=False)

    assert artifact.read_text(encoding="utf-8") == '{"old":true}\n'


def test_oracle_verifier_requires_typed_clean_replay_results() -> None:
    payload = {
        **_gate_envelope("oracle_replay"),
        "direct_replay": True,
        "controller_used": False,
        "replays": [
            {
                "outcome": "success",
                "raw_reward": 0.0,
                "truncated": True,
                "worker_id": "worker-1",
                "reset_seed": 5,
                "namespace_fresh": True,
                "api_state_cleared": True,
                "watchdog_active": True,
            }
        ]
        * 2,
    }

    with pytest.raises(common.GateArtifactError, match="ProgramReplayResultV1"):
        common.verify_oracle_gate_artifact(payload)


def test_oracle_verifier_rejects_top_level_reset_evidence_spoof() -> None:
    result = _replay_result(
        program_sample_id="oracle-0", source="oracle = True\n", success=True
    ).to_dict()
    result["diagnostics"] = {}
    payload = {
        **_gate_envelope("oracle_replay"),
        "direct_replay": True,
        "controller_used": False,
        "replays": [
            {
                "result": result,
                "worker_id": "worker-1",
                "reset_seed": 5,
                "namespace_fresh": True,
                "api_state_cleared": True,
                "watchdog_active": True,
            }
        ]
        * 2,
    }

    with pytest.raises(common.GateArtifactError, match="capsule_reset_evidence"):
        common.verify_oracle_gate_artifact(payload)


def test_guided_verifier_rejects_unlinked_string_only_success() -> None:
    group = _verified_group()
    payload = {
        **_gate_envelope("guided"),
        "task_instance": _verified_task().to_dict(),
        "learning_group": group.to_dict(),
        "original_prompt": "stack the cubes",
        "training_input_contains_critique": False,
        "base_results": [
            _replay_result(
                program_sample_id=member.program_sample_id,
                source=member.response,
                success=False,
            ).to_dict()
            for member in group.members[:7]
        ],
        "selected_repair": {"pt_outcome": "success", "p_hat_outcome": "success"},
    }

    with pytest.raises(common.GateArtifactError, match="repair_attempts|selected repair"):
        common.verify_guided_gate_artifact(payload)


def test_guided_verifier_requires_complete_fixed_2x2_repair_attempts() -> None:
    payload = _guided_gate_payload()
    payload["repair_attempts"] = payload["repair_attempts"][:-1]

    with pytest.raises(common.GateArtifactError, match="exactly 4 attempts"):
        common.verify_guided_gate_artifact(payload)


def test_guided_verifier_selects_first_success_in_fixed_attempt_order() -> None:
    payload = _guided_gate_payload()
    attempts = payload["repair_attempts"]
    first = attempts[0]
    first["selected"] = False
    second = deepcopy(first)
    second_trajectory_id = (
        f"{payload['learning_group']['group_uid']}:p0-0:trajectory-1"
    )
    second.update(
        {
            "trajectory_index": 1,
            "repair_trajectory_id": second_trajectory_id,
            "selected": True,
        }
    )
    second["trace"]["repair_trajectory_id"] = second_trajectory_id
    for event in [*second["trace"]["edits"], *second["trace"]["audits"]]:
        event["repair_trajectory_id"] = second_trajectory_id
    second["pt_result"]["program_sample_id"] = f"{second_trajectory_id}:pt"
    attempts[1] = second
    payload["learning_group"]["members"][-1][
        "repair_trajectory_id"
    ] = second_trajectory_id
    payload["selected_repair"] = {
        "p0_rank": 0,
        "trajectory_index": 1,
        "trace": deepcopy(second["trace"]),
        "p0_result": deepcopy(payload["base_results"][0]),
        "pt_result": deepcopy(second["pt_result"]),
        "p_hat_result": deepcopy(second["revision_result"]),
    }

    with pytest.raises(common.GateArtifactError, match="first successful"):
        common.verify_guided_gate_artifact(payload)


def test_guided_verifier_enforces_deterministic_p0_selection() -> None:
    payload = _guided_gate_payload()
    for attempt in payload["repair_attempts"][2:]:
        attempt["p0_program_sample_id"] = "base-2"

    with pytest.raises(common.GateArtifactError, match="deterministic P0 selection"):
        common.verify_guided_gate_artifact(payload)


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("loss_mode", "vanilla", "capsule_critique"),
        ("capsule_gamma", 0.2, "capsule_gamma"),
        ("reference_kl_enabled", False, "reference KL"),
        ("reference_kl_coef", 0.0, "reference_kl_coef"),
        ("rollout_mode", "async", "synchronous"),
        ("ppo_epochs", 2, "ppo_epochs"),
        ("ppo_mini_batch_size", 4, "ppo_mini_batch_size"),
        ("data_parallel_world_size", 3, "data_parallel_world_size"),
        ("sequence_parallel_size", 2, "sequence_parallel_size"),
        ("actor_update_rpcs", 2, "actor update RPC"),
        ("optimizer_step_after", 2, "step delta"),
    ],
)
def test_trainer_verifier_requires_capsule_loss_contract(
    field_name: str,
    invalid_value: object,
    message: str,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    payload = _complete_gate_payloads(checkpoint)["trainer"]
    payload[field_name] = invalid_value

    with pytest.raises(common.GateArtifactError, match=message):
        common.verify_trainer_gate_artifact(payload)


def test_server_config_validator_reuses_canonical_training_contract(tmp_path: Path) -> None:
    config_path = _server_config(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["controller_service"]["endpoint"] = payload["program_service"]["endpoint"]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(common.ConfigValidationError, match="separate endpoints"):
        common.load_and_validate_server_config(config_path, check_runtime_paths=True)


def test_main_validate_import_path_does_not_import_torch() -> None:
    code = """
import builtins
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise RuntimeError('torch import is forbidden on validate-only import path')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import capx.rl.capsule.main_ppo
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_gate7_requires_all_six_verified_artifacts(tmp_path: Path) -> None:
    assert hasattr(analyze_artifacts, "audit_gate_directory")
    with pytest.raises(common.GateArtifactError, match="missing gate artifact"):
        analyze_artifacts.audit_gate_directory(tmp_path)


def test_gate7_validate_only_audits_inputs_without_writing(tmp_path: Path) -> None:
    with pytest.raises(common.GateArtifactError, match="missing gate artifact"):
        analyze_artifacts.main(
            [
                "--input-dir",
                str(tmp_path),
                "--output-json",
                str(tmp_path / "summary.json"),
                "--output-report",
                str(tmp_path / "report.md"),
                "--validate-only",
            ]
        )

    assert not (tmp_path / "summary.json").exists()
    assert not (tmp_path / "report.md").exists()


def _checkpoint_contract(checkpoint: Path) -> dict[str, object]:
    file_count = common.artifact_tree_file_count(checkpoint)
    sha256 = common.artifact_tree_sha256(checkpoint)
    manifest = checkpoint.parent / "checkpoint_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_file_count": file_count,
                "checkpoint_sha256": sha256,
                "optimizer_step_before": 0,
                "optimizer_step_after": 1,
                "optimizer_step_delta": 1,
            }
        ),
        encoding="utf-8",
    )
    return {
        "checkpoint_file_count": file_count,
        "checkpoint_sha256": sha256,
        "checkpoint_manifest": str(manifest.resolve()),
    }


def _verl_provenance() -> dict[str, object]:
    return {
        "source_path": "/pinned/verl-source",
        "expected_sha": "c" * 40,
        "actual_sha": "c" * 40,
        "clean": True,
        "worker_count": 1,
        "worker_ranks": [0],
        "worker_module_paths": ["/pinned/verl-source/verl/__init__.py"],
    }


def _complete_gate_payloads(checkpoint: Path) -> dict[str, dict[str, object]]:
    if checkpoint.is_dir() and not any(checkpoint.iterdir()):
        (checkpoint / "state.bin").write_bytes(b"checkpoint")
    checkpoint_contract = _checkpoint_contract(checkpoint)
    oracle_result = _replay_result(
        program_sample_id="oracle-0", source="oracle = True\n", success=True
    )
    oracle_records = [
        {
            "result": oracle_result.to_dict(),
            "worker_id": "worker-1",
            "reset_seed": 5,
            "namespace_fresh": True,
            "api_state_cleared": True,
            "watchdog_active": True,
        }
        for _ in range(2)
    ]

    collector_selected_batch_results = [
        _replay_result(
            program_sample_id=f"collector-base-{rank}",
            source=f"collector_failed_{rank} = True\n",
            success=False,
        )
        for rank in range(7)
    ]
    collector_base_results = collector_selected_batch_results[:2]
    collector_records = []
    for rank, result in enumerate(collector_base_results):
        for trajectory_index in range(2):
            draft = RepairDraft(
                task_id=result.task_id,
                environment_seed=result.environment_seed,
                program_sample_id=result.program_sample_id,
                repair_trajectory_id=f"collector-repair-{rank}-{trajectory_index}",
                base_source=result.source,
                base_units=[BaseUnitSpan("whole", 0, len(result.source), result.source)],
            )
            draft.submit({"action": "finish", "rationale": "complete"})
            collector_records.append(
                {
                    "p0_rank": rank,
                    "trajectory_index": trajectory_index,
                    "trace": draft.to_trace().to_dict(),
                }
            )

    group = _verified_group()
    guided_provenance = _guided_provenance(group)

    return {
        "preflight": {
            **_gate_envelope("preflight"),
            "failed_checks": [],
            "checks": {
                "git_sha": "d" * 40,
                "verl_source_path": "/pinned/verl-source",
                "verl_expected_sha": "c" * 40,
                "verl_actual_sha": "c" * 40,
                "verl_sha_matches": True,
                "dependency_lock_present": True,
                "cuda_available": True,
                "egl_configured": True,
                "program_model_exists": True,
                "program_api_key_present": True,
                "controller_api_key_present": True,
                "program_endpoint_ready": True,
                "controller_endpoint_ready": True,
                "pyroki_endpoint_ready": True,
                "resolved_environment_sha256": "e" * 64,
                "verl_resolved_config_sha256": "f" * 64,
                "dataset_path": str((checkpoint.parent / "dataset.jsonl").resolve()),
                "dataset_sha256": "9" * 64,
                "dataset_task_count": 1,
                "dataset_task_identities": [
                    {"task_id": "cube-stack-5", "environment_seed": 5}
                ],
            },
        },
        "seed": {
            **_gate_envelope("seed"),
            "seeds": [5, 6, 5],
            "initial_state_sha256": ["a" * 64, "b" * 64, "a" * 64],
        },
        "oracle_replay": {
            **_gate_envelope("oracle_replay"),
            "direct_replay": True,
            "controller_used": False,
            "replays": oracle_records,
            "replay_event_count": 2,
            "attempt_event_count": 2,
            "retry_count": 0,
            "infra_failures": 0,
            "evaluator_failures": 0,
            "worker_replacements": 0,
        },
        "collector": {
            **_gate_envelope("collector"),
            "controller_frozen": True,
            "intermediate_replay_count": 0,
            "p0_count": 2,
            "repair_trajectories_per_p0": 2,
            "base_results": [result.to_dict() for result in collector_base_results],
            "selected_batch_index": 0,
            "selected_batch_results": [
                result.to_dict() for result in collector_selected_batch_results
            ],
            "discarded_batches": [],
            "replay_events": [
                {
                    "batch_index": 0,
                    "base_index": index,
                    "selected_batch": True,
                    "result": result.to_dict(),
                }
                for index, result in enumerate(collector_selected_batch_results)
            ],
            "replay_event_count": 7,
            "attempt_event_count": 7,
            "retry_count": 0,
            "infra_failures": 0,
            "evaluator_failures": 0,
            "worker_replacements": 0,
            "repair_traces": collector_records,
        },
        "guided": {
            **_gate_envelope("guided"),
            "task_instance": _verified_task().to_dict(),
            "original_prompt": "stack the cubes",
            "training_input_contains_critique": False,
            "learning_group": group.to_dict(),
            **guided_provenance,
        },
        "trainer": {
            **_gate_envelope("trainer"),
            "learning_group": group.to_dict(),
            "actor_update_rpcs": 1,
            "optimizer_steps": 1,
            "optimizer_step_before": 0,
            "optimizer_step_after": 1,
            "gradient_norm": 0.25,
            "checkpoint": str(checkpoint),
            "group_rewards": [0, 0, 0, 0, 0, 0, 0, 1],
            "guided_token_mask_present": True,
            "guided_token_count": 4,
            "guided_mask_response_only": True,
            "rollout_is": False,
            "norm_adv_by_std_in_grpo": False,
            "loss_mode": "capsule_critique",
            "capsule_gamma": 0.1,
            "reference_kl_enabled": True,
            "reference_kl_coef": 0.001,
            "rollout_mode": "sync",
            "ppo_epochs": 1,
            "ppo_mini_batch_size": 8,
            "data_parallel_world_size": 1,
            "sequence_parallel_size": 1,
            "verl_provenance_before": _verl_provenance(),
            "verl_provenance_after": _verl_provenance(),
            "actor_update_skipped": False,
            "metrics": {"actor/pg_loss": 0.5, "capsule/guided_loss": -0.1},
            **checkpoint_contract,
            "guided_artifact_sha256": "f" * 64,
        },
    }


def _write_gate_payloads(
    directory: Path, payloads: dict[str, dict[str, object]]
) -> None:
    for gate, filename in analyze_artifacts.REQUIRED_GATE_FILES.items():
        if gate == "trainer":
            guided_path = directory / analyze_artifacts.REQUIRED_GATE_FILES["guided"]
            payloads["trainer"]["guided_artifact_sha256"] = common.artifact_file_sha256(
                guided_path
            )
        (directory / filename).write_text(
            json.dumps(payloads[gate], ensure_ascii=False), encoding="utf-8"
        )


def test_gate7_verifies_complete_typed_chain_and_emits_hash_manifest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    payloads = _complete_gate_payloads(checkpoint)
    _write_gate_payloads(tmp_path, payloads)

    summary = analyze_artifacts.audit_gate_directory(tmp_path)

    assert summary["runtime_verified"] is True
    assert summary["dataset_sha256"] == "9" * 64
    assert summary["resolved_environment_sha256"] == "e" * 64
    assert summary["verl_resolved_config_sha256"] == "f" * 64
    assert summary["typed_task_identities"] == [
        {
            "task_id": "cube-stack-5",
            "environment_seed": 5,
            "initial_state_sha256": "a" * 64,
        }
    ]
    assert summary["gate_statuses"] == {gate: "passed" for gate in payloads}
    assert [entry["gate"] for entry in summary["gate_chain"]] == list(payloads)
    assert summary["gate_chain"][0]["previous_sha256"] is None
    assert summary["gate_chain"][1]["previous_sha256"] == summary["gate_chain"][0][
        "sha256"
    ]


def test_gate7_rejects_noncanonical_gate_evidence(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    payloads = _complete_gate_payloads(checkpoint)
    payloads["collector"]["execution_mode"] = "custom_runner"
    _write_gate_payloads(tmp_path, payloads)

    with pytest.raises(common.GateArtifactError, match="noncanonical"):
        analyze_artifacts.audit_gate_directory(tmp_path)


def test_gate7_rejects_success_and_failure_evidence_for_same_gate(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    payloads = _complete_gate_payloads(checkpoint)
    _write_gate_payloads(tmp_path, payloads)
    seed_path = tmp_path / analyze_artifacts.REQUIRED_GATE_FILES["seed"]
    common.write_gate_failure_artifact(
        seed_path,
        gate="seed",
        run_id="capsule-smoke-001",
        config_sha256="c" * 64,
        git_sha="d" * 40,
        error=RuntimeError("late failure evidence"),
        stage="artifact_publish",
    )

    with pytest.raises(common.GateArtifactError, match="both success and failure"):
        analyze_artifacts.audit_gate_directory(tmp_path)


def test_gate7_rejects_symlink_gate_artifact_in_run_directory(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    payloads = _complete_gate_payloads(checkpoint)
    _write_gate_payloads(tmp_path, payloads)
    seed_path = tmp_path / analyze_artifacts.REQUIRED_GATE_FILES["seed"]
    external = tmp_path / "external-seed.json"
    seed_path.replace(external)
    seed_path.symlink_to(external)

    assert common.gate_failure_artifact_path(seed_path) == seed_path.with_name(
        f"{seed_path.name}.failure.json"
    )
    with pytest.raises(common.GateArtifactError, match="symlink"):
        analyze_artifacts.audit_gate_directory(tmp_path)


def test_gate7_rejects_gate_bytes_changed_during_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    payloads = _complete_gate_payloads(checkpoint)
    _write_gate_payloads(tmp_path, payloads)
    original_loader = analyze_artifacts._load_gate_artifact_snapshot
    mutated = False

    def load_then_mutate(path: Path, gate: str):
        nonlocal mutated
        snapshot = original_loader(path, gate)
        if gate == "preflight" and not mutated:
            path.write_bytes(path.read_bytes() + b"\n")
            mutated = True
        return snapshot

    monkeypatch.setattr(
        analyze_artifacts, "_load_gate_artifact_snapshot", load_then_mutate
    )

    with pytest.raises(common.GateArtifactError, match="changed during Gate 7 audit"):
        analyze_artifacts.audit_gate_directory(tmp_path)


def test_gate7_publishes_json_and_markdown_as_one_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    payloads = _complete_gate_payloads(checkpoint)
    _write_gate_payloads(tmp_path, payloads)
    output_json = tmp_path / "audit.json"
    output_report = tmp_path / "audit.md"
    real_link = analyze_artifacts.os.link

    def fail_report_publish(source: object, destination: object) -> None:
        if Path(destination) == output_report:
            raise OSError("report publication failed")
        real_link(source, destination)

    monkeypatch.setattr(analyze_artifacts.os, "link", fail_report_publish)

    with pytest.raises(OSError, match="report publication failed"):
        analyze_artifacts.main(
            [
                "--input-dir",
                str(tmp_path),
                "--output-json",
                str(output_json),
                "--output-report",
                str(output_report),
            ]
        )

    assert not output_json.exists()
    assert not output_report.exists()
    assert not list(tmp_path.glob(".audit.*.tmp"))


def test_gate7_pair_rollback_never_deletes_replaced_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    payloads = _complete_gate_payloads(checkpoint)
    _write_gate_payloads(tmp_path, payloads)
    output_json = tmp_path / "audit.json"
    output_report = tmp_path / "audit.md"
    real_link = analyze_artifacts.os.link

    def replace_json_then_fail_report(source: object, destination: object) -> None:
        destination_path = Path(destination)
        if destination_path == output_report:
            output_json.unlink()
            output_json.write_text("foreign replacement\n", encoding="utf-8")
            raise OSError("report publication failed after replacement")
        real_link(source, destination)

    monkeypatch.setattr(analyze_artifacts.os, "link", replace_json_then_fail_report)

    with pytest.raises(OSError, match="after replacement"):
        analyze_artifacts.main(
            [
                "--input-dir",
                str(tmp_path),
                "--output-json",
                str(output_json),
                "--output-report",
                str(output_report),
            ]
        )

    assert output_json.read_text(encoding="utf-8") == "foreign replacement\n"
    assert not output_report.exists()


def test_gate7_rejects_trainer_group_different_from_guided_group(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    payloads = _complete_gate_payloads(checkpoint)
    payloads["trainer"]["learning_group"]["group_uid"] = "different-group"
    _write_gate_payloads(tmp_path, payloads)

    with pytest.raises(common.GateArtifactError, match="exact verified guided group"):
        analyze_artifacts.audit_gate_directory(tmp_path)


def test_gate7_links_seed5_initial_state_across_replay_gates(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    payloads = _complete_gate_payloads(checkpoint)
    payloads["seed"]["initial_state_sha256"] = ["9" * 64, "b" * 64, "9" * 64]
    _write_gate_payloads(tmp_path, payloads)

    with pytest.raises(common.GateArtifactError, match="seed-5 initial state"):
        analyze_artifacts.audit_gate_directory(tmp_path)


def test_gate7_rejects_dataset_bytes_changed_between_gates(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    payloads = _complete_gate_payloads(checkpoint)
    payloads["collector"]["dataset_sha256"] = "8" * 64
    _write_gate_payloads(tmp_path, payloads)

    with pytest.raises(common.GateArtifactError, match="dataset SHA"):
        analyze_artifacts.audit_gate_directory(tmp_path)


@pytest.mark.parametrize(
    ("field_name", "message"),
    [
        ("resolved_environment_sha256", "resolved environment SHA"),
        ("verl_resolved_config_sha256", "resolved VeRL config SHA"),
    ],
)
def test_gate7_rejects_runtime_dependency_changed_between_gates(
    field_name: str, message: str, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    payloads = _complete_gate_payloads(checkpoint)
    payloads["collector"][field_name] = "8" * 64
    _write_gate_payloads(tmp_path, payloads)

    with pytest.raises(common.GateArtifactError, match=message):
        analyze_artifacts.audit_gate_directory(tmp_path)


def test_gate7_rejects_different_typed_seed5_task_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    payloads = _complete_gate_payloads(checkpoint)
    for replay in payloads["oracle_replay"]["replays"]:
        replay["result"]["task_id"] = "different-cube-stack-task"
    _write_gate_payloads(tmp_path, payloads)

    with pytest.raises(common.GateArtifactError, match="seed-5 task identity"):
        analyze_artifacts.audit_gate_directory(tmp_path)


def test_gate7_requires_seed5_task_identity_to_exist_in_preflight_dataset(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    payloads = _complete_gate_payloads(checkpoint)
    payloads["preflight"]["checks"]["dataset_task_identities"] = [
        {"task_id": "another-task", "environment_seed": 5}
    ]
    _write_gate_payloads(tmp_path, payloads)

    with pytest.raises(common.GateArtifactError, match="preflight dataset"):
        analyze_artifacts.audit_gate_directory(tmp_path)


def test_gate7_rejects_trainer_dependency_hash_different_from_gate5(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    payloads = _complete_gate_payloads(checkpoint)
    _write_gate_payloads(tmp_path, payloads)
    trainer_path = tmp_path / analyze_artifacts.REQUIRED_GATE_FILES["trainer"]
    trainer_payload = json.loads(trainer_path.read_text(encoding="utf-8"))
    trainer_payload["guided_artifact_sha256"] = "0" * 64
    trainer_path.write_text(json.dumps(trainer_payload), encoding="utf-8")

    with pytest.raises(common.GateArtifactError, match="Gate 5 artifact"):
        analyze_artifacts.audit_gate_directory(tmp_path)


def test_gate7_rejects_checkpoint_bytes_changed_after_gate6(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    payloads = _complete_gate_payloads(checkpoint)
    _write_gate_payloads(tmp_path, payloads)
    (checkpoint / "state.bin").write_bytes(b"tampered")

    with pytest.raises(common.GateArtifactError, match="checkpoint_sha256"):
        analyze_artifacts.audit_gate_directory(tmp_path)


def test_external_gate_publishes_runner_output_from_unique_staging_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _server_config(tmp_path)
    artifact = tmp_path / "seed.json"

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        staging = Path(argv[argv.index("--output") + 1])
        assert staging != artifact
        staging.write_text('{"fresh":true}\n', encoding="utf-8")
        assert _kwargs["capture_output"] is True
        assert _kwargs["text"] is True
        return subprocess.CompletedProcess(argv, 0, "runner out\n", "runner err\n")

    monkeypatch.setattr(common.subprocess, "run", fake_run)
    plan = common.ExternalGatePlan(
        gate_name="seed",
        config_path=config_path,
        artifact_path=artifact,
        runner_command=(
            f"{sys.executable} fake_runner.py --config {{config}} --output {{artifact}}"
        ),
    )

    common.run_external_gate(plan, validate_only=False)

    assert artifact.read_text(encoding="utf-8") == '{"fresh":true}\n'
    assert (tmp_path / "seed.json.stdout.log").read_text(encoding="utf-8") == "runner out\n"
    assert (tmp_path / "seed.json.stderr.log").read_text(encoding="utf-8") == "runner err\n"
    assert not list(tmp_path.glob(".seed.json.*.tmp"))


def test_external_gate_direct_publish_passes_final_artifact_to_locked_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _server_config(tmp_path)
    artifact = tmp_path / "trainer.json"

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        destination = Path(argv[argv.index("--output") + 1])
        assert destination == artifact
        destination.write_text('{"fresh":true}\n', encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "trainer out\n", "")

    monkeypatch.setattr(common.subprocess, "run", fake_run)
    plan = common.ExternalGatePlan(
        gate_name="trainer",
        config_path=config_path,
        artifact_path=artifact,
        runner_command=(
            f"{sys.executable} fake_runner.py --config {{config}} --output {{artifact}}"
        ),
        direct_artifact_publish=True,
    )

    common.run_external_gate(plan, validate_only=False)

    assert artifact.read_text(encoding="utf-8") == '{"fresh":true}\n'
    assert (tmp_path / "trainer.json.stdout.log").read_text(encoding="utf-8") == "trainer out\n"
    assert (tmp_path / "trainer.json.stderr.log").read_text(encoding="utf-8") == ""


def test_one_step_trainer_forbids_runner_override(tmp_path: Path) -> None:
    config_path = _server_config(tmp_path)

    with pytest.raises(SystemExit):
        one_step_trainer_smoke.main(
            [
                "--config",
                str(config_path),
                "--artifact",
                str(tmp_path / "trainer.json"),
                "--validate-only",
                "--runner-command",
                "python unsafe_override.py",
            ]
        )


@pytest.mark.parametrize(
    "entrypoint",
    (
        check_seed_determinism.main,
        oracle_clean_replay.main,
        controller_collector_smoke.main,
        build_verified_group.main,
    ),
)
def test_canonical_collection_gate_wrappers_forbid_runner_override(
    entrypoint: object, tmp_path: Path
) -> None:
    config_path = _server_config(tmp_path)

    with pytest.raises(SystemExit):
        entrypoint(
            [
                "--config",
                str(config_path),
                "--artifact",
                str(tmp_path / "gate.json"),
                "--validate-only",
                "--runner-command",
                "python custom_runner.py",
            ]
        )


def test_external_gate_does_not_publish_invalid_staging_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _server_config(tmp_path)
    artifact = tmp_path / "seed.json"

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        staging = Path(argv[argv.index("--output") + 1])
        staging.write_text('{"passed":false}\n', encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    def reject(_payload: object) -> None:
        raise common.GateArtifactError("invalid staged evidence")

    monkeypatch.setattr(common.subprocess, "run", fake_run)
    plan = common.ExternalGatePlan(
        gate_name="seed",
        config_path=config_path,
        artifact_path=artifact,
        runner_command=(
            f"{sys.executable} fake_runner.py --config {{config}} --output {{artifact}}"
        ),
    )

    with pytest.raises(common.GateArtifactError, match="invalid staged evidence"):
        common.run_external_gate(plan, validate_only=False, verifier=reject)

    assert not artifact.exists()
    failure = json.loads(
        (tmp_path / "seed.json.failure.json").read_text(encoding="utf-8")
    )
    assert failure["passed"] is False
    assert failure["gate"] == "seed"
    assert failure["exception"] == {
        "type": "GateArtifactError",
        "message": "invalid staged evidence",
        "stage": "artifact_verification",
    }
    assert not list(tmp_path.glob(".seed.json.*.tmp"))


def test_external_gate_persists_nonzero_runner_failure_and_captured_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _server_config(tmp_path)
    artifact = tmp_path / "seed.json"

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 17, "partial out\n", "child crashed\n")

    monkeypatch.setattr(common.subprocess, "run", fake_run)
    monkeypatch.setattr(common, "_available_git_sha", lambda _root: "c" * 40)
    plan = common.ExternalGatePlan(
        gate_name="seed_determinism",
        config_path=config_path,
        artifact_path=artifact,
        runner_command=(
            f"{sys.executable} fake_runner.py --config {{config}} --output {{artifact}}"
        ),
    )

    with pytest.raises(common.GateExecutionError, match="status 17"):
        common.run_external_gate(plan, validate_only=False)

    failure = json.loads(
        (tmp_path / "seed.json.failure.json").read_text(encoding="utf-8")
    )
    assert not artifact.exists()
    assert failure["schema_version"] == 1
    assert failure["gate"] == "seed"
    assert failure["passed"] is False
    assert failure["run_id"] == tmp_path.name
    assert failure["config_sha256"] == common.artifact_file_sha256(config_path)
    assert failure["git_sha"] == "c" * 40
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert failure["dataset_sha256"] == common.artifact_file_sha256(
        config["runtime"]["dataset_path"]
    )
    assert failure["exception"]["stage"] == "runner_exit"
    assert (tmp_path / "seed.json.stdout.log").read_text(encoding="utf-8") == "partial out\n"
    assert (tmp_path / "seed.json.stderr.log").read_text(encoding="utf-8") == "child crashed\n"


@pytest.mark.parametrize("invalid_version", [True, 1.0])
def test_failure_artifact_requires_exact_integer_schema_version(
    invalid_version: object,
) -> None:
    payload = {
        "schema_version": invalid_version,
        "gate": "seed",
        "passed": False,
        "run_id": "run-01",
        "config_sha256": "c" * 64,
        "git_sha": "d" * 40,
        "exception": {
            "type": "RuntimeError",
            "message": "failed",
            "stage": "runtime_dispatch",
        },
    }

    with pytest.raises(common.GateArtifactError, match="schema_version"):
        common._verify_failure_artifact(
            payload,
            gate="seed",
            run_id="run-01",
            config_sha256="c" * 64,
        )


def test_external_gate_promotes_child_failure_before_cleaning_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _server_config(tmp_path)
    artifact = tmp_path / "seed.json"
    child_payload: dict[str, object] = {}
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_sha256 = common.artifact_file_sha256(config["runtime"]["dataset_path"])

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        staging = Path(argv[argv.index("--output") + 1])
        child_failure = common.write_gate_failure_artifact(
            staging,
            gate="seed",
            run_id=tmp_path.name,
            config_sha256=common.artifact_file_sha256(config_path),
            git_sha="a" * 40,
            dataset_sha256=dataset_sha256,
            error=RuntimeError("child runtime failed"),
            stage="runtime_dispatch",
        )
        child_payload.update(json.loads(child_failure.read_text(encoding="utf-8")))
        return subprocess.CompletedProcess(argv, 1, "child plan\n", "traceback\n")

    monkeypatch.setattr(common.subprocess, "run", fake_run)
    plan = common.ExternalGatePlan(
        gate_name="seed",
        config_path=config_path,
        artifact_path=artifact,
        runner_command=(
            f"{sys.executable} fake_runner.py --config {{config}} --output {{artifact}}"
        ),
    )

    with pytest.raises(common.GateExecutionError, match="status 1"):
        common.run_external_gate(plan, validate_only=False)

    final_failure = tmp_path / "seed.json.failure.json"
    assert json.loads(final_failure.read_text(encoding="utf-8")) == child_payload
    assert not list(tmp_path.glob(".seed.json.*.tmp.failure.json"))
    assert (tmp_path / "seed.json.stdout.log").is_file()
    assert (tmp_path / "seed.json.stderr.log").is_file()


def test_external_gate_direct_publish_reuses_child_failure_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _server_config(tmp_path)
    artifact = tmp_path / "trainer.json"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_sha256 = common.artifact_file_sha256(config["runtime"]["dataset_path"])

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        destination = Path(argv[argv.index("--output") + 1])
        common.write_gate_failure_artifact(
            destination,
            gate="trainer",
            run_id=tmp_path.name,
            config_sha256=common.artifact_file_sha256(config_path),
            git_sha="b" * 40,
            dataset_sha256=dataset_sha256,
            error=RuntimeError("checkpoint validation failed"),
            stage="artifact_verification",
        )
        return subprocess.CompletedProcess(argv, 2, "", "trainer traceback\n")

    monkeypatch.setattr(common.subprocess, "run", fake_run)
    plan = common.ExternalGatePlan(
        gate_name="trainer",
        config_path=config_path,
        artifact_path=artifact,
        runner_command=(
            f"{sys.executable} fake_runner.py --config {{config}} --output {{artifact}}"
        ),
        direct_artifact_publish=True,
    )

    with pytest.raises(common.GateExecutionError, match="status 2"):
        common.run_external_gate(plan, validate_only=False)

    failure = json.loads(
        (tmp_path / "trainer.json.failure.json").read_text(encoding="utf-8")
    )
    assert failure["exception"]["message"] == "checkpoint validation failed"
    assert failure["exception"]["stage"] == "artifact_verification"
    assert (tmp_path / "trainer.json.stderr.log").read_text(encoding="utf-8") == (
        "trainer traceback\n"
    )


@pytest.mark.parametrize(
    "claimed_suffix",
    (".failure.json", ".stdout.log", ".stderr.log"),
)
def test_external_gate_refuses_to_overwrite_existing_failure_or_log(
    claimed_suffix: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _server_config(tmp_path)
    artifact = tmp_path / "seed.json"
    claimed = tmp_path / f"seed.json{claimed_suffix}"
    claimed.write_bytes(b"original immutable evidence\n")

    def forbidden_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("runner must not start when an evidence path is claimed")

    monkeypatch.setattr(common.subprocess, "run", forbidden_run)
    plan = common.ExternalGatePlan(
        gate_name="seed",
        config_path=config_path,
        artifact_path=artifact,
        runner_command=(
            f"{sys.executable} fake_runner.py --config {{config}} --output {{artifact}}"
        ),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        common.run_external_gate(plan, validate_only=False)

    assert claimed.read_bytes() == b"original immutable evidence\n"
    assert not artifact.exists()


def test_external_gate_keeps_staged_child_failure_when_final_publication_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _server_config(tmp_path)
    artifact = tmp_path / "seed.json"
    final_failure = tmp_path / "seed.json.failure.json"
    real_link = common.os.link
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_sha256 = common.artifact_file_sha256(config["runtime"]["dataset_path"])

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        staging = Path(argv[argv.index("--output") + 1])
        common.write_gate_failure_artifact(
            staging,
            gate="seed",
            run_id=tmp_path.name,
            config_sha256=common.artifact_file_sha256(config_path),
            git_sha="a" * 40,
            dataset_sha256=dataset_sha256,
            error=RuntimeError("unique child failure"),
            stage="runtime_dispatch",
        )

        def fail_final_failure_link(source: object, destination: object) -> None:
            if Path(destination) == final_failure:
                raise OSError("failure publication unavailable")
            real_link(source, destination)

        monkeypatch.setattr(common.os, "link", fail_final_failure_link)
        return subprocess.CompletedProcess(argv, 3, "", "child failed\n")

    monkeypatch.setattr(common.subprocess, "run", fake_run)
    plan = common.ExternalGatePlan(
        gate_name="seed",
        config_path=config_path,
        artifact_path=artifact,
        runner_command=(
            f"{sys.executable} fake_runner.py --config {{config}} --output {{artifact}}"
        ),
    )

    with pytest.raises(common.GateExecutionError, match="status 3") as caught:
        common.run_external_gate(plan, validate_only=False)

    staged_failures = list(tmp_path.glob(".seed.json.*.tmp.failure.json"))
    assert not final_failure.exists()
    assert len(staged_failures) == 1
    assert json.loads(staged_failures[0].read_text(encoding="utf-8"))["exception"][
        "message"
    ] == "unique child failure"
    assert caught.value.failure_artifact_recording_errors
