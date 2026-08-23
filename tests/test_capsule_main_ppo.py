from __future__ import annotations

import os
import json
from pathlib import Path

import pytest
import yaml

from capx.rl.capsule import main_ppo
from capx.rl.capsule.provenance import (
    file_sha256,
    project_path,
    runtime_dependency_hashes,
)
from capx.rl.capsule.schema import TaskInstanceV1
from capx.rl.capsule.compat import (
    PINNED_VERL_SHA,
    VeRLCompatibilityError,
    VeRLCompatibilityReport,
)
from test_capsule_config import valid_config


def write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "capsule.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    runtime = config.get("runtime", {})
    manifest_value = runtime.get("bundle_manifest_path")
    if isinstance(manifest_value, str):
        try:
            dataset_path = project_path(config, runtime["dataset_path"], "runtime.dataset_path")
            resolved_verl_path = project_path(
                config,
                runtime["verl_resolved_config_path"],
                "runtime.verl_resolved_config_path",
            )
            dependencies = runtime_dependency_hashes(config)
            records = [
                TaskInstanceV1.from_json(line)
                for line in dataset_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, KeyError, ValueError):
            return path
        typed_identities = [
            {
                "task_id": task.task_id,
                "environment_seed": task.environment_seed,
                "initial_state_sha256": task.initial_state_sha256,
            }
            for task in records
        ]
        audit_path = project_path(
            config, runtime["gate7_audit_path"], "runtime.gate7_audit_path"
        )
        audit_path.write_text(
            json.dumps(
                {
                    "runtime_verified": True,
                    "run_id": "capsule-smoke-001",
                    "config_sha256": file_sha256(path),
                    "dataset_sha256": file_sha256(dataset_path),
                    **dependencies,
                    "typed_task_identities": typed_identities,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        manifest_path = project_path(
            config, manifest_value, "runtime.bundle_manifest_path"
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_type": "capsule_seed_resolved_dataset",
                    "record_count": len(records),
                    "dataset_path": str(dataset_path),
                    "dataset_sha256": file_sha256(dataset_path),
                    "config_path": str(path.resolve()),
                    "config_sha256": file_sha256(path),
                    "source_config_path": str(path.resolve()),
                    "source_config_sha256": file_sha256(path),
                    "source_dataset_path": str(dataset_path),
                    "source_dataset_sha256": file_sha256(dataset_path),
                    "gate7_audit_path": str(audit_path),
                    "gate7_audit_sha256": file_sha256(audit_path),
                    "gate7_run_id": "capsule-smoke-001",
                    "gate7_typed_task_identities": typed_identities,
                    "output_dataset_path": str(dataset_path),
                    "output_dataset_sha256": file_sha256(dataset_path),
                    "output_config_path": str(path.resolve()),
                    "output_config_sha256": file_sha256(path),
                    "output_task_identities": typed_identities,
                    "resolved_environment_sha256": dependencies[
                        "resolved_environment_sha256"
                    ],
                    "verl_resolved_config_path": str(resolved_verl_path),
                    "verl_resolved_config_sha256": dependencies[
                        "verl_resolved_config_sha256"
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return path


def valid_local_config(tmp_path: Path) -> dict:
    config = valid_config()
    project_root = tmp_path / "project"
    dataset = project_root / "data" / "tasks.jsonl"
    program_model = project_root / "models" / "program"
    resolved_verl_config = project_root / "configs" / "verl.yaml"
    environment_config = project_root / "configs" / "environment.yaml"
    gate7_audit = project_root / "bundle" / "gate07_audit.json"
    bundle_manifest = project_root / "bundle" / "bundle_manifest.json"
    output_parent = project_root / "outputs"
    dataset.parent.mkdir(parents=True)
    dataset.write_text(
        TaskInstanceV1(
            task_id="cube-stack",
            environment_seed=5,
            prompt="stack",
            environment="robosuite_cube_stack",
            api="franka_control_privileged",
            privilege="privileged",
            initial_state_sha256="a" * 64,
        ).to_json()
        + "\n",
        encoding="utf-8",
    )
    program_model.mkdir(parents=True)
    resolved_verl_config.parent.mkdir(parents=True)
    resolved_verl_config.write_text("trainer: {}\n", encoding="utf-8")
    environment_config.write_text("task: CubeStack\n", encoding="utf-8")
    gate7_audit.parent.mkdir(parents=True)
    output_parent.mkdir(parents=True)
    config["runtime"].update(
        {
            "project_root": str(project_root),
            "dataset_path": "data/tasks.jsonl",
            "program_model_path": "models/program",
            "verl_resolved_config_path": "configs/verl.yaml",
            "bundle_manifest_path": "bundle/bundle_manifest.json",
            "gate7_audit_path": "bundle/gate07_audit.json",
            "output_dir": "outputs/capsule-run",
        }
    )
    config["task"]["config_path"] = "configs/environment.yaml"
    return config


def compatible_report(source_path: str | Path) -> VeRLCompatibilityReport:
    return VeRLCompatibilityReport(
        source_path=str(Path(source_path).resolve()),
        expected_sha=PINNED_VERL_SHA,
        actual_sha=PINNED_VERL_SHA,
        compatible=True,
        rollout_is_slot="rollout_is_weights",
        checked_symbols=(
            "register_policy_loss",
            "compute_policy_loss_vanilla",
            "compute_log_prob",
            "compute_ref_log_prob",
            "update_actor",
        ),
    )


@pytest.mark.parametrize("validation_flag", ["--validate-only", "--dry-run"])
def test_validation_mode_checks_static_config_and_never_loads_trainer_factory(
    validation_flag: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = valid_local_config(tmp_path)
    path = write_config(tmp_path, config)
    monkeypatch.setattr(
        main_ppo,
        "check_verl_compatibility",
        lambda source_path, expected_sha: compatible_report(source_path),
    )
    provenance_calls: list[dict] = []

    def project_git_sha(received_config):
        provenance_calls.append(received_config)
        return "a" * 40

    monkeypatch.setattr(main_ppo, "_project_git_sha", project_git_sha)
    monkeypatch.setattr(
        main_ppo,
        "register_capsule_critique_policy_loss",
        lambda: (_ for _ in ()).throw(
            AssertionError("validate-only must not import or register VeRL")
        ),
    )
    monkeypatch.setattr(
        main_ppo,
        "bind_pinned_verl_import",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("validate-only must not bind or import VeRL")
        ),
    )
    monkeypatch.setattr(
        main_ppo,
        "run_training",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("validation mode must not construct a trainer")
        ),
    )

    def forbidden_loader(_path: str):
        raise AssertionError("validation mode must not import trainer code")

    exit_code = main_ppo.main(
        ["--config", str(path), validation_flag],
        factory_loader=forbidden_loader,
    )

    assert exit_code == 0
    assert provenance_calls == [config]
    output = capsys.readouterr().out
    assert '"mode": "validate-only"' in output
    assert '"project_git_sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"' in output
    assert PINNED_VERL_SHA in output


@pytest.mark.parametrize(
    ("runtime_field", "tampered_text", "message"),
    [
        ("dataset_path", "tampered dataset\n", "dataset SHA"),
        ("verl_resolved_config_path", "trainer: {tampered: true}\n", "VeRL config SHA"),
        ("gate7_audit_path", '{"runtime_verified":false}\n', "Gate7 audit SHA"),
    ],
)
def test_validate_only_rejects_tampered_bundle_provenance(
    runtime_field: str,
    tampered_text: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = valid_local_config(tmp_path)
    path = write_config(tmp_path, config)
    tampered_path = project_path(
        config, config["runtime"][runtime_field], f"runtime.{runtime_field}"
    )
    tampered_path.write_text(tampered_text, encoding="utf-8")
    monkeypatch.setattr(main_ppo, "_project_git_sha", lambda _config: "a" * 40)
    monkeypatch.setattr(
        main_ppo,
        "check_verl_compatibility",
        lambda source_path, expected_sha: compatible_report(source_path),
    )

    assert main_ppo.main(["--config", str(path), "--validate-only"]) == 2
    assert message in capsys.readouterr().err


def test_validate_only_requires_bundle_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = valid_local_config(tmp_path)
    path = write_config(tmp_path, config)
    project_path(
        config,
        config["runtime"]["bundle_manifest_path"],
        "runtime.bundle_manifest_path",
    ).unlink()
    monkeypatch.setattr(main_ppo, "_project_git_sha", lambda _config: "a" * 40)

    assert main_ppo.main(["--config", str(path), "--validate-only"]) == 2
    assert "bundle_manifest_path" in capsys.readouterr().err


def test_bundle_manifest_requires_exact_integer_schema_version(tmp_path: Path) -> None:
    config = valid_local_config(tmp_path)
    path = write_config(tmp_path, config)
    manifest_path = project_path(
        config,
        config["runtime"]["bundle_manifest_path"],
        "runtime.bundle_manifest_path",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(main_ppo.TrainerFactoryError, match="schema_version"):
        main_ppo.verify_bundle_provenance(config, path)


def test_bundle_manifest_rejects_duplicate_task_seed_with_different_state(
    tmp_path: Path,
) -> None:
    config = valid_local_config(tmp_path)
    path = write_config(tmp_path, config)
    manifest_path = project_path(
        config,
        config["runtime"]["bundle_manifest_path"],
        "runtime.bundle_manifest_path",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    duplicate = dict(manifest["gate7_typed_task_identities"][0])
    duplicate["initial_state_sha256"] = "b" * 64
    manifest["gate7_typed_task_identities"].append(duplicate)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(main_ppo.TrainerFactoryError, match="duplicate identities"):
        main_ppo.verify_bundle_provenance(config, path)


def test_validate_only_reports_missing_environment_dependency_as_typed_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = valid_local_config(tmp_path)
    path = write_config(tmp_path, config)
    project_path(
        config,
        config["task"]["config_path"],
        "task.config_path",
    ).unlink()
    monkeypatch.setattr(main_ppo, "_project_git_sha", lambda _config: "a" * 40)

    assert main_ppo.main(["--config", str(path), "--validate-only"]) == 2
    assert "cannot hash runtime dependencies" in capsys.readouterr().err


def test_validate_only_reports_config_failure_without_running_compatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = valid_config()
    config["actor_rollout_ref"]["rollout"]["n"] = 7
    path = write_config(tmp_path, config)

    def forbidden_compat(*_args, **_kwargs):
        raise AssertionError("invalid config must fail before compatibility checks")

    monkeypatch.setattr(main_ppo, "check_verl_compatibility", forbidden_compat)

    assert main_ppo.main(["--config", str(path), "--validate-only"]) == 2
    assert "n=8" in capsys.readouterr().err


def test_validate_only_surfaces_typed_uninitialized_verl_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_config(tmp_path, valid_local_config(tmp_path))
    monkeypatch.setattr(main_ppo, "_project_git_sha", lambda _config: "a" * 40)

    def unavailable(*_args, **_kwargs):
        raise VeRLCompatibilityError("uninitialized_source", "VeRL source is not initialized")

    monkeypatch.setattr(main_ppo, "check_verl_compatibility", unavailable)

    assert main_ppo.main(["--config", str(path), "--validate-only"]) == 2
    error = capsys.readouterr().err
    assert "uninitialized_source" in error
    assert "not initialized" in error


@pytest.mark.parametrize("validation_flag", ["--validate-only", "--dry-run"])
def test_validation_mode_requires_nonempty_trainer_factory(
    validation_flag: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = valid_local_config(tmp_path)
    config["trainer_factory"] = "  "
    path = write_config(tmp_path, config)
    monkeypatch.setattr(
        main_ppo,
        "check_verl_compatibility",
        lambda source_path, expected_sha: compatible_report(source_path),
    )
    monkeypatch.setattr(main_ppo, "_project_git_sha", lambda _config: "a" * 40)

    assert main_ppo.main(["--config", str(path), validation_flag]) == 2
    assert "trainer_factory" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("field_name", "invalid_kind", "expected_kind"),
    [
        ("dataset_path", "missing", "file"),
        ("dataset_path", "directory", "file"),
        ("program_model_path", "missing", "directory"),
        ("program_model_path", "file", "directory"),
        ("verl_resolved_config_path", "missing", "file"),
        ("verl_resolved_config_path", "directory", "file"),
    ],
)
def test_validate_only_requires_existing_runtime_paths_of_the_expected_kind(
    field_name: str,
    invalid_kind: str,
    expected_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = valid_local_config(tmp_path)
    invalid_path = tmp_path / f"invalid-{field_name}"
    if invalid_kind == "directory":
        invalid_path.mkdir()
    elif invalid_kind == "file":
        invalid_path.write_text("invalid", encoding="utf-8")
    config["runtime"][field_name] = str(invalid_path)
    path = write_config(tmp_path, config)
    monkeypatch.setattr(
        main_ppo,
        "check_verl_compatibility",
        lambda source_path, expected_sha: compatible_report(source_path),
    )
    monkeypatch.setattr(main_ppo, "_project_git_sha", lambda _config: "a" * 40)

    assert main_ppo.main(["--config", str(path), "--validate-only"]) == 2
    error = capsys.readouterr().err
    assert f"runtime.{field_name}" in error
    assert f"existing {expected_kind}" in error


def test_validate_only_checks_nearest_existing_output_parent_is_writable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = valid_local_config(tmp_path)
    existing_parent = Path(config["runtime"]["project_root"]) / "publish-here"
    existing_parent.mkdir()
    config["runtime"]["output_dir"] = "publish-here/future/capsule-run"
    path = write_config(tmp_path, config)
    monkeypatch.setattr(
        main_ppo,
        "check_verl_compatibility",
        lambda source_path, expected_sha: compatible_report(source_path),
    )
    monkeypatch.setattr(main_ppo, "_project_git_sha", lambda _config: "a" * 40)
    access_calls: list[tuple[Path, int]] = []

    def writable(candidate: str | os.PathLike[str], mode: int) -> bool:
        access_calls.append((Path(candidate), mode))
        return True

    monkeypatch.setattr(main_ppo.os, "access", writable)

    assert main_ppo.main(["--config", str(path), "--validate-only"]) == 0
    assert access_calls == [(existing_parent.resolve(), os.W_OK)]


def test_validate_only_rejects_nonwritable_output_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = valid_local_config(tmp_path)
    path = write_config(tmp_path, config)
    monkeypatch.setattr(main_ppo.os, "access", lambda _path, _mode: False)
    monkeypatch.setattr(main_ppo, "_project_git_sha", lambda _config: "a" * 40)

    assert main_ppo.main(["--config", str(path), "--validate-only"]) == 2
    error = capsys.readouterr().err
    assert "runtime.output_dir" in error
    assert "not writable" in error


def test_validate_only_rejects_output_path_that_is_an_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = valid_local_config(tmp_path)
    output_file = tmp_path / "output-file"
    output_file.write_text("not a directory", encoding="utf-8")
    config["runtime"]["output_dir"] = str(output_file)
    path = write_config(tmp_path, config)
    monkeypatch.setattr(main_ppo, "_project_git_sha", lambda _config: "a" * 40)

    assert main_ppo.main(["--config", str(path), "--validate-only"]) == 2
    error = capsys.readouterr().err
    assert "runtime.output_dir" in error
    assert "not a directory" in error


def test_validate_only_fails_closed_on_project_checkout_provenance_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = write_config(tmp_path, valid_local_config(tmp_path))
    monkeypatch.setattr(
        main_ppo,
        "_project_git_sha",
        lambda _config: (_ for _ in ()).throw(
            main_ppo.TrainerFactoryError("project checkout has untracked files")
        ),
    )

    assert main_ppo.main(["--config", str(path), "--validate-only"]) == 2
    assert "untracked files" in capsys.readouterr().err


def test_non_validate_mode_requires_explicit_project_trainer_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = valid_config()
    config["trainer_factory"] = ""
    path = write_config(tmp_path, config)
    monkeypatch.setattr(
        main_ppo,
        "check_verl_compatibility",
        lambda source_path, expected_sha: compatible_report(source_path),
    )
    monkeypatch.setattr(main_ppo, "_project_git_sha", lambda _config: "a" * 40)

    assert main_ppo.main(["--config", str(path)]) == 2
    assert "trainer_factory" in capsys.readouterr().err


def test_non_validate_mode_builds_only_explicit_factory_and_calls_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = valid_local_config(tmp_path)
    config["trainer_factory"] = "tests.fake_capsule:create_trainer"
    path = write_config(tmp_path, config)
    monkeypatch.setattr(
        main_ppo,
        "check_verl_compatibility",
        lambda source_path, expected_sha: compatible_report(source_path),
    )
    monkeypatch.setattr(main_ppo, "_project_git_sha", lambda _config: "a" * 40)
    events: list[str] = []
    monkeypatch.setattr(
        main_ppo,
        "bind_pinned_verl_import",
        lambda source_path: events.append(f"bind:{source_path}"),
    )
    monkeypatch.setattr(
        main_ppo,
        "register_capsule_critique_policy_loss",
        lambda: events.append("register") or True,
    )

    class FakeTrainer:
        def fit(self):
            events.append("fit")
            return {"dry_fake": True}

    def loader(dotted_path: str):
        events.append(f"load:{dotted_path}")

        def factory(received_config):
            assert received_config["capsule"]["group_size"] == 8
            events.append("factory")
            return FakeTrainer()

        return factory

    assert main_ppo.main(["--config", str(path)], factory_loader=loader) == 0
    resolved_verl = main_ppo._runtime_path(config, "verl_source_path")
    assert events == [
        f"bind:{resolved_verl}",
        "register",
        "load:tests.fake_capsule:create_trainer",
        "factory",
        "fit",
    ]


def test_non_validate_mode_rejects_bundle_drift_during_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = valid_local_config(tmp_path)
    config["trainer_factory"] = "tests.fake_capsule:create_trainer"
    path = write_config(tmp_path, config)
    dataset_path = project_path(
        config, config["runtime"]["dataset_path"], "runtime.dataset_path"
    )
    monkeypatch.setattr(
        main_ppo,
        "check_verl_compatibility",
        lambda source_path, expected_sha: compatible_report(source_path),
    )
    monkeypatch.setattr(main_ppo, "_project_git_sha", lambda _config: "a" * 40)
    monkeypatch.setattr(main_ppo, "bind_pinned_verl_import", lambda _path: None)
    monkeypatch.setattr(main_ppo, "register_capsule_critique_policy_loss", lambda: True)

    class MutatingTrainer:
        def fit(self):
            dataset_path.write_bytes(dataset_path.read_bytes() + b"\n")
            return {"fit": True}

    assert main_ppo.main(
        ["--config", str(path)],
        factory_loader=lambda _path: lambda _config: MutatingTrainer(),
    ) == 2
    assert "dataset SHA" in capsys.readouterr().err


def test_loss_registration_failure_is_fail_closed_before_factory_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_config(tmp_path, valid_local_config(tmp_path))
    monkeypatch.setattr(
        main_ppo,
        "check_verl_compatibility",
        lambda source_path, expected_sha: compatible_report(source_path),
    )
    monkeypatch.setattr(main_ppo, "bind_pinned_verl_import", lambda _path: None)
    monkeypatch.setattr(main_ppo, "register_capsule_critique_policy_loss", lambda: False)
    monkeypatch.setattr(main_ppo, "_project_git_sha", lambda _config: "a" * 40)

    def forbidden_loader(_path: str):
        raise AssertionError("factory must not load when loss registration fails")

    assert main_ppo.main(["--config", str(path)], factory_loader=forbidden_loader) == 2
    assert "register" in capsys.readouterr().err


def test_non_validate_mode_rejects_dirty_project_before_binding_verl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_config(tmp_path, valid_local_config(tmp_path))
    monkeypatch.setattr(
        main_ppo,
        "check_verl_compatibility",
        lambda source_path, expected_sha: compatible_report(source_path),
    )
    monkeypatch.setattr(
        main_ppo,
        "_project_git_sha",
        lambda _config: (_ for _ in ()).throw(
            main_ppo.TrainerFactoryError("project checkout has untracked files")
        ),
    )
    monkeypatch.setattr(
        main_ppo,
        "bind_pinned_verl_import",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("dirty project must fail before VeRL binding")
        ),
    )

    assert main_ppo.main(["--config", str(path)]) == 2
    assert "untracked" in capsys.readouterr().err


def test_factory_without_fit_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_config(tmp_path, valid_config())
    monkeypatch.setattr(
        main_ppo,
        "check_verl_compatibility",
        lambda source_path, expected_sha: compatible_report(source_path),
    )
    monkeypatch.setattr(main_ppo, "bind_pinned_verl_import", lambda _path: None)
    monkeypatch.setattr(main_ppo, "register_capsule_critique_policy_loss", lambda: True)

    with pytest.raises(main_ppo.TrainerFactoryError, match="fit"):
        main_ppo.run_training(
            main_ppo.load_and_validate_config(path)[0],
            factory_loader=lambda _path: lambda _config: object(),
        )
