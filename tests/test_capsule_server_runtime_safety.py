from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from capx.rl.capsule.controller import OpenAICompatibleControllerTransport
from capx.rl.capsule.group import deterministic_group_uid
from capx.rl.capsule.schema import TaskInstanceV1
from capx.rl.capsule.server_factory import (
    CapsuleServerRuntime,
    ServerFactoryError,
    VeRLWorkerSession,
    _bind_pinned_verl_import,
    _capsule_data_parallel_world_size,
    _close_runtime_resources,
    _configure_verl_training_schedule,
    _pinned_ray_runtime_env,
    _schedule_training_tasks,
    load_task_instances,
)


def _task_config(tmp_path: Path, dataset: Path) -> dict[str, object]:
    return {
        "runtime": {
            "project_root": str(tmp_path),
            "dataset_path": str(dataset),
        },
        "task": {
            "environment": "robosuite_cube_stack",
            "api": "franka_control_privileged",
            "privilege": "privileged",
        },
    }


def _task_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": 1,
        "task_id": "cube-stack-5",
        "environment_seed": 5,
        "prompt": "stack the cubes",
        "environment": "robosuite_cube_stack",
        "api": "franka_control_privileged",
        "privilege": "privileged",
        "initial_state_sha256": "a" * 64,
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize("field", ["environment", "api", "privilege"])
def test_task_loader_rejects_dataset_provenance_different_from_runtime_config(
    field: str, tmp_path: Path
) -> None:
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(
        json.dumps(_task_row(**{field: "different"})) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ServerFactoryError, match=field):
        load_task_instances(_task_config(tmp_path, dataset))


def test_formal_runtime_defaults_to_strict_seed_resolved_task_loader() -> None:
    runtime = CapsuleServerRuntime({})

    assert runtime.task_loader is load_task_instances


def test_epoch_scheduler_assigns_unique_collection_and_group_ids() -> None:
    task = TaskInstanceV1.from_dict(_task_row())

    scheduled = _schedule_training_tasks((task,), 2)

    assert len(scheduled) == 2
    assert scheduled[0].metadata["capsule_collection_id"] != scheduled[1].metadata[
        "capsule_collection_id"
    ]
    assert deterministic_group_uid(scheduled[0]) != deterministic_group_uid(scheduled[1])
    assert scheduled[0].initial_state_sha256 == scheduled[1].initial_state_sha256


def _restore_verl_modules(previous: dict[str, object]) -> None:
    for name in tuple(sys.modules):
        if name == "verl" or name.startswith("verl."):
            del sys.modules[name]
    sys.modules.update(previous)


def test_pinned_verl_binding_rejects_an_already_imported_different_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pinned = tmp_path / "pinned"
    installed = tmp_path / "installed"
    (pinned / "verl").mkdir(parents=True)
    (installed / "verl").mkdir(parents=True)
    (pinned / "verl" / "__init__.py").write_text("SOURCE = 'pinned'\n", encoding="utf-8")
    (installed / "verl" / "__init__.py").write_text(
        "SOURCE = 'installed'\n", encoding="utf-8"
    )
    previous = {
        name: module
        for name, module in sys.modules.items()
        if name == "verl" or name.startswith("verl.")
    }
    previous_sys_path = list(sys.path)
    _restore_verl_modules({})
    monkeypatch.syspath_prepend(str(installed))
    try:
        imported = importlib.import_module("verl")
        assert imported.SOURCE == "installed"

        with pytest.raises(ServerFactoryError, match="pinned checkout"):
            _bind_pinned_verl_import(pinned)
    finally:
        _restore_verl_modules(previous)
        sys.path[:] = previous_sys_path


def test_pinned_verl_binding_imports_exact_validated_checkout(
    tmp_path: Path,
) -> None:
    pinned = tmp_path / "pinned"
    (pinned / "verl").mkdir(parents=True)
    (pinned / "verl" / "__init__.py").write_text("SOURCE = 'pinned'\n", encoding="utf-8")
    previous = {
        name: module
        for name, module in sys.modules.items()
        if name == "verl" or name.startswith("verl.")
    }
    previous_sys_path = list(sys.path)
    _restore_verl_modules({})
    try:
        _bind_pinned_verl_import(pinned)

        imported = importlib.import_module("verl")
        assert imported.SOURCE == "pinned"
        assert Path(imported.__file__).resolve().is_relative_to((pinned / "verl").resolve())
    finally:
        _restore_verl_modules(previous)
        sys.path[:] = previous_sys_path


def test_ray_runtime_env_propagates_pinned_verl_without_mutating_input(tmp_path: Path) -> None:
    pinned = tmp_path / "pinned"
    runtime_env = {"env_vars": {"PYTHONPATH": "/existing", "KEEP": "yes"}}
    pinned_sha = "a" * 40

    resolved = _pinned_ray_runtime_env(runtime_env, pinned, pinned_sha)

    assert resolved is not runtime_env
    assert runtime_env["env_vars"]["PYTHONPATH"] == "/existing"
    assert resolved["env_vars"]["CAPX_PINNED_VERL_SOURCE_PATH"] == str(pinned.resolve())
    assert resolved["env_vars"]["CAPX_PINNED_VERL_SHA"] == pinned_sha
    assert resolved["env_vars"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert resolved["env_vars"]["PYTHONPATH"].split(os.pathsep) == [
        str(pinned.resolve()),
        "/existing",
    ]
    assert resolved["env_vars"]["KEEP"] == "yes"


def test_verl_schedule_is_bounded_by_dataset_rows_and_epochs() -> None:
    config = OmegaConf.create(
        {
            "trainer": {"total_epochs": 3, "total_training_steps": 999},
            "actor_rollout_ref": {"actor": {"optim": {"total_training_steps": 999}}},
        }
    )

    total_epochs, total_steps = _configure_verl_training_schedule(config, dataset_row_count=2)

    assert (total_epochs, total_steps) == (3, 6)
    assert config.trainer.total_training_steps == 6
    assert config.actor_rollout_ref.actor.optim.total_training_steps == 6


@pytest.mark.parametrize(("gpus", "nodes", "expected"), [(1, 1, 1), (4, 1, 4), (4, 2, 8)])
def test_capsule_world_size_must_evenly_partition_one_7_plus_1_group(
    gpus: int, nodes: int, expected: int
) -> None:
    config = OmegaConf.create({"trainer": {"n_gpus_per_node": gpus, "nnodes": nodes}})

    assert _capsule_data_parallel_world_size(config) == expected


@pytest.mark.parametrize(("gpus", "nodes"), [(3, 1), (6, 1), (8, 2)])
def test_capsule_world_size_rejects_non_divisors_or_more_than_eight(
    gpus: int, nodes: int
) -> None:
    config = OmegaConf.create({"trainer": {"n_gpus_per_node": gpus, "nnodes": nodes}})

    with pytest.raises(ServerFactoryError, match="divisible"):
        _capsule_data_parallel_world_size(config)


def test_worker_session_requires_identical_optimizer_steps_from_all_ranks() -> None:
    actor = type("Actor", (), {"get_capsule_optimizer_step": lambda self: [3, 3]})()
    session = VeRLWorkerSession(
        actor_rollout_wg=actor,
        ref_policy_wg=object(),
        tokenizer=object(),
        data_proto_factory=lambda **_kwargs: object(),
        ray_module=object(),
        owns_ray=True,
    )

    assert session.optimizer_step() == 3
    actor.get_capsule_optimizer_step = lambda: [3, 4]
    with pytest.raises(ServerFactoryError, match="agree across ranks"):
        session.optimizer_step()


def test_worker_session_requires_matching_driver_and_all_rank_verl_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pinned = tmp_path / "pinned-verl"
    module_path = pinned / "verl" / "__init__.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("", encoding="utf-8")
    pinned_sha = "a" * 40
    report = type(
        "Report",
        (),
        {"actual_sha": pinned_sha, "source_path": str(pinned.resolve()), "compatible": True},
    )()
    monkeypatch.setattr(
        "capx.rl.capsule.server_factory.check_verl_compatibility",
        lambda _path, _sha: report,
    )
    records = [
        {
            "rank": rank,
            "source_path": str(pinned.resolve()),
            "module_path": str(module_path.resolve()),
            "expected_sha": pinned_sha,
            "actual_sha": pinned_sha,
            "clean": True,
        }
        for rank in range(2)
    ]
    actor = type(
        "Actor", (), {"get_capsule_verl_provenance": lambda self: records}
    )()
    session = VeRLWorkerSession(
        actor_rollout_wg=actor,
        ref_policy_wg=object(),
        tokenizer=object(),
        data_proto_factory=lambda **_kwargs: object(),
        ray_module=object(),
        owns_ray=True,
        data_parallel_world_size=2,
        verl_source_path=pinned,
        verl_pinned_sha=pinned_sha,
    )

    evidence = session.verl_provenance()

    assert evidence["actual_sha"] == pinned_sha
    assert evidence["clean"] is True
    assert evidence["worker_count"] == 2
    assert evidence["worker_ranks"] == [0, 1]

    records[1]["actual_sha"] = "b" * 40
    with pytest.raises(ServerFactoryError, match="provenance"):
        session.verl_provenance()


def test_runtime_preserves_setup_error_and_closes_workers(tmp_path: Path) -> None:
    task = TaskInstanceV1.from_dict(_task_row())
    events: list[str] = []

    class _Workers:
        total_epochs = 0

        def close(self) -> None:
            events.append("workers")
            raise RuntimeError("close failed")

    runtime = CapsuleServerRuntime(
        {
            "runtime": {"output_dir": str(tmp_path / "outputs")},
            "task": {},
            "capsule": {},
            "controller_service": {},
            "program_service": {},
        },
        worker_starter=lambda _config: _Workers(),  # type: ignore[arg-type]
        task_loader=lambda _config: (task,),
    )

    with pytest.raises(ServerFactoryError, match="total_epochs"):
        runtime.fit()

    assert events == ["workers"]


def test_runtime_teardown_attempts_all_resources_after_an_earlier_failure() -> None:
    events: list[str] = []

    class _Repair:
        def close(self) -> None:
            events.append("repair")
            raise RuntimeError("repair close failed")

    class _Evaluator:
        def close(self) -> None:
            events.append("evaluator")

    class _Workers:
        def close(self) -> None:
            events.append("workers")

    with pytest.raises(RuntimeError, match="repair close failed"):
        _close_runtime_resources(_Repair(), _Evaluator(), _Workers())  # type: ignore[arg-type]

    assert events == ["repair", "evaluator", "workers"]


def test_controller_transport_close_releases_existing_client_without_creating_one() -> None:
    events: list[str] = []

    class _Client:
        def close(self) -> None:
            events.append("client")

    transport = object.__new__(OpenAICompatibleControllerTransport)
    transport._client = _Client()

    transport.close()
    transport.close()

    assert events == ["client"]
