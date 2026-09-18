from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from capx.rl.capsule.schema import ProgramReplayResultV1, ReplayOutcome
from scripts.capsule_rl import common, cube_lift_privileged_replay_smoke as smoke


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAIN_CONFIG = (
    REPOSITORY_ROOT
    / "env_configs"
    / "cube_lifting"
    / "capsule_rl"
    / "franka_robosuite_cube_lift_capsule_smoke.yaml"
)
ENVIRONMENT_CONFIG = MAIN_CONFIG.with_name(
    "franka_robosuite_cube_lift_privileged_clean_replay.yaml"
)
SOURCE_TASK = MAIN_CONFIG.with_name("cube_lift_capsule_source_tasks.jsonl")
HASH_5 = "5" * 64
HASH_6 = "6" * 64


def _write_inputs(
    tmp_path: Path,
    *,
    config_changes: dict[str, Any] | None = None,
    environment_changes: dict[str, Any] | None = None,
    source_rows: list[Any] | None = None,
) -> tuple[Path, Path, Path]:
    config = yaml.safe_load(MAIN_CONFIG.read_text(encoding="utf-8"))
    environment = yaml.safe_load(ENVIRONMENT_CONFIG.read_text(encoding="utf-8"))
    canonical_source = json.loads(SOURCE_TASK.read_text(encoding="utf-8"))
    if config_changes:
        config["task"].update(config_changes)
    if environment_changes:
        environment.update(environment_changes)

    environment_path = tmp_path / "environment.yaml"
    environment_path.write_text(
        yaml.safe_dump(environment, sort_keys=False), encoding="utf-8"
    )
    config["task"]["config_path"] = str(environment_path)
    config_path = tmp_path / "capsule.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    source_path = tmp_path / "source.jsonl"
    rows = [canonical_source] if source_rows is None else source_rows
    source_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return config_path, environment_path, source_path


def _success_result(
    *,
    initial_hash: str = HASH_5,
    program_sample_id: str = "cube-lift-red-cube:seed-5:oracle",
    attempt_history: list[dict[str, Any]] | None = None,
    reset_evidence: dict[str, Any] | None = None,
) -> ProgramReplayResultV1:
    history = (
        [
            {
                "attempt": 1,
                "outcome": "success",
                "worker_replaced": False,
                "retry_scheduled": False,
                "error_type": None,
                "error_message": None,
            }
        ]
        if attempt_history is None
        else attempt_history
    )
    evidence = (
        {
            "namespace_fresh": True,
            "api_state_cleared": True,
            "api_reset_count": 1,
            "api_reset_confirmed_count": 1,
        }
        if reset_evidence is None
        else reset_evidence
    )
    return ProgramReplayResultV1(
        task_id="cube-lift-red-cube",
        environment_seed=5,
        program_sample_id=program_sample_id,
        source="open_gripper()\nclose_gripper()",
        initial_state_sha256=initial_hash,
        outcome=ReplayOutcome.SUCCESS,
        raw_reward=1.0,
        binary_reward=1.0,
        task_completed=True,
        terminated=True,
        attempts=1,
        diagnostics={
            "reset_info": {
                "initial_state_sha256": initial_hash,
                "capsule_reset_evidence": evidence,
            },
            "evaluator_attempt_history": history,
        },
    )


class _Probe:
    def __init__(
        self,
        hashes: tuple[str, str, str] = (HASH_5, HASH_6, HASH_5),
        oracle_code: object = "open_gripper()\nclose_gripper()",
    ) -> None:
        self.hashes = iter(hashes)
        self.oracle_code = oracle_code
        self.reset_calls: list[tuple[int, dict[str, Any]]] = []
        self.closed = False

    def reset(self, *, seed: int, options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        self.reset_calls.append((seed, options))
        return {}, {"initial_state_sha256": next(self.hashes)}

    def close(self) -> None:
        self.closed = True


def _runtime_components(
    *,
    results: list[ProgramReplayResultV1] | None = None,
    pids: list[int | None] | None = None,
    probe: _Probe | None = None,
) -> SimpleNamespace:
    state = SimpleNamespace(
        probe=_Probe() if probe is None else probe,
        factory_instances=[],
        backend_instances=[],
        evaluator_instances=[],
        results=list(results or [_success_result(), _success_result()]),
        pids=list(pids or [1701, 1701]),
    )

    class Factory:
        def __init__(self, config_path: str, config_bytes: bytes | None = None) -> None:
            self.config_path = config_path
            self.config_bytes = config_bytes
            self.calls: list[object] = []
            state.factory_instances.append(self)

        def __call__(self, task: object) -> _Probe:
            self.calls.append(task)
            return state.probe

    class Backend:
        def __init__(self, factory: object, *, start_method: str) -> None:
            self.factory = factory
            self.start_method = start_method
            self.current_pid: int | None = None
            self.closed = False
            state.backend_instances.append(self)

        @property
        def worker_pid(self) -> int | None:
            return self.current_pid

        def close(self) -> None:
            self.closed = True

    class Evaluator:
        def __init__(
            self, backend: Backend, *, timeout_s: float, max_failure_retries: int
        ) -> None:
            self.backend = backend
            self.timeout_s = timeout_s
            self.max_failure_retries = max_failure_retries
            self.calls: list[tuple[object, str, int, str | None]] = []
            self.closed = False
            state.evaluator_instances.append(self)

        def evaluate_program(
            self,
            task: object,
            source: str,
            seed: int,
            *,
            program_sample_id: str | None = None,
        ) -> ProgramReplayResultV1:
            index = len(self.calls)
            self.calls.append((task, source, seed, program_sample_id))
            self.backend.current_pid = state.pids[index]
            return state.results[index]

        def close(self) -> None:
            self.closed = True
            self.backend.close()

    state.components = smoke.RuntimeComponents(
        environment_factory_type=Factory,
        replay_backend_type=Backend,
        replay_evaluator_type=Evaluator,
    )
    return state


def _execute_with_fakes(
    tmp_path: Path,
    *,
    state: SimpleNamespace | None = None,
) -> tuple[dict[str, Any], SimpleNamespace, Path]:
    config_path, _environment_path, source_path = _write_inputs(tmp_path)
    inputs = smoke.load_smoke_inputs(config_path, source_path)
    runtime = _runtime_components() if state is None else state
    output = tmp_path / "cube_lift_smoke.json"
    payload = smoke.execute_smoke(
        inputs,
        seed_sequence=(5, 6, 5),
        replay_seed=5,
        replays=2,
        timeout_s=180.0,
        output_path=output,
        readiness_checker=lambda host, port: None,
        runtime_loader=lambda: runtime.components,
    )
    return payload, runtime, output


def test_load_smoke_inputs_validates_profile_environment_and_source(tmp_path: Path) -> None:
    config_path, environment_path, source_path = _write_inputs(tmp_path)

    inputs = smoke.load_smoke_inputs(config_path, source_path)

    assert inputs.profile.name == "robosuite_cube_lift_privileged_highlevel"
    assert inputs.environment_path == environment_path.resolve()
    assert inputs.source_row == json.loads(source_path.read_text(encoding="utf-8"))
    assert (inputs.pyroki_host, inputs.pyroki_port) == ("127.0.0.1", 8116)
    assert inputs.config_sha256 == common.artifact_file_sha256(config_path)
    assert inputs.environment_sha256 == common.artifact_file_sha256(environment_path)
    assert inputs.source_sha256 == common.artifact_file_sha256(source_path)


def test_load_smoke_inputs_hashes_the_bytes_it_validated_when_files_mutate_after_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, environment_path, source_path = _write_inputs(tmp_path)
    original_bytes = {
        "config": config_path.read_bytes(),
        "environment": environment_path.read_bytes(),
        "source": source_path.read_bytes(),
    }
    real_source_loader = smoke._load_source_row

    def mutate_after_all_reads(raw_bytes: bytes, path: Path) -> dict[str, str]:
        row = dict(real_source_loader(raw_bytes, path))
        config_path.write_bytes(b"changed config bytes\n")
        environment_path.write_bytes(b"changed environment bytes\n")
        source_path.write_bytes(b"changed source bytes\n")
        return row

    monkeypatch.setattr(smoke, "_load_source_row", mutate_after_all_reads)

    inputs = smoke.load_smoke_inputs(config_path, source_path)

    assert config_path.read_bytes() != original_bytes["config"]
    assert environment_path.read_bytes() != original_bytes["environment"]
    assert source_path.read_bytes() != original_bytes["source"]
    assert inputs.config_sha256 == hashlib.sha256(original_bytes["config"]).hexdigest()
    assert inputs.environment_sha256 == hashlib.sha256(
        original_bytes["environment"]
    ).hexdigest()
    assert inputs.source_sha256 == hashlib.sha256(original_bytes["source"]).hexdigest()


@pytest.mark.parametrize(
    "task_changes",
    (
        {"profile": "robosuite_cube_stack_privileged", "environment": "robosuite_cube_stack"},
        {"profile": None},
    ),
)
def test_load_smoke_inputs_rejects_non_explicit_lift_profile(
    tmp_path: Path, task_changes: dict[str, Any]
) -> None:
    config_path, _environment_path, source_path = _write_inputs(
        tmp_path, config_changes=task_changes
    )

    with pytest.raises(smoke.CubeLiftSmokeError, match="explicit.*Cube Lift profile"):
        smoke.load_smoke_inputs(config_path, source_path)


@pytest.mark.parametrize(
    "source_rows, message",
    (
        ([], "exactly one"),
        ([{}, {}], "exactly one"),
        ([[]], "mapping"),
        ([{"task_id": "wrong", "prompt": "wrong"}], "canonical"),
        ([{"task_id": "cube-lift-red-cube", "prompt": "wrong"}], "canonical"),
        (
            [
                {
                    **json.loads(SOURCE_TASK.read_text(encoding="utf-8")),
                    "environment_seed": 5,
                }
            ],
            "exact keys",
        ),
    ),
)
def test_load_smoke_inputs_rejects_source_contract_drift(
    tmp_path: Path, source_rows: list[Any], message: str
) -> None:
    config_path, _environment_path, source_path = _write_inputs(
        tmp_path, source_rows=source_rows
    )

    with pytest.raises(smoke.CubeLiftSmokeError, match=message):
        smoke.load_smoke_inputs(config_path, source_path)


def test_load_smoke_inputs_uses_environment_profile_validator(tmp_path: Path) -> None:
    config_path, _environment_path, source_path = _write_inputs(
        tmp_path, environment_changes={"record_video": True}
    )

    with pytest.raises(smoke.CubeLiftSmokeError, match="record_video"):
        smoke.load_smoke_inputs(config_path, source_path)


def test_validate_only_stops_before_socket_runtime_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _environment_path, source_path = _write_inputs(tmp_path)
    output = tmp_path / "must-not-exist.json"

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("validate-only crossed a side-effect boundary")

    monkeypatch.setattr(smoke, "_check_pyroki_ready", forbidden)
    monkeypatch.setattr(smoke, "_load_runtime_components", forbidden)
    monkeypatch.setattr(common, "atomic_write_json", forbidden)

    assert smoke.main(
        [
            "--config",
            str(config_path),
            "--source-task",
            str(source_path),
            "--output",
            str(output),
            "--validate-only",
        ]
    ) == 0
    assert not output.exists()


@pytest.mark.parametrize(
    "changes, message",
    (
        ({"seed_sequence": (5, 5, 6)}, "5,6,5"),
        ({"replay_seed": 6}, "replay seed 5"),
        ({"replays": 1}, "exactly two"),
        ({"timeout_s": 0.0}, "finite and positive"),
        ({"timeout_s": float("inf")}, "finite and positive"),
    ),
)
def test_execution_contract_rejects_wrong_arguments(
    tmp_path: Path, changes: dict[str, Any], message: str
) -> None:
    arguments: dict[str, Any] = {
        "seed_sequence": (5, 6, 5),
        "replay_seed": 5,
        "replays": 2,
        "timeout_s": 180.0,
        "output_path": tmp_path / "smoke.json",
    }
    arguments.update(changes)

    with pytest.raises(smoke.CubeLiftSmokeError, match=message):
        smoke.validate_execution_contract(**arguments)


def test_execution_contract_rejects_finite_noncanonical_timeout(tmp_path: Path) -> None:
    with pytest.raises(smoke.CubeLiftSmokeError, match="exactly 180"):
        smoke.validate_execution_contract(
            seed_sequence=(5, 6, 5),
            replay_seed=5,
            replays=2,
            timeout_s=179.0,
            output_path=tmp_path / "smoke.json",
        )


@pytest.mark.parametrize("name", ("gate03_oracle.json", "gate-07-result.json"))
def test_execution_contract_rejects_formal_gate_artifact_names(
    tmp_path: Path, name: str
) -> None:
    with pytest.raises(smoke.CubeLiftSmokeError, match="formal Gate"):
        smoke.validate_execution_contract(
            seed_sequence=(5, 6, 5),
            replay_seed=5,
            replays=2,
            timeout_s=180.0,
            output_path=tmp_path / name,
        )


def test_existing_output_fails_before_socket_or_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _environment_path, source_path = _write_inputs(tmp_path)
    output = tmp_path / "smoke.json"
    output.write_text("original\n", encoding="utf-8")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("collision must fail before runtime")

    monkeypatch.setattr(smoke, "_check_pyroki_ready", forbidden)
    monkeypatch.setattr(smoke, "_load_runtime_components", forbidden)

    with pytest.raises(FileExistsError, match="already exists"):
        smoke.main(
            [
                "--config",
                str(config_path),
                "--source-task",
                str(source_path),
                "--output",
                str(output),
            ]
        )
    assert output.read_text(encoding="utf-8") == "original\n"


def test_pyroki_readiness_failure_precedes_runtime_import_and_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _environment_path, source_path = _write_inputs(tmp_path)

    def not_ready(host: str, port: int) -> None:
        assert (host, port) == ("127.0.0.1", 8116)
        raise smoke.CubeLiftSmokeError("PyRoKi endpoint is not ready")

    def forbidden_runtime() -> None:
        raise AssertionError("runtime modules must load only after readiness")

    monkeypatch.setattr(smoke, "_check_pyroki_ready", not_ready)
    monkeypatch.setattr(smoke, "_load_runtime_components", forbidden_runtime)

    with pytest.raises(smoke.CubeLiftSmokeError, match="PyRoKi.*not ready"):
        smoke.main(
            [
                "--config",
                str(config_path),
                "--source-task",
                str(source_path),
                "--output",
                str(tmp_path / "smoke.json"),
            ]
        )


def test_seed_hash_contract_accepts_only_repeatable_seed_five() -> None:
    assert smoke.validate_seed_hashes((5, 6, 5), (HASH_5, HASH_6, HASH_5)) == HASH_5


@pytest.mark.parametrize(
    "hashes",
    (
        (HASH_5, HASH_6, "7" * 64),
        (HASH_5, HASH_5, HASH_5),
        ("A" * 64, HASH_6, "A" * 64),
        ("short", HASH_6, "short"),
    ),
)
def test_seed_hash_contract_reports_all_hashes_on_failure(hashes: tuple[str, str, str]) -> None:
    with pytest.raises(smoke.CubeLiftSmokeError) as caught:
        smoke.validate_seed_hashes((5, 6, 5), hashes)

    for initial_hash in hashes:
        assert initial_hash in str(caught.value)


def test_build_task_instance_uses_source_profile_and_real_seed_hash(tmp_path: Path) -> None:
    config_path, _environment_path, source_path = _write_inputs(tmp_path)
    inputs = smoke.load_smoke_inputs(config_path, source_path)

    task = smoke.build_task_instance(inputs, environment_seed=5, initial_hash=HASH_5)

    assert task.to_dict() == {
        "schema_version": 1,
        "task_id": "cube-lift-red-cube",
        "environment_seed": 5,
        "prompt": inputs.source_row["prompt"],
        "environment": "robosuite_cube_lift",
        "api": "franka_control_privileged",
        "privilege": "privileged",
        "initial_state_sha256": HASH_5,
        "metadata": {"task_profile": "robosuite_cube_lift_privileged_highlevel"},
    }


def test_execute_smoke_uses_one_backend_and_evaluator_for_both_replays(
    tmp_path: Path,
) -> None:
    payload, state, output = _execute_with_fakes(tmp_path)

    assert state.probe.reset_calls == [
        (5, {"capsule_smoke": "seed"}),
        (6, {"capsule_smoke": "seed"}),
        (5, {"capsule_smoke": "seed"}),
    ]
    assert state.probe.closed is True
    assert len(state.factory_instances) == 1
    assert state.factory_instances[0].calls == [None]
    assert len(state.backend_instances) == 1
    assert state.backend_instances[0].start_method == "spawn"
    assert len(state.evaluator_instances) == 1
    evaluator = state.evaluator_instances[0]
    assert evaluator.timeout_s == 180.0
    assert evaluator.max_failure_retries == 0
    assert len(evaluator.calls) == 2
    assert {call[3] for call in evaluator.calls} == {
        "cube-lift-red-cube:seed-5:oracle"
    }
    assert evaluator.closed is True
    assert state.backend_instances[0].closed is True
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_execute_smoke_pins_validated_environment_bytes_before_probe(
    tmp_path: Path,
) -> None:
    config_path, environment_path, source_path = _write_inputs(tmp_path)
    validated_environment_bytes = environment_path.read_bytes()
    inputs = smoke.load_smoke_inputs(config_path, source_path)
    environment_path.write_bytes(b"env: {_target_: changed.Environment}\n")
    state = _runtime_components()

    payload = smoke.execute_smoke(
        inputs,
        seed_sequence=(5, 6, 5),
        replay_seed=5,
        replays=2,
        timeout_s=180.0,
        output_path=tmp_path / "smoke.json",
        readiness_checker=lambda host, port: None,
        runtime_loader=lambda: state.components,
    )

    assert state.factory_instances[0].config_bytes == validated_environment_bytes
    assert payload["environment_sha256"] == hashlib.sha256(
        validated_environment_bytes
    ).hexdigest()


def test_execute_smoke_artifact_is_explicitly_non_training_and_non_gate(tmp_path: Path) -> None:
    payload, _state, _output = _execute_with_fakes(tmp_path)

    assert payload["mode"] == "cube_lift_privileged_replay_smoke_v1"
    assert "gate" not in payload["mode"].lower()
    assert payload["passed"] is True
    assert payload["seed_sequence"] == [5, 6, 5]
    assert payload["initial_state_sha256"] == [HASH_5, HASH_6, HASH_5]
    assert payload["worker_ids"] == [1701, 1701]
    assert len(payload["replays"]) == 2
    assert all(record["outcome"] == "success" for record in payload["replays"])
    assert payload["profile"] == {
        "name": "robosuite_cube_lift_privileged_highlevel",
        "environment": "robosuite_cube_lift",
        "api": "franka_control_privileged",
        "privilege": "privileged",
    }
    assert payload["render_enabled"] is False
    assert payload["record_video"] is False
    for field_name in (
        "program_actor_used",
        "controller_used",
        "ray_used",
        "verl_used",
        "optimizer_used",
    ):
        assert payload[field_name] is False
    assert "runtime_verified" not in payload
    assert payload["config_sha256"] == common.artifact_file_sha256(payload["config_path"])
    assert payload["environment_sha256"] == common.artifact_file_sha256(
        payload["environment_path"]
    )
    assert payload["source_task_sha256"] == common.artifact_file_sha256(
        payload["source_task_path"]
    )


def test_execute_smoke_closes_probe_when_seed_contract_fails(tmp_path: Path) -> None:
    state = _runtime_components(probe=_Probe((HASH_5, HASH_5, HASH_5)))
    config_path, _environment_path, source_path = _write_inputs(tmp_path)
    inputs = smoke.load_smoke_inputs(config_path, source_path)

    with pytest.raises(smoke.CubeLiftSmokeError, match="seed reset hashes"):
        smoke.execute_smoke(
            inputs,
            seed_sequence=(5, 6, 5),
            replay_seed=5,
            replays=2,
            timeout_s=180.0,
            output_path=tmp_path / "smoke.json",
            readiness_checker=lambda host, port: None,
            runtime_loader=lambda: state.components,
        )

    assert state.probe.closed is True
    assert state.backend_instances == []


def test_execute_smoke_closes_evaluator_when_replay_validation_fails(tmp_path: Path) -> None:
    bad = _success_result(initial_hash="7" * 64)
    state = _runtime_components(results=[_success_result(), bad])
    config_path, _environment_path, source_path = _write_inputs(tmp_path)
    inputs = smoke.load_smoke_inputs(config_path, source_path)

    with pytest.raises(smoke.CubeLiftSmokeError, match="initial state hash"):
        smoke.execute_smoke(
            inputs,
            seed_sequence=(5, 6, 5),
            replay_seed=5,
            replays=2,
            timeout_s=180.0,
            output_path=tmp_path / "smoke.json",
            readiness_checker=lambda host, port: None,
            runtime_loader=lambda: state.components,
        )

    assert state.evaluator_instances[0].closed is True
    assert state.backend_instances[0].closed is True
    assert not (tmp_path / "smoke.json").exists()


@pytest.mark.parametrize(
    "field_name, mutation, message",
    (
        ("pid", lambda result: None, "worker PID"),
        (
            "outcome",
            lambda result: object.__setattr__(result, "outcome", ReplayOutcome.TASK_FAILURE),
            "outcome",
        ),
        (
            "binary_reward",
            lambda result: object.__setattr__(result, "binary_reward", 0.0),
            "binary reward",
        ),
        (
            "task_completed",
            lambda result: object.__setattr__(result, "task_completed", False),
            "task completion",
        ),
        (
            "attempts",
            lambda result: object.__setattr__(result, "attempts", 2),
            "exactly one attempt",
        ),
        (
            "attempt_history",
            lambda result: object.__setattr__(
                result,
                "diagnostics",
                {
                    **dict(result.diagnostics),
                    "evaluator_attempt_history": [
                        {
                            "attempt": 1,
                            "outcome": "success",
                            "worker_replaced": True,
                            "retry_scheduled": False,
                        }
                    ],
                },
            ),
            "attempt history",
        ),
        (
            "reset_evidence",
            lambda result: object.__setattr__(
                result,
                "diagnostics",
                {
                    **dict(result.diagnostics),
                    "reset_info": {
                        "initial_state_sha256": HASH_5,
                        "capsule_reset_evidence": {
                            "namespace_fresh": True,
                            "api_state_cleared": False,
                            "api_reset_count": 1,
                            "api_reset_confirmed_count": 0,
                        },
                    },
                },
            ),
            "reset evidence",
        ),
    ),
)
def test_execute_smoke_fails_closed_on_replay_evidence_drift(
    tmp_path: Path,
    field_name: str,
    mutation: Any,
    message: str,
) -> None:
    first = _success_result()
    second = _success_result()
    pids: list[int | None] = [1701, 1701]
    if field_name == "pid":
        pids[1] = None
    else:
        mutation(second)
    state = _runtime_components(results=[first, second], pids=pids)
    config_path, _environment_path, source_path = _write_inputs(tmp_path)
    inputs = smoke.load_smoke_inputs(config_path, source_path)

    with pytest.raises(smoke.CubeLiftSmokeError, match=message):
        smoke.execute_smoke(
            inputs,
            seed_sequence=(5, 6, 5),
            replay_seed=5,
            replays=2,
            timeout_s=180.0,
            output_path=tmp_path / "smoke.json",
            readiness_checker=lambda host, port: None,
            runtime_loader=lambda: state.components,
        )

    assert state.evaluator_instances[0].closed is True
    assert not (tmp_path / "smoke.json").exists()


def test_execute_smoke_rejects_worker_pid_drift(tmp_path: Path) -> None:
    state = _runtime_components(pids=[1701, 1702])
    config_path, _environment_path, source_path = _write_inputs(tmp_path)
    inputs = smoke.load_smoke_inputs(config_path, source_path)

    with pytest.raises(smoke.CubeLiftSmokeError, match="same persistent worker"):
        smoke.execute_smoke(
            inputs,
            seed_sequence=(5, 6, 5),
            replay_seed=5,
            replays=2,
            timeout_s=180.0,
            output_path=tmp_path / "smoke.json",
            readiness_checker=lambda host, port: None,
            runtime_loader=lambda: state.components,
        )

    assert state.evaluator_instances[0].closed is True


def test_execute_smoke_uses_atomic_immutable_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published: list[tuple[Path, dict[str, Any]]] = []
    real_atomic_write = common.atomic_write_json

    def recording_publish(path: str | Path, payload: dict[str, Any]) -> Path:
        published.append((Path(path), deepcopy(payload)))
        return real_atomic_write(path, payload)

    monkeypatch.setattr(common, "atomic_write_json", recording_publish)

    payload, _state, output = _execute_with_fakes(tmp_path)

    assert published == [(output, payload)]
    with pytest.raises(FileExistsError, match="already exists"):
        common.atomic_write_json(output, payload)
