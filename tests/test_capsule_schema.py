import json
import pickle

import pytest

from capx.rl.capsule.schema import (
    CommittedEditV1,
    LearningGroupV1,
    LearningMemberV1,
    ProgramReplayResultV1,
    ReplayOutcome,
    RepairAuditV1,
    RepairTraceV1,
    SourceUnitV1,
    TaskInstanceV1,
    source_sha256,
)


def make_task() -> TaskInstanceV1:
    return TaskInstanceV1(
        task_id="cube-stack-5",
        environment_seed=5,
        prompt="Stack the red cube on the blue cube.",
        environment="RobosuiteCubeStack",
        api="FrankaControlPrivilegedApi",
        privilege="privileged",
        initial_state_sha256="a" * 64,
    )


def test_task_schema_json_round_trip_preserves_seed_and_environment_contract():
    task = make_task()

    restored = TaskInstanceV1.from_json(task.to_json())

    assert restored == task
    assert json.loads(task.to_json())["schema_version"] == 1
    assert restored.environment_seed == 5
    assert restored.api == "FrankaControlPrivilegedApi"


def test_schema_version_rejects_bool_and_float_aliases() -> None:
    for invalid in (True, 1.0):
        payload = make_task().to_dict()
        payload["schema_version"] = invalid
        with pytest.raises((TypeError, ValueError), match="schema_version"):
            TaskInstanceV1.from_dict(payload)


@pytest.mark.parametrize("attempts", [True, 1.5, 0])
def test_replay_attempts_require_a_positive_integer(attempts: object) -> None:
    with pytest.raises((TypeError, ValueError), match="attempts"):
        ProgramReplayResultV1(
            task_id="cube-stack-5",
            environment_seed=5,
            program_sample_id="sample-3",
            source="move()",
            initial_state_sha256="a" * 64,
            outcome=ReplayOutcome.TASK_FAILURE,
            raw_reward=0.1,
            binary_reward=0.0,
            task_completed=False,
            attempts=attempts,  # type: ignore[arg-type]
        )


def test_task_failure_cannot_encode_completed_success_facts() -> None:
    with pytest.raises(ValueError, match="non-success"):
        ProgramReplayResultV1(
            task_id="cube-stack-5",
            environment_seed=5,
            program_sample_id="sample-3",
            source="move()",
            initial_state_sha256="a" * 64,
            outcome=ReplayOutcome.TASK_FAILURE,
            raw_reward=1.0,
            binary_reward=0.0,
            task_completed=True,
        )


@pytest.mark.parametrize("error_type", [None, ""])
def test_program_error_requires_nonempty_error_type(error_type: str | None) -> None:
    with pytest.raises(ValueError, match="PROGRAM_ERROR.*error_type"):
        ProgramReplayResultV1(
            task_id="cube-stack-5",
            environment_seed=5,
            program_sample_id="sample-3",
            source="move()",
            initial_state_sha256="a" * 64,
            outcome=ReplayOutcome.PROGRAM_ERROR,
            raw_reward=0.0,
            binary_reward=0.0,
            task_completed=False,
            error_type=error_type,
        )


def test_step_budget_exhausted_requires_truncated_true() -> None:
    with pytest.raises(ValueError, match="STEP_BUDGET_EXHAUSTED.*truncated"):
        ProgramReplayResultV1(
            task_id="cube-stack-5",
            environment_seed=5,
            program_sample_id="sample-3",
            source="move()",
            initial_state_sha256="a" * 64,
            outcome=ReplayOutcome.STEP_BUDGET_EXHAUSTED,
            raw_reward=0.0,
            binary_reward=0.0,
            task_completed=False,
            truncated=False,
        )


@pytest.mark.parametrize("outcome", [ReplayOutcome.INFRA_ERROR, ReplayOutcome.EVALUATOR_ERROR])
def test_unknown_reward_outcomes_require_raw_reward_to_be_null(outcome: ReplayOutcome) -> None:
    with pytest.raises(ValueError, match="infra/evaluator raw_reward must be null"):
        ProgramReplayResultV1(
            task_id="cube-stack-5",
            environment_seed=5,
            program_sample_id="sample-3",
            source="move()",
            initial_state_sha256="a" * 64,
            outcome=outcome,
            raw_reward=0.0,
            binary_reward=None,
            task_completed=False,
        )


def test_program_timeout_allows_watchdog_result_without_environment_truncation() -> None:
    result = ProgramReplayResultV1(
        task_id="cube-stack-5",
        environment_seed=5,
        program_sample_id="sample-3",
        source="while True: pass",
        initial_state_sha256="a" * 64,
        outcome=ReplayOutcome.PROGRAM_TIMEOUT,
        raw_reward=None,
        binary_reward=0.0,
        task_completed=False,
        truncated=False,
        error_type="WorkerTimedOutError",
    )

    assert result.truncated is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"truncated": True},
        {"error_type": "ValueError"},
        {"raw_reward": None},
    ],
)
def test_task_failure_matches_evaluator_classifier_inverse(overrides: dict) -> None:
    arguments = {
        "task_id": "cube-stack-5",
        "environment_seed": 5,
        "program_sample_id": "sample-3",
        "source": "move()",
        "initial_state_sha256": "a" * 64,
        "outcome": ReplayOutcome.TASK_FAILURE,
        "raw_reward": 0.2,
        "binary_reward": 0.0,
        "task_completed": False,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match="TASK_FAILURE"):
        ProgramReplayResultV1(**arguments)


@pytest.mark.parametrize(
    "overrides",
    [
        {"raw_reward": 0.0},
        {"error_type": None},
        {"truncated": True},
    ],
)
def test_program_timeout_matches_watchdog_classifier_inverse(overrides: dict) -> None:
    arguments = {
        "task_id": "cube-stack-5",
        "environment_seed": 5,
        "program_sample_id": "sample-3",
        "source": "while True: pass",
        "initial_state_sha256": "a" * 64,
        "outcome": ReplayOutcome.PROGRAM_TIMEOUT,
        "raw_reward": None,
        "binary_reward": 0.0,
        "task_completed": False,
        "truncated": False,
        "error_type": "WorkerTimedOutError",
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match="PROGRAM_TIMEOUT"):
        ProgramReplayResultV1(**arguments)


@pytest.mark.parametrize(
    ("outcome", "error_type"),
    [
        (ReplayOutcome.STEP_BUDGET_EXHAUSTED, "ValueError"),
        (ReplayOutcome.PROGRAM_ERROR, "ValueError"),
    ],
)
def test_payload_semantic_outcomes_require_numeric_raw_reward(
    outcome: ReplayOutcome, error_type: str
) -> None:
    with pytest.raises(ValueError, match="raw_reward"):
        ProgramReplayResultV1(
            task_id="cube-stack-5",
            environment_seed=5,
            program_sample_id="sample-3",
            source="move()",
            initial_state_sha256="a" * 64,
            outcome=outcome,
            raw_reward=None,
            binary_reward=0.0,
            task_completed=False,
            truncated=outcome is ReplayOutcome.STEP_BUDGET_EXHAUSTED,
            error_type=error_type,
        )


def test_learning_contract_rejects_bool_reward_duplicate_ids_and_non_bool_skip() -> None:
    with pytest.raises((TypeError, ValueError), match="reward"):
        LearningMemberV1(
            member_type="base",
            program_sample_id="duplicate",
            prompt="stack",
            response="pass",
            reward=True,  # type: ignore[arg-type]
        )
    member = LearningMemberV1(
        member_type="base",
        program_sample_id="duplicate",
        prompt="stack",
        response="pass",
        reward=0.0,
    )
    with pytest.raises(ValueError, match="program_sample_id"):
        LearningGroupV1(
            task_id="cube-stack-5",
            environment_seed=5,
            group_uid="group",
            initial_state_sha256="a" * 64,
            members=(member, member),
        )
    with pytest.raises(TypeError, match="skip_actor_update"):
        LearningGroupV1(
            task_id="cube-stack-5",
            environment_seed=5,
            group_uid="group",
            initial_state_sha256="a" * 64,
            members=(member,),
            skip_actor_update=0,  # type: ignore[arg-type]
        )


def test_json_artifact_mappings_are_recursively_frozen_and_pickle_safe():
    metadata = {"nested": {"values": [1, 2]}}
    task = TaskInstanceV1(
        task_id="cube-stack-5",
        environment_seed=5,
        prompt="Stack cubes",
        environment="RobosuiteCubeStack",
        api="FrankaControlPrivilegedApi",
        privilege="privileged",
        initial_state_sha256="a" * 64,
        metadata=metadata,
    )

    metadata["nested"]["values"].append(3)

    assert task.metadata["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        task.metadata["new"] = True
    with pytest.raises(TypeError):
        task.metadata["nested"]["new"] = True
    assert pickle.loads(pickle.dumps(task)) == task
    assert TaskInstanceV1.from_json(task.to_json()) == task


def test_all_mapping_artifacts_detach_from_mutable_constructor_inputs():
    diagnostics = {"nested": {"status": "before"}}
    action = {"args": {"target": "base:move"}}
    member_metadata = {"provenance": {"rank": 0}}
    group_metadata = {"audit": {"selected": False}}
    replay = ProgramReplayResultV1(
        task_id="cube-stack-5",
        environment_seed=5,
        program_sample_id="sample-3",
        source="move()",
        initial_state_sha256="a" * 64,
        outcome=ReplayOutcome.TASK_FAILURE,
        raw_reward=0.1,
        binary_reward=0.0,
        task_completed=False,
        diagnostics=diagnostics,
    )
    audit = RepairAuditV1(
        task_id="cube-stack-5",
        environment_seed=5,
        program_sample_id="sample-3",
        repair_trajectory_id="repair-1",
        turn_index=1,
        event_type="inspect",
        status="observed",
        action=action,
    )
    member = LearningMemberV1(
        member_type="base",
        program_sample_id="sample-3",
        prompt="Stack cubes",
        response="move()",
        reward=0.0,
        metadata=member_metadata,
    )
    group = LearningGroupV1(
        task_id="cube-stack-5",
        environment_seed=5,
        group_uid="uid",
        members=(member,),
        initial_state_sha256="a" * 64,
        metadata=group_metadata,
    )

    diagnostics["nested"]["status"] = "after"
    action["args"]["target"] = "base:other"
    member_metadata["provenance"]["rank"] = 9
    group_metadata["audit"]["selected"] = True

    assert replay.diagnostics["nested"]["status"] == "before"
    assert audit.action["args"]["target"] == "base:move"
    assert member.metadata["provenance"]["rank"] == 0
    assert group.metadata["audit"]["selected"] is False


def test_schema_version_is_rejected_instead_of_silently_migrated():
    payload = make_task().to_dict()
    payload["schema_version"] = 2

    with pytest.raises(ValueError, match="schema_version"):
        TaskInstanceV1.from_dict(payload)


@pytest.mark.parametrize("outcome", [ReplayOutcome.INFRA_ERROR, ReplayOutcome.EVALUATOR_ERROR])
def test_infrastructure_outcomes_allow_unknown_binary_reward(outcome: ReplayOutcome):
    result = ProgramReplayResultV1(
        task_id="cube-stack-5",
        environment_seed=5,
        program_sample_id="sample-3",
        source="move()",
        initial_state_sha256="a" * 64,
        outcome=outcome,
        raw_reward=None,
        binary_reward=None,
        task_completed=False,
    )

    assert ProgramReplayResultV1.from_json(result.to_json()) == result


def test_semantic_outcome_requires_binary_reward():
    with pytest.raises(ValueError, match="binary_reward"):
        ProgramReplayResultV1(
            task_id="cube-stack-5",
            environment_seed=5,
            program_sample_id="sample-3",
            source="move()",
            initial_state_sha256="a" * 64,
            outcome=ReplayOutcome.TASK_FAILURE,
            raw_reward=0.25,
            binary_reward=None,
            task_completed=False,
        )


def test_replay_outcome_must_be_typed_at_construction_time():
    with pytest.raises(TypeError, match="outcome"):
        ProgramReplayResultV1(
            task_id="cube-stack-5",
            environment_seed=5,
            program_sample_id="sample-3",
            source="move()",
            initial_state_sha256="a" * 64,
            outcome=123,
            raw_reward=0.0,
            binary_reward=0.0,
            task_completed=False,
        )


@pytest.mark.parametrize("field_name", ["task_completed", "terminated", "truncated"])
def test_replay_boolean_fields_are_strict(field_name):
    arguments = {
        "task_id": "cube-stack-5",
        "environment_seed": 5,
        "program_sample_id": "sample-3",
        "source": "move()",
        "initial_state_sha256": "a" * 64,
        "outcome": ReplayOutcome.TASK_FAILURE,
        "raw_reward": 0.0,
        "binary_reward": 0.0,
        "task_completed": False,
    }
    arguments[field_name] = "false"

    with pytest.raises(TypeError, match=field_name):
        ProgramReplayResultV1(**arguments)


def test_replay_rewards_and_json_payloads_reject_non_finite_numbers():
    with pytest.raises(ValueError, match="raw_reward"):
        ProgramReplayResultV1(
            task_id="cube-stack-5",
            environment_seed=5,
            program_sample_id="sample-3",
            source="move()",
            initial_state_sha256="a" * 64,
            outcome=ReplayOutcome.SUCCESS,
            raw_reward=float("nan"),
            binary_reward=1.0,
            task_completed=True,
        )

    with pytest.raises(ValueError, match="finite"):
        ProgramReplayResultV1(
            task_id="cube-stack-5",
            environment_seed=5,
            program_sample_id="sample-3",
            source="move()",
            initial_state_sha256="a" * 64,
            outcome=ReplayOutcome.INFRA_ERROR,
            raw_reward=None,
            binary_reward=None,
            task_completed=False,
            diagnostics={"latency": float("nan")},
        )


def test_replay_result_hashes_source_and_round_trips_typed_outcome():
    result = ProgramReplayResultV1(
        task_id="cube-stack-5",
        environment_seed=5,
        program_sample_id="sample-3",
        source="move()\n",
        initial_state_sha256="a" * 64,
        outcome=ReplayOutcome.SUCCESS,
        raw_reward=1.0,
        binary_reward=1.0,
        task_completed=True,
    )

    payload = result.to_dict()
    restored = ProgramReplayResultV1.from_dict(payload)

    assert payload["outcome"] == "success"
    assert restored.outcome is ReplayOutcome.SUCCESS
    assert restored.source_sha256 == source_sha256("move()\n")


@pytest.mark.parametrize(
    ("raw_reward", "error_type"),
    [(None, None), (0.99, None), (1.0, "RuntimeError")],
)
def test_success_requires_cube_stack_clean_success_invariants(raw_reward, error_type):
    with pytest.raises(ValueError, match="success"):
        ProgramReplayResultV1(
            task_id="cube-stack-5",
            environment_seed=5,
            program_sample_id="sample-3",
            source="move()",
            initial_state_sha256="a" * 64,
            outcome=ReplayOutcome.SUCCESS,
            raw_reward=raw_reward,
            binary_reward=1.0,
            task_completed=True,
            error_type=error_type,
        )


def test_trace_audit_and_learning_group_have_json_round_trips():
    audit = RepairAuditV1(
        task_id="cube-stack-5",
        environment_seed=5,
        program_sample_id="sample-3",
        repair_trajectory_id="repair-2",
        turn_index=1,
        event_type="inspect",
        status="observed",
        message="looked at group_1",
        action={"target": "base:group_1"},
    )
    trace = RepairTraceV1(
        task_id="cube-stack-5",
        environment_seed=5,
        program_sample_id="sample-3",
        repair_trajectory_id="repair-2",
        base_source="move()",
        base_units=(SourceUnitV1("base:group_1", 0, 6, "move()", "base"),),
        audits=(audit,),
        final_source="move()",
    )
    group = LearningGroupV1(
        task_id="cube-stack-5",
        environment_seed=5,
        group_uid="cube-stack-5:seed-5",
        initial_state_sha256="a" * 64,
        members=(
            LearningMemberV1(
                member_type="base",
                program_sample_id="sample-3",
                prompt="Stack cubes",
                response="move()",
                reward=0.0,
            ),
        ),
    )

    assert RepairAuditV1.from_json(audit.to_json()) == audit
    assert RepairTraceV1.from_json(trace.to_json()) == trace
    assert LearningGroupV1.from_json(group.to_json()) == group


def test_learning_group_requires_auditable_initial_state_hash():
    with pytest.raises(ValueError, match="initial_state_sha256"):
        LearningGroupV1(
            task_id="cube-stack-5",
            environment_seed=5,
            group_uid="cube-stack-5:seed-5",
            members=(
                LearningMemberV1(
                    member_type="base",
                    program_sample_id="sample-3",
                    prompt="Stack cubes",
                    response="move()",
                    reward=0.0,
                ),
            ),
        )


def test_environment_seed_program_sample_and_repair_trajectory_are_not_conflated():
    trace = RepairTraceV1(
        task_id="task",
        environment_seed=17,
        program_sample_id="program-17-4",
        repair_trajectory_id="trajectory-9",
        base_source="pass",
        base_units=(SourceUnitV1("base:root", 0, 4, "pass", "base"),),
        final_source="pass",
    )

    payload = trace.to_dict()

    assert payload["environment_seed"] == 17
    assert payload["program_sample_id"] == "program-17-4"
    assert payload["repair_trajectory_id"] == "trajectory-9"


@pytest.mark.parametrize(
    ("target", "origin"),
    [("base:", "base"), ("base:unit:extra", "base"), ("recovery:x", "recovery")],
)
def test_source_unit_rejects_malformed_stable_targets(target, origin):
    with pytest.raises(ValueError, match="target"):
        SourceUnitV1(target, 0, 4, "pass", origin)


def test_committed_edit_rejects_malformed_stable_target():
    with pytest.raises(ValueError, match="target"):
        CommittedEditV1(
            edit_index=0,
            turn_index=1,
            action="replace",
            target="recovery:generation-only",
            origin="recovery",
            input_revision=0,
            output_revision=1,
            input_sha256="a" * 64,
            output_sha256="b" * 64,
            rationale="",
            before_source="pass",
            after_source="move()",
        )


@pytest.mark.parametrize("field_name", ["start_offset", "end_offset"])
@pytest.mark.parametrize("invalid", [True, 1.5])
def test_source_unit_offsets_require_strict_integers(
    field_name: str, invalid: object
) -> None:
    arguments = {
        "target": "base:root",
        "start_offset": 0,
        "end_offset": 4,
        "source": "pass",
        "origin": "base",
    }
    arguments[field_name] = invalid

    with pytest.raises(TypeError, match=field_name):
        SourceUnitV1(**arguments)  # type: ignore[arg-type]


def test_source_unit_source_requires_string() -> None:
    with pytest.raises(TypeError, match="source"):
        SourceUnitV1(
            target="base:root",
            start_offset=0,
            end_offset=4,
            source=123,  # type: ignore[arg-type]
            origin="base",
        )


@pytest.mark.parametrize(
    "field_name", ["edit_index", "turn_index", "input_revision", "output_revision"]
)
@pytest.mark.parametrize("invalid", [True, 1.5])
def test_committed_edit_indices_require_strict_integers(
    field_name: str, invalid: object
) -> None:
    arguments = {
        "edit_index": 0,
        "turn_index": 1,
        "action": "replace",
        "target": "base:root",
        "origin": "base",
        "input_revision": 0,
        "output_revision": 1,
        "input_sha256": "a" * 64,
        "output_sha256": "b" * 64,
        "rationale": "fix",
        "before_source": "pass",
        "after_source": "move()",
    }
    arguments[field_name] = invalid

    with pytest.raises(TypeError, match=field_name):
        CommittedEditV1(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", ["rationale", "before_source", "after_source"])
def test_committed_edit_source_fields_require_strings(field_name: str) -> None:
    arguments = {
        "edit_index": 0,
        "turn_index": 1,
        "action": "replace",
        "target": "base:root",
        "origin": "base",
        "input_revision": 0,
        "output_revision": 1,
        "input_sha256": "a" * 64,
        "output_sha256": "b" * 64,
        "rationale": "fix",
        "before_source": "pass",
        "after_source": "move()",
    }
    arguments[field_name] = 123

    with pytest.raises(TypeError, match=field_name):
        CommittedEditV1(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field_name",
    [
        "task_id",
        "program_sample_id",
        "repair_trajectory_id",
        "event_type",
        "status",
        "message",
    ],
)
def test_repair_audit_scalar_fields_require_strings(field_name: str) -> None:
    arguments = {
        "task_id": "cube-stack-5",
        "environment_seed": 5,
        "program_sample_id": "sample-3",
        "repair_trajectory_id": "repair-1",
        "turn_index": 1,
        "event_type": "inspect",
        "status": "observed",
        "message": "checked",
    }
    arguments[field_name] = 123

    with pytest.raises(TypeError, match=field_name):
        RepairAuditV1(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("turn_index", [True, 1.5])
def test_repair_audit_turn_index_requires_strict_integer(turn_index: object) -> None:
    with pytest.raises(TypeError, match="turn_index"):
        RepairAuditV1(
            task_id="cube-stack-5",
            environment_seed=5,
            program_sample_id="sample-3",
            repair_trajectory_id="repair-1",
            turn_index=turn_index,  # type: ignore[arg-type]
            event_type="inspect",
            status="observed",
        )
