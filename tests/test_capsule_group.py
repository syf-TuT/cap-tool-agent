from __future__ import annotations

import json
from dataclasses import replace

import pytest

from capx.rl.capsule.group import (
    CandidateCollectionError,
    CapsuleGroupAssembler,
    CollectionInfrastructureError,
    GroupAssemblyResult,
    GroupDiscarded,
    ProgramCandidate,
    deterministic_group_uid,
)
from capx.rl.capsule.repair import BaseUnitSpan, RepairDraft
from capx.rl.capsule.schema import (
    ProgramReplayResultV1,
    ReplayOutcome,
    TaskInstanceV1,
)


INITIAL_HASH = "a" * 64


def make_task(seed: int = 5) -> TaskInstanceV1:
    return TaskInstanceV1(
        task_id="cube-stack",
        environment_seed=seed,
        prompt="Stack the red cube on the blue cube.",
        environment="robosuite_cube_stack",
        api="franka_privileged",
        privilege="privileged",
        initial_state_sha256=INITIAL_HASH,
    )


def replay_result(
    task: TaskInstanceV1,
    candidate: ProgramCandidate,
    *,
    reward: float = 0.0,
    raw_reward: float | None = 0.0,
    outcome: ReplayOutcome | None = None,
) -> ProgramReplayResultV1:
    selected_outcome = outcome or (
        ReplayOutcome.SUCCESS if reward == 1.0 else ReplayOutcome.TASK_FAILURE
    )
    unknown = selected_outcome in {ReplayOutcome.INFRA_ERROR, ReplayOutcome.EVALUATOR_ERROR}
    return ProgramReplayResultV1(
        task_id=task.task_id,
        environment_seed=task.environment_seed,
        program_sample_id=candidate.program_sample_id,
        source=candidate.source,
        initial_state_sha256=task.initial_state_sha256,
        outcome=selected_outcome,
        raw_reward=None if unknown else raw_reward,
        binary_reward=None if unknown else reward,
        task_completed=selected_outcome is ReplayOutcome.SUCCESS,
        error_type="FakeFailure" if unknown else None,
    )


class ScriptedCallbacks:
    def __init__(
        self,
        *,
        base_rewards: list[float] | None = None,
        base_raw_rewards: list[float | None] | None = None,
        pt_success: set[tuple[int, int]] | None = None,
        revision_success: set[tuple[int, int]] | None = None,
        revision_source: str = "fixed = True\n",
    ) -> None:
        self.base_rewards = base_rewards or [0.0] * 8
        self.base_raw_rewards = base_raw_rewards or list(self.base_rewards)
        self.pt_success = pt_success or set()
        self.revision_success = revision_success or set()
        self.revision_source = revision_source
        self.base_calls: list[int] = []
        self.repair_calls: list[tuple[int, int, str, str]] = []
        self.repair_base_results: list[ProgramReplayResultV1] = []
        self.revision_calls: list[tuple[int, int]] = []
        self.evaluator_calls: list[str] = []
        self.candidates: dict[str, ProgramCandidate] = {}

    def base_sampler(self, task: TaskInstanceV1, base_index: int) -> ProgramCandidate:
        del task
        self.base_calls.append(base_index)
        candidate = ProgramCandidate(
            program_sample_id=f"base-{base_index}",
            source=f"value_{base_index} = {base_index}\n",
        )
        self.candidates[candidate.program_sample_id] = candidate
        return candidate

    def repair_collector(
        self,
        task: TaskInstanceV1,
        p0: ProgramCandidate,
        p0_result: ProgramReplayResultV1,
        p0_rank: int,
        trajectory_index: int,
        repair_trajectory_id: str,
    ):
        self.repair_base_results.append(p0_result)
        self.repair_calls.append(
            (p0_rank, trajectory_index, p0.program_sample_id, repair_trajectory_id)
        )
        draft = RepairDraft(
            task_id=task.task_id,
            environment_seed=task.environment_seed,
            program_sample_id=p0.program_sample_id,
            repair_trajectory_id=repair_trajectory_id,
            base_source=p0.source,
            base_units=[BaseUnitSpan("whole", 0, len(p0.source), p0.source)],
        )
        draft.submit(
            {
                "action": "replace",
                "target": "base:whole",
                "source": p0.source + "repaired = True\n",
                "rationale": "repair",
            }
        )
        draft.submit({"action": "finish", "rationale": "done"})
        return draft.to_trace()

    def revision_generator(
        self,
        task: TaskInstanceV1,
        p0: ProgramCandidate,
        trace,
        revision_prompt,
        p0_rank: int,
        trajectory_index: int,
    ) -> ProgramCandidate:
        del task, p0, trace, revision_prompt
        self.revision_calls.append((p0_rank, trajectory_index))
        candidate = ProgramCandidate(
            program_sample_id=f"revision-{p0_rank}-{trajectory_index}",
            source=self.revision_source,
        )
        self.candidates[candidate.program_sample_id] = candidate
        return candidate

    def evaluator(
        self, task: TaskInstanceV1, candidate: ProgramCandidate
    ) -> ProgramReplayResultV1:
        self.evaluator_calls.append(candidate.program_sample_id)
        if candidate.program_sample_id.startswith("base-"):
            index = int(candidate.program_sample_id.removeprefix("base-"))
            return replay_result(
                task,
                candidate,
                reward=self.base_rewards[index],
                raw_reward=self.base_raw_rewards[index],
            )
        if candidate.program_sample_id.endswith(":pt"):
            prefix = candidate.program_sample_id.rsplit(":pt", 1)[0]
            rank = int(prefix.split(":p0-")[1].split(":", 1)[0])
            trajectory = int(prefix.rsplit("trajectory-", 1)[1])
            success = (rank, trajectory) in self.pt_success
            return replay_result(
                task,
                candidate,
                reward=float(success),
                raw_reward=float(success),
            )
        rank, trajectory = map(
            int, candidate.program_sample_id.removeprefix("revision-").split("-")
        )
        success = (rank, trajectory) in self.revision_success
        return replay_result(
            task,
            candidate,
            reward=float(success),
            raw_reward=float(success),
        )


def make_assembler(callbacks: ScriptedCallbacks, **kwargs) -> CapsuleGroupAssembler:
    return CapsuleGroupAssembler(
        base_sampler=callbacks.base_sampler,
        repair_collector=callbacks.repair_collector,
        revision_generator=callbacks.revision_generator,
        clean_evaluator=callbacks.evaluator,
        **kwargs,
    )


def test_any_success_in_first_seven_uses_eight_bases_and_never_repairs() -> None:
    callbacks = ScriptedCallbacks(base_rewards=[0, 0, 1, 0, 0, 0, 0, 1])

    result = make_assembler(callbacks).assemble(make_task())

    assert callbacks.base_calls == list(range(8))
    assert callbacks.repair_calls == []
    assert len(result.group.members) == 8
    assert all(member.member_type == "base" for member in result.group.members)
    assert [member.reward for member in result.group.members] == callbacks.base_rewards
    assert len(result.base_results) == 8


def test_all_failed_bases_build_seven_plus_first_successful_guided_member() -> None:
    callbacks = ScriptedCallbacks(
        pt_success={(0, 0), (0, 1), (1, 0), (1, 1)},
        revision_success={(0, 1), (1, 0)},
    )

    result = make_assembler(callbacks).assemble(make_task())

    assert callbacks.base_calls == list(range(7))
    assert [(rank, index) for rank, index, *_ in callbacks.repair_calls] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert [result.program_sample_id for result in callbacks.repair_base_results] == [
        sample_id for _, _, sample_id, _ in callbacks.repair_calls
    ]
    assert all(result.binary_reward == 0.0 for result in callbacks.repair_base_results)
    assert len(result.repair_attempts) == 4
    assert len(result.group.members) == 8
    guided = result.group.members[-1]
    assert guided.member_type == "critique_guided_revision"
    assert guided.program_sample_id == "revision-0-1"
    assert guided.prompt == make_task().prompt
    assert guided.response == "fixed = True\n"
    assert "repair" not in guided.prompt.lower()
    assert guided.repair_trajectory_id == callbacks.repair_calls[1][3]
    assert result.repair_attempts[1].selected is True
    assert sum(attempt.selected for attempt in result.repair_attempts) == 1


def test_no_successful_revision_falls_back_to_eighth_base() -> None:
    callbacks = ScriptedCallbacks(pt_success={(0, 0), (0, 1), (1, 0), (1, 1)})

    result = make_assembler(callbacks).assemble(make_task())

    assert callbacks.base_calls == list(range(8))
    assert len(result.base_results) == 8
    assert all(member.member_type == "base" for member in result.group.members)


def test_p0_ranking_uses_partial_reward_then_max_regex_token_distance() -> None:
    callbacks = ScriptedCallbacks(
        base_raw_rewards=[0.2, 0.9, 0.0, 0.4, 0.1, 0.9, 0.3, 0.0]
    )
    custom_sources = [
        "move(a)\n",
        "move(red_cube)\n",
        "move(red_cube)\n",
        "move(red_cube, blue_cube)\n",
        "pass\n",
        "for i in range(100):\n    rotate(robot, i, force=True)\n",
        "move(red_cube)\n",
        "unused = 8\n",
    ]

    def sampler(task: TaskInstanceV1, index: int) -> ProgramCandidate:
        del task
        callbacks.base_calls.append(index)
        return ProgramCandidate(f"base-{index}", custom_sources[index])

    assembler = CapsuleGroupAssembler(
        base_sampler=sampler,
        repair_collector=callbacks.repair_collector,
        revision_generator=callbacks.revision_generator,
        clean_evaluator=callbacks.evaluator,
    )

    assembler.assemble(make_task())

    selected_samples = [sample_id for _, _, sample_id, _ in callbacks.repair_calls]
    assert selected_samples == ["base-1", "base-1", "base-5", "base-5"]


def test_pt_failure_does_not_call_revision_generator_but_all_four_repairs_run() -> None:
    callbacks = ScriptedCallbacks(pt_success={(1, 1)}, revision_success={(1, 1)})

    result = make_assembler(callbacks).assemble(make_task())

    assert len(callbacks.repair_calls) == 4
    assert callbacks.revision_calls == [(1, 1)]
    assert result.group.members[-1].program_sample_id == "revision-1-1"


@pytest.mark.parametrize(
    ("candidate", "expected_reason"),
    [
        (ProgramCandidate("bad", "x = 1\n", finish_reason="length"), "incomplete_program"),
        (ProgramCandidate("bad", "if True:\n"), "syntax_error"),
        (ProgramCandidate("bad", "x = 1 " * 2049), "response_overflow"),
    ],
)
def test_revision_rejections_are_audited_without_evaluation_or_truncation(
    candidate: ProgramCandidate, expected_reason: str
) -> None:
    callbacks = ScriptedCallbacks(pt_success={(0, 0)})

    def revision_generator(*args, **kwargs) -> ProgramCandidate:
        del args, kwargs
        callbacks.revision_calls.append((0, 0))
        return candidate

    assembler = CapsuleGroupAssembler(
        base_sampler=callbacks.base_sampler,
        repair_collector=callbacks.repair_collector,
        revision_generator=revision_generator,
        clean_evaluator=callbacks.evaluator,
    )

    result = assembler.assemble(make_task())

    rejected = result.repair_attempts[0]
    assert rejected.rejection_reason == expected_reason
    assert rejected.revision_source == candidate.source
    assert "bad" not in callbacks.evaluator_calls
    assert all(member.response != candidate.source for member in result.group.members)

    artifact = result.to_dict()
    assert artifact["repair_attempts"][0]["revision_source"] == candidate.source
    json.dumps(artifact, allow_nan=False)
    assert GroupAssemblyResult.from_dict(artifact).to_dict() == artifact


def test_prompt_overflow_is_audited_and_never_calls_revision_generator() -> None:
    callbacks = ScriptedCallbacks(pt_success={(0, 0), (0, 1), (1, 0), (1, 1)})

    result = make_assembler(callbacks, revision_input_token_limit=1).assemble(make_task())

    assert callbacks.revision_calls == []
    assert {attempt.rejection_reason for attempt in result.repair_attempts} == {
        "input_overflow"
    }
    assert all(member.member_type == "base" for member in result.group.members)


def test_revision_response_budget_counts_raw_tokens_plus_eos() -> None:
    callbacks = ScriptedCallbacks(
        pt_success={(0, 0), (0, 1), (1, 0), (1, 1)},
        revision_success={(0, 0)},
        revision_source="fixed = True\n",
    )
    prompt_inputs: list[str] = []
    response_inputs: list[str] = []

    def prompt_counter(text: str) -> int:
        prompt_inputs.append(text)
        return 1

    def raw_response_counter(text: str) -> int:
        response_inputs.append(text)
        return 2

    result = make_assembler(
        callbacks,
        revision_prompt_token_counter=prompt_counter,
        revision_response_token_counter=raw_response_counter,
        revision_response_token_limit=2,
    ).assemble(make_task())

    assert prompt_inputs and all(text.startswith("You are regenerating") for text in prompt_inputs)
    assert response_inputs == ["fixed = True\n"] * 4
    assert {attempt.rejection_reason for attempt in result.repair_attempts} == {
        "response_overflow"
    }
    assert "uses 3 tokens" in result.repair_attempts[0].rejection_message
    assert all(member.member_type == "base" for member in result.group.members)


def test_revision_response_exactly_fits_when_raw_tokens_leave_one_eos_slot() -> None:
    callbacks = ScriptedCallbacks(
        pt_success={(0, 0), (0, 1), (1, 0), (1, 1)},
        revision_success={(0, 0)},
        revision_source="fixed = True\n",
    )

    result = make_assembler(
        callbacks,
        revision_prompt_token_counter=lambda _text: 1,
        revision_response_token_counter=lambda _text: 1,
        revision_response_token_limit=2,
    ).assemble(make_task())

    assert result.group.members[-1].member_type == "critique_guided_revision"
    assert result.repair_attempts[0].selected is True


def test_group_uid_is_deterministic_and_isolated_by_environment_seed() -> None:
    first = make_assembler(ScriptedCallbacks(base_rewards=[1] * 8)).assemble(make_task(5))
    repeated = make_assembler(ScriptedCallbacks(base_rewards=[1] * 8)).assemble(make_task(5))
    other_seed = make_assembler(ScriptedCallbacks(base_rewards=[1] * 8)).assemble(make_task(6))

    assert first.group.group_uid == repeated.group.group_uid
    assert first.group.group_uid != other_seed.group.group_uid
    assert {member.prompt for member in first.group.members} == {make_task(5).prompt}


@pytest.mark.parametrize(
    "changed_task",
    [
        replace(make_task(), prompt="Use a different instruction."),
        replace(make_task(), environment="different_environment"),
        replace(make_task(), api="different_api"),
        replace(make_task(), privilege="different_privilege"),
        replace(make_task(), metadata={"collection": "different"}),
    ],
)
def test_group_uid_binds_the_complete_task_contract(changed_task: TaskInstanceV1) -> None:
    assert deterministic_group_uid(make_task()) != deterministic_group_uid(changed_task)


def test_unexpected_sampler_and_evaluator_errors_are_not_masked_as_group_discards() -> None:
    callbacks = ScriptedCallbacks(base_rewards=[1.0] * 8)

    def broken_sampler(_task: TaskInstanceV1, _index: int) -> ProgramCandidate:
        raise RuntimeError("sampler runtime bug")

    sampler_assembler = CapsuleGroupAssembler(
        base_sampler=broken_sampler,
        repair_collector=callbacks.repair_collector,
        revision_generator=callbacks.revision_generator,
        clean_evaluator=callbacks.evaluator,
    )
    with pytest.raises(RuntimeError, match="sampler runtime bug"):
        sampler_assembler.assemble(make_task())

    def broken_evaluator(_task: TaskInstanceV1, _candidate: ProgramCandidate):
        raise RuntimeError("evaluator runtime bug")

    evaluator_assembler = CapsuleGroupAssembler(
        base_sampler=callbacks.base_sampler,
        repair_collector=callbacks.repair_collector,
        revision_generator=callbacks.revision_generator,
        clean_evaluator=broken_evaluator,
    )
    with pytest.raises(RuntimeError, match="evaluator runtime bug"):
        evaluator_assembler.assemble(make_task())


def test_typed_invalid_base_candidate_discards_only_the_seed_group() -> None:
    callbacks = ScriptedCallbacks()

    def invalid_candidate(_task: TaskInstanceV1, _index: int) -> ProgramCandidate:
        raise CandidateCollectionError("sample omitted EOS")

    assembler = CapsuleGroupAssembler(
        base_sampler=invalid_candidate,
        repair_collector=callbacks.repair_collector,
        revision_generator=callbacks.revision_generator,
        clean_evaluator=callbacks.evaluator,
    )

    with pytest.raises(GroupDiscarded, match="sample omitted EOS"):
        assembler.assemble(make_task())


def test_unexpected_repair_and_revision_errors_propagate() -> None:
    callbacks = ScriptedCallbacks(pt_success={(0, 0)})

    def broken_collector(*_args, **_kwargs):
        raise RuntimeError("collector runtime bug")

    collector_assembler = CapsuleGroupAssembler(
        base_sampler=callbacks.base_sampler,
        repair_collector=broken_collector,
        revision_generator=callbacks.revision_generator,
        clean_evaluator=callbacks.evaluator,
    )
    with pytest.raises(RuntimeError, match="collector runtime bug"):
        collector_assembler.assemble(make_task())

    def broken_revision(*_args, **_kwargs):
        raise RuntimeError("revision runtime bug")

    revision_assembler = CapsuleGroupAssembler(
        base_sampler=callbacks.base_sampler,
        repair_collector=callbacks.repair_collector,
        revision_generator=broken_revision,
        clean_evaluator=callbacks.evaluator,
    )
    with pytest.raises(RuntimeError, match="revision runtime bug"):
        revision_assembler.assemble(make_task())


@pytest.mark.parametrize("rewards", [[0.0] * 8, [1.0] * 8])
def test_constant_reward_groups_are_persisted_but_skip_actor_update(
    rewards: list[float],
) -> None:
    callbacks = ScriptedCallbacks(base_rewards=rewards)

    result = make_assembler(callbacks).assemble(make_task())

    assert len(result.group.members) == 8
    assert result.group.skip_actor_update is True


def test_mixed_reward_group_is_not_skipped() -> None:
    callbacks = ScriptedCallbacks(base_rewards=[1, 0, 0, 0, 0, 0, 0, 0])
    result = make_assembler(callbacks).assemble(make_task())
    assert result.group.skip_actor_update is False


@pytest.mark.parametrize("outcome", [ReplayOutcome.INFRA_ERROR, ReplayOutcome.EVALUATOR_ERROR])
def test_unknown_base_reward_discards_entire_group(outcome: ReplayOutcome) -> None:
    callbacks = ScriptedCallbacks()

    def evaluator(task: TaskInstanceV1, candidate: ProgramCandidate):
        result = callbacks.evaluator(task, candidate)
        if candidate.program_sample_id == "base-3":
            return replay_result(task, candidate, outcome=outcome, raw_reward=None)
        return result

    assembler = CapsuleGroupAssembler(
        base_sampler=callbacks.base_sampler,
        repair_collector=callbacks.repair_collector,
        revision_generator=callbacks.revision_generator,
        clean_evaluator=evaluator,
    )

    with pytest.raises(GroupDiscarded, match="unknown clean replay reward") as captured:
        assembler.assemble(make_task())
    assert captured.value.partial_repair_attempts == ()


@pytest.mark.parametrize("stage", ["pt", "revision"])
def test_unknown_repair_replay_reward_discards_entire_group(stage: str) -> None:
    callbacks = ScriptedCallbacks(pt_success={(0, 0)})

    def evaluator(task: TaskInstanceV1, candidate: ProgramCandidate):
        result = callbacks.evaluator(task, candidate)
        is_target = (
            stage == "pt" and candidate.program_sample_id.endswith(":p0-0:trajectory-0:pt")
        ) or (stage == "revision" and candidate.program_sample_id == "revision-0-0")
        if is_target:
            return replay_result(task, candidate, outcome=ReplayOutcome.INFRA_ERROR)
        return result

    assembler = CapsuleGroupAssembler(
        base_sampler=callbacks.base_sampler,
        repair_collector=callbacks.repair_collector,
        revision_generator=callbacks.revision_generator,
        clean_evaluator=evaluator,
    )

    with pytest.raises(GroupDiscarded, match="unknown clean replay reward") as captured:
        assembler.assemble(make_task())

    partial_attempts = captured.value.partial_repair_attempts
    assert len(partial_attempts) == 1
    assert partial_attempts[0].trace is not None
    assert partial_attempts[0].trace.reconstruct() == partial_attempts[0].trace.final_source
    if stage == "revision":
        assert partial_attempts[0].pt_result is not None
        assert partial_attempts[0].revision_source is not None


def test_replay_source_sample_seed_and_hash_must_match_candidate() -> None:
    callbacks = ScriptedCallbacks()

    def evaluator(task: TaskInstanceV1, candidate: ProgramCandidate):
        result = callbacks.evaluator(task, candidate)
        if candidate.program_sample_id == "base-0":
            return replace(result, program_sample_id="wrong-sample")
        return result

    assembler = CapsuleGroupAssembler(
        base_sampler=callbacks.base_sampler,
        repair_collector=callbacks.repair_collector,
        revision_generator=callbacks.revision_generator,
        clean_evaluator=evaluator,
    )

    with pytest.raises(GroupDiscarded, match="program_sample_id"):
        assembler.assemble(make_task())


def test_untrusted_trace_ids_are_rejected_but_other_trajectories_continue() -> None:
    callbacks = ScriptedCallbacks(pt_success={(0, 1)}, revision_success={(0, 1)})
    valid_collector = callbacks.repair_collector

    def collector(task, p0, p0_result, rank, index, trajectory_id):
        trace = valid_collector(task, p0, p0_result, rank, index, trajectory_id)
        if (rank, index) == (0, 0):
            # Drop the finish audit so construction remains schema-valid while the immutable
            # trajectory identity itself is intentionally untrusted.
            return replace(
                trace,
                repair_trajectory_id="untrusted",
                audits=(),
                event_sequence_sha256="",
            )
        return trace

    assembler = CapsuleGroupAssembler(
        base_sampler=callbacks.base_sampler,
        repair_collector=collector,
        revision_generator=callbacks.revision_generator,
        clean_evaluator=callbacks.evaluator,
    )

    result = assembler.assemble(make_task())

    assert len(callbacks.repair_calls) == 4
    assert result.repair_attempts[0].rejection_reason == "trace_mismatch"
    assert result.group.members[-1].program_sample_id == "revision-0-1"


def test_typed_controller_infrastructure_failure_discards_group() -> None:
    callbacks = ScriptedCallbacks()

    def unavailable_collector(*_args, **_kwargs):
        raise CollectionInfrastructureError("controller authentication failed")

    assembler = CapsuleGroupAssembler(
        base_sampler=callbacks.base_sampler,
        repair_collector=unavailable_collector,
        revision_generator=callbacks.revision_generator,
        clean_evaluator=callbacks.evaluator,
    )

    with pytest.raises(GroupDiscarded, match="controller authentication failed") as captured:
        assembler.assemble(make_task())

    assert len(captured.value.partial_repair_attempts) == 1
    assert captured.value.partial_repair_attempts[0].rejection_reason == (
        "repair_infrastructure_error"
    )
