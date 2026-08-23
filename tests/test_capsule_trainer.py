from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from capx.rl.capsule.group import (
    GroupAssemblyResult,
    GroupDiscarded,
    RepairAttempt,
    deterministic_group_uid,
)
from capx.rl.capsule.repair import BaseUnitSpan, RepairDraft
from capx.rl.capsule.schema import (
    LearningGroupV1,
    LearningMemberV1,
    ProgramReplayResultV1,
    ReplayOutcome,
    TaskInstanceV1,
)
from capx.rl.capsule.trainer import (
    AtomicJsonArtifactSink,
    CapsuleCritiqueRayTrainer,
    MemoryArtifactSink,
    TokenBudgetExceeded,
    TokenizerGroupEncoder,
)


def _task() -> TaskInstanceV1:
    return TaskInstanceV1(
        task_id="cube-stack-5",
        environment_seed=5,
        prompt="ORIGINAL TASK PROMPT",
        environment="cube-stack",
        api="privileged",
        privilege="privileged",
        initial_state_sha256="a" * 64,
    )


def _assembly(rewards: list[float]) -> GroupAssemblyResult:
    group_uid = deterministic_group_uid(_task())
    members = []
    for index, reward in enumerate(rewards):
        guided = index == 7 and reward == 1.0 and rewards[:7] == [0.0] * 7
        members.append(
            LearningMemberV1(
                member_type="critique_guided_revision" if guided else "base",
                program_sample_id=f"sample-{index}",
                repair_trajectory_id=(
                    f"{group_uid}:p0-0:trajectory-0" if guided else None
                ),
                prompt="ORIGINAL TASK PROMPT",
                response=f"program_{index}()",
                reward=reward,
                metadata=(
                    {"private_critique": "P0 and rho must not reach encoder"}
                    if guided
                    else {}
                ),
            )
        )
    group = LearningGroupV1(
        task_id="cube-stack-5",
        environment_seed=5,
        group_uid=group_uid,
        initial_state_sha256="a" * 64,
        members=tuple(members),
        skip_actor_update=len(set(rewards)) == 1,
    )
    base_member_count = 7 if members[-1].member_type == "critique_guided_revision" else 8
    base_results = tuple(
        ProgramReplayResultV1(
            task_id="cube-stack-5",
            environment_seed=5,
            program_sample_id=member.program_sample_id,
            source=member.response,
            initial_state_sha256="a" * 64,
            outcome=ReplayOutcome.SUCCESS if member.reward == 1.0 else ReplayOutcome.TASK_FAILURE,
            raw_reward=member.reward,
            binary_reward=member.reward,
            task_completed=member.reward == 1.0,
        )
        for member in members[:base_member_count]
    )
    repair_attempts: tuple[RepairAttempt, ...] = ()
    if rewards[:7] == [0.0] * 7:
        attempts: list[RepairAttempt] = []
        for p0_rank, p0_sample_id in enumerate(("sample-0", "sample-1")):
            for trajectory_index in range(2):
                trajectory_id = f"{group_uid}:p0-{p0_rank}:trajectory-{trajectory_index}"
                attempts.append(
                    RepairAttempt(
                        p0_rank=p0_rank,
                        trajectory_index=trajectory_index,
                        p0_program_sample_id=p0_sample_id,
                        repair_trajectory_id=trajectory_id,
                        status="rejected",
                        rejection_reason="collector_error",
                        rejection_message="fixture rejection",
                    )
                )
        if members[-1].member_type == "critique_guided_revision":
            trajectory_id = f"{group_uid}:p0-0:trajectory-0"
            p0_source = members[0].response
            draft = RepairDraft(
                task_id="cube-stack-5",
                environment_seed=5,
                program_sample_id="sample-0",
                repair_trajectory_id=trajectory_id,
                base_source=p0_source,
                base_units=(
                    BaseUnitSpan("program", 0, len(p0_source), p0_source),
                ),
            )
            draft.submit(
                {
                    "action": "replace",
                    "target": "base:program",
                    "source": "pt_program()",
                }
            )
            trace = draft.to_trace()
            attempts[0] = RepairAttempt(
                p0_rank=0,
                trajectory_index=0,
                p0_program_sample_id="sample-0",
                repair_trajectory_id=trajectory_id,
                status="guided_success",
                trace=trace,
                pt_result=ProgramReplayResultV1(
                    task_id="cube-stack-5",
                    environment_seed=5,
                    program_sample_id=f"{trajectory_id}:pt",
                    source=trace.final_source,
                    initial_state_sha256="a" * 64,
                    outcome=ReplayOutcome.SUCCESS,
                    raw_reward=1.0,
                    binary_reward=1.0,
                    task_completed=True,
                ),
                revision_program_sample_id=members[-1].program_sample_id,
                revision_source=members[-1].response,
                revision_result=ProgramReplayResultV1(
                    task_id="cube-stack-5",
                    environment_seed=5,
                    program_sample_id=members[-1].program_sample_id,
                    source=members[-1].response,
                    initial_state_sha256="a" * 64,
                    outcome=ReplayOutcome.SUCCESS,
                    raw_reward=1.0,
                    binary_reward=1.0,
                    task_completed=True,
                ),
                selected=True,
            )
        repair_attempts = tuple(attempts)
    return GroupAssemblyResult(
        group=group,
        base_results=base_results,
        repair_attempts=repair_attempts,
    )


class _Assembler:
    def __init__(self, events: list[str], assembly: GroupAssemblyResult) -> None:
        self.events = events
        self.assembly = assembly

    def assemble(self, _task: TaskInstanceV1) -> GroupAssemblyResult:
        self.events.extend(["generate", "score", "repair"])
        return self.assembly


class _Encoder:
    def __init__(self) -> None:
        self.prompts: tuple[str, ...] = ()
        self.responses: tuple[str, ...] = ()

    def encode(self, prompts: tuple[str, ...], responses: tuple[str, ...]) -> dict[str, Any]:
        self.prompts = prompts
        self.responses = responses
        response_mask = torch.tensor([[1, 1, 0, 0]] * 7 + [[1, 1, 1, 0]], dtype=torch.bool)
        return {
            "input_ids": torch.ones((8, 8), dtype=torch.long),
            "attention_mask": torch.ones((8, 8), dtype=torch.bool),
            "position_ids": torch.arange(8).repeat(8, 1),
            "responses": torch.ones((8, 4), dtype=torch.long),
            "response_mask": response_mask,
        }


class _Actor:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.updated_batch: Any = None

    def compute_log_prob(self, batch):
        self.events.append("old_logprob")
        return {"old_log_probs": torch.zeros_like(batch["response_mask"], dtype=torch.float32)}

    def update_actor(self, batch):
        self.events.append("update")
        self.updated_batch = batch
        return {"metrics": {"finite_gradient": True}}


class _Reference:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def compute_ref_log_prob(self, batch):
        self.events.append("reference_logprob")
        return {"ref_log_prob": torch.zeros_like(batch["response_mask"], dtype=torch.float32)}


def _config() -> dict[str, Any]:
    return {"algorithm": {"rollout_is": False, "rollout_is_threshold": None}}


def test_fit_audits_discarded_group_and_continues_with_later_task() -> None:
    infra_result = ProgramReplayResultV1(
        task_id="cube-stack-5",
        environment_seed=5,
        program_sample_id="infra-sample",
        source="pass\n",
        initial_state_sha256="a" * 64,
        outcome=ReplayOutcome.INFRA_ERROR,
        raw_reward=None,
        binary_reward=None,
        task_completed=False,
        attempts=3,
        error_type="WorkerCrashedError",
        error_message="worker remained poisoned",
    )
    partial_attempt = RepairAttempt(
        p0_rank=0,
        trajectory_index=0,
        p0_program_sample_id="sample-0",
        repair_trajectory_id="partial-repair-0",
        status="rejected",
        rejection_reason="unknown_replay_reward",
        rejection_message="typed evaluator returned infra_error",
    )

    class _History:
        def __init__(self) -> None:
            self.values: tuple[ProgramReplayResultV1, ...] = ()

        def drain_history(self) -> tuple[ProgramReplayResultV1, ...]:
            values = self.values
            self.values = ()
            return values

    class _DiscardThenAssemble:
        def __init__(self, assembly: GroupAssemblyResult) -> None:
            self.assembly = assembly
            self.calls = 0
            self.clean_evaluator = _History()

        def assemble(self, _task: TaskInstanceV1) -> GroupAssemblyResult:
            self.calls += 1
            if self.calls == 1:
                self.clean_evaluator.values = (infra_result,)
                raise GroupDiscarded(
                    "infra_retry_exhausted",
                    "worker remained poisoned",
                    partial_repair_attempts=(partial_attempt,),
                )
            return self.assembly

    events: list[str] = []
    trainer = CapsuleCritiqueRayTrainer(
        assembler=_DiscardThenAssemble(_assembly([0.0] * 8)),
        batch_encoder=_Encoder(),
        actor_rollout_wg=_Actor(events),
        artifact_sink=MemoryArtifactSink(),
        config=_config(),
        event_log=events,
    )

    results = trainer.fit((_task(), _task()))

    assert len(results) == 1
    assert trainer.discarded_count == 1
    assert trainer.discard_reasons == ("infra_retry_exhausted",)
    assert len(trainer.discarded_groups) == 1
    record = trainer.discarded_groups[0]
    assert record.task_index == 0
    assert record.task_id == "cube-stack-5"
    assert record.environment_seed == 5
    assert record.initial_state_sha256 == "a" * 64
    assert record.reason == "infra_retry_exhausted"
    assert record.message == "infra_retry_exhausted: worker remained poisoned"
    assert record.replay_results == (infra_result,)
    assert record.partial_repair_attempts == (partial_attempt,)
    assert record.to_dict()["replay_event_count"] == 1
    assert record.to_dict()["retry_count"] == 2
    assert record.to_dict()["infra_failures"] == 3
    assert events[0] == "discard:infra_retry_exhausted"


def test_fit_does_not_swallow_non_discard_exceptions() -> None:
    class _BrokenAssembler:
        def assemble(self, _task: TaskInstanceV1) -> GroupAssemblyResult:
            raise RuntimeError("programming defect")

    trainer = CapsuleCritiqueRayTrainer(
        assembler=_BrokenAssembler(),
        batch_encoder=_Encoder(),
        actor_rollout_wg=_Actor([]),
        artifact_sink=MemoryArtifactSink(),
        config=_config(),
    )

    with pytest.raises(RuntimeError, match="programming defect"):
        trainer.fit((_task(),))

    assert trainer.discarded_count == 0
    assert trainer.discarded_groups == ()


def test_reference_worker_is_required_when_actor_reference_kl_is_enabled() -> None:
    events: list[str] = []
    config = _config()
    config["actor_rollout_ref"] = {"actor": {"use_kl_loss": True}}

    with pytest.raises(ValueError, match="ref_policy_wg"):
        CapsuleCritiqueRayTrainer(
            assembler=_Assembler(events, _assembly([0.0] * 7 + [1.0])),
            batch_encoder=_Encoder(),
            actor_rollout_wg=_Actor(events),
            artifact_sink=MemoryArtifactSink(),
            config=config,
        )


def test_one_step_injects_binary_scores_guided_mask_and_mean_advantage_in_order() -> None:
    events: list[str] = []
    encoder = _Encoder()
    actor = _Actor(events)
    sink = MemoryArtifactSink()
    trainer = CapsuleCritiqueRayTrainer(
        assembler=_Assembler(events, _assembly([0.0] * 7 + [1.0])),
        batch_encoder=encoder,
        actor_rollout_wg=actor,
        ref_policy_wg=_Reference(events),
        artifact_sink=sink,
        config=_config(),
        event_log=events,
    )

    result = trainer.run_step(_task())

    assert result.skipped_actor_update is False
    assert events == [
        "generate",
        "score",
        "repair",
        "inject",
        "old_logprob",
        "reference_logprob",
        "advantage",
        "update",
    ]
    assert encoder.prompts == ("ORIGINAL TASK PROMPT",) * 8
    assert all("P0" not in prompt and "rho" not in prompt for prompt in encoder.prompts)
    batch = actor.updated_batch
    assert batch["uid"] == (deterministic_group_uid(_task()),) * 8
    assert batch["token_level_scores"][:7].sum().item() == 0.0
    assert batch["token_level_scores"][7].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert torch.equal(batch["token_level_scores"], batch["token_level_rewards"])
    assert batch["advantages"][0].tolist() == [-0.125, -0.125, 0.0, 0.0]
    assert batch["advantages"][7].tolist() == [0.875, 0.875, 0.875, 0.0]
    assert batch["guided_token_mask"][0].tolist() == [False] * 4
    assert batch["guided_token_mask"][7].tolist() == [True, True, True, False]
    assert batch["rollout_is_weights"] is batch["guided_token_mask"]
    assert len(sink.artifacts) == 1
    assert sink.artifacts[0].guided_token_mask[7] == (True, True, True, False)


def test_trainer_rejects_missing_typed_base_replay_provenance() -> None:
    events: list[str] = []
    assembly = replace(
        _assembly([0.0] * 7 + [1.0]),
        base_results=(),
    )
    trainer = CapsuleCritiqueRayTrainer(
        assembler=_Assembler(events, assembly),
        batch_encoder=_Encoder(),
        actor_rollout_wg=_Actor(events),
        ref_policy_wg=_Reference(events),
        artifact_sink=MemoryArtifactSink(),
        config=_config(),
    )

    with pytest.raises(ValueError, match="base_results"):
        trainer.run_step(_task())


def test_trainer_rejects_guided_member_without_matching_selected_revision_replay() -> None:
    events: list[str] = []
    assembly = _assembly([0.0] * 7 + [1.0])
    selected = assembly.repair_attempts[0]
    assert selected.revision_result is not None
    corrupted = replace(
        selected,
        revision_result=replace(
            selected.revision_result,
            program_sample_id="forged-guided-sample",
        ),
    )
    assembly = replace(
        assembly,
        repair_attempts=(corrupted, *assembly.repair_attempts[1:]),
    )
    trainer = CapsuleCritiqueRayTrainer(
        assembler=_Assembler(events, assembly),
        batch_encoder=_Encoder(),
        actor_rollout_wg=_Actor(events),
        ref_policy_wg=_Reference(events),
        artifact_sink=MemoryArtifactSink(),
        config=_config(),
    )

    with pytest.raises(ValueError, match="revision replay"):
        trainer.run_step(_task())


def test_trainer_rejects_repair_attempts_for_the_wrong_deterministic_p0_selection() -> None:
    events: list[str] = []
    assembly = _assembly([0.0] * 8)
    corrupted_attempts = tuple(
        replace(attempt, p0_program_sample_id="sample-2")
        if attempt.p0_rank == 0
        else attempt
        for attempt in assembly.repair_attempts
    )
    assembly = replace(assembly, repair_attempts=corrupted_attempts)
    trainer = CapsuleCritiqueRayTrainer(
        assembler=_Assembler(events, assembly),
        batch_encoder=_Encoder(),
        actor_rollout_wg=_Actor(events),
        artifact_sink=MemoryArtifactSink(),
        config=_config(),
    )

    with pytest.raises(ValueError, match="P0 selection"):
        trainer.run_step(_task())


def test_rejected_trace_mismatch_remains_audit_only_for_base_fallback() -> None:
    events: list[str] = []
    fallback = _assembly([0.0] * 8)
    guided_fixture = _assembly([0.0] * 7 + [1.0])
    valid_trace = guided_fixture.repair_attempts[0].trace
    assert valid_trace is not None
    invalid_trace = replace(
        valid_trace,
        final_source="not the reconstructed PT",
        final_source_sha256="",
    )
    rejected = replace(
        fallback.repair_attempts[0],
        trace=invalid_trace,
        rejection_reason="trace_mismatch",
    )
    fallback = replace(
        fallback,
        repair_attempts=(rejected, *fallback.repair_attempts[1:]),
    )
    sink = MemoryArtifactSink()
    trainer = CapsuleCritiqueRayTrainer(
        assembler=_Assembler(events, fallback),
        batch_encoder=_Encoder(),
        actor_rollout_wg=_Actor(events),
        artifact_sink=sink,
        config=_config(),
    )

    result = trainer.run_step(_task())

    assert result.skipped_actor_update is True
    assert len(sink.artifacts) == 1


def test_constant_reward_group_is_persisted_and_skips_actor_calls() -> None:
    events: list[str] = []
    actor = _Actor(events)
    sink = MemoryArtifactSink()
    trainer = CapsuleCritiqueRayTrainer(
        assembler=_Assembler(events, _assembly([0.0] * 8)),
        batch_encoder=_Encoder(),
        actor_rollout_wg=actor,
        artifact_sink=sink,
        config=_config(),
        event_log=events,
    )

    result = trainer.run_step(_task())

    assert result.skipped_actor_update is True
    assert events == ["generate", "score", "repair", "inject", "advantage"]
    assert actor.updated_batch is None
    assert len(sink.artifacts) == 1
    assert sink.artifacts[0].skipped_actor_update is True


def test_failed_actor_update_does_not_publish_an_immutable_success_artifact() -> None:
    class _FailingActor(_Actor):
        def update_actor(self, batch):
            self.events.append("update_failed")
            raise RuntimeError("synthetic actor failure")

    events: list[str] = []
    sink = MemoryArtifactSink()
    trainer = CapsuleCritiqueRayTrainer(
        assembler=_Assembler(events, _assembly([0.0] * 7 + [1.0])),
        batch_encoder=_Encoder(),
        actor_rollout_wg=_FailingActor(events),
        ref_policy_wg=_Reference(events),
        artifact_sink=sink,
        config=_config(),
    )

    with pytest.raises(RuntimeError, match="synthetic actor failure"):
        trainer.run_step(_task())

    assert sink.artifacts == []


@pytest.mark.parametrize(
    ("rewards", "incorrect_skip"),
    [([0.0] * 8, False), ([0.0] * 7 + [1.0], True)],
)
def test_skip_flag_must_match_rewards(rewards: list[float], incorrect_skip: bool) -> None:
    events: list[str] = []
    assembly = _assembly(rewards)
    assembly = replace(
        assembly,
        group=replace(assembly.group, skip_actor_update=incorrect_skip),
    )
    trainer = CapsuleCritiqueRayTrainer(
        assembler=_Assembler(events, assembly),
        batch_encoder=_Encoder(),
        actor_rollout_wg=_Actor(events),
        artifact_sink=MemoryArtifactSink(),
        config=_config(),
    )

    with pytest.raises(ValueError, match="skip_actor_update"):
        trainer.run_step(_task())


def test_existing_rollout_importance_slot_is_never_overwritten() -> None:
    class _EncoderWithImportanceWeights(_Encoder):
        def encode(self, prompts, responses):
            batch = super().encode(prompts, responses)
            batch["rollout_is_weights"] = torch.ones((8, 4), dtype=torch.float32)
            return batch

    events: list[str] = []
    trainer = CapsuleCritiqueRayTrainer(
        assembler=_Assembler(events, _assembly([0.0] * 7 + [1.0])),
        batch_encoder=_EncoderWithImportanceWeights(),
        actor_rollout_wg=_Actor(events),
        artifact_sink=MemoryArtifactSink(),
        config=_config(),
    )

    with pytest.raises(ValueError, match="already populated"):
        trainer.run_step(_task())


def _replace_members(
    assembly: GroupAssemblyResult, members: list[LearningMemberV1]
) -> GroupAssemblyResult:
    return replace(assembly, group=replace(assembly.group, members=tuple(members)))


@pytest.mark.parametrize(
    "case", ["multiple", "not_last", "zero_reward", "missing_trace", "base_success"]
)
def test_guided_member_must_satisfy_strict_seven_plus_one_contract(case: str) -> None:
    assembly = _assembly([0.0] * 7 + [1.0])
    members = list(assembly.group.members)
    if case == "multiple":
        members[6] = replace(
            members[6],
            member_type="critique_guided_revision",
            reward=1.0,
            repair_trajectory_id="repair-extra",
        )
    elif case == "not_last":
        members[6] = replace(
            members[6],
            member_type="critique_guided_revision",
            reward=1.0,
            repair_trajectory_id="repair-moved",
        )
        members[7] = replace(
            members[7], member_type="base", repair_trajectory_id=None
        )
    elif case == "zero_reward":
        members[7] = replace(members[7], reward=0.0)
    elif case == "missing_trace":
        members[7] = replace(members[7], repair_trajectory_id=None)
    elif case == "base_success":
        members[0] = replace(members[0], reward=1.0)
    malformed = _replace_members(assembly, members)
    malformed = replace(
        malformed,
        group=replace(
            malformed.group,
            skip_actor_update=len({member.reward for member in members}) == 1,
        ),
    )
    events: list[str] = []
    trainer = CapsuleCritiqueRayTrainer(
        assembler=_Assembler(events, malformed),
        batch_encoder=_Encoder(),
        actor_rollout_wg=_Actor(events),
        artifact_sink=MemoryArtifactSink(),
        config=_config(),
    )

    with pytest.raises(ValueError, match="guided"):
        trainer.run_step(_task())


class _FakeDataProto:
    def __init__(
        self,
        batch: dict[str, Any],
        marker: str,
        *,
        non_tensor_key: str = "preserved",
        meta_key: str = "preserved_meta",
    ) -> None:
        self.batch = batch
        self.non_tensor_batch = {
            non_tensor_key: np.full((8,), marker, dtype=object)
        }
        self.meta_info = {meta_key: marker}

    def union(self, other):
        source = other.batch if hasattr(other, "batch") else other
        self.batch.update(source)
        if hasattr(other, "non_tensor_batch"):
            for key, value in other.non_tensor_batch.items():
                if key in self.non_tensor_batch:
                    assert np.array_equal(self.non_tensor_batch[key], value)
                else:
                    self.non_tensor_batch[key] = value
        if hasattr(other, "meta_info"):
            for key, value in other.meta_info.items():
                if key in self.meta_info:
                    assert self.meta_info[key] == value
                else:
                    self.meta_info[key] = value
        return self

    def check_consistency(self) -> None:
        for value in self.non_tensor_batch.values():
            assert isinstance(value, np.ndarray)
            assert value.dtype == object
            assert value.shape == (8,)


class _DataProtoEncoder(_Encoder):
    def encode(self, prompts, responses):
        return _FakeDataProto(super().encode(prompts, responses), "yes")


class _DataProtoActor(_Actor):
    def compute_log_prob(self, batch):
        self.events.append("old_logprob")
        batch.check_consistency()
        return _FakeDataProto(
            {"old_log_probs": torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)},
            "actor",
            non_tensor_key="actor_value",
            meta_key="actor_meta",
        )


def test_dataproto_like_batch_union_preserves_non_tensor_and_meta() -> None:
    events: list[str] = []
    actor = _DataProtoActor(events)
    trainer = CapsuleCritiqueRayTrainer(
        assembler=_Assembler(events, _assembly([0.0] * 7 + [1.0])),
        batch_encoder=_DataProtoEncoder(),
        actor_rollout_wg=actor,
        artifact_sink=MemoryArtifactSink(),
        config=_config(),
        event_log=events,
    )

    trainer.run_step(_task())

    assert actor.updated_batch.non_tensor_batch["preserved"].tolist() == ["yes"] * 8
    assert actor.updated_batch.meta_info["preserved_meta"] == "yes"
    assert actor.updated_batch.non_tensor_batch["uid"].dtype == object
    assert actor.updated_batch.non_tensor_batch["uid"].tolist() == [
        deterministic_group_uid(_task())
    ] * 8
    assert actor.updated_batch.non_tensor_batch["actor_value"].tolist() == ["actor"] * 8
    assert actor.updated_batch.meta_info["actor_meta"] == "actor"
    assert actor.updated_batch.meta_info["global_token_num"] == [8] * 8
    assert "old_log_probs" in actor.updated_batch.batch


def test_tokenizer_encoder_can_build_a_dataproto_like_batch() -> None:
    created: list[dict[str, torch.Tensor]] = []

    def factory(tensors: dict[str, torch.Tensor]) -> _FakeDataProto:
        created.append(tensors)
        return _FakeDataProto(tensors, "factory")

    encoder = TokenizerGroupEncoder(
        _CharacterTokenizer(),
        prompt_token_limit=4,
        response_token_limit=3,
        batch_factory=factory,
    )

    batch = encoder.encode(("ab",) * 8, ("c",) * 8)

    assert isinstance(batch, _FakeDataProto)
    assert batch.batch is created[0]
    assert set(batch.batch) >= {
        "input_ids",
        "attention_mask",
        "position_ids",
        "responses",
        "response_mask",
    }


@pytest.mark.parametrize("missing", ["input_ids", "attention_mask", "position_ids", "responses"])
def test_trainer_rejects_missing_required_tensor(missing: str) -> None:
    class _IncompleteEncoder(_Encoder):
        def encode(self, prompts, responses):
            batch = super().encode(prompts, responses)
            del batch[missing]
            return batch

    events: list[str] = []
    trainer = CapsuleCritiqueRayTrainer(
        assembler=_Assembler(events, _assembly([0.0] * 7 + [1.0])),
        batch_encoder=_IncompleteEncoder(),
        actor_rollout_wg=_Actor(events),
        artifact_sink=MemoryArtifactSink(),
        config=_config(),
    )

    with pytest.raises(KeyError, match=missing):
        trainer.run_step(_task())


def test_trainer_rejects_misaligned_response_tensor() -> None:
    class _MisalignedEncoder(_Encoder):
        def encode(self, prompts, responses):
            batch = super().encode(prompts, responses)
            batch["responses"] = torch.ones((8, 3), dtype=torch.long)
            return batch

    events: list[str] = []
    trainer = CapsuleCritiqueRayTrainer(
        assembler=_Assembler(events, _assembly([0.0] * 7 + [1.0])),
        batch_encoder=_MisalignedEncoder(),
        actor_rollout_wg=_Actor(events),
        artifact_sink=MemoryArtifactSink(),
        config=_config(),
    )

    with pytest.raises(ValueError, match="responses and response_mask"):
        trainer.run_step(_task())


class _CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


def test_tokenizer_encoder_left_pads_prompt_right_pads_response_and_adds_eos() -> None:
    encoder = TokenizerGroupEncoder(
        _CharacterTokenizer(),
        prompt_token_limit=4,
        response_token_limit=3,
    )

    batch = encoder.encode(("ab", "wxyz"), ("c", "de"))

    assert batch["prompts"].tolist() == [[0, 0, ord("a"), ord("b")], [ord(c) for c in "wxyz"]]
    assert batch["responses"].tolist() == [
        [ord("c"), 99, 0],
        [ord("d"), ord("e"), 99],
    ]
    assert batch["response_mask"].tolist() == [[True, True, False], [True, True, True]]
    assert batch["attention_mask"].tolist() == [
        [False, False, True, True, True, True, False],
        [True, True, True, True, True, True, True],
    ]
    assert batch["position_ids"].tolist() == [
        [0, 0, 0, 1, 2, 3, 3],
        [0, 1, 2, 3, 4, 5, 6],
    ]


@pytest.mark.parametrize(
    ("prompts", "responses", "expected_field", "observed", "limit"),
    [
        (("abcde",), ("x",), "prompt", 5, 4),
        (("x",), ("abc",), "response", 4, 3),
    ],
)
def test_tokenizer_encoder_rejects_overflow_without_truncation(
    prompts, responses, expected_field, observed, limit
) -> None:
    encoder = TokenizerGroupEncoder(
        _CharacterTokenizer(),
        prompt_token_limit=4,
        response_token_limit=3,
    )

    with pytest.raises(TokenBudgetExceeded) as error:
        encoder.encode(prompts, responses)

    assert error.value.field == expected_field
    assert error.value.observed_tokens == observed
    assert error.value.token_limit == limit


def test_tokenizer_encoder_rejects_empty_or_misaligned_batches() -> None:
    encoder = TokenizerGroupEncoder(_CharacterTokenizer())

    with pytest.raises(ValueError, match="same non-zero batch size"):
        encoder.encode(("prompt",), ())
    with pytest.raises(ValueError, match="empty response"):
        encoder.encode(("prompt",), ("",))


def test_atomic_json_sink_writes_complete_immutable_artifacts_for_repeated_group_uid(
    tmp_path, monkeypatch
) -> None:
    events: list[str] = []
    memory = MemoryArtifactSink()
    trainer = CapsuleCritiqueRayTrainer(
        assembler=_Assembler(events, _assembly([0.0] * 8)),
        batch_encoder=_Encoder(),
        actor_rollout_wg=_Actor(events),
        artifact_sink=memory,
        config=_config(),
    )
    artifact = trainer.run_step(_task()).artifact
    fsynced: list[Any] = []
    import capx.rl.capsule.trainer as trainer_module

    monkeypatch.setattr(
        trainer_module,
        "_fsync_directory",
        lambda path: fsynced.append(path),
        raising=False,
    )
    sink = AtomicJsonArtifactSink(tmp_path)

    sink.write(artifact)

    first_destination = tmp_path / f"00000000-{deterministic_group_uid(_task())}.json"
    payload = json.loads(first_destination.read_text(encoding="utf-8"))
    assert payload["guided_token_mask"] == [[False] * 4] * 8
    assert payload["skipped_actor_update"] is True
    assert fsynced == [tmp_path]
    first_bytes = first_destination.read_bytes()

    sink.write(artifact)

    second_destination = tmp_path / f"00000001-{deterministic_group_uid(_task())}.json"
    assert second_destination.read_bytes() == first_bytes
    assert first_destination.read_bytes() == first_bytes
    assert fsynced == [tmp_path, tmp_path]


def test_atomic_json_sink_resumes_sequence_without_overwriting_after_restart(tmp_path) -> None:
    events: list[str] = []
    memory = MemoryArtifactSink()
    trainer = CapsuleCritiqueRayTrainer(
        assembler=_Assembler(events, _assembly([0.0] * 8)),
        batch_encoder=_Encoder(),
        actor_rollout_wg=_Actor(events),
        artifact_sink=memory,
        config=_config(),
    )
    artifact = trainer.run_step(_task()).artifact
    first_sink = AtomicJsonArtifactSink(tmp_path)
    first_sink.write(artifact)
    group_uid = deterministic_group_uid(_task())
    first_bytes = (tmp_path / f"00000000-{group_uid}.json").read_bytes()

    AtomicJsonArtifactSink(tmp_path).write(artifact)

    assert (tmp_path / f"00000000-{group_uid}.json").read_bytes() == first_bytes
    assert (tmp_path / f"00000001-{group_uid}.json").read_bytes() == first_bytes


def test_atomic_json_sink_cleans_temporary_file_when_link_fails(tmp_path, monkeypatch) -> None:
    events: list[str] = []
    memory = MemoryArtifactSink()
    trainer = CapsuleCritiqueRayTrainer(
        assembler=_Assembler(events, _assembly([0.0] * 8)),
        batch_encoder=_Encoder(),
        actor_rollout_wg=_Actor(events),
        artifact_sink=memory,
        config=_config(),
    )
    artifact = trainer.run_step(_task()).artifact
    import capx.rl.capsule.trainer as trainer_module

    monkeypatch.setattr(
        trainer_module.os,
        "link",
        lambda *_args: (_ for _ in ()).throw(OSError("boom")),
    )

    with pytest.raises(OSError, match="boom"):
        AtomicJsonArtifactSink(tmp_path).write(artifact)

    assert list(tmp_path.iterdir()) == []
