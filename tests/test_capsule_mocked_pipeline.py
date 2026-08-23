from __future__ import annotations

from typing import Any

import torch

from capx.rl.capsule.group import CapsuleGroupAssembler, ProgramCandidate
from capx.rl.capsule.repair import BaseUnitSpan, RepairDraft
from capx.rl.capsule.schema import ProgramReplayResultV1, ReplayOutcome, TaskInstanceV1
from capx.rl.capsule.trainer import (
    CapsuleCritiqueRayTrainer,
    MemoryArtifactSink,
    TokenizerGroupEncoder,
)


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [2 + ord(character) for character in text]


class _RecordingEncoder(TokenizerGroupEncoder):
    def __init__(self) -> None:
        super().__init__(_Tokenizer(), prompt_token_limit=64, response_token_limit=64)
        self.seen_prompts: tuple[str, ...] = ()
        self.seen_responses: tuple[str, ...] = ()

    def encode(self, prompts, responses):
        self.seen_prompts = prompts
        self.seen_responses = responses
        return super().encode(prompts, responses)


def _result(
    task: TaskInstanceV1,
    candidate: ProgramCandidate,
    *,
    success: bool,
    raw_reward: float,
) -> ProgramReplayResultV1:
    return ProgramReplayResultV1(
        task_id=task.task_id,
        environment_seed=task.environment_seed,
        program_sample_id=candidate.program_sample_id,
        source=candidate.source,
        initial_state_sha256=task.initial_state_sha256,
        outcome=ReplayOutcome.SUCCESS if success else ReplayOutcome.TASK_FAILURE,
        raw_reward=1.0 if success else raw_reward,
        binary_reward=float(success),
        task_completed=success,
    )


class _ScriptedCollection:
    def __init__(self) -> None:
        self.base_sources: set[str] = set()
        self.repair_calls = 0
        self.revision_calls = 0

    def sample_base(self, _task: TaskInstanceV1, base_index: int) -> ProgramCandidate:
        source = f"attempt_{base_index} = False\n"
        self.base_sources.add(source)
        return ProgramCandidate(f"base-{base_index}", source)

    def collect_repair(
        self,
        task: TaskInstanceV1,
        p0: ProgramCandidate,
        p0_result: ProgramReplayResultV1,
        _p0_rank: int,
        _trajectory_index: int,
        trajectory_id: str,
    ):
        assert p0_result.program_sample_id == p0.program_sample_id
        assert p0_result.binary_reward == 0.0
        self.repair_calls += 1
        draft = RepairDraft(
            task_id=task.task_id,
            environment_seed=task.environment_seed,
            program_sample_id=p0.program_sample_id,
            repair_trajectory_id=trajectory_id,
            base_source=p0.source,
            base_units=[BaseUnitSpan("whole", 0, len(p0.source), p0.source)],
        )
        draft.submit(
            {
                "action": "replace",
                "target": "base:whole",
                "source": "pt_is_correct = True\n",
                "rationale": "replace the failed plan",
            }
        )
        draft.submit({"action": "finish"})
        return draft.to_trace()

    def generate_revision(
        self,
        _task,
        p0,
        trace,
        revision_prompt,
        p0_rank,
        trajectory_index,
    ) -> ProgramCandidate:
        self.revision_calls += 1
        assert p0.source in revision_prompt.text
        assert trace.final_source_sha256 in revision_prompt.text
        return ProgramCandidate(
            f"revision-{p0_rank}-{trajectory_index}",
            "independent_success = True\n",
        )

    def evaluate(self, task: TaskInstanceV1, candidate: ProgramCandidate):
        if candidate.program_sample_id.startswith("base-"):
            index = int(candidate.program_sample_id.removeprefix("base-"))
            return _result(task, candidate, success=False, raw_reward=index / 10)
        return _result(task, candidate, success=True, raw_reward=1.0)


class _Actor:
    def __init__(self) -> None:
        self.updated: dict[str, Any] | None = None

    def compute_log_prob(self, batch):
        return {"old_log_probs": torch.zeros_like(batch["response_mask"], dtype=torch.float32)}

    def update_actor(self, batch):
        self.updated = batch
        return {"metrics": {"mock_update_only": True}}


class _Reference:
    def compute_ref_log_prob(self, batch):
        return {"ref_log_prob": torch.zeros_like(batch["response_mask"], dtype=torch.float32)}


def test_scripted_collection_to_guided_training_batch_without_external_runtime() -> None:
    task = TaskInstanceV1(
        task_id="cube-stack-mocked",
        environment_seed=5,
        prompt="Stack the cubes.",
        environment="robosuite_cube_stack",
        api="franka_privileged",
        privilege="privileged",
        initial_state_sha256="a" * 64,
    )
    collection = _ScriptedCollection()
    assembler = CapsuleGroupAssembler(
        base_sampler=collection.sample_base,
        repair_collector=collection.collect_repair,
        revision_generator=collection.generate_revision,
        clean_evaluator=collection.evaluate,
        token_counter=len,
    )
    encoder = _RecordingEncoder()
    actor = _Actor()
    sink = MemoryArtifactSink()
    trainer = CapsuleCritiqueRayTrainer(
        assembler=assembler,
        batch_encoder=encoder,
        actor_rollout_wg=actor,
        ref_policy_wg=_Reference(),
        artifact_sink=sink,
        config={"algorithm": {"rollout_is": False, "rollout_is_threshold": None}},
    )

    result = trainer.run_step(task)

    assert collection.repair_calls == 4
    assert collection.revision_calls == 4
    assert [member.reward for member in result.artifact.assembly.group.members] == [0.0] * 7 + [1.0]
    assert result.artifact.assembly.group.members[-1].member_type == "critique_guided_revision"
    assert encoder.seen_prompts == (task.prompt,) * 8
    assert encoder.seen_responses[-1] == "independent_success = True\n"
    assert all(source not in "".join(encoder.seen_prompts) for source in collection.base_sources)
    assert actor.updated is not None
    assert actor.updated["guided_token_mask"].sum().item() > 0
    assert len(sink.artifacts) == 1
