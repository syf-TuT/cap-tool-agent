import pytest

from capx.rl.capsule.repair import BaseUnitSpan, RepairDraft
from capx.rl.capsule.revision import (
    RevisionRejection,
    RevisionRejectionReason,
    build_revision_prompt,
    validate_complete_program,
)
from capx.rl.capsule.schema import TaskInstanceV1


def make_task() -> TaskInstanceV1:
    return TaskInstanceV1(
        task_id="cube-stack-5",
        environment_seed=5,
        prompt="Stack the cubes.",
        environment="RobosuiteCubeStack",
        api="FrankaControlPrivilegedApi",
        privilege="privileged",
        initial_state_sha256="a" * 64,
    )


def make_trace():
    source = "move()"
    draft = RepairDraft(
        task_id="cube-stack-5",
        environment_seed=5,
        program_sample_id="sample-1",
        repair_trajectory_id="repair-1",
        base_source=source,
        base_units=(BaseUnitSpan("move", 0, len(source)),),
    )
    draft.submit(
        {
            "action": "replace",
            "target": "base:move",
            "source": "move_safely()",
            "rationale": "the direct move collided",
        }
    )
    return draft.to_trace()


def test_revision_prompt_contains_task_p0_and_full_edit_history_without_training_projection():
    trace = make_trace()

    prompt = build_revision_prompt(
        make_task(),
        trace,
        token_counter=lambda text: len(text.split()),
    )

    assert "Stack the cubes." in prompt.text
    assert "move()" in prompt.text
    assert "move_safely()" in prompt.text
    assert "the direct move collided" in prompt.text
    assert prompt.response_token_limit == 2048
    assert prompt.input_token_limit == 8192


def test_revision_prompt_overflow_is_rejected_without_truncation():
    trace = make_trace()

    with pytest.raises(RevisionRejection) as exc_info:
        build_revision_prompt(
            make_task(),
            trace,
            token_counter=lambda _text: 8193,
            input_token_limit=8192,
        )

    assert exc_info.value.reason is RevisionRejectionReason.INPUT_OVERFLOW


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        ("", RevisionRejectionReason.INCOMPLETE_PROGRAM),
        ("```python\nmove()\n```", RevisionRejectionReason.MARKDOWN_FENCE),
        ("if True:\n", RevisionRejectionReason.SYNTAX_ERROR),
        ("return 1\n", RevisionRejectionReason.SYNTAX_ERROR),
    ],
)
def test_incomplete_or_wrapped_revision_response_is_typed_rejection(response, reason):
    with pytest.raises(RevisionRejection) as exc_info:
        validate_complete_program(response, token_counter=lambda text: len(text.split()))

    assert exc_info.value.reason is reason


def test_response_overflow_is_rejected_instead_of_truncated():
    source = "\n".join("move()" for _ in range(5))

    with pytest.raises(RevisionRejection) as exc_info:
        validate_complete_program(
            source,
            token_counter=lambda _text: 2049,
            response_token_limit=2048,
        )

    assert exc_info.value.reason is RevisionRejectionReason.RESPONSE_OVERFLOW


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
def test_length_terminated_response_is_rejected_even_when_prefix_parses(finish_reason):
    with pytest.raises(RevisionRejection) as exc_info:
        validate_complete_program(
            "move()\n",
            token_counter=lambda _text: 1,
            finish_reason=finish_reason,
        )

    assert exc_info.value.reason is RevisionRejectionReason.INCOMPLETE_PROGRAM


def test_valid_complete_python_program_is_returned_byte_for_byte():
    source = "target = get_target()\nmove(target)\n"

    assert validate_complete_program(source, token_counter=lambda _text: 12) == source
