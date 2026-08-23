import json

import pytest

from capx.rl.capsule.repair import BaseUnitSpan, RepairDraft, RepairInvariantError
from capx.rl.capsule.schema import RepairTraceV1


BASE_SOURCE = "x = 1\nmove(x)\ncheck()\n"
BASE_UNITS = (
    BaseUnitSpan("setup", 0, 5),
    BaseUnitSpan("move", 6, 13),
    BaseUnitSpan("check", 14, 21),
)


def make_draft(*, max_turns: int = 12) -> RepairDraft:
    return RepairDraft(
        task_id="cube-stack-5",
        environment_seed=5,
        program_sample_id="sample-2",
        repair_trajectory_id="repair-1",
        base_source=BASE_SOURCE,
        base_units=BASE_UNITS,
        max_turns=max_turns,
    )


def test_controller_turn_limit_cannot_exceed_twelve():
    with pytest.raises(RepairInvariantError, match="at most 12"):
        make_draft(max_turns=13)


def test_failed_commit_is_atomic_and_does_not_leak_into_next_revision():
    draft = make_draft()
    target = {
        "action": "append",
        "generation_id": "recovery_1",
        "unit_id": "retry",
        "rationale": "exercise hash failure",
    }

    rejected = draft.submit({**target, "source": "\ud800"})
    accepted = draft.submit({**target, "source": "retry()"})

    assert rejected.committed is False
    assert accepted.committed is True
    assert draft.current_revision == 1
    assert draft.current_source.endswith("retry()")
    assert "\ud800" not in draft.current_source


def test_append_and_replace_base_appended_and_already_patched_source():
    draft = make_draft()

    append = draft.submit(
        {
            "action": "append",
            "generation_id": "recovery_1",
            "unit_id": "retry",
            "source": "retry()",
            "rationale": "first recovery",
        }
    )
    replace_base = draft.submit(
        {
            "action": "replace",
            "target": "base:move",
            "source": "move_safely(x)",
            "rationale": "avoid collision",
        }
    )
    replace_appended = draft.submit(
        {
            "action": "replace",
            "target": "recovery:recovery_1:retry",
            "source": "retry_safely()",
            "rationale": "fix recovery",
        }
    )
    replace_again = draft.submit(
        {
            "action": "replace",
            "target": "base:move",
            "source": "move_precisely(x)",
            "rationale": "second patch of same stable target",
        }
    )
    draft.submit({"action": "finish", "rationale": "ready"})

    assert append.committed is True
    assert replace_base.edit.origin == "base"
    assert replace_appended.edit.origin == "recovery"
    assert replace_again.edit.before_source == "move_safely(x)"
    assert draft.current_revision == 4
    assert draft.current_source == "x = 1\nmove_precisely(x)\ncheck()\n\nretry_safely()"
    assert draft.finished is True

    trace = draft.to_trace()
    assert trace.reconstruct() == draft.current_source
    assert [edit.input_revision for edit in trace.edits] == [0, 1, 2, 3]
    assert [edit.output_revision for edit in trace.edits] == [1, 2, 3, 4]


def test_invalid_inspect_and_json_parse_failure_only_enter_audit():
    draft = make_draft()

    inspected = draft.submit({"action": "inspect", "target": "base:move"})
    invalid = draft.submit(
        {"action": "replace", "target": "base:missing", "source": "pass"}
    )
    parse_failure = draft.submit_json("{not-json")

    assert inspected.committed is False
    assert invalid.committed is False
    assert parse_failure.committed is False
    assert draft.current_revision == 0
    assert draft.current_source == BASE_SOURCE
    assert [item.event_type for item in draft.audits] == ["inspect", "invalid", "parse_failure"]
    assert len(draft.to_trace().edits) == 0


@pytest.mark.parametrize(
    "action",
    [
        {
            "action": "append",
            "generation_id": "recovery_1",
            "unit_id": "retry",
            "source": "retry()",
            "rationale": 7,
        },
        {
            "action": "replace",
            "target": "base:move",
            "source": "move_safely(x)",
            "rationale": 7,
        },
        {"action": "inspect", "message": 7},
        {"action": "finish", "rationale": 7},
    ],
)
def test_non_string_controller_text_fields_are_audited_without_mutation(
    action: dict[str, object],
) -> None:
    draft = make_draft()

    result = draft.submit(action)

    assert result.committed is False
    assert result.audit is not None
    assert result.audit.event_type == "invalid"
    assert draft.current_revision == 0
    assert draft.current_source == BASE_SOURCE
    assert draft.finished is False


def test_duplicate_recovery_target_is_invalid_and_does_not_advance_revision():
    draft = make_draft()
    action = {
        "action": "append",
        "generation_id": "recovery_1",
        "unit_id": "retry",
        "source": "retry()",
    }

    draft.submit(action)
    duplicate = draft.submit(action)

    assert duplicate.committed is False
    assert draft.current_revision == 1
    assert draft.audits[-1].event_type == "invalid"


def test_every_controller_submission_consumes_one_of_twelve_turns():
    draft = make_draft(max_turns=12)

    for _ in range(12):
        draft.submit({"action": "inspect", "target": "base:move"})

    assert draft.turn_count == 12
    with pytest.raises(RepairInvariantError, match="turn limit"):
        draft.submit({"action": "inspect", "target": "base:move"})


def test_trace_reconstruction_detects_tampered_edit_hash():
    draft = make_draft()
    draft.submit(
        {
            "action": "replace",
            "target": "base:move",
            "source": "move_safely(x)",
        }
    )
    payload = draft.to_trace().to_dict()
    payload["edits"][0]["output_sha256"] = "0" * 64

    tampered = RepairTraceV1.from_dict(json.loads(json.dumps(payload)))
    with pytest.raises(RepairInvariantError, match="output hash"):
        tampered.reconstruct()


def _mixed_edit_and_audit_trace_payload() -> dict[str, object]:
    draft = make_draft()
    draft.submit({"action": "inspect", "target": "base:move"})
    draft.submit(
        {
            "action": "replace",
            "target": "base:move",
            "source": "move_safely(x)",
        }
    )
    draft.submit({"action": "inspect", "target": "base:check"})
    draft.submit(
        {
            "action": "replace",
            "target": "base:move",
            "source": "move_precisely(x)",
        }
    )
    return json.loads(draft.to_trace().to_json())


def test_trace_rejects_tampered_edit_tuple_order() -> None:
    payload = _mixed_edit_and_audit_trace_payload()
    payload["edits"].reverse()

    with pytest.raises(ValueError, match="edits.*turn_index.*strictly increasing"):
        RepairTraceV1.from_dict(payload)


def test_trace_rejects_tampered_audit_tuple_order() -> None:
    payload = _mixed_edit_and_audit_trace_payload()
    payload["audits"].reverse()

    with pytest.raises(ValueError, match="audits.*turn_index.*strictly increasing"):
        RepairTraceV1.from_dict(payload)


def test_trace_rejects_cross_list_turn_chronology_tampering() -> None:
    payload = _mixed_edit_and_audit_trace_payload()
    # Preserve per-tuple ordering and global contiguous indices while changing which event
    # occurred at turns 2 and 3.  The immutable event-chain hash must still detect this.
    payload["audits"][1]["turn_index"] = 2
    payload["edits"][0]["turn_index"] = 3

    with pytest.raises(ValueError, match="event_sequence_sha256"):
        RepairTraceV1.from_dict(payload)


def test_base_spans_must_match_the_immutable_p0_source():
    with pytest.raises(RepairInvariantError, match="span source"):
        RepairDraft(
            task_id="task",
            environment_seed=1,
            program_sample_id="sample",
            repair_trajectory_id="repair",
            base_source=BASE_SOURCE,
            base_units=(BaseUnitSpan("move", 0, 4, expected_source="nope"),),
        )
