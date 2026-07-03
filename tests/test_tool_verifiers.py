from capx.tools.schema import ToolCall, ToolResult
from capx.tools.verifiers import StepVerifier


def test_verifier_passes_successful_perception_result():
    verifier = StepVerifier()
    result = ToolResult(
        tool="segment_sam3_text_prompt",
        status="success",
        output_summary={"count": 1, "best_score": 0.92, "mask_area": 1200},
    )

    feedback = verifier.verify(
        step_id=1,
        tool_call=ToolCall(tool="segment_sam3_text_prompt"),
        result=result,
        before={},
        after={},
    )

    assert feedback.status == "success"


def test_verifier_flags_low_confidence_mask():
    verifier = StepVerifier()
    result = ToolResult(
        tool="segment_sam3_text_prompt",
        status="success",
        output_summary={"count": 1, "best_score": 0.2, "mask_area": 10},
    )

    feedback = verifier.verify(
        step_id=1,
        tool_call=ToolCall(tool="segment_sam3_text_prompt"),
        result=result,
        before={},
        after={},
    )

    assert feedback.status == "warning"
    assert feedback.failure_type == "low_confidence_mask"
    assert "segment_sam3_text_prompt" in feedback.recommended_next_tools


def test_verifier_flags_missing_molmo_point():
    verifier = StepVerifier()
    result = ToolResult(
        tool="point_prompt_molmo",
        status="success",
        output_summary={
            "type": "dict",
            "nested_refs": {
                "red cube": {
                    "ref": "point_prompt_molmo.0.red cube",
                    "type": "tuple",
                    "repr": "(None, None)",
                }
            },
        },
    )

    feedback = verifier.verify(
        step_id=1,
        tool_call=ToolCall(tool="point_prompt_molmo"),
        result=result,
        before={},
        after={},
    )

    assert feedback.status == "warning"
    assert feedback.failure_type == "point_not_found"
    assert "point_prompt_molmo" in feedback.recommended_next_tools


def test_verifier_converts_failed_result_to_feedback():
    verifier = StepVerifier()
    result = ToolResult.failed(tool="solve_ik", failure_type="exception", message="bad")

    feedback = verifier.verify(
        step_id=2,
        tool_call=ToolCall(tool="solve_ik"),
        result=result,
        before={},
        after={},
    )

    assert feedback.status == "failed"
    assert feedback.failure_type == "exception"
