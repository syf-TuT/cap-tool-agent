from capx.tools.franka_metadata import FRANKA_TOOL_METADATA


def test_franka_tool_metadata_marks_core_tools():
    assert FRANKA_TOOL_METADATA["segment_sam3_text_prompt"]["tags"] == ["perception"]
    assert "low_confidence_mask" in FRANKA_TOOL_METADATA["segment_sam3_text_prompt"]["failure_modes"]
    assert FRANKA_TOOL_METADATA["solve_ik"]["tags"] == ["planning"]
    assert FRANKA_TOOL_METADATA["move_to_joints"]["tags"] == ["execution"]
