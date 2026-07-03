from __future__ import annotations

from typing import Any

from capx.tools.schema import StepFeedback, ToolCall, ToolResult

MIN_MASK_SCORE = 0.4
MIN_MASK_AREA = 50
PERCEPTION_TOOLS = {
    "segment_sam3_text_prompt",
    "segment_sam3_point_prompt",
    "segment_sam2",
    "point_prompt_molmo",
}
EXECUTION_TOOLS = {
    "move_to_joints",
    "move_to_joints_arm0",
    "move_to_joints_arm1",
    "move_to_joints_both",
    "open_gripper",
    "close_gripper",
    "open_gripper_arm0",
    "close_gripper_arm0",
    "open_gripper_arm1",
    "close_gripper_arm1",
}


class StepVerifier:
    def verify(
        self,
        *,
        step_id: int,
        tool_call: ToolCall,
        result: ToolResult,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> StepFeedback:
        if result.status in {"failed", "invalid"}:
            return StepFeedback(
                step_id=step_id,
                tool=tool_call.tool,
                status=result.status,
                failure_stage=self._failure_stage(tool_call.tool),
                failure_type=result.failure_type,
                evidence={"message": result.message},
                repair_hints=self._repair_hints(tool_call.tool, result.failure_type),
                recommended_next_tools=self._recommended_next_tools(tool_call.tool),
            )

        if tool_call.tool in PERCEPTION_TOOLS:
            feedback = self._verify_perception(step_id, tool_call, result)
            if feedback is not None:
                return feedback

        evidence = {}
        if tool_call.tool in EXECUTION_TOOLS:
            evidence["reward_before"] = before.get("reward")
            evidence["reward_after"] = after.get("reward")

        return StepFeedback(
            step_id=step_id,
            tool=tool_call.tool,
            status=result.status,
            evidence=evidence,
        )

    def _verify_perception(
        self,
        step_id: int,
        tool_call: ToolCall,
        result: ToolResult,
    ) -> StepFeedback | None:
        summary = result.output_summary
        if not isinstance(summary, dict):
            return None
        if tool_call.tool == "point_prompt_molmo":
            missing_points = [
                ref
                for ref, nested_summary in (summary.get("nested_refs") or {}).items()
                if isinstance(nested_summary, dict) and "None" in str(nested_summary.get("repr", ""))
            ]
            if missing_points:
                return StepFeedback(
                    step_id=step_id,
                    tool=tool_call.tool,
                    status="warning",
                    failure_stage="perception",
                    failure_type="point_not_found",
                    evidence={"missing_points": missing_points},
                    repair_hints=["retry point prompting with a clearer prompt or use segmentation"],
                    recommended_next_tools=[tool_call.tool, "segment_sam3_text_prompt"],
                )
        score = summary.get("best_score")
        area = summary.get("mask_area")
        if (score is not None and score < MIN_MASK_SCORE) or (
            area is not None and area < MIN_MASK_AREA
        ):
            return StepFeedback(
                step_id=step_id,
                tool=tool_call.tool,
                status="warning",
                failure_stage="perception",
                failure_type="low_confidence_mask",
                evidence={"best_score": score, "mask_area": area},
                repair_hints=["retry segmentation with a clearer prompt or point prompt"],
                recommended_next_tools=[tool_call.tool],
            )
        return None

    def _failure_stage(self, tool: str) -> str:
        if tool in PERCEPTION_TOOLS:
            return "perception"
        if tool in EXECUTION_TOOLS:
            return "execution"
        if "ik" in tool or "grasp" in tool or "plan" in tool:
            return "planning"
        return "tool_call"

    def _repair_hints(self, tool: str, failure_type: str | None) -> list[str]:
        if failure_type == "unknown_tool":
            return ["choose one of the registered tools"]
        if tool in PERCEPTION_TOOLS:
            return ["retry perception with a more specific prompt"]
        if "ik" in tool:
            return ["choose another target pose or raise the pregrasp height"]
        return []

    def _recommended_next_tools(self, tool: str) -> list[str]:
        if tool in PERCEPTION_TOOLS:
            return [tool]
        if "ik" in tool:
            return ["solve_ik"]
        return [tool]
