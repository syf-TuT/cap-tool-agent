"""Single-trial execution for CaP-X environments.

This module handles single trial execution including code generation,
multi-turn decisions, and visual feedback. It contains the core trial
loop extracted from launch.py, covering:

- Initial code generation and oracle code handling
- Code block execution with multi-turn regeneration
- Visual feedback capture and image/video differencing
- Trial artifact saving (code, logs, per-turn videos, combined video)
"""

from __future__ import annotations

import ast
import base64
import copy
import gc
import hashlib
import io
import json
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from capx.envs.configs.instantiate import instantiate
from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.runtime_control import (
    CapsuleExecutor,
    CodeRegion,
    CodeRegionGroup,
    ProgramContractViolation,
    RuntimeAction,
    RuntimeEvent,
    RuntimeTrace,
    analyze_capsule_program_contract_details,
    build_capsule_prompt,
    build_runtime_feedback,
    parse_runtime_action_response,
    replace_region_source,
    segment_python_code,
    segment_python_code_groups,
)
from capx.runtime_control.feedback import validate_progress_mode
from capx.runtime_control.prompts import (
    build_capsule_recovery_prompt,
    build_capsule_terminal_recovery_prompt,
)
from capx.runtime_control.segmenter import ROBOT_SIDE_EFFECT_CALLS
from capx.runtime_control.side_effects import collect_side_effect_calls

from capx.llm.client import (
    VLM_MODELS,
    ModelQueryArgs,
    query_model as _query_model,
    query_model_ensemble as _query_model_ensemble,
    query_single_model_ensemble as _query_single_model_ensemble,
)
from capx.llm.context import llm_call_stage
from capx.utils.launch_utils import (
    TrialSummary,
    _build_multi_turn_decision_prompt,
    _build_multi_turn_decision_prompt_legacy,
    _extract_code,
    _get_visual_feedback,
    _parse_multi_turn_decision,
    _save_trial_artifacts,
)
from capx.utils.video_utils import _encode_video_base64, _write_video

# Use TYPE_CHECKING to avoid circular imports for type hints only
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from capx.envs.launch import LaunchArgs


MULTITURN_LIMIT = 10

# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------

def _annotate_code_blocks(
    code_blocks: list[str],
    code_block_metadata: list[dict[str, Any]],
) -> str:
    """Join code blocks into a single string with ``# Code block N`` headers."""
    annotated = []
    for i, (block, metadata) in enumerate(zip(code_blocks, code_block_metadata, strict=False)):
        annotated.append(f"# Code block {i}\n{block}")
    return "\n\n".join(annotated)


def _build_log_lines(
    final_code: str,
    info_step: dict[str, Any],
    reward: float,
    terminated: bool,
    truncated: bool,
    num_regenerations: int,
    num_finishes: int,
    num_code_blocks: int,
    *,
    prefix: str = "",
    stderr_override: str | None = None,
) -> list[str]:
    """Build the standard log-line list used for both normal and timeout summaries."""
    stderr = stderr_override if stderr_override is not None else info_step.get("stderr", "")
    lines = ["-" * 100]
    if prefix:
        lines.append(prefix)
    lines.extend([
        "Generated program:",
        final_code if final_code else "(no program available)",
        "\n\nEnvironment response:",
        f"  Sandbox failed: {info_step.get('sandbox_rc', 1)}",
        f"  Stdout: {info_step.get('stdout', '')}",
        f"  Stderr: {stderr}",
        f"  Reward: {reward}",
        f"  Task Completed: {info_step.get('task_completed', False)}",
        f"  Terminated: {terminated}, Truncated: {truncated}",
        f"  Num Regenerations: {num_regenerations}",
        f"  Num Finishes: {num_finishes}",
        f"  Num Code Blocks: {num_code_blocks}",
        "-" * 100,
    ])
    return lines


# ---------------------------------------------------------------------------
# Trial video directory helper
# ---------------------------------------------------------------------------

def _trial_video_dir(
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
) -> str:
    """Return the trial output directory path used for video saving."""
    return os.path.join(
        config["output_dir"],
        f"trial_{trial:02d}_sandboxrc_{info_step['sandbox_rc']}_reward_{reward:.3f}"
        f"_taskcompleted_{int(info_step.get('task_completed', False))}",
    )


def _save_trial_video(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
    num_code_blocks: int,
    *,
    suffix_extra: str = "",
) -> None:
    """Save recorded video frames from the environment, if available."""
    if not config["record_video"] or not hasattr(env, "get_video_frames"):
        return
    frames = env.get_video_frames(clear=True)
    if not frames or not config["output_dir"]:
        return

    base_dir = _trial_video_dir(config, trial, info_step, reward)
    suffix = f"{reward:.3f}"
    if suffix_extra:
        suffix += f"_{suffix_extra}"

    if isinstance(frames, list):
        _write_video(frames, base_dir, suffix=suffix)
    elif isinstance(frames, dict):
        for key, frame in frames.items():
            _write_video(frame, base_dir, suffix=f"{suffix}_{key}")


def _save_turn_and_combined_videos(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
    turn_frame_ranges: list[tuple[int, int]],
) -> None:
    """Save per-turn videos and a combined video of all turns.

    Gets all frames from the environment (clearing the buffer), then writes:
      - ``video_turn_00.mp4``, ``video_turn_01.mp4``, ... for each turn
      - ``video_combined.mp4`` for the full trial
      - If wrist camera is enabled: ``video_turn_00_wrist.mp4``, etc.
    """
    if not config["record_video"] or not config["output_dir"]:
        return
    if not hasattr(env, "get_video_frames"):
        return

    all_frames = env.get_video_frames(clear=True)
    if not all_frames:
        return

    base_dir = _trial_video_dir(config, trial, info_step, reward)

    # all_frames may be a list (Robosuite) or a dict of lists (R1Pro multi-camera).
    # Normalise to a list for slicing; dict case is handled by _write_multi_video.
    if isinstance(all_frames, dict):
        # Multi-camera: write each camera stream as a combined video
        for key, frames in all_frames.items():
            if frames:
                _write_video(frames, base_dir, suffix=f"combined_{key}")
        return

    # Per-turn videos
    for i, (start, end) in enumerate(turn_frame_ranges):
        turn_frames = all_frames[start:end]
        if turn_frames:
            _write_video(turn_frames, base_dir, suffix=f"turn_{i:02d}")

    # Combined video
    _write_video(all_frames, base_dir, suffix="combined")

    # Wrist camera videos
    if config.get("use_wrist_camera") and hasattr(env, "get_wrist_video_frames"):
        wrist_frames = env.get_wrist_video_frames(clear=True)
        if wrist_frames:
            for i, (start, end) in enumerate(turn_frame_ranges):
                wrist_turn = wrist_frames[start:end]
                if wrist_turn:
                    _write_video(wrist_turn, base_dir, suffix=f"turn_{i:02d}_wrist")
            _write_video(wrist_frames, base_dir, suffix="combined_wrist")


# ---------------------------------------------------------------------------
# Visual feedback and image differencing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapsuleVisual:
    """One current camera observation prepared for a Capsule prompt."""

    camera: str
    data_url: str
    image: Image.Image
    width: int
    height: int
    sha256: str

    def metadata(self) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "width": self.width,
            "height": self.height,
            "sha256": self.sha256,
        }


def _capsule_capture_error(camera: str, error: str) -> dict[str, str]:
    return {"camera": camera, "error": error}


def _visual_payloads(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _decode_image_data_url(data_url: str) -> tuple[str, bytes]:
    header, encoded = data_url.split(",", 1)
    if not header.startswith("data:") or ";base64" not in header:
        raise ValueError("visual payload is not a base64 data URL")
    media_type = header[5:].split(";", 1)[0] or "application/octet-stream"
    return media_type, base64.b64decode(encoded, validate=True)


def _capsule_visual_from_payload(
    camera: str,
    data_url: Any,
    image: Any,
) -> CapsuleVisual:
    if not isinstance(data_url, str) or not isinstance(image, Image.Image):
        raise ValueError("visual capture returned no image")
    _, png_bytes = _decode_image_data_url(data_url)
    independent_image = image.copy()
    width, height = independent_image.size
    return CapsuleVisual(
        camera=camera,
        data_url=data_url,
        image=independent_image,
        width=width,
        height=height,
        sha256=hashlib.sha256(png_bytes).hexdigest(),
    )


def _try_capsule_visual(
    camera: str,
    data_url: Any,
    image: Any,
) -> CapsuleVisual | None:
    try:
        return _capsule_visual_from_payload(camera, data_url, image)
    except Exception:
        return None


def _capture_capsule_main(env: CodeExecutionEnvBase) -> CapsuleVisual | None:
    try:
        data_url, image = _get_visual_feedback(env, use_wrist_camera=False)
    except Exception:
        return None
    return _try_capsule_visual("main", data_url, image)


def _capture_capsule_visuals(
    env: CodeExecutionEnvBase,
    *,
    use_wrist_camera: bool = False,
) -> tuple[list[CapsuleVisual], list[dict[str, str]]]:
    """Capture current main/wrist observations without retaining stale frames."""
    if not use_wrist_camera:
        main_record = _capture_capsule_main(env)
        if main_record is not None:
            return [main_record], []
        return [], [_capsule_capture_error("main", "capture_failed")]

    combined_data_urls: list[Any] = []
    combined_images: list[Any] = []
    try:
        data_urls, images = _get_visual_feedback(env, use_wrist_camera=True)
        combined_data_urls = _visual_payloads(data_urls)
        combined_images = _visual_payloads(images)
    except Exception:
        combined_data_urls = combined_images = []

    if len(combined_data_urls) >= 2 and len(combined_images) >= 2:
        main_record = _try_capsule_visual(
            "main", combined_data_urls[0], combined_images[0]
        )
        wrist_record = _try_capsule_visual(
            "wrist", combined_data_urls[1], combined_images[1]
        )
        if main_record is None:
            main_record = _capture_capsule_main(env)

        records = [record for record in (main_record, wrist_record) if record is not None]
        errors = []
        if main_record is None:
            errors.append(_capsule_capture_error("main", "capture_failed"))
        if wrist_record is None:
            errors.append(_capsule_capture_error("wrist", "capture_failed"))
        return records, errors

    main_record = _capture_capsule_main(env)
    records = [main_record] if main_record is not None else []
    errors = []
    if main_record is None:
        errors.append(_capsule_capture_error("main", "capture_failed"))
    wrist_error = (
        "camera_unavailable" if not hasattr(env, "render_wrist") else "capture_failed"
    )
    errors.append(_capsule_capture_error("wrist", wrist_error))
    return records, errors


def _attach_capsule_visuals(
    prompt: list[dict[str, Any]],
    records: list[CapsuleVisual],
    errors: list[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Return a prompt copy with current camera observations appended."""
    attached = copy.deepcopy(prompt)
    if not attached:
        attached.append({"role": "user", "content": []})

    content = attached[-1].get("content", [])
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    elif not isinstance(content, list):
        content = [{"type": "text", "text": str(content)}]
    else:
        content = list(content)
    attached[-1]["content"] = content

    for record in records:
        content.extend([
            {"type": "text", "text": f"Current {record.camera}-camera view"},
            {"type": "image_url", "image_url": {"url": record.data_url}},
        ])
    for error in errors:
        camera = error.get("camera", "unknown")
        error_code = error.get("error", "capture_failed")
        content.append({
            "type": "text",
            "text": f"Current {camera}-camera view unavailable ({error_code}).",
        })
    return attached


_DATA_URL_PATTERN = re.compile(
    r"data:(?P<media>[-\w.+/]+);base64,(?P<payload>[A-Za-z0-9+/=]+)"
)


def _sanitize_text_data_urls(value: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        try:
            payload = base64.b64decode(match.group("payload"), validate=True)
        except (ValueError, TypeError):
            payload = match.group("payload").encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return f"[image_reference sha256={digest}]"

    sanitized = _DATA_URL_PATTERN.sub(replacement, value)
    if "base64," in sanitized:
        prefix, _, _ = sanitized.partition("base64,")
        return f"{prefix}[binary_payload_redacted]"
    return sanitized


def _image_reference_metadata(
    value: Any,
    *,
    camera: str,
    artifact_by_sha256: Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        image_url = value.get("url", "")
    else:
        image_url = value
    image_url = image_url if isinstance(image_url, str) else str(image_url)

    width = height = None
    media_type = "unknown"
    try:
        media_type, image_bytes = _decode_image_data_url(image_url)
        digest = hashlib.sha256(image_bytes).hexdigest()
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
    except Exception:
        digest = hashlib.sha256(image_url.encode("utf-8")).hexdigest()

    artifact = artifact_by_sha256.get(digest)
    artifact_metadata = artifact if isinstance(artifact, Mapping) else {}
    path = artifact_metadata.get("path") if artifact_metadata else artifact
    return {
        "camera": artifact_metadata.get("camera", camera),
        "path": str(path).replace("\\", "/") if path is not None else None,
        "width": artifact_metadata.get("width", width),
        "height": artifact_metadata.get("height", height),
        "sha256": digest,
        "media_type": artifact_metadata.get("media_type", media_type),
    }


def _sanitize_multimodal_prompt(
    prompt: Any,
    artifact_by_sha256: Mapping[str, Any] | None = None,
) -> Any:
    """Deep-copy a prompt while replacing image payloads with artifact metadata."""
    artifacts = artifact_by_sha256 or {}

    def sanitize(value: Any, *, camera: str = "unknown") -> Any:
        if isinstance(value, list):
            sanitized_items = []
            next_camera = camera
            image_count = 0
            for item in value:
                if isinstance(item, Mapping) and item.get("type") == "text":
                    text = str(item.get("text", "")).lower()
                    if "wrist-camera" in text or "wrist camera" in text:
                        next_camera = "wrist"
                    elif (
                        "main-camera" in text
                        or "main camera" in text
                        or "image of the initial state" in text
                    ):
                        next_camera = "main"
                if isinstance(item, Mapping) and item.get("type") == "image_url":
                    item_camera = next_camera
                    if item_camera == "unknown" and image_count == 0:
                        item_camera = "main"
                    sanitized_items.append(sanitize(item, camera=item_camera))
                    image_count += 1
                    next_camera = "unknown"
                else:
                    sanitized_items.append(sanitize(item, camera=next_camera))
            return sanitized_items
        if isinstance(value, Mapping):
            if value.get("type") == "image_url" or "image_url" in value:
                image_value = value.get("image_url", value)
                return {
                    "type": "image_reference",
                    "image_reference": _image_reference_metadata(
                        image_value,
                        camera=camera,
                        artifact_by_sha256=artifacts,
                    ),
                }
            return {key: sanitize(item, camera=camera) for key, item in value.items()}
        if isinstance(value, str):
            return _sanitize_text_data_urls(value)
        return copy.deepcopy(value)

    return sanitize(copy.deepcopy(prompt))


def _save_capsule_visuals(
    records: list[CapsuleVisual],
    output_dir: str | os.PathLike[str] | None,
    *,
    trial_id: int,
    step_id: int,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Persist captured PNG payloads and return hashes mapped to relative paths."""
    if output_dir is None:
        return {}, []

    output_root = Path(output_dir)
    visual_dir = output_root / f"capsule_visuals_trial_{trial_id:02d}"
    artifact_by_sha256: dict[str, str] = {}
    errors: list[dict[str, str]] = []
    try:
        visual_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return {}, [_capsule_capture_error("all", "save_failed")]

    for record in records:
        relative_path = Path(visual_dir.name) / f"step_{step_id:02d}_{record.camera}.png"
        try:
            _, png_bytes = _decode_image_data_url(record.data_url)
            if hashlib.sha256(png_bytes).hexdigest() != record.sha256:
                raise ValueError("visual payload hash mismatch")
            (output_root / relative_path).write_bytes(png_bytes)
            artifact_by_sha256[record.sha256] = relative_path.as_posix()
        except (OSError, ValueError, TypeError):
            errors.append(_capsule_capture_error(record.camera, "save_failed"))
    return artifact_by_sha256, errors

def _capture_initial_visual_feedback(
    env: CodeExecutionEnvBase,
    obs: dict[str, Any],
    config: dict[str, Any],
    args: LaunchArgs,
    visual_differencing_args: ModelQueryArgs,
) -> tuple[list, list[str], str]:
    """Capture the initial environment image and optionally describe it.

    Returns:
        (visual_feedback_imgs, visual_feedback_base64_history, task_description)
    """
    visual_feedback_imgs: list = []
    visual_feedback_base64_history: list[str] = []
    task_description = ""

    use_wrist = config.get("use_wrist_camera", False)

    needs_visual = (
        (config["use_visual_feedback"] and args.model in VLM_MODELS)
        or (config["use_img_differencing"] and visual_differencing_args.model in VLM_MODELS)
        or config.get("use_video_differencing", False)
    )
    if not (needs_visual and hasattr(env, "render")):
        return visual_feedback_imgs, visual_feedback_base64_history, task_description

    initial_base64, initial_img = _get_visual_feedback(env)
    visual_feedback_imgs.append(initial_img)
    visual_feedback_base64_history.append(initial_base64)
    task_description = copy.deepcopy(obs["full_prompt"][-1]["content"][0]["text"])

    # Also capture wrist camera image for multiview initial description
    initial_wrist_base64 = None
    if use_wrist and hasattr(env, "render_wrist"):
        wrist_img = env.render_wrist()
        if wrist_img is not None:
            pil_wrist = Image.fromarray(wrist_img)
            buf = io.BytesIO()
            pil_wrist.save(buf, format="png")
            initial_wrist_base64 = (
                f"data:image/png;base64,"
                f"{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            )
            visual_feedback_imgs.append(pil_wrist)

    # Append image to the prompt for VLM visual feedback
    if config["use_visual_feedback"]:
        obs["full_prompt"][-1]["content"][0]["text"] += (
            "\n\nIncluded below is an image of the initial state of the environment."
        )
        obs["full_prompt"][-1]["content"].append(
            {"type": "image_url", "image_url": {"url": initial_base64}}
        )
        if initial_wrist_base64 is not None:
            obs["full_prompt"][-1]["content"].append(
                {
                    "type": "text",
                    "text": "Included below is an image from the robot's wrist camera.",
                }
            )
            obs["full_prompt"][-1]["content"].append(
                {"type": "image_url", "image_url": {"url": initial_wrist_base64}}
            )

    # Image differencing: ask a VLM to describe the initial scene
    if config["use_img_differencing"] or config.get("use_video_differencing", False):
        description = _describe_initial_scene(
            visual_differencing_args, task_description, initial_base64,
            wrist_image_base64=initial_wrist_base64,
        )
        feedback = f"The initial state of the environment is described as follows:\n{description}"
        obs["full_prompt"][-1]["content"][0]["text"] += f"\n\n{feedback}"
        if args.debug:
            print(description)

    return visual_feedback_imgs, visual_feedback_base64_history, task_description


def _describe_initial_scene(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    image_base64: str,
    wrist_image_base64: str | None = None,
) -> str:
    """Query a VLM to describe the initial environment state."""
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "Describe the initial state of the environment with the goal of the "
                "task in mind. You should try to provide objective information and no "
                "assumptions. Do *NOT* write any code."
            ),
        },
        {"type": "text", "text": "Main camera view:"},
        {"type": "image_url", "image_url": {"url": image_base64}},
    ]
    if wrist_image_base64 is not None:
        user_content.extend([
            {"type": "text", "text": "Wrist camera view:"},
            {"type": "image_url", "image_url": {"url": wrist_image_base64}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes the initial state of the "
                "environment with the goal of the task in mind. You should try to provide "
                "objective information and no assumptions. Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    with llm_call_stage("visual_feedback"):
        return _query_model(visual_differencing_args, prompt)["content"]


def _get_visual_differencing_feedback(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    visual_feedback_base64_history: list[str],
    wrist_base64_history: list[str] | None = None,
) -> str | None:
    """Query a VLM to describe what changed between the two most recent frames.

    Args:
        wrist_base64_history: Optional history of wrist camera images.  When provided
            and has >=2 entries, the before/after wrist images are included in the prompt.
    """
    if len(visual_feedback_base64_history) < 2:
        return None

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "Describe the difference between the current state of the "
                "environment and the previous state of the environment with the "
                "goal of the task in mind and whether the task has been completed. "
                "You should try to provide objective information and no assumptions. "
                "Do *NOT* write any code.."
            ),
        },
        {"type": "text", "text": "Previous state (main camera):"},
        {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-2]}},
        {"type": "text", "text": "Current state (main camera):"},
        {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-1]}},
    ]

    if wrist_base64_history and len(wrist_base64_history) >= 2:
        user_content.extend([
            {"type": "text", "text": "Previous state (wrist camera):"},
            {"type": "image_url", "image_url": {"url": wrist_base64_history[-2]}},
            {"type": "text", "text": "Current state (wrist camera):"},
            {"type": "image_url", "image_url": {"url": wrist_base64_history[-1]}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes the difference between the "
                "current state of the environment and the previous state of the environment "
                "with the goal of the task in mind and whether the task has been completed. "
                "You should try to provide objective information and no assumptions. "
                "Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    with llm_call_stage("visual_feedback"):
        return _query_model(visual_differencing_args, prompt)["content"]


# ---------------------------------------------------------------------------
# Video differencing
# ---------------------------------------------------------------------------

def _get_video_differencing_feedback(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    turn_frames: list[np.ndarray],
    wrist_turn_frames: list[np.ndarray] | None = None,
) -> str | None:
    """Query a VLM with a video of the turn execution to describe what happened.

    Args:
        visual_differencing_args: Model query args for the VDM model.
        task_description: The task goal.
        turn_frames: RGB frames from the main camera for this turn.
        wrist_turn_frames: RGB frames from the wrist camera for this turn (optional).

    Returns:
        Text description of the execution, or None if no frames.
    """
    if not turn_frames:
        return None

    video_base64 = _encode_video_base64(turn_frames)

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "The following video shows the robot executing code in the "
                "environment from the main camera view. Describe what happened "
                "during execution, including what actions the robot took, how "
                "the objects in the scene changed, and whether the task appears "
                "to have been completed. Provide objective information and no "
                "assumptions. Do *NOT* write any code."
            ),
        },
        {"type": "text", "text": "Main camera video:"},
        {"type": "image_url", "image_url": {"url": video_base64}},
    ]

    if wrist_turn_frames:
        wrist_video_base64 = _encode_video_base64(wrist_turn_frames)
        user_content.extend([
            {
                "type": "text",
                "text": (
                    "The following video shows the same execution from the "
                    "robot's wrist-mounted camera (eye-in-hand view), providing "
                    "a close-up perspective of the gripper and objects being "
                    "manipulated."
                ),
            },
            {"type": "text", "text": "Wrist camera video:"},
            {"type": "image_url", "image_url": {"url": wrist_video_base64}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that analyzes robot execution "
                "videos. You describe what happened during the robot's code "
                "execution, what actions were taken, how the environment "
                "changed, and whether the task appears to have been completed. "
                "Provide objective information and no assumptions. "
                "Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    with llm_call_stage("visual_feedback"):
        return _query_model(visual_differencing_args, prompt)["content"]


# ---------------------------------------------------------------------------
# Initial code generation
# ---------------------------------------------------------------------------

def _query_initial_code(
    args: LaunchArgs,
    config: dict[str, Any],
    obs: dict[str, Any],
    artifact_by_sha256: Mapping[str, Any] | None = None,
) -> tuple[str, str | None, dict | None]:
    """Query the model for the initial code generation.

    Returns:
        (raw_code, reasoning, ensemble_data)
    """
    # Save the initial prompt
    with open(os.path.join(config["output_dir"], "initial_prompt.txt"), "w") as f:
        f.write(
            str(
                _sanitize_multimodal_prompt(
                    obs["full_prompt"], artifact_by_sha256=artifact_by_sha256
                )
            )
        )

    ensemble_data = None
    if config["use_parallel_ensemble"]:
        if config.get("use_multimodel", False):
            print("RUNNING MULTIMODEL ENSEMBLE QUERY")
            with llm_call_stage("initial_code"):
                out = _query_model_ensemble(args, obs["full_prompt"], is_multiturn=False)
        else:
            print("RUNNING SINGLE MODEL ENSEMBLE QUERY")
            with llm_call_stage("initial_code"):
                out = _query_single_model_ensemble(
                    args, obs["full_prompt"], args.model, is_multiturn=False
                )
        ensemble_data = {
            "ensemble_candidates_txt": out["ensemble_candidates_txt"],
            "ensemble_synthesis_txt": out["ensemble_synthesis_txt"],
        }
    else:
        with llm_call_stage("initial_code"):
            out = _query_model(args, obs["full_prompt"])

    return out["content"], out["reasoning"], ensemble_data


# ---------------------------------------------------------------------------
# Multi-turn decision handling
# ---------------------------------------------------------------------------

def _handle_multi_turn_step(
    env: CodeExecutionEnvBase,
    obs: dict[str, Any],
    args: LaunchArgs,
    config: dict[str, Any],
    visual_differencing_args: ModelQueryArgs,
    multi_turn_prompt: str,
    code_blocks: list[str],
    code_block_idx: int,
    info_step: dict[str, Any],
    task_description: str,
    visual_feedback_imgs: list,
    visual_feedback_base64_history: list[str],
    stderr_history: list[str],
    turn_frames: list[np.ndarray] | None = None,
    wrist_turn_frames: list[np.ndarray] | None = None,
    wrist_base64_history: list[str] | None = None,
) -> tuple[str, str | None, str | None, dict | None, list | None]:
    """Execute one multi-turn decision step.

    Captures visual feedback, builds the decision prompt, queries the model,
    and returns the parsed decision.

    Args:
        turn_frames: Frames from the main camera for this turn (for video differencing).
        wrist_turn_frames: Frames from the wrist camera for this turn (for video differencing).
        wrist_base64_history: History of wrist camera base64 images for image-based
            differencing with multiview.

    Returns:
        (decision, new_code, reasoning, multiturn_ensemble_entry)
        where decision is "regenerate", "finish", or "continue".
    """
    use_wrist = config.get("use_wrist_camera", False)

    executed_code = "\n".join(code_blocks[:code_block_idx])
    complete_multi_turn_prompt = multi_turn_prompt.format(
        executed_code=executed_code,
        console_stdout=info_step["stdout"],
        console_stderr=info_step["stderr"],
    )

    if info_step["stderr"] != "":
        stderr_history.append(info_step["stderr"])

    # Capture visual feedback if applicable
    visual_feedback_base64 = None
    needs_visual = (
        (config["use_visual_feedback"] and args.model in VLM_MODELS)
        or (config["use_img_differencing"] and visual_differencing_args.model in VLM_MODELS)
    )
    if needs_visual and hasattr(env, "render"):
        vf_base64, vf_img = _get_visual_feedback(env)
        visual_feedback_imgs.append(vf_img)
        visual_feedback_base64_history.append(vf_base64)

        # Also capture wrist camera snapshot for image-based multiview
        if use_wrist and hasattr(env, "render_wrist") and wrist_base64_history is not None:
            wrist_result = _get_visual_feedback(env, use_wrist_camera=True)
            if wrist_result[0] is not None and isinstance(wrist_result[0], list) and len(wrist_result[0]) > 1:
                wrist_base64_history.append(wrist_result[0][1])  # index 1 = wrist image

    # Determine differencing feedback
    differencing_feedback = None
    is_video_feedback = False

    if config.get("use_video_differencing") and turn_frames:
        # Video-based differencing: pass video of this turn to VDM
        differencing_feedback = _get_video_differencing_feedback(
            visual_differencing_args, task_description, turn_frames, wrist_turn_frames,
        )
        is_video_feedback = True
    elif config["use_img_differencing"] and len(visual_feedback_base64_history) >= 2:
        # Image-based differencing: pass before/after images to VDM
        differencing_feedback = _get_visual_differencing_feedback(
            visual_differencing_args, task_description, visual_feedback_base64_history,
            wrist_base64_history=wrist_base64_history,
        )

    # Only pass visual feedback to prompt if visual_feedback is enabled
    if not config["use_visual_feedback"]:
        visual_feedback_base64 = None
    elif needs_visual and hasattr(env, "render"):
        visual_feedback_base64 = visual_feedback_base64_history[-1] if visual_feedback_base64_history else None

    # Build decision prompt
    if args.use_legacy_multi_turn_decision_prompt:
        print("Using legacy multi-turn decision prompt")
        decision_prompt = _build_multi_turn_decision_prompt_legacy(
            obs, complete_multi_turn_prompt, visual_feedback_base64, differencing_feedback,
            is_video_feedback=is_video_feedback,
        )
    else:
        decision_prompt = _build_multi_turn_decision_prompt(
            obs, complete_multi_turn_prompt, visual_feedback_base64, differencing_feedback,
            is_video_feedback=is_video_feedback,
        )

    # Query model
    multiturn_ensemble_entry = None
    if config["use_parallel_ensemble"]:
        if config.get("use_multimodel", False):
            print("RUNNING MULTITURN MULTIMODEL ENSEMBLE QUERY")
            with llm_call_stage("multi_turn"):
                content = _query_model_ensemble(args, decision_prompt, is_multiturn=True)
        else:
            print("RUNNING MULTITURN SINGLE MODEL ENSEMBLE QUERY")
            with llm_call_stage("multi_turn"):
                content = _query_single_model_ensemble(
                    args, decision_prompt, args.model, is_multiturn=True
                )
        multiturn_ensemble_entry = {
            "ensemble_candidates_txt": content.get("ensemble_candidates_txt", ""),
            "ensemble_synthesis_txt": content.get("ensemble_synthesis_txt", ""),
        }
    else:
        with llm_call_stage("multi_turn"):
            content = _query_model(args, decision_prompt)

    reasoning = content["reasoning"]
    decision, new_code = _parse_multi_turn_decision(content["content"])

    return decision, new_code, reasoning, multiturn_ensemble_entry, decision_prompt


# ---------------------------------------------------------------------------
# Core single-trial execution
# ---------------------------------------------------------------------------

def _task_text_from_obs(obs: dict[str, Any]) -> str:
    try:
        content = obs["full_prompt"][-1]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return str(content)


def _whole_source_fallback_units(
    source: str,
) -> tuple[list[CodeRegion], list[CodeRegionGroup]]:
    line_count = max(1, len(source.splitlines()))
    region = CodeRegion(
        region_id="region_1",
        start_line=1,
        end_line=line_count,
        source=source,
    )
    group = CodeRegionGroup(
        group_id="group_1",
        start_line=1,
        end_line=line_count,
        source=source,
        region_ids=[region.region_id],
    )
    return [region], [group]


def _initial_syntax_error_history_entry(exc: SyntaxError) -> dict[str, Any]:
    return {
        "step_id": 0,
        "action": {"action": "initial_parse"},
        "event": {
            "action": "initial_parse",
            "status": "failed",
            "region_id": "group_1",
            "message": str(exc),
            "evidence": {
                "exception_type": type(exc).__name__,
                "lineno": exc.lineno,
                "offset": exc.offset,
                "text": exc.text,
            },
        },
        "feedback": {
            "status": "failed",
            "region_id": "group_1",
            "message": (
                "Initial source is invalid Python. Use patch_group to replace the "
                "complete group_1 source with valid Python."
            ),
            "repair_hints": [
                "Patch the complete group_1 source before attempting execution."
            ],
        },
    }


def _run_capsule_trial(
    env: CodeExecutionEnvBase,
    trial: int,
    args: LaunchArgs,
    config: dict[str, Any],
    *,
    initial_code: str | None = None,
    scripted_actions: list[dict[str, Any]] | None = None,
) -> TrialSummary:
    mode = str(config.get("capsule_control_mode", "auto_forward"))
    if mode == "auto_forward":
        return _run_capsule_auto_forward_loop(
            env=env,
            trial=trial,
            args=args,
            config=config,
            initial_code=initial_code,
        )
    if mode == "llm_step":
        return _run_capsule_llm_step_loop(
            env=env,
            trial=trial,
            args=args,
            config=config,
            initial_code=initial_code,
            scripted_actions=scripted_actions,
        )
    raise ValueError(f"Unsupported capsule_control_mode: {mode}")


def _run_capsule_auto_forward_loop(
    env: CodeExecutionEnvBase,
    trial: int,
    args: LaunchArgs,
    config: dict[str, Any],
    *,
    initial_code: str | None = None,
) -> TrialSummary:
    output_dir = config.get("output_dir")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if config.get("capsule_execution_granularity", "semantic_group") != "semantic_group":
        raise ValueError(
            "capsule_control_mode='auto_forward' only supports effect-bounded "
            "execution units via capsule_execution_granularity='semantic_group'"
        )

    obs, _ = env.reset(options={"trial": trial}, seed=trial)
    if config.get("record_video") and hasattr(env, "enable_video_capture"):
        env.enable_video_capture(
            True,
            clear=True,
            wrist_camera=config.get("use_wrist_camera", False),
        )

    if initial_code is not None:
        source = initial_code
    elif config.get("use_oracle_code", False):
        source = env.oracle_code
    else:
        raw_code, _, _ = _query_initial_code(args, config, obs)
        source = "\n\n".join(_extract_code(raw_code))

    max_regions_per_group = int(config.get("capsule_max_regions_per_group", 20))
    side_effect_calls = collect_side_effect_calls(getattr(env, "_apis", {}).values())
    if not side_effect_calls:
        side_effect_calls = ROBOT_SIDE_EFFECT_CALLS
    recovery_observation_functions = _collect_recovery_observation_functions(
        getattr(env, "_apis", {}).values()
    )
    initial_syntax_error: SyntaxError | None = None
    try:
        regions = segment_python_code(source)
        groups = segment_python_code_groups(
            source,
            regions,
            max_regions_per_group=max_regions_per_group,
            side_effect_calls=side_effect_calls,
        )
    except SyntaxError as exc:
        initial_syntax_error = exc
        regions, groups = _whole_source_fallback_units(source)
    region_by_id = {region.region_id: region for region in regions}
    group_by_id = {group.group_id: group for group in groups}

    trace = RuntimeTrace()
    executor = CapsuleExecutor(
        base_globals=env._build_capsule_globals(trace=trace),
        trace=trace,
    )
    max_steps = int(config.get("max_capsule_steps", 12))
    action_query_args = copy.copy(args)
    default_action_max_tokens = getattr(args, "max_tokens", 2048 * 10)
    action_query_args.max_tokens = int(
        config.get("capsule_action_max_tokens", default_action_max_tokens)
    )
    history: list[dict[str, Any]] = []
    if initial_syntax_error is not None:
        history.append(_initial_syntax_error_history_entry(initial_syntax_error))
    step_metrics: list[dict[str, Any]] = []
    prompts: list[list[dict[str, Any]]] = []
    failed = False
    executed_regions = 0
    best_reward_so_far: float | None = None
    executed_side_effect_regions: set[str] = set()
    executed_side_effect_groups: set[str] = set()
    recovery_side_effect_budget = 0
    reward_drop_guard_min_best_reward = float(
        config.get("capsule_reward_drop_guard_min_best_reward", 0.6)
    )
    reward_drop_guard_threshold = float(
        config.get("capsule_reward_drop_guard_threshold", 0.25)
    )
    terminal_recovery_attempted = False

    group_index = 0
    group_attempts = 0
    while group_attempts < max_steps:
        if group_index >= len(groups):
            terminal_state = _capsule_state_snapshot(env)
            terminal_reward = _state_reward(terminal_state)
            terminal_succeeded = bool(terminal_state.get("task_completed")) or (
                terminal_reward is not None and terminal_reward >= 1.0
            )
            if terminal_succeeded or terminal_recovery_attempted or not groups:
                break

            terminal_recovery_attempted = True
            last_unit = groups[-1]
            recovery_step_id = len(history) + 1
            recovery_action: RuntimeAction | None = None
            recovery_prompt = build_capsule_terminal_recovery_prompt(
                task=_task_text_from_obs(obs),
                last_unit=last_unit,
                history_tail=history,
                trace_summary=executor.trace.summary() if executor.trace is not None else {},
                side_effect_ledger={
                    "executed_side_effect_regions": sorted(executed_side_effect_regions),
                    "executed_side_effect_groups": sorted(executed_side_effect_groups),
                },
                terminal_state=terminal_state,
                recovery_observation_functions=recovery_observation_functions,
            )
            prompts.append(recovery_prompt)
            try:
                with llm_call_stage("capsule_recovery"):
                    response = _query_model(action_query_args, recovery_prompt)
                response_content = response["content"]
                try:
                    recovery_action = parse_runtime_action_response(response_content)
                except ValueError:
                    recovery_action = _coerce_terminal_append_recovery_action(
                        response_content,
                        recovery_observation_functions=recovery_observation_functions,
                    )
                    if recovery_action is None:
                        raise
                recovery_action.step_id = recovery_step_id
            except ValueError as exc:
                recovery_event = RuntimeEvent(
                    action="invalid",
                    status="invalid",
                    message=str(exc),
                    evidence={"exception_type": type(exc).__name__},
                )
            else:
                if recovery_action.action not in {"append_recovery", "finish"}:
                    recovery_event = RuntimeEvent(
                        action=recovery_action.action,
                        status="invalid",
                        region_id=_runtime_action_unit_id(recovery_action),
                        message=(
                            f"{recovery_action.action} is not allowed during terminal "
                            "recovery."
                        ),
                    )
                else:
                    recovery_event = _no_rollback_guard_event(
                        recovery_action,
                        executed_side_effect_regions,
                        executed_side_effect_groups,
                        group_by_id,
                        recovery_observation_functions=recovery_observation_functions,
                    )
                    if recovery_event is None:
                        recovery_event = _execute_runtime_action(
                            recovery_action,
                            executor,
                            source,
                            region_by_id,
                            group_by_id,
                            recovery_observation_functions=recovery_observation_functions,
                            append_recovery_insert_after_line=last_unit.end_line,
                        )
                    if (
                        recovery_action.action == "append_recovery"
                        and recovery_event.status == "success"
                    ):
                        recovery_side_effect_budget = 1

            recovery_after_state = _capsule_state_snapshot(env)
            recovery_trace_events = recovery_event.evidence.get("trace_events", [])
            recovery_feedback_action = recovery_action or RuntimeAction("invalid", {})
            recovery_feedback = build_runtime_feedback(
                step_id=recovery_step_id,
                action=recovery_feedback_action,
                event=recovery_event,
                region=None,
                trace_events=recovery_trace_events,
                before_state=terminal_state,
                after_state=recovery_after_state,
            )
            history.append(
                {
                    "step_id": recovery_step_id,
                    "action": (
                        recovery_action.to_dict()
                        if recovery_action is not None
                        else {"action": "invalid"}
                    ),
                    "event": recovery_event.to_dict(),
                    "feedback": recovery_feedback.to_dict(),
                    "trace_events": recovery_trace_events,
                    "state_before": terminal_state,
                    "state_after": recovery_after_state,
                }
            )
            recovery_metric, best_reward_so_far = _capsule_step_metric(
                step_id=recovery_step_id,
                action=recovery_action,
                event=recovery_event,
                before_state=terminal_state,
                after_state=recovery_after_state,
                executed_regions=executed_regions,
                best_reward_so_far=best_reward_so_far,
                recovery_execution_attempt=False,
            )
            step_metrics.append(recovery_metric)
            if recovery_event.status in {"failed", "invalid"}:
                failed = True
                break
            if recovery_action is None or recovery_action.action == "finish":
                break
            if recovery_action.action == "append_recovery":
                previous_source = source
                previous_regions = regions
                previous_groups = groups
                source = str(recovery_event.evidence["source"])
                regions = segment_python_code(source)
                groups = segment_python_code_groups(
                    source,
                    regions,
                    max_regions_per_group=max_regions_per_group,
                    side_effect_calls=side_effect_calls,
                )
                region_by_id = {region.region_id: region for region in regions}
                group_by_id = {group.group_id: group for group in groups}
                (
                    executed_side_effect_regions,
                    executed_side_effect_groups,
                ) = _remap_executed_side_effect_ledger(
                    recovery_action,
                    previous_source,
                    source,
                    previous_regions,
                    regions,
                    previous_groups,
                    groups,
                    executed_side_effect_regions,
                    executed_side_effect_groups,
                )
                group_index = _first_group_index_starting_after_line(
                    groups,
                    last_unit.end_line,
                    default=len(groups),
                )
                continue
            break

        group = groups[group_index]
        group_attempts += 1
        step_id = len(history) + 1
        action = RuntimeAction("run_group", {"group_id": group.group_id})
        action.step_id = step_id
        before_state = _capsule_state_snapshot(env)
        consumes_recovery_side_effect = False

        event = _no_rollback_guard_event(
            action,
            executed_side_effect_regions,
            executed_side_effect_groups,
            group_by_id,
            recovery_observation_functions=recovery_observation_functions,
        )
        if event is None:
            event = _reward_drop_guard_event(
                action,
                before_state,
                best_reward_so_far,
                region_by_id,
                group_by_id,
                side_effect_calls,
                recovery_side_effect_budget=recovery_side_effect_budget,
                min_best_reward=reward_drop_guard_min_best_reward,
                drop_threshold=reward_drop_guard_threshold,
                recovery_observation_functions=recovery_observation_functions,
            )
        if event is None:
            consumes_recovery_side_effect = (
                recovery_side_effect_budget > 0
                and _runtime_action_targets_side_effect_unit(
                    action,
                    region_by_id,
                    group_by_id,
                    side_effect_calls,
                )
            )
            event = _execute_runtime_action(
                action,
                executor,
                source,
                region_by_id,
                group_by_id,
                recovery_observation_functions=recovery_observation_functions,
            )
            if consumes_recovery_side_effect:
                recovery_side_effect_budget -= 1

        _record_runtime_side_effect_execution(
            action,
            event,
            region_by_id,
            group_by_id,
            executed_side_effect_regions,
            executed_side_effect_groups,
            side_effect_calls,
        )

        if event.status != "invalid":
            executed_regions += len(group.region_ids)

        after_state = _capsule_state_snapshot(env)
        trace_events = event.evidence.get("trace_events", [])
        feedback = build_runtime_feedback(
            step_id=step_id,
            action=action,
            event=event,
            region=group,
            trace_events=trace_events,
            before_state=before_state,
            after_state=after_state,
        )
        history.append(
            {
                "step_id": step_id,
                "action": action.to_dict(),
                "event": event.to_dict(),
                "feedback": feedback.to_dict(),
                "trace_events": trace_events,
                "state_before": before_state,
                "state_after": after_state,
            }
        )
        metric, best_reward_so_far = _capsule_step_metric(
            step_id=step_id,
            action=action,
            event=event,
            before_state=before_state,
            after_state=after_state,
            executed_regions=executed_regions,
            best_reward_so_far=best_reward_so_far,
            recovery_execution_attempt=consumes_recovery_side_effect,
        )
        step_metrics.append(metric)

        recovery_continued = False
        if event.status in {"failed", "invalid"}:
            recovery_step_id = len(history) + 1
            recovery_before_state = after_state
            recovery_action: RuntimeAction | None = None
            recovery_source_unit = None
            recovery_consumes_side_effect = False
            recovery_prompt = build_capsule_recovery_prompt(
                task=_task_text_from_obs(obs),
                failed_unit=group,
                history_tail=history,
                trace_summary=executor.trace.summary() if executor.trace is not None else {},
                side_effect_ledger={
                    "executed_side_effect_regions": sorted(executed_side_effect_regions),
                    "executed_side_effect_groups": sorted(executed_side_effect_groups),
                },
                recovery_observation_functions=recovery_observation_functions,
            )
            prompts.append(recovery_prompt)
            try:
                with llm_call_stage("capsule_recovery"):
                    response = _query_model(action_query_args, recovery_prompt)
                recovery_action = parse_runtime_action_response(response["content"])
                recovery_action.step_id = recovery_step_id
            except ValueError as exc:
                recovery_event = RuntimeEvent(
                    action="invalid",
                    status="invalid",
                    message=str(exc),
                    evidence={"exception_type": type(exc).__name__},
                )
            else:
                recovery_region_id = str(recovery_action.args.get("region_id", ""))
                recovery_group_id = str(recovery_action.args.get("group_id", ""))
                recovery_source_unit = region_by_id.get(recovery_region_id) or group_by_id.get(
                    recovery_group_id
                )
                recovery_event = _validate_recovery_action(
                    recovery_action,
                    failed_unit=group,
                    region_by_id=region_by_id,
                    group_by_id=group_by_id,
                )
                if recovery_event is None:
                    recovery_event = _no_rollback_guard_event(
                        recovery_action,
                        executed_side_effect_regions,
                        executed_side_effect_groups,
                        group_by_id,
                        recovery_observation_functions=recovery_observation_functions,
                    )
                if recovery_event is None:
                    recovery_event = _reward_drop_guard_event(
                        recovery_action,
                        recovery_before_state,
                        best_reward_so_far,
                        region_by_id,
                        group_by_id,
                        side_effect_calls,
                        recovery_side_effect_budget=recovery_side_effect_budget,
                        min_best_reward=reward_drop_guard_min_best_reward,
                        drop_threshold=reward_drop_guard_threshold,
                        recovery_observation_functions=recovery_observation_functions,
                    )
                if recovery_event is None:
                    recovery_consumes_side_effect = (
                        recovery_side_effect_budget > 0
                        and _runtime_action_targets_side_effect_unit(
                            recovery_action,
                            region_by_id,
                            group_by_id,
                            side_effect_calls,
                        )
                    )
                    recovery_event = _execute_runtime_action(
                        recovery_action,
                        executor,
                        source,
                        region_by_id,
                        group_by_id,
                        recovery_observation_functions=recovery_observation_functions,
                        append_recovery_insert_after_line=group.end_line,
                    )
                    if recovery_consumes_side_effect:
                        recovery_side_effect_budget -= 1
                if (
                    recovery_action.action == "append_recovery"
                    and recovery_event.status == "success"
                ):
                    recovery_side_effect_budget = 1
                _record_runtime_side_effect_execution(
                    recovery_action,
                    recovery_event,
                    region_by_id,
                    group_by_id,
                    executed_side_effect_regions,
                    executed_side_effect_groups,
                    side_effect_calls,
                )

            recovery_after_state = _capsule_state_snapshot(env)
            recovery_trace_events = recovery_event.evidence.get("trace_events", [])
            recovery_feedback_action = recovery_action or RuntimeAction("invalid", {})
            recovery_feedback = build_runtime_feedback(
                step_id=recovery_step_id,
                action=recovery_feedback_action,
                event=recovery_event,
                region=recovery_source_unit,
                trace_events=recovery_trace_events,
                before_state=recovery_before_state,
                after_state=recovery_after_state,
            )
            history.append(
                {
                    "step_id": recovery_step_id,
                    "action": (
                        recovery_action.to_dict()
                        if recovery_action is not None
                        else {"action": "invalid"}
                    ),
                    "event": recovery_event.to_dict(),
                    "feedback": recovery_feedback.to_dict(),
                    "trace_events": recovery_trace_events,
                    "state_before": recovery_before_state,
                    "state_after": recovery_after_state,
                }
            )
            recovery_metric, best_reward_so_far = _capsule_step_metric(
                step_id=recovery_step_id,
                action=recovery_action,
                event=recovery_event,
                before_state=recovery_before_state,
                after_state=recovery_after_state,
                executed_regions=executed_regions,
                best_reward_so_far=best_reward_so_far,
                recovery_execution_attempt=recovery_consumes_side_effect,
            )
            step_metrics.append(recovery_metric)
            if (
                recovery_action is not None
                and recovery_action.action in {"patch_region", "patch_group", "append_recovery"}
                and recovery_event.status == "success"
            ):
                recovery_action_name = recovery_action.action
                recovery_group_id = str(recovery_action.args.get("group_id", ""))
                recovery_region_id = str(recovery_action.args.get("region_id", ""))
                previous_source = source
                previous_regions = regions
                previous_groups = groups
                source = str(recovery_event.evidence["source"])
                regions = segment_python_code(source)
                groups = segment_python_code_groups(
                    source,
                    regions,
                    max_regions_per_group=max_regions_per_group,
                    side_effect_calls=side_effect_calls,
                )
                region_by_id = {region.region_id: region for region in regions}
                group_by_id = {group.group_id: group for group in groups}
                (
                    executed_side_effect_regions,
                    executed_side_effect_groups,
                ) = _remap_executed_side_effect_ledger(
                    recovery_action,
                    previous_source,
                    source,
                    previous_regions,
                    regions,
                    previous_groups,
                    groups,
                    executed_side_effect_regions,
                    executed_side_effect_groups,
                )
                if recovery_action_name == "append_recovery":
                    group_index = _first_group_index_starting_after_line(
                        groups,
                        group.end_line,
                        default=len(groups) - 1,
                    )
                elif recovery_action_name == "patch_group":
                    group_index = _group_index_by_id(groups, recovery_group_id, default=group_index)
                else:
                    group_index = _group_index_for_region(
                        groups,
                        recovery_region_id,
                        default=group_index,
                    )
                recovery_continued = True
            if recovery_event.status in {"failed", "invalid"}:
                failed = True
                break
            if recovery_event.action == "finish" or (
                recovery_action.action == "finish" if recovery_action is not None else False
            ):
                break
            if not recovery_continued:
                recovery_reward_after = _state_reward(recovery_after_state)
                failed = not (
                    bool(recovery_after_state.get("task_completed"))
                    or (recovery_reward_after is not None and recovery_reward_after >= 1.0)
                )
                break
            continue

        reward_after = _state_reward(after_state)
        if (
            event.status in {"failed", "invalid"}
            or bool(after_state.get("task_completed"))
            or (reward_after is not None and reward_after >= 1.0)
        ):
            break
        group_index += 1

    reward = _safe_compute_reward(env)
    task_completed = _safe_task_completed(env)
    task_succeeded = bool(task_completed) or reward >= 1.0
    sandbox_rc = 0 if task_succeeded and not failed else 1
    code_path = None
    if output_dir:
        code_path = os.path.join(output_dir, f"capsule_code_trial_{trial:02d}.py")
        with open(code_path, "w") as f:
            f.write(source)
        with open(os.path.join(output_dir, f"capsule_trace_trial_{trial:02d}.json"), "w") as f:
            json.dump(history, f, indent=2, default=str)
        with open(os.path.join(output_dir, f"capsule_prompts_trial_{trial:02d}.json"), "w") as f:
            json.dump(prompts, f, indent=2, default=str)
        metrics_path = os.path.join(output_dir, f"capsule_step_metrics_trial_{trial:02d}.jsonl")
        with open(metrics_path, "w") as f:
            for metric in step_metrics:
                f.write(json.dumps(metric, default=str) + "\n")

    if config.get("record_video"):
        info_step = {
            "sandbox_rc": sandbox_rc,
            "task_completed": bool(task_completed),
        }
        _save_trial_video(
            env,
            config,
            trial,
            info_step,
            reward,
            executed_regions,
            suffix_extra="capsule",
        )

    log = (
        f"Capsule actions: {len(history)}\n"
        f"Executed regions: {executed_regions}\n"
        f"Reward: {reward}\n"
        f"Task Completed: {task_completed}"
    )
    return TrialSummary(
        trial=trial,
        success=sandbox_rc == 0,
        reward=reward,
        terminated=bool(task_completed) or reward >= 1.0,
        truncated=False,
        sandbox_rc=sandbox_rc,
        log=log,
        task_completed=task_completed,
        code_path=code_path,
        num_regenerations=0,
        num_finishes=0,
        num_code_blocks=executed_regions,
    )


def _run_capsule_llm_step_loop(
    env: CodeExecutionEnvBase,
    trial: int,
    args: LaunchArgs,
    config: dict[str, Any],
    *,
    initial_code: str | None = None,
    scripted_actions: list[dict[str, Any]] | None = None,
    stop_after_failed_event: bool = False,
    stop_after_task_success: bool = False,
) -> TrialSummary:
    progress_mode = validate_progress_mode(
        str(config.get("capsule_progress_mode", "dense"))
    )
    prompt_state_level = _validate_capsule_state_level(
        str(config.get("capsule_prompt_state_level", "full")),
        config_name="capsule_prompt_state_level",
    )
    diagnostic_state_level = _validate_capsule_state_level(
        str(config.get("capsule_diagnostic_state_level", "none")),
        config_name="capsule_diagnostic_state_level",
        allow_none=True,
    )
    obs, _ = env.reset(options={"trial": trial}, seed=trial)
    output_dir = config.get("output_dir")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if config.get("record_video") and hasattr(env, "enable_video_capture"):
        env.enable_video_capture(
            True,
            clear=True,
            wrist_camera=config.get("use_wrist_camera", False),
        )

    raw_code: str
    if initial_code is not None:
        raw_code = initial_code
        source = initial_code
    elif config.get("use_oracle_code", False):
        raw_code = env.oracle_code
        source = env.oracle_code
    else:
        raw_code, _, _ = _query_initial_code(args, config, obs)
        source = "\n\n".join(_extract_code(raw_code))

    max_regions_per_group = int(config.get("capsule_max_regions_per_group", 20))
    use_semantic_groups = (
        config.get("capsule_execution_granularity", "semantic_group") == "semantic_group"
    )
    side_effect_calls = collect_side_effect_calls(getattr(env, "_apis", {}).values())
    if not side_effect_calls:
        side_effect_calls = ROBOT_SIDE_EFFECT_CALLS
    recovery_observation_functions = _collect_recovery_observation_functions(
        getattr(env, "_apis", {}).values()
    )
    initial_syntax_error: SyntaxError | None = None
    try:
        regions = segment_python_code(source)
        groups = (
            segment_python_code_groups(
                source,
                regions,
                max_regions_per_group=max_regions_per_group,
                side_effect_calls=side_effect_calls,
            )
            if use_semantic_groups
            else []
        )
    except SyntaxError as exc:
        initial_syntax_error = exc
        regions, fallback_groups = _whole_source_fallback_units(source)
        groups = fallback_groups if use_semantic_groups else []
    region_by_id = {region.region_id: region for region in regions}
    group_by_id = {group.group_id: group for group in groups}
    validate_program_contract = _coerce_config_bool(
        config.get("capsule_validate_program_contract", False)
    )
    program_contract_analysis = (
        analyze_capsule_program_contract_details(
            source,
            regions,
            groups,
            side_effect_calls=side_effect_calls,
        )
        if validate_program_contract and initial_syntax_error is None
        else None
    )
    if program_contract_analysis is not None:
        program_contract_violations = list(program_contract_analysis.violations)
        contract_effectful_region_ids = set(
            program_contract_analysis.effectful_region_ids
        )
        contract_effectful_group_ids = set(
            program_contract_analysis.effectful_group_ids
        )
    else:
        program_contract_violations = []
        contract_effectful_region_ids = set()
        contract_effectful_group_ids = set()
    trace = RuntimeTrace()
    executor = CapsuleExecutor(
        base_globals=env._build_capsule_globals(trace=trace),
        trace=trace,
    )
    max_steps = int(config.get("max_capsule_steps", 12))
    action_query_args = copy.copy(args)
    default_action_max_tokens = getattr(args, "max_tokens", 2048 * 10)
    action_query_args.max_tokens = int(
        config.get("capsule_action_max_tokens", default_action_max_tokens)
    )
    script = scripted_actions if scripted_actions is not None else config.get("scripted_actions")
    script_idx = 0
    history: list[dict[str, Any]] = []
    if initial_syntax_error is not None:
        history.append(_initial_syntax_error_history_entry(initial_syntax_error))
    step_metrics: list[dict[str, Any]] = []
    diagnostic_states: list[dict[str, Any]] = []
    prompts: list[list[dict[str, Any]]] = []
    failed = False
    finished = False
    loop_exit_reason: str | None = None
    executed_regions = 0
    best_reward_so_far: float | None = None
    executed_side_effect_regions: set[str] = set()
    executed_side_effect_groups: set[str] = set()
    recovery_side_effect_budget = 0
    pending_recovery_actions: list[RuntimeAction] = []
    reward_drop_guard_min_best_reward = float(
        config.get("capsule_reward_drop_guard_min_best_reward", 0.6)
    )
    reward_drop_guard_threshold = float(
        config.get("capsule_reward_drop_guard_threshold", 0.25)
    )
    llm_step_compact_context = _coerce_config_bool(
        config.get("capsule_llm_step_compact_context", True)
    )
    action_history_max_entries = int(config.get("capsule_action_history_max_entries", 4))
    action_trace_max_events = int(config.get("capsule_action_trace_max_events", 5))
    action_source_preview_chars = int(
        config.get("capsule_action_source_preview_chars", 240)
    )
    action_prompt_char_budget = int(
        config.get("capsule_action_prompt_char_budget", 60000)
    )
    require_task_success_for_finish = _coerce_config_bool(
        config.get("capsule_require_task_success_for_finish", False)
    )

    for step_id in range(1, max_steps + 1):
        action: RuntimeAction | None = None
        before_state = _capsule_state_snapshot(env, state_level=prompt_state_level)
        diagnostic_before_state = (
            _capsule_state_snapshot(env, state_level=diagnostic_state_level)
            if diagnostic_state_level != "none"
            else None
        )
        source_unit_for_feedback = None
        consumes_recovery_side_effect = False
        forced_recovery_action = False
        action_prompt_chars = 0
        action_prompt_char_budget_metric = None
        action_prompt_over_budget = False

        try:
            if pending_recovery_actions:
                forced_recovery_action = True
                action = pending_recovery_actions.pop(0)
            else:
                prompt = build_capsule_prompt(
                    task=_task_text_from_obs(obs),
                    regions=regions,
                    groups=groups if use_semantic_groups else None,
                    history=history,
                    trace_summary=executor.trace.summary() if executor.trace is not None else {},
                    contract_violations=[
                        item.to_dict() for item in program_contract_violations
                    ],
                    side_effect_ledger={
                        "executed_side_effect_groups": sorted(executed_side_effect_groups),
                        "executed_side_effect_regions": sorted(executed_side_effect_regions),
                    },
                    recovery_observation_functions=recovery_observation_functions,
                    compact_context=llm_step_compact_context,
                    history_max_entries=action_history_max_entries,
                    trace_max_events=action_trace_max_events,
                    source_preview_chars=action_source_preview_chars,
                    focused_source_max_units=1 if llm_step_compact_context else 0,
                    prompt_char_budget=(
                        action_prompt_char_budget if llm_step_compact_context else None
                    ),
                )
                action_prompt_chars = len(json.dumps(prompt, default=str))
                action_prompt_char_budget_metric = (
                    action_prompt_char_budget if llm_step_compact_context else None
                )
                action_prompt_over_budget = (
                    action_prompt_char_budget_metric is not None
                    and action_prompt_char_budget_metric > 0
                    and action_prompt_chars > action_prompt_char_budget_metric
                )
                prompts.append(prompt)

            if action is None and script is not None:
                if script_idx >= len(script):
                    action = RuntimeAction(action="finish", args={})
                else:
                    action = RuntimeAction.from_mapping(script[script_idx])
                    script_idx += 1
            elif action is None:
                with llm_call_stage("capsule_action"):
                    response = _query_model(action_query_args, prompt)
                action = parse_runtime_action_response(response["content"])
            action.step_id = step_id
        except ValueError as exc:
            event = RuntimeEvent(
                action="invalid",
                status="invalid",
                message=str(exc),
                evidence={"exception_type": type(exc).__name__},
            )
            failed = True
        else:
            region_id = str(action.args.get("region_id", ""))
            group_id = str(action.args.get("group_id", ""))
            source_unit_for_feedback = region_by_id.get(region_id) or group_by_id.get(group_id)
            event = _finish_success_guard_event(
                action,
                before_state,
                require_task_success=require_task_success_for_finish,
            )
            if event is None:
                event = (
                    _program_contract_guard_event(
                        action,
                        program_contract_violations,
                        contract_effectful_region_ids,
                        contract_effectful_group_ids,
                    )
                    if validate_program_contract
                    else None
                )
            if event is None:
                event = _no_rollback_guard_event(
                    action,
                    executed_side_effect_regions,
                    executed_side_effect_groups,
                    group_by_id,
                    recovery_observation_functions=recovery_observation_functions,
                )
            if event is None:
                event = _reward_drop_guard_event(
                    action,
                    before_state,
                    best_reward_so_far,
                    region_by_id,
                    group_by_id,
                    side_effect_calls,
                    recovery_side_effect_budget=recovery_side_effect_budget,
                    min_best_reward=reward_drop_guard_min_best_reward,
                    drop_threshold=reward_drop_guard_threshold,
                    recovery_observation_functions=recovery_observation_functions,
                )
            if event is None:
                consumes_recovery_side_effect = (
                    recovery_side_effect_budget > 0
                    and _runtime_action_targets_side_effect_unit(
                        action,
                        region_by_id,
                        group_by_id,
                        side_effect_calls,
                    )
                )
                event = _execute_runtime_action(
                    action,
                    executor,
                    source,
                    region_by_id,
                    group_by_id,
                    recovery_observation_functions=recovery_observation_functions,
                )
                if consumes_recovery_side_effect:
                    recovery_side_effect_budget -= 1
            if event.status in {"failed", "invalid"}:
                failed = True
                if forced_recovery_action:
                    pending_recovery_actions.clear()
            if (
                action.action in {"run_region", "resume_from_region"}
                and event.status != "invalid"
            ):
                executed_regions += 1
            elif action.action == "run_group":
                group = group_by_id.get(group_id)
                if event.status != "invalid":
                    executed_regions += len(group.region_ids) if group is not None else 0
            elif action.action == "append_recovery" and event.status == "success":
                recovery_side_effect_budget = 1
            _record_runtime_side_effect_execution(
                action,
                event,
                region_by_id,
                group_by_id,
                executed_side_effect_regions,
                executed_side_effect_groups,
                side_effect_calls,
            )

        after_state = _capsule_state_snapshot(env, state_level=prompt_state_level)
        if diagnostic_before_state is not None:
            diagnostic_states.append(
                {
                    "step_id": step_id,
                    "state_before": diagnostic_before_state,
                    "state_after": _capsule_state_snapshot(
                        env, state_level=diagnostic_state_level
                    ),
                }
            )
        trace_events = event.evidence.get("trace_events", [])
        feedback_action = action if action is not None else RuntimeAction(action="invalid", args={})
        feedback = build_runtime_feedback(
            step_id=step_id,
            action=feedback_action,
            event=event,
            region=source_unit_for_feedback,
            trace_events=trace_events,
            before_state=before_state,
            after_state=after_state,
            progress_mode=progress_mode,
            side_effect_calls=side_effect_calls,
        )

        history.append(
            {
                "step_id": step_id,
                "action": action.to_dict() if action is not None else {"action": "invalid"},
                "event": event.to_dict(),
                "feedback": feedback.to_dict(),
                "trace_events": trace_events,
                "state_before": before_state,
                "state_after": after_state,
            }
        )
        metric, best_reward_so_far = _capsule_step_metric(
            step_id=step_id,
            action=action,
            event=event,
            before_state=before_state,
            after_state=after_state,
            executed_regions=executed_regions,
            best_reward_so_far=best_reward_so_far,
            recovery_execution_attempt=consumes_recovery_side_effect,
        )
        metric["action_prompt_chars"] = action_prompt_chars
        metric["action_prompt_compact_context"] = llm_step_compact_context
        metric["action_prompt_char_budget"] = action_prompt_char_budget_metric
        metric["action_prompt_over_budget"] = action_prompt_over_budget
        metric["program_contract_valid"] = not program_contract_violations
        metric["program_contract_violation_count"] = len(
            program_contract_violations
        )
        metric["program_contract_violation_codes"] = sorted(
            {item.code for item in program_contract_violations}
        )
        metric["budget_exhausted"] = False
        step_metrics.append(metric)

        if (
            action is not None
            and action.action in {"patch_region", "patch_group", "append_recovery"}
            and event.status == "success"
        ):
            previous_source = source
            previous_source_line_count = len(source.splitlines())
            previous_regions = regions
            previous_groups = groups
            source = str(event.evidence["source"])
            regions = segment_python_code(source)
            groups = (
                segment_python_code_groups(
                    source,
                    regions,
                    max_regions_per_group=max_regions_per_group,
                    side_effect_calls=side_effect_calls,
                )
                if use_semantic_groups
                else []
            )
            region_by_id = {region.region_id: region for region in regions}
            group_by_id = {group.group_id: group for group in groups}
            (
                executed_side_effect_regions,
                executed_side_effect_groups,
            ) = _remap_executed_side_effect_ledger(
                action,
                previous_source,
                source,
                previous_regions,
                regions,
                previous_groups,
                groups,
                executed_side_effect_regions,
                executed_side_effect_groups,
            )
            program_contract_analysis = (
                analyze_capsule_program_contract_details(
                    source,
                    regions,
                    groups,
                    side_effect_calls=side_effect_calls,
                )
                if validate_program_contract
                else None
            )
            if program_contract_analysis is not None:
                program_contract_violations = list(
                    program_contract_analysis.violations
                )
                contract_effectful_region_ids = set(
                    program_contract_analysis.effectful_region_ids
                )
                contract_effectful_group_ids = set(
                    program_contract_analysis.effectful_group_ids
                )
            else:
                program_contract_violations = []
                contract_effectful_region_ids = set()
                contract_effectful_group_ids = set()
            if action.action == "append_recovery":
                pending_recovery_actions = _runtime_actions_for_appended_recovery(
                    regions,
                    groups,
                    use_semantic_groups=use_semantic_groups,
                    insert_after_line=previous_source_line_count,
                )
                recovery_side_effect_budget = max(
                    recovery_side_effect_budget,
                    _count_side_effect_runtime_actions(
                        pending_recovery_actions,
                        region_by_id,
                        group_by_id,
                        side_effect_calls,
                    ),
                )

        if stop_after_failed_event and event.status in {"failed", "invalid"}:
            loop_exit_reason = "failed_event"
            break
        reward_after = _state_reward(after_state)
        if (
            action is not None
            and action.action == "finish"
            and event.status == "success"
        ):
            finished = True
            loop_exit_reason = "accepted_finish"
            break
        if (stop_after_task_success or require_task_success_for_finish) and (
            bool(after_state.get("task_completed"))
            or (reward_after is not None and reward_after >= 1.0)
        ):
            loop_exit_reason = "task_success"
            break

    reward = _safe_compute_reward(env)
    task_completed = _safe_task_completed(env)
    task_succeeded = bool(task_completed) or reward >= 1.0
    budget_exhausted = (
        loop_exit_reason is None and not finished and not task_succeeded
    )
    if budget_exhausted and step_metrics:
        step_metrics[-1]["budget_exhausted"] = True
    sandbox_rc = 0 if task_succeeded and not failed else 1
    code_path = None
    if output_dir:
        code_path = os.path.join(output_dir, f"capsule_code_trial_{trial:02d}.py")
        with open(code_path, "w") as f:
            f.write(source)
        with open(os.path.join(output_dir, f"capsule_trace_trial_{trial:02d}.json"), "w") as f:
            json.dump(history, f, indent=2, default=str)
        with open(os.path.join(output_dir, f"capsule_prompts_trial_{trial:02d}.json"), "w") as f:
            json.dump(prompts, f, indent=2, default=str)
        metrics_path = os.path.join(output_dir, f"capsule_step_metrics_trial_{trial:02d}.jsonl")
        with open(metrics_path, "w") as f:
            for metric in step_metrics:
                f.write(json.dumps(metric, default=str) + "\n")
        if diagnostic_state_level != "none":
            diagnostics_path = os.path.join(
                output_dir, f"capsule_diagnostics_trial_{trial:02d}.jsonl"
            )
            with open(diagnostics_path, "w") as f:
                for diagnostic_state in diagnostic_states:
                    f.write(json.dumps(diagnostic_state, default=str) + "\n")

    if config.get("record_video"):
        info_step = {
            "sandbox_rc": sandbox_rc,
            "task_completed": bool(task_completed),
        }
        _save_trial_video(
            env,
            config,
            trial,
            info_step,
            reward,
            executed_regions,
            suffix_extra="capsule",
        )

    log = (
        f"Capsule actions: {len(history)}\n"
        f"Executed regions: {executed_regions}\n"
        f"Reward: {reward}\n"
        f"Task Completed: {task_completed}\n"
        f"Budget Exhausted: {budget_exhausted}"
    )
    return TrialSummary(
        trial=trial,
        success=sandbox_rc == 0,
        reward=reward,
        terminated=task_succeeded,
        truncated=budget_exhausted,
        sandbox_rc=sandbox_rc,
        log=log,
        task_completed=task_completed,
        code_path=code_path,
        num_regenerations=0,
        num_finishes=1 if finished else 0,
        num_code_blocks=executed_regions,
    )


def _finish_success_guard_event(
    action: RuntimeAction,
    state: dict[str, Any],
    *,
    require_task_success: bool,
) -> RuntimeEvent | None:
    if action.action != "finish" or not require_task_success:
        return None
    reward = _state_reward(state)
    if state.get("task_completed") is True or (reward is not None and reward >= 1.0):
        return None
    return RuntimeEvent(
        action="finish",
        status="warning",
        message=(
            "Finish rejected because the environment success predicate is not satisfied."
        ),
        evidence={
            "task_completed": state.get("task_completed"),
            "reward": reward,
        },
    )


def _validate_patched_source(
    *, action: str, region_id: str, source: str
) -> RuntimeEvent | None:
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return RuntimeEvent(
            action=action,
            status="invalid",
            region_id=region_id,
            message=f"Patched source is invalid Python: {exc}",
            evidence={
                "exception_type": type(exc).__name__,
                "lineno": exc.lineno,
                "offset": exc.offset,
                "text": exc.text,
            },
        )
    return None


def _coerce_terminal_append_recovery_action(
    content: str,
    *,
    recovery_observation_functions: set[str] | None,
) -> RuntimeAction | None:
    source = _terminal_python_payload(content)
    if source is None:
        return None
    try:
        recovery_tree = ast.parse(source)
    except SyntaxError:
        return None

    fresh_state_functions = (
        {"get_observation"}
        if recovery_observation_functions is None
        else recovery_observation_functions
    )
    if not fresh_state_functions:
        return None
    if not any(_ast_calls_function(recovery_tree, name) for name in fresh_state_functions):
        return None
    return RuntimeAction("append_recovery", {"source": source})


def _terminal_python_payload(content: str) -> str | None:
    text = content.strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = text.splitlines()
        language = lines[0].strip("`").strip().lower()
        if language not in {"", "python", "py"}:
            return None
        if len(lines) < 3 or not lines[-1].strip().startswith("```"):
            return None
        text = "\n".join(lines[1:-1]).strip()
    return text or None


def _execute_runtime_action(
    action: RuntimeAction,
    executor: CapsuleExecutor,
    source: str,
    region_by_id: dict[str, Any],
    group_by_id: dict[str, Any] | None = None,
    recovery_observation_functions: set[str] | None = None,
    append_recovery_insert_after_line: int | None = None,
) -> RuntimeEvent:
    group_by_id = group_by_id or {}
    if action.action == "finish":
        return RuntimeEvent(action="finish", status="success")

    if action.action == "run_region":
        region_id = str(action.args.get("region_id", ""))
        region = region_by_id.get(region_id)
        if region is None:
            return RuntimeEvent(
                action=action.action,
                status="invalid",
                region_id=region_id,
                message=f"Unknown region: {region_id}",
            )
        return executor.run_region(region)

    if action.action == "run_group":
        group_id = str(action.args.get("group_id", ""))
        group = group_by_id.get(group_id)
        if group is None:
            return RuntimeEvent(
                action=action.action,
                status="invalid",
                region_id=group_id,
                message=f"Unknown group: {group_id}",
            )
        event = executor.run_region(group)
        event.action = "run_group"
        return event

    if action.action == "inspect_trace":
        last_n = _runtime_trace_last_n(action.args.get("last_n", 8))
        failed_only = bool(action.args.get("failed_only", False))
        return RuntimeEvent(
            action=action.action,
            status="success",
            evidence=(
                executor.trace.summary(max_events=last_n, failed_only=failed_only)
                if executor.trace is not None
                else {
                    "event_count": 0,
                    "primitive_call_counts": {},
                    "failed_event_count": 0,
                    "recent_events": [],
                    "failed_events": [],
                }
            ),
        )

    if action.action == "inspect_variables":
        names = action.args.get("names", [])
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(name, str) for name in names)
        ):
            return RuntimeEvent(
                action=action.action,
                status="invalid",
                message="inspect_variables requires args.names as a non-empty list of strings",
            )
        evidence = {
            str(name): _summarize_runtime_value(executor.globals.get(str(name)))
            for name in names
        }
        return RuntimeEvent(action=action.action, status="success", evidence=evidence)

    if action.action == "patch_region":
        region_id = str(action.args.get("region_id", ""))
        replacement = _runtime_patch_replacement(action.args)
        region = region_by_id.get(region_id)
        if region is None or not isinstance(replacement, str):
            return RuntimeEvent(
                action=action.action,
                status="invalid",
                region_id=region_id,
                message=(
                    "patch_region requires args.region_id and args.source as strings; "
                    "new_source and patch are accepted as compatibility aliases"
                ),
            )
        patched = replace_region_source(source, region, replacement)
        syntax_error_event = _validate_patched_source(
            action=action.action,
            region_id=region_id,
            source=patched,
        )
        if syntax_error_event is not None:
            return syntax_error_event
        return RuntimeEvent(
            action=action.action,
            status="success",
            region_id=region_id,
            evidence={"source": patched},
        )

    if action.action == "patch_group":
        group_id = str(action.args.get("group_id", ""))
        replacement = _runtime_patch_replacement(action.args)
        group = group_by_id.get(group_id)
        if group is None or not isinstance(replacement, str):
            return RuntimeEvent(
                action=action.action,
                status="invalid",
                region_id=group_id,
                message=(
                    "patch_group requires args.group_id and args.source as strings; "
                    "new_source and patch are accepted as compatibility aliases"
                ),
            )
        patched = replace_region_source(source, group, replacement)
        syntax_error_event = _validate_patched_source(
            action=action.action,
            region_id=group_id,
            source=patched,
        )
        if syntax_error_event is not None:
            return syntax_error_event
        return RuntimeEvent(
            action=action.action,
            status="success",
            region_id=group_id,
            evidence={"source": patched},
        )

    if action.action == "append_recovery":
        recovery_source = action.args.get("source")
        if not isinstance(recovery_source, str) or not recovery_source.strip():
            return RuntimeEvent(
                action=action.action,
                status="invalid",
                message="append_recovery requires args.source as a non-empty string.",
            )
        try:
            recovery_tree = ast.parse(recovery_source)
        except SyntaxError as exc:
            return RuntimeEvent(
                action=action.action,
                status="invalid",
                message=f"append_recovery source must parse as Python: {exc.msg}",
                evidence={"exception_type": type(exc).__name__},
            )
        fresh_state_functions = (
            {"get_observation"}
            if recovery_observation_functions is None
            else recovery_observation_functions
        )
        if not fresh_state_functions:
            return RuntimeEvent(
                action=action.action,
                status="invalid",
                message=(
                    "append_recovery is unavailable because the active API does not declare "
                    "a fresh-state observation function."
                ),
            )
        if not any(_ast_calls_function(recovery_tree, name) for name in fresh_state_functions):
            allowed_calls = ", ".join(f"{name}()" for name in sorted(fresh_state_functions))
            return RuntimeEvent(
                action=action.action,
                status="invalid",
                message=f"append_recovery source must call at least one of: {allowed_calls}.",
            )
        patched = _append_recovery_source(
            source,
            recovery_source,
            insert_after_line=append_recovery_insert_after_line,
        )
        return RuntimeEvent(
            action=action.action,
            status="success",
            evidence={"source": patched},
        )

    if action.action == "rollback_to_checkpoint":
        return RuntimeEvent(
            action=action.action,
            status="invalid",
            message=(
                "Rollback is disabled. Continue from the current physical state with "
                "fresh observation and recovery code."
            ),
        )

    if action.action == "resume_from_region":
        region_id = str(action.args.get("region_id", ""))
        region = region_by_id.get(region_id)
        if region is None:
            return RuntimeEvent(
                action=action.action,
                status="invalid",
                region_id=region_id,
                message=f"Unknown region: {region_id}",
            )
        return executor.run_region(region)

    return RuntimeEvent(action=action.action, status="invalid", message="Unsupported runtime action")


def _no_rollback_guard_event(
    action: RuntimeAction,
    executed_side_effect_regions: set[str],
    executed_side_effect_groups: set[str],
    group_by_id: dict[str, CodeRegionGroup] | None = None,
    *,
    recovery_observation_functions: set[str] | None = None,
) -> RuntimeEvent | None:
    group_by_id = group_by_id or {}
    recovery_hint = _recovery_instruction(recovery_observation_functions)
    if action.action in {"run_group", "patch_group"}:
        group_id = str(action.args.get("group_id", ""))
        group = group_by_id.get(group_id)
        contains_executed_region = bool(
            group is not None
            and executed_side_effect_regions.intersection(group.region_ids)
        )
        if group_id in executed_side_effect_groups or contains_executed_region:
            return _already_executed_side_effect_event(action.action, group_id, recovery_hint)
        return None

    if action.action in {"run_region", "patch_region", "resume_from_region"}:
        region_id = str(action.args.get("region_id", ""))
        if region_id in executed_side_effect_regions:
            return _already_executed_side_effect_event(action.action, region_id, recovery_hint)
        return None

    return None


def _reward_drop_guard_event(
    action: RuntimeAction,
    before_state: dict[str, Any],
    best_reward_so_far: float | None,
    region_by_id: dict[str, Any],
    group_by_id: dict[str, Any],
    side_effect_calls: set[str] | None = None,
    *,
    recovery_side_effect_budget: int,
    min_best_reward: float,
    drop_threshold: float,
    recovery_observation_functions: set[str] | None = None,
) -> RuntimeEvent | None:
    if recovery_side_effect_budget > 0:
        return None
    if side_effect_calls is None:
        side_effect_calls = ROBOT_SIDE_EFFECT_CALLS
    if not _runtime_action_targets_side_effect_unit(
        action,
        region_by_id,
        group_by_id,
        side_effect_calls,
    ):
        return None

    current_reward = _state_reward(before_state)
    if best_reward_so_far is None or current_reward is None:
        return None

    reward_drop = best_reward_so_far - current_reward
    if best_reward_so_far < min_best_reward or reward_drop < drop_threshold:
        return None

    unit_id = _runtime_action_unit_id(action)
    recovery_hint = _recovery_instruction(recovery_observation_functions)
    return RuntimeEvent(
        action=action.action,
        status="invalid",
        region_id=unit_id,
        message=(
            f"Current reward dropped from best_reward={best_reward_so_far:.3f} "
            f"to {current_reward:.3f}. Rollback is disabled. {recovery_hint} "
            "Finish instead if the task is complete."
        ),
        evidence={
            "best_reward_so_far": best_reward_so_far,
            "current_reward": current_reward,
            "reward_drop_from_best": reward_drop,
            "drop_threshold": drop_threshold,
            "min_best_reward": min_best_reward,
        },
    )


def _program_contract_guard_event(
    action: RuntimeAction,
    violations: list[ProgramContractViolation],
    effectful_region_ids: set[str],
    effectful_group_ids: set[str],
) -> RuntimeEvent | None:
    if not violations:
        return None
    if action.action == "run_group":
        target_is_effectful = str(action.args.get("group_id", "")) in effectful_group_ids
    elif action.action in {"run_region", "resume_from_region"}:
        target_is_effectful = (
            str(action.args.get("region_id", "")) in effectful_region_ids
        )
    else:
        target_is_effectful = False
    if not target_is_effectful:
        return None
    return RuntimeEvent(
        action=action.action,
        status="invalid",
        region_id=str(
            action.args.get("group_id") or action.args.get("region_id") or ""
        )
        or None,
        message=(
            "Robot side-effect execution is blocked until the Capsule-ready "
            "program contract is repaired."
        ),
        evidence={
            "program_contract_violations": [item.to_dict() for item in violations]
        },
    )


def _runtime_action_targets_side_effect_unit(
    action: RuntimeAction,
    region_by_id: dict[str, Any],
    group_by_id: dict[str, Any],
    side_effect_calls: set[str] | None = None,
) -> bool:
    if side_effect_calls is None:
        side_effect_calls = ROBOT_SIDE_EFFECT_CALLS
    if action.action == "run_group":
        group = group_by_id.get(str(action.args.get("group_id", "")))
        return bool(group is not None and group.has_robot_side_effect)

    if action.action in {"run_region", "resume_from_region"}:
        region = region_by_id.get(str(action.args.get("region_id", "")))
        return bool(
            region is not None
            and _region_has_side_effect_call(region, side_effect_calls)
        )

    return False


def _runtime_action_unit_id(action: RuntimeAction) -> str | None:
    unit_id = action.args.get("group_id") or action.args.get("region_id")
    return str(unit_id) if unit_id else None


RECOVERY_ACTIONS: set[str] = {
    "inspect_trace",
    "inspect_variables",
    "patch_group",
    "patch_region",
    "append_recovery",
    "resume_from_region",
    "finish",
}


def _validate_recovery_action(
    action: RuntimeAction,
    *,
    failed_unit: CodeRegion | CodeRegionGroup,
    region_by_id: dict[str, Any],
    group_by_id: dict[str, Any],
) -> RuntimeEvent | None:
    if action.action not in RECOVERY_ACTIONS:
        return RuntimeEvent(
            action=action.action,
            status="invalid",
            region_id=_runtime_action_unit_id(action),
            message=f"{action.action} is not allowed during recovery.",
        )

    if action.action in {"patch_group"}:
        group_id = str(action.args.get("group_id", ""))
        if not isinstance(failed_unit, CodeRegionGroup) or group_id != failed_unit.group_id:
            return RuntimeEvent(
                action=action.action,
                status="invalid",
                region_id=group_id,
                message="Recovery patch_group must target the failed group.",
            )
        if group_id not in group_by_id:
            return None

    if action.action in {"patch_region", "resume_from_region"}:
        region_id = str(action.args.get("region_id", ""))
        if not _recovery_region_is_local(failed_unit, region_id):
            return RuntimeEvent(
                action=action.action,
                status="invalid",
                region_id=region_id,
                message=(
                    "Recovery patch_region/resume_from_region must target the failed "
                    "region or a member region of the failed group."
                ),
            )
        if region_id not in region_by_id:
            return None

    return None


def _recovery_region_is_local(
    failed_unit: CodeRegion | CodeRegionGroup,
    region_id: str,
) -> bool:
    if isinstance(failed_unit, CodeRegionGroup):
        return region_id in failed_unit.region_ids
    return region_id == failed_unit.region_id


def _record_runtime_side_effect_execution(
    action: RuntimeAction,
    event: RuntimeEvent,
    region_by_id: dict[str, CodeRegion],
    group_by_id: dict[str, CodeRegionGroup],
    executed_side_effect_regions: set[str],
    executed_side_effect_groups: set[str],
    side_effect_calls: set[str],
) -> None:
    if event.status == "invalid" or not _event_has_side_effect_trace(
        event, side_effect_calls
    ):
        return

    if action.action in {"run_region", "resume_from_region"}:
        region_id = str(action.args.get("region_id", ""))
        if region_id not in region_by_id:
            return
        executed_side_effect_regions.add(region_id)
        executed_side_effect_groups.update(
            group.group_id
            for group in group_by_id.values()
            if region_id in group.region_ids
        )
        return

    if action.action == "run_group":
        group_id = str(action.args.get("group_id", ""))
        group = group_by_id.get(group_id)
        if group is None:
            return
        executed_side_effect_groups.add(group_id)
        executed_side_effect_regions.update(group.region_ids)
        executed_side_effect_groups.update(
            candidate.group_id
            for candidate in group_by_id.values()
            if executed_side_effect_regions.intersection(candidate.region_ids)
        )


def _event_has_side_effect_trace(event: RuntimeEvent, side_effect_calls: set[str]) -> bool:
    trace_events = event.evidence.get("trace_events", [])
    if not isinstance(trace_events, list):
        return False
    return any(trace_event.get("name") in side_effect_calls for trace_event in trace_events)


def _region_has_side_effect_call(region: CodeRegion, side_effect_calls: set[str]) -> bool:
    try:
        tree = ast.parse(region.source)
    except SyntaxError:
        return False
    return any(_ast_calls_function(tree, name) for name in side_effect_calls)


def _group_index_by_id(
    groups: list[CodeRegionGroup],
    group_id: str,
    *,
    default: int,
) -> int:
    for idx, group in enumerate(groups):
        if group.group_id == group_id:
            return idx
    return default


def _next_group_index_after_group(groups: list[CodeRegionGroup], group_id: str) -> int:
    return _group_index_by_id(groups, group_id, default=len(groups) - 1) + 1


def _first_group_index_starting_after_line(
    groups: list[CodeRegionGroup],
    line_number: int,
    *,
    default: int,
) -> int:
    for idx, group in enumerate(groups):
        if group.start_line > line_number:
            return idx
    return default


def _runtime_actions_for_appended_recovery(
    regions: list[CodeRegion],
    groups: list[CodeRegionGroup],
    *,
    use_semantic_groups: bool,
    insert_after_line: int,
) -> list[RuntimeAction]:
    if use_semantic_groups:
        return [
            RuntimeAction("run_group", {"group_id": group.group_id})
            for group in groups
            if group.start_line > insert_after_line
        ]
    return [
        RuntimeAction("run_region", {"region_id": region.region_id})
        for region in regions
        if region.start_line > insert_after_line
    ]


def _count_side_effect_runtime_actions(
    actions: list[RuntimeAction],
    region_by_id: dict[str, Any],
    group_by_id: dict[str, Any],
    side_effect_calls: set[str],
) -> int:
    return sum(
        1
        for action in actions
        if _runtime_action_targets_side_effect_unit(
            action,
            region_by_id,
            group_by_id,
            side_effect_calls,
        )
    )


def _group_index_for_region(
    groups: list[CodeRegionGroup],
    region_id: str,
    *,
    default: int,
) -> int:
    for idx, group in enumerate(groups):
        if region_id in group.region_ids:
            return idx
    return default


def _already_executed_side_effect_event(
    action_name: str, unit_id: str, recovery_hint: str
) -> RuntimeEvent:
    return RuntimeEvent(
        action=action_name,
        status="invalid",
        region_id=unit_id,
        message=(
            f"{unit_id} already executed robot-side-effect code and rollback is disabled. "
            f"{recovery_hint}"
        ),
    )


def _remap_executed_side_effect_ledger(
    action: RuntimeAction,
    previous_source: str,
    current_source: str,
    previous_regions: list[CodeRegion],
    current_regions: list[CodeRegion],
    previous_groups: list[CodeRegionGroup],
    current_groups: list[CodeRegionGroup],
    executed_region_ids: set[str],
    executed_group_ids: set[str],
) -> tuple[set[str], set[str]]:
    edit_span = _runtime_source_edit_span(
        action,
        previous_source,
        current_source,
        previous_regions,
        previous_groups,
    )
    if edit_span is None:
        return (
            {region.region_id for region in current_regions},
            {group.group_id for group in current_groups},
        )
    edit_start_line, edit_end_line, line_delta = edit_span
    previous_region_by_id = {
        region.region_id: region for region in previous_regions
    }
    remapped_region_ids: set[str] = set()
    lineage_unresolved = False
    for region_id in executed_region_ids:
        previous_region = previous_region_by_id.get(region_id)
        if previous_region is None:
            lineage_unresolved = True
            continue
        matches = _lineage_matches_after_edit(
            previous_region,
            current_regions,
            edit_start_line=edit_start_line,
            edit_end_line=edit_end_line,
            line_delta=line_delta,
            id_attribute="region_id",
        )
        if not matches:
            lineage_unresolved = True
        remapped_region_ids.update(matches)

    anchored_group_ids = {
        group.group_id
        for group in previous_groups
        if executed_region_ids.intersection(group.region_ids)
    }
    previous_group_by_id = {group.group_id: group for group in previous_groups}
    remapped_group_ids: set[str] = set()
    for group_id in executed_group_ids - anchored_group_ids:
        previous_group = previous_group_by_id.get(group_id)
        if previous_group is None:
            lineage_unresolved = True
            continue
        matches = _lineage_matches_after_edit(
            previous_group,
            current_groups,
            edit_start_line=edit_start_line,
            edit_end_line=edit_end_line,
            line_delta=line_delta,
            id_attribute="group_id",
        )
        if not matches:
            lineage_unresolved = True
        remapped_group_ids.update(matches)

    if lineage_unresolved:
        remapped_region_ids.update(region.region_id for region in current_regions)
        remapped_group_ids.update(group.group_id for group in current_groups)
    remapped_group_ids.update(
        group.group_id
        for group in current_groups
        if remapped_region_ids.intersection(group.region_ids)
    )
    return remapped_region_ids, remapped_group_ids


def _runtime_source_edit_span(
    action: RuntimeAction,
    previous_source: str,
    current_source: str,
    previous_regions: list[CodeRegion],
    previous_groups: list[CodeRegionGroup],
) -> tuple[int, int, int] | None:
    line_delta = len(current_source.splitlines()) - len(previous_source.splitlines())
    if action.action == "patch_region":
        target_id = str(action.args.get("region_id", ""))
        target = next(
            (region for region in previous_regions if region.region_id == target_id),
            None,
        )
        if target is None:
            return None
        return target.start_line, target.end_line, line_delta
    if action.action == "patch_group":
        target_id = str(action.args.get("group_id", ""))
        target = next(
            (group for group in previous_groups if group.group_id == target_id),
            None,
        )
        if target is None:
            return None
        return target.start_line, target.end_line, line_delta
    if action.action == "append_recovery":
        insertion_line = len(previous_source.splitlines()) + 1
        return insertion_line, insertion_line - 1, line_delta
    return None


def _lineage_matches_after_edit(
    previous_unit: CodeRegion | CodeRegionGroup,
    current_units: list[CodeRegion] | list[CodeRegionGroup],
    *,
    edit_start_line: int,
    edit_end_line: int,
    line_delta: int,
    id_attribute: str,
) -> set[str]:
    if previous_unit.end_line < edit_start_line:
        expected_start = previous_unit.start_line
        expected_end = previous_unit.end_line
    elif previous_unit.start_line > edit_end_line:
        expected_start = previous_unit.start_line + line_delta
        expected_end = previous_unit.end_line + line_delta
    else:
        return set()

    exact_source_candidates = [
        unit
        for unit in current_units
        if unit.source.rstrip("\n") == previous_unit.source.rstrip("\n")
    ]
    expected_candidates = [
        unit
        for unit in exact_source_candidates
        if unit.start_line == expected_start and unit.end_line == expected_end
    ]
    matches = (
        expected_candidates
        if len(expected_candidates) == 1
        else exact_source_candidates
    )
    return {str(getattr(unit, id_attribute)) for unit in matches}


def _runtime_patch_replacement(args: dict[str, Any]) -> Any:
    for key in ("source", "new_source", "patch"):
        replacement = args.get(key)
        if isinstance(replacement, str):
            return replacement
    return None


def _runtime_trace_last_n(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 8
    return max(0, min(parsed, 50))


def _coerce_config_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    return bool(value)


def _append_recovery_source(
    source: str,
    recovery_source: str,
    *,
    insert_after_line: int | None = None,
) -> str:
    base = source.rstrip("\n")
    recovery = recovery_source.strip("\n")
    if insert_after_line is not None:
        return _insert_recovery_source_after_line(source, recovery, insert_after_line)
    if not base:
        return f"{recovery}\n"
    return f"{base}\n\n{recovery}\n"


def _insert_recovery_source_after_line(
    source: str,
    recovery_source: str,
    insert_after_line: int,
) -> str:
    lines = source.splitlines()
    insert_at = max(0, min(insert_after_line, len(lines)))
    barrier_name = "__capsule_recovery_barrier"
    # The duplicate definition prevents bounded grouping from merging recovery with future effects.
    recovery_block = [
        f"{barrier_name} = None",
        *recovery_source.splitlines(),
        f"{barrier_name} = {barrier_name}",
    ]
    patched_lines = [*lines[:insert_at], *recovery_block, *lines[insert_at:]]
    return "\n".join(patched_lines) + "\n"


def _ast_calls_function(tree: ast.AST, function_name: str) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == function_name:
            return True
        if isinstance(func, ast.Attribute) and func.attr == function_name:
            return True
    return False


def _recovery_instruction(recovery_observation_functions: set[str] | None) -> str:
    functions = (
        {"get_observation"}
        if recovery_observation_functions is None
        else recovery_observation_functions
    )
    if not functions:
        return "append_recovery is unavailable because no fresh-state function is declared."
    calls = ", ".join(f"{name}()" for name in sorted(functions))
    return (
        f"Use append_recovery that calls at least one fresh-state function ({calls}) "
        "before executing more robot-side-effect code from the current physical state."
    )


def _collect_recovery_observation_functions(apis: Any) -> set[str]:
    functions: set[str] = set()
    for api in apis:
        public_functions = set(api.functions())
        capability = getattr(api, "recovery_observation_functions", None)
        declared = set(capability()) if callable(capability) else {"get_observation"}
        functions.update(declared & public_functions)
    return functions


def _region_has_robot_side_effect(region: Any) -> bool:
    try:
        tree = ast.parse(region.source)
    except SyntaxError:
        return False
    return any(_ast_calls_function(tree, call_name) for call_name in ROBOT_SIDE_EFFECT_CALLS)


def _safe_compute_reward(env: CodeExecutionEnvBase) -> float:
    try:
        return float(env.compute_reward())
    except Exception:
        return 0.0


def _validate_capsule_state_level(
    state_level: str,
    *,
    config_name: str = "state_level",
    allow_none: bool = False,
) -> str:
    allowed = ("none", "full", "proprioceptive") if allow_none else (
        "full",
        "proprioceptive",
    )
    if state_level not in allowed:
        raise ValueError(
            f"{config_name} must use one of the allowed values: {', '.join(allowed)}"
        )
    return state_level


def _capsule_state_snapshot(
    env: CodeExecutionEnvBase,
    *,
    state_level: str = "full",
) -> dict[str, Any]:
    _validate_capsule_state_level(state_level)
    snapshot: dict[str, Any] = {
        "reward": _safe_compute_reward(env),
        "task_completed": _safe_task_completed(env),
    }
    low_level_env = getattr(env, "low_level_env", None)
    if low_level_env is None:
        return snapshot

    for attr in ("_step_count", "_sim_step_count", "_gripper_fraction"):
        if hasattr(low_level_env, attr):
            snapshot[attr.removeprefix("_")] = _jsonable_metric_value(getattr(low_level_env, attr))

    gripper_pose = getattr(low_level_env, "gripper_link_wxyz_xyz", None)
    if gripper_pose is not None:
        snapshot["gripper_wxyz_xyz"] = _jsonable_metric_value(gripper_pose)

    current_joints = getattr(low_level_env, "_current_joints", None)
    if current_joints is not None:
        snapshot["robot_joint_pos"] = _jsonable_metric_value(current_joints)

    if state_level == "full":
        object_poses = _capsule_object_pose_snapshot(low_level_env)
        if object_poses:
            snapshot["object_poses"] = object_poses

    return snapshot


def _capsule_step_metric(
    *,
    step_id: int,
    action: RuntimeAction | None,
    event: RuntimeEvent,
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    executed_regions: int,
    best_reward_so_far: float | None,
    recovery_execution_attempt: bool,
) -> tuple[dict[str, Any], float | None]:
    reward_before = _state_reward(before_state)
    reward_after = _state_reward(after_state)
    reward_delta = (
        reward_after - reward_before
        if reward_before is not None and reward_after is not None
        else None
    )
    reward_candidates = [
        reward
        for reward in (best_reward_so_far, reward_before, reward_after)
        if reward is not None
    ]
    updated_best = max(reward_candidates) if reward_candidates else best_reward_so_far
    reward_drop = (
        updated_best - reward_after
        if updated_best is not None and reward_after is not None
        else None
    )

    region_id = event.region_id
    if region_id is None and action is not None:
        region_id = action.args.get("region_id") or action.args.get("group_id")

    trace_event_count = len(event.evidence.get("trace_events", []))
    append_recovery_source_appended = (
        action is not None
        and action.action == "append_recovery"
        and event.status == "success"
    )
    recovery_execution_reward_improved = bool(
        recovery_execution_attempt and reward_delta is not None and reward_delta > 0
    )
    recovery_execution_trace_improved = bool(
        recovery_execution_attempt and trace_event_count > 0
    )

    metric = {
        "step_id": step_id,
        "action": action.action if action is not None else "invalid",
        "region_id": region_id,
        "event_action": event.action,
        "event_status": event.status,
        "event_message": event.message,
        "duration_s": event.duration_s,
        "trace_event_count": trace_event_count,
        "executed_regions_so_far": executed_regions,
        "reward_before": reward_before,
        "reward_after": reward_after,
        "reward_delta": reward_delta,
        "best_reward_so_far": updated_best,
        "reward_drop_from_best": reward_drop,
        "append_recovery_source_appended": append_recovery_source_appended,
        "recovery_execution_attempt": recovery_execution_attempt,
        "recovery_execution_reward_improved": recovery_execution_reward_improved,
        "recovery_execution_trace_improved": recovery_execution_trace_improved,
        "recovery_execution_improved": (
            recovery_execution_reward_improved or recovery_execution_trace_improved
        ),
        "recovery_execution_effective": recovery_execution_reward_improved,
        "task_completed_before": before_state.get("task_completed"),
        "task_completed_after": after_state.get("task_completed"),
        "state_before": before_state,
        "state_after": after_state,
    }
    return metric, updated_best


def _state_reward(state: dict[str, Any]) -> float | None:
    value = state.get("reward")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _jsonable_metric_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable_metric_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable_metric_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_metric_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _capsule_object_pose_snapshot(low_level_env: Any) -> dict[str, Any]:
    get_all_object_poses = getattr(low_level_env, "_get_all_object_poses", None)
    if callable(get_all_object_poses):
        try:
            raw_object_poses = get_all_object_poses()
        except Exception:
            raw_object_poses = None
        normalized_poses = _normalize_named_object_poses(raw_object_poses)
        if normalized_poses:
            return normalized_poses

    robosuite_env = getattr(low_level_env, "robosuite_env", None)
    sim = getattr(robosuite_env, "sim", None)
    if robosuite_env is None or sim is None:
        return {}

    object_poses: dict[str, Any] = {}
    for attr_name, obj in vars(robosuite_env).items():
        root_body = getattr(obj, "root_body", None)
        if not root_body:
            continue
        try:
            body_id = sim.model.body_name2id(root_body)
            object_poses[attr_name] = {
                "body": root_body,
                "pos": _jsonable_metric_value(sim.data.body_xpos[body_id]),
                "quat_wxyz": _jsonable_metric_value(sim.data.body_xquat[body_id]),
            }
        except Exception as exc:
            object_poses[attr_name] = {
                "body": root_body,
                "error": type(exc).__name__,
            }
    return object_poses


def _normalize_named_object_poses(raw_object_poses: Any) -> dict[str, Any]:
    if not isinstance(raw_object_poses, Mapping):
        return {}

    normalized: dict[str, Any] = {}
    for object_name, raw_pose in raw_object_poses.items():
        if isinstance(raw_pose, Mapping):
            pose = {
                str(key): _jsonable_metric_value(value)
                for key, value in raw_pose.items()
            }
        elif isinstance(raw_pose, (list, tuple)) and len(raw_pose) == 2:
            position, quaternion_wxyz = raw_pose
            pose = {
                "pos": _jsonable_metric_value(position),
                "quat_wxyz": _jsonable_metric_value(quaternion_wxyz),
            }
        else:
            continue
        normalized[str(object_name)] = pose
    return normalized


def _safe_task_completed(env: CodeExecutionEnvBase) -> bool | None:
    low_level_env = getattr(env, "low_level_env", None)
    if low_level_env is not None and hasattr(low_level_env, "task_completed"):
        try:
            return bool(low_level_env.task_completed())
        except Exception:
            return None
    return None


def _summarize_runtime_value(value: Any) -> dict[str, Any]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        summary = {"type": type(value).__name__, "shape": list(shape)}
        if isinstance(value, np.ndarray) and value.size <= 32 and value.dtype.kind in "biuf":
            summary["value"] = value.tolist()
        return summary
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def _run_single_trial(
    env: CodeExecutionEnvBase,
    trial: int,
    args: LaunchArgs,
    config: dict[str, Any],
    multi_turn_prompt: str | None,
    partial_artifacts: dict[str, Any] | None = None,
) -> TrialSummary:
    """Execute a single trial end-to-end.

    Steps:
        1. Reset the environment.
        2. Capture initial visual feedback (if configured).
        3. Query the model for initial code generation.
        4. Execute code blocks one-by-one, with optional multi-turn regeneration.
        5. Save artifacts (code, logs, per-turn videos, combined video) and return a TrialSummary.
    """
    if config.get("agent_mode", "code") == "capsule":
        return _run_capsule_trial(env, trial, args, config)

    trial_start_time = time.time()

    use_video_diff = config.get("use_video_differencing", False)
    use_wrist = config.get("use_wrist_camera", False)

    # --- 1. Reset environment ---
    obs, _ = env.reset(options={"trial": trial}, seed=trial)
    obs["full_prompt"] = copy.deepcopy(obs["full_prompt"])
    _patch_libero_goal(env, obs)

    if config["record_video"] and hasattr(env, "enable_video_capture"):
        env.enable_video_capture(True, clear=True, wrist_camera=use_wrist)
    elif use_video_diff and hasattr(env, "enable_video_capture"):
        # Video differencing needs frame recording even without record_video
        env.enable_video_capture(True, clear=True, wrist_camera=use_wrist)

    # --- Shared trial state ---
    code_blocks: list[str] = []
    code_block_metadata: list[dict[str, Any]] = []
    all_responses: list[dict[str, Any]] = []
    stderr_history: list[str] = []
    num_regenerations = 0
    num_finishes = 0
    info_step: dict[str, Any] = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
    reward = 0.0
    terminated = truncated = False
    sandbox_rc_override = None
    ensemble_data = None
    multiturn_ensemble_data: list[dict[str, Any]] = []

    # Per-turn frame tracking (for video differencing and per-turn video saving)
    turn_frame_ranges: list[tuple[int, int]] = []

    # Wrist camera base64 history for image-based multiview differencing
    wrist_base64_history: list[str] | None = [] if use_wrist else None

    visual_differencing_args = ModelQueryArgs(
        model=args.visual_differencing_model,
        server_url=args.visual_differencing_model_server_url,
        api_key=args.visual_differencing_model_api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        debug=args.debug,
    )

    if config["use_img_differencing"] or use_video_diff:
        assert visual_differencing_args.model in VLM_MODELS, (
            "Image/video differencing model must be in the list of VLM models"
        )

    # --- 2. Capture initial visual feedback ---
    visual_feedback_imgs, visual_feedback_base64_history, task_description = (
        _capture_initial_visual_feedback(env, obs, config, args, visual_differencing_args)
    )

    # Seed wrist base64 history with initial wrist image
    if use_wrist and wrist_base64_history is not None and hasattr(env, "render_wrist"):
        wrist_img = env.render_wrist()
        if wrist_img is not None:
            pil_wrist = Image.fromarray(wrist_img)
            buf = io.BytesIO()
            pil_wrist.save(buf, format="png")
            wrist_base64_history.append(
                f"data:image/png;base64,"
                f"{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            )

    # --- 3. Initial code generation ---
    if config["use_oracle_code"]:
        raw_code = env.oracle_code
        with open(os.path.join(config["output_dir"], "oracle_code.py"), "w") as f:
            f.write(raw_code)
        reasoning = None
        ensemble_data = None
    else:
        raw_code, reasoning, ensemble_data = _query_initial_code(args, config, obs)

    # Initialize partial artifacts for timeout recovery
    if partial_artifacts is not None:
        partial_artifacts.update({
            "raw_code": raw_code,
            "code_blocks": code_blocks,
            "code_block_metadata": code_block_metadata,
            "all_responses": all_responses,
            "visual_feedback_imgs": visual_feedback_imgs,
            "info_step": info_step,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "num_regenerations": num_regenerations,
            "num_finishes": num_finishes,
            "num_code_blocks": 0,
            "ensemble_data": ensemble_data,
            "multiturn_ensemble_data": multiturn_ensemble_data,
        })

    # Parse initial code into blocks
    initial_blocks = _extract_code(raw_code)
    code_blocks.extend(initial_blocks)
    code_block_metadata.extend([{"generation": 0, "regenerated": False}] * len(initial_blocks))
    all_responses.append({
        "block_idx": [0],
        "code_blocks": initial_blocks,
        "decision": "initial",
        "initial_prompt": copy.deepcopy(obs["full_prompt"]),
        "reasoning": reasoning if reasoning is not None else "",
    })

    with open(os.path.join(config["output_dir"], "all_responses.json"), "w") as f:
        json.dump(all_responses, f)

    if args.debug:
        with open(os.path.join(config["output_dir"], "code_init.txt"), "w") as f:
            f.write("\n".join(initial_blocks))

    # --- 4. Execute code blocks (with optional multi-turn) ---
    info_step = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
    reward = 0.0
    terminated = truncated = False
    code_block_idx = 0

    # Track whether we're recording frames (for video diff or record_video)
    recording_frames = (
        (config["record_video"] or use_video_diff)
        and hasattr(env, "get_video_frame_count")
    )

    while code_block_idx < len(code_blocks) and code_block_idx <= MULTITURN_LIMIT:
        code = code_blocks[code_block_idx]
        code_block_idx += 1

        # Record frame index before step
        frame_start = env.get_video_frame_count() if recording_frames else 0

        obs_next, reward, terminated, truncated, info_step = env.step(code)

        # Record frame index after step
        frame_end = env.get_video_frame_count() if recording_frames else 0
        turn_frame_ranges.append((frame_start, frame_end))

        if partial_artifacts is not None:
            partial_artifacts.update({
                "info_step": info_step,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            })

        obs = obs_next

        # Multi-turn decision
        if multi_turn_prompt:
            if "terminated episode" in info_step["stderr"]:
                truncated = True
                break
            max_regenerations = config.get("max_regenerations")
            if max_regenerations is not None and num_regenerations >= int(max_regenerations):
                print(f"Reached max_regenerations={max_regenerations}; stopping multi-turn loop")
                break

            # Get turn frames for video differencing
            turn_frames = None
            wrist_turn_frames = None
            if use_video_diff and recording_frames:
                turn_frames = env.get_video_frames_range(frame_start, frame_end)
                if use_wrist and hasattr(env, "get_wrist_video_frames_range"):
                    wrist_turn_frames = env.get_wrist_video_frames_range(
                        frame_start, frame_end,
                    )

            decision, new_code, mt_reasoning, mt_ensemble, decision_prompt = _handle_multi_turn_step(
                env, obs, args, config, visual_differencing_args,
                multi_turn_prompt, code_blocks, code_block_idx, info_step,
                task_description, visual_feedback_imgs, visual_feedback_base64_history,
                stderr_history,
                turn_frames=turn_frames,
                wrist_turn_frames=wrist_turn_frames,
                wrist_base64_history=wrist_base64_history,
            )

            if mt_ensemble is not None:
                mt_ensemble["regeneration"] = num_regenerations + 1
                multiturn_ensemble_data.append(mt_ensemble)

            if decision == "regenerate":
                print("Model chose to regenerate code")
                new_blocks = _extract_code(new_code)
                all_responses.append({
                    "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                    "block_idx": [code_block_idx],
                    "code_blocks": new_blocks,
                    "decision": "regenerate",
                    "reasoning": mt_reasoning if mt_reasoning is not None else "",
                })
                del code_blocks[code_block_idx:]
                del code_block_metadata[code_block_idx:]
                code_blocks.extend(new_blocks)
                code_block_metadata.extend(
                    [{"generation": num_regenerations + 1, "regenerated": True,
                      "regenerated_at_idx": code_block_idx}]
                    * len(new_blocks)
                )
                num_regenerations += 1
                if partial_artifacts is not None:
                    partial_artifacts["num_regenerations"] = num_regenerations

            elif decision == "finish":
                all_responses.append({
                    "decision": "finish",
                    "reasoning": mt_reasoning if mt_reasoning is not None else (new_code or ""),
                })
                print("Model chose to finish")
                num_finishes += 1
                if partial_artifacts is not None:
                    partial_artifacts["num_finishes"] = num_finishes
                break

        print(f"Code block {code_block_idx} done")
        print(f"Number of code blocks: {len(code_blocks)}")

        # Save intermediate artifacts (code, logs) per code block
        final_code = _annotate_code_blocks(code_blocks, code_block_metadata)
        _save_trial_artifacts(
            config, trial, info_step["sandbox_rc"], reward,
            info_step.get("task_completed", False), final_code, raw_code,
            all_responses, ["-" * 100, "Generated program:", final_code],
            visual_feedback_imgs,
        )

        # Only save intermediate video if NOT doing per-turn saving
        # (per-turn saving is deferred to after the loop to avoid clearing the buffer)
        if not recording_frames:
            _save_trial_video(
                env, config, trial, info_step, reward, len(code_blocks),
                suffix_extra=str(len(code_blocks)),
            )

    print("Code blocks done")

    # --- 5. Build final summary ---
    final_code = _annotate_code_blocks(code_blocks, code_block_metadata)
    num_code_blocks = len(code_blocks)

    if partial_artifacts is not None:
        partial_artifacts["final_code"] = final_code
        partial_artifacts["num_code_blocks"] = num_code_blocks

    # Override sandbox_rc for terminated-episode stderr
    if "executing action in terminated episode" in info_step["stderr"]:
        sandbox_rc_override = 0
    if sandbox_rc_override is not None:
        info_step["sandbox_rc"] = sandbox_rc_override

    stderr = "\n\n".join(stderr_history) if stderr_history else info_step["stderr"]
    log_lines = _build_log_lines(
        final_code, info_step, reward, terminated, truncated,
        num_regenerations, num_finishes, num_code_blocks,
        stderr_override=stderr,
    )

    code_path = _save_trial_artifacts(
        config, trial, info_step["sandbox_rc"], reward,
        info_step.get("task_completed", False), final_code, raw_code,
        all_responses, log_lines, visual_feedback_imgs,
        ensemble_data=ensemble_data,
        multiturn_ensemble_data=multiturn_ensemble_data,
    )

    # Save per-turn and combined videos
    if recording_frames and turn_frame_ranges:
        _save_turn_and_combined_videos(
            env, config, trial, info_step, reward, turn_frame_ranges,
        )
    else:
        _save_trial_video(env, config, trial, info_step, reward, num_code_blocks)

    success = info_step["sandbox_rc"] == 0

    # --- Evolving skill library integration (opt-in) ---
    if config.get("evolve_skill_library", False) and info_step.get("task_completed", False):
        try:
            from capx.skills import SkillLibrary

            skill_lib_path = config.get("skill_library_path", None)
            skill_lib = SkillLibrary(path=skill_lib_path)
            task_name = config.get("task_name", f"trial_{trial}")
            new_skills = skill_lib.extract_from_code(final_code, task_name=task_name)
            skill_lib.save()
            if new_skills:
                print(f"[SkillLibrary] Extracted {len(new_skills)} new skill(s): {new_skills}")
        except Exception as exc:
            print(f"[SkillLibrary] Skill extraction failed: {exc}")

    print(f"Trial {trial} took {time.time() - trial_start_time:.2f} seconds")

    gc.collect()

    return TrialSummary(
        trial=trial,
        success=success,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        sandbox_rc=info_step["sandbox_rc"],
        log="\n".join(log_lines),
        task_completed=info_step.get("task_completed", None),
        code_path=code_path,
        num_regenerations=num_regenerations,
        num_finishes=num_finishes,
        num_code_blocks=num_code_blocks,
    )


def _patch_libero_goal(env: CodeExecutionEnvBase, obs: dict[str, Any]) -> None:
    """Inject the LIBERO task language into the prompt template if applicable."""
    if not hasattr(env.low_level_env, "handle"):
        return
    handle = env.low_level_env.handle
    if (
        hasattr(handle, "task_language")
        and "libero_environment_goal" in obs["full_prompt"][-1]["content"][0]["text"]
    ):
        goal = getattr(handle, "task_language")
        obs["full_prompt"][-1]["content"][0]["text"] = (
            obs["full_prompt"][-1]["content"][0]["text"].format(
                libero_environment_goal=goal
            )
        )
