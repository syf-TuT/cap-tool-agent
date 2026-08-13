import base64
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import capx.envs.trial as trial_module
import numpy as np
import pytest
from PIL import Image
from capx.envs.trial import (
    _attach_capsule_visuals,
    _capture_capsule_visuals,
    _describe_initial_scene,
    _execute_runtime_action,
    _get_video_differencing_feedback,
    _get_visual_differencing_feedback,
    _handle_multi_turn_step,
    _no_rollback_guard_event,
    _program_contract_guard_event,
    _query_initial_code,
    _reward_drop_guard_event,
    _run_capsule_trial,
    _run_single_trial,
    _sanitize_multimodal_prompt,
    _save_capsule_visuals,
    _summarize_runtime_value,
)
from capx.llm.client import ModelQueryArgs
from capx.llm.context import get_trial_llm_context, trial_llm_context
from capx.runtime_control.contract import ProgramContractViolation
from capx.runtime_control.executor import CapsuleExecutor
from capx.runtime_control.schema import CodeRegion, CodeRegionGroup, RuntimeAction
from capx.runtime_control.trace import wrap_function_for_trace


class FakeApi:
    def __init__(self):
        self.moved = False
        self.observed = False

    def functions(self):
        return {
            "get_observation": self.get_observation,
            "get_pose": self.get_pose,
            "move_to": self.move_to,
        }

    def get_observation(self):
        self.observed = True
        return {"state": "current"}

    def get_pose(self, name):
        return [1, 2, 3]

    def move_to(self, pose):
        self.moved = True


class FakeCapsuleEnv:
    oracle_code = ""

    def __init__(self):
        self.api = FakeApi()
        self.low_level_env = object()
        self._apis = {"fake": self.api}

    def reset(self, *, seed=None, options=None):
        return {
            "full_prompt": [
                {"role": "system", "content": "x"},
                {"role": "user", "content": [{"type": "text", "text": "task"}]},
            ]
        }, {}

    def _build_capsule_globals(self, trace=None):
        globals_dict = {}
        for api in self._apis.values():
            for fn_name, fn in api.functions().items():
                globals_dict[fn_name] = (
                    wrap_function_for_trace(fn_name, fn, trace) if trace is not None else fn
                )
        return globals_dict

    def compute_reward(self):
        return 1.0


class FakeIncompleteCapsuleEnv(FakeCapsuleEnv):
    def compute_reward(self):
        return 0.0


class FakeHandleRecoveryApi:
    def __init__(self):
        self.observed = False

    def functions(self):
        return {"get_handle0_pos": self.get_handle0_pos}

    def recovery_observation_functions(self):
        return {"get_handle0_pos"}

    def get_handle0_pos(self):
        self.observed = True
        return [0.1, 0.2, 0.3]


class FakeHandleRecoveryEnv(FakeIncompleteCapsuleEnv):
    def __init__(self):
        self.api = FakeHandleRecoveryApi()
        self.low_level_env = object()
        self._apis = {"fake": self.api}


class FakeRewardDropApi(FakeApi):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.moves = []

    def move_to(self, pose):
        self.moved = True
        self.moves.append(pose)
        if pose == "good":
            self.env.reward = 0.75
        elif pose == "bad":
            self.env.reward = 0.1
        elif pose == "recover":
            self.env.reward = 1.0


class FakeRewardDropCapsuleEnv(FakeCapsuleEnv):
    def __init__(self):
        self.reward = 0.0
        self.api = FakeRewardDropApi(self)
        self.low_level_env = object()
        self._apis = {"fake": self.api}

    def compute_reward(self):
        return self.reward


class FakeCustomMoveApi:
    def __init__(self):
        self.moves = []

    def functions(self):
        return {"custom_move": self.custom_move}

    def side_effect_functions(self):
        return {"custom_move"}

    def custom_move(self, pose):
        self.moves.append(pose)


class FakeCustomMoveCapsuleEnv(FakeIncompleteCapsuleEnv):
    def __init__(self):
        self.api = FakeCustomMoveApi()
        self.low_level_env = object()
        self._apis = {"fake": self.api}


class FakeGripperApi:
    def __init__(self):
        self.calls = []

    def functions(self):
        return {"close_gripper": self.close_gripper}

    def side_effect_functions(self):
        return {"close_gripper"}

    def close_gripper(self):
        self.calls.append("close_gripper")


class FakeGripperCapsuleEnv(FakeIncompleteCapsuleEnv):
    def __init__(self):
        self.api = FakeGripperApi()
        self.low_level_env = object()
        self._apis = {"fake": self.api}


class FakeUnsafeNonPrivilegedCapsuleEnv(FakeGripperCapsuleEnv):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(privileged=False)
        self.low_level_env = SimpleNamespace(
            _get_all_object_poses=lambda: {"secret_object": "SIM_TRUTH_SENTINEL"}
        )

    def _build_capsule_globals(self, trace=None):
        globals_dict = super()._build_capsule_globals(trace=trace)
        globals_dict["env"] = self.low_level_env
        globals_dict["APIS"] = self._apis
        return globals_dict


class FakeSuccessfulNonPrivilegedCapsuleEnv(FakeUnsafeNonPrivilegedCapsuleEnv):
    def compute_reward(self):
        return 1.0 if self.api.calls else 0.0


class FakeSuccessfulRecoveryApi(FakeGripperApi):
    def functions(self):
        functions = dict(super().functions())
        functions["get_observation"] = self.get_observation
        return functions

    def recovery_observation_functions(self):
        return {"get_observation"}

    def get_observation(self):
        return {"gripper": "current"}


class FakeSuccessfulRecoveryNonPrivilegedCapsuleEnv(
    FakeSuccessfulNonPrivilegedCapsuleEnv
):
    def __init__(self):
        self.api = FakeSuccessfulRecoveryApi()
        self.low_level_env = object()
        self._apis = {"fake": self.api}
        self.cfg = SimpleNamespace(privileged=False)


class FakePrivilegedCapsuleEnv(FakeGripperCapsuleEnv):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(privileged=True)


class FakeTraceFailureApi(FakeApi):
    def functions(self):
        functions = dict(super().functions())
        functions["fail_primitive"] = self.fail_primitive
        return functions

    def fail_primitive(self):
        raise RuntimeError("primitive failed")


class FakeTraceFailureEnv(FakeIncompleteCapsuleEnv):
    def __init__(self):
        self.api = FakeTraceFailureApi()
        self.low_level_env = object()
        self._apis = {"fake": self.api}


class FakeVideoCapsuleEnv(FakeCapsuleEnv):
    def __init__(self):
        super().__init__()
        self.video_capture_args = None
        self.video_clear_requested = None

    def enable_video_capture(self, enabled, *, clear=False, wrist_camera=False):
        self.video_capture_args = {
            "enabled": enabled,
            "clear": clear,
            "wrist_camera": wrist_camera,
        }

    def get_video_frames(self, *, clear=False):
        self.video_clear_requested = clear
        return ["frame-1", "frame-2"]


class FakeLiberoLowLevelEnv:
    def __init__(self):
        self._step_count = 3
        self._sim_step_count = 12
        self._gripper_fraction = np.float32(0.25)
        self.gripper_link_wxyz_xyz = np.array(
            [1.0, 0.0, 0.0, 0.0, 0.4, -0.2, 0.3], dtype=np.float32
        )
        self._current_joints = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    def task_completed(self):
        return False

    def _get_all_object_poses(self):
        return {
            "alphabet_soup": (
                np.array([0.2, 0.1, 0.05]),
                np.array([1.0, 0.0, 0.0, 0.0]),
            )
        }


class FakeLiberoCapsuleEnv(FakeIncompleteCapsuleEnv):
    def __init__(self):
        super().__init__()
        self.low_level_env = FakeLiberoLowLevelEnv()


def _record_current_stage():
    context = get_trial_llm_context()
    assert context is not None
    call_index = context.next_call_index()
    context.record_attempt(
        call_index=call_index,
        attempt=1,
        mode="nonstreaming",
        http_status=200,
        ttfb_ms=1,
        first_content_ms=1,
        started_monotonic=1.0,
        finished_monotonic=1.001,
        remaining_before_ms=None,
        outcome="success",
        error_kind=None,
        retry_scheduled=False,
    )
    return {"content": "FINISH", "reasoning": None}


def _telemetry_stages(path):
    return [json.loads(line)["stage"] for line in path.read_text().splitlines()]


def _capsule_step_metrics(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _json_string_payload(value: str) -> str:
    return json.dumps(value)[1:-1]


def test_capsule_state_level_proprioceptive_excludes_object_truth():
    snapshot = trial_module._capsule_state_snapshot(
        FakeLiberoCapsuleEnv(), state_level="proprioceptive"
    )

    assert snapshot["reward"] == 0.0
    assert snapshot["task_completed"] is False
    assert snapshot["step_count"] == 3
    assert snapshot["sim_step_count"] == 12
    assert snapshot["gripper_fraction"] == pytest.approx(0.25)
    assert snapshot["gripper_wxyz_xyz"] == pytest.approx(
        [1.0, 0.0, 0.0, 0.0, 0.4, -0.2, 0.3]
    )
    assert snapshot["robot_joint_pos"] == pytest.approx([0.1, 0.2, 0.3])
    assert "object_poses" not in snapshot


def test_capsule_state_level_full_includes_jsonable_libero_object_poses():
    snapshot = trial_module._capsule_state_snapshot(
        FakeLiberoCapsuleEnv(), state_level="full"
    )

    assert snapshot["object_poses"]["alphabet_soup"] == {
        "pos": [0.2, 0.1, 0.05],
        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    json.dumps(snapshot)


def test_capsule_state_level_rejects_invalid_value():
    with pytest.raises(ValueError, match=r"allowed.*full.*proprioceptive"):
        trial_module._capsule_state_snapshot(
            FakeLiberoCapsuleEnv(), state_level="privileged"
        )


def test_capsule_state_level_snapshots_do_not_share_nested_values():
    env = FakeLiberoCapsuleEnv()
    public_state = trial_module._capsule_state_snapshot(env, state_level="full")
    diagnostic_state = trial_module._capsule_state_snapshot(env, state_level="full")

    public_state["robot_joint_pos"][0] = 99.0
    public_state["object_poses"]["alphabet_soup"]["pos"][0] = 88.0
    public_state["object_poses"]["alphabet_soup"]["new"] = {"nested": ["changed"]}

    assert diagnostic_state["robot_joint_pos"][0] == pytest.approx(0.1)
    assert diagnostic_state["object_poses"]["alphabet_soup"] == {
        "pos": [0.2, 0.1, 0.05],
        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    }


def test_capsule_llm_step_separates_prompt_and_diagnostic_state_artifacts(tmp_path):
    trial_module._run_capsule_llm_step_loop(
        FakeLiberoCapsuleEnv(),
        trial=2,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 2,
            "capsule_prompt_state_level": "proprioceptive",
            "capsule_diagnostic_state_level": "full",
        },
        initial_code="x = 1\n",
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "finish", "args": {}},
        ],
    )

    ordinary_paths = [
        tmp_path / "capsule_prompts_trial_02.json",
        tmp_path / "capsule_trace_trial_02.json",
        tmp_path / "capsule_step_metrics_trial_02.jsonl",
    ]
    for path in ordinary_paths:
        artifact = path.read_text()
        assert "alphabet_soup" not in artifact
        assert '"pos": [0.2, 0.1, 0.05]' not in artifact

    diagnostic_path = tmp_path / "capsule_diagnostics_trial_02.jsonl"
    rows = [json.loads(line) for line in diagnostic_path.read_text().splitlines()]
    assert [row["step_id"] for row in rows] == [1, 2]
    assert rows[0]["state_before"]["object_poses"]["alphabet_soup"] == {
        "pos": [0.2, 0.1, 0.05],
        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    assert rows[1]["state_after"]["object_poses"]["alphabet_soup"] == {
        "pos": [0.2, 0.1, 0.05],
        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    }


def test_capsule_llm_step_diagnostic_state_artifact_none_writes_no_file(tmp_path):
    trial_module._run_capsule_llm_step_loop(
        FakeLiberoCapsuleEnv(),
        trial=3,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 1,
            "capsule_diagnostic_state_level": "none",
        },
        initial_code="x = 1\n",
        scripted_actions=[{"action": "finish", "args": {}}],
    )

    assert not (tmp_path / "capsule_diagnostics_trial_03.jsonl").exists()


@pytest.mark.parametrize(
    ("config_key", "state_level"),
    [
        ("capsule_prompt_state_level", "none"),
        ("capsule_diagnostic_state_level", "privileged"),
    ],
)
def test_capsule_llm_step_rejects_invalid_configured_state_level_before_action(
    tmp_path, config_key, state_level
):
    env = FakeLiberoCapsuleEnv()

    with pytest.raises(ValueError, match=r"allowed"):
        trial_module._run_capsule_llm_step_loop(
            env,
            trial=4,
            args=SimpleNamespace(model="test", use_oracle_code=False),
            config={
                "output_dir": str(tmp_path),
                "max_capsule_steps": 1,
                config_key: state_level,
            },
            initial_code='move_to("blocked")\n',
            scripted_actions=[
                {"action": "run_group", "args": {"group_id": "group_1"}}
            ],
        )

    assert env.api.moved is False


def test_initial_generation_query_uses_initial_code_stage(tmp_path, monkeypatch):
    telemetry_path = tmp_path / "initial.jsonl"
    monkeypatch.setattr(
        "capx.envs.trial._query_model", lambda args, prompt: _record_current_stage()
    )
    args = SimpleNamespace(model="test")
    config = {
        "output_dir": str(tmp_path),
        "use_parallel_ensemble": False,
    }

    with trial_llm_context(trial=1, telemetry_path=telemetry_path):
        _query_initial_code(
            args,
            config,
            {"full_prompt": [{"role": "user", "content": "task"}]},
        )

    assert _telemetry_stages(telemetry_path) == ["initial_code"]


class _FakeCapsuleVisualEnv:
    def __init__(self, *, fail_wrist=False):
        self.fail_wrist = fail_wrist

    def render(self):
        return np.full((2, 3, 3), [10, 20, 30], dtype=np.uint8)

    def render_wrist(self):
        if self.fail_wrist:
            raise RuntimeError("camera token must never reach an artifact")
        return np.full((1, 2, 3), [40, 50, 60], dtype=np.uint8)


def _decoded_data_url(data_url):
    return base64.b64decode(data_url.split(",", 1)[1])


def _capsule_test_visual(camera, rgb, *, data_url_padding=0):
    class _SolidColorEnv:
        def render(self):
            return np.full((2, 3, 3), rgb, dtype=np.uint8)

    data_url, image = trial_module._get_visual_feedback(
        _SolidColorEnv(), use_wrist_camera=False
    )
    if data_url_padding:
        data_url = data_url.replace(",", "," + " " * data_url_padding, 1)
    return trial_module._capsule_visual_from_payload(camera, data_url, image)


def _prompt_image_urls(prompt):
    return [
        item["image_url"]["url"]
        for message in prompt
        for item in (
            message.get("content", [])
            if isinstance(message.get("content", []), list)
            else []
        )
        if isinstance(item, dict) and item.get("type") == "image_url"
    ]


class _FakeLiberoGoalVisualEnv(FakeCapsuleEnv):
    def __init__(self):
        super().__init__()
        self.shared_prompt = [
            {"role": "system", "content": "x"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Goal: {libero_environment_goal}",
                    }
                ],
            },
        ]
        self.low_level_env = SimpleNamespace(
            handle=SimpleNamespace(
                task_language="Pick the alphabet soup and place it in the basket"
            )
        )

    def reset(self, *, seed=None, options=None):
        return {"full_prompt": self.shared_prompt}, {}

    def render(self):
        return np.full((2, 3, 3), [10, 20, 30], dtype=np.uint8)

    def render_wrist(self):
        return np.full((1, 2, 3), [40, 50, 60], dtype=np.uint8)


def test_capture_capsule_visuals_returns_png_hashed_main_and_wrist_records():
    records, errors = _capture_capsule_visuals(
        _FakeCapsuleVisualEnv(), use_wrist_camera=True
    )

    assert errors == []
    assert [record.camera for record in records] == ["main", "wrist"]
    assert [(record.width, record.height) for record in records] == [(3, 2), (2, 1)]
    for record in records:
        png_bytes = _decoded_data_url(record.data_url)
        assert record.sha256 == hashlib.sha256(png_bytes).hexdigest()
        assert record.metadata() == {
            "camera": record.camera,
            "width": record.width,
            "height": record.height,
            "sha256": record.sha256,
        }
        assert record.image is not None
        assert record.image.size == (record.width, record.height)


def test_capture_capsule_visuals_prefers_combined_multiview_feedback(monkeypatch):
    env = _FakeCapsuleVisualEnv()
    combined_feedback = trial_module._get_visual_feedback(env, use_wrist_camera=True)
    calls = []

    def fake_visual_feedback(requested_env, use_wrist_camera=False):
        calls.append((requested_env, use_wrist_camera))
        return combined_feedback

    monkeypatch.setattr(trial_module, "_get_visual_feedback", fake_visual_feedback)

    records, errors = _capture_capsule_visuals(env, use_wrist_camera=True)

    assert calls == [(env, True)]
    assert errors == []
    assert [record.camera for record in records] == ["main", "wrist"]
    assert [(record.width, record.height) for record in records] == [(3, 2), (2, 1)]
    for index, record in enumerate(records):
        png_bytes = _decoded_data_url(combined_feedback[0][index])
        assert record.sha256 == hashlib.sha256(png_bytes).hexdigest()


def test_capture_capsule_visuals_falls_back_to_current_main_after_combined_failure(
    monkeypatch,
):
    env = _FakeCapsuleVisualEnv()
    get_visual_feedback = trial_module._get_visual_feedback
    current_main = get_visual_feedback(env, use_wrist_camera=False)
    calls = []

    def fake_visual_feedback(requested_env, use_wrist_camera=False):
        calls.append((requested_env, use_wrist_camera))
        if use_wrist_camera:
            raise RuntimeError("combined capture failed")
        return current_main

    monkeypatch.setattr(trial_module, "_get_visual_feedback", fake_visual_feedback)

    records, errors = _capture_capsule_visuals(env, use_wrist_camera=True)

    assert calls == [(env, True), (env, False)]
    assert [record.camera for record in records] == ["main"]
    assert records[0].sha256 == hashlib.sha256(
        _decoded_data_url(current_main[0])
    ).hexdigest()
    assert errors == [{"camera": "wrist", "error": "capture_failed"}]


def test_capture_capsule_visuals_keeps_current_main_when_wrist_capture_fails():
    records, errors = _capture_capsule_visuals(
        _FakeCapsuleVisualEnv(fail_wrist=True), use_wrist_camera=True
    )

    assert [record.camera for record in records] == ["main"]
    assert errors == [{"camera": "wrist", "error": "capture_failed"}]
    assert "camera token" not in json.dumps(errors)


def test_attach_capsule_visuals_returns_new_multimodal_prompt_without_mutating_input():
    records, _ = _capture_capsule_visuals(_FakeCapsuleVisualEnv(), use_wrist_camera=True)
    original = [{"role": "user", "content": [{"type": "text", "text": "task"}]}]

    attached = _attach_capsule_visuals(
        original,
        records[:1],
        [{"camera": "wrist", "error": "capture_failed"}],
    )

    assert original == [
        {"role": "user", "content": [{"type": "text", "text": "task"}]}
    ]
    assert attached is not original
    assert [item["type"] for item in attached[-1]["content"]] == [
        "text",
        "text",
        "image_url",
        "text",
    ]
    assert attached[-1]["content"][1]["text"] == "Current main-camera view"
    assert attached[-1]["content"][2]["image_url"]["url"].startswith("data:image/png")
    assert "Current wrist-camera view unavailable" in attached[-1]["content"][3]["text"]


def test_sanitize_capsule_visual_prompt_replaces_images_without_mutation():
    records, _ = _capture_capsule_visuals(_FakeCapsuleVisualEnv(), use_wrist_camera=False)
    record = records[0]
    prompt = _attach_capsule_visuals(
        [{"role": "user", "content": [{"type": "text", "text": "task"}]}],
        records,
        [],
    )
    prompt[-1]["content"].append(
        {
            "type": "image_url",
            "image_url": {"url": "https://secret.example/frame.png?token=private"},
        }
    )

    sanitized = _sanitize_multimodal_prompt(
        prompt, {record.sha256: "capsule_visuals_trial_02/step_00_main.png"}
    )

    assert prompt[-1]["content"][2]["type"] == "image_url"
    serialized = json.dumps(sanitized)
    assert "data:image" not in serialized
    assert "base64," not in serialized
    assert "secret.example" not in serialized
    references = [
        item["image_reference"]
        for item in sanitized[-1]["content"]
        if item.get("type") == "image_reference"
    ]
    assert references[0] == {
        "camera": "main",
        "path": "capsule_visuals_trial_02/step_00_main.png",
        "width": 3,
        "height": 2,
        "sha256": record.sha256,
        "media_type": "image/png",
    }
    assert references[1]["camera"] == "unknown"
    assert references[1]["path"] is None
    assert references[1]["width"] is None
    assert references[1]["height"] is None
    assert len(references[1]["sha256"]) == 64


def test_sanitize_capsule_visual_prompt_removes_unstructured_base64_payload():
    prompt = [{"role": "user", "content": "diagnostic base64,U0VDUkVU"}]

    sanitized = _sanitize_multimodal_prompt(prompt, {})

    serialized = json.dumps(sanitized)
    assert "U0VDUkVU" not in serialized
    assert "base64," not in serialized


@pytest.mark.parametrize(
    "plain_text",
    [
        "The source data: values are tabular.",
        "SSE data: [DONE]",
        'payload = {"data": "data: ordinary value"}',
        'config = {"label": "base64, is an encoding marker"}',
    ],
)
def test_sanitize_capsule_visual_prompt_preserves_plain_data_text(plain_text):
    prompt = ({"role": "user", "content": (plain_text,)},)

    sanitized = _sanitize_multimodal_prompt(prompt, {})

    assert sanitized == [{"role": "user", "content": [plain_text]}]


@pytest.mark.parametrize(
    "unsafe_data_url",
    [
        "data:image/png;base64,U0VD-UkVU",
        "DATA:IMAGE/PNG;BASE64,U0VDUkVU",
        "data:image/png;base64,U0VD \n UkVU",
        "data:image/png;base64,U0VD$UkVU",
    ],
)
def test_sanitize_capsule_visual_prompt_fail_closes_malformed_data_urls(
    unsafe_data_url,
):
    prompt = [{"role": "user", "content": f"before {unsafe_data_url} after"}]

    sanitized = _sanitize_multimodal_prompt(prompt, {})

    serialized = json.dumps(sanitized)
    assert "U0VD" not in serialized
    assert "UkVU" not in serialized
    assert "data:" not in serialized.lower()
    assert "base64," not in serialized.lower()


def test_sanitize_capsule_visual_prompt_recurses_tuples_and_rejects_unsafe_paths():
    records, _ = _capture_capsule_visuals(_FakeCapsuleVisualEnv(), use_wrist_camera=False)
    record = records[0]
    prompt = (
        {
            "role": "user",
            "content": (
                {"type": "image_url", "image_url": {"url": record.data_url}},
                ("data:image/png;base64,U0VDUkVU",),
            ),
        },
    )

    sanitized = _sanitize_multimodal_prompt(
        prompt,
        {record.sha256: {"path": "../secret.png", "camera": "main"}},
    )

    assert isinstance(sanitized, list)
    assert isinstance(sanitized[0]["content"], list)
    assert isinstance(sanitized[0]["content"][1], list)
    serialized = json.dumps(sanitized)
    assert "data:" not in serialized.lower()
    assert "base64," not in serialized.lower()
    assert "secret.png" not in serialized
    assert sanitized[0]["content"][0]["image_reference"]["path"] is None


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "/absolute/secret.png",
        "../secret.png",
        "a/../../b.png",
        r"C:\secret.png",
        r"\\server\share\secret.png",
    ],
)
def test_sanitize_capsule_visual_prompt_omits_unsafe_artifact_paths(unsafe_path):
    records, _ = _capture_capsule_visuals(_FakeCapsuleVisualEnv(), use_wrist_camera=False)
    record = records[0]
    prompt = [{"type": "image_url", "image_url": {"url": record.data_url}}]

    sanitized = _sanitize_multimodal_prompt(
        prompt, {record.sha256: unsafe_path}
    )

    assert sanitized[0]["image_reference"]["path"] is None
    assert unsafe_path not in json.dumps(sanitized)


def test_sanitize_capsule_visual_prompt_identifies_legacy_main_and_wrist_labels():
    records, _ = _capture_capsule_visuals(_FakeCapsuleVisualEnv(), use_wrist_camera=True)
    prompt = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Included below is an image of the initial state.",
                },
                {"type": "image_url", "image_url": {"url": records[0].data_url}},
                {
                    "type": "text",
                    "text": "Included below is an image from the robot's wrist camera.",
                },
                {"type": "image_url", "image_url": {"url": records[1].data_url}},
            ],
        }
    ]

    sanitized = _sanitize_multimodal_prompt(prompt, {})

    references = [
        item["image_reference"]
        for item in sanitized[-1]["content"]
        if item.get("type") == "image_reference"
    ]
    assert [reference["camera"] for reference in references] == ["main", "wrist"]


def test_save_capsule_visuals_writes_decoded_png_bytes_and_returns_relative_mapping(
    tmp_path,
):
    records, _ = _capture_capsule_visuals(_FakeCapsuleVisualEnv(), use_wrist_camera=True)

    artifact_by_sha256, errors = _save_capsule_visuals(
        records, tmp_path, trial_id=2, step_id=3
    )

    assert errors == []
    assert artifact_by_sha256 == {
        records[0].sha256: "capsule_visuals_trial_02/step_03_main.png",
        records[1].sha256: "capsule_visuals_trial_02/step_03_wrist.png",
    }
    for record in records:
        saved = tmp_path / artifact_by_sha256[record.sha256]
        assert saved.read_bytes() == _decoded_data_url(record.data_url)
        with Image.open(saved) as image:
            assert image.size == (record.width, record.height)


def test_save_capsule_visuals_preserves_camera_paths_when_images_have_same_hash(
    tmp_path,
):
    main = _capsule_test_visual("main", [12, 34, 56])
    wrist = replace(main, camera="wrist")

    artifact_by_sha256, errors = _save_capsule_visuals(
        [main, wrist], tmp_path, trial_id=5, step_id=0
    )
    prompt = _attach_capsule_visuals(
        [{"role": "user", "content": [{"type": "text", "text": "task"}]}],
        [main, wrist],
        [],
    )
    sanitized = _sanitize_multimodal_prompt(prompt, artifact_by_sha256)

    assert errors == []
    references = [
        item["image_reference"]
        for item in sanitized[-1]["content"]
        if item.get("type") == "image_reference"
    ]
    assert references[0]["camera"] == "main"
    assert references[0]["path"].endswith("step_00_main.png")
    assert references[1]["camera"] == "wrist"
    assert references[1]["path"].endswith("step_00_wrist.png")
    assert (tmp_path / references[0]["path"]).is_file()
    assert (tmp_path / references[1]["path"]).is_file()


def test_capsule_visual_error_normalization_drops_exception_text_and_unsafe_fields():
    normalized = trial_module._normalize_capsule_visual_errors(
        [
            {
                "camera": "../../secret-camera",
                "error": "save_failed: SECRET_EXCEPTION_TEXT",
                "path": "../secret-path.png",
                "exception": "SECRET_EXCEPTION_TEXT",
            }
        ]
    )

    assert normalized == [{"camera": "unknown", "error": "unknown_error"}]
    serialized = json.dumps(normalized)
    assert "SECRET" not in serialized
    assert "secret" not in serialized


def test_save_capsule_visuals_without_output_dir_is_a_noop():
    records, _ = _capture_capsule_visuals(_FakeCapsuleVisualEnv(), use_wrist_camera=False)

    assert _save_capsule_visuals(records, None, trial_id=0, step_id=0) == ({}, [])


def test_save_capsule_visuals_reports_directory_creation_failure(tmp_path):
    records, _ = _capture_capsule_visuals(_FakeCapsuleVisualEnv(), use_wrist_camera=False)
    invalid_output_dir = tmp_path / "not-a-directory"
    invalid_output_dir.write_text("occupied")

    artifact_by_sha256, errors = _save_capsule_visuals(
        records, invalid_output_dir, trial_id=0, step_id=0
    )

    assert artifact_by_sha256 == {}
    assert errors == [{"camera": "all", "error": "save_failed"}]


def test_save_capsule_visuals_rejects_camera_name_outside_allowlist(tmp_path):
    records, _ = _capture_capsule_visuals(_FakeCapsuleVisualEnv(), use_wrist_camera=False)
    unsafe_record = replace(records[0], camera="../escaped")

    artifact_by_sha256, errors = _save_capsule_visuals(
        [unsafe_record], tmp_path, trial_id=0, step_id=0
    )

    assert artifact_by_sha256 == {}
    assert errors == [{"camera": "unknown", "error": "invalid_camera"}]
    assert list(tmp_path.rglob("*.png")) == []


def test_save_capsule_visuals_revalidates_public_record_before_writing(tmp_path):
    records, _ = _capture_capsule_visuals(_FakeCapsuleVisualEnv(), use_wrist_camera=False)
    invalid_bytes = b"not a png"
    invalid_record = replace(
        records[0],
        data_url="data:image/png;base64," + base64.b64encode(invalid_bytes).decode(),
        sha256=hashlib.sha256(invalid_bytes).hexdigest(),
    )

    artifact_by_sha256, errors = _save_capsule_visuals(
        [invalid_record], tmp_path, trial_id=0, step_id=0
    )

    assert artifact_by_sha256 == {}
    assert errors == [{"camera": "main", "error": "save_failed"}]
    assert list(tmp_path.rglob("*.png")) == []


def test_capsule_visual_rejects_data_url_with_non_token_base64_parameter():
    records, _ = _capture_capsule_visuals(_FakeCapsuleVisualEnv(), use_wrist_camera=False)
    record = records[0]
    malformed = record.data_url.replace(";base64,", ";base64evil,")

    with pytest.raises(ValueError, match="base64"):
        trial_module._capsule_visual_from_payload("main", malformed, record.image)


def test_capsule_visual_rejects_non_png_media_type():
    records, _ = _capture_capsule_visuals(_FakeCapsuleVisualEnv(), use_wrist_camera=False)
    record = records[0]
    wrong_media_type = record.data_url.replace("data:image/png", "data:text/plain")

    with pytest.raises(ValueError, match="PNG"):
        trial_module._capsule_visual_from_payload(
            "main", wrong_media_type, record.image
        )


def test_capsule_visual_rejects_decoded_bytes_that_are_not_png():
    data_url = "data:image/png;base64," + base64.b64encode(b"not a png").decode()

    with pytest.raises(ValueError, match="PNG"):
        trial_module._capsule_visual_from_payload(
            "main", data_url, Image.new("RGB", (1, 1))
        )


@pytest.mark.parametrize(
    "supplied_image",
    [Image.new("RGB", (9, 9)), Image.new("RGBA", (3, 2))],
    ids=["dimensions", "mode"],
)
def test_capsule_visual_rejects_pil_metadata_mismatch(supplied_image):
    env = _FakeCapsuleVisualEnv()
    data_url, _ = trial_module._get_visual_feedback(env, use_wrist_camera=False)

    with pytest.raises(ValueError, match="metadata"):
        trial_module._capsule_visual_from_payload("main", data_url, supplied_image)


def test_query_initial_code_sanitizes_initial_prompt_multimodal_artifact(
    tmp_path, monkeypatch
):
    records, _ = _capture_capsule_visuals(_FakeCapsuleVisualEnv(), use_wrist_camera=False)
    record = records[0]
    prompt = _attach_capsule_visuals(
        [{"role": "user", "content": [{"type": "text", "text": "task"}]}],
        records,
        [],
    )
    monkeypatch.setattr(
        "capx.envs.trial._query_model",
        lambda args, live_prompt: {"content": "code", "reasoning": None},
    )

    _query_initial_code(
        SimpleNamespace(model="test"),
        {"output_dir": str(tmp_path), "use_parallel_ensemble": False},
        {"full_prompt": prompt},
        artifact_by_sha256={
            record.sha256: "capsule_visuals_trial_00/step_00_main.png"
        },
    )

    artifact = (tmp_path / "initial_prompt.txt").read_text()
    assert "image_reference" in artifact
    assert record.sha256 in artifact
    assert "data:image" not in artifact
    assert "base64," not in artifact


def test_capsule_llm_step_libero_goal_before_initial_query_and_shared_prompt_isolation(
    tmp_path, monkeypatch
):
    env = _FakeLiberoGoalVisualEnv()
    captured_prompts = []

    def fake_initial_code(args, config, obs, artifact_by_sha256=None):
        captured_prompts.append(
            {
                "prompt": json.loads(json.dumps(obs["full_prompt"])),
                "artifacts": dict(artifact_by_sha256 or {}),
            }
        )
        return "x = 1\n", None, None

    monkeypatch.setattr(trial_module, "_query_initial_code", fake_initial_code)

    for trial_id in range(2):
        trial_module._run_capsule_llm_step_loop(
            env,
            trial=trial_id,
            args=SimpleNamespace(model="test", use_oracle_code=False),
            config={
                "output_dir": str(tmp_path),
                "max_capsule_steps": 1,
                "use_visual_feedback": True,
                "use_wrist_camera": True,
            },
            scripted_actions=[{"action": "finish", "args": {}}],
        )

    goal = "Pick the alphabet soup and place it in the basket"
    assert len(captured_prompts) == 2
    for captured in captured_prompts:
        prompt_text = json.dumps(captured["prompt"])
        assert goal in prompt_text
        assert "libero_environment_goal" not in prompt_text
        assert len(_prompt_image_urls(captured["prompt"])) == 2
        assert len(captured["artifacts"]) == 2

    assert env.shared_prompt[-1]["content"] == [
        {"type": "text", "text": "Goal: {libero_environment_goal}"}
    ]


def test_capsule_llm_step_visual_feedback_uses_current_pairs_and_sanitized_artifacts(
    tmp_path, monkeypatch
):
    step_0 = [
        _capsule_test_visual("main", [10, 11, 12], data_url_padding=50000),
        _capsule_test_visual("wrist", [20, 21, 22]),
    ]
    step_1 = [
        _capsule_test_visual("main", [30, 31, 32]),
        _capsule_test_visual("wrist", [40, 41, 42]),
    ]
    step_2 = [
        _capsule_test_visual("main", [50, 51, 52]),
        _capsule_test_visual("wrist", [60, 61, 62]),
    ]
    capture_batches = iter([(step_0, []), (step_1, []), (step_2, [])])
    capture_calls = []

    def fake_capture(env, *, use_wrist_camera=False):
        capture_calls.append(use_wrist_camera)
        return next(capture_batches)

    live_prompts = []
    responses = iter(
        [
            {"content": "x = 1\n", "reasoning": None},
            {"content": '{"action":"run_group","args":{"group_id":"group_1"}}'},
            {"content": '{"action":"finish","args":{}}'},
        ]
    )

    def fake_query_model(args, prompt):
        live_prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(trial_module, "_capture_capsule_visuals", fake_capture)
    monkeypatch.setattr(trial_module, "_query_model", fake_query_model)

    trial_module._run_capsule_llm_step_loop(
        FakeCapsuleEnv(),
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 2,
            "use_parallel_ensemble": False,
            "use_visual_feedback": True,
            "capsule_action_visual_feedback": True,
            "use_wrist_camera": True,
        },
    )

    assert capture_calls == [True, True, True]
    assert _prompt_image_urls(live_prompts[0]) == [
        step_0[0].data_url,
        step_0[1].data_url,
    ]
    assert _prompt_image_urls(live_prompts[1]) == [
        step_0[0].data_url,
        step_0[1].data_url,
    ]
    assert _prompt_image_urls(live_prompts[2]) == [
        step_1[0].data_url,
        step_1[1].data_url,
    ]
    assert step_0[0].data_url not in json.dumps(live_prompts[2])
    assert step_0[1].data_url not in json.dumps(live_prompts[2])

    initial_artifact = (tmp_path / "initial_prompt.txt").read_text()
    prompt_artifact_path = tmp_path / "capsule_prompts_trial_00.json"
    prompt_artifact = prompt_artifact_path.read_text()
    for artifact in (initial_artifact, prompt_artifact):
        assert "image_reference" in artifact
        assert "data:image" not in artifact
        assert "base64," not in artifact

    saved_images = sorted(
        (tmp_path / "capsule_visuals_trial_00").glob("step_*.png")
    )
    assert [path.name for path in saved_images] == [
        "step_00_main.png",
        "step_00_wrist.png",
        "step_01_main.png",
        "step_01_wrist.png",
        "step_02_main.png",
        "step_02_wrist.png",
    ]
    sanitized_prompts = json.loads(prompt_artifact_path.read_text())
    for prompt in sanitized_prompts:
        for item in prompt[-1]["content"]:
            if item.get("type") != "image_reference":
                continue
            reference = item["image_reference"]
            assert reference["camera"] in {"main", "wrist"}
            assert reference["width"] == 3
            assert reference["height"] == 2
            assert len(reference["sha256"]) == 64
            assert (tmp_path / reference["path"]).is_file()

    rows = _capsule_step_metrics(
        tmp_path / "capsule_step_metrics_trial_00.jsonl"
    )
    assert [row["action_prompt_image_count"] for row in rows] == [2, 2]
    assert [row["visual_capture_errors"] for row in rows] == [[], []]
    assert [image["path"] for image in rows[0]["action_prompt_images"]] == [
        "capsule_visuals_trial_00/step_00_main.png",
        "capsule_visuals_trial_00/step_00_wrist.png",
    ]
    assert [image["path"] for image in rows[1]["action_prompt_images"]] == [
        "capsule_visuals_trial_00/step_01_main.png",
        "capsule_visuals_trial_00/step_01_wrist.png",
    ]
    text_only_prompt = json.loads(json.dumps(live_prompts[1]))
    text_only_prompt[-1]["content"] = text_only_prompt[-1]["content"][:1]
    assert rows[0]["action_prompt_chars"] == len(
        json.dumps(text_only_prompt, default=str)
    )


def test_capsule_llm_step_visual_feedback_clears_failed_camera_for_next_prompt(
    tmp_path, monkeypatch
):
    step_0 = [
        _capsule_test_visual("main", [1, 2, 3]),
        _capsule_test_visual("wrist", [4, 5, 6]),
    ]
    step_1 = [_capsule_test_visual("main", [7, 8, 9])]
    step_2 = [_capsule_test_visual("main", [10, 11, 12])]
    batches = iter(
        [
            (step_0, []),
            (step_1, [{"camera": "wrist", "error": "capture_failed"}]),
            (step_2, [{"camera": "wrist", "error": "capture_failed"}]),
        ]
    )
    live_prompts = []
    responses = iter(
        [
            {"content": "x = 1\n", "reasoning": None},
            {"content": '{"action":"inspect_trace","args":{}}'},
            {"content": '{"action":"finish","args":{}}'},
        ]
    )

    monkeypatch.setattr(
        trial_module,
        "_capture_capsule_visuals",
        lambda env, *, use_wrist_camera=False: next(batches),
    )

    def fake_query_model(args, prompt):
        live_prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(trial_module, "_query_model", fake_query_model)

    trial_module._run_capsule_llm_step_loop(
        FakeCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 2,
            "use_parallel_ensemble": False,
            "capsule_action_visual_feedback": True,
            "use_wrist_camera": True,
        },
    )

    second_action_prompt = live_prompts[2]
    assert _prompt_image_urls(second_action_prompt) == [step_1[0].data_url]
    assert step_0[1].data_url not in json.dumps(second_action_prompt)
    assert "Current wrist-camera view unavailable (capture_failed)." in json.dumps(
        second_action_prompt
    )
    rows = _capsule_step_metrics(
        tmp_path / "capsule_step_metrics_trial_01.jsonl"
    )
    assert rows[1]["action_prompt_image_count"] == 1
    assert rows[1]["visual_capture_errors"] == [
        {"camera": "wrist", "error": "capture_failed"}
    ]


def test_capsule_llm_step_visual_feedback_keeps_save_failures_out_of_live_prompt(
    tmp_path, monkeypatch
):
    record = _capsule_test_visual("main", [21, 22, 23])
    live_prompts = []

    monkeypatch.setattr(
        trial_module,
        "_capture_capsule_visuals",
        lambda env, *, use_wrist_camera=False: ([record], []),
    )
    monkeypatch.setattr(
        trial_module,
        "_save_capsule_visuals",
        lambda records, output_dir, *, trial_id, step_id: (
            {},
            [{"camera": "main", "error": "save_failed"}],
        ),
    )

    def fake_query_model(args, prompt):
        live_prompts.append(prompt)
        return {"content": '{"action":"finish","args":{}}'}

    monkeypatch.setattr(trial_module, "_query_model", fake_query_model)

    trial_module._run_capsule_llm_step_loop(
        FakeCapsuleEnv(),
        trial=5,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 1,
            "capsule_action_visual_feedback": True,
        },
        initial_code="x = 1\n",
    )

    assert _prompt_image_urls(live_prompts[0]) == [record.data_url]
    serialized_live_prompt = json.dumps(live_prompts[0])
    assert "Current main-camera view unavailable (save_failed)." not in (
        serialized_live_prompt
    )
    assert "save_failed" not in serialized_live_prompt
    prompt_artifact = (
        tmp_path / "capsule_prompts_trial_05.json"
    ).read_text()
    assert "save_failed" not in prompt_artifact

    rows = _capsule_step_metrics(
        tmp_path / "capsule_step_metrics_trial_05.jsonl"
    )
    assert rows[0]["visual_capture_errors"] == []
    assert rows[0]["visual_artifact_errors"] == [
        {"camera": "main", "error": "save_failed"}
    ]


def test_capsule_llm_step_visual_feedback_audits_terminal_refresh_failures(
    tmp_path, monkeypatch
):
    record = _capsule_test_visual("main", [31, 32, 33])
    captures = iter(
        [
            ([record], []),
            ([], [{"camera": "main", "error": "capture_failed"}]),
        ]
    )
    saves = iter(
        [
            ({record.sha256: "capsule_visuals_trial_06/step_00_main.png"}, []),
            ({}, [{"camera": "all", "error": "save_failed"}]),
        ]
    )

    monkeypatch.setattr(
        trial_module,
        "_capture_capsule_visuals",
        lambda env, *, use_wrist_camera=False: next(captures),
    )
    monkeypatch.setattr(
        trial_module,
        "_save_capsule_visuals",
        lambda records, output_dir, *, trial_id, step_id: next(saves),
    )

    summary = trial_module._run_capsule_llm_step_loop(
        FakeCapsuleEnv(),
        trial=6,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 1,
            "capsule_action_visual_feedback": True,
        },
        initial_code="x = 1\n",
        scripted_actions=[{"action": "finish", "args": {}}],
    )

    rows = _capsule_step_metrics(
        tmp_path / "capsule_step_metrics_trial_06.jsonl"
    )
    assert rows[0]["visual_capture_errors"] == []
    assert rows[0]["visual_artifact_errors"] == []
    assert rows[0]["post_action_visual_capture_errors"] == [
        {"camera": "main", "error": "capture_failed"}
    ]
    assert rows[0]["post_action_visual_artifact_errors"] == [
        {"camera": "all", "error": "save_failed"}
    ]
    assert '"step_id": 1' in summary.log
    assert '"error": "capture_failed"' in summary.log
    assert '"error": "save_failed"' in summary.log
    assert '"visual_capture_errors":' in summary.log
    assert '"visual_artifact_errors":' in summary.log
    assert '"visual_artifacts_complete": false' in summary.log


def test_capsule_llm_step_visual_feedback_can_keep_action_prompts_text_only(
    tmp_path, monkeypatch
):
    records = [
        _capsule_test_visual("main", [1, 1, 1]),
        _capsule_test_visual("wrist", [2, 2, 2]),
    ]
    capture_calls = []

    def fake_capture(env, *, use_wrist_camera=False):
        capture_calls.append(use_wrist_camera)
        return records, []

    live_prompts = []
    responses = iter(
        [
            {"content": "x = 1\n", "reasoning": None},
            {"content": '{"action":"inspect_trace","args":{}}'},
            {"content": '{"action":"finish","args":{}}'},
        ]
    )

    monkeypatch.setattr(trial_module, "_capture_capsule_visuals", fake_capture)

    def fake_query_model(args, prompt):
        live_prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(trial_module, "_query_model", fake_query_model)

    trial_module._run_capsule_llm_step_loop(
        FakeCapsuleEnv(),
        trial=2,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 2,
            "use_parallel_ensemble": False,
            "use_visual_feedback": True,
            "capsule_action_visual_feedback": False,
            "use_wrist_camera": True,
        },
    )

    assert capture_calls == [True]
    assert len(_prompt_image_urls(live_prompts[0])) == 2
    assert _prompt_image_urls(live_prompts[1]) == []
    assert _prompt_image_urls(live_prompts[2]) == []
    rows = _capsule_step_metrics(
        tmp_path / "capsule_step_metrics_trial_02.jsonl"
    )
    assert [row["action_prompt_image_count"] for row in rows] == [0, 0]
    assert [row["action_prompt_images"] for row in rows] == [[], []]
    assert [row["post_action_visual_capture_errors"] for row in rows] == [[], []]
    assert [row["post_action_visual_artifact_errors"] for row in rows] == [[], []]


def test_capsule_llm_step_visual_feedback_refreshes_after_all_action_outcomes(
    tmp_path, monkeypatch
):
    records = [_capsule_test_visual("main", [9, 9, 9])]
    capture_calls = []

    def fake_capture(env, *, use_wrist_camera=False):
        capture_calls.append(use_wrist_camera)
        return records, []

    monkeypatch.setattr(trial_module, "_capture_capsule_visuals", fake_capture)

    trial_module._run_capsule_llm_step_loop(
        FakeIncompleteCapsuleEnv(),
        trial=3,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 6,
            "capsule_action_visual_feedback": True,
            "capsule_require_task_success_for_finish": True,
        },
        initial_code="x = 1\n",
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "inspect_variables", "args": {"names": ["x"]}},
            {
                "action": "patch_group",
                "args": {"group_id": "group_1", "source": "x = 2\n"},
            },
            {"action": "finish", "args": {}},
            {"action": "run_group", "args": {"group_id": "missing"}},
            {"action": "unsupported", "args": {}},
        ],
    )

    assert capture_calls == [False] * 7
    trace = json.loads((tmp_path / "capsule_trace_trial_03.json").read_text())
    assert [entry["event"]["status"] for entry in trace] == [
        "success",
        "success",
        "success",
        "warning",
        "invalid",
        "invalid",
    ]


def test_capsule_llm_step_visual_feedback_refreshes_for_forced_recovery_without_prompt(
    tmp_path, monkeypatch
):
    records = [_capsule_test_visual("main", [3, 3, 3])]
    capture_calls = []

    def fake_capture(env, *, use_wrist_camera=False):
        capture_calls.append(use_wrist_camera)
        return records, []

    monkeypatch.setattr(trial_module, "_capture_capsule_visuals", fake_capture)

    trial_module._run_capsule_llm_step_loop(
        FakeRewardDropCapsuleEnv(),
        trial=4,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 2,
            "capsule_action_visual_feedback": True,
        },
        initial_code='move_to("bad")\n',
        scripted_actions=[
            {
                "action": "append_recovery",
                "args": {
                    "source": 'obs = get_observation()\nmove_to("recover")\n'
                },
            }
        ],
    )

    assert capture_calls == [False, False, False]
    rows = _capsule_step_metrics(
        tmp_path / "capsule_step_metrics_trial_04.jsonl"
    )
    assert rows[0]["action_prompt_image_count"] == 1
    assert rows[1]["action_prompt_image_count"] == 0
    assert rows[1]["action_prompt_images"] == []
    assert rows[1]["visual_capture_errors"] == []


def test_visual_model_queries_use_visual_feedback_stage(tmp_path, monkeypatch):
    telemetry_path = tmp_path / "visual.jsonl"
    monkeypatch.setattr(
        "capx.envs.trial._query_model", lambda args, prompt: _record_current_stage()
    )
    monkeypatch.setattr("capx.envs.trial._encode_video_base64", lambda frames: "video-data")
    query_args = ModelQueryArgs(
        model="test",
        server_url="http://example.test",
        api_key=None,
        temperature=0.2,
        max_tokens=100,
    )

    with trial_llm_context(trial=1, telemetry_path=telemetry_path):
        _describe_initial_scene(query_args, "task", "image-data")
        _get_visual_differencing_feedback(
            query_args,
            "task",
            ["before-image", "after-image"],
        )
        _get_video_differencing_feedback(query_args, "task", [object()])

    assert _telemetry_stages(telemetry_path) == [
        "visual_feedback",
        "visual_feedback",
        "visual_feedback",
    ]


def test_multi_turn_decision_query_uses_multi_turn_stage(tmp_path, monkeypatch):
    telemetry_path = tmp_path / "multi_turn.jsonl"
    monkeypatch.setattr(
        "capx.envs.trial._query_model", lambda args, prompt: _record_current_stage()
    )
    monkeypatch.setattr(
        "capx.envs.trial._build_multi_turn_decision_prompt",
        lambda *args, **kwargs: [{"role": "user", "content": "decide"}],
    )
    args = SimpleNamespace(
        model="test",
        use_legacy_multi_turn_decision_prompt=False,
    )
    config = {
        "use_visual_feedback": False,
        "use_img_differencing": False,
        "use_video_differencing": False,
        "use_wrist_camera": False,
        "use_parallel_ensemble": False,
    }

    with trial_llm_context(trial=1, telemetry_path=telemetry_path):
        decision, *_ = _handle_multi_turn_step(
            object(),
            {"full_prompt": []},
            args,
            config,
            SimpleNamespace(model="test"),
            "executed={executed_code} stdout={console_stdout} stderr={console_stderr}",
            ["print('done')"],
            1,
            {"stdout": "", "stderr": ""},
            "task",
            [],
            [],
            [],
        )

    assert decision == "finish"
    assert _telemetry_stages(telemetry_path) == ["multi_turn"]


def test_capsule_action_query_uses_capsule_action_stage(tmp_path, monkeypatch):
    telemetry_path = tmp_path / "capsule.jsonl"

    def fake_query(args, prompt):
        _record_current_stage()
        return {"content": '{"action":"finish","args":{}}', "reasoning": None}

    monkeypatch.setattr("capx.envs.trial._query_model", fake_query)
    with trial_llm_context(trial=1, telemetry_path=telemetry_path):
        _run_capsule_trial(
            env=FakeCapsuleEnv(),
            trial=1,
            args=SimpleNamespace(model="test", use_oracle_code=False, max_tokens=100),
            config={
                "capsule_control_mode": "llm_step",
                "output_dir": str(tmp_path),
                "max_capsule_steps": 1,
                "use_parallel_ensemble": False,
                "use_multimodel": False,
            },
            initial_code="x = 1\n",
        )

    assert _telemetry_stages(telemetry_path) == ["capsule_action"]


def test_capsule_trial_defaults_to_auto_forward(monkeypatch):
    expected = object()

    def fake_auto_forward(**kwargs):
        return expected

    def fail_llm_step(**kwargs):
        raise AssertionError("omitting capsule_control_mode should use auto_forward")

    monkeypatch.setattr(trial_module, "_run_capsule_auto_forward_loop", fake_auto_forward)
    monkeypatch.setattr(trial_module, "_run_capsule_llm_step_loop", fail_llm_step)

    result = _run_capsule_trial(
        env=FakeCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(),
        config={},
        initial_code="x = 1\n",
    )

    assert result is expected


def test_capsule_auto_forward_runs_groups_without_capsule_action_llm(tmp_path, monkeypatch):
    telemetry_path = tmp_path / "auto_forward.jsonl"
    env = FakeRewardDropCapsuleEnv()

    def fake_query(args, prompt):
        _record_current_stage()
        return {"content": '{"action": "finish", "args": {}}', "reasoning": None}

    monkeypatch.setattr("capx.envs.trial._query_model", fake_query)

    with trial_llm_context(trial=1, telemetry_path=telemetry_path):
        summary = _run_capsule_trial(
            env=env,
            trial=1,
            args=SimpleNamespace(model="test", use_oracle_code=False, max_tokens=100),
            config={
                "output_dir": str(tmp_path),
                "use_runtime_control": True,
                "capsule_control_mode": "auto_forward",
                "max_capsule_steps": 8,
                "capsule_max_regions_per_group": 1,
                "use_parallel_ensemble": False,
                "use_multimodel": False,
            },
            initial_code='move_to("good")\nmove_to("recover")\n',
        )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())
    telemetry_stages = _telemetry_stages(telemetry_path) if telemetry_path.exists() else []

    assert summary.sandbox_rc == 0
    assert env.api.moves == ["good", "recover"]
    assert [entry["event"]["action"] for entry in trace] == ["run_group", "run_group"]
    assert [entry["event"]["region_id"] for entry in trace] == ["group_1", "group_2"]
    assert "capsule_action" not in telemetry_stages


def test_nonprivileged_auto_forward_strict_violation_never_executes(tmp_path, monkeypatch):
    env = FakeUnsafeNonPrivilegedCapsuleEnv()

    monkeypatch.setattr(
        trial_module,
        "_query_model",
        lambda args, prompt: {
            "content": '{"action": "finish", "args": {}}',
            "reasoning": None,
        },
    )

    summary = _run_capsule_trial(
        env=env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "capsule_control_mode": "auto_forward",
            "capsule_validate_program_contract": False,
            "max_capsule_steps": 2,
        },
        initial_code="runner = close_gripper\nrunner()\n",
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_00.json").read_text())

    assert trace[0]["event"]["status"] == "invalid"
    assert trace[0]["event"]["evidence"]["strict_subset_violations"]
    assert env.api.calls == []
    assert summary.sandbox_rc == 1


def test_nonprivileged_auto_forward_reanalyzes_recovery_patch(tmp_path, monkeypatch):
    env = FakeSuccessfulNonPrivilegedCapsuleEnv()

    monkeypatch.setattr(
        trial_module,
        "_query_model",
        lambda args, prompt: {
            "content": (
                '{"action": "patch_group", "args": {"group_id": "group_1", '
                '"source": "close_gripper()\\n"}}'
            ),
            "reasoning": None,
        },
    )

    summary = _run_capsule_trial(
        env=env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "capsule_control_mode": "auto_forward",
            "capsule_validate_program_contract": False,
            "max_capsule_steps": 3,
        },
        initial_code="runner = close_gripper\nrunner()\n",
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_00.json").read_text())

    assert [entry["event"]["status"] for entry in trace] == [
        "invalid",
        "success",
        "success",
    ]
    assert env.api.calls == ["close_gripper"]
    assert summary.sandbox_rc == 0


def test_capsule_auto_forward_executes_effect_bounded_groups_in_source_order(
    tmp_path, monkeypatch
):
    env = FakeRewardDropCapsuleEnv()

    def fail_if_prompted(*args, **kwargs):
        raise AssertionError("auto_forward should not build capsule action prompts")

    monkeypatch.setattr("capx.envs.trial.build_capsule_prompt", fail_if_prompted)

    summary = _run_capsule_trial(
        env=env,
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "use_runtime_control": True,
            "capsule_control_mode": "auto_forward",
            "max_capsule_steps": 8,
            "capsule_max_regions_per_group": 1,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='move_to("good")\nmove_to("recover")\n',
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())
    metrics = _capsule_step_metrics(tmp_path / "capsule_step_metrics_trial_01.jsonl")

    assert summary.sandbox_rc == 0
    assert env.api.moves == ["good", "recover"]
    assert [entry["event"]["region_id"] for entry in trace] == ["group_1", "group_2"]
    assert [row["action"] for row in metrics] == ["run_group", "run_group"]
    assert [row["region_id"] for row in metrics] == ["group_1", "group_2"]
    assert [row["event_status"] for row in metrics] == ["success", "success"]


def test_capsule_auto_forward_never_replays_successful_side_effect_group(
    tmp_path, monkeypatch
):
    env = FakeRewardDropCapsuleEnv()
    original_segment_groups = trial_module.segment_python_code_groups

    def duplicated_first_group(*args, **kwargs):
        groups = original_segment_groups(*args, **kwargs)
        return [groups[0], groups[0]]

    def fail_if_prompted(*args, **kwargs):
        raise AssertionError("auto_forward should not build capsule action prompts")

    def fake_query(args, prompt):
        return {"content": '{"action": "finish", "args": {}}', "reasoning": None}

    monkeypatch.setattr("capx.envs.trial.segment_python_code_groups", duplicated_first_group)
    monkeypatch.setattr("capx.envs.trial.build_capsule_prompt", fail_if_prompted)
    monkeypatch.setattr("capx.envs.trial._query_model", fake_query)

    summary = _run_capsule_trial(
        env=env,
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "use_runtime_control": True,
            "capsule_control_mode": "auto_forward",
            "max_capsule_steps": 8,
            "capsule_max_regions_per_group": 1,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='move_to("good")\n',
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())
    metrics = _capsule_step_metrics(tmp_path / "capsule_step_metrics_trial_01.jsonl")

    assert summary.sandbox_rc == 1
    assert env.api.moves == ["good"]
    assert [entry["event"]["status"] for entry in trace] == ["success", "invalid", "success"]
    assert trace[1]["event"]["region_id"] == "group_1"
    assert "already executed robot-side-effect code" in trace[1]["event"]["message"]
    assert [row["event_status"] for row in metrics] == ["success", "invalid", "success"]


def test_capsule_auto_forward_rejects_region_granularity(tmp_path):
    try:
        _run_capsule_trial(
            env=FakeCapsuleEnv(),
            trial=1,
            args=SimpleNamespace(model="test", use_oracle_code=False),
            config={
                "output_dir": str(tmp_path),
                "use_runtime_control": True,
                "capsule_control_mode": "auto_forward",
                "capsule_execution_granularity": "region",
                "max_capsule_steps": 4,
                "use_parallel_ensemble": False,
                "use_multimodel": False,
            },
            initial_code='move_to([1, 2, 3])\n',
        )
    except ValueError as exc:
        assert "auto_forward" in str(exc)
        assert "effect-bounded execution units" in str(exc)
    else:
        raise AssertionError(
            "auto_forward should reject non-effect-bounded execution granularity"
        )


def test_capsule_auto_forward_initial_code_query_does_not_query_capsule_action(
    tmp_path, monkeypatch
):
    telemetry_path = tmp_path / "auto_forward_initial.jsonl"
    env = FakeCapsuleEnv()

    def fake_query(args, prompt):
        _record_current_stage()
        return {
            "content": '```python\nmove_to([1, 2, 3])\n```',
            "reasoning": None,
        }

    monkeypatch.setattr("capx.envs.trial._query_model", fake_query)

    with trial_llm_context(trial=1, telemetry_path=telemetry_path):
        summary = _run_capsule_trial(
            env=env,
            trial=1,
            args=SimpleNamespace(model="test", use_oracle_code=False, max_tokens=100),
            config={
                "output_dir": str(tmp_path),
                "use_runtime_control": True,
                "capsule_control_mode": "auto_forward",
                "max_capsule_steps": 4,
                "use_parallel_ensemble": False,
                "use_multimodel": False,
            },
        )

    assert summary.sandbox_rc == 0
    assert env.api.moved is True
    assert _telemetry_stages(telemetry_path) == ["initial_code"]


def test_capsule_auto_forward_stops_after_failed_group(tmp_path, monkeypatch):
    env = FakeRewardDropCapsuleEnv()

    def fake_query(args, prompt):
        return {"content": '{"action": "finish", "args": {}}', "reasoning": None}

    monkeypatch.setattr("capx.envs.trial._query_model", fake_query)

    summary = _run_capsule_trial(
        env=env,
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "use_runtime_control": True,
            "capsule_control_mode": "auto_forward",
            "max_capsule_steps": 4,
            "capsule_max_regions_per_group": 1,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='raise RuntimeError("boom")\nmove_to("recover")\n',
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 1
    assert env.api.moves == []
    assert [entry["event"]["action"] for entry in trace] == ["run_group", "finish"]
    assert trace[0]["event"]["status"] == "failed"


def test_capsule_auto_forward_calls_llm_only_after_group_failure(tmp_path, monkeypatch):
    telemetry_path = tmp_path / "auto_forward_recovery.jsonl"
    prompts = []
    queries = []

    def fail_if_action_prompted(*args, **kwargs):
        raise AssertionError("auto_forward should not build capsule action prompts")

    def fake_recovery_prompt(**kwargs):
        prompts.append(kwargs)
        return [{"role": "user", "content": [{"type": "text", "text": "recover"}]}]

    def fake_query(args, prompt):
        _record_current_stage()
        queries.append(prompt)
        return {"content": '{"action": "finish", "args": {}}', "reasoning": None}

    monkeypatch.setattr("capx.envs.trial.build_capsule_prompt", fail_if_action_prompted)
    monkeypatch.setattr("capx.envs.trial.build_capsule_recovery_prompt", fake_recovery_prompt)
    monkeypatch.setattr("capx.envs.trial._query_model", fake_query)

    with trial_llm_context(trial=1, telemetry_path=telemetry_path):
        summary = _run_capsule_trial(
            env=FakeRewardDropCapsuleEnv(),
            trial=1,
            args=SimpleNamespace(model="test", use_oracle_code=False, max_tokens=100),
            config={
                "output_dir": str(tmp_path),
                "use_runtime_control": True,
                "capsule_control_mode": "auto_forward",
                "max_capsule_steps": 4,
                "capsule_max_regions_per_group": 1,
                "use_parallel_ensemble": False,
                "use_multimodel": False,
            },
            initial_code='move_to("good")\nraise RuntimeError("boom")\n',
        )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 1
    assert [entry["event"]["status"] for entry in trace] == ["success", "failed", "success"]
    assert [entry["event"]["action"] for entry in trace] == ["run_group", "run_group", "finish"]
    assert len(prompts) == 1
    assert prompts[0]["failed_unit"].group_id == "group_2"
    assert len(queries) == 1
    assert _telemetry_stages(telemetry_path) == ["capsule_recovery"]


def test_capsule_auto_forward_append_recovery_continues_to_success(tmp_path, monkeypatch):
    telemetry_path = tmp_path / "auto_forward_append_recovery.jsonl"
    env = FakeRewardDropCapsuleEnv()
    queries = []

    def fail_if_action_prompted(*args, **kwargs):
        raise AssertionError("auto_forward should not build capsule action prompts")

    def fake_query(args, prompt):
        _record_current_stage()
        queries.append(prompt)
        return {
            "content": (
                '{"action": "append_recovery", "args": {"source": '
                '"state = get_observation()\\nmove_to(\\"recover\\")"}}'
            ),
            "reasoning": None,
        }

    monkeypatch.setattr("capx.envs.trial.build_capsule_prompt", fail_if_action_prompted)
    monkeypatch.setattr("capx.envs.trial._query_model", fake_query)

    with trial_llm_context(trial=1, telemetry_path=telemetry_path):
        summary = _run_capsule_trial(
            env=env,
            trial=1,
            args=SimpleNamespace(model="test", use_oracle_code=False, max_tokens=100),
            config={
                "output_dir": str(tmp_path),
                "use_runtime_control": True,
                "capsule_control_mode": "auto_forward",
                "max_capsule_steps": 6,
                "capsule_max_regions_per_group": 1,
                "use_parallel_ensemble": False,
                "use_multimodel": False,
            },
            initial_code='move_to("good")\nraise RuntimeError("boom")\n',
        )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 0
    assert env.api.observed is True
    assert env.api.moves == ["good", "recover"]
    assert [entry["event"]["action"] for entry in trace[:3]] == [
        "run_group",
        "run_group",
        "append_recovery",
    ]
    assert [entry["event"]["status"] for entry in trace[:3]] == [
        "success",
        "failed",
        "success",
    ]
    assert trace[-1]["event"]["action"] == "run_group"
    assert trace[-1]["event"]["status"] == "success"
    assert len(queries) == 1
    assert _telemetry_stages(telemetry_path) == ["capsule_recovery"]


def test_capsule_auto_forward_terminal_incomplete_appends_recovery(
    tmp_path, monkeypatch
):
    telemetry_path = tmp_path / "auto_forward_terminal_recovery.jsonl"
    env = FakeRewardDropCapsuleEnv()
    queries = []

    def fail_if_action_prompted(*args, **kwargs):
        raise AssertionError("auto_forward should not build capsule action prompts")

    def fake_query(args, prompt):
        _record_current_stage()
        queries.append(prompt)
        return {
            "content": (
                '{"action": "append_recovery", "args": {"source": '
                '"state = get_observation()\\nmove_to(\\"recover\\")"}}'
            ),
            "reasoning": None,
        }

    monkeypatch.setattr("capx.envs.trial.build_capsule_prompt", fail_if_action_prompted)
    monkeypatch.setattr("capx.envs.trial._query_model", fake_query)

    with trial_llm_context(trial=1, telemetry_path=telemetry_path):
        summary = _run_capsule_trial(
            env=env,
            trial=1,
            args=SimpleNamespace(model="test", use_oracle_code=False, max_tokens=100),
            config={
                "output_dir": str(tmp_path),
                "use_runtime_control": True,
                "capsule_control_mode": "auto_forward",
                "max_capsule_steps": 6,
                "capsule_max_regions_per_group": 1,
                "use_parallel_ensemble": False,
                "use_multimodel": False,
            },
            initial_code='move_to("good")\n',
        )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 0
    assert env.api.observed is True
    assert env.api.moves == ["good", "recover"]
    assert [entry["event"]["action"] for entry in trace[:3]] == [
        "run_group",
        "append_recovery",
        "run_group",
    ]
    assert all(entry["event"]["action"] == "run_group" for entry in trace[2:])
    assert trace[1]["state_before"]["reward"] < 1.0
    assert len(queries) == 1
    assert _telemetry_stages(telemetry_path) == ["capsule_recovery"]


def test_capsule_auto_forward_terminal_recovery_accepts_raw_python_content(
    tmp_path, monkeypatch
):
    telemetry_path = tmp_path / "auto_forward_terminal_recovery_raw_python.jsonl"
    env = FakeRewardDropCapsuleEnv()

    def fake_query(args, prompt):
        _record_current_stage()
        return {
            "content": 'state = get_observation()\nmove_to("recover")\n',
            "reasoning": None,
        }

    monkeypatch.setattr("capx.envs.trial._query_model", fake_query)

    with trial_llm_context(trial=1, telemetry_path=telemetry_path):
        summary = _run_capsule_trial(
            env=env,
            trial=1,
            args=SimpleNamespace(model="test", use_oracle_code=False, max_tokens=100),
            config={
                "output_dir": str(tmp_path),
                "use_runtime_control": True,
                "capsule_control_mode": "auto_forward",
                "max_capsule_steps": 6,
                "capsule_max_regions_per_group": 1,
                "use_parallel_ensemble": False,
                "use_multimodel": False,
            },
            initial_code='move_to("good")\n',
        )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 0
    assert env.api.observed is True
    assert env.api.moves == ["good", "recover"]
    assert trace[1]["event"]["action"] == "append_recovery"
    assert trace[1]["event"]["status"] == "success"
    assert trace[2]["event"]["action"] == "run_group"
    assert _telemetry_stages(telemetry_path) == ["capsule_recovery"]


def test_capsule_auto_forward_append_recovery_runs_before_future_groups(
    tmp_path, monkeypatch
):
    telemetry_path = tmp_path / "auto_forward_append_recovery_before_future.jsonl"
    env = FakeRewardDropCapsuleEnv()

    def fake_query(args, prompt):
        _record_current_stage()
        return {
            "content": (
                '{"action": "append_recovery", "args": {"source": '
                '"state = get_observation()\\nmove_to(\\"recover\\")"}}'
            ),
            "reasoning": None,
        }

    monkeypatch.setattr("capx.envs.trial._query_model", fake_query)

    with trial_llm_context(trial=1, telemetry_path=telemetry_path):
        summary = _run_capsule_trial(
            env=env,
            trial=1,
            args=SimpleNamespace(model="test", use_oracle_code=False, max_tokens=100),
            config={
                "output_dir": str(tmp_path),
                "use_runtime_control": True,
                "capsule_control_mode": "auto_forward",
                "max_capsule_steps": 6,
                "capsule_max_regions_per_group": 1,
                "use_parallel_ensemble": False,
                "use_multimodel": False,
            },
            initial_code='move_to("good")\nraise RuntimeError("boom")\nmove_to("future")\n',
        )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 0
    assert env.api.observed is True
    assert env.api.moves == ["good", "recover"]
    assert [entry["event"]["action"] for entry in trace[:4]] == [
        "run_group",
        "run_group",
        "append_recovery",
        "run_group",
    ]
    assert trace[3]["action"]["args"]["group_id"] == "group_3"
    assert "future" not in env.api.moves
    assert _telemetry_stages(telemetry_path) == ["capsule_recovery"]


def test_capsule_auto_forward_append_recovery_isolated_from_bounded_future_groups(
    tmp_path, monkeypatch
):
    telemetry_path = tmp_path / "auto_forward_append_recovery_bounded_future.jsonl"
    env = FakeRewardDropCapsuleEnv()
    future_source = "\n".join(f'move_to("future_{idx}")' for idx in range(10))

    def fake_query(args, prompt):
        _record_current_stage()
        return {
            "content": (
                '{"action": "append_recovery", "args": {"source": '
                '"state = get_observation()\\nmove_to(\\"recover\\")"}}'
            ),
            "reasoning": None,
        }

    monkeypatch.setattr("capx.envs.trial._query_model", fake_query)

    with trial_llm_context(trial=1, telemetry_path=telemetry_path):
        summary = _run_capsule_trial(
            env=env,
            trial=1,
            args=SimpleNamespace(model="test", use_oracle_code=False, max_tokens=100),
            config={
                "output_dir": str(tmp_path),
                "use_runtime_control": True,
                "capsule_control_mode": "auto_forward",
                "max_capsule_steps": 6,
                "capsule_max_regions_per_group": 1,
                "use_parallel_ensemble": False,
                "use_multimodel": False,
            },
            initial_code=(
                'move_to("good")\n'
                'raise RuntimeError("boom")\n'
                f"{future_source}\n"
            ),
        )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 0
    assert env.api.observed is True
    assert env.api.moves == ["good", "recover"]
    assert not any(str(move).startswith("future_") for move in env.api.moves)
    assert trace[3]["action"]["args"]["group_id"] == "group_3"
    assert "group_3" in [
        entry["action"]["args"].get("group_id")
        for entry in trace
        if entry["event"]["action"] == "run_group"
    ]
    assert _telemetry_stages(telemetry_path) == ["capsule_recovery"]


def test_capsule_auto_forward_stops_when_recovery_action_is_invalid(tmp_path, monkeypatch):
    telemetry_path = tmp_path / "auto_forward_invalid_recovery.jsonl"
    env = FakeRewardDropCapsuleEnv()

    def fail_if_action_prompted(*args, **kwargs):
        raise AssertionError("auto_forward should not build capsule action prompts")

    def fake_query(args, prompt):
        _record_current_stage()
        return {"content": '{"action": "not_supported", "args": {}}', "reasoning": None}

    monkeypatch.setattr("capx.envs.trial.build_capsule_prompt", fail_if_action_prompted)
    monkeypatch.setattr("capx.envs.trial._query_model", fake_query)

    with trial_llm_context(trial=1, telemetry_path=telemetry_path):
        summary = _run_capsule_trial(
            env=env,
            trial=1,
            args=SimpleNamespace(model="test", use_oracle_code=False, max_tokens=100),
            config={
                "output_dir": str(tmp_path),
                "use_runtime_control": True,
                "capsule_control_mode": "auto_forward",
                "max_capsule_steps": 6,
                "capsule_max_regions_per_group": 1,
                "use_parallel_ensemble": False,
                "use_multimodel": False,
            },
            initial_code='move_to("good")\nraise RuntimeError("boom")\nmove_to("recover")\n',
        )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 1
    assert env.api.moves == ["good"]
    assert [entry["event"]["status"] for entry in trace] == ["success", "failed", "invalid"]
    assert "Unsupported runtime action" in trace[2]["event"]["message"]
    assert _telemetry_stages(telemetry_path) == ["capsule_recovery"]


def test_capsule_auto_forward_rejects_recovery_run_group_action(tmp_path, monkeypatch):
    telemetry_path = tmp_path / "auto_forward_recovery_run_group.jsonl"
    env = FakeRewardDropCapsuleEnv()

    def fake_query(args, prompt):
        _record_current_stage()
        return {
            "content": '{"action": "run_group", "args": {"group_id": "group_2"}}',
            "reasoning": None,
        }

    monkeypatch.setattr("capx.envs.trial._query_model", fake_query)

    with trial_llm_context(trial=1, telemetry_path=telemetry_path):
        summary = _run_capsule_trial(
            env=env,
            trial=1,
            args=SimpleNamespace(model="test", use_oracle_code=False, max_tokens=100),
            config={
                "output_dir": str(tmp_path),
                "use_runtime_control": True,
                "capsule_control_mode": "auto_forward",
                "max_capsule_steps": 6,
                "capsule_max_regions_per_group": 1,
                "use_parallel_ensemble": False,
                "use_multimodel": False,
            },
            initial_code='raise RuntimeError("boom")\nmove_to("recover")\n',
        )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 1
    assert env.api.moves == []
    assert [entry["event"]["status"] for entry in trace] == ["failed", "invalid"]
    assert "not allowed during recovery" in trace[1]["event"]["message"]
    assert _telemetry_stages(telemetry_path) == ["capsule_recovery"]


def test_capsule_auto_forward_recovery_cannot_replay_failed_side_effect_group(
    tmp_path, monkeypatch
):
    telemetry_path = tmp_path / "auto_forward_recovery_replay_failed_side_effect.jsonl"
    env = FakeRewardDropCapsuleEnv()

    def single_failing_side_effect_group(source, regions, **kwargs):
        return [
            CodeRegionGroup(
                group_id="group_1",
                start_line=1,
                end_line=2,
                source=source,
                region_ids=[region.region_id for region in regions],
                primitive_calls=["move_to"],
                defined_names=[],
                used_names=["move_to"],
                has_robot_side_effect=True,
            )
        ]

    def fake_query(args, prompt):
        _record_current_stage()
        return {
            "content": (
                '{"action": "patch_group", "args": {"group_id": "group_1", '
                '"source": "move_to(\\"recover\\")"}}'
            ),
            "reasoning": None,
        }

    monkeypatch.setattr(
        "capx.envs.trial.segment_python_code_groups",
        single_failing_side_effect_group,
    )
    monkeypatch.setattr("capx.envs.trial._query_model", fake_query)

    with trial_llm_context(trial=1, telemetry_path=telemetry_path):
        summary = _run_capsule_trial(
            env=env,
            trial=1,
            args=SimpleNamespace(model="test", use_oracle_code=False, max_tokens=100),
            config={
                "output_dir": str(tmp_path),
                "use_runtime_control": True,
                "capsule_control_mode": "auto_forward",
                "max_capsule_steps": 6,
                "capsule_max_regions_per_group": 2,
                "use_parallel_ensemble": False,
                "use_multimodel": False,
            },
            initial_code='move_to("good")\nraise RuntimeError("boom")\n',
        )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 1
    assert env.api.moves == ["good"]
    assert [entry["event"]["status"] for entry in trace] == ["failed", "invalid"]
    assert "already executed robot-side-effect code" in trace[1]["event"]["message"]
    assert _telemetry_stages(telemetry_path) == ["capsule_recovery"]


def test_capsule_auto_forward_recovery_cannot_patch_future_group(tmp_path, monkeypatch):
    telemetry_path = tmp_path / "auto_forward_recovery_future_patch.jsonl"
    env = FakeRewardDropCapsuleEnv()

    def fake_query(args, prompt):
        _record_current_stage()
        return {
            "content": (
                '{"action": "patch_group", "args": {"group_id": "group_2", '
                '"source": "move_to(\\"recover\\")"}}'
            ),
            "reasoning": None,
        }

    monkeypatch.setattr("capx.envs.trial._query_model", fake_query)

    with trial_llm_context(trial=1, telemetry_path=telemetry_path):
        summary = _run_capsule_trial(
            env=env,
            trial=1,
            args=SimpleNamespace(model="test", use_oracle_code=False, max_tokens=100),
            config={
                "output_dir": str(tmp_path),
                "use_runtime_control": True,
                "capsule_control_mode": "auto_forward",
                "max_capsule_steps": 6,
                "capsule_max_regions_per_group": 1,
                "use_parallel_ensemble": False,
                "use_multimodel": False,
            },
            initial_code='raise RuntimeError("boom")\nmove_to("recover")\n',
        )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 1
    assert env.api.moves == []
    assert [entry["event"]["status"] for entry in trace] == ["failed", "invalid"]
    assert "must target the failed group" in trace[1]["event"]["message"]
    assert _telemetry_stages(telemetry_path) == ["capsule_recovery"]


def test_capsule_auto_forward_stops_after_reward_success(tmp_path):
    env = FakeRewardDropCapsuleEnv()

    summary = _run_capsule_trial(
        env=env,
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "use_runtime_control": True,
            "capsule_control_mode": "auto_forward",
            "max_capsule_steps": 4,
            "capsule_max_regions_per_group": 1,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='move_to("recover")\nmove_to("bad")\n',
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 0
    assert env.api.moves == ["recover"]
    assert [entry["event"]["action"] for entry in trace] == ["run_group"]
    assert trace[0]["state_after"]["reward"] == 1.0


def test_capsule_auto_forward_blocks_side_effect_after_reward_drop_from_best(tmp_path, monkeypatch):
    env = FakeRewardDropCapsuleEnv()

    def fake_query(args, prompt):
        return {"content": '{"action": "finish", "args": {}}', "reasoning": None}

    monkeypatch.setattr("capx.envs.trial._query_model", fake_query)

    summary = _run_capsule_trial(
        env=env,
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "use_runtime_control": True,
            "capsule_control_mode": "auto_forward",
            "max_capsule_steps": 4,
            "capsule_max_regions_per_group": 1,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='move_to("good")\nmove_to("bad")\nmove_to("worse")\n',
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 1
    assert env.api.moves == ["good", "bad"]
    assert trace[2]["event"]["status"] == "invalid"
    assert trace[2]["event"]["evidence"]["safety_failure"] == (
        "reward_drop_guard"
    )
    assert "reward dropped" in trace[2]["event"]["message"]
    assert "append_recovery" in trace[2]["event"]["message"]


def test_capsule_llm_step_mode_keeps_existing_action_loop(tmp_path):
    env = FakeCapsuleEnv()

    summary = _run_capsule_trial(
        env=env,
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "use_runtime_control": True,
            "capsule_control_mode": "llm_step",
            "scripted_actions": [
                {"action": "run_group", "args": {"group_id": "group_1"}},
                {"action": "finish", "args": {}},
            ],
            "max_capsule_steps": 3,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='pose = get_pose("cube")\nmove_to(pose)\n',
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 0
    assert env.api.moved is True
    assert [entry["event"]["action"] for entry in trace] == ["run_group", "finish"]


def test_capsule_llm_step_program_contract_blocks_effects_until_patch(tmp_path):
    env = FakeCustomMoveCapsuleEnv()
    source = (
        "def move_cube():\n"
        '    custom_move("blocked")\n'
        "\n"
        "move_cube()\n"
    )

    trial_module._run_capsule_llm_step_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 4,
            "capsule_validate_program_contract": True,
        },
        initial_code=source,
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {
                "action": "patch_group",
                "args": {
                    "group_id": "group_1",
                    "source": 'custom_move("repaired")\n',
                },
            },
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_00.json").read_text())
    metrics = _capsule_step_metrics(tmp_path / "capsule_step_metrics_trial_00.jsonl")
    prompts = json.loads((tmp_path / "capsule_prompts_trial_00.json").read_text())

    assert [entry["event"]["status"] for entry in trace] == [
        "invalid",
        "success",
        "success",
        "success",
    ]
    assert env.api.moves == ["repaired"]
    assert "program_contract_violations" in trace[0]["event"]["evidence"]
    assert trace[0]["event"]["evidence"]["program_contract_violations"][0][
        "code"
    ] == "effectful_helper"
    assert metrics[0]["program_contract_valid"] is False
    assert metrics[0]["program_contract_violation_count"] >= 1
    assert "effectful_helper" in metrics[0]["program_contract_violation_codes"]
    assert metrics[1]["program_contract_valid"] is False
    assert metrics[2]["program_contract_valid"] is True
    assert metrics[2]["program_contract_violation_count"] == 0
    assert metrics[2]["program_contract_violation_codes"] == []
    assert "Capsule-ready program contract violations" in str(prompts[0])
    assert "Capsule-ready program contract violations" in str(prompts[1])
    assert "Capsule-ready program contract violations" not in str(prompts[2])


def test_capsule_llm_step_program_contract_flag_false_preserves_execution(tmp_path):
    env = FakePrivilegedCapsuleEnv()

    trial_module._run_capsule_llm_step_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 1,
            "capsule_validate_program_contract": False,
        },
        initial_code=(
            "runner = close_gripper\n"
            "runner()\n"
        ),
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_00.json").read_text())
    metrics = _capsule_step_metrics(tmp_path / "capsule_step_metrics_trial_00.jsonl")

    assert trace[0]["event"]["status"] == "success"
    assert env.api.calls == ["close_gripper"]
    assert metrics[0]["program_contract_valid"] is True
    assert metrics[0]["program_contract_violation_count"] == 0
    assert metrics[0]["program_contract_violation_codes"] == []


@pytest.mark.parametrize(
    "source",
    [
        (
            "runner = close_gripper\n"
            "if True:\n"
            "    runner = close_gripper\n"
            "runner()\n"
        ),
        (
            "import sys\n"
            "print(sys._getframe().f_locals)\n"
        ),
        "close_gripper.__call__()\n",
    ],
)
def test_nonprivileged_llm_step_enforces_strict_subset_when_contract_flag_false(
    tmp_path, source
):
    env = FakeUnsafeNonPrivilegedCapsuleEnv()

    trial_module._run_capsule_llm_step_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 1,
            "capsule_validate_program_contract": False,
        },
        initial_code=source,
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
        ],
    )

    trace_text = (tmp_path / "capsule_trace_trial_00.json").read_text()
    trace = json.loads(trace_text)
    prompts = (tmp_path / "capsule_prompts_trial_00.json").read_text()
    metrics = _capsule_step_metrics(tmp_path / "capsule_step_metrics_trial_00.jsonl")

    assert trace[0]["event"]["status"] == "invalid"
    assert trace[0]["event"]["evidence"]["strict_subset_violations"]
    assert env.api.calls == []
    assert "SIM_TRUTH_SENTINEL" not in trace_text
    assert "SIM_TRUTH_SENTINEL" not in prompts
    assert "strict_subset_violation" in prompts
    assert metrics[0]["strict_subset_valid"] is False
    assert metrics[0]["strict_subset_violation_count"] >= 1
    assert metrics[0]["program_contract_valid"] is False
    assert "strict_subset_violation" in metrics[0][
        "program_contract_violation_codes"
    ]


def test_nonprivileged_llm_step_reanalyzes_strict_source_after_patch(tmp_path):
    env = FakeSuccessfulNonPrivilegedCapsuleEnv()

    summary = trial_module._run_capsule_llm_step_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 3,
            "capsule_validate_program_contract": False,
        },
        initial_code="runner = close_gripper\nrunner()\n",
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {
                "action": "patch_group",
                "args": {
                    "group_id": "group_1",
                    "source": "close_gripper()\n",
                },
            },
            {"action": "run_group", "args": {"group_id": "group_1"}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_00.json").read_text())
    metrics = _capsule_step_metrics(tmp_path / "capsule_step_metrics_trial_00.jsonl")

    assert [entry["event"]["status"] for entry in trace] == [
        "invalid",
        "success",
        "success",
    ]
    assert env.api.calls == ["close_gripper"]
    assert [row["strict_subset_valid"] for row in metrics] == [False, False, True]
    assert metrics[-1]["program_contract_valid"] is True
    assert summary.sandbox_rc == 0


def test_llm_step_invalid_action_is_recoverable_before_task_success(
    tmp_path, monkeypatch
):
    env = FakeSuccessfulNonPrivilegedCapsuleEnv()
    responses = iter(
        [
            {"content": '{"action": "not_supported", "args": {}}'},
            {
                "content": (
                    '{"action": "run_group", "args": {"group_id": "group_1"}}'
                )
            },
        ]
    )
    monkeypatch.setattr(
        trial_module,
        "_query_model",
        lambda args, prompt: next(responses),
    )

    summary = trial_module._run_capsule_llm_step_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 2,
            "capsule_validate_program_contract": False,
        },
        initial_code="close_gripper()\n",
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_00.json").read_text())

    assert [entry["event"]["status"] for entry in trace] == [
        "invalid",
        "success",
    ]
    assert env.api.calls == ["close_gripper"]
    assert summary.sandbox_rc == 0


def test_llm_step_safety_failure_stays_failed_after_append_recovery(tmp_path):
    env = FakeSuccessfulRecoveryNonPrivilegedCapsuleEnv()

    summary = trial_module._run_capsule_llm_step_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 4,
            "capsule_validate_program_contract": False,
        },
        initial_code="close_gripper()\n",
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {
                "action": "append_recovery",
                "args": {
                    "source": "state = get_observation()\nresult = len(state)\n",
                },
            },
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_00.json").read_text())

    assert [entry["event"]["status"] for entry in trace] == [
        "success",
        "invalid",
        "success",
        "success",
    ]
    assert trace[1]["event"]["evidence"]["safety_failure"] == (
        "side_effect_replay"
    )
    assert env.api.calls == ["close_gripper"]
    assert summary.reward == 1.0
    assert summary.sandbox_rc == 1


def test_nonprivileged_strict_preflight_skips_segmentation_for_helper_flood(
    tmp_path, monkeypatch
):
    env = FakeUnsafeNonPrivilegedCapsuleEnv()
    source = "\n".join(f"def helper_{index}():\n    return {index}" for index in range(257))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("strict preflight must run before source segmentation")

    monkeypatch.setattr(trial_module, "segment_python_code", fail_if_called)
    monkeypatch.setattr(trial_module, "segment_python_code_groups", fail_if_called)

    trial_module._run_capsule_llm_step_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 1,
            "capsule_validate_program_contract": False,
        },
        initial_code=source,
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_00.json").read_text())

    assert trace[0]["event"]["status"] == "invalid"
    violation = trace[0]["event"]["evidence"]["strict_subset_violations"][0]
    assert violation["region_ids"] == ["region_1"]
    assert violation["group_ids"] == ["group_1"]
    assert env.api.calls == []


def test_nonprivileged_capsule_globals_block_raw_env_truth_without_prompt_leak(
    tmp_path,
):
    env = FakeUnsafeNonPrivilegedCapsuleEnv()

    trial_module._run_capsule_llm_step_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 2,
            "capsule_validate_program_contract": False,
        },
        initial_code="print(env._get_all_object_poses())\n",
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "finish", "args": {}},
        ],
    )

    trace_text = (tmp_path / "capsule_trace_trial_00.json").read_text()
    prompt_text = (tmp_path / "capsule_prompts_trial_00.json").read_text()
    trace = json.loads(trace_text)

    assert trace[0]["event"]["status"] == "invalid"
    assert trace[0]["event"]["evidence"]["strict_subset_violations"]
    assert "SIM_TRUTH_SENTINEL" not in trace_text
    assert "SIM_TRUTH_SENTINEL" not in prompt_text
    assert env.api.calls == []


def test_nonprivileged_public_api_is_traced_and_no_replay_guarded(tmp_path):
    env = FakeUnsafeNonPrivilegedCapsuleEnv()

    trial_module._run_capsule_llm_step_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 2,
            "capsule_validate_program_contract": False,
        },
        initial_code="close_gripper()\n",
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "run_group", "args": {"group_id": "group_1"}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_00.json").read_text())

    assert [entry["event"]["status"] for entry in trace] == [
        "success",
        "invalid",
    ]
    assert trace[0]["trace_events"][0]["name"] == "close_gripper"
    assert env.api.calls == ["close_gripper"]


@pytest.mark.parametrize(
    "source",
    [
        "alias = close_gripper\nalias()\n",
        "alias = lambda: (close_gripper(), close_gripper())\nalias()\n",
        'globals()["close_gripper"]()\n',
        'APIS["fake"].close_gripper()\n',
        "close_gripper.__wrapped__()\n",
    ],
)
def test_capsule_contract_blocks_alias_and_dynamic_effects_before_execution(
    tmp_path,
    source,
):
    env = FakeUnsafeNonPrivilegedCapsuleEnv()

    trial_module._run_capsule_llm_step_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 1,
            "capsule_validate_program_contract": True,
        },
        initial_code=source,
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_00.json").read_text())

    assert trace[0]["event"]["status"] == "invalid"
    assert trace[0]["event"]["evidence"]["program_contract_violations"]
    assert env.api.calls == []


@pytest.mark.parametrize(
    "source",
    [
        (
            "print(close_gripper.__closure__[0].cell_contents.__self__.env"
            "._get_all_object_poses())\n"
        ),
        "close_gripper.__self__\n",
        "vars(close_gripper)\n",
        "dir(close_gripper)\n",
        "(lambda: (close_gripper(), close_gripper()))()\n",
        "list(map(lambda _: close_gripper(), [1]))\n",
        "(alias,) = (close_gripper,)\nalias()\n",
        "(alias := close_gripper)()\n",
    ],
)
def test_capsule_contract_blocks_reflection_lambda_and_destructuring(
    tmp_path,
    source,
):
    env = FakeUnsafeNonPrivilegedCapsuleEnv()
    env.api.env = env.low_level_env

    trial_module._run_capsule_llm_step_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 1,
            "capsule_validate_program_contract": True,
        },
        initial_code=source,
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
        ],
    )

    trace_text = (tmp_path / "capsule_trace_trial_00.json").read_text()
    prompt_text = (tmp_path / "capsule_prompts_trial_00.json").read_text()
    trace = json.loads(trace_text)

    assert trace[0]["event"]["status"] == "invalid"
    assert trace[0]["event"]["evidence"]["program_contract_violations"]
    assert "SIM_TRUTH_SENTINEL" not in trace_text
    assert "SIM_TRUTH_SENTINEL" not in prompt_text
    assert env.api.calls == []


def test_capsule_contract_blocks_effectful_class_definition(tmp_path):
    env = FakeUnsafeNonPrivilegedCapsuleEnv()

    trial_module._run_capsule_llm_step_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 1,
            "capsule_validate_program_contract": True,
        },
        initial_code=(
            "class Unsafe:\n"
            "    def act(self):\n"
            "        close_gripper()\n"
            "obj = Unsafe()\n"
            "obj.act()\n"
        ),
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_00.json").read_text())

    assert trace[0]["event"]["status"] == "invalid"
    assert env.api.calls == []


@pytest.mark.parametrize(
    "source",
    [
        'runner = eval\nrunner("close_gripper()")\n',
        'runner = exec\nrunner("close_gripper()")\n',
        'r = eval\ns = r\ns("close_gripper()")\n',
        (
            "ga = getattr\n"
            "ga(ga(close_gripper, '__closure__')[0], 'cell_contents')()\n"
        ),
        (
            "ga = getattr\n"
            "raw = ga(ga(close_gripper, '__closure__')[0], 'cell_contents')\n"
            "api = ga(raw, '__self__')\n"
            "print(api.env._get_all_object_poses())\n"
        ),
    ],
)
def test_capsule_contract_blocks_forbidden_builtin_aliases(tmp_path, source):
    env = FakeUnsafeNonPrivilegedCapsuleEnv()
    env.api.env = env.low_level_env

    trial_module._run_capsule_llm_step_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 1,
            "capsule_validate_program_contract": True,
        },
        initial_code=source,
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
        ],
    )

    trace_text = (tmp_path / "capsule_trace_trial_00.json").read_text()
    prompt_text = (tmp_path / "capsule_prompts_trial_00.json").read_text()
    trace = json.loads(trace_text)

    assert trace[0]["event"]["status"] == "invalid"
    assert trace[0]["event"]["evidence"]["program_contract_violations"]
    assert "SIM_TRUTH_SENTINEL" not in trace_text
    assert "SIM_TRUTH_SENTINEL" not in prompt_text
    assert env.api.calls == []


def test_program_contract_guard_only_blocks_side_effect_execution_units():
    violation = ProgramContractViolation(
        code="effectful_helper",
        message="helper is effectful",
        start_line=1,
        end_line=2,
    )
    blocked = _program_contract_guard_event(
        RuntimeAction("run_group", {"group_id": "group_2"}),
        [violation],
        {"region_2"},
        {"group_2"},
    )

    assert blocked is not None
    assert blocked.status == "invalid"
    assert blocked.region_id == "group_2"
    assert blocked.evidence["program_contract_violations"] == [violation.to_dict()]
    for action in [
        RuntimeAction("run_group", {"group_id": "group_1"}),
        RuntimeAction("patch_group", {"group_id": "group_2", "source": "x = 1"}),
        RuntimeAction("inspect_trace", {}),
        RuntimeAction("finish", {}),
        RuntimeAction("append_recovery", {"source": "x = 1"}),
    ]:
        assert (
            _program_contract_guard_event(
                action,
                [violation],
                {"region_2"},
                {"group_2"},
            )
            is None
        )


def test_capsule_contract_region_guard_tracks_transitive_helper_effects(tmp_path):
    env = FakeGripperCapsuleEnv()

    trial_module._run_capsule_llm_step_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 4,
            "capsule_max_regions_per_group": 1,
            "capsule_validate_program_contract": True,
        },
        initial_code=(
            "def move():\n"
            "    close_gripper()\n"
            "move()\n"
        ),
        scripted_actions=[
            {"action": "run_region", "args": {"region_id": "region_1"}},
            {"action": "run_region", "args": {"region_id": "region_2"}},
            {"action": "resume_from_region", "args": {"region_id": "region_2"}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_00.json").read_text())

    assert [entry["event"]["status"] for entry in trace] == [
        "success",
        "invalid",
        "invalid",
        "success",
    ]
    assert trace[0]["event"]["region_id"] == "region_1"
    assert trace[1]["event"]["region_id"] == "region_2"
    assert trace[2]["event"]["region_id"] == "region_2"
    assert trace[1]["event"]["evidence"]["program_contract_violations"]
    assert trace[2]["event"]["evidence"]["program_contract_violations"]
    assert env.api.calls == []


def test_capsule_contract_guard_blocks_effectful_comprehension(tmp_path):
    env = FakeGripperCapsuleEnv()

    trial_module._run_capsule_llm_step_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 2,
            "capsule_validate_program_contract": True,
        },
        initial_code="[close_gripper() for _ in range(2)]\n",
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_00.json").read_text())

    assert trace[0]["event"]["status"] == "invalid"
    assert trace[0]["event"]["evidence"]["program_contract_violations"][0][
        "code"
    ] == "effectful_control_flow"
    assert env.api.calls == []


def test_capsule_no_replay_ledger_survives_patch_group_renumbering(tmp_path):
    env = FakeCustomMoveCapsuleEnv()

    trial_module._run_capsule_llm_step_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 4,
            "capsule_max_regions_per_group": 1,
        },
        initial_code=(
            "x = 1\n"
            'custom_move("same")\n'
            'custom_move("same")\n'
        ),
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_2"}},
            {
                "action": "patch_group",
                "args": {
                    "group_id": "group_1",
                    "source": "x = 2\ny = 3\n",
                },
            },
            {"action": "run_group", "args": {"group_id": "group_3"}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_00.json").read_text())
    prompts = json.loads((tmp_path / "capsule_prompts_trial_00.json").read_text())

    assert [entry["event"]["status"] for entry in trace] == [
        "success",
        "success",
        "invalid",
        "success",
    ]
    assert "already executed robot-side-effect code" in trace[2]["event"]["message"]
    assert env.api.moves == ["same"]
    third_prompt_text = prompts[2][1]["content"][0]["text"]
    assert '"executed_side_effect_groups": [\n    "group_3"\n  ]' in third_prompt_text


def test_capsule_no_replay_rejects_second_resume_from_side_effect_region(tmp_path):
    env = FakeCustomMoveCapsuleEnv()

    trial_module._run_capsule_llm_step_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={"output_dir": str(tmp_path), "max_capsule_steps": 3},
        initial_code='custom_move("once")\n',
        scripted_actions=[
            {"action": "resume_from_region", "args": {"region_id": "region_1"}},
            {"action": "resume_from_region", "args": {"region_id": "region_1"}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_00.json").read_text())

    assert [entry["event"]["status"] for entry in trace] == [
        "success",
        "invalid",
        "success",
    ]
    assert env.api.moves == ["once"]


def test_capsule_region_execution_seals_containing_group(tmp_path):
    env = FakeCustomMoveCapsuleEnv()

    trial_module._run_capsule_llm_step_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={"output_dir": str(tmp_path), "max_capsule_steps": 3},
        initial_code='x = 1\ncustom_move("once")\n',
        scripted_actions=[
            {"action": "run_region", "args": {"region_id": "region_2"}},
            {
                "action": "patch_group",
                "args": {
                    "group_id": "group_1",
                    "source": 'x = 2\ncustom_move("patched")\n',
                },
            },
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_00.json").read_text())
    code = (tmp_path / "capsule_code_trial_00.py").read_text()

    assert [entry["event"]["status"] for entry in trace] == [
        "success",
        "invalid",
        "success",
    ]
    assert env.api.moves == ["once"]
    assert code == 'x = 1\ncustom_move("once")\n'


def test_capsule_resume_execution_seals_containing_group(tmp_path):
    env = FakeCustomMoveCapsuleEnv()

    trial_module._run_capsule_llm_step_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={"output_dir": str(tmp_path), "max_capsule_steps": 3},
        initial_code='x = 1\ncustom_move("once")\n',
        scripted_actions=[
            {"action": "resume_from_region", "args": {"region_id": "region_2"}},
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_00.json").read_text())

    assert [entry["event"]["status"] for entry in trace] == [
        "success",
        "invalid",
        "success",
    ]
    assert env.api.moves == ["once"]


def test_capsule_side_effect_lineage_uses_edit_span_with_duplicate_source(tmp_path):
    env = FakeCustomMoveCapsuleEnv()
    initial_source = 'x = 1\ncustom_move("same")\n'
    replacement = 'custom_move("same")\nx = 2\n'
    patched_source = f'{replacement}custom_move("same")\n'
    current_regions = trial_module.segment_python_code(patched_source)
    current_groups = trial_module.segment_python_code_groups(
        patched_source,
        current_regions,
        max_regions_per_group=1,
        side_effect_calls={"custom_move"},
    )
    moved_effect_group_id = current_groups[-1].group_id

    trial_module._run_capsule_llm_step_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 4,
            "capsule_max_regions_per_group": 1,
        },
        initial_code=initial_source,
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_2"}},
            {
                "action": "patch_group",
                "args": {"group_id": "group_1", "source": replacement},
            },
            {"action": "run_group", "args": {"group_id": moved_effect_group_id}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_00.json").read_text())
    prompts = json.loads((tmp_path / "capsule_prompts_trial_00.json").read_text())
    third_prompt_text = prompts[2][1]["content"][0]["text"]

    assert trace[2]["event"]["status"] == "invalid"
    assert env.api.moves == ["same"]
    assert moved_effect_group_id in third_prompt_text
    assert (
        f'"executed_side_effect_groups": [\n    "{moved_effect_group_id}"\n  ]'
        in third_prompt_text
    )


def test_capsule_llm_step_reanalyzes_forced_appended_recovery(tmp_path):
    env = FakeRewardDropCapsuleEnv()
    recovery_source = (
        "state = get_observation()\n"
        "def recover():\n"
        '    move_to("recover")\n'
        "recover()\n"
    )

    trial_module._run_capsule_llm_step_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 3,
            "capsule_validate_program_contract": True,
        },
        initial_code='move_to("bad")\n',
        scripted_actions=[
            {"action": "append_recovery", "args": {"source": recovery_source}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_00.json").read_text())
    metrics = _capsule_step_metrics(tmp_path / "capsule_step_metrics_trial_00.jsonl")

    assert [entry["event"]["action"] for entry in trace] == [
        "append_recovery",
        "run_group",
        "finish",
    ]
    assert [entry["event"]["status"] for entry in trace] == [
        "success",
        "invalid",
        "success",
    ]
    assert env.api.moves == []
    assert trace[1]["event"]["evidence"]["program_contract_violations"][0][
        "code"
    ] == "effectful_helper"
    assert metrics[0]["program_contract_valid"] is True
    assert metrics[1]["program_contract_valid"] is False


def test_capsule_llm_step_uses_compact_action_prompt_by_default(tmp_path, monkeypatch):
    prompts = []
    long_source = "\n".join(f"value_{idx} = {idx}" for idx in range(100))

    def fake_query_model(args, prompt):
        prompts.append(prompt)
        return {"content": '{"action": "finish", "args": {}}'}

    monkeypatch.setattr("capx.envs.trial._query_model", fake_query_model)

    summary = trial_module._run_capsule_llm_step_loop(
        FakeCapsuleEnv(),
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "capsule_control_mode": "llm_step",
            "max_capsule_steps": 1,
        },
        initial_code=long_source,
    )

    assert summary.num_finishes == 1
    text = prompts[0][1]["content"][0]["text"]
    assert "Compact generated code regions" in text
    assert long_source not in text
    assert _json_string_payload(long_source) not in text

    rows = _capsule_step_metrics(tmp_path / "capsule_step_metrics_trial_00.jsonl")
    assert rows[0]["action_prompt_chars"] == len(json.dumps(prompts[0], default=str))
    assert rows[0]["action_prompt_compact_context"] is True


def test_capsule_llm_step_can_disable_compact_action_prompt(tmp_path, monkeypatch):
    prompts = []
    source = "x = 1"

    def fake_query_model(args, prompt):
        prompts.append(prompt)
        return {"content": '{"action": "finish", "args": {}}'}

    monkeypatch.setattr("capx.envs.trial._query_model", fake_query_model)

    trial_module._run_capsule_llm_step_loop(
        FakeCapsuleEnv(),
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "capsule_control_mode": "llm_step",
            "max_capsule_steps": 1,
            "capsule_llm_step_compact_context": False,
        },
        initial_code=source,
    )

    text = prompts[0][1]["content"][0]["text"]
    assert "Generated code regions" in text
    assert "Compact generated code regions" not in text

    rows = _capsule_step_metrics(tmp_path / "capsule_step_metrics_trial_00.jsonl")
    assert rows[0]["action_prompt_chars"] == len(json.dumps(prompts[0], default=str))
    assert rows[0]["action_prompt_compact_context"] is False


def test_capsule_llm_step_treats_string_false_as_non_compact_context(
    tmp_path, monkeypatch
):
    prompts = []
    source = "x = 1"

    def fake_query_model(args, prompt):
        prompts.append(prompt)
        return {"content": '{"action": "finish", "args": {}}'}

    monkeypatch.setattr("capx.envs.trial._query_model", fake_query_model)

    trial_module._run_capsule_llm_step_loop(
        FakeCapsuleEnv(),
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "capsule_control_mode": "llm_step",
            "max_capsule_steps": 1,
            "capsule_llm_step_compact_context": "false",
        },
        initial_code=source,
    )

    text = prompts[0][1]["content"][0]["text"]
    assert "Generated code regions" in text
    assert "Compact generated code regions" not in text

    rows = _capsule_step_metrics(tmp_path / "capsule_step_metrics_trial_00.jsonl")
    assert rows[0]["action_prompt_compact_context"] is False


def test_capsule_llm_step_records_prompt_budget_overflow_after_fallback(
    tmp_path, monkeypatch
):
    prompts = []
    source = "\n".join(f"value_{idx} = {idx}" for idx in range(100))

    def fake_query_model(args, prompt):
        prompts.append(prompt)
        return {"content": '{"action": "finish", "args": {}}'}

    monkeypatch.setattr("capx.envs.trial._query_model", fake_query_model)

    trial_module._run_capsule_llm_step_loop(
        FakeCapsuleEnv(),
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "capsule_control_mode": "llm_step",
            "max_capsule_steps": 1,
            "capsule_action_prompt_char_budget": 1,
        },
        initial_code=source,
    )

    rows = _capsule_step_metrics(tmp_path / "capsule_step_metrics_trial_00.jsonl")
    assert rows[0]["action_prompt_chars"] == len(json.dumps(prompts[0], default=str))
    assert rows[0]["action_prompt_char_budget"] == 1
    assert rows[0]["action_prompt_over_budget"] is True


def test_capsule_llm_step_compact_prompt_does_not_replay_full_patched_source(
    tmp_path, monkeypatch
):
    prompts = []
    patched_body = "\n".join(f"patched_{idx} = {idx}" for idx in range(200))
    patched_source = f"PATCHED_SOURCE = {patched_body!r}\n"
    responses = iter(
        [
            {
                "content": json.dumps(
                    {
                        "action": "patch_group",
                        "args": {"group_id": "group_1", "source": patched_source},
                    }
                )
            },
            {"content": '{"action": "finish", "args": {}}'},
        ]
    )

    def fake_query_model(args, prompt):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr("capx.envs.trial._query_model", fake_query_model)

    trial_module._run_capsule_llm_step_loop(
        FakeCapsuleEnv(),
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "capsule_control_mode": "llm_step",
            "max_capsule_steps": 2,
        },
        initial_code="x = 1\n",
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_00.json").read_text())

    assert trace[0]["event"]["action"] == "patch_group"
    assert trace[0]["event"]["status"] == "success"
    assert patched_source in trace[0]["event"]["evidence"]["source"]
    assert trace[1]["event"]["action"] == "finish"
    assert len(prompts) == 2
    second_prompt_text = prompts[1][1]["content"][0]["text"]
    assert "Recent runtime history summary" in second_prompt_text
    assert "patch_group" in second_prompt_text
    assert "group_1" in second_prompt_text
    assert "success" in second_prompt_text
    assert patched_source not in second_prompt_text
    assert _json_string_payload(patched_source) not in second_prompt_text
    assert "patched_199 = 199" not in second_prompt_text


def test_capsule_llm_step_rejects_patch_after_failed_side_effect_group(
    tmp_path, monkeypatch
):
    env = FakeRewardDropCapsuleEnv()

    def single_failing_side_effect_group(source, regions, **kwargs):
        return [
            CodeRegionGroup(
                group_id="group_1",
                start_line=1,
                end_line=2,
                source=source,
                region_ids=[region.region_id for region in regions],
                primitive_calls=["move_to"],
                defined_names=[],
                used_names=["move_to"],
                has_robot_side_effect=True,
            )
        ]

    monkeypatch.setattr(
        "capx.envs.trial.segment_python_code_groups",
        single_failing_side_effect_group,
    )

    summary = _run_capsule_trial(
        env=env,
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "use_runtime_control": True,
            "capsule_control_mode": "llm_step",
            "scripted_actions": [
                {"action": "run_group", "args": {"group_id": "group_1"}},
                {
                    "action": "patch_group",
                    "args": {"group_id": "group_1", "source": 'move_to("recover")'},
                },
            ],
            "max_capsule_steps": 3,
            "capsule_max_regions_per_group": 2,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='move_to("good")\nraise RuntimeError("boom")\n',
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 1
    assert env.api.moves == ["good"]
    assert [entry["event"]["status"] for entry in trace] == ["failed", "invalid", "success"]
    assert "already executed robot-side-effect code" in trace[1]["event"]["message"]


def test_capsule_llm_step_rejects_replay_after_failed_side_effect_region(
    tmp_path, monkeypatch
):
    env = FakeRewardDropCapsuleEnv()

    def single_failing_side_effect_region(source):
        return [CodeRegion(region_id="region_1", start_line=1, end_line=2, source=source)]

    monkeypatch.setattr("capx.envs.trial.segment_python_code", single_failing_side_effect_region)

    summary = _run_capsule_trial(
        env=env,
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "use_runtime_control": True,
            "capsule_control_mode": "llm_step",
            "capsule_execution_granularity": "region",
            "scripted_actions": [
                {"action": "run_region", "args": {"region_id": "region_1"}},
                {"action": "run_region", "args": {"region_id": "region_1"}},
            ],
            "max_capsule_steps": 3,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='move_to("good")\nraise RuntimeError("boom")\n',
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 1
    assert env.api.moves == ["good"]
    assert [entry["event"]["status"] for entry in trace] == ["failed", "invalid", "success"]
    assert "already executed robot-side-effect code" in trace[1]["event"]["message"]


def test_capsule_llm_step_marks_failed_dynamic_side_effect_group_region(
    tmp_path, monkeypatch
):
    env = FakeCustomMoveCapsuleEnv()

    def single_dynamic_side_effect_region(source):
        return [CodeRegion(region_id="region_1", start_line=1, end_line=2, source=source)]

    def single_dynamic_side_effect_group(source, regions, **kwargs):
        return [
            CodeRegionGroup(
                group_id="group_1",
                start_line=1,
                end_line=2,
                source=source,
                region_ids=[region.region_id for region in regions],
                primitive_calls=["custom_move"],
                defined_names=[],
                used_names=["custom_move"],
                has_robot_side_effect=True,
            )
        ]

    monkeypatch.setattr("capx.envs.trial.segment_python_code", single_dynamic_side_effect_region)
    monkeypatch.setattr(
        "capx.envs.trial.segment_python_code_groups",
        single_dynamic_side_effect_group,
    )

    summary = _run_capsule_trial(
        env=env,
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "use_runtime_control": True,
            "capsule_control_mode": "llm_step",
            "scripted_actions": [
                {"action": "run_group", "args": {"group_id": "group_1"}},
                {"action": "run_region", "args": {"region_id": "region_1"}},
            ],
            "max_capsule_steps": 3,
            "capsule_max_regions_per_group": 2,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='custom_move("good")\nraise RuntimeError("boom")\n',
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 1
    assert env.api.moves == ["good"]
    assert [entry["event"]["status"] for entry in trace] == ["failed", "invalid", "success"]
    assert "already executed robot-side-effect code" in trace[1]["event"]["message"]


def test_capsule_llm_step_marks_failed_dynamic_side_effect_region(
    tmp_path, monkeypatch
):
    env = FakeCustomMoveCapsuleEnv()

    def single_dynamic_side_effect_region(source):
        return [CodeRegion(region_id="region_1", start_line=1, end_line=2, source=source)]

    monkeypatch.setattr("capx.envs.trial.segment_python_code", single_dynamic_side_effect_region)

    summary = _run_capsule_trial(
        env=env,
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "use_runtime_control": True,
            "capsule_control_mode": "llm_step",
            "capsule_execution_granularity": "region",
            "scripted_actions": [
                {"action": "run_region", "args": {"region_id": "region_1"}},
                {"action": "run_region", "args": {"region_id": "region_1"}},
            ],
            "max_capsule_steps": 3,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='custom_move("good")\nraise RuntimeError("boom")\n',
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 1
    assert env.api.moves == ["good"]
    assert [entry["event"]["status"] for entry in trace] == ["failed", "invalid", "success"]
    assert "already executed robot-side-effect code" in trace[1]["event"]["message"]


def test_capsule_llm_step_rejects_patch_after_successful_dynamic_side_effect_group(
    tmp_path, monkeypatch
):
    env = FakeCustomMoveCapsuleEnv()

    def single_dynamic_side_effect_region(source):
        return [CodeRegion(region_id="region_1", start_line=1, end_line=1, source=source)]

    def single_dynamic_side_effect_group(source, regions, **kwargs):
        return [
            CodeRegionGroup(
                group_id="group_1",
                start_line=1,
                end_line=1,
                source=source,
                region_ids=[region.region_id for region in regions],
                primitive_calls=["custom_move"],
                defined_names=[],
                used_names=["custom_move"],
                has_robot_side_effect=True,
            )
        ]

    monkeypatch.setattr("capx.envs.trial.segment_python_code", single_dynamic_side_effect_region)
    monkeypatch.setattr(
        "capx.envs.trial.segment_python_code_groups",
        single_dynamic_side_effect_group,
    )

    summary = _run_capsule_trial(
        env=env,
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "use_runtime_control": True,
            "capsule_control_mode": "llm_step",
            "scripted_actions": [
                {"action": "run_group", "args": {"group_id": "group_1"}},
                {"action": "patch_region", "args": {"region_id": "region_1", "source": "x = 1"}},
            ],
            "max_capsule_steps": 3,
            "capsule_max_regions_per_group": 2,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='custom_move("good")\n',
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 1
    assert env.api.moves == ["good"]
    assert [entry["event"]["status"] for entry in trace] == ["success", "invalid", "success"]
    assert "already executed robot-side-effect code" in trace[1]["event"]["message"]


class FakeMultiTurnEnv:
    def __init__(self):
        self.steps = []
        self.low_level_env = object()
        self.obs = {
            "full_prompt": [
                {"role": "system", "content": "x"},
                {"role": "user", "content": [{"type": "text", "text": "task"}]},
            ]
        }

    def reset(self, *, seed=None, options=None):
        return self.obs, {}

    def step(self, code):
        self.steps.append(code)
        return self.obs, 0.0, False, False, {
            "sandbox_rc": 0,
            "stdout": "",
            "stderr": "",
            "task_completed": False,
        }


def test_capsule_trial_runs_scripted_regions(tmp_path):
    summary = _run_capsule_trial(
        env=FakeCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 4,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code="x = 1\nRESULT = x + 1\n",
        scripted_actions=[
            {"action": "run_region", "args": {"region_id": "region_1"}},
            {"action": "run_region", "args": {"region_id": "region_2"}},
            {"action": "finish", "args": {}},
        ],
    )

    assert summary.sandbox_rc == 0
    assert summary.num_code_blocks == 2
    assert summary.num_finishes == 1


def test_capsule_trial_runs_scripted_group(tmp_path):
    env = FakeCapsuleEnv()

    summary = _run_capsule_trial(
        env=env,
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 3,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='pose = get_pose("cube")\nmove_to(pose)\n',
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())
    assert env.api.moved is True
    assert summary.sandbox_rc == 0
    assert summary.num_code_blocks == 2
    assert trace[0]["event"]["action"] == "run_group"
    assert trace[0]["feedback"]["region_id"] == "group_1"
    assert trace[0]["event"]["evidence"]["source_span"] == {"start_line": 1, "end_line": 2}


def test_capsule_trial_defaults_to_loose_group_cap(tmp_path):
    source = "\n".join(f"v{i} = {i}" for i in range(8))

    _run_capsule_trial(
        env=FakeCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 2,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code=source,
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())
    assert trace[0]["event"]["evidence"]["source_span"] == {"start_line": 1, "end_line": 8}


def test_capsule_trial_writes_original_source_after_group_normalization(tmp_path):
    source = "\n".join(
        [
            "def pick():",
            "    move_to([1, 2, 3])",
            "pick()",
            "x = 1",
        ]
    )

    summary = _run_capsule_trial(
        env=FakeCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 2,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code=source,
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "finish", "args": {}},
        ],
    )

    assert Path(summary.code_path).read_text() == source


def test_capsule_trial_patches_group_and_regroups(tmp_path):
    summary = _run_capsule_trial(
        env=FakeCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 4,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='pose = get_pose("cube")\nmove_to(pose)\nRESULT = "old"\n',
        scripted_actions=[
            {
                "action": "patch_group",
                "args": {
                    "group_id": "group_1",
                    "source": 'pose = get_pose("cube")\nRESULT = "patched"\nmove_to(pose)',
                },
            },
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())
    patched_source = Path(summary.code_path).read_text()

    assert summary.sandbox_rc == 0
    assert trace[0]["event"]["action"] == "patch_group"
    assert trace[1]["event"]["region_id"] == "group_1"
    assert 'RESULT = "patched"' in patched_source


def test_capsule_repairs_invalid_initial_source_with_patch_group(tmp_path):
    summary = _run_capsule_trial(
        env=FakeCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 4,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code="value = (\n",
        scripted_actions=[
            {
                "action": "patch_group",
                "args": {
                    "group_id": "group_1",
                    "source": "value = 1\nRESULT = value\n",
                },
            },
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 0
    assert trace[0]["event"]["action"] == "initial_parse"
    assert trace[0]["event"]["evidence"]["exception_type"] == "SyntaxError"
    assert trace[1]["event"]["action"] == "patch_group"
    assert trace[2]["event"]["action"] == "run_group"


def test_capsule_retries_after_syntax_error_in_group_patch(tmp_path):
    summary = _run_capsule_trial(
        env=FakeCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 4,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code="value = (\n",
        scripted_actions=[
            {
                "action": "patch_group",
                "args": {"group_id": "group_1", "source": "value = [\n"},
            },
            {
                "action": "patch_group",
                "args": {
                    "group_id": "group_1",
                    "source": "value = 1\nRESULT = value\n",
                },
            },
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())
    final_source = Path(summary.code_path).read_text()

    assert trace[1]["event"]["action"] == "patch_group"
    assert trace[1]["event"]["status"] == "invalid"
    assert trace[1]["event"]["evidence"]["exception_type"] == "SyntaxError"
    assert trace[2]["event"]["status"] == "success"
    assert trace[3]["event"]["status"] == "success"
    assert final_source == "value = 1\nRESULT = value\n"


def test_capsule_trial_appends_recovery_and_regroups(tmp_path):
    env = FakeCapsuleEnv()

    summary = _run_capsule_trial(
        env=env,
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 4,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='pose = get_pose("cube")\nmove_to(pose)\n',
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {
                "action": "append_recovery",
                "args": {"source": 'obs = get_observation()\nRESULT = "recovered"'},
            },
            {"action": "run_group", "args": {"group_id": "group_2"}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())
    patched_source = Path(summary.code_path).read_text()

    assert summary.sandbox_rc == 0
    assert env.api.observed is True
    assert trace[1]["event"]["action"] == "append_recovery"
    assert trace[2]["event"]["region_id"] == "group_2"
    assert 'RESULT = "recovered"' in patched_source


def test_append_recovery_requires_fresh_observation(tmp_path):
    summary = _run_capsule_trial(
        env=FakeIncompleteCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 2,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code="x = 1\n",
        scripted_actions=[
            {"action": "append_recovery", "args": {"source": 'RESULT = "no observation"'}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 1
    assert trace[0]["event"]["status"] == "invalid"
    assert "get_observation" in trace[0]["event"]["message"]


def test_append_recovery_accepts_api_declared_fresh_state_function(tmp_path):
    env = FakeHandleRecoveryEnv()
    summary = _run_capsule_trial(
        env=env,
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 3,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code="x = 1\n",
        scripted_actions=[
            {"action": "append_recovery", "args": {"source": "handle = get_handle0_pos()"}},
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 1  # The fake task never succeeds, but recovery is valid.
    assert trace[0]["event"]["status"] == "success"
    assert trace[1]["event"]["status"] == "success"
    assert env.api.observed is True


def test_append_recovery_rejects_blind_code_for_task_specific_observation_api(tmp_path):
    _run_capsule_trial(
        env=FakeHandleRecoveryEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 2,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code="x = 1\n",
        scripted_actions=[
            {"action": "append_recovery", "args": {"source": 'RESULT = "blind"'}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert trace[0]["event"]["status"] == "invalid"
    assert "get_handle0_pos" in trace[0]["event"]["message"]


def test_append_recovery_rejects_api_without_fresh_state_capability():
    event = _execute_runtime_action(
        RuntimeAction("append_recovery", {"source": "x = 1"}),
        CapsuleExecutor(base_globals={}),
        "",
        {},
        recovery_observation_functions=set(),
    )

    assert event.status == "invalid"
    assert "does not declare" in event.message


def test_capsule_trial_rejects_rerun_of_executed_side_effect_group(tmp_path):
    summary = _run_capsule_trial(
        env=FakeCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 3,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='pose = get_pose("cube")\nmove_to(pose)\n',
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 1
    assert trace[1]["event"]["status"] == "invalid"
    assert "append_recovery" in trace[1]["event"]["message"]


def test_capsule_llm_step_prompt_includes_executed_side_effect_ledger(tmp_path):
    _run_capsule_trial(
        env=FakeIncompleteCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 2,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='pose = get_pose("cube")\nmove_to(pose)\n',
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "finish", "args": {}},
        ],
    )

    prompts = json.loads((tmp_path / "capsule_prompts_trial_01.json").read_text())
    second_prompt_text = prompts[1][1]["content"][0]["text"]

    assert "Side-effect execution ledger" in second_prompt_text
    assert '"executed_side_effect_groups": [\n    "group_1"\n  ]' in second_prompt_text
    assert '"execution_state": "executed_side_effect"' in second_prompt_text
    assert '"run_allowed": false' in second_prompt_text
    assert '{"action": "run_group", "args": {"group_id": "group_1"}}' not in second_prompt_text


def test_no_rollback_guard_uses_task_specific_recovery_function():
    event = _no_rollback_guard_event(
        RuntimeAction("run_group", {"group_id": "group_1"}),
        set(),
        {"group_1"},
        recovery_observation_functions={"get_handle0_pos"},
    )

    assert event is not None
    assert "get_handle0_pos()" in event.message
    assert "get_observation()" not in event.message


def test_reward_drop_guard_uses_task_specific_recovery_function():
    group = CodeRegionGroup(
        group_id="group_1",
        start_line=1,
        end_line=1,
        source="move_to('next')",
        region_ids=["region_1"],
        primitive_calls=["move_to"],
        defined_names=[],
        used_names=["move_to"],
        has_robot_side_effect=True,
    )
    event = _reward_drop_guard_event(
        RuntimeAction("run_group", {"group_id": "group_1"}),
        {"reward": 0.01},
        0.1,
        {},
        {"group_1": group},
        recovery_side_effect_budget=0,
        min_best_reward=0.05,
        drop_threshold=0.03,
        recovery_observation_functions={"get_handle0_pos"},
    )

    assert event is not None
    assert "get_handle0_pos()" in event.message
    assert "get_observation()" not in event.message


def test_capsule_action_query_uses_separate_max_tokens(tmp_path, monkeypatch):
    observed_max_tokens = []

    def fake_query_model(args, prompt):
        observed_max_tokens.append(args.max_tokens)
        return {"content": '{"action": "finish", "args": {}}', "reasoning": None}

    monkeypatch.setattr("capx.envs.trial._query_model", fake_query_model)

    _run_capsule_trial(
        env=FakeCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False, max_tokens=8192),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 1,
            "capsule_action_max_tokens": 512,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code="x = 1\n",
    )

    assert observed_max_tokens == [512]


def test_capsule_trial_rejects_patch_of_executed_side_effect_group(tmp_path):
    summary = _run_capsule_trial(
        env=FakeCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 3,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='pose = get_pose("cube")\nmove_to(pose)\n',
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {
                "action": "patch_group",
                "args": {"group_id": "group_1", "source": 'obs = get_observation()'},
            },
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 1
    assert trace[1]["event"]["status"] == "invalid"
    assert "already executed" in trace[1]["event"]["message"]
    assert "append_recovery" in trace[1]["event"]["message"]


def test_capsule_trial_blocks_side_effect_after_reward_drop_from_best(tmp_path):
    env = FakeRewardDropCapsuleEnv()

    summary = _run_capsule_trial(
        env=env,
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 4,
            "capsule_max_regions_per_group": 1,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='move_to("good")\nmove_to("bad")\nmove_to("worse")\n',
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "run_group", "args": {"group_id": "group_2"}},
            {"action": "run_group", "args": {"group_id": "group_3"}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 1
    assert env.api.moves == ["good", "bad"]
    assert trace[2]["event"]["status"] == "invalid"
    assert "reward dropped" in trace[2]["event"]["message"]
    assert "append_recovery" in trace[2]["event"]["message"]


def test_capsule_trial_allows_recovery_side_effect_after_append_recovery(tmp_path):
    env = FakeRewardDropCapsuleEnv()

    summary = _run_capsule_trial(
        env=env,
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 6,
            "capsule_max_regions_per_group": 1,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='move_to("good")\nmove_to("bad")\n',
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "run_group", "args": {"group_id": "group_2"}},
            {
                "action": "append_recovery",
                "args": {"source": 'obs = get_observation()\nmove_to("recover")'},
            },
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 0
    assert env.api.observed is True
    assert env.api.moves == ["good", "bad", "recover"]
    assert trace[4]["event"]["status"] == "success"


def test_capsule_llm_step_auto_executes_appended_recovery_before_next_action(tmp_path):
    env = FakeRewardDropCapsuleEnv()

    summary = _run_capsule_trial(
        env=env,
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 6,
            "capsule_max_regions_per_group": 1,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='move_to("good")\nmove_to("bad")\n',
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "run_group", "args": {"group_id": "group_2"}},
            {
                "action": "append_recovery",
                "args": {"source": 'obs = get_observation()\nmove_to("recover")'},
            },
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 0
    assert env.api.observed is True
    assert env.api.moves == ["good", "bad", "recover"]
    assert [entry["event"]["action"] for entry in trace] == [
        "run_group",
        "run_group",
        "append_recovery",
        "run_group",
        "run_group",
        "finish",
    ]
    assert trace[3]["action"]["args"]["group_id"] == "group_3"
    assert trace[4]["action"]["args"]["group_id"] == "group_4"


def test_capsule_llm_step_allows_entire_appended_recovery_block_after_reward_drop(
    tmp_path,
):
    env = FakeRewardDropCapsuleEnv()

    summary = _run_capsule_trial(
        env=env,
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 7,
            "capsule_max_regions_per_group": 1,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='move_to("good")\nmove_to("bad")\n',
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "run_group", "args": {"group_id": "group_2"}},
            {
                "action": "append_recovery",
                "args": {
                    "source": (
                        'obs = get_observation()\nmove_to("hold")\nmove_to("recover")'
                    )
                },
            },
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 0
    assert env.api.observed is True
    assert env.api.moves == ["good", "bad", "hold", "recover"]
    assert [entry["event"]["status"] for entry in trace] == [
        "success",
        "success",
        "success",
        "success",
        "success",
        "success",
        "success",
    ]
    assert trace[4]["action"]["args"]["group_id"] == "group_4"
    assert trace[5]["action"]["args"]["group_id"] == "group_5"


def test_capsule_metrics_split_append_source_from_recovery_execution(tmp_path):
    _run_capsule_trial(
        env=FakeRewardDropCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 5,
            "capsule_max_regions_per_group": 1,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='move_to("good")\nmove_to("bad")\n',
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "run_group", "args": {"group_id": "group_2"}},
            {
                "action": "append_recovery",
                "args": {"source": 'obs = get_observation()\nmove_to("recover")'},
            },
        ],
    )

    metrics_path = tmp_path / "capsule_step_metrics_trial_01.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]

    assert rows[2]["action"] == "append_recovery"
    assert rows[2]["append_recovery_source_appended"] is True
    assert rows[2]["recovery_execution_attempt"] is False
    assert rows[2]["recovery_execution_improved"] is False

    assert rows[4]["action"] == "run_group"
    assert rows[4]["recovery_execution_attempt"] is True
    assert rows[4]["recovery_execution_reward_improved"] is True
    assert rows[4]["recovery_execution_trace_improved"] is True
    assert rows[4]["recovery_execution_improved"] is True
    assert rows[4]["recovery_execution_effective"] is True


def test_runtime_variable_summary_includes_only_small_array_values():
    import numpy as np

    small = _summarize_runtime_value(np.array([1.0, 2.0, 3.0]))
    large = _summarize_runtime_value(np.zeros((8, 8)))

    assert small["value"] == [1.0, 2.0, 3.0]
    assert "value" not in large


def test_runtime_variable_summary_safely_handles_non_numpy_shape():
    class TensorLike:
        shape = (3,)
        dtype = "float32"

        def size(self):
            return 3

    summary = _summarize_runtime_value(TensorLike())

    assert summary["shape"] == [3]
    assert "value" not in summary


def test_capsule_trial_allows_patch_of_executed_non_side_effect_group(tmp_path):
    summary = _run_capsule_trial(
        env=FakeCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 4,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code="x = 1\nRESULT = x\n",
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "patch_group", "args": {"group_id": "group_1", "source": "x = 2\nRESULT = x"}},
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())
    patched_source = Path(summary.code_path).read_text()

    assert summary.sandbox_rc == 0
    assert trace[1]["event"]["status"] == "success"
    assert "x = 2" in patched_source


def test_multiturn_trial_stops_after_max_regenerations(tmp_path, monkeypatch):
    decision_calls = []

    def fake_initial_code(args, config, obs):
        return "print('initial')", None, None

    def fake_multi_turn_step(*args, **kwargs):
        decision_calls.append(1)
        return "regenerate", "print('regenerated')", None, None, []

    monkeypatch.setattr("capx.envs.trial._query_initial_code", fake_initial_code)
    monkeypatch.setattr("capx.envs.trial._handle_multi_turn_step", fake_multi_turn_step)

    summary = _run_single_trial(
        env=FakeMultiTurnEnv(),
        trial=1,
        args=SimpleNamespace(
            model="test",
            visual_differencing_model="test",
            visual_differencing_model_server_url="http://example.test",
            visual_differencing_model_api_key=None,
            max_tokens=100,
            temperature=0.2,
            reasoning_effort="minimal",
            debug=False,
        ),
        config={
            "output_dir": str(tmp_path),
            "agent_mode": "code",
            "record_video": False,
            "use_visual_feedback": False,
            "use_img_differencing": False,
            "use_video_differencing": False,
            "use_wrist_camera": False,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
            "use_oracle_code": False,
            "save_multiturn_prompts": False,
            "max_regenerations": 1,
        },
        multi_turn_prompt=(
            "executed={executed_code}\nstdout={console_stdout}\nstderr={console_stderr}"
        ),
    )

    assert len(decision_calls) == 1
    assert summary.num_regenerations == 1
    assert len(summary.log) > 0


def test_capsule_trial_marks_exhausted_budget_as_failed(tmp_path):
    summary = _run_capsule_trial(
        env=FakeIncompleteCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 2,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code="x = 1\nRESULT = x + 1\n",
        scripted_actions=[
            {"action": "run_region", "args": {"region_id": "region_1"}},
            {"action": "inspect_variables", "args": {"names": ["x"]}},
        ],
    )

    assert summary.sandbox_rc == 1
    assert summary.success is False
    assert summary.num_finishes == 0


def test_capsule_llm_step_rejects_premature_finish_and_continues(tmp_path):
    env = FakeRewardDropCapsuleEnv()

    summary = trial_module._run_capsule_llm_step_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 3,
            "capsule_require_task_success_for_finish": True,
            "capsule_progress_mode": "sparse_terminal",
        },
        initial_code='move_to("recover")\n',
        scripted_actions=[
            {"action": "finish", "args": {}},
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "inspect_variables", "args": {"names": []}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_00.json").read_text())

    assert summary.reward == 1.0
    assert summary.num_finishes == 0
    assert [entry["event"]["action"] for entry in trace] == ["finish", "run_group"]
    assert trace[0]["event"]["status"] == "warning"
    assert trace[0]["feedback"]["status"] == "warning"
    assert trace[0]["feedback"]["message"] == (
        "Finish rejected because the environment success predicate is not satisfied."
    )


def test_capsule_llm_step_rejects_invalid_progress_mode_before_side_effect(tmp_path):
    env = FakeRewardDropCapsuleEnv()

    with pytest.raises(ValueError, match="progress_mode.*dense.*sparse_terminal"):
        trial_module._run_capsule_llm_step_loop(
            env,
            trial=0,
            args=SimpleNamespace(model="test", use_oracle_code=False),
            config={
                "output_dir": str(tmp_path),
                "max_capsule_steps": 1,
                "capsule_progress_mode": "invalid",
            },
            initial_code='move_to("recover")\n',
            scripted_actions=[
                {"action": "run_group", "args": {"group_id": "group_1"}},
            ],
        )

    assert env.api.moves == []


def test_capsule_llm_step_stops_immediately_on_success_when_finish_guard_enabled(tmp_path):
    env = FakeRewardDropCapsuleEnv()

    summary = trial_module._run_capsule_llm_step_loop(
        env,
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 2,
            "capsule_require_task_success_for_finish": True,
        },
        initial_code='move_to("recover")\n',
        scripted_actions=[
            {"action": "run_group", "args": {"group_id": "group_1"}},
            {"action": "inspect_variables", "args": {"names": []}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_00.json").read_text())

    assert summary.reward == 1.0
    assert summary.num_finishes == 0
    assert [entry["event"]["action"] for entry in trace] == ["run_group"]


def test_capsule_llm_step_marks_budget_exhausted_without_event_failure(tmp_path):
    summary = trial_module._run_capsule_llm_step_loop(
        FakeIncompleteCapsuleEnv(),
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 2,
        },
        initial_code="x = 1\nRESULT = x + 1\n",
        scripted_actions=[
            {"action": "run_region", "args": {"region_id": "region_1"}},
            {"action": "inspect_variables", "args": {"names": ["x"]}},
        ],
    )

    metrics = _capsule_step_metrics(tmp_path / "capsule_step_metrics_trial_00.jsonl")

    assert summary.truncated is True
    assert summary.failure_kind is None
    assert metrics[-1]["budget_exhausted"] is True
    assert "Budget Exhausted: True" in summary.log


def test_capsule_llm_step_marks_zero_step_budget_exhausted(tmp_path):
    summary = trial_module._run_capsule_llm_step_loop(
        FakeIncompleteCapsuleEnv(),
        trial=0,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "output_dir": str(tmp_path),
            "max_capsule_steps": 0,
        },
        initial_code="x = 1\n",
        scripted_actions=[],
    )

    metrics_path = tmp_path / "capsule_step_metrics_trial_00.jsonl"

    assert summary.truncated is True
    assert summary.num_finishes == 0
    assert metrics_path.read_text() == ""
    assert "Budget Exhausted: True" in summary.log


def test_capsule_trial_finish_without_completion_is_failed(tmp_path):
    summary = _run_capsule_trial(
        env=FakeIncompleteCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 1,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code="x = 1\n",
        scripted_actions=[{"action": "finish", "args": {}}],
    )

    assert summary.sandbox_rc == 1
    assert summary.success is False
    assert summary.task_completed is None
    assert summary.num_finishes == 1


def test_capsule_trial_records_video_when_requested(tmp_path, monkeypatch):
    video_writes = []

    def fake_write_video(frames, base_dir, *, suffix):
        video_writes.append({"frames": frames, "base_dir": base_dir, "suffix": suffix})

    monkeypatch.setattr("capx.envs.trial._write_video", fake_write_video)
    env = FakeVideoCapsuleEnv()

    _run_capsule_trial(
        env=env,
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "record_video": True,
            "use_wrist_camera": False,
            "max_capsule_steps": 4,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code="RESULT = 1\n",
        scripted_actions=[{"action": "finish", "args": {}}],
    )

    assert env.video_capture_args == {
        "enabled": True,
        "clear": True,
        "wrist_camera": False,
    }
    assert env.video_clear_requested is True
    assert video_writes == [
        {
            "frames": ["frame-1", "frame-2"],
            "base_dir": str(tmp_path / "trial_01_sandboxrc_0_reward_1.000_taskcompleted_0"),
            "suffix": "1.000_capsule",
        }
    ]


def test_capsule_trial_writes_trace_and_feedback_artifact(tmp_path):
    _run_capsule_trial(
        env=FakeCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 4,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='pose = get_pose("cube")\nmove_to(pose)\nRESULT = "done"\n',
        scripted_actions=[
            {"action": "run_region", "args": {"region_id": "region_1"}},
            {"action": "run_region", "args": {"region_id": "region_2"}},
            {"action": "inspect_trace", "args": {}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())
    assert trace[0]["feedback"]["region_id"] == "region_1"
    assert trace[0]["trace_events"][0]["name"] == "get_pose"
    assert trace[2]["event"]["evidence"]["recent_events"][0]["name"] == "get_pose"
    assert trace[2]["event"]["evidence"]["event_count"] == 2
    assert "events" not in trace[2]["event"]["evidence"]


def test_inspect_trace_supports_last_n_and_failed_only(tmp_path):
    _run_capsule_trial(
        env=FakeTraceFailureEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 4,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code=(
            'get_pose("cube")\n'
            "fail_primitive()\n"
        ),
        scripted_actions=[
            {"action": "run_region", "args": {"region_id": "region_1"}},
            {"action": "run_region", "args": {"region_id": "region_2"}},
            {"action": "inspect_trace", "args": {"last_n": 1, "failed_only": True}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())
    evidence = trace[2]["event"]["evidence"]

    assert evidence["event_count"] == 2
    assert evidence["failed_event_count"] == 1
    assert [event["name"] for event in evidence["recent_events"]] == ["fail_primitive"]
    assert evidence["recent_events"][0]["status"] == "failed"


def test_capsule_trial_writes_step_metrics_jsonl(tmp_path):
    _run_capsule_trial(
        env=FakeCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
            "capsule_control_mode": "llm_step",
            "output_dir": str(tmp_path),
            "max_capsule_steps": 3,
            "use_parallel_ensemble": False,
            "use_multimodel": False,
        },
        initial_code='pose = get_pose("cube")\nmove_to(pose)\n',
        scripted_actions=[
            {"action": "run_region", "args": {"region_id": "region_1"}},
            {"action": "finish", "args": {}},
        ],
    )

    metrics_path = tmp_path / "capsule_step_metrics_trial_01.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]

    assert [row["step_id"] for row in rows] == [1, 2]
    assert rows[0]["action"] == "run_region"
    assert rows[0]["event_status"] == "success"
    assert rows[0]["reward_before"] == 1.0
    assert rows[0]["reward_after"] == 1.0
    assert rows[0]["best_reward_so_far"] == 1.0
    assert rows[0]["reward_drop_from_best"] == 0.0
    assert rows[0]["state_after"]["reward"] == 1.0


def test_patch_region_accepts_new_source_alias():
    source = "x = 1\ny = x + 1\n"
    region = CodeRegion("region_2", 2, 2, "y = x + 1")
    event = _execute_runtime_action(
        RuntimeAction(
            "patch_region",
            {"region_id": "region_2", "new_source": "y = x + 2"},
        ),
        CapsuleExecutor(base_globals={}),
        source,
        {"region_2": region},
    )

    assert event.status == "success"
    assert event.evidence["source"] == "x = 1\ny = x + 2\n"


def test_patch_region_accepts_patch_alias():
    source = "x = 1\ny = x + 1\n"
    region = CodeRegion("region_2", 2, 2, "y = x + 1")
    event = _execute_runtime_action(
        RuntimeAction(
            "patch_region",
            {"region_id": "region_2", "patch": "y = x + 3"},
        ),
        CapsuleExecutor(base_globals={}),
        source,
        {"region_2": region},
    )

    assert event.status == "success"
    assert event.evidence["source"] == "x = 1\ny = x + 3\n"


def test_patch_region_rejects_syntax_error_in_complete_source():
    source = "x = 1\ny = x + 1\n"
    region = CodeRegion("region_2", 2, 2, "y = x + 1")
    event = _execute_runtime_action(
        RuntimeAction(
            "patch_region",
            {"region_id": "region_2", "source": "y = ("},
        ),
        CapsuleExecutor(base_globals={}),
        source,
        {"region_2": region},
    )

    assert event.status == "invalid"
    assert event.region_id == "region_2"
    assert event.evidence["exception_type"] == "SyntaxError"
    assert "source" not in event.evidence


def test_inspect_variables_requires_names():
    event = _execute_runtime_action(
        RuntimeAction("inspect_variables", {"region_id": "region_1"}),
        CapsuleExecutor(base_globals={}),
        "x = 1\n",
        {},
    )

    assert event.status == "invalid"
    assert "args.names" in event.message
