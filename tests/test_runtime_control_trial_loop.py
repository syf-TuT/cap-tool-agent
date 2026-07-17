import json
from pathlib import Path
from types import SimpleNamespace

from capx.envs.trial import (
    _describe_initial_scene,
    _execute_runtime_action,
    _get_video_differencing_feedback,
    _get_visual_differencing_feedback,
    _handle_multi_turn_step,
    _no_rollback_guard_event,
    _query_initial_code,
    _reward_drop_guard_event,
    _run_capsule_trial,
    _run_single_trial,
    _summarize_runtime_value,
)
from capx.llm.client import ModelQueryArgs
from capx.llm.context import get_trial_llm_context, trial_llm_context
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
                "output_dir": str(tmp_path),
                "max_capsule_steps": 1,
                "use_parallel_ensemble": False,
                "use_multimodel": False,
            },
            initial_code="x = 1\n",
        )

    assert _telemetry_stages(telemetry_path) == ["capsule_action"]


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


def test_capsule_auto_forward_stops_after_failed_group(tmp_path):
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
        initial_code='raise RuntimeError("boom")\nmove_to("recover")\n',
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 1
    assert env.api.moves == []
    assert [entry["event"]["action"] for entry in trace] == ["run_group"]
    assert trace[0]["event"]["status"] == "failed"


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
            {"action": "run_group", "args": {"group_id": "group_3"}},
            {"action": "run_group", "args": {"group_id": "group_4"}},
            {"action": "finish", "args": {}},
        ],
    )

    trace = json.loads((tmp_path / "capsule_trace_trial_01.json").read_text())

    assert summary.sandbox_rc == 0
    assert env.api.observed is True
    assert env.api.moves == ["good", "bad", "recover"]
    assert trace[4]["event"]["status"] == "success"


def test_capsule_metrics_split_append_source_from_recovery_execution(tmp_path):
    _run_capsule_trial(
        env=FakeRewardDropCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
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
            {"action": "run_group", "args": {"group_id": "group_3"}},
            {"action": "run_group", "args": {"group_id": "group_4"}},
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


def test_capsule_trial_finish_without_completion_is_failed(tmp_path):
    summary = _run_capsule_trial(
        env=FakeIncompleteCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
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
    assert trace[2]["event"]["evidence"]["events"][0]["name"] == "get_pose"


def test_capsule_trial_writes_step_metrics_jsonl(tmp_path):
    _run_capsule_trial(
        env=FakeCapsuleEnv(),
        trial=1,
        args=SimpleNamespace(model="test", use_oracle_code=False),
        config={
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
