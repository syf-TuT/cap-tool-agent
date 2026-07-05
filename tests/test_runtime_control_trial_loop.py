import json
from pathlib import Path
from types import SimpleNamespace

from capx.envs.trial import _execute_runtime_action, _run_capsule_trial, _run_single_trial
from capx.runtime_control.executor import CapsuleExecutor
from capx.runtime_control.schema import CodeRegion, RuntimeAction
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


def test_inspect_variables_requires_names():
    event = _execute_runtime_action(
        RuntimeAction("inspect_variables", {"region_id": "region_1"}),
        CapsuleExecutor(base_globals={}),
        "x = 1\n",
        {},
    )

    assert event.status == "invalid"
    assert "args.names" in event.message
