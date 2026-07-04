import json
from types import SimpleNamespace

from capx.envs.trial import _execute_runtime_action, _run_capsule_trial
from capx.runtime_control.executor import CapsuleExecutor
from capx.runtime_control.schema import CodeRegion, RuntimeAction
from capx.runtime_control.trace import wrap_function_for_trace


class FakeApi:
    def __init__(self):
        self.moved = False

    def functions(self):
        return {"get_pose": self.get_pose, "move_to": self.move_to}

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
