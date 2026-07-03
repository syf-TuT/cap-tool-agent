from types import SimpleNamespace

from capx.envs.trial import _run_capsule_trial


class FakeCapsuleEnv:
    oracle_code = ""

    def reset(self, *, seed=None, options=None):
        return {
            "full_prompt": [
                {"role": "system", "content": "x"},
                {"role": "user", "content": [{"type": "text", "text": "task"}]},
            ]
        }, {}

    def _build_capsule_globals(self):
        return {}

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
