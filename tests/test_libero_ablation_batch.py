from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from capx.envs.scripts import run_libero_batch as batch


class FakeSuite:
    n_tasks = 2

    def get_task(self, task_id):
        return SimpleNamespace(name=f"task_{task_id}")

    def get_task_init_states(self, task_id):
        return [0, 1, 2] if task_id == 0 else [0, 1]


@pytest.mark.parametrize("selected", [False, True])
def test_ablation_batch_uses_matching_trials_and_separate_outputs(monkeypatch, tmp_path, selected):
    suites = ["libero_goal"] if selected else list(batch.STANDARD_SUITES)
    models = ["provider/model_a", "provider/model_b"]
    args = batch.LiberoBatchLaunchArgs(
        ablation=True,
        suites=suites if selected else None,
        task_ids=[1] if selected else None,
        total_trials=1 if selected else None,
        models=models,
        output_dir=str(tmp_path),
    )
    launches = []
    monkeypatch.setattr(batch, "_load_benchmark_dict", lambda: dict.fromkeys(suites, FakeSuite))
    monkeypatch.setattr(batch, "launch_main", launches.append)
    batch.main(args)

    expected_task_ids = [1] if selected else [0, 1]
    expected_runs = {
        (model, suite, task_id, group)
        for model in models
        for suite in suites
        for task_id in expected_task_ids
        for group in batch.ABLATION_CONFIGS
    }
    actual_runs = set()
    for launch in launches:
        config_path = Path(launch.config_path)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        low_level = config["env"]["cfg"]["low_level"]
        suite, task_id = low_level["suite_name"], low_level["task_id"]
        group = config_path.parent.parent.name
        model_dir = launch.model.replace("/", "_")
        run_dir = tmp_path / suite / f"task_{task_id}" / group
        assert config_path == run_dir / model_dir / "config.yaml"
        assert Path(config["output_dir"]) == run_dir / "run"
        assert config["trials"] == (1 if selected else 3 - task_id)
        assert launch.total_trials == args.total_trials
        assert launch.temperature == args.temperature
        assert launch.max_tokens == args.max_tokens
        actual_runs.add((launch.model, suite, task_id, group))
    assert len(launches) == len(expected_runs)
    assert actual_runs == expected_runs


@pytest.mark.parametrize(
    ("states", "overrides", "error"),
    [
        ([], {}, "No initial states"),
        (None, {}, "Cannot load initial states"),
        ([0], {"total_trials": 2}, "--total-trials"),
        ([0], {"task_ids": [2]}, "task ID"),
        ([0], {"suites": ["libero_10"]}, "--ablation requires suites"),
    ],
)
def test_ablation_batch_validates_before_launch(monkeypatch, tmp_path, states, overrides, error):
    def get_states(self, task_id):
        return states if task_id == 1 else [0, 1, 2]

    monkeypatch.setattr(FakeSuite, "get_task_init_states", get_states)
    monkeypatch.setattr(batch, "_load_benchmark_dict", lambda: {"libero_goal": FakeSuite})
    monkeypatch.setattr(batch, "launch_main", lambda _: pytest.fail("Unexpected launch"))
    options = {"suites": ["libero_goal"], **overrides}
    args = batch.LiberoBatchLaunchArgs(ablation=True, output_dir=str(tmp_path), **options)

    with pytest.raises(ValueError, match=error):
        batch.main(args)
    assert not list(tmp_path.iterdir())


def test_ordinary_batch_keeps_the_pro_defaults(monkeypatch, tmp_path):
    args = batch.LiberoBatchLaunchArgs(output_dir=str(tmp_path))
    launches = []
    monkeypatch.setattr(batch, "_load_benchmark_dict", lambda: dict.fromkeys(batch.PRO_SUITES, FakeSuite))
    monkeypatch.setattr(batch, "launch_main", launches.append)

    batch.main(args)

    assert len(launches) == 2 * len(batch.PRO_SUITES)
    for launch in launches:
        config_path = Path(launch.config_path)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        cfg = config["env"]["cfg"]
        assert config_path.parent.parent.name in batch.PRO_SUITES
        assert cfg["low_level"]["suite_name"] == config_path.parent.parent.name
        assert cfg["apis"] == ["FrankaLiberoApiReducedSkillLibrary"]
        assert config["use_parallel_ensemble"] is True
