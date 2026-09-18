import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from capx.utils.eval_utils import ExperimentParser


def _load_script_module(name: str) -> ModuleType:
    script_path = (
        Path(__file__).parents[1] / "scripts" / "skill_library_compilation" / f"{name}.py"
    )
    spec = importlib.util.spec_from_file_location(f"test_{name}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT_MODULES = [
    _load_script_module(name)
    for name in ("parse_outputs", "summarize_analysis", "compile_skill_library")
]


def _make_trial_dir(experiment_dir: Path, trial: int) -> None:
    (experiment_dir / f"trial_{trial}_sandboxrc_0_reward_1.0_taskcompleted_1").mkdir()


def test_experiment_parser_prefers_trial_prompt_then_legacy_fallback(tmp_path):
    legacy_prompt = tmp_path / "initial_prompt.txt"
    trial_prompt = tmp_path / "initial_prompt_trial_00.txt"
    legacy_prompt.write_text("legacy", encoding="utf-8")
    trial_prompt.write_text("trial zero", encoding="utf-8")
    _make_trial_dir(tmp_path, 0)
    _make_trial_dir(tmp_path, 1)

    trials = ExperimentParser(tmp_path).parse_trials()

    assert trials[0].initial_prompt_txt_path == trial_prompt.absolute()
    assert trials[1].initial_prompt_txt_path == legacy_prompt.absolute()


@pytest.mark.parametrize(
    "module",
    SCRIPT_MODULES,
)
def test_skill_library_scripts_detect_trial_scoped_prompt_artifacts(tmp_path, module):
    (tmp_path / "initial_prompt_trial_07.txt").write_text("prompt", encoding="utf-8")

    assert module.is_experiment_dir(tmp_path)
