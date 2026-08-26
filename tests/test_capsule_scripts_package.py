from __future__ import annotations

import importlib
from pathlib import Path


def test_repository_scripts_package_wins_over_verl_scripts() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    scripts_package = importlib.import_module("scripts")

    assert scripts_package.__file__ is not None
    assert Path(scripts_package.__file__).resolve() == repository_root / "scripts" / "__init__.py"
    importlib.import_module("scripts.capsule_rl.launch_owned_services")
    importlib.import_module("scripts.capsule_rl.cube_lift_privileged_replay_smoke")
