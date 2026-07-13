from pathlib import Path

import yaml


def test_two_arm_lift_has_task_local_reward_guard_without_rollback():
    config_path = (
        Path(__file__).parents[1]
        / "env_configs"
        / "two_arm_lift"
        / "franka_robosuite_two_arm_lift.yaml"
    )
    config = yaml.safe_load(config_path.read_text())

    assert config["capsule_reward_drop_guard_min_best_reward"] == 0.05
    assert config["capsule_reward_drop_guard_threshold"] == 0.03
    assert "rollback_policy" not in config
    assert "capsule_rollback_policy" not in config
