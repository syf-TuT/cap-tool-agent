from pathlib import Path

import yaml


CONFIGS = [
    "env_configs/cube_stack/franka_robosuite_cube_stack_tool_vdm.yaml",
    "env_configs/cube_lifting/franka_robosuite_cube_lifting_tool_vdm.yaml",
    "env_configs/cube_restack/franka_robosuite_cube_restack_tool_vdm.yaml",
]


def test_tool_yaml_configs_define_tool_mode():
    for path in CONFIGS:
        data = yaml.safe_load(Path(path).read_text())
        assert data["agent_mode"] == "tool"
        assert data["max_tool_steps"] > 0
        assert data["env"]["cfg"]["apis"] == ["FrankaControlApiReducedSkillLibrary"]
