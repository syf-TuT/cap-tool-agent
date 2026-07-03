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


def test_state_first_tool_yaml_uses_true_state_without_vision_servers():
    data = yaml.safe_load(
        Path(
            "env_configs/cube_stack/franka_robosuite_cube_stack_tool_state_first.yaml"
        ).read_text()
    )

    assert data["agent_mode"] == "tool"
    assert data["max_tool_steps"] > 0
    assert data["env"]["cfg"]["apis"] == ["FrankaStateControlApi"]
    assert data["use_img_differencing"] is False

    server_targets = [server["_target_"] for server in data["api_servers"]]
    assert server_targets == ["capx.serving.launch_pyroki_server.main"]

    prompt = data["env"]["cfg"]["prompt"]
    assert "cubeA_pos" in prompt
    assert "cubeA_quat" in prompt
    assert "cubeB_pos" in prompt
    assert "cubeB_quat" in prompt
    assert "segment" not in prompt.lower()
