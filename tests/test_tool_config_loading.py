from types import SimpleNamespace

import yaml

from capx.utils.launch_utils import _load_config


def test_load_config_reads_tool_mode_fields(tmp_path):
    cfg_path = tmp_path / "tool.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "env": {"_target_": "fake.Env"},
                "agent_mode": "tool",
                "max_tool_steps": 7,
                "tool_feedback_level": "repair_hint",
                "trials": 1,
            }
        )
    )
    args = SimpleNamespace(
        config_path=str(cfg_path),
        total_trials=None,
        num_workers=None,
        record_video=None,
        output_dir=None,
        use_oracle_code=None,
        use_visual_feedback=None,
        use_img_differencing=None,
        use_video_differencing=None,
        use_wrist_camera=None,
        use_parallel_ensemble=None,
        use_multimodel=None,
        web_ui=None,
        web_ui_port=None,
        server_url="http://127.0.0.1:8110/chat/completions",
        visual_differencing_model="google/gemini-3.1-pro-preview",
        visual_differencing_model_server_url="http://127.0.0.1:8110/chat/completions",
        visual_differencing_model_api_key=None,
    )

    _, config, _ = _load_config(args)

    assert config["agent_mode"] == "tool"
    assert config["max_tool_steps"] == 7
    assert config["tool_feedback_level"] == "repair_hint"
