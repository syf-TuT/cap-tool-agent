import inspect
from types import SimpleNamespace

import capx.integrations as integrations
from capx.integrations import base_api
from capx.integrations.franka import libero as libero_module
from capx.integrations.vision import molmo


def test_instantiate_api_preserves_env_only_factories():
    factory_name = "TestEnvOnlyApiFactory"
    low_level_env = object()
    captured = []

    def env_only_factory(env):
        captured.append(env)
        return "legacy-api"

    base_api.register_api(factory_name, env_only_factory)

    api = base_api.instantiate_api(factory_name, low_level_env, SimpleNamespace())

    assert api == "legacy-api"
    assert captured == [low_level_env]


def test_franka_libero_registry_passes_molmo_config(monkeypatch):
    captured = {}

    def fake_api(env, **kwargs):
        captured["env"] = env
        captured.update(kwargs)
        return "configured-api"

    monkeypatch.setattr(integrations, "FrankaLiberoApi", fake_api)
    low_level_env = object()
    cfg = SimpleNamespace(
        molmo_base_url="http://molmo.example.test/v1",
        molmo_model_name="test/Molmo",
    )

    api = base_api.instantiate_api("FrankaLiberoApi", low_level_env, cfg)

    assert api == "configured-api"
    assert captured == {
        "env": low_level_env,
        "use_sam3": True,
        "molmo_base_url": "http://molmo.example.test/v1",
        "molmo_model_name": "test/Molmo",
    }


def test_franka_libero_registry_factory_supports_direct_env_call(monkeypatch):
    captured = {}

    def fake_api(env, **kwargs):
        captured["env"] = env
        captured.update(kwargs)
        return "default-api"

    monkeypatch.setattr(integrations, "FrankaLiberoApi", fake_api)
    low_level_env = object()

    api = base_api.get_api("FrankaLiberoApi")(low_level_env)

    assert api == "default-api"
    assert captured == {
        "env": low_level_env,
        "use_sam3": True,
        "molmo_base_url": None,
        "molmo_model_name": None,
    }


def test_franka_libero_constructor_passes_molmo_endpoint(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "capx.integrations.motion.pyroki_context.get_pyroki_context",
        lambda *args, **kwargs: SimpleNamespace(robot=object(), target_link_name="panda_hand"),
    )
    monkeypatch.setattr(libero_module, "init_sam3", lambda: object())
    monkeypatch.setattr(libero_module, "init_sam3_point_prompt", lambda: object())
    monkeypatch.setattr(libero_module, "init_contact_graspnet", lambda: object())
    monkeypatch.setattr(libero_module, "init_contact_graspnet_point_clouds", lambda: object())
    monkeypatch.setattr(libero_module, "init_pyroki", lambda: object())

    def fake_init_molmo(*, model_name, base_url):
        captured.update(model_name=model_name, base_url=base_url)
        return object()

    monkeypatch.setattr(libero_module, "init_molmo", fake_init_molmo)

    libero_module.FrankaLiberoApi(
        object(),
        molmo_base_url="http://molmo.example.test/v1",
        molmo_model_name="test/Molmo",
    )

    assert captured == {
        "model_name": "test/Molmo",
        "base_url": "http://molmo.example.test/v1",
    }

    captured.clear()
    libero_module.FrankaLiberoApi(
        object(),
        molmo_base_url=None,
        molmo_model_name=None,
    )
    assert captured == {
        "model_name": molmo.DEFAULT_MOLMO_MODEL_NAME,
        "base_url": molmo.DEFAULT_MOLMO_BASE_URL,
    }


def test_molmo_defaults_are_exported_and_used_by_initializer():
    parameters = inspect.signature(molmo.init_molmo).parameters

    assert molmo.DEFAULT_MOLMO_BASE_URL == "http://127.0.0.1:8122/v1"
    assert molmo.DEFAULT_MOLMO_MODEL_NAME == "allenai/Molmo2-8B"
    assert parameters["base_url"].default == molmo.DEFAULT_MOLMO_BASE_URL
    assert parameters["model_name"].default == molmo.DEFAULT_MOLMO_MODEL_NAME
