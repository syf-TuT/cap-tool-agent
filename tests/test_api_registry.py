from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from capx.integrations import base_api


def test_reregistered_api_replaces_factory_without_stale_cache():
    name = "TestReregisteredApi"
    env = object()
    base_api.register_api(name, lambda received_env: ("old", received_env))

    assert base_api.get_api(name)(env) == ("old", env)

    base_api.register_api(name, lambda received_env: ("new", received_env))

    assert base_api.get_api(name)(env) == ("new", env)
    assert base_api.instantiate_api(name, env, SimpleNamespace()) == ("new", env)


def test_registration_snapshot_keeps_factories_in_one_immutable_record():
    name = "TestAtomicApiRegistration"

    def legacy_factory(env):
        return ("legacy", env)

    def config_factory(env, cfg):
        return ("configured", env, cfg)

    base_api.register_api(name, legacy_factory, config_factory=config_factory)

    registration = base_api._get_api_registration(name)

    assert registration.factory is legacy_factory
    assert registration.config_factory is config_factory
    with pytest.raises(FrozenInstanceError):
        registration.factory = config_factory
