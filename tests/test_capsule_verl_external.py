from __future__ import annotations

import importlib
import sys

import pytest

from capx.rl.capsule import compat
from capx.rl.capsule import policy_loss


MODULE_NAME = "capx.rl.capsule.verl_external"


def test_verl_external_registers_loss_in_each_importing_worker(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setenv("CAPX_PINNED_VERL_SOURCE_PATH", "/pinned/verl-source")
    monkeypatch.setenv("CAPX_PINNED_VERL_SHA", "a" * 40)
    monkeypatch.setattr(
        compat,
        "check_verl_compatibility",
        lambda path, sha: calls.append(f"git:{path}:{sha}"),
    )
    monkeypatch.setattr(
        compat,
        "verify_imported_verl_path",
        lambda path: calls.append(f"verify:{path}"),
    )
    monkeypatch.setattr(
        policy_loss,
        "register_capsule_critique_policy_loss",
        lambda: calls.append("register") or True,
    )
    sys.modules.pop(MODULE_NAME, None)

    module = importlib.import_module(MODULE_NAME)

    assert calls == [
        f"git:/pinned/verl-source:{'a' * 40}",
        "verify:/pinned/verl-source",
        "register",
    ]
    assert callable(module.initialize_verl_external)
    sys.modules.pop(MODULE_NAME, None)


def test_verl_external_fails_closed_when_verl_is_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_PINNED_VERL_SOURCE_PATH", "/pinned/verl-source")
    monkeypatch.setenv("CAPX_PINNED_VERL_SHA", "a" * 40)
    monkeypatch.setattr(compat, "check_verl_compatibility", lambda _path, _sha: None)
    monkeypatch.setattr(compat, "verify_imported_verl_path", lambda _path: None)
    monkeypatch.setattr(
        policy_loss,
        "register_capsule_critique_policy_loss",
        lambda: False,
    )
    sys.modules.pop(MODULE_NAME, None)

    with pytest.raises(RuntimeError, match="VeRL is unavailable"):
        importlib.import_module(MODULE_NAME)

    sys.modules.pop(MODULE_NAME, None)


def test_verl_external_requires_worker_pinned_path_marker(monkeypatch) -> None:
    monkeypatch.delenv("CAPX_PINNED_VERL_SOURCE_PATH", raising=False)
    monkeypatch.delenv("CAPX_PINNED_VERL_SHA", raising=False)
    monkeypatch.setattr(
        policy_loss,
        "register_capsule_critique_policy_loss",
        lambda: (_ for _ in ()).throw(AssertionError("must verify before registration")),
    )
    sys.modules.pop(MODULE_NAME, None)

    with pytest.raises(RuntimeError, match="CAPX_PINNED_VERL_SOURCE_PATH"):
        importlib.import_module(MODULE_NAME)

    sys.modules.pop(MODULE_NAME, None)


def test_verl_external_requires_worker_pinned_sha_marker(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_PINNED_VERL_SOURCE_PATH", "/pinned/verl-source")
    monkeypatch.delenv("CAPX_PINNED_VERL_SHA", raising=False)
    monkeypatch.setattr(
        policy_loss,
        "register_capsule_critique_policy_loss",
        lambda: (_ for _ in ()).throw(AssertionError("must verify before registration")),
    )
    sys.modules.pop(MODULE_NAME, None)

    with pytest.raises(RuntimeError, match="CAPX_PINNED_VERL_SHA"):
        importlib.import_module(MODULE_NAME)

    sys.modules.pop(MODULE_NAME, None)
