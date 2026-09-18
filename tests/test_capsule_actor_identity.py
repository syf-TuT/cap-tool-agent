from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient

from capx.rl.capsule import actor_identity
from capx.rl.capsule.actor_identity import (
    ActorIdentityError,
    actor_binding_sha256,
    build_actor_identity,
    create_actor_identity_app,
    verify_actor_identity_payload,
)
from scripts.capsule_rl import server_preflight


def _identity_config(tmp_path: Path) -> dict[str, object]:
    model = tmp_path / "Qwen2.5-Coder-7B-Instruct"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen2"}\n', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"model-bytes")
    verl = tmp_path / "verl"
    verl.mkdir()
    resolved = tmp_path / "resolved-verl.yaml"
    resolved.write_text(
        yaml.safe_dump(
            {
                "actor_rollout_ref": {
                    "model": {
                        "lora_rank": 16,
                        "lora_alpha": 32,
                        "target_modules": "all-linear",
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {
        "runtime": {
            "project_root": str(tmp_path),
            "program_model_path": str(model),
            "verl_source_path": str(verl),
            "verl_pinned_sha": "a" * 40,
            "verl_resolved_config_path": str(resolved),
        },
        "program_service": {
            "mode": "actor_identity",
            "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "api_key_env": "CAPX_PROGRAM_API_KEY",
        },
    }


def test_actor_identity_binds_model_tree_lora_verl_and_resolved_config(
    tmp_path: Path,
) -> None:
    config = _identity_config(tmp_path)

    identity = build_actor_identity(config)

    assert identity["schema_version"] == 1
    assert identity["service_role"] == "program_actor_identity"
    assert identity["serves_generation"] is False
    assert identity["program_model_file_count"] == 2
    assert len(identity["program_model_sha256"]) == 64
    assert identity["lora_rank"] == 16
    assert identity["lora_alpha"] == 32
    assert identity["lora_target_modules"] == ["all-linear"]
    assert identity["verl_pinned_sha"] == "a" * 40
    assert len(identity["verl_resolved_config_sha256"]) == 64
    assert len(identity["actor_binding_sha256"]) == 64

    changed = deepcopy(config)
    Path(changed["runtime"]["program_model_path"], "model.safetensors").write_bytes(
        b"changed-model-bytes"
    )
    assert build_actor_identity(changed)["actor_binding_sha256"] != identity[
        "actor_binding_sha256"
    ]


def test_actor_identity_rejects_symlinks_and_wrong_lora_contract(tmp_path: Path) -> None:
    config = _identity_config(tmp_path)
    model = Path(config["runtime"]["program_model_path"])
    link = model / "linked-config.json"
    try:
        link.symlink_to(model / "config.json")
    except OSError:
        pytest.skip("filesystem does not allow symlinks")

    with pytest.raises(ActorIdentityError, match="symlink"):
        build_actor_identity(config)

    link.unlink()
    resolved = Path(config["runtime"]["verl_resolved_config_path"])
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    payload["actor_rollout_ref"]["model"]["lora_rank"] = 8
    resolved.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ActorIdentityError, match="rank=16"):
        build_actor_identity(config)


@pytest.mark.parametrize("bad_target", [["all-linear", 7], ["all-linear", ""]])
def test_actor_identity_rejects_malformed_target_module_sequences(
    bad_target: list[object],
    tmp_path: Path,
) -> None:
    config = _identity_config(tmp_path)
    resolved = Path(config["runtime"]["verl_resolved_config_path"])
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    payload["actor_rollout_ref"]["model"]["target_modules"] = bad_target
    resolved.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ActorIdentityError, match="target_modules"):
        build_actor_identity(config)


def test_actor_identity_rejects_symlinked_parent_of_configured_model(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    config = _identity_config(real_root)
    linked_root = tmp_path / "linked"
    try:
        linked_root.symlink_to(real_root, target_is_directory=True)
    except OSError:
        pytest.skip("filesystem does not allow directory symlinks")
    config["runtime"]["program_model_path"] = str(
        linked_root / "Qwen2.5-Coder-7B-Instruct"
    )

    with pytest.raises(ActorIdentityError, match="symlink"):
        build_actor_identity(config)


def test_actor_identity_parent_replacement_cannot_redirect_model_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted_parent = tmp_path / "trusted-parent"
    trusted_parent.mkdir()
    config = _identity_config(tmp_path)
    model_name = "Qwen2.5-Coder-7B-Instruct"
    original_model = tmp_path / model_name
    original_model.rename(trusted_parent / model_name)
    config["runtime"]["program_model_path"] = str(trusted_parent / model_name)

    attacker_parent = tmp_path / "attacker-parent"
    attacker_parent.mkdir()
    attacker_model = attacker_parent / model_name
    attacker_model.mkdir()
    (attacker_model / "config.json").write_text(
        '{"model_type":"attacker"}\n', encoding="utf-8"
    )
    (attacker_model / "model.safetensors").write_bytes(b"attacker-model-bytes")
    displaced_parent = tmp_path / "displaced-trusted-parent"

    original_tree_identity = actor_identity._model_tree_identity
    swapped = False

    def swap_parent_before_hash(*args: object, **kwargs: object) -> tuple[int, str]:
        nonlocal swapped
        if not swapped:
            trusted_parent.rename(displaced_parent)
            attacker_parent.rename(trusted_parent)
            swapped = True
        return original_tree_identity(*args, **kwargs)

    monkeypatch.setattr(actor_identity, "_model_tree_identity", swap_parent_before_hash)

    with pytest.raises(ActorIdentityError, match="changed|replaced"):
        build_actor_identity(config)

    assert swapped is True


def test_actor_identity_finally_revalidates_an_early_model_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _identity_config(tmp_path)
    early_file = Path(config["runtime"]["program_model_path"]) / "config.json"
    trusted = early_file.read_bytes()
    original_scan = actor_identity._snapshot_regular_tree
    raced = False

    def mutate_after_final_metadata_scan(*args: object, **kwargs: object):
        nonlocal raced
        nodes = original_scan(*args, **kwargs)
        if kwargs.get("digest") is None and not raced:
            raced = True
            early_file.write_bytes(b'{"model_type":"attacker"}\n')
            early_file.write_bytes(trusted)
        return nodes

    monkeypatch.setattr(
        actor_identity, "_snapshot_regular_tree", mutate_after_final_metadata_scan
    )

    with pytest.raises(ActorIdentityError, match="changed after it was scanned"):
        build_actor_identity(config)

    assert raced is True
    assert early_file.read_bytes() == trusted


def test_actor_identity_cli_builds_from_one_stable_config_a_to_b_to_a_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _identity_config(tmp_path)
    config_path = tmp_path / "actor-identity.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    trusted = config_path.read_bytes()
    replacement_config = deepcopy(config)
    replacement_config["program_service"]["model"] = "attacker-model"
    replacement = yaml.safe_dump(replacement_config, sort_keys=False).encode()
    original_snapshot = actor_identity.read_stable_regular_file
    original_read_text = Path.read_text
    observed_models: list[str] = []

    def snapshot_during_a_to_b_to_a(candidate: str | Path, *, label: str):
        snapshot = original_snapshot(candidate, label=label)
        if label == "actor identity config":
            config_path.write_bytes(replacement)
            config_path.write_bytes(trusted)
        return snapshot

    def read_attacker_then_restore(self: Path, *args: object, **kwargs: object) -> str:
        if self.resolve() != config_path.resolve():
            return original_read_text(self, *args, **kwargs)
        config_path.write_bytes(replacement)
        payload = original_read_text(self, *args, **kwargs)
        config_path.write_bytes(trusted)
        return payload

    def capture_identity(received_config):
        observed_models.append(str(received_config["program_service"]["model"]))
        return build_actor_identity(config)

    monkeypatch.setattr(
        actor_identity, "read_stable_regular_file", snapshot_during_a_to_b_to_a
    )
    monkeypatch.setattr(Path, "read_text", read_attacker_then_restore)
    monkeypatch.setattr(actor_identity, "build_actor_identity", capture_identity)
    monkeypatch.setattr(actor_identity, "create_actor_identity_app", lambda *_a, **_k: object())
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setenv("CAPX_PROGRAM_API_KEY", "identity-secret")

    actor_identity.main(["--config", str(config_path)])

    assert config_path.read_bytes() == trusted
    assert observed_models == ["Qwen/Qwen2.5-Coder-7B-Instruct"]


def test_actor_identity_service_requires_bearer_and_has_no_generation_route(
    tmp_path: Path,
) -> None:
    identity = build_actor_identity(_identity_config(tmp_path))
    client = TestClient(create_actor_identity_app(identity, bearer_token="identity-secret"))

    assert client.get("/v1/capx/actor-identity").status_code == 401
    assert (
        client.get(
            "/v1/capx/actor-identity",
            headers={"Authorization": "Bearer wrong"},
        ).status_code
        == 403
    )
    response = client.get(
        "/v1/capx/actor-identity",
        headers={"Authorization": "Bearer identity-secret"},
    )
    assert response.status_code == 200
    assert response.json() == identity
    assert (
        client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer identity-secret"},
            json={"messages": []},
        ).status_code
        == 404
    )


@pytest.mark.parametrize(
    ("field_name", "bad_value", "message"),
    [
        ("schema_version", True, "schema_version"),
        ("service_role", "generation", "service_role"),
        ("serves_generation", True, "serves_generation"),
        ("program_model_file_count", True, "file_count"),
        ("lora_target_modules", ["q_proj"], "target_modules"),
    ],
)
def test_actor_identity_service_rejects_rebound_but_semantically_invalid_identity(
    field_name: str,
    bad_value: object,
    message: str,
    tmp_path: Path,
) -> None:
    invalid = build_actor_identity(_identity_config(tmp_path))
    invalid[field_name] = bad_value
    invalid["actor_binding_sha256"] = actor_binding_sha256(invalid)

    with pytest.raises(ActorIdentityError, match=message):
        create_actor_identity_app(invalid, bearer_token="identity-secret")


def test_gate_cross_validation_rejects_any_actor_identity_drift(tmp_path: Path) -> None:
    expected = build_actor_identity(_identity_config(tmp_path))
    verify_actor_identity_payload(expected, expected)

    for field in (
        "model",
        "program_model_sha256",
        "lora_rank",
        "lora_alpha",
        "lora_target_modules",
        "verl_pinned_sha",
        "verl_resolved_config_sha256",
        "actor_binding_sha256",
    ):
        drifted = deepcopy(expected)
        drifted[field] = 17 if field in {"lora_rank", "lora_alpha"} else "drift"
        with pytest.raises(ActorIdentityError, match=field):
            verify_actor_identity_payload(drifted, expected)


class _IdentityResponse:
    def __init__(self, body: bytes, *, url: str, content_length: int | None = None) -> None:
        self.body = body
        self.url = url
        self.status = 200
        self.headers = (
            {} if content_length is None else {"Content-Length": str(content_length)}
        )

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        return self.body

    def geturl(self) -> str:
        return self.url


def test_gate_identity_fetch_uses_bearer_and_rejects_redirect_or_oversize() -> None:
    endpoint = "http://127.0.0.1:8101/v1"
    identity_url = endpoint + "/capx/actor-identity"
    body = b'{"schema_version":1}'
    requests: list[object] = []

    class _Opener:
        def __init__(self, response: _IdentityResponse) -> None:
            self.response = response

        def open(self, request, *, timeout: float):
            assert timeout == 5.0
            requests.append(request)
            return self.response

    payload = server_preflight._fetch_actor_identity(
        endpoint,
        "identity-secret",
        opener=_Opener(_IdentityResponse(body, url=identity_url)),
    )
    assert payload == {"schema_version": 1}
    assert requests[0].full_url == identity_url
    assert requests[0].get_header("Authorization") == "Bearer identity-secret"

    with pytest.raises(server_preflight.GateArtifactError, match="redirect"):
        server_preflight._fetch_actor_identity(
            endpoint,
            "identity-secret",
            opener=_Opener(
                _IdentityResponse(body, url="http://wrong.invalid/actor-identity")
            ),
        )
    with pytest.raises(server_preflight.GateArtifactError, match="too large"):
        server_preflight._fetch_actor_identity(
            endpoint,
            "identity-secret",
            opener=_Opener(
                _IdentityResponse(body, url=identity_url, content_length=65_537)
            ),
        )
