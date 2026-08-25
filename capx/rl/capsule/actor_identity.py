"""Authenticated, generation-free identity binding for the Capsule Program actor."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .provenance import project_root_path

IDENTITY_ROUTE = "/v1/capx/actor-identity"
IDENTITY_SCHEMA_VERSION = 1
ACTOR_BINDING_FIELDS = (
    "schema_version",
    "service_role",
    "serves_generation",
    "model",
    "program_model_path",
    "program_model_file_count",
    "program_model_sha256",
    "lora_rank",
    "lora_alpha",
    "lora_target_modules",
    "verl_source_path",
    "verl_pinned_sha",
    "verl_resolved_config_path",
    "verl_resolved_config_sha256",
)


class ActorIdentityError(ValueError):
    """The Program actor identity cannot be established or does not match."""


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ActorIdentityError(f"{field_name} must be a mapping")
    return value


def _lexical_path(config: Mapping[str, Any], value: object, field_name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ActorIdentityError(f"{field_name} must be a non-empty path")
    raw = Path(value).expanduser()
    path = raw if raw.is_absolute() else project_root_path(config) / raw
    path = Path(os.path.abspath(path))
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ActorIdentityError(
                f"{field_name} must not contain a symlink component: {component}"
            )
    return path


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _scan_regular_tree(
    path: Path,
) -> tuple[
    tuple[tuple[str, str, tuple[int, int, int, int, int, int]], ...],
    tuple[tuple[Path, str, tuple[int, int, int, int, int, int]], ...],
]:
    """Snapshot a tree without following links and reject every special node."""

    nodes: list[tuple[str, str, tuple[int, int, int, int, int, int]]] = []
    files: list[tuple[Path, str, tuple[int, int, int, int, int, int]]] = []

    def visit(directory: Path, relative_root: Path) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as error:
            raise ActorIdentityError(f"cannot scan program model tree: {error}") from error
        for entry in entries:
            relative = (relative_root / entry.name).as_posix()
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ActorIdentityError(
                    f"cannot stat program model tree entry {relative}: {error}"
                ) from error
            identity = _stat_identity(entry_stat)
            if stat.S_ISLNK(entry_stat.st_mode):
                raise ActorIdentityError(
                    f"program model tree must not contain symlinks: {entry.path}"
                )
            entry_path = Path(entry.path)
            if stat.S_ISDIR(entry_stat.st_mode):
                nodes.append((relative, "directory", identity))
                visit(entry_path, relative_root / entry.name)
            elif stat.S_ISREG(entry_stat.st_mode):
                nodes.append((relative, "file", identity))
                files.append((entry_path, relative, identity))
            else:
                raise ActorIdentityError(
                    f"program model tree contains a non-regular node: {entry.path}"
                )

    try:
        root_stat = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ActorIdentityError(f"cannot stat program model directory: {error}") from error
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ActorIdentityError(f"program model must be an existing directory: {path}")
    nodes.append(("", "directory", _stat_identity(root_stat)))
    visit(path, Path())
    return tuple(nodes), tuple(files)


def _read_stable_regular_file(
    path: Path,
    expected_identity: tuple[int, int, int, int, int, int] | None,
    *,
    label: str,
) -> bytes:
    """Read one unchanged regular file and reject final-component link substitution."""

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ActorIdentityError(f"cannot open {label}: {error}") from error
    try:
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode):
            raise ActorIdentityError(f"{label} must be a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    except OSError as error:
        raise ActorIdentityError(f"cannot read {label}: {error}") from error
    finally:
        os.close(descriptor)
    try:
        lexical_after = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ActorIdentityError(f"cannot restat {label}: {error}") from error
    observed = {
        _stat_identity(opened_before),
        _stat_identity(opened_after),
        _stat_identity(lexical_after),
    }
    if len(observed) != 1 or (
        expected_identity is not None and _stat_identity(opened_before) != expected_identity
    ):
        raise ActorIdentityError(f"{label} changed while it was being read: {path}")
    body = b"".join(chunks)
    if len(body) != opened_after.st_size:
        raise ActorIdentityError(f"{label} size changed while it was being read: {path}")
    return body


def _hash_stable_regular_file(
    path: Path,
    expected_identity: tuple[int, int, int, int, int, int],
    *,
    label: str,
    digest: Any,
) -> None:
    """Stream one unchanged file into a digest without retaining model bytes in RAM."""

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ActorIdentityError(f"cannot open {label}: {error}") from error
    total = 0
    try:
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode):
            raise ActorIdentityError(f"{label} must be a regular file: {path}")
        if _stat_identity(opened_before) != expected_identity:
            raise ActorIdentityError(f"{label} changed before it was read: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
        opened_after = os.fstat(descriptor)
    except OSError as error:
        raise ActorIdentityError(f"cannot read {label}: {error}") from error
    finally:
        os.close(descriptor)
    try:
        lexical_after = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ActorIdentityError(f"cannot restat {label}: {error}") from error
    if (
        _stat_identity(opened_after) != expected_identity
        or _stat_identity(lexical_after) != expected_identity
        or total != opened_after.st_size
    ):
        raise ActorIdentityError(f"{label} changed while it was being hashed: {path}")


def _model_tree_identity(path: Path) -> tuple[int, str]:
    before, files = _scan_regular_tree(path)
    if not files:
        raise ActorIdentityError("program model directory must contain regular files")
    digest = hashlib.sha256()
    for file_path, relative_text, expected_identity in files:
        relative = relative_text.encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(expected_identity[3].to_bytes(8, "big"))
        _hash_stable_regular_file(
            file_path,
            expected_identity,
            label=f"program model file {relative_text}",
            digest=digest,
        )
    after, _post_files = _scan_regular_tree(path)
    if after != before:
        raise ActorIdentityError("program model tree changed while it was being hashed")
    return len(files), digest.hexdigest()


def _normalized_lora_targets(value: object) -> list[str]:
    if isinstance(value, str):
        targets = [value.strip()] if value.strip() else []
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ActorIdentityError(
                "LoRA target_modules must contain only non-empty strings"
            )
        targets = [item.strip() for item in value]
    else:
        targets = []
    if targets != ["all-linear"]:
        raise ActorIdentityError("LoRA target_modules must resolve to all-linear")
    return targets


def actor_binding_sha256(identity: Mapping[str, Any]) -> str:
    missing = [field for field in ACTOR_BINDING_FIELDS if field not in identity]
    if missing:
        raise ActorIdentityError(
            "actor identity is missing binding field(s): " + ", ".join(missing)
        )
    canonical = json.dumps(
        {field: identity[field] for field in ACTOR_BINDING_FIELDS},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_actor_identity_payload(identity: Mapping[str, Any]) -> None:
    """Validate the complete identity schema and its canonical binding digest."""

    required = {*ACTOR_BINDING_FIELDS, "actor_binding_sha256"}
    missing = required.difference(identity)
    unexpected = set(identity).difference(required)
    if missing:
        raise ActorIdentityError(
            "actor identity is missing field(s): " + ", ".join(sorted(missing))
        )
    if unexpected:
        raise ActorIdentityError(
            "actor identity contains unexpected field(s): "
            + ", ".join(sorted(unexpected))
        )
    if type(identity["schema_version"]) is not int or identity["schema_version"] != 1:
        raise ActorIdentityError("actor identity schema_version must be 1")
    if identity["service_role"] != "program_actor_identity":
        raise ActorIdentityError("actor identity service_role is invalid")
    if identity["serves_generation"] is not False:
        raise ActorIdentityError("actor identity must record serves_generation=false")
    if not isinstance(identity["model"], str) or not identity["model"].strip():
        raise ActorIdentityError("actor identity model must be non-empty")
    for field_name in (
        "program_model_path",
        "verl_source_path",
        "verl_resolved_config_path",
    ):
        value = identity[field_name]
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ActorIdentityError(f"actor identity {field_name} must be an absolute path")
    file_count = identity["program_model_file_count"]
    if isinstance(file_count, bool) or not isinstance(file_count, int) or file_count < 1:
        raise ActorIdentityError("actor identity program_model_file_count must be positive")
    for field_name in (
        "program_model_sha256",
        "verl_resolved_config_sha256",
        "actor_binding_sha256",
    ):
        if not _lower_hex(identity[field_name], 64):
            raise ActorIdentityError(f"actor identity {field_name} must be lowercase SHA-256")
    if not _lower_hex(identity["verl_pinned_sha"], 40):
        raise ActorIdentityError("actor identity verl_pinned_sha must be a full lowercase Git SHA")
    if identity["lora_rank"] != 16 or type(identity["lora_rank"]) is not int:
        raise ActorIdentityError("actor identity lora_rank must be 16")
    if identity["lora_alpha"] != 32 or type(identity["lora_alpha"]) is not int:
        raise ActorIdentityError("actor identity lora_alpha must be 32")
    if identity["lora_target_modules"] != ["all-linear"]:
        raise ActorIdentityError(
            "actor identity lora_target_modules must be exactly ['all-linear']"
        )
    if actor_binding_sha256(identity) != identity["actor_binding_sha256"]:
        raise ActorIdentityError("actor identity field actor_binding_sha256 is invalid")


def build_actor_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the complete local actor binding without importing a model runtime."""

    runtime = _mapping(config.get("runtime"), "runtime")
    service = _mapping(config.get("program_service"), "program_service")
    if service.get("mode") != "actor_identity":
        raise ActorIdentityError("program_service.mode must be actor_identity")
    model_name = service.get("model")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ActorIdentityError("program_service.model must be non-empty")

    model_path = _lexical_path(
        config, runtime.get("program_model_path"), "runtime.program_model_path"
    )
    model_path = model_path.resolve()
    model_file_count, model_sha256 = _model_tree_identity(model_path)
    resolved_config_path = _lexical_path(
        config,
        runtime.get("verl_resolved_config_path"),
        "runtime.verl_resolved_config_path",
    )
    if not resolved_config_path.is_file():
        raise ActorIdentityError(
            f"resolved VeRL config must be an existing file: {resolved_config_path}"
        )
    resolved_config_bytes = _read_stable_regular_file(
        resolved_config_path,
        None,
        label="resolved VeRL config",
    )
    try:
        resolved_config = yaml.safe_load(resolved_config_bytes.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as error:
        raise ActorIdentityError(f"cannot parse resolved VeRL config: {error}") from error
    model_config = _mapping(
        _mapping(
            _mapping(resolved_config, "resolved VeRL config").get("actor_rollout_ref"),
            "resolved VeRL config.actor_rollout_ref",
        ).get("model"),
        "resolved VeRL config.actor_rollout_ref.model",
    )
    lora_rank = model_config.get("lora_rank")
    if isinstance(lora_rank, bool) or not isinstance(lora_rank, int) or lora_rank != 16:
        raise ActorIdentityError("resolved VeRL LoRA contract requires rank=16")
    lora_alpha = model_config.get("lora_alpha")
    if isinstance(lora_alpha, bool) or not isinstance(lora_alpha, int) or lora_alpha != 32:
        raise ActorIdentityError("resolved VeRL LoRA contract requires alpha=32")
    lora_targets = _normalized_lora_targets(model_config.get("target_modules"))

    verl_path = _lexical_path(
        config, runtime.get("verl_source_path"), "runtime.verl_source_path"
    )
    if not verl_path.is_dir():
        raise ActorIdentityError(f"VeRL source must be an existing directory: {verl_path}")
    pinned_sha = runtime.get("verl_pinned_sha")
    if (
        not isinstance(pinned_sha, str)
        or len(pinned_sha) != 40
        or any(character not in "0123456789abcdef" for character in pinned_sha)
    ):
        raise ActorIdentityError("runtime.verl_pinned_sha must be a lowercase full Git SHA")
    identity: dict[str, Any] = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "service_role": "program_actor_identity",
        "serves_generation": False,
        "model": model_name,
        "program_model_path": str(model_path),
        "program_model_file_count": model_file_count,
        "program_model_sha256": model_sha256,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_target_modules": lora_targets,
        "verl_source_path": str(verl_path.resolve()),
        "verl_pinned_sha": pinned_sha,
        "verl_resolved_config_path": str(resolved_config_path.resolve()),
        "verl_resolved_config_sha256": hashlib.sha256(resolved_config_bytes).hexdigest(),
    }
    identity["actor_binding_sha256"] = actor_binding_sha256(identity)
    validate_actor_identity_payload(identity)
    return identity


def verify_actor_identity_payload(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    """Require a service payload to equal an independently recomputed local identity."""

    validate_actor_identity_payload(expected)
    for field in (*ACTOR_BINDING_FIELDS, "actor_binding_sha256"):
        if actual.get(field) != expected.get(field):
            raise ActorIdentityError(f"actor identity field {field} does not match local bytes")
    validate_actor_identity_payload(actual)


def create_actor_identity_app(identity: Mapping[str, Any], *, bearer_token: str) -> Any:
    """Create a tiny FastAPI app with exactly one authenticated identity route."""

    if not isinstance(bearer_token, str) or not bearer_token:
        raise ActorIdentityError("actor identity bearer token must be non-empty")
    verified_identity = json.loads(
        json.dumps(dict(identity), ensure_ascii=False, sort_keys=True, allow_nan=False)
    )
    validate_actor_identity_payload(verified_identity)

    from fastapi import FastAPI, Header, HTTPException

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get(IDENTITY_ROUTE)
    def get_actor_identity(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Bearer authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        supplied = authorization.removeprefix("Bearer ")
        if not hmac.compare_digest(supplied, bearer_token):
            raise HTTPException(status_code=403, detail="invalid bearer token")
        return dict(verified_identity)

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the Capsule Program actor identity.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8101)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    identity = build_actor_identity(_mapping(config, "config"))
    service = _mapping(config.get("program_service"), "program_service")
    key_env = service.get("api_key_env")
    if not isinstance(key_env, str) or not key_env:
        raise ActorIdentityError("program_service.api_key_env must be non-empty")
    token = os.environ.get(key_env)
    if not token:
        raise ActorIdentityError(f"Program actor identity credential {key_env!r} is unset")
    app = create_actor_identity_app(identity, bearer_token=token)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()


__all__ = [
    "ACTOR_BINDING_FIELDS",
    "ActorIdentityError",
    "IDENTITY_ROUTE",
    "actor_binding_sha256",
    "build_actor_identity",
    "create_actor_identity_app",
    "validate_actor_identity_payload",
    "verify_actor_identity_payload",
]
