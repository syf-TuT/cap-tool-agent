"""Authenticated, generation-free identity binding for the Capsule Program actor."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .stable_io import (
    PinnedPath,
    StableFileSnapshot,
    StablePathError,
    full_stat_identity,
    pin_absolute_path,
    read_stable_regular_file,
)

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
    runtime = _mapping(config.get("runtime"), "runtime")
    configured_root = runtime.get("project_root")
    repository_root = Path(__file__).resolve().parents[3]
    if isinstance(configured_root, str) and configured_root:
        root_value = Path(configured_root).expanduser()
        project_root = (
            root_value
            if root_value.is_absolute()
            else repository_root / root_value
        )
    else:
        project_root = repository_root
    return Path(os.path.abspath(raw if raw.is_absolute() else project_root / raw))


def _tree_entry_descriptor(
    parent_descriptor: int,
    name: str,
    *,
    directory: bool,
    label: str,
) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    if directory:
        flags |= os.O_DIRECTORY
    descriptor: int | None = None
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        opened = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise ActorIdentityError(
            f"cannot open {label} without following symlinks: {error}"
        ) from error
    expected_kind = (
        stat.S_ISDIR(opened.st_mode) if directory else stat.S_ISREG(opened.st_mode)
    )
    if not expected_kind or full_stat_identity(named) != full_stat_identity(opened):
        os.close(descriptor)
        raise ActorIdentityError(f"{label} changed or was replaced while it was opened")
    return descriptor, opened


@dataclass(frozen=True)
class _HeldTreeEntry:
    relative: str
    descriptor: int
    parent_descriptor: int
    name: str
    identity: tuple[int, int, int, int, int, int]


def _snapshot_regular_tree(
    directory_descriptor: int,
    relative_root: Path,
    *,
    digest: Any | None,
    held_entries: list[_HeldTreeEntry] | None = None,
) -> tuple[tuple[str, str, tuple[int, int, int, int, int, int]], ...]:
    nodes: list[tuple[str, str, tuple[int, int, int, int, int, int]]] = []
    try:
        names = sorted(os.listdir(directory_descriptor))
    except OSError as error:
        raise ActorIdentityError(f"cannot list program model tree: {error}") from error
    for name in names:
        relative = (relative_root / name).as_posix()
        try:
            entry_stat = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise ActorIdentityError(
                f"cannot stat program model tree entry {relative}: {error}"
            ) from error
        identity = full_stat_identity(entry_stat)
        if stat.S_ISLNK(entry_stat.st_mode):
            raise ActorIdentityError(
                f"program model tree must not contain symlinks: {relative}"
            )
        if stat.S_ISDIR(entry_stat.st_mode):
            nodes.append((relative, "directory", identity))
            child_descriptor, opened = _tree_entry_descriptor(
                directory_descriptor,
                name,
                directory=True,
                label=f"program model directory {relative}",
            )
            if held_entries is not None:
                held_entries.append(
                    _HeldTreeEntry(
                        relative=relative,
                        descriptor=child_descriptor,
                        parent_descriptor=directory_descriptor,
                        name=name,
                        identity=identity,
                    )
                )
            try:
                nodes.extend(
                    _snapshot_regular_tree(
                        child_descriptor,
                        relative_root / name,
                        digest=digest,
                        held_entries=held_entries,
                    )
                )
                opened_after = os.fstat(child_descriptor)
                named_after = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise ActorIdentityError(
                    f"cannot restat program model directory {relative}: {error}"
                ) from error
            finally:
                if held_entries is None:
                    os.close(child_descriptor)
            if (
                full_stat_identity(opened) != identity
                or full_stat_identity(opened_after) != identity
                or full_stat_identity(named_after) != identity
            ):
                raise ActorIdentityError(
                    f"program model directory {relative} changed while it was scanned"
                )
        elif stat.S_ISREG(entry_stat.st_mode):
            nodes.append((relative, "file", identity))
            file_descriptor, opened = _tree_entry_descriptor(
                directory_descriptor,
                name,
                directory=False,
                label=f"program model file {relative}",
            )
            if held_entries is not None:
                held_entries.append(
                    _HeldTreeEntry(
                        relative=relative,
                        descriptor=file_descriptor,
                        parent_descriptor=directory_descriptor,
                        name=name,
                        identity=identity,
                    )
                )
            total = 0
            try:
                if digest is not None:
                    relative_bytes = relative.encode("utf-8")
                    digest.update(len(relative_bytes).to_bytes(8, "big"))
                    digest.update(relative_bytes)
                    digest.update(entry_stat.st_size.to_bytes(8, "big"))
                    while True:
                        chunk = os.read(file_descriptor, 1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        digest.update(chunk)
                opened_after = os.fstat(file_descriptor)
                named_after = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise ActorIdentityError(
                    f"cannot hash program model file {relative}: {error}"
                ) from error
            finally:
                if held_entries is None:
                    os.close(file_descriptor)
            if (
                full_stat_identity(opened) != identity
                or full_stat_identity(opened_after) != identity
                or full_stat_identity(named_after) != identity
                or (digest is not None and total != entry_stat.st_size)
            ):
                raise ActorIdentityError(
                    f"program model file {relative} changed while it was hashed"
                )
        else:
            raise ActorIdentityError(
                f"program model tree contains a non-regular node: {relative}"
            )
    return tuple(nodes)


def _validate_held_tree_entries(
    root_descriptor: int,
    root_identity: tuple[int, int, int, int, int, int],
    held_entries: Sequence[_HeldTreeEntry],
) -> None:
    if full_stat_identity(os.fstat(root_descriptor)) != root_identity:
        raise ActorIdentityError("program model root changed while its tree was hashed")
    for entry in held_entries:
        try:
            opened = os.fstat(entry.descriptor)
            named = os.stat(
                entry.name,
                dir_fd=entry.parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise ActorIdentityError(
                f"program model tree entry {entry.relative} changed after it was scanned: "
                f"{error}"
            ) from error
        if (
            full_stat_identity(opened) != entry.identity
            or full_stat_identity(named) != entry.identity
        ):
            raise ActorIdentityError(
                f"program model tree entry {entry.relative} changed after it was scanned"
            )


def _model_tree_identity(pinned_model: PinnedPath) -> tuple[int, str]:
    root_descriptor = pinned_model.descriptor
    root_before = os.fstat(root_descriptor)
    if not stat.S_ISDIR(root_before.st_mode):
        raise ActorIdentityError(
            f"program model must be an existing directory: {pinned_model.path}"
        )
    root_identity = full_stat_identity(root_before)
    digest = hashlib.sha256()
    held_entries: list[_HeldTreeEntry] = []
    try:
        before = (("", "directory", root_identity),) + _snapshot_regular_tree(
            root_descriptor,
            Path(),
            digest=digest,
            held_entries=held_entries,
        )
        root_after_hash = os.fstat(root_descriptor)
        after = (
            ("", "directory", full_stat_identity(root_after_hash)),
        ) + _snapshot_regular_tree(
            root_descriptor,
            Path(),
            digest=None,
        )
        _validate_held_tree_entries(root_descriptor, root_identity, held_entries)
        if before != after or full_stat_identity(root_after_hash) != root_identity:
            raise ActorIdentityError("program model tree changed while it was being hashed")
        file_count = sum(node_type == "file" for _path, node_type, _identity in before)
        if not file_count:
            raise ActorIdentityError("program model directory must contain regular files")
        return file_count, digest.hexdigest()
    finally:
        for entry in reversed(held_entries):
            try:
                os.close(entry.descriptor)
            except OSError:
                pass


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


def build_actor_identity(
    config: Mapping[str, Any],
    *,
    resolved_config_snapshot: StableFileSnapshot | None = None,
) -> dict[str, Any]:
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
    try:
        with pin_absolute_path(
            model_path,
            label="program model directory",
            directory=True,
        ) as pinned_model:
            model_file_count, model_sha256 = _model_tree_identity(pinned_model)
            pinned_model.validate()
    except StablePathError as error:
        raise ActorIdentityError(str(error)) from error
    resolved_config_path = _lexical_path(
        config,
        runtime.get("verl_resolved_config_path"),
        "runtime.verl_resolved_config_path",
    )
    if resolved_config_snapshot is None:
        try:
            resolved_config_snapshot = read_stable_regular_file(
                resolved_config_path,
                label="resolved VeRL config",
            )
        except StablePathError as error:
            raise ActorIdentityError(str(error)) from error
    elif resolved_config_snapshot.path != resolved_config_path:
        raise ActorIdentityError(
            "supplied resolved VeRL config snapshot path does not match runtime config"
        )
    resolved_config_bytes = resolved_config_snapshot.raw_bytes
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
    try:
        with pin_absolute_path(
            verl_path,
            label="VeRL source directory",
            directory=True,
        ) as pinned_verl:
            pinned_verl.validate()
    except StablePathError as error:
        raise ActorIdentityError(str(error)) from error
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
        "verl_source_path": str(verl_path),
        "verl_pinned_sha": pinned_sha,
        "verl_resolved_config_path": str(resolved_config_path),
        "verl_resolved_config_sha256": resolved_config_snapshot.sha256,
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
    config_path = Path(os.path.abspath(args.config.expanduser()))
    try:
        config_snapshot = read_stable_regular_file(
            config_path,
            label="actor identity config",
        )
        config = yaml.safe_load(config_snapshot.raw_bytes.decode("utf-8"))
    except (StablePathError, UnicodeError, yaml.YAMLError) as error:
        raise ActorIdentityError(f"cannot parse actor identity config: {error}") from error
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
