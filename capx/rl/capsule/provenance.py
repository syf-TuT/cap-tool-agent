"""Torch-free hashes for runtime inputs shared by smoke gates and formal training."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    """Hash one file exactly as stored, independent of its serialization format."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_root_path(config: Mapping[str, Any]) -> Path:
    """Resolve the configured project root with the repository as the stable default."""

    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("runtime must be a mapping")
    repository_root = Path(__file__).resolve().parents[3]
    configured = runtime.get("project_root")
    if isinstance(configured, str) and configured:
        path = Path(configured).expanduser()
        return path.resolve() if path.is_absolute() else (repository_root / path).resolve()
    return repository_root


def project_path(config: Mapping[str, Any], value: object, field_name: str) -> Path:
    """Resolve one required path against :func:`project_root_path`."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root_path(config) / path).resolve()


def resolved_environment_sha256(config: Mapping[str, Any]) -> str:
    """Hash the resolved task mapping together with its environment YAML bytes."""

    task = config.get("task")
    if not isinstance(task, Mapping):
        raise ValueError("task must be a mapping")
    environment_config = task.get("config_path")
    environment_config_hash: str | None = None
    if isinstance(environment_config, str) and environment_config:
        environment_config_hash = file_sha256(
            project_path(config, environment_config, "task.config_path")
        )
    serialized = json.dumps(
        {
            "task": dict(task),
            "environment_config_sha256": environment_config_hash,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def runtime_dependency_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    """Return hashes for mutable runtime dependencies that every gate must pin."""

    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("runtime must be a mapping")
    resolved_verl_path = project_path(
        config,
        runtime.get("verl_resolved_config_path"),
        "runtime.verl_resolved_config_path",
    )
    return {
        "resolved_environment_sha256": resolved_environment_sha256(config),
        "verl_resolved_config_sha256": file_sha256(resolved_verl_path),
    }


__all__ = [
    "file_sha256",
    "project_path",
    "project_root_path",
    "resolved_environment_sha256",
    "runtime_dependency_hashes",
]
