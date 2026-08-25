"""Cryptographically bind a llama.cpp runtime tree to an official release archive."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import tarfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


ATTESTATION_ARTIFACT_TYPE = "llama_cpp_b10516_runtime_attestation"
ATTESTATION_SCHEMA_VERSION = 1
_BUILD_PATTERN = re.compile(r"(?im)^\s*(?:version|build)\s*:\s*(\d+)\b")
_DYNAMIC_LOADER_ENV_NAMES = frozenset(
    {"LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT"}
)


class LlamaRuntimeAttestationError(ValueError):
    """The executable tree is not an exact safe materialization of the archive."""


def llama_build_number(version_text: str) -> int | None:
    match = _BUILD_PATTERN.search(version_text)
    return int(match.group(1)) if match is not None else None


def sanitize_dynamic_loader_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy an environment without loader injection controls."""

    source = os.environ if environment is None else environment
    return {
        name: value
        for name, value in source.items()
        if name not in _DYNAMIC_LOADER_ENV_NAMES and not name.startswith("DYLD_")
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_sha256(stream: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _real_file(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise LlamaRuntimeAttestationError(f"{label} is unavailable: {path}") from error
    if not stat.S_ISREG(mode):
        raise LlamaRuntimeAttestationError(f"{label} must be a real regular file: {path}")


def _member_name(raw_name: str) -> str:
    if not raw_name or "\0" in raw_name or "\\" in raw_name:
        raise LlamaRuntimeAttestationError(
            f"unsafe archive member path: {raw_name!r}"
        )
    path = PurePosixPath(raw_name)
    if path.is_absolute():
        raise LlamaRuntimeAttestationError(
            f"unsafe archive member path: {raw_name!r}"
        )
    parts = [part for part in path.parts if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise LlamaRuntimeAttestationError(
            f"unsafe archive member path: {raw_name!r}"
        )
    return "." if not parts else "/".join(parts)


def _symlink_target(member_name: str, raw_target: str) -> str:
    if not raw_target or "\0" in raw_target or "\\" in raw_target:
        raise LlamaRuntimeAttestationError(
            f"unsafe symlink target for {member_name!r}: {raw_target!r}"
        )
    target = PurePosixPath(raw_target)
    if target.is_absolute():
        raise LlamaRuntimeAttestationError(
            f"unsafe symlink target for {member_name!r}: {raw_target!r}"
        )
    stack = list(PurePosixPath(member_name).parent.parts)
    for part in target.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not stack:
                raise LlamaRuntimeAttestationError(
                    f"unsafe symlink target for {member_name!r}: {raw_target!r}"
                )
            stack.pop()
        else:
            stack.append(part)
    if not stack:
        return "."
    return "/".join(stack)


def _runtime_path(root: Path, member_name: str) -> Path:
    return root if member_name == "." else root.joinpath(*member_name.split("/"))


def _require_real_ancestors(root: Path, path: Path) -> None:
    relative = path.relative_to(root)
    current = root
    for part in relative.parts[:-1]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except OSError as error:
            raise LlamaRuntimeAttestationError(
                f"runtime member parent is unavailable: {current}"
            ) from error
        if not stat.S_ISDIR(mode):
            raise LlamaRuntimeAttestationError(
                f"runtime member parent must be a real directory: {current}"
            )


def _resolve_archive_link(
    start: str, members: Mapping[str, tarfile.TarInfo]
) -> str:
    current = start
    seen: set[str] = set()
    while True:
        if current in seen:
            raise LlamaRuntimeAttestationError(
                f"archive symlink cycle includes {start!r}"
            )
        seen.add(current)
        target = members.get(current)
        if target is None:
            raise LlamaRuntimeAttestationError(
                f"archive symlink {start!r} targets missing member {current!r}"
            )
        if target.issym():
            current = _symlink_target(current, target.linkname)
            continue
        if not target.isfile():
            raise LlamaRuntimeAttestationError(
                f"archive symlink {start!r} does not resolve to a regular file"
            )
        return current


def _canonical_tree_sha256(records: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        sorted(records, key=lambda record: str(record["archive_member"])),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def attest_llama_cpp_runtime(
    *,
    archive_path: str | Path,
    expected_archive_sha256: str,
    binary_path: str | Path,
    gguf_path: str | Path,
    expected_gguf_sha256: str,
    expected_build_number: int,
    version_tag: str,
) -> dict[str, Any]:
    """Return a strict, reproducible proof for the executable release tree."""

    archive = Path(archive_path).expanduser().absolute()
    binary = Path(binary_path).expanduser().absolute()
    gguf = Path(gguf_path).expanduser().absolute()
    _real_file(archive, "llama.cpp archive")
    _real_file(binary, "llama-server")
    _real_file(gguf, "controller GGUF")
    archive_sha256 = _file_sha256(archive)
    if archive_sha256 != expected_archive_sha256:
        raise LlamaRuntimeAttestationError(
            "llama.cpp release archive SHA-256 mismatch"
        )
    gguf_sha256 = _file_sha256(gguf)
    if gguf_sha256 != expected_gguf_sha256:
        raise LlamaRuntimeAttestationError("controller GGUF SHA-256 mismatch")

    members: dict[str, tarfile.TarInfo] = {}
    try:
        with tarfile.open(archive, mode="r:gz") as release:
            for member in release.getmembers():
                name = _member_name(member.name)
                if name in members:
                    raise LlamaRuntimeAttestationError(
                        f"duplicate archive member path: {name!r}"
                    )
                if not (member.isdir() or member.isfile() or member.issym()):
                    raise LlamaRuntimeAttestationError(
                        f"unsupported archive member type: {name!r}"
                    )
                members[name] = member

            for name, member in members.items():
                if member.issym():
                    target_member = _symlink_target(name, member.linkname)
                    _resolve_archive_link(target_member, members)

            server_members = [
                name for name in members if PurePosixPath(name).name == "llama-server"
            ]
            if len(server_members) != 1:
                raise LlamaRuntimeAttestationError(
                    "release archive must contain exactly one llama-server"
                )
            binary_member_name = server_members[0]
            binary_member = members[binary_member_name]
            if not binary_member.isfile():
                raise LlamaRuntimeAttestationError(
                    "archive llama-server member must be a regular file"
                )

            root = archive.parent
            expected_binary = _runtime_path(root, binary_member_name).absolute()
            if binary != expected_binary:
                raise LlamaRuntimeAttestationError(
                    "configured binary_path does not correspond to archive llama-server member"
                )
            records: list[dict[str, Any]] = []
            regular_file_count = 0
            symlink_count = 0
            for name, member in sorted(members.items()):
                runtime_path = _runtime_path(root, name)
                if name != ".":
                    _require_real_ancestors(root, runtime_path)
                try:
                    mode = runtime_path.lstat().st_mode
                except OSError as error:
                    raise LlamaRuntimeAttestationError(
                        f"archive runtime member is missing: {runtime_path}"
                    ) from error
                if member.isdir():
                    if not stat.S_ISDIR(mode):
                        raise LlamaRuntimeAttestationError(
                            f"archive directory is not a real runtime directory: {runtime_path}"
                        )
                    continue
                if member.isfile():
                    if not stat.S_ISREG(mode):
                        raise LlamaRuntimeAttestationError(
                            f"archive regular member is not a real file: {runtime_path}"
                        )
                    if runtime_path.stat().st_size != member.size:
                        raise LlamaRuntimeAttestationError(
                            f"runtime member size mismatch: {name!r}"
                        )
                    source = release.extractfile(member)
                    if source is None:
                        raise LlamaRuntimeAttestationError(
                            f"cannot read archive regular member: {name!r}"
                        )
                    with source:
                        archive_member_sha256 = _stream_sha256(source)
                    runtime_sha256 = _file_sha256(runtime_path)
                    if runtime_sha256 != archive_member_sha256:
                        raise LlamaRuntimeAttestationError(
                            f"runtime member SHA-256 mismatch: {name!r}"
                        )
                    records.append(
                        {
                            "archive_member": name,
                            "sha256": runtime_sha256,
                            "size": member.size,
                            "type": "regular",
                        }
                    )
                    regular_file_count += 1
                    continue

                if not stat.S_ISLNK(mode):
                    raise LlamaRuntimeAttestationError(
                        f"archive symlink is not a runtime symlink: {runtime_path}"
                    )
                actual_target = os.readlink(runtime_path)
                if actual_target != member.linkname:
                    raise LlamaRuntimeAttestationError(
                        f"runtime symlink target mismatch: {name!r}"
                    )
                target_member = _symlink_target(name, member.linkname)
                resolved_member = _resolve_archive_link(target_member, members)
                expected_target = _runtime_path(root, resolved_member).resolve(strict=True)
                if runtime_path.resolve(strict=True) != expected_target:
                    raise LlamaRuntimeAttestationError(
                        f"runtime symlink resolves to wrong member: {name!r}"
                    )
                records.append(
                    {
                        "archive_member": name,
                        "link_target": member.linkname,
                        "resolved_archive_member": resolved_member,
                        "type": "symlink",
                    }
                )
                symlink_count += 1
    except (tarfile.TarError, OSError) as error:
        if isinstance(error, LlamaRuntimeAttestationError):
            raise
        raise LlamaRuntimeAttestationError(
            f"cannot inspect llama.cpp release archive: {error}"
        ) from error

    if not os.access(binary, os.X_OK):
        raise LlamaRuntimeAttestationError("llama-server is not executable")
    binary_sha256 = _file_sha256(binary)
    try:
        version_result = subprocess.run(
            [str(binary), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
            env=sanitize_dynamic_loader_environment(),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise LlamaRuntimeAttestationError(
            f"cannot execute llama-server version attestation: {error}"
        ) from error
    version_text = version_result.stdout + "\n" + version_result.stderr
    build_number = llama_build_number(version_text)
    if build_number != expected_build_number:
        raise LlamaRuntimeAttestationError(
            f"llama-server build must be {expected_build_number}, got {build_number!r}"
        )

    return {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "artifact_type": ATTESTATION_ARTIFACT_TYPE,
        "version_tag": version_tag,
        "archive_path": str(archive),
        "archive_sha256": archive_sha256,
        "binary_path": str(binary),
        "binary_archive_member": binary_member_name,
        "binary_sha256": binary_sha256,
        "gguf_path": str(gguf),
        "gguf_sha256": gguf_sha256,
        "build_number": build_number,
        "runtime_tree_sha256": _canonical_tree_sha256(records),
        "regular_file_count": regular_file_count,
        "symlink_count": symlink_count,
    }
