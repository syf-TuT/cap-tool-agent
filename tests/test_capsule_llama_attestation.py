from __future__ import annotations

import hashlib
import io
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from scripts.capsule_rl.llama_attestation import (
    LlamaRuntimeAttestationError,
    attest_llama_cpp_runtime,
)


def _add_regular(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.mode = 0o755 if name.endswith("llama-server") else 0o644
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


def _write_archive(
    path: Path,
    *,
    prefix: str = "",
    extra_members: list[tuple[tarfile.TarInfo, bytes | None]] | None = None,
) -> None:
    member_prefix = f"{prefix}/" if prefix else ""
    with tarfile.open(path, "w:gz") as archive:
        _add_regular(archive, f"{member_prefix}llama-server", b"official-server-bytes")
        _add_regular(
            archive,
            f"{member_prefix}libllama.so.1",
            b"official-library-bytes",
        )
        symlink = tarfile.TarInfo(f"{member_prefix}libllama.so")
        symlink.type = tarfile.SYMTYPE
        symlink.linkname = "libllama.so.1"
        archive.addfile(symlink)
        for member, data in extra_members or []:
            archive.addfile(member, io.BytesIO(data) if data is not None else None)


def _materialize_runtime(root: Path) -> tuple[Path, Path, Path]:
    archive_path = root / "llama-b10516.tar.gz"
    _write_archive(archive_path)
    binary_path = root / "llama-server"
    binary_path.write_bytes(b"official-server-bytes")
    binary_path.chmod(0o755)
    (root / "libllama.so.1").write_bytes(b"official-library-bytes")
    os.symlink("libllama.so.1", root / "libllama.so")
    gguf_path = root / "controller.gguf"
    gguf_path.write_bytes(b"official-gguf")
    return archive_path, binary_path, gguf_path


def _materialize_official_layout(root: Path) -> tuple[Path, Path, Path]:
    archive_path = root / "llama-b10516-bin-ubuntu-x64.tar.gz"
    _write_archive(archive_path, prefix="llama-b10516")
    runtime_root = root / "llama-b10516"
    runtime_root.mkdir()
    binary_path = runtime_root / "llama-server"
    binary_path.write_bytes(b"official-server-bytes")
    binary_path.chmod(0o755)
    (runtime_root / "libllama.so.1").write_bytes(b"official-library-bytes")
    os.symlink("libllama.so.1", runtime_root / "libllama.so")
    gguf_path = root / "controller.gguf"
    gguf_path.write_bytes(b"official-gguf")
    return archive_path, binary_path, gguf_path


def _attest(
    monkeypatch: pytest.MonkeyPatch,
    archive_path: Path,
    binary_path: Path,
    gguf_path: Path,
    captured_run: dict[str, object] | None = None,
) -> dict[str, object]:
    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert argv == [str(binary_path), "--version"]
        if captured_run is not None:
            captured_run.update(_kwargs)
        return subprocess.CompletedProcess(argv, 0, "version: 10516 (b95502ba)\n", "")

    monkeypatch.setattr(
        "scripts.capsule_rl.llama_attestation.subprocess.run", fake_run
    )
    return attest_llama_cpp_runtime(
        archive_path=archive_path,
        expected_archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        binary_path=binary_path,
        gguf_path=gguf_path,
        expected_gguf_sha256=hashlib.sha256(gguf_path.read_bytes()).hexdigest(),
        expected_build_number=10516,
        version_tag="b10516",
    )


def test_attestation_version_probe_removes_dynamic_loader_injection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive_path, binary_path, gguf_path = _materialize_runtime(tmp_path)
    for name in ("LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "DYLD_INSERT_LIBRARIES"):
        monkeypatch.setenv(name, f"hostile-{name}")
    monkeypatch.setenv("UNRELATED_RUNTIME_VALUE", "retained")
    captured: dict[str, object] = {}

    _attest(
        monkeypatch,
        archive_path,
        binary_path,
        gguf_path,
        captured_run=captured,
    )

    environment = captured["env"]
    assert isinstance(environment, dict)
    for name in ("LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "DYLD_INSERT_LIBRARIES"):
        assert name not in environment
    assert environment["UNRELATED_RUNTIME_VALUE"] == "retained"


def test_attestation_binds_every_regular_file_and_safe_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive_path, binary_path, gguf_path = _materialize_runtime(tmp_path)

    attestation = _attest(monkeypatch, archive_path, binary_path, gguf_path)

    assert attestation == {
        "schema_version": 1,
        "artifact_type": "llama_cpp_b10516_runtime_attestation",
        "version_tag": "b10516",
        "archive_path": str(archive_path),
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "binary_path": str(binary_path),
        "binary_archive_member": "llama-server",
        "binary_sha256": hashlib.sha256(binary_path.read_bytes()).hexdigest(),
        "gguf_path": str(gguf_path),
        "gguf_sha256": hashlib.sha256(gguf_path.read_bytes()).hexdigest(),
        "build_number": 10516,
        "runtime_tree_sha256": attestation["runtime_tree_sha256"],
        "regular_file_count": 2,
        "symlink_count": 1,
    }
    assert len(str(attestation["runtime_tree_sha256"])) == 64


def test_attestation_accepts_official_b10516_top_level_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive_path, binary_path, gguf_path = _materialize_official_layout(tmp_path)

    attestation = _attest(monkeypatch, archive_path, binary_path, gguf_path)

    assert attestation["binary_archive_member"] == "llama-b10516/llama-server"
    assert attestation["binary_path"] == str(binary_path)


def test_attestation_rejects_any_runtime_member_byte_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive_path, binary_path, gguf_path = _materialize_runtime(tmp_path)
    (tmp_path / "libllama.so.1").write_bytes(
        b"X" * len(b"official-library-bytes")
    )

    with pytest.raises(LlamaRuntimeAttestationError, match="runtime member SHA-256"):
        _attest(monkeypatch, archive_path, binary_path, gguf_path)


def test_attestation_rejects_binary_path_not_matching_archive_member(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive_path, binary_path, gguf_path = _materialize_runtime(tmp_path)
    moved = tmp_path / "bin" / "llama-server"
    moved.parent.mkdir()
    moved.write_bytes(binary_path.read_bytes())
    moved.chmod(0o755)

    with pytest.raises(LlamaRuntimeAttestationError, match="binary_path does not correspond"):
        _attest(monkeypatch, archive_path, moved, gguf_path)


@pytest.mark.parametrize(
    ("member", "message"),
    [
        (tarfile.TarInfo("../escape"), "unsafe archive member path"),
        (tarfile.TarInfo("copy/llama-server"), "exactly one llama-server"),
    ],
)
def test_attestation_rejects_unsafe_or_multiple_server_members(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    member: tarfile.TarInfo,
    message: str,
) -> None:
    member.size = 5
    member.mode = 0o755
    archive_path, binary_path, gguf_path = _materialize_runtime(tmp_path)
    _write_archive(archive_path, extra_members=[(member, b"extra")])

    with pytest.raises(LlamaRuntimeAttestationError, match=message):
        _attest(monkeypatch, archive_path, binary_path, gguf_path)


def test_attestation_rejects_special_nodes_and_escaping_symlinks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive_path, binary_path, gguf_path = _materialize_runtime(tmp_path)
    fifo = tarfile.TarInfo("runtime.pipe")
    fifo.type = tarfile.FIFOTYPE
    _write_archive(archive_path, extra_members=[(fifo, None)])
    with pytest.raises(LlamaRuntimeAttestationError, match="unsupported archive member type"):
        _attest(monkeypatch, archive_path, binary_path, gguf_path)

    escape = tarfile.TarInfo("escape-link")
    escape.type = tarfile.SYMTYPE
    escape.linkname = "../outside"
    _write_archive(archive_path, extra_members=[(escape, None)])
    with pytest.raises(LlamaRuntimeAttestationError, match="unsafe symlink target"):
        _attest(monkeypatch, archive_path, binary_path, gguf_path)
