"""Exclusive, recoverable checkpoint publication for Capsule server runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def checkpoint_tree_files(path: str | Path) -> tuple[Path, tuple[Path, ...]]:
    lexical_root = Path(path).expanduser()
    if lexical_root.is_symlink():
        raise ValueError(f"checkpoint path must not be a symlink: {lexical_root}")
    root = lexical_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"checkpoint path does not exist: {root}")
    if root.is_file():
        return root.parent, (root,)
    files: list[Path] = []
    for entry in root.rglob("*"):
        if entry.is_symlink():
            raise ValueError(f"checkpoint tree must not contain symlinks: {entry}")
        if entry.is_file():
            files.append(entry)
    files.sort(key=lambda item: item.relative_to(root).as_posix())
    return root, tuple(files)


def checkpoint_tree_sha256(path: str | Path) -> str:
    root, files = checkpoint_tree_files(path)
    digest = hashlib.sha256()
    for file_path in files:
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(file_path.stat().st_size.to_bytes(8, "big"))
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointEvidence:
    path: Path
    file_count: int
    sha256: str
    manifest_path: Path


class AtomicCheckpointClaim:
    """Own one run directory, stage a checkpoint, and publish it without replacement."""

    def __init__(self, final_path: str | Path, *, claim_root: str | Path) -> None:
        self.final_path = Path(final_path).expanduser().resolve()
        self.claim_root = Path(claim_root).expanduser().resolve()
        if self.final_path == self.claim_root or not self.final_path.is_relative_to(
            self.claim_root
        ):
            raise ValueError("final checkpoint path must be a child of claim_root")
        self._token = uuid.uuid4().hex
        self._marker = self.claim_root / ".capsule_checkpoint_claim"
        self._staging_root = self.claim_root / f".staging-{self._token}"
        self._staging_path = self._staging_root / self.final_path.relative_to(
            self.claim_root
        )
        self._claimed = False
        self._published = False

    def __enter__(self) -> AtomicCheckpointClaim:
        self.claim_root.parent.mkdir(parents=True, exist_ok=True)
        self.claim_root.mkdir(exist_ok=False)
        try:
            self._marker.write_text(self._token + "\n", encoding="ascii")
        except BaseException:
            self.claim_root.rmdir()
            raise
        self._claimed = True
        return self

    @staticmethod
    def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, allow_nan=False, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def publish(
        self,
        save_callback: Callable[[Path], None],
        *,
        optimizer_step_before: int,
        optimizer_step_after: int,
    ) -> CheckpointEvidence:
        if not self._claimed or self._published:
            raise RuntimeError("checkpoint claim is not active")
        self._staging_path.parent.mkdir(parents=True, exist_ok=False)
        save_callback(self._staging_path)
        _root, files = checkpoint_tree_files(self._staging_path)
        if not files:
            raise RuntimeError("checkpoint writer created no regular files")
        sha256 = checkpoint_tree_sha256(self._staging_path)
        self.final_path.parent.mkdir(parents=True, exist_ok=False)
        self._staging_path.rename(self.final_path)
        shutil.rmtree(self._staging_root)
        manifest_path = self.claim_root / "checkpoint_manifest.json"
        self._write_manifest(
            manifest_path,
            {
                "schema_version": 1,
                "checkpoint": str(self.final_path),
                "checkpoint_file_count": len(files),
                "checkpoint_sha256": sha256,
                "optimizer_step_before": optimizer_step_before,
                "optimizer_step_after": optimizer_step_after,
                "optimizer_step_delta": optimizer_step_after - optimizer_step_before,
            },
        )
        self._published = True
        return CheckpointEvidence(
            path=self.final_path,
            file_count=len(files),
            sha256=sha256,
            manifest_path=manifest_path,
        )

    def abort(self) -> None:
        if not self._claimed or not self.claim_root.exists():
            return
        try:
            owner = self._marker.read_text(encoding="ascii").strip()
        except OSError:
            owner = ""
        if owner != self._token:
            raise RuntimeError(
                "refusing to clean checkpoint claim without matching owner marker: "
                f"{self.claim_root}"
            )
        shutil.rmtree(self.claim_root)
        self._claimed = False

    def commit(self) -> None:
        """Release a successfully published claim without altering checkpoint bytes."""

        if not self._claimed or not self._published:
            raise RuntimeError("checkpoint claim has no published checkpoint to commit")
        self._marker.unlink()
        self._claimed = False

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        del exc_type, traceback
        if exc is not None or not self._published:
            self.abort()
        else:
            self.commit()
        return False


__all__ = [
    "AtomicCheckpointClaim",
    "CheckpointEvidence",
    "checkpoint_tree_files",
    "checkpoint_tree_sha256",
]
