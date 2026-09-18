"""Descriptor-pinned, no-follow reads for security-sensitive provenance inputs."""

from __future__ import annotations

import ctypes
import hashlib
import os
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


class StablePathError(RuntimeError):
    """A path could not be pinned or changed while its bytes were consumed."""


@dataclass(frozen=True)
class MutationWatch:
    """One path that must remain mutation-free while a runtime consumes it."""

    path: Path
    label: str
    recursive: bool = False


_BasicIdentity = tuple[int, int, int]
_FullIdentity = tuple[int, int, int, int, int, int]


def _basic_identity(value: os.stat_result) -> _BasicIdentity:
    return (value.st_dev, value.st_ino, value.st_mode)


def full_stat_identity(value: os.stat_result) -> _FullIdentity:
    """Return metadata that changes for replacement or in-place content mutation."""

    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_openat_support() -> None:
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
    missing_flags = [name for name in required_flags if not hasattr(os, name)]
    supported = (
        os.name == "posix"
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.stat in os.supports_follow_symlinks
        and os.listdir in os.supports_fd
        and not missing_flags
    )
    if not supported:
        detail = ", ".join(missing_flags) if missing_flags else "openat/fstatat/listdir(fd)"
        raise StablePathError(
            "descriptor-relative no-symlink path traversal is unavailable; "
            f"refusing provenance read ({detail})"
        )


@dataclass
class PinnedPath:
    """An absolute path whose complete component chain is held by descriptors."""

    path: Path
    label: str
    descriptors: tuple[int, ...]
    component_names: tuple[str, ...]
    component_identities: tuple[_BasicIdentity, ...]
    _closed: bool = False

    @property
    def descriptor(self) -> int:
        if self._closed:
            raise StablePathError(f"{self.label} descriptor chain is already closed")
        return self.descriptors[-1]

    def validate(self) -> None:
        """Verify every held parent still names the exact child descriptor opened earlier."""

        if self._closed:
            raise StablePathError(f"{self.label} descriptor chain is already closed")
        for index, name in enumerate(self.component_names):
            parent_descriptor = self.descriptors[index]
            child_descriptor = self.descriptors[index + 1]
            expected = self.component_identities[index + 1]
            try:
                named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
                opened = os.fstat(child_descriptor)
            except OSError as error:
                raise StablePathError(
                    f"{self.label} path changed or was replaced during access: {self.path}: "
                    f"{error}"
                ) from error
            if _basic_identity(named) != expected or _basic_identity(opened) != expected:
                raise StablePathError(
                    f"{self.label} path changed or was replaced during access: {self.path}"
                )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for descriptor in reversed(self.descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass

    def __enter__(self) -> PinnedPath:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def pin_absolute_path(
    path: str | Path,
    *,
    label: str,
    directory: bool,
) -> PinnedPath:
    """Open every component with ``openat`` and ``O_NOFOLLOW``, retaining the chain."""

    _require_openat_support()
    lexical = Path(os.path.abspath(Path(path).expanduser()))
    if not lexical.is_absolute():  # pragma: no cover - abspath guarantees this
        raise StablePathError(f"{label} must be an absolute path: {lexical}")
    names = tuple(lexical.parts[1:])
    if not names:
        raise StablePathError(f"{label} must not be the filesystem root")
    if any(name in {"", ".", ".."} for name in names):
        raise StablePathError(f"{label} contains an unsafe path component: {lexical}")

    directory_flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    regular_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptors: list[int] = []
    identities: list[_BasicIdentity] = []
    try:
        root_descriptor = os.open(os.sep, directory_flags)
        descriptors.append(root_descriptor)
        identities.append(_basic_identity(os.fstat(root_descriptor)))
        for index, name in enumerate(names):
            is_final = index == len(names) - 1
            flags = directory_flags if not is_final or directory else regular_flags
            parent_descriptor = descriptors[-1]
            child_descriptor: int | None = None
            try:
                child_descriptor = os.open(name, flags, dir_fd=parent_descriptor)
                named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
                opened = os.fstat(child_descriptor)
            except OSError as error:
                if child_descriptor is not None:
                    os.close(child_descriptor)
                raise StablePathError(
                    f"cannot open {label} without following symlink components: "
                    f"{lexical}: {error}"
                ) from error
            expected_kind = (
                stat.S_ISDIR(opened.st_mode)
                if not is_final or directory
                else stat.S_ISREG(opened.st_mode)
            )
            if not expected_kind or _basic_identity(named) != _basic_identity(opened):
                os.close(child_descriptor)
                raise StablePathError(
                    f"{label} changed or was replaced while its path was opened: {lexical}"
                )
            descriptors.append(child_descriptor)
            identities.append(_basic_identity(opened))
        pinned = PinnedPath(
            path=lexical,
            label=label,
            descriptors=tuple(descriptors),
            component_names=names,
            component_identities=tuple(identities),
        )
        pinned.validate()
        return pinned
    except BaseException:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


@dataclass(frozen=True)
class StableFileSnapshot:
    """Bytes and SHA-256 produced from one unchanged, descriptor-pinned file read."""

    path: Path
    raw_bytes: bytes
    sha256: str
    identity: _FullIdentity


def read_stable_regular_file(path: str | Path, *, label: str) -> StableFileSnapshot:
    """Read, hash, and validate one regular file without reopening its pathname."""

    with pin_absolute_path(path, label=label, directory=False) as pinned:
        descriptor = pinned.descriptor
        try:
            before = os.fstat(descriptor)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        except OSError as error:
            raise StablePathError(f"cannot read {label}: {pinned.path}: {error}") from error
        body = b"".join(chunks)
        identity = full_stat_identity(before)
        if (
            not stat.S_ISREG(before.st_mode)
            or full_stat_identity(after) != identity
            or len(body) != after.st_size
        ):
            raise StablePathError(
                f"{label} changed or was replaced while its bytes were read: {pinned.path}"
            )
        pinned.validate()
        parent_descriptor = pinned.descriptors[-2]
        final_name = pinned.component_names[-1]
        try:
            named_after = os.stat(
                final_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise StablePathError(
                f"{label} changed or was replaced after its bytes were read: "
                f"{pinned.path}: {error}"
            ) from error
        if full_stat_identity(named_after) != identity:
            raise StablePathError(
                f"{label} changed or was replaced after its bytes were read: {pinned.path}"
            )
        return StableFileSnapshot(
            path=pinned.path,
            raw_bytes=body,
            sha256=hashlib.sha256(body).hexdigest(),
            identity=identity,
        )


_IN_MODIFY = 0x00000002
_IN_ATTRIB = 0x00000004
_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_FROM = 0x00000040
_IN_MOVED_TO = 0x00000080
_IN_CREATE = 0x00000100
_IN_DELETE = 0x00000200
_IN_DELETE_SELF = 0x00000400
_IN_MOVE_SELF = 0x00000800
_IN_DONT_FOLLOW = 0x02000000
_MUTATION_MASK = (
    _IN_MODIFY
    | _IN_ATTRIB
    | _IN_CLOSE_WRITE
    | _IN_MOVED_FROM
    | _IN_MOVED_TO
    | _IN_CREATE
    | _IN_DELETE
    | _IN_DELETE_SELF
    | _IN_MOVE_SELF
    | _IN_DONT_FOLLOW
)


def _recursive_directory_paths(root: Path, *, label: str) -> tuple[Path, ...]:
    directories: list[Path] = [root]
    index = 0
    while index < len(directories):
        directory = directories[index]
        index += 1
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise StablePathError(f"cannot scan {label} for mutation watches: {error}") from error
        for entry in entries:
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise StablePathError(
                    f"cannot stat {label} entry while installing mutation watches: "
                    f"{entry.path}: {error}"
                ) from error
            if stat.S_ISLNK(entry_stat.st_mode):
                raise StablePathError(
                    f"{label} mutation guard refuses symlink tree entry: {entry.path}"
                )
            if stat.S_ISDIR(entry_stat.st_mode):
                directories.append(Path(entry.path))
            elif not stat.S_ISREG(entry_stat.st_mode):
                raise StablePathError(
                    f"{label} mutation guard refuses non-regular tree entry: {entry.path}"
                )
    return tuple(directories)


class PathMutationGuard:
    """Linux inotify fail-fast guard for A-to-B-to-A runtime input mutation.

    Hashing before and after a model load cannot detect bytes that are swapped in and then
    restored.  The guard records every write, metadata change, create/delete, or rename event
    queued by the kernel while the consumer is active.  It intentionally fails closed when
    inotify or the required recursive watches are unavailable.
    """

    def __init__(self, descriptor: int, watches: tuple[MutationWatch, ...]) -> None:
        self._descriptor = descriptor
        self.watches = watches
        self._closed = False

    @classmethod
    def open(cls, watches: Iterable[MutationWatch]) -> PathMutationGuard:
        requested = tuple(watches)
        if not requested:
            raise StablePathError("at least one runtime mutation watch is required")
        if os.name != "posix" or not sys_platform_linux():
            raise StablePathError(
                "Linux inotify is required to bind mutable model/VeRL paths during runtime"
            )
        libc = ctypes.CDLL(None, use_errno=True)
        init = libc.inotify_init1
        init.argtypes = [ctypes.c_int]
        init.restype = ctypes.c_int
        add_watch = libc.inotify_add_watch
        add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        add_watch.restype = ctypes.c_int
        descriptor = init(os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
        if descriptor < 0:
            error_number = ctypes.get_errno()
            raise StablePathError(
                f"cannot initialize Linux runtime mutation guard: {os.strerror(error_number)}"
            )
        normalized: list[MutationWatch] = []
        watched_paths: set[Path] = set()
        try:
            for watch in requested:
                path = Path(os.path.abspath(watch.path.expanduser()))
                with pin_absolute_path(
                    path,
                    label=watch.label,
                    directory=watch.recursive,
                ) as pinned:
                    pinned.validate()
                candidates = (
                    _recursive_directory_paths(path, label=watch.label)
                    if watch.recursive
                    else (path,)
                )
                for candidate in candidates:
                    if candidate in watched_paths:
                        continue
                    watch_descriptor = add_watch(
                        descriptor,
                        os.fsencode(candidate),
                        _MUTATION_MASK,
                    )
                    if watch_descriptor < 0:
                        error_number = ctypes.get_errno()
                        raise StablePathError(
                            f"cannot guard {watch.label} against runtime mutation: "
                            f"{candidate}: {os.strerror(error_number)}"
                        )
                    watched_paths.add(candidate)
                normalized.append(
                    MutationWatch(path=path, label=watch.label, recursive=watch.recursive)
                )
            guard = cls(descriptor, tuple(normalized))
            guard.assert_unchanged(context="while mutation watches were installed")
            return guard
        except BaseException:
            os.close(descriptor)
            raise

    def assert_unchanged(self, *, context: str) -> None:
        if self._closed:
            raise StablePathError("runtime mutation guard is already closed")
        changed = False
        try:
            while True:
                try:
                    payload = os.read(self._descriptor, 64 * 1024)
                except BlockingIOError:
                    break
                if not payload:
                    break
                changed = True
        except OSError as error:
            raise StablePathError(f"cannot read runtime mutation guard: {error}") from error
        if changed:
            labels = ", ".join(watch.label for watch in self.watches)
            raise StablePathError(f"guarded runtime input changed {context}: {labels}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self._descriptor)

    def __enter__(self) -> PathMutationGuard:
        return self

    def __exit__(self, error_type: object, *_args: object) -> None:
        try:
            if error_type is None:
                self.assert_unchanged(context="during training")
        finally:
            self.close()


def sys_platform_linux() -> bool:
    """Avoid importing platform in the provenance-only hot path."""

    return os.uname().sysname == "Linux" if hasattr(os, "uname") else False


__all__ = [
    "MutationWatch",
    "PathMutationGuard",
    "PinnedPath",
    "StableFileSnapshot",
    "StablePathError",
    "full_stat_identity",
    "pin_absolute_path",
    "read_stable_regular_file",
]
