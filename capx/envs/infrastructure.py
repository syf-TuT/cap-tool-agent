"""Typed failures shared by environment launch and trial execution."""

from __future__ import annotations

from typing import Any


class InfrastructureFailure(RuntimeError):
    """Signals a non-program failure in a required runtime service."""

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        self.kind = kind
        self.message = message
        self.evidence = evidence or {}
        super().__init__(message)


class ServiceReadinessError(InfrastructureFailure):
    """Signals that required services did not become ready before launch."""
