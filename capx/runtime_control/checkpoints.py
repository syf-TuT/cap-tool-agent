from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NamespaceCheckpoint:
    checkpoint_id: str
    label: str
    values: dict[str, Any]
    skipped: list[str] = field(default_factory=list)


class NamespaceCheckpointStore:
    def __init__(self) -> None:
        self._checkpoints: dict[str, NamespaceCheckpoint] = {}
        self._count = 0

    def save(self, label: str, namespace: dict[str, Any]) -> str:
        self._count += 1
        checkpoint_id = f"checkpoint_{self._count}"
        values: dict[str, Any] = {}
        skipped: list[str] = []
        for key, value in namespace.items():
            try:
                values[key] = copy.deepcopy(value)
            except Exception:
                skipped.append(key)
        self._checkpoints[checkpoint_id] = NamespaceCheckpoint(
            checkpoint_id=checkpoint_id,
            label=label,
            values=values,
            skipped=skipped,
        )
        return checkpoint_id

    def restore(self, checkpoint_id: str) -> dict[str, Any]:
        if checkpoint_id not in self._checkpoints:
            raise KeyError(f"Unknown checkpoint: {checkpoint_id}")
        return copy.deepcopy(self._checkpoints[checkpoint_id].values)

    def skipped(self, checkpoint_id: str) -> list[str]:
        if checkpoint_id not in self._checkpoints:
            raise KeyError(f"Unknown checkpoint: {checkpoint_id}")
        return list(self._checkpoints[checkpoint_id].skipped)
