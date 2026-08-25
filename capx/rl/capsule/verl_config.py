"""Typed VeRL configuration extensions required by Capsule policy loss."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

from verl.workers.config import PolicyLossConfig


@dataclass
class CapsulePolicyLossConfig(PolicyLossConfig):
    """Add the fixed guided-token weighting constant to VeRL's loss config."""

    capsule_gamma: float = 0.1

    def __post_init__(self) -> None:
        parent_post_init = getattr(super(), "__post_init__", None)
        if callable(parent_post_init):
            parent_post_init()
        if (
            isinstance(self.capsule_gamma, bool)
            or not isinstance(self.capsule_gamma, Real)
            or self.capsule_gamma <= 0
        ):
            raise ValueError("capsule_gamma must be positive")
