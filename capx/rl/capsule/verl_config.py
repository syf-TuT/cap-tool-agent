"""Typed VeRL configuration extensions required by Capsule policy loss."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

from verl.workers.config import PolicyLossConfig

from .config_validation import GRPO_BC_LOSS_MODE, validate_bc_coef


@dataclass
class CapsulePolicyLossConfig(PolicyLossConfig):
    """Configure the guided objective used by the Capsule policy loss."""

    capsule_gamma: float = 0.1
    guided_objective: str = "probability_shaping"
    bc_coef: float | None = None

    def __post_init__(self) -> None:
        parent_post_init = getattr(super(), "__post_init__", None)
        if callable(parent_post_init):
            parent_post_init()
        if self.loss_mode == GRPO_BC_LOSS_MODE:
            validate_bc_coef(self.bc_coef)
        if self.guided_objective not in ("probability_shaping", "log_probability"):
            raise ValueError(f"unsupported guided_objective: {self.guided_objective!r}")
        if (
            isinstance(self.capsule_gamma, bool)
            or not isinstance(self.capsule_gamma, Real)
            or self.capsule_gamma <= 0
        ):
            raise ValueError("capsule_gamma must be positive")
