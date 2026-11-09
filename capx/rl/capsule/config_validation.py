"""Torch-free validation helpers shared by local and server Capsule entrypoints."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

GRPO_BC_LOSS_MODE = "grpo_bc"
_MISSING = object()
_NO_DEFAULT = object()


def validate_bc_coef(value: Any) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("grpo_bc requires a finite positive bc_coef")


def _get(config: Any, key: str, default: Any = _NO_DEFAULT) -> Any:
    if isinstance(config, Mapping):
        value = config.get(key, _MISSING)
    else:
        value = getattr(config, key, _MISSING)
        if value is _MISSING:
            getter = getattr(config, "get", None)
            if callable(getter):
                try:
                    value = getter(key, _MISSING)
                except TypeError:
                    value = _MISSING
    if value is _MISSING and default is _NO_DEFAULT:
        raise KeyError(key)
    return default if value is _MISSING else value


def validate_capsule_training_config(config: Any) -> None:
    """Reject configs that could reinterpret or overwrite the guided-mask tensor slot."""

    algorithm = _get(config, "algorithm", _MISSING)
    if algorithm is _MISSING:
        raise ValueError("Capsule config requires an algorithm section")
    rollout_is = _get(algorithm, "rollout_is", _MISSING)
    threshold = _get(algorithm, "rollout_is_threshold", _MISSING)
    if rollout_is is not False or threshold is not None:
        raise ValueError(
            "standard rollout importance sampling must be disabled: set "
            "algorithm.rollout_is=false and algorithm.rollout_is_threshold=null"
        )
    rollout_ref = _get(config, "actor_rollout_ref", {})
    actor = _get(rollout_ref, "actor", {})
    policy_loss = _get(actor, "policy_loss", {})
    mode = _get(policy_loss, "loss_mode", None)
    bc_coef = _get(policy_loss, "bc_coef", None)
    if mode == GRPO_BC_LOSS_MODE:
        validate_bc_coef(bc_coef)
    elif bc_coef is not None:
        raise ValueError("bc_coef requires grpo_bc loss mode")


__all__ = ["validate_capsule_training_config"]
