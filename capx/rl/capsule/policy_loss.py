"""Pure Torch Capsule-Critique policy loss and a lazy VeRL adapter.

`rollout_is_weights` is an existing optional tensor slot in the pinned VeRL actor API.  In the
Capsule loss mode that slot is *only* a transport for the boolean ``guided_token_mask`` artifact;
standard rollout importance sampling must be disabled by configuration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from .config_validation import validate_capsule_training_config

CAPSULE_CRITIQUE_LOSS_MODE = "capsule_critique"
GUIDED_TOKEN_MASK_FIELD = "guided_token_mask"
VERL_MASK_SLOT = "rollout_is_weights"
_MISSING = object()
_NO_DEFAULT = object()


def _require_tensor_shape_and_dtype(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    guided_token_mask: torch.Tensor,
) -> None:
    tensors = {
        "old_log_prob": old_log_prob,
        "log_prob": log_prob,
        "advantages": advantages,
        "response_mask": response_mask,
        GUIDED_TOKEN_MASK_FIELD: guided_token_mask,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    shape = log_prob.shape
    for name, tensor in tensors.items():
        if tensor.shape != shape:
            raise ValueError(f"{name} shape must match log_prob shape {tuple(shape)}")
        if tensor.device != log_prob.device:
            raise ValueError(f"{name} must be on the same device as log_prob")
    for name, tensor in (
        ("old_log_prob", old_log_prob),
        ("log_prob", log_prob),
        ("advantages", advantages),
    ):
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must have a floating dtype")
    if response_mask.dtype != torch.bool:
        raise TypeError("response_mask must have torch.bool dtype")
    if guided_token_mask.dtype != torch.bool:
        raise TypeError(f"{GUIDED_TOKEN_MASK_FIELD} must have torch.bool dtype")
    if torch.any(guided_token_mask & ~response_mask).item():
        raise ValueError(f"{GUIDED_TOKEN_MASK_FIELD} must be a subset of response_mask")


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def capsule_critique_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    guided_token_mask: torch.Tensor,
    *,
    clip_ratio: float = 0.2,
    clip_ratio_low: float | None = None,
    clip_ratio_high: float | None = None,
    clip_ratio_c: float = 3.0,
    capsule_gamma: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mix pinned VeRL dual-clipped PPO base tokens with guided probability shaping.

    Base-token math is byte-for-byte equivalent in operation order to the pinned VeRL vanilla
    objective.  Guided tokens use ``-A * sigmoid(log(p) - log(gamma))`` and never consult their
    old log-probability.  The combined loss is a mean over non-padding response tokens.
    """

    _require_tensor_shape_and_dtype(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        guided_token_mask,
    )
    numeric_parameters = {
        "clip_ratio": clip_ratio,
        "clip_ratio_c": clip_ratio_c,
        "capsule_gamma": capsule_gamma,
    }
    for name, value in numeric_parameters.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if clip_ratio < 0:
        raise ValueError("clip_ratio must be non-negative")
    if clip_ratio_c <= 1.0:
        raise ValueError("clip_ratio_c must be greater than 1.0")
    if capsule_gamma <= 0:
        raise ValueError("capsule_gamma must be positive")
    selected_low = clip_ratio if clip_ratio_low is None else clip_ratio_low
    selected_high = clip_ratio if clip_ratio_high is None else clip_ratio_high
    for name, value in (
        ("clip_ratio_low", selected_low),
        ("clip_ratio_high", selected_high),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    # Zero guided positions before every base metric/objective so changing guided old-logprobs
    # cannot change even diagnostic values.
    raw_negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.where(
        guided_token_mask,
        torch.zeros_like(raw_negative_approx_kl),
        torch.clamp(raw_negative_approx_kl, min=-20.0, max=20.0),
    )
    ratio = torch.exp(negative_approx_kl)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(
        ratio,
        1 - selected_low,
        1 + selected_high,
    )
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.minimum(pg_losses3, clip_pg_losses1)
    base_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # sigmoid is numerically stable even for very small/large current probabilities.
    guided_probability_weight = torch.sigmoid(log_prob - math.log(float(capsule_gamma)))
    guided_losses = -advantages * guided_probability_weight
    losses = torch.where(guided_token_mask, guided_losses, base_losses)
    pg_loss = _masked_mean(losses, response_mask)

    base_mask = response_mask & ~guided_token_mask
    ppo_kl = _masked_mean(-negative_approx_kl, base_mask)
    pg_clipfrac = _masked_mean(
        torch.gt(pg_losses2, pg_losses1).to(log_prob.dtype),
        base_mask,
    )
    pg_clipfrac_lower = _masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3).to(log_prob.dtype)
        * (advantages < 0).to(log_prob.dtype),
        base_mask,
    )
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def _config_get(config: Any, key: str, default: Any = _NO_DEFAULT) -> Any:
    if isinstance(config, Mapping):
        try:
            return config[key]
        except (AttributeError, KeyError):
            pass
    else:
        getter = getattr(config, "get", None)
        if callable(getter):
            try:
                value = getter(key, _MISSING)
            except TypeError:
                value = _MISSING
            if value is not _MISSING:
                return value
        if hasattr(config, key):
            return getattr(config, key)
    if default is _NO_DEFAULT:
        raise KeyError(key)
    return default


def _policy_loss_config_get(config: Any, key: str, default: Any) -> Any:
    direct = _config_get(config, key, _MISSING)
    if direct is not _MISSING:
        return direct
    policy_loss = _config_get(config, "policy_loss", None)
    if policy_loss is not None:
        nested = _config_get(policy_loss, key, _MISSING)
        if nested is not _MISSING:
            return nested
    return default


def verl_capsule_critique_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Any | None = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """VeRL policy-loss signature; reinterpret its optional IS slot as the guided mask."""

    if config is None:
        raise ValueError("Capsule policy loss requires an actor config")
    if loss_agg_mode != "token-mean":
        raise ValueError("Capsule policy loss currently requires loss_agg_mode='token-mean'")
    if rollout_is_weights is None:
        raise ValueError(
            f"Capsule policy loss requires the boolean {GUIDED_TOKEN_MASK_FIELD} in VeRL "
            f"slot {VERL_MASK_SLOT}; refusing to silently treat guided tokens as base tokens"
        )
    guided_token_mask = rollout_is_weights
    if not isinstance(guided_token_mask, torch.Tensor) or guided_token_mask.dtype != torch.bool:
        raise TypeError(
            f"VeRL {VERL_MASK_SLOT} must carry the boolean {GUIDED_TOKEN_MASK_FIELD}, "
            "not numeric rollout importance weights"
        )
    return capsule_critique_policy_loss(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        guided_token_mask,
        clip_ratio=_config_get(config, "clip_ratio", 0.2),
        clip_ratio_low=_config_get(config, "clip_ratio_low", None),
        clip_ratio_high=_config_get(config, "clip_ratio_high", None),
        clip_ratio_c=_config_get(config, "clip_ratio_c", 3.0),
        capsule_gamma=_policy_loss_config_get(config, "capsule_gamma", 0.1),
    )


def map_guided_token_mask_to_verl_slot(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Copy an artifact batch and map its explicit guided mask into VeRL's optional slot."""

    if not isinstance(batch, Mapping):
        raise TypeError("batch must be a mapping")
    if "response_mask" not in batch or GUIDED_TOKEN_MASK_FIELD not in batch:
        raise KeyError(f"batch requires response_mask and {GUIDED_TOKEN_MASK_FIELD}")
    response_mask = batch["response_mask"]
    guided_token_mask = batch[GUIDED_TOKEN_MASK_FIELD]
    if not isinstance(response_mask, torch.Tensor) or response_mask.dtype != torch.bool:
        raise TypeError("response_mask must be a boolean torch.Tensor")
    if (
        not isinstance(guided_token_mask, torch.Tensor)
        or guided_token_mask.dtype != torch.bool
    ):
        raise TypeError(f"{GUIDED_TOKEN_MASK_FIELD} must be a boolean torch.Tensor")
    if response_mask.shape != guided_token_mask.shape:
        raise ValueError(f"{GUIDED_TOKEN_MASK_FIELD} shape must match response_mask")
    if response_mask.device != guided_token_mask.device:
        raise ValueError(f"{GUIDED_TOKEN_MASK_FIELD} must be on the response_mask device")
    if torch.any(guided_token_mask & ~response_mask).item():
        raise ValueError(f"{GUIDED_TOKEN_MASK_FIELD} must be a subset of response_mask")
    existing = batch.get(VERL_MASK_SLOT)
    if existing is not None and existing is not guided_token_mask:
        raise ValueError(
            f"{VERL_MASK_SLOT} is already populated; standard rollout importance sampling "
            "must be disabled before mapping the guided mask"
        )
    mapped = dict(batch)
    mapped[VERL_MASK_SLOT] = guided_token_mask
    return mapped


def register_capsule_critique_policy_loss() -> bool:
    """Lazily register the loss with VeRL; return ``False`` when VeRL is unavailable."""

    try:
        from verl.trainer.ppo import core_algos
    except ModuleNotFoundError as error:
        if error.name == "verl" or (error.name and error.name.startswith("verl.")):
            return False
        raise
    registry = core_algos.POLICY_LOSS_REGISTRY
    existing = registry.get(CAPSULE_CRITIQUE_LOSS_MODE)
    if existing is verl_capsule_critique_policy_loss:
        return True
    if existing is not None:
        raise RuntimeError(
            f"VeRL loss mode {CAPSULE_CRITIQUE_LOSS_MODE!r} is already registered"
        )
    core_algos.register_policy_loss(CAPSULE_CRITIQUE_LOSS_MODE)(
        verl_capsule_critique_policy_loss
    )
    return True


__all__ = [
    "CAPSULE_CRITIQUE_LOSS_MODE",
    "GUIDED_TOKEN_MASK_FIELD",
    "VERL_MASK_SLOT",
    "capsule_critique_policy_loss",
    "map_guided_token_mask_to_verl_slot",
    "register_capsule_critique_policy_loss",
    "validate_capsule_training_config",
    "verl_capsule_critique_policy_loss",
]
