"""Dependency-light helpers for synchronizing Robosuite episode RNG state."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def _set_sampler_rng(
    sampler: Any,
    rng: np.random.Generator,
    *,
    seen: set[int] | None = None,
) -> None:
    if sampler is None:
        return
    if seen is None:
        seen = set()
    sampler_id = id(sampler)
    if sampler_id in seen:
        return
    seen.add(sampler_id)

    if hasattr(sampler, "rng"):
        sampler.rng = rng
    children = getattr(sampler, "samplers", None)
    if isinstance(children, Mapping):
        child_samplers = children.values()
    elif isinstance(children, Iterable) and not isinstance(children, (str, bytes)):
        child_samplers = children
    else:
        return
    for child in child_samplers:
        _set_sampler_rng(child, rng, seen=seen)


def reseed_robosuite_owner(owner: Any, seed: int) -> None:
    """Point the wrapper, Robosuite env, and every placement sampler at one RNG."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    robosuite_env = owner.robosuite_env
    rng = np.random.default_rng(seed)
    owner._rng = rng
    robosuite_env.seed = seed
    robosuite_env.rng = rng
    _set_sampler_rng(getattr(robosuite_env, "placement_initializer", None), rng)


__all__ = ["reseed_robosuite_owner"]
