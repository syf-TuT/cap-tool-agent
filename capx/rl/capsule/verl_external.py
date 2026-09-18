"""VeRL worker import hook that registers the Capsule policy loss in each process."""

from __future__ import annotations

import os

from .compat import (
    VeRLCompatibilityError,
    check_verl_compatibility,
    verify_imported_verl_path,
)
from .policy_loss import register_capsule_critique_policy_loss


def initialize_verl_external() -> None:
    pinned_source = os.environ.get("CAPX_PINNED_VERL_SOURCE_PATH")
    if not pinned_source:
        raise RuntimeError(
            "CAPX_PINNED_VERL_SOURCE_PATH is required in every Capsule VeRL worker"
        )
    pinned_sha = os.environ.get("CAPX_PINNED_VERL_SHA")
    if not pinned_sha:
        raise RuntimeError("CAPX_PINNED_VERL_SHA is required in every Capsule VeRL worker")
    try:
        check_verl_compatibility(pinned_source, pinned_sha)
        verify_imported_verl_path(pinned_source)
    except VeRLCompatibilityError as error:
        raise RuntimeError(f"VeRL worker import identity check failed: {error}") from error
    if register_capsule_critique_policy_loss() is not True:
        raise RuntimeError("VeRL is unavailable while importing the Capsule worker extension")


initialize_verl_external()


__all__ = ["initialize_verl_external"]
