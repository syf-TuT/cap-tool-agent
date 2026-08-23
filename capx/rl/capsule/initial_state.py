"""Canonical Cube Stack initial-state hashing without simulator imports."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

_POSE_NAMES = ("primary", "secondary")
_POSE_LENGTH = 7
_JOINT_COUNT = 7
_DECIMALS = 10


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    result = round(result, _DECIMALS)
    return 0.0 if result == 0.0 else result


def _canonical_pose(values: Sequence[object], name: str) -> list[float]:
    if len(values) != _POSE_LENGTH:
        raise ValueError(f"{name} pose must contain xyz plus a WXYZ quaternion")
    position = [_finite_float(value, f"{name} pose") for value in values[:3]]
    quaternion = [_finite_float(value, f"{name} quaternion") for value in values[3:]]
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm == 0.0:
        raise ValueError(f"{name} quaternion must be non-zero")
    quaternion = [_finite_float(value / norm, f"{name} quaternion") for value in quaternion]
    first_nonzero = next((value for value in quaternion if value != 0.0), 1.0)
    if first_nonzero < 0.0:
        quaternion = [-value if value != 0.0 else 0.0 for value in quaternion]
    return position + quaternion


def canonicalize_cube_initial_state(
    cube_poses: Mapping[str, Sequence[object]],
    robot_joints: Sequence[object],
) -> dict[str, Any]:
    """Return a stable JSON-compatible representation for Cube Stack reset state."""

    if set(cube_poses) != set(_POSE_NAMES):
        raise ValueError("cube_poses must contain exactly primary and secondary")
    if len(robot_joints) != _JOINT_COUNT:
        raise ValueError("robot_joints must contain exactly seven values")
    canonical_poses = {
        name: _canonical_pose(cube_poses[name], name)
        for name in _POSE_NAMES
    }
    canonical_joints = [_finite_float(value, "robot joint") for value in robot_joints]
    return {"cube_poses": canonical_poses, "robot_joints": canonical_joints}


def cube_initial_state_sha256(
    cube_poses: Mapping[str, Sequence[object]],
    robot_joints: Sequence[object],
) -> str:
    payload = canonicalize_cube_initial_state(cube_poses, robot_joints)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cube_initial_state_sha256_from_observation(
    observation: Mapping[str, Any],
    robot_joints: Sequence[object],
) -> str | None:
    """Hash privileged state, while leaving existing visual resets unchanged."""

    cube_poses = observation.get("cube_poses")
    if cube_poses is None:
        return None
    if not isinstance(cube_poses, Mapping):
        raise ValueError("observation cube_poses must be a mapping")
    return cube_initial_state_sha256(cube_poses, robot_joints)


__all__ = [
    "canonicalize_cube_initial_state",
    "cube_initial_state_sha256",
    "cube_initial_state_sha256_from_observation",
]
