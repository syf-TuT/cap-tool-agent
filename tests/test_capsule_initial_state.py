from __future__ import annotations

import pytest

from capx.rl.capsule.initial_state import (
    canonicalize_cube_initial_state,
    cube_initial_state_sha256,
    cube_initial_state_sha256_from_observation,
)


def _poses() -> dict[str, list[float]]:
    return {
        "secondary": [0.2, -0.0, 0.03, -1.0, -0.0, -0.0, -0.0],
        "primary": [0.1, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0],
    }


def test_cube_initial_state_hash_is_order_independent_and_normalizes_float_noise() -> None:
    poses = _poses()
    reordered = {"primary": poses["primary"], "secondary": poses["secondary"]}
    noisy_joints = [0.0, 0.1 + 1e-12, 0.2, 0.3, 0.4, 0.5, -0.0]
    exact_joints = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0]

    assert cube_initial_state_sha256(poses, noisy_joints) == cube_initial_state_sha256(
        reordered, exact_joints
    )
    canonical = canonicalize_cube_initial_state(poses, exact_joints)
    assert list(canonical["cube_poses"]) == ["primary", "secondary"]
    assert canonical["cube_poses"]["secondary"][3:] == [1.0, 0.0, 0.0, 0.0]
    assert canonical["robot_joints"][-1] == 0.0


def test_cube_initial_state_hash_changes_with_state() -> None:
    poses = _poses()
    changed = _poses()
    changed["primary"] = [0.11, *changed["primary"][1:]]

    assert cube_initial_state_sha256(poses, [0.0] * 7) != cube_initial_state_sha256(
        changed, [0.0] * 7
    )


def test_non_privileged_observation_without_cube_poses_keeps_reset_compatible() -> None:
    assert cube_initial_state_sha256_from_observation({}, [0.0] * 7) is None


@pytest.mark.parametrize(
    ("poses", "joints"),
    [
        ({"primary": [0.0, 0.0, float("nan"), 1.0, 0.0, 0.0, 0.0]}, [0.0] * 7),
        (_poses(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("inf")]),
        (_poses(), [0.0] * 6),
    ],
)
def test_cube_initial_state_rejects_invalid_state(
    poses: dict[str, list[float]], joints: list[float]
) -> None:
    with pytest.raises(ValueError):
        canonicalize_cube_initial_state(poses, joints)
