from __future__ import annotations

import pytest

from capx.rl.capsule.initial_state import (
    canonicalize_cube_lift_initial_state,
    canonicalize_cube_initial_state,
    cube_lift_initial_state_sha256,
    cube_lift_initial_state_sha256_from_observation,
    cube_initial_state_sha256,
    cube_initial_state_sha256_from_observation,
)


def _poses() -> dict[str, list[float]]:
    return {
        "secondary": [0.2, -0.0, 0.03, -1.0, -0.0, -0.0, -0.0],
        "primary": [0.1, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0],
    }


def _lift_poses() -> dict[str, list[float]]:
    return {"primary": [0.1, -0.0, 0.03, 2.0, 0.0, 0.0, 0.0]}


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


def test_cube_stack_initial_state_hash_regression_vector_is_unchanged() -> None:
    assert cube_initial_state_sha256(
        _poses(), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0]
    ) == "ce23215e037db9daf4cc8b375a401da9fe696e5065a2a1be397b72bc8a0056af"


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


def test_cube_lift_initial_state_accepts_primary_pose_and_normalizes_representation() -> None:
    poses = _lift_poses()
    equivalent_poses = {
        "primary": [0.1 + 1e-12, 0.0, 0.03, -2.0, -0.0, -0.0, -0.0]
    }
    exact_joints = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0]
    noisy_joints = [-0.0, 0.1 + 1e-12, 0.2, 0.3, 0.4, 0.5, -0.0]

    assert cube_lift_initial_state_sha256(poses, exact_joints) == (
        cube_lift_initial_state_sha256(equivalent_poses, noisy_joints)
    )
    canonical = canonicalize_cube_lift_initial_state(poses, exact_joints)
    assert list(canonical["cube_poses"]) == ["primary"]
    assert canonical["cube_poses"]["primary"] == [0.1, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0]
    assert canonical["robot_joints"][-1] == 0.0


def test_cube_lift_initial_state_hash_changes_with_cube_position_or_robot_joint() -> None:
    poses = _lift_poses()
    moved_cube = _lift_poses()
    moved_cube["primary"][0] = 0.11
    joints = [0.0] * 7
    moved_joint = [0.01, *joints[1:]]
    baseline = cube_lift_initial_state_sha256(poses, joints)

    assert baseline != cube_lift_initial_state_sha256(moved_cube, joints)
    assert baseline != cube_lift_initial_state_sha256(poses, moved_joint)


@pytest.mark.parametrize(
    ("poses", "joints"),
    [
        ({}, [0.0] * 7),
        (
            {
                **_lift_poses(),
                "secondary": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            },
            [0.0] * 7,
        ),
        ({"primary": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]}, [0.0] * 7),
        ({"primary": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, [0.0] * 7),
        ({"primary": [True, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}, [0.0] * 7),
        ({"primary": [float("nan"), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}, [0.0] * 7),
        (_lift_poses(), [0.0] * 6),
        (_lift_poses(), [0.0] * 8),
        (_lift_poses(), [False, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (_lift_poses(), [0.0, 0.0, 0.0, 0.0, 0.0, float("inf"), 0.0]),
    ],
)
def test_cube_lift_initial_state_rejects_invalid_state(
    poses: dict[str, list[object]], joints: list[object]
) -> None:
    with pytest.raises(ValueError):
        canonicalize_cube_lift_initial_state(poses, joints)


def test_cube_lift_non_privileged_observation_without_cube_poses_stays_compatible() -> None:
    assert cube_lift_initial_state_sha256_from_observation({}, [0.0] * 7) is None


def test_cube_lift_observation_rejects_non_mapping_cube_poses() -> None:
    with pytest.raises(ValueError, match="observation cube_poses must be a mapping"):
        cube_lift_initial_state_sha256_from_observation({"cube_poses": []}, [0.0] * 7)
