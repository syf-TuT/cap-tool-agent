from __future__ import annotations

import numpy as np

from capx.envs.simulators.robosuite_cubes_restack import (
    FrankaRobosuiteCubesRestackLowLevel,
)


def _cube_positions(observation: dict) -> np.ndarray:
    cube_poses = observation["cube_poses"]
    return np.stack(
        [
            np.asarray(cube_poses["primary"][:3], dtype=np.float64),
            np.asarray(cube_poses["secondary"][:3], dtype=np.float64),
        ]
    )


def test_cube_restack_reset_reproduces_initial_cube_positions_for_seed() -> None:
    env = FrankaRobosuiteCubesRestackLowLevel(
        seed=5,
        privileged=True,
        enable_render=False,
    )
    try:
        first_seed_five, _ = env.reset(seed=5)
        seed_six, _ = env.reset(seed=6)
        second_seed_five, _ = env.reset(seed=5)
    finally:
        env.robosuite_env.close()

    first_positions = _cube_positions(first_seed_five)
    second_positions = _cube_positions(second_seed_five)
    alternate_positions = _cube_positions(seed_six)

    np.testing.assert_allclose(first_positions, second_positions, rtol=0.0, atol=1e-7)
    assert not np.allclose(first_positions, alternate_positions, rtol=0.0, atol=1e-7)

