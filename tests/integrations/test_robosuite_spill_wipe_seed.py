"""Requires the prepared Linux Robosuite/MuJoCo environment."""
import numpy as np

from capx.envs.simulators.robosuite_spill_wipe import FrankaRobosuiteSpillWipeLowLevel


def test_spill_wipe_reset_reproduces_robot_and_spill():
    paired_states = []
    for privileged in (True, False):
        env = FrankaRobosuiteSpillWipeLowLevel(privileged=privileged, enable_render=False)
        states = []
        try:
            for seed in (1, 2, 1):
                env.reset(seed=seed)
                sim = env.robosuite_env.sim
                markers = env.robosuite_env.model.mujoco_arena.markers
                positions = [sim.data.body_xpos[sim.model.body_name2id(m.root_body)] for m in markers]
                states.append(np.concatenate((sim.data.qpos.copy(), np.array(positions).ravel())))
        finally:
            env.close()
        np.testing.assert_allclose(states[0], states[2], atol=1e-10, rtol=0)
        assert not np.allclose(states[0], states[1])
        paired_states.append(states)
    np.testing.assert_allclose(paired_states[0], paired_states[1], atol=1e-10, rtol=0)
