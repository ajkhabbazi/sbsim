"""EnergyPlusEnv in SBSim's environment format: a TF-Agents PyEnvironment.

Specs follow SBSim's Environment (smart_control/environment/environment.py):
actions are float32 in [-1, 1], observations a flat float32 vector, rewards
returned with ts.transition (discount factor, default 1) and ts.termination,
and a step after the last one starts a new episode.

Action 0 is the heating setpoint and action 1 the cooling setpoint. -1 is the
low end of each range in eplus_env.py and +1 the high end (0% to 100%).

Observations are the 8 readings of eplus_env.py with heating and cooling in kW
instead of W, so no input to the agent's networks is in the thousands.
"""
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from eplus_env import EnergyPlusEnv, HEAT_RANGE, COOL_RANGE, OBS_NAMES


def to_setpoints(action):
    """Map an action in [-1, 1] to (heating, cooling) setpoints in C."""
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    heat = HEAT_RANGE[0] + (a[0] + 1.0) / 2.0 * (HEAT_RANGE[1] - HEAT_RANGE[0])
    cool = COOL_RANGE[0] + (a[1] + 1.0) / 2.0 * (COOL_RANGE[1] - COOL_RANGE[0])
    return float(heat), float(cool)


def to_action(heat, cool):
    """Inverse of to_setpoints: setpoints in C to an action in [-1, 1]."""
    a0 = 2.0 * (heat - HEAT_RANGE[0]) / (HEAT_RANGE[1] - HEAT_RANGE[0]) - 1.0
    a1 = 2.0 * (cool - COOL_RANGE[0]) / (COOL_RANGE[1] - COOL_RANGE[0]) - 1.0
    return np.array([a0, a1], dtype=np.float32)


class EnergyPlusTFEnv(py_environment.PyEnvironment):
    def __init__(self, out_dir="out-tf-env", discount_factor=1.0):
        super().__init__()
        self._env = EnergyPlusEnv(out_dir=out_dir)
        self._discount = np.float32(discount_factor)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.float32, name="action", minimum=-1.0, maximum=1.0)
        self._observation_spec = array_spec.ArraySpec(
            shape=(len(OBS_NAMES),), dtype=np.float32, name="observation")
        self._episode_ended = False

    @property
    def out_dir(self):
        return self._env.out_dir

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    @staticmethod
    def _to_kw(obs):
        obs = obs.copy()
        obs[-2:] /= 1000.0     # heating and cooling, W -> kW
        return obs

    def _reset(self):
        self._episode_ended = False
        return ts.restart(self._to_kw(self._env.reset()))

    def _step(self, action):
        if self._episode_ended:
            return self.reset()
        obs, reward, done, _ = self._env.step(to_setpoints(action))
        obs = self._to_kw(obs)
        if done:
            self._episode_ended = True
            return ts.termination(obs, np.float32(reward))
        return ts.transition(obs, np.float32(reward), self._discount)

    def close(self):
        self._env.close()
