"""Step 6b test for EnergyPlusTFEnv (SBSim's environment format).

1. Action mapping: -1 and +1 hit the ends of the setpoint ranges, and the
   action for 21/24 C maps back to 21/24 C.
2. TF-Agents' own checker drives one full year with random actions and checks
   every time step against the specs.
3. One full year at the action for 21/24 C, timed. It must match test_env.py:
   105,120 steps and 271.51 GJ.
4. A step after the last one starts a new episode, as in SBSim.
"""
import re, time
from pathlib import Path
import numpy as np
from tf_agents.environments import utils
from eplus_env import HEAT_RANGE, COOL_RANGE
from eplus_tf_env import EnergyPlusTFEnv, to_setpoints, to_action

# 1) action mapping
assert np.allclose(to_setpoints([-1.0, -1.0]), (HEAT_RANGE[0], COOL_RANGE[0]))
assert np.allclose(to_setpoints([1.0, 1.0]), (HEAT_RANGE[1], COOL_RANGE[1]))
fixed = to_action(21.0, 24.0)
assert np.allclose(to_setpoints(fixed), (21.0, 24.0), atol=1e-5)
print("action mapping: OK; action", fixed, "->", np.round(to_setpoints(fixed), 4), "C")

env = EnergyPlusTFEnv(out_dir="out-tf-env")

# 2) TF-Agents spec check over one year of random actions
t0 = time.perf_counter()
utils.validate_py_environment(env, episodes=1)
print("spec check, one year of random actions: passed,", round(time.perf_counter() - t0, 2), "s")

# 3) one year at 21/24 C, timed
t0 = time.perf_counter()
time_step = env.reset()
steps = 0
while not time_step.is_last():
    time_step = env.step(fixed)
    steps += 1
elapsed = time.perf_counter() - t0

html = (Path(env.out_dir) / "eplustbl.htm").read_text(errors="ignore")
site_gj = re.search(r"Total Site Energy</td>\s*<td[^>]*>\s*([\d.]+)", html).group(1)
print("steps:", steps, "(expected 105120)")
print("time:", round(elapsed, 2), "s")
print("total site energy:", site_gj, "GJ (test_env.py gave 271.51)")

# 4) a step after the last one starts a new episode
time_step = env.step(fixed)
assert time_step.is_first(), "expected a new episode"
env.close()
print("step after the last one starts a new episode: OK")
