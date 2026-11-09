"""Step 4a test for EnergyPlusEnv.

1. One full year at fixed 21/24 C through reset()/step(), timed. With the same
   setpoints as run_control.py it must give the same annual site energy
   (271.51 GJ) and 105,120 steps.
2. A second run stopped after 10 steps, to check that close() ends a run
   cleanly and a new reset() works in the same process.
"""
import re, time
from pathlib import Path
from eplus_env import EnergyPlusEnv, OBS_NAMES

env = EnergyPlusEnv(out_dir="out-env")

# 1) full year, timed
t0 = time.perf_counter()
obs = env.reset()
first_obs = obs
steps, energy_kwh, done = 0, 0.0, False
while not done:
    obs, reward, done, info = env.step((21.0, 24.0))
    steps += 1
    energy_kwh -= reward
elapsed = time.perf_counter() - t0

html = (Path(env.out_dir) / "eplustbl.htm").read_text(errors="ignore")
site_gj = re.search(r"Total Site Energy</td>\s*<td[^>]*>\s*([\d.]+)", html).group(1)

print("first observation:")
for name, value in zip(OBS_NAMES, first_obs):
    print(f"  {name:20s} {value:10.2f}")
print("steps:", steps, "(expected 105120)")
print("time:", round(elapsed, 2), "s")
print("ideal-loads heating + cooling:", round(energy_kwh), "kWh")
print("total site energy:", site_gj, "GJ (run_control.py gave 271.51)")

# 2) early stop and a fresh reset
t0 = time.perf_counter()
env.reset()
for _ in range(10):
    env.step((20.0, 25.0))
env.close()
print("early stop after 10 steps: OK,", round(time.perf_counter() - t0, 2), "s")
