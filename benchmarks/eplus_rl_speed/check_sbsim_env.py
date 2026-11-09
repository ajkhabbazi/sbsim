"""Step 7a: check that SBSim's RL environment runs on Windows, and time its steps.

Uses the public SBSim code in the folder set in bench_paths.py
(branch bench/rl-speed, from upstream copybara_push) and the same environment
setup as SBSim's own train.py: building sb1, sim_config_1_day.gin (1-day episodes).
"""
import os, sys, time
import numpy as np
from bench_paths import SBSIM_DIR

SBSIM = str(SBSIM_DIR)
sys.path.insert(0, SBSIM)

# config also registers the gin functions SBSim's configs call (paths, histogram reducer)
from smart_control.reinforcement_learning.utils.config import CONFIG_PATH
from smart_control.reinforcement_learning.utils.environment import create_and_setup_environment

N_STEPS = 12   # one simulated hour of 5-minute steps

config = os.path.join(CONFIG_PATH, "sim_config_1_day.gin")
t0 = time.perf_counter()
env = create_and_setup_environment(config, metrics_path=None)
print("load:", round(time.perf_counter() - t0, 2), "s")
print("action spec:", env.action_spec())
print("observation size:", env.observation_spec().shape)
print("steps per episode:", env._num_timesteps_in_episode)

t0 = time.perf_counter()
time_step = env.reset()
print("reset:", round(time.perf_counter() - t0, 2), "s")

action = np.zeros(env.action_spec().shape, dtype=np.float32)   # middle of each action range
times = []
for _ in range(N_STEPS):
    t0 = time.perf_counter()
    time_step = env.step(action)
    times.append(time.perf_counter() - t0)
print(f"{N_STEPS} steps: mean {np.mean(times):.3f} s per step "
      f"(first {times[0]:.3f} s, last {times[-1]:.3f} s)")
print("last reward:", float(time_step.reward))
