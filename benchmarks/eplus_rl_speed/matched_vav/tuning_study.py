"""Benchmark late full-grid CPU optimizations with exact state comparisons."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import median

import numpy as np

from . import benchmark
from . import rl_env
from . import settings


def _rollout(args, solver, actions, output):
  output.mkdir(parents=True)
  env = rl_env.make_env(
      'sbsim', args.model_dir, args.epw, args.energyplus_dir, output,
      sbsim_solver=solver, episode_days=7,
  )
  sim = env.backend.sim
  state = sha256()
  try:
    env.reset()
    for action in actions:
      env.step(action)
      state.update(sim.building.temp.tobytes())
      state.update(sim.building.input_q.tobytes())
      state.update(bytes(v.thermostat._current_mode.value for v in sim.hvac.vavs.values()))
  finally:
    env.close()
  if (
      env.total_steps != len(actions) or env.completed_episodes != 1
      or len(env.reset_times) != 1
  ):
    raise RuntimeError('Expected one continuous seven-day episode')
  return dict(
      warm_step_ms=1000 * float(np.mean(env.step_times[1:])),
      first_step_s=env.step_times[0],
      reset_and_steps_s=env.reset_times[0] + sum(env.step_times),
      state_sha256=state.hexdigest(),
      action_sha256=sha256(actions.tobytes()).hexdigest(),
      **benchmark._engine_info(env, 'sbsim'),
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--energyplus-dir', type=Path, required=True)
  parser.add_argument('--epw', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument(
      '--model-dir', type=Path,
      default=Path(__file__).with_name('models') / 'zone100_office',
  )
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=False)
  methods = ('optimized-lu', 'tuned-lu')
  steps = 7 * rl_env.EPISODE_STEPS
  runs = []
  for seed in range(3):
    actions = np.random.default_rng(seed).uniform(
        -1, 1, (steps, 2)
    ).astype(np.float32)
    pair = []
    for solver in methods if seed % 2 == 0 else methods[::-1]:
      record = _rollout(args, solver, actions, args.output / f'{seed}-{solver}')
      pair.append(dict(record, seed=seed))
      print(seed, solver, f"{record['warm_step_ms']:.3f} ms", flush=True)
    if pair[0]['state_sha256'] != pair[1]['state_sha256']:
      raise RuntimeError(f'Full-grid HVAC or temperature state changed for seed {seed}')
    runs.extend(pair)
  report = dict(
      steps=steps,
      timestep_seconds=settings.STEP_SECONDS,
      cpu_affinity=sorted(os.sched_getaffinity(0)),
      epw=args.epw.name,
      epw_sha256=sha256(args.epw.read_bytes()).hexdigest(),
      medians={
          solver: {
              key: median(r[key] for r in runs if r['solver'] == solver)
              for key in ('warm_step_ms', 'reset_and_steps_s')
          }
          for solver in methods
      },
      runs=runs,
  )
  (args.output / 'summary.json').write_text(
      json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8'
  )


if __name__ == '__main__':
  main()
