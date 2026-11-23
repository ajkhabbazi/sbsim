"""Time the two-longwave models and replay fixed HVAC forcing."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from . import benchmark
from . import coupled_radiation
from . import generate
from . import rl_env
from . import settings


METHODS = (
    ('full-nonlinear', 'sbsim', 'coupled-radiation', 1),
    ('air-nonlinear', 'sbsim', 'coupled-radiation', None),
    ('air-linear', 'sbsim', 'linear-radiation', None),
    ('air-wall-linear', 'sbsim', 'linear-radiation-envelope', None),
    ('eplus', 'eplus', 'jax', None),
)


def _replay(backend, hvac_heat):
  backend.reset()
  weather = backend.weather
  tick = pd.Timedelta(seconds=settings.STEP_SECONDS)
  frames = []
  for i, heat in enumerate(hvac_heat):
    stamp = weather.start + i * tick
    backend.sim._current_timestamp = stamp
    backend.sim.building.input_q = heat.copy()
    backend.sim.finite_differences_timestep(
        ambient_temperature=weather.get_current_temp(stamp),
        convection_coefficient=weather.get_air_convection_coefficient(stamp),
    )
    frames.append(backend._read(stamp)[:-1])
  return np.stack(frames)


def _errors(candidate, reference):
  error = np.abs(candidate - reference)
  return dict(zone_mae_c=float(error.mean()), zone_max_abs_c=float(error.max()))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--energyplus-dir', type=Path, required=True)
  parser.add_argument('--epw', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument(
      '--model-dir', type=Path,
      default=Path(__file__).with_name('models') / 'zone100_office',
  )
  parser.add_argument('--episode-days', type=int, default=7)
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()
  if args.episode_days < 1:
    parser.error('--episode-days must be positive')
  args.output.mkdir(parents=True, exist_ok=False)
  plan, run_day = generate.load_plan(args.model_dir)
  start = perf_counter()
  coupled_radiation._interior_view_factors(plan.grid.shape, plan.grid.astype(np.int8).tobytes())
  shared_view_setup_s = perf_counter() - start
  count = args.episode_days * rl_env.EPISODE_STEPS
  actions = np.random.default_rng(args.seed).uniform(-1, 1, (count, 2)).astype(np.float32)
  hvac_heat = np.empty((count, *plan.grid.shape))
  reports, traces, fixed_heat = {}, {}, {}
  for name, engine, solver, block in METHODS:
    output = args.output / name
    output.mkdir()
    start = perf_counter()
    env = rl_env.make_env(
        engine, args.model_dir, args.epw, args.energyplus_dir, output,
        sbsim_solver=solver, air_block_cells=block, episode_days=args.episode_days,
    )
    setup_s = perf_counter() - start
    frames, solves = [], Counter()
    try:
      env.reset()
      for i, action in enumerate(actions):
        if name == 'full-nonlinear':
          hvac_heat[i] = env.backend.sim.building.input_q
        obs = env.step(action).observation[:-1].astype(np.float64)
        frames.append(obs * 10 + 22.5)
        if engine == 'sbsim':
          solves[env.backend.sim.last_outer_iteration_count] += 1
    finally:
      env.close()
    if env.total_steps != count or env.completed_episodes != 1 or len(env.reset_times) != 1:
      raise RuntimeError(f'Incomplete experiment for {name}')
    traces[name] = np.stack(frames)
    reports[name] = dict(
        setup_s=setup_s, reset_s=env.reset_times[0], first_step_s=env.step_times[0],
        warm_step_ms=1000 * float(np.mean(env.step_times[1:])),
        heat_solves=dict(sorted(solves.items())),
        **benchmark._engine_info(env, engine),
    )
    if engine == 'sbsim':
      reports[name]['error_vs_full_nonlinear'] = _errors(
          traces[name], traces['full-nonlinear']
      )
    print(name, f"{reports[name]['warm_step_ms']:.3f} ms", flush=True)
    if engine == 'sbsim' and name != 'full-nonlinear':
      fixed_heat[name] = _replay(env.backend, hvac_heat)
      reports[name]['fixed_heat_error_vs_full_nonlinear'] = _errors(
          fixed_heat[name], traces['full-nonlinear']
      )
  reports['air-linear']['error_vs_air_nonlinear'] = _errors(
      traces['air-linear'], traces['air-nonlinear']
  )
  reports['air-linear']['fixed_heat_error_vs_air_nonlinear'] = _errors(
      fixed_heat['air-linear'], fixed_heat['air-nonlinear']
  )
  report = dict(
      cpu_affinity=sorted(os.sched_getaffinity(0)), start_date=run_day.isoformat(),
      steps=count, timestep_seconds=settings.STEP_SECONDS, seed=args.seed,
      shared_view_setup_s=shared_view_setup_s, epw=args.epw.name,
      epw_sha256=sha256(args.epw.read_bytes()).hexdigest(),
      action_sha256=sha256(actions.tobytes()).hexdigest(), variants=reports,
  )
  (args.output / 'summary.json').write_text(
      json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8'
  )


if __name__ == '__main__':
  main()
