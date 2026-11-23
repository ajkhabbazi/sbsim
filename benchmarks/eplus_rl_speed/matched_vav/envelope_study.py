"""Compare full-grid and grouped envelope models with exterior radiation."""

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


_VARIANTS = {
    'fast-lu': ('sbsim', 'fast-lu', None),
    'eplus': ('eplus', 'fast-lu', None),
    'optimized-lu': ('sbsim', 'optimized-lu', None),
    'wall-only': ('sbsim', 'reduced-envelope', 1),
    'reduced-envelope': ('sbsim', 'reduced-envelope', None),
    'condensed-envelope': ('sbsim', 'condensed-envelope', None),
    'jax': ('sbsim', 'jax', None),
}


def _run(args, variant, actions, output, *, grid=None, reference_modes=None):
  engine, solver, block = _VARIANTS[variant]
  output.mkdir(parents=True)
  env = rl_env.make_env(
      engine, args.model_dir, args.epw, args.energyplus_dir, output,
      sbsim_solver=solver, air_block_cells=block,
      episode_days=len(actions) // rl_env.EPISODE_STEPS,
  )
  sim = env.backend.sim if engine == 'sbsim' else None
  grid_max = 0.0
  modes_changed = 0
  observations = []
  try:
    env.reset()
    if variant == 'fast-lu':
      grid = np.empty((len(actions), *sim.building.temp.shape))
      reference_modes = np.empty((len(actions), len(env.backend.names)), dtype=np.uint8)
    for idx, action in enumerate(actions):
      frame = env.step(action)
      observations.append(frame.observation[:-1].astype(np.float64))
      if sim is None:
        continue
      modes = np.fromiter(
          (v.thermostat._current_mode.value for v in sim.hvac.vavs.values()),
          dtype=np.uint8,
      )
      if variant == 'fast-lu':
        grid[idx] = sim.building.temp
        reference_modes[idx] = modes
      else:
        grid_max = max(grid_max, float(np.max(np.abs(sim.building.temp - grid[idx]))))
        modes_changed += int(np.count_nonzero(modes != reference_modes[idx]))
  finally:
    env.close()
  if (
      env.total_steps != len(actions) or env.completed_episodes != 1
      or len(env.reset_times) != 1
  ):
    raise RuntimeError('Expected one completed, continuous episode')
  timings = np.asarray(env.step_times)
  row = dict(
      variant=variant, steps=len(actions), reset_s=env.reset_times[0],
      warm_step_ms=float(timings[1:].mean() * 1000),
      first_day_warm_step_ms=float(timings[1:rl_env.EPISODE_STEPS].mean() * 1000),
      reset_and_steps_s=float(env.reset_times[0] + timings.sum()),
      action_sha256=sha256(actions.tobytes()).hexdigest(),
      **benchmark._engine_info(env, engine),
  )
  if sim is not None:
    row.update(grid_max_abs_c=grid_max, changed_vav_modes=modes_changed)
  return row, np.stack(observations), grid, reference_modes


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
  args = parser.parse_args()
  if args.episode_days < 1:
    parser.error('--episode-days must be positive')
  args.output.mkdir(parents=True, exist_ok=False)
  steps = args.episode_days * rl_env.EPISODE_STEPS
  runs = []
  for seed in range(3):
    actions = np.random.default_rng(seed).uniform(-1, 1, (steps, 2)).astype(np.float32)
    row, reference, grid, modes = _run(
        args, 'fast-lu', actions, args.output / f'{seed}-fast-lu'
    )
    runs.append(dict(row, seed=seed, zone_mae_c=0., zone_max_abs_c=0.))
    traces, records = {}, {}
    alternatives = tuple(_VARIANTS)[1:-1]
    for name in alternatives[seed:] + alternatives[:seed]:
      row, trace, _, _ = _run(
          args, name, actions, args.output / f'{seed}-{name}',
          grid=grid, reference_modes=modes,
      )
      if name != 'eplus':
        error = np.abs(trace - reference) * 10
        row.update(zone_mae_c=float(error.mean()), zone_max_abs_c=float(error.max()))
        traces[name] = trace
      if name == 'optimized-lu' and (
          row['grid_max_abs_c'] > 1e-7 or row['changed_vav_modes']
      ):
        raise RuntimeError(f'Full-grid optimization changed the trajectory: {row}')
      records[name] = dict(row, seed=seed)
      print(seed, name, f"{row['warm_step_ms']:.3f} ms", flush=True)
    condensed_error = np.abs(traces['condensed-envelope'] - traces['reduced-envelope'])
    records['condensed-envelope']['grouped_zone_max_abs_c'] = float(
        condensed_error.max() * 10
    )
    runs.extend(records.values())
    if seed == 0:
      one_day = actions[:rl_env.EPISODE_STEPS]
      jax, _, _, _ = _run(
          args, 'jax', one_day, args.output / 'jax-one-day', grid=grid,
          reference_modes=modes,
      )
      print('jax one day', f"{jax['warm_step_ms']:.3f} ms", flush=True)
  medians = {
      solver: {
          key: median(r[key] for r in runs if r['variant'] == solver)
          for key in ('warm_step_ms', 'reset_and_steps_s')
      }
      for solver in _VARIANTS if solver != 'jax'
  }
  report = dict(
      cpu_affinity=sorted(os.sched_getaffinity(0)), steps=steps,
      timestep_seconds=settings.STEP_SECONDS, episode_days=args.episode_days,
      epw=args.epw.name, epw_sha256=sha256(args.epw.read_bytes()).hexdigest(),
      medians=medians, runs=runs, jax_one_day=jax,
  )
  (args.output / 'summary.json').write_text(
      json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8'
  )


if __name__ == '__main__':
  main()
