"""Measure exterior radiation ablations on the matched office VAV model."""

import argparse
from collections import Counter
from collections import defaultdict
from functools import wraps
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from . import benchmark
from . import generate
from . import rl_env
from . import settings


def _meter(instance, name, charges):
  method = getattr(instance, name)

  @wraps(method)
  def timed(*args, **kwargs):
    start = perf_counter()
    try:
      return method(*args, **kwargs)
    finally:
      charges[name] += perf_counter() - start

  setattr(instance, name, timed)


def _run(args, variant, actions, output, *, profile=False):
  engine = 'eplus' if variant == 'eplus' else 'sbsim'
  solver = 'reduced-air' if variant == 'reduced-air' else args.sbsim_solver
  mode = variant if variant in ('linear-longwave', 'solar-only', 'off') else 'full'
  output.mkdir(parents=True)
  env = rl_env.make_env(
      engine, args.model_dir, args.epw, args.energyplus_dir, output,
      sbsim_solver=solver, episode_days=args.episode_days,
      exterior_radiation=mode,
  )
  iterations, charges, observations = Counter(), defaultdict(float), []
  backend = env.backend
  sim = backend.sim if engine == 'sbsim' else None
  try:
    env.reset()
    for i, action in enumerate(actions):
      frame = env.step(action)
      observations.append(frame.observation[:-1])
      if sim is not None:
        iterations[sim.last_outer_iteration_count] += 1
      if profile and not i:
        # Exclude static geometry and factorization setup from the profile.
        for method in (
            'setup_step_sim', '_prepare_exterior_radiation',
            '_iterate_temperature_timestep',
        ):
          _meter(sim, method, charges)
  finally:
    env.close()
  if env.total_steps != len(actions) or len(env.reset_times) != 1:
    raise RuntimeError('Expected one continuous rollout per variant')
  if not profile and env.completed_episodes != 1:
    raise RuntimeError('Expected one completed episode')
  if mode != 'full' and iterations != {1: len(actions)}:
    raise RuntimeError('Expected one linear solve per timestep')
  warm = np.asarray(env.step_times[1:])
  info = benchmark._engine_info(env, engine)
  report = dict(
      variant=variant, steps=env.total_steps, reset_s=env.reset_times[0],
      first_step_s=env.step_times[0], warm_step_ms=float(warm.mean() * 1000),
      action_sha256=sha256(actions.tobytes()).hexdigest(), **info,
  )
  if sim is not None:
    report.update(
        thermal_solves=dict(sorted(iterations.items())),
        triangular_solves=sim.triangular_solve_count,
        factorizations=sim.factorization_count,
    )
  if profile:
    profile_ms = {name: time * 1000 / len(warm) for name, time in charges.items()}
    profile_ms['other'] = report['warm_step_ms'] - sum(profile_ms.values())
    report['profile_ms_per_step'] = profile_ms
  return report, np.stack(observations)


def _replay_fixed_hvac_heat(args, actions, modes):
  plan, run_day = generate.load_plan(args.model_dir)
  office = generate.read_manifest(args.model_dir).get('office', False)
  names = [room.name for room in plan.rooms]
  models = {
      mode: rl_env._Sbsim(
          plan, args.epw, run_day, names, office=office, solver=args.sbsim_solver,
          episode_days=args.episode_days, exterior_radiation=mode,
      )
      for mode in modes
  }
  for backend in models.values():
    backend.reset()
  reference = models.pop('full')
  errors = {mode: [] for mode in models}
  for action in actions:
    stamp = reference.sim.current_timestamp
    power = reference.sim.building.input_q.copy()
    reference_t = reference.step(rl_env.setpoints(action))[:-1]
    for mode, backend in models.items():
      backend.sim._current_timestamp = stamp
      backend.sim.building.input_q = power
      if not backend.sim.finite_differences_timestep(
          ambient_temperature=backend.weather.get_current_temp(stamp),
          convection_coefficient=backend.weather.get_air_convection_coefficient(stamp),
      ):
        raise RuntimeError(f'Fixed HVAC heat replay failed: {mode}/{stamp}')
      errors[mode].append(np.abs(backend._read(stamp)[:-1] - reference_t))
  return {
      mode: dict(zone_mae_c=float(np.mean(error)), zone_max_abs_c=float(np.max(error)))
      for mode, error in errors.items()
  }


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
  parser.add_argument('--sbsim-solver', choices=['fast-lu', 'tuned-lu'], default='fast-lu')
  parser.add_argument('--include-linear-longwave', action='store_true')
  args = parser.parse_args()
  if args.episode_days < 1:
    parser.error('--episode-days must be positive')
  args.output.mkdir(parents=True, exist_ok=False)
  modes = ('full', 'solar-only', 'off')
  if args.include_linear_longwave:
    modes = ('full', 'linear-longwave', 'solar-only', 'off')
  variants = ('eplus', *modes)
  steps = args.episode_days * rl_env.EPISODE_STEPS
  runs = []
  for seed in range(3):
    actions = np.random.default_rng(seed).uniform(-1, 1, (steps, 2)).astype(np.float32)
    records, temperatures = {}, {}
    for variant in variants[seed:] + variants[:seed]:
      result, trace = _run(args, variant, actions, args.output / f'{seed}-{variant}')
      records[variant] = dict(result, seed=seed)
      temperatures[variant] = trace
      print(seed, variant, f"{result['warm_step_ms']:.3f} ms", flush=True)
    for variant, row in records.items():
      if variant != 'eplus':
        error = np.abs(temperatures[variant] - temperatures['full']) * 10.0
        row.update(
            zone_mae_c=float(error.mean()), zone_max_abs_c=float(error.max()),
        )
      runs.append(row)
  profile_actions = actions[:rl_env.EPISODE_STEPS]
  profiles = [
      _run(args, variant, profile_actions, args.output / f'profile-{variant}',
           profile=True)[0]
      for variant in ('reduced-air', *modes)
  ]
  medians = {}
  for variant in variants:
    selected = [r for r in runs if r['variant'] == variant]
    row = {metric: median(r[metric] for r in selected) for metric in (
        'warm_step_ms', 'reset_s', 'first_step_s', 'zone_mae_c'
    ) if metric in selected[0]}
    if 'zone_max_abs_c' in selected[0]:
      row['zone_max_abs_c'] = max(r['zone_max_abs_c'] for r in selected)
    medians[variant] = row
  report = dict(
      cpu_affinity=sorted(os.sched_getaffinity(0)),
      sbsim_solver=args.sbsim_solver,
      episode_days=args.episode_days, steps=steps,
      timestep_seconds=settings.STEP_SECONDS,
      epw=args.epw.name, epw_sha256=sha256(args.epw.read_bytes()).hexdigest(),
      medians=medians, runs=runs,
      profiles=profiles,
      fixed_hvac_heat_seed0=_replay_fixed_hvac_heat(
          args, np.random.default_rng(0).uniform(-1, 1, (steps, 2)).astype(np.float32),
          modes,
      ),
  )
  (args.output / 'summary.json').write_text(
      json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8'
  )


if __name__ == '__main__':
  main()
