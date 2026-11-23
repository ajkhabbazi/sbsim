"""Compare the matched VAV solvers at fixed geometry and across CV sizes."""

import argparse
from dataclasses import replace
from datetime import date
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
from .floorplan import PlanSpec


def _run(engine, solver, block, model, epw, energyplus, output, actions):
  start = perf_counter()
  env = rl_env.make_env(
      engine, model, epw, energyplus, output, sbsim_solver=solver,
      air_block_cells=block,
  )
  setup_s = perf_counter() - start
  frames = []
  grid_hash = sha256() if engine == 'sbsim' else None
  try:
    env.reset()
    for action in actions:
      frame = env.step(action)
      frames.append(frame.observation[:-1].astype(np.float64) * 10 + 22.5)
      if grid_hash is not None:
        grid_hash.update(env.backend.sim.building.temp.tobytes())
    if env.total_steps != len(actions) or env.completed_episodes != 1:
      raise RuntimeError('Expected one completed 288-step episode')
  finally:
    env.close()
  metadata = benchmark._engine_info(env, engine)
  metadata.pop('solver', None)
  return dict(
      setup_s=setup_s,
      reset_s=env.reset_times[0],
      first_step_s=env.step_times[0],
      warm_step_ms=1000 * float(np.mean(env.step_times[1:])),
      grid_temperature_sha256=grid_hash.hexdigest() if grid_hash else None,
      **metadata,
  ), np.stack(frames)


def _replay_hvac_heat(model, epw):
  """Isolate the heat-balance approximation from thermostat feedback."""
  plan, run_day = generate.load_plan(model)
  rooms = [room.name for room in plan.rooms]
  source = rl_env._Sbsim(plan, epw, run_day, rooms, office=True, solver='fast-lu')
  reduced = {
      label: rl_env._Sbsim(
          plan, epw, run_day, rooms, office=True, solver='reduced-air',
          air_block_cells=block,
      )
      for label, block in (('air-blocks-2', 2), ('air-connected-zone', None))
  }
  source.reset()
  for candidate in reduced.values():
    candidate.reset()
  errors = {label: [] for label in reduced}
  actions = np.random.default_rng(0).uniform(
      -1, 1, (rl_env.EPISODE_STEPS, 2)
  ).astype(np.float32)
  for action in actions:
    stamp = source.sim.current_timestamp
    hvac_heat = source.sim.building.input_q.copy()
    reference = source.step(rl_env.setpoints(action))[:-1]
    for label, candidate in reduced.items():
      candidate.sim._current_timestamp = stamp
      candidate.sim.building.input_q = hvac_heat
      weather = candidate.weather
      if not candidate.sim.finite_differences_timestep(
          ambient_temperature=weather.get_current_temp(stamp),
          convection_coefficient=weather.get_air_convection_coefficient(stamp),
      ):
        raise RuntimeError(f'HVAC heat replay failed to converge: {label}/{stamp}')
      errors[label].append(np.abs(candidate._read(stamp)[:-1] - reference))
  return {
      label: dict(zone_mae_c=float(np.mean(err)), zone_max_abs_c=float(np.max(err)))
      for label, err in errors.items()
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--epw', type=Path, required=True)
  parser.add_argument('--energyplus-dir', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=False)

  aligned = args.output / 'models'
  fine, coarse = aligned / 'fine', aligned / 'coarse'
  spec = PlanSpec(100, room_width_m=6, wall_thickness_m=1, window_width_m=2)
  for resolution, model in ((0.5, fine), (1.0, coarse)):
    generate.write_models(
        model, replace(spec, cv_size_m=resolution), date(2024, 7, 15),
        '24.2', office=True,
    )
  energyplus_model = (fine / 'model.epJSON').read_bytes()
  if energyplus_model != (coarse / 'model.epJSON').read_bytes():
    raise RuntimeError('EnergyPlus model changed with SBSim CV size')

  models = Path(__file__).with_name('models')
  variants = {
      'eplus': ('eplus', 'jax', None),
      'jax': ('sbsim', 'jax', None),
      'cached-lu': ('sbsim', 'cached-lu', None),
      'fast-lu': ('sbsim', 'fast-lu', None),
      'air-blocks-2': ('sbsim', 'reduced-air', 2),
      'air-blocks-4': ('sbsim', 'reduced-air', 4),
      'air-connected-zone': ('sbsim', 'reduced-air', None),
      'fast-lu-1m': ('sbsim', 'fast-lu', None),
  }
  cases = {
      'zone05': (
          models / 'zone05',
          ('eplus', 'jax', 'cached-lu', 'fast-lu', 'air-blocks-2',
           'air-connected-zone'),
      ),
      'zone100_office': (
          models / 'zone100_office',
          ('eplus', 'jax', 'cached-lu', 'fast-lu', 'air-blocks-2',
           'air-blocks-4', 'air-connected-zone'),
      ),
      'zone100_aligned': (
          fine, ('eplus', 'cached-lu', 'fast-lu', 'fast-lu-1m'),
      ),
  }
  seeds = [0, 1, 2]
  runs, summaries = [], []
  for case, (model, names) in cases.items():
    for seed in seeds:
      actions = np.random.default_rng(seed).uniform(
          -1, 1, (rl_env.EPISODE_STEPS, 2)
      ).astype(np.float32)
      frames, paired = {}, {}
      for name in names[seed:] + names[:seed]:
        engine, solver, block = variants[name]
        output = args.output / 'runs' / case / str(seed) / name
        output.mkdir(parents=True)
        metrics, frames[name] = _run(
            engine, solver, block, coarse if name == 'fast-lu-1m' else model,
            args.epw, args.energyplus_dir, output, actions,
        )
        paired[name] = dict(
            case=case, variant=name, seed=seed,
            action_sha256=sha256(actions.tobytes()).hexdigest(), **metrics,
        )
        print(case, seed, name, f"{metrics['warm_step_ms']:.4f} ms", flush=True)
      if (paired['cached-lu']['grid_temperature_sha256']
          != paired['fast-lu']['grid_temperature_sha256']):
        raise RuntimeError(f'Full-grid temperature mismatch for {case}/{seed}')
      reference = frames['cached-lu']
      for name in names:
        row = paired[name]
        if name != 'eplus':
          error = np.abs(frames[name] - reference)
          row['zone_mae_c'] = float(error.mean())
          row['zone_max_abs_c'] = float(error.max())
        runs.append(row)
    for name in names:
      selected = [r for r in runs if (r['case'], r['variant']) == (case, name)]
      row = dict(case=case, variant=name)
      for key in ('warm_step_ms', 'first_step_s', 'setup_s', 'reset_s',
                  'zone_mae_c'):
        if key in selected[0]:
          row[key] = median(r[key] for r in selected)
      if 'zone_max_abs_c' in selected[0]:
        row['zone_max_abs_c'] = max(r['zone_max_abs_c'] for r in selected)
        row['active_heat_nodes'] = selected[0]['active_heat_nodes']
      summaries.append(row)

  report = dict(
      cpu_affinity=sorted(os.sched_getaffinity(0)),
      timestep_seconds=300,
      steps=rl_env.EPISODE_STEPS,
      seeds=seeds,
      epw=args.epw.name,
      epw_sha256=sha256(args.epw.read_bytes()).hexdigest(),
      aligned_energyplus_sha256=sha256(energyplus_model).hexdigest(),
      fixed_hvac_heat_replay_seed0=_replay_hvac_heat(
          models / 'zone100_office', args.epw
      ),
      summaries=summaries,
      runs=runs,
  )
  (args.output / 'comparison.json').write_text(
      json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8'
  )


if __name__ == '__main__':
  main()
