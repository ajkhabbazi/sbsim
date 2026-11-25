"""Replay the saved full-reference HVAC with a grouped nonlinear heat balance.

This is a targeted supplement to results/restructure_cpu/physics.npz, not a new
full benchmark. It uses the same public GroupedJaxSimulator path and sparse LU
as the saved grouped linear physics replay, changing only radiation iteration.
"""

import argparse
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd


def replay_saved_hvac(sim, weather, start, forcing, diffuser_indices, steps):
  """Apply recorded heat without stepping the thermostat or VAV again."""
  if steps < 1 or forcing.ndim != 2 or forcing.shape[0] < steps:
    raise ValueError('HVAC replay requires at least the requested positive steps')
  if forcing.shape[1] != len(diffuser_indices):
    raise ValueError('HVAC forcing and diffuser indices do not agree')
  forcing = np.asarray(forcing, dtype=np.float64)
  if not np.isfinite(forcing).all():
    raise ValueError('HVAC forcing contains a nonfinite value')
  zones = len(sim.hvac.vavs) if hasattr(sim, 'hvac') else len(
      sim.building.get_zone_average_temps()
  )
  temperatures = np.empty((steps, zones))
  durations = np.empty(steps)
  solves = np.empty(steps, dtype=int)
  interval_end = []
  for step in range(steps):
    stamp = start + step * pd.Timedelta(minutes=5)
    sim._current_timestamp = stamp  # Matches run.py._heat_balance_trace.
    sim.building.input_q.flat[diffuser_indices] = forcing[step]
    begin = perf_counter()
    sim.finite_differences_timestep(
        ambient_temperature=weather.get_current_temp(stamp),
        convection_coefficient=weather.get_air_convection_coefficient(stamp),
    )
    temperatures[step] = list(sim.building.get_zone_average_temps().values())
    durations[step] = perf_counter() - begin
    solves[step] = sim.last_solve_count
    # finite_differences_timestep updates the building, not simulator time.
    # _heat_balance_trace likewise derives its report timestamps from the start.
    expected_end = start + (step + 1) * pd.Timedelta(minutes=5)
    interval_end.append(expected_end)
    if (step + 1) % 288 == 0:
      print(f'Grouped nonlinear replay: {step + 1}/{steps}', flush=True)
  return dict(
      temperatures_k=temperatures,
      step_s=durations,
      linear_solves=solves,
      interval_end=np.array([stamp.isoformat() for stamp in interval_end]),
  )


def _replay_arguments(setup, epw):
  """Match run.py's per-physics-section grouped solver override."""
  return SimpleNamespace(
      days=setup['days'],
      run_day=date.fromisoformat(setup['start_day']),
      energyplus_version=setup['energyplus_input_version'],
      epw=epw.resolve(),
      seed=setup['seed'],
      platform='cpu',
      grouped_method='sparse_lu',
      physics_grouped_method='sparse_lu',
  )


def run_saved_case(source, epw, output):
  """Run the complete existing 100-zone replay, preserving its source files."""
  # Heavy benchmark/JAX imports remain deferred until this replay is explicitly
  # requested, so analyzing saved artifacts does not initialize a JAX device.
  from . import run as study

  source = source.resolve()
  if not epw.is_file():
    raise ValueError('EPW weather file does not exist')
  if output.exists():
    raise ValueError('Replay output already exists; choose a new file')
  summary = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
  setup = summary['setup']
  if summary.get('physics_zones') != 100 or setup.get('platform') != 'cpu':
    raise ValueError('This supplement targets the saved 100-zone CPU case')
  if sha256(epw.read_bytes()).hexdigest() != setup.get('epw_sha256'):
    raise ValueError('EPW hash does not match the saved benchmark')
  args = _replay_arguments(setup, epw)
  plan, weather, _ = study._model(args, source, 100, needs_energyplus=False)
  with np.load(source / 'physics.npz', allow_pickle=False) as data:
    forcing = data['hvac_diffuser_w']
    diffuser_indices = data['diffuser_flat_indices']
    source_end = data['interval_end']
    source_zones = data['zones'].tolist()
  environment = study._SbsimOffice(
      plan, weather, args, 'grouped_jax',
      linearized_radiative_heat_transfer=False,
  )
  simulation = environment.sim
  if (not simulation.grouped or simulation.solver_method != 'sparse_lu'
      or simulation.linearized_radiative_heat_transfer
      or simulation._linear_heat_balance):
    raise ValueError('Expected prepared grouped nonlinear sparse LU replay')
  source_grouped = next(
      row for row in summary['physics']['metrics']
      if row['variant'] == 'grouped_jax'
  )
  states = (simulation.thermal_node_count, simulation.mass_node_count)
  if states != (source_grouped['thermal_nodes'], source_grouped['mass_nodes']):
    raise ValueError('Prepared grouped state counts differ from the saved source')
  environment.reset(args.seed)
  if source_zones != [room.name for room in plan.rooms]:
    raise ValueError('The local plan differs from the saved zone order')
  if not np.array_equal(
      np.flatnonzero(simulation.building.diffusers), diffuser_indices
  ):
    raise ValueError('The local HVAC diffusers differ from the saved source')
  simulation._current_timestamp = weather.start
  result = replay_saved_hvac(
      simulation, weather, weather.start, forcing, diffuser_indices,
      len(source_end)
  )
  if not np.array_equal(result['interval_end'], source_end):
    raise ValueError('Replay timestamps differ from the saved source')
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
      output,
      grouped_nonlinear_c=result['temperatures_k'] - 273.15,
      step_s=result['step_s'],
      linear_solves=result['linear_solves'],
      interval_end=result['interval_end'],
      zones=np.array(source_zones),
      source_physics_sha256=np.array(sha256(
          (source / 'physics.npz').read_bytes()
      ).hexdigest()),
      source_summary_sha256=np.array(sha256(
          (source / 'summary.json').read_bytes()
      ).hexdigest()),
      solver_method=np.array('sparse_lu'),
      nonlinear=np.array(True),
      prepared_nonlinear=np.array(not simulation._linear_heat_balance),
      thermal_node_count=np.array(simulation.thermal_node_count),
      mass_node_count=np.array(simulation.mass_node_count),
  )
  print(f'Wrote {output.resolve()}', flush=True)
  print(f"Final replayed interval: {result['interval_end'][-1]}", flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--source', type=Path, required=True)
  parser.add_argument('--epw', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  run_saved_case(args.source, args.epw, args.output)


if __name__ == '__main__':
  main()
