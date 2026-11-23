"""Paired CPU benchmark for exterior radiation on partitioned enclosures."""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
import time

import jax
import numpy as np
import pandas as pd
import tensorflow as tf

from reference import application_grid_time_sensitivity as solvers
from reference import partitioned_enclosure_grid_benchmark as grid
from smart_control.simulator import constants
from smart_control.simulator import jax_simulator
from smart_control.simulator import weather_controller


APPLICATIONS = ('base_rad', 'base_rad_interior_mass')
SOLVERS = (
    'numpy_scipy',
    'tf_cg',
    'tf_cg_resident_fp32',
    'jax_cg',
    'jax_cg_resident',
    'jax_cg_resident_fp32',
)
START = pd.Timestamp('2025-06-21 11:00', tz='America/Los_Angeles')
CV_SIZE_CM = 20.0
ROOM_INTERIOR_CVS = 25
AMBIENT = 305.0
CONVECTION = 12.0


class _BenchmarkWeather(weather_controller.WeatherController):
  """Measures the weather component of the actual solver entry point."""

  def __init__(self):
    super().__init__(AMBIENT, AMBIENT, convection_coefficient=CONVECTION)
    self.sample_count = 0
    self.sample_seconds = 0.0

  def get_exterior_radiation(self, timestamp, ambient_temperature):
    start = time.perf_counter()
    sample = super().get_exterior_radiation(timestamp, ambient_temperature)
    self.sample_seconds += time.perf_counter() - start
    self.sample_count += 1
    return sample


def _glazed_floor_plan(rooms_per_side):
  plan = grid._create_partitioned_floor_plan(rooms_per_side, ROOM_INTERIOR_CVS)
  tile_size = ROOM_INTERIOR_CVS + 2
  # Five glazing CVs centered on each perimeter room's exterior facade.
  for room in range(rooms_per_side):
    midpoint = 1 + room * tile_size + tile_size // 2
    window = slice(midpoint - 2, midpoint + 3)
    glass = constants.FENESTRATION_VALUE_IN_FILE_INPUT
    plan[1, window] = plan[-2, window] = glass
    plan[window, 1] = plan[window, -2] = glass
  return plan


def _initialize(simulator, template):
  grid._apply_initial_state(simulator, template)
  grid._set_forcing(simulator)
  if not isinstance(simulator._weather_controller, _BenchmarkWeather):
    simulator._weather_controller = _BenchmarkWeather()
  simulator._weather_controller.sample_count = 0
  simulator._weather_controller.sample_seconds = 0.0
  simulator._current_timestamp = START


def _step(simulator, step, timestep):
  simulator._current_timestamp = START + pd.Timedelta(step * timestep, unit='s')
  start = time.perf_counter()
  converged = simulator.finite_differences_timestep(
      ambient_temperature=AMBIENT, convection_coefficient=CONVECTION
  )
  elapsed_ms = (time.perf_counter() - start) * 1000.0
  if not converged:
    raise RuntimeError(
        f'{type(simulator).__name__} did not converge on step {step + 1} '
        f'(exterior={simulator.include_exterior_radiative_heat_transfer}, '
        f'outer_iterations={simulator.last_outer_iteration_count}, '
        f'max_delta={simulator.last_outer_max_delta:.9g} K, '
        f'limit={simulator._convergence_threshold:g} K)'
    )
  return elapsed_ms


def _trajectory(simulator, template, args):
  _initialize(simulator, template)
  temperatures, mass_temperatures, timings, iterations = [], [], [], []
  for step in range(args.steps):
    timings.append(_step(simulator, step, args.time_step_seconds))
    iterations.append(simulator.last_outer_iteration_count)
    temperature, mass = grid._simulator_state(simulator)
    temperatures.append(temperature)
    if mass is not None:
      mass_temperatures.append(mass)
  weather = simulator._weather_controller
  expected_samples = (
      args.steps if template.include_exterior_radiative_heat_transfer else 0
  )
  if weather.sample_count != expected_samples:
    raise RuntimeError(
        f'Expected {expected_samples} weather samples, got'
        f' {weather.sample_count}'
    )
  return (
      np.asarray(temperatures),
      np.asarray(mass_temperatures) if mass_temperatures else None,
      np.asarray(timings),
      np.asarray(iterations),
      weather.sample_seconds * 1000.0 / args.steps,
  )


def _reference(args, template):
  simulator = jax_simulator.JaxSimulator(
      *solvers._simulator_args(
          copy.deepcopy(template),
          args.time_step_seconds,
          grid.REFERENCE_CONVERGENCE_THRESHOLD,
          args.iteration_limit,
      ),
      relative_convergence_threshold=None,
      solver_method='spsolve',
  )
  return _trajectory(simulator, template, args)[:2]


def _benchmark(args, rooms_per_side, application, plan):
  glazed = grid._create_building(
      rooms_per_side,
      ROOM_INTERIOR_CVS,
      CV_SIZE_CM,
      application,
      floor_plan=plan,
      include_exterior_radiation=True,
  )
  templates = {enabled: copy.deepcopy(glazed) for enabled in (False, True)}
  # Keep the glazing geometry and interior longwave coefficients identical.
  templates[False].include_exterior_radiative_heat_transfer = False
  references = {
      enabled: _reference(args, template)
      for enabled, template in templates.items()
  }
  active = glazed.len_neighbors >= 2
  indoor = plan == constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT
  physics_difference = references[True][0] - references[False][0]
  for solver in args.solvers:
    cases, warmups = {}, {}
    for enabled in args.exterior_states:
      template = templates[enabled]
      simulator = solvers._build_benchmark_solver(
          args, solver, copy.deepcopy(template)
      )
      _initialize(simulator, template)
      warmup_start = time.perf_counter()
      if solver in solvers.RESIDENT_SOLVERS:
        simulator.warmup(
            ambient_temperature=AMBIENT, convection_coefficient=CONVECTION
        )
      else:
        _step(simulator, 0, args.time_step_seconds)
      warmups[enabled] = (time.perf_counter() - warmup_start) * 1000.0
      cases[enabled] = simulator

    # Alternate ON/OFF order between repeats to reduce ordering bias.
    for repeat in range(args.repeats):
      order = args.exterior_states
      for enabled in order if repeat % 2 == 0 else reversed(order):
        simulator = cases[enabled]
        temperatures, masses, timings, iterations, weather_ms = _trajectory(
            simulator, templates[enabled], args
        )
        ref_temperature, ref_masses = references[enabled]
        error = np.abs(temperatures[:, active] - ref_temperature[:, active])
        mass_error = (
            np.abs(
                masses[:, glazed.interior_mass_mask]
                - ref_masses[:, glazed.interior_mass_mask]
            ).max()
            if masses is not None
            else np.nan
        )
        if not np.all(np.isfinite(error)) or (
            masses is not None and not np.isfinite(mass_error)
        ):
          raise RuntimeError(f'{solver} produced nonfinite temperatures')
        yield {
            'rooms_per_side': rooms_per_side,
            'rooms': rooms_per_side**2,
            'active_cvs': int(active.sum()),
            'glazing_cvs': int(
                np.count_nonzero(
                    plan == constants.FENESTRATION_VALUE_IN_FILE_INPUT
                )
            ),
            'application': application,
            'solver': solver,
            'exterior': enabled,
            'repeat': repeat + 1,
            'steps': args.steps,
            'time_step_seconds': args.time_step_seconds,
            'convergence_threshold_K': simulator._convergence_threshold,
            'cg_tolerance': getattr(simulator, 'cg_tolerance', np.nan),
            'cold_warmup_ms': warmups[enabled],
            'trajectory_ms': timings.sum(),
            'mean_step_ms': timings.mean(),
            'median_step_ms': np.median(timings),
            'mean_weather_ms': weather_ms,
            'mean_outer_iterations': iterations.mean(),
            'max_air_grid_error_K': error.max(),
            'max_mass_error_K': mass_error,
            'max_exterior_effect_grid_K': np.max(
                np.abs(physics_difference[:, active])
            ),
            'max_exterior_effect_indoor_K': np.max(
                np.abs(physics_difference[:, indoor])
            ),
            'final_mean_indoor_exterior_effect_K': physics_difference[
                -1, indoor
            ].mean(),
        }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--rooms-per-side', nargs='+', type=int, default=[1, 2])
  parser.add_argument('--steps', type=int, default=12)
  parser.add_argument('--time-step-seconds', type=float, default=300.0)
  parser.add_argument('--repeats', type=int, default=3)
  parser.add_argument(
      '--exterior', choices=('both', 'on', 'off'), default='both'
  )
  parser.add_argument('--solvers', nargs='+', choices=SOLVERS, default=SOLVERS)
  parser.add_argument(
      '--applications', nargs='+', choices=APPLICATIONS, default=APPLICATIONS
  )
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  args.exterior_states = {
      'both': (False, True),
      'off': (False,),
      'on': (True,),
  }[args.exterior]
  if (
      min(
          *args.rooms_per_side, args.steps, args.repeats, args.time_step_seconds
      )
      <= 0
  ):
    parser.error('Grid sizes, steps, repeats and timestep must be positive.')
  if jax.default_backend() != 'cpu' or tf.config.list_physical_devices('GPU'):
    parser.error('Select CPU for both TensorFlow and JAX before running.')
  args.convergence_threshold = 1e-4
  args.tensorflow_convergence_threshold = 1.5e-4
  args.resident_convergence_threshold = 1.5e-4
  args.cg_tolerance = 1e-10
  args.resident_cg_tolerance = 1e-8
  args.iteration_limit = 5000
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open('x', newline='', encoding='utf-8') as file:
    writer = None
    for rooms_per_side in args.rooms_per_side:
      plan = _glazed_floor_plan(rooms_per_side)
      for application in args.applications:
        for i, row in enumerate(
            _benchmark(args, rooms_per_side, application, plan)
        ):
          if writer is None:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
          writer.writerow(row)
          file.flush()
          if (i + 1) % (len(args.exterior_states) * args.repeats) == 0:
            print(
                f'{row["rooms"]} room(s) {application} {row["solver"]} done',
                flush=True,
            )


if __name__ == '__main__':
  main()
