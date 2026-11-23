"""Small timing comparison for the heat-balance solution paths.

Compares the iterative TensorFlow sweep against the SciPy and JAX sparse
direct solves on the small test fixture, and the two direct solves on the
partitioned-enclosure grids of `partitioned_enclosure_grid_benchmark`.

This is a reproducibility helper, not a performance regression test. Runtime
depends heavily on the machine and TensorFlow/SciPy/JAX builds. JAX `spsolve`
is native only on CUDA; on CPU it calls back into SciPy.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import math
from statistics import median
import time
from unittest import mock

import jax
import numpy as np
import pandas as pd

from reference.application_grid_time_sensitivity import APPLICATIONS
from reference.partitioned_enclosure_grid_benchmark import _apply_initial_state
from reference.partitioned_enclosure_grid_benchmark import _create_building
from reference.three_solver_sensitivity_benchmark import _simulator_args
from smart_control.simulator import jax_simulator
from smart_control.simulator import numpy_scipy_simulator
from smart_control.simulator import simulator_flexible_floor_plan_test
from smart_control.simulator import tf_simulator
from smart_control.simulator import weather_controller as weather_controller_py


DEFAULT_CONVERGENCE_THRESHOLD = 1e-4
MASS_REFERENCE_TOLERANCE = 5e-5
DIRECT_SOLVER_CLASSES = {
    "numpy_scipy": numpy_scipy_simulator.NumpyScipySimulator,
    "jax": jax_simulator.JaxSimulator,
}


@dataclass(frozen=True)
class PhysicsCase:
  include_interior_mass: bool
  include_radiation: bool
  air_reference_tolerance: float


def _create_simulator(case: PhysicsCase, solver: str):
  fixture = simulator_flexible_floor_plan_test.FlexibleFloorplanSimulatorTest()
  _, building = fixture._create_simulator_and_building(  # pylint: disable=protected-access
      initial_temp=292.0,
      include_interior_mass=case.include_interior_mass,
      include_radiative_heat_transfer=case.include_radiation,
      convergence_threshold=DEFAULT_CONVERGENCE_THRESHOLD,
      iteration_limit=2000,
  )
  building.conductivity[:] = 0.5

  source_coordinates = np.argwhere(building.diffusers > 0.0)
  source = tuple(source_coordinates[len(source_coordinates) // 2])
  simulator_class = DIRECT_SOLVER_CLASSES.get(solver, tf_simulator.TFSimulator)
  simulator = simulator_class(
      building,
      fixture._create_small_hvac(),  # pylint: disable=protected-access
      mock.create_autospec(weather_controller_py.WeatherController),
      30.0,
      DEFAULT_CONVERGENCE_THRESHOLD,
      2000,
      2000,
      pd.Timestamp("2012-12-21"),
      relative_convergence_threshold=None,
  )
  if case.include_interior_mass:
    simulator.building.interior_mass_temp[
        simulator.building.interior_mass_mask
    ] = np.linspace(
        286.0,
        298.0,
        int(np.sum(simulator.building.interior_mass_mask)),
    )
  return simulator, source


def _warm_up_jax(simulator) -> None:
  """Builds the static system and compiles the solve without moving state."""
  simulator.update_temperature_estimates(
      simulator.building.temp.copy(),
      ambient_temperature=305.0,
      convection_coefficient=12.0,
  )


def _forcing(step: int) -> tuple[float, float, float]:
  """Returns changing ambient temperature, film coefficient, and net heat."""
  phase = 2.0 * math.pi * step / 6.0
  ambient_temperature = 305.0 + 0.3 * math.sin(phase)
  convection_coefficient = 12.0 + 0.2 * math.sin(phase)
  thermal_source = (50.0, -30.0, 15.0)[step % 3]
  return ambient_temperature, convection_coefficient, thermal_source


def _set_forcing(simulator, source, step: int) -> tuple[float, float]:
  ambient_temperature, convection_coefficient, thermal_source = _forcing(step)
  simulator.building.input_q.fill(0.0)
  simulator.building.input_q[source] = thermal_source
  return ambient_temperature, convection_coefficient


def _iterate_to_reference(
    simulator,
    reference_temperature: np.ndarray,
    reference_mass_temperature: np.ndarray | None,
    ambient_temperature: float,
    convection_coefficient: float,
    air_reference_tolerance: float,
    max_sweeps: int,
) -> tuple[int, float, float | None]:
  temperature = simulator.building.temp.copy()
  previous_mass_temperature = None
  mass_temperature = None
  if reference_mass_temperature is not None:
    previous_mass_temperature = simulator.building.interior_mass_temp.copy()
    mass_temperature = previous_mass_temperature.copy()

  for sweep in range(1, max_sweeps + 1):
    temperature, _ = simulator.update_temperature_estimates(
        temperature,
        ambient_temperature=ambient_temperature,
        convection_coefficient=convection_coefficient,
    )
    if reference_mass_temperature is not None:
      simulator.building.interior_mass_temp = previous_mass_temperature
      mass_temperature, _ = simulator.update_interior_mass_temperatures(
          temperature
      )
      simulator.building.interior_mass_temp = mass_temperature

    air_difference = float(np.max(np.abs(temperature - reference_temperature)))
    mass_difference = None
    if reference_mass_temperature is not None:
      mass_difference = float(
          np.max(np.abs(mass_temperature - reference_mass_temperature))
      )
    if air_difference <= air_reference_tolerance and (
        mass_difference is None
        or mass_difference <= MASS_REFERENCE_TOLERANCE
    ):
      simulator.building.temp = temperature
      return sweep, air_difference, mass_difference

  raise RuntimeError(
      "iterative solver did not reach the direct solution within "
      f"{max_sweeps} sweeps: air_difference={air_difference}, "
      f"mass_difference={mass_difference}"
  )


def _run_pair(
    case: PhysicsCase, steps: int, max_sweeps: int
) -> tuple[float, float, float, float, float, float | None, float]:
  direct, direct_source = _create_simulator(case, "numpy_scipy")
  jax_direct, jax_source = _create_simulator(case, "jax")
  iterative, iterative_source = _create_simulator(case, "tf")
  _warm_up_jax(jax_direct)
  direct_elapsed = 0.0
  jax_elapsed = 0.0
  iterative_elapsed = 0.0
  maximum_jax_difference = 0.0
  total_sweeps = 0
  maximum_air_difference = 0.0
  maximum_mass_difference = None

  for step in range(steps):
    ambient_temperature, convection_coefficient = _set_forcing(
        direct, direct_source, step
    )
    _set_forcing(jax_direct, jax_source, step)
    _set_forcing(iterative, iterative_source, step)

    start = time.perf_counter()
    direct_converged = direct.finite_differences_timestep(
        ambient_temperature=ambient_temperature,
        convection_coefficient=convection_coefficient,
    )
    direct_elapsed += time.perf_counter() - start
    if not direct_converged:
      raise RuntimeError("direct solver did not converge")

    start = time.perf_counter()
    jax_converged = jax_direct.finite_differences_timestep(
        ambient_temperature=ambient_temperature,
        convection_coefficient=convection_coefficient,
    )
    jax_elapsed += time.perf_counter() - start
    if not jax_converged:
      raise RuntimeError("JAX direct solver did not converge")

    reference_temperature = direct.building.temp.copy()
    maximum_jax_difference = max(
        maximum_jax_difference,
        float(np.max(np.abs(jax_direct.building.temp - reference_temperature))),
    )
    # Both direct solvers continue from the SciPy state so that each step
    # compares one solve rather than accumulated drift.
    jax_direct.building.temp = reference_temperature.copy()
    reference_mass_temperature = None
    if case.include_interior_mass:
      reference_mass_temperature = direct.building.interior_mass_temp.copy()
      maximum_jax_difference = max(
          maximum_jax_difference,
          float(
              np.max(
                  np.abs(
                      jax_direct.building.interior_mass_temp
                      - reference_mass_temperature
                  )
              )
          ),
      )
      jax_direct.building.interior_mass_temp = reference_mass_temperature.copy()

    start = time.perf_counter()
    sweeps, air_difference, mass_difference = _iterate_to_reference(
        iterative,
        reference_temperature,
        reference_mass_temperature,
        ambient_temperature,
        convection_coefficient,
        case.air_reference_tolerance,
        max_sweeps,
    )
    iterative_elapsed += time.perf_counter() - start
    total_sweeps += sweeps
    maximum_air_difference = max(maximum_air_difference, air_difference)
    if mass_difference is not None:
      maximum_mass_difference = max(
          maximum_mass_difference or 0.0, mass_difference
      )

    iterative.building.temp = reference_temperature.copy()
    if reference_mass_temperature is not None:
      iterative.building.interior_mass_temp = reference_mass_temperature.copy()

  return (
      iterative_elapsed * 1000.0 / steps,
      direct_elapsed * 1000.0 / steps,
      jax_elapsed * 1000.0 / steps,
      total_sweeps / steps,
      maximum_air_difference,
      maximum_mass_difference,
      maximum_jax_difference,
  )


def _benchmark_case(
    case: PhysicsCase, steps: int, repeats: int, max_sweeps: int
) -> tuple:
  iterative_times = []
  direct_times = []
  jax_times = []
  sweeps_per_step = []
  air_differences = []
  mass_differences = []
  jax_differences = []
  for _ in range(repeats):
    (
        iterative_time,
        direct_time,
        jax_time,
        sweeps,
        air_difference,
        mass_difference,
        jax_difference,
    ) = _run_pair(case, steps, max_sweeps)
    iterative_times.append(iterative_time)
    direct_times.append(direct_time)
    jax_times.append(jax_time)
    jax_differences.append(jax_difference)
    sweeps_per_step.append(sweeps)
    air_differences.append(air_difference)
    if mass_difference is not None:
      mass_differences.append(mass_difference)

  return (
      median(iterative_times),
      median(direct_times),
      median(jax_times),
      median(sweeps_per_step),
      max(air_differences),
      max(mass_differences) if mass_differences else None,
      max(jax_differences),
  )


def _run_partitioned_case(
    application: str,
    rooms_per_side: int,
    args: argparse.Namespace,
) -> dict[str, float | int]:
  """Times the SciPy and JAX direct solves on one partitioned grid."""
  room_interior_size = int(round(args.room_size_meters * 100.0 / args.cv_size_cm))
  template = _create_building(
      rooms_per_side, room_interior_size, args.cv_size_cm, application
  )
  simulators = {}
  build_ms = {}
  for solver, simulator_class in DIRECT_SOLVER_CLASSES.items():
    building = copy.deepcopy(template)
    start = time.perf_counter()
    simulator = simulator_class(
        *_simulator_args(
            building,
            args.partitioned_time_step_seconds,
            DEFAULT_CONVERGENCE_THRESHOLD,
            args.partitioned_iteration_limit,
        ),
        relative_convergence_threshold=None,
    )
    build_ms[solver] = (time.perf_counter() - start) * 1000.0
    # Construction resets the interior mass to the building's initial
    # temperature; restore the template's state.
    _apply_initial_state(simulator, template)
    source = tuple(np.argwhere(simulator.building.diffusers > 0.0)[0])
    simulator.building.input_q[source] = 50.0
    simulators[solver] = simulator

  start = time.perf_counter()
  _warm_up_jax(simulators["jax"])
  warmup_ms = (time.perf_counter() - start) * 1000.0

  elapsed_ms = {solver: [] for solver in simulators}
  outer_iterations = {solver: 0 for solver in simulators}
  maximum_difference = 0.0
  for step in range(args.partitioned_steps):
    for solver, simulator in simulators.items():
      start = time.perf_counter()
      converged = simulator.finite_differences_timestep(
          ambient_temperature=305.0, convection_coefficient=12.0
      )
      elapsed_ms[solver].append((time.perf_counter() - start) * 1000.0)
      if not converged:
        raise RuntimeError(
            f"{solver} did not converge: application={application}, "
            f"rooms_per_side={rooms_per_side}, step={step}"
        )
      outer_iterations[solver] += simulator.last_outer_iteration_count
    maximum_difference = max(
        maximum_difference,
        float(
            np.max(
                np.abs(
                    simulators["jax"].building.temp
                    - simulators["numpy_scipy"].building.temp
                )
            )
        ),
    )

  active_cvs = sum(
      len(neighbors) > 1
      for row in template.neighbors
      for neighbors in row
  )
  return {
      "grid": template.temp.shape[0],
      "active_cvs": active_cvs,
      "scipy_build_ms": build_ms["numpy_scipy"],
      "jax_build_ms": build_ms["jax"],
      "jax_warmup_ms": warmup_ms,
      "scipy_step_ms": float(np.median(elapsed_ms["numpy_scipy"])),
      "jax_step_ms": float(np.median(elapsed_ms["jax"])),
      "scipy_outer": outer_iterations["numpy_scipy"] / args.partitioned_steps,
      "jax_outer": outer_iterations["jax"] / args.partitioned_steps,
      "jax_diff_K": maximum_difference,
  }


def _print_partitioned(args: argparse.Namespace) -> None:
  print(
      f"\npartitioned enclosure grid: jax_backend={jax.default_backend()} "
      f"room_size_m={args.room_size_meters:g} cv_size_cm={args.cv_size_cm:g} "
      f"time_step_sec={args.partitioned_time_step_seconds:g} "
      f"steps={args.partitioned_steps}"
  )
  print(
      "application rooms grid active_cvs scipy_step_ms jax_step_ms speedup "
      "jax_warmup_ms outer_scipy outer_jax jax_diff_K"
  )
  for application in args.partitioned_applications:
    for rooms_per_side in args.partitioned_rooms_per_side:
      row = _run_partitioned_case(application, rooms_per_side, args)
      print(
          f"{application:22s} {rooms_per_side**2:5d} {row['grid']:4d} "
          f"{row['active_cvs']:10d} {row['scipy_step_ms']:13.3f} "
          f"{row['jax_step_ms']:11.3f} "
          f"{row['scipy_step_ms'] / row['jax_step_ms']:7.2f} "
          f"{row['jax_warmup_ms']:13.3f} {row['scipy_outer']:11.1f} "
          f"{row['jax_outer']:9.1f} {row['jax_diff_K']:10.3g}",
          flush=True,
      )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=1)
  parser.add_argument("--repeats", type=int, default=1)
  parser.add_argument("--max-sweeps", type=int, default=20000)
  parser.add_argument(
      "--skip-fixture",
      action="store_true",
      help="Skip the small-fixture iterative/direct comparison.",
  )
  parser.add_argument(
      "--partitioned-rooms-per-side",
      default="1,2,4,8",
      help="Comma-separated room counts per side; empty to skip.",
  )
  parser.add_argument(
      "--partitioned-applications", default="base,base_rad"
  )
  parser.add_argument("--partitioned-steps", type=int, default=3)
  parser.add_argument(
      "--partitioned-time-step-seconds", type=float, default=300.0
  )
  parser.add_argument("--partitioned-iteration-limit", type=int, default=5000)
  parser.add_argument("--room-size-meters", type=float, default=5.0)
  parser.add_argument("--cv-size-cm", type=float, default=20.0)
  args = parser.parse_args()
  if args.steps < 1 or args.repeats < 1 or args.max_sweeps < 1:
    parser.error("--steps, --repeats, and --max-sweeps must be positive")
  if args.partitioned_steps < 1:
    parser.error("--partitioned-steps must be positive")
  args.partitioned_rooms_per_side = tuple(
      int(value) for value in args.partitioned_rooms_per_side.split(",") if value
  )
  args.partitioned_applications = tuple(
      value.strip() for value in args.partitioned_applications.split(",")
  )
  unknown_applications = set(args.partitioned_applications) - set(APPLICATIONS)
  if unknown_applications:
    parser.error(f"unknown applications: {sorted(unknown_applications)}")
  if not np.isclose(
      round(args.room_size_meters * 100.0 / args.cv_size_cm) * args.cv_size_cm,
      args.room_size_meters * 100.0,
  ):
    parser.error("--room-size-meters must be an integer multiple of --cv-size-cm")

  if not args.skip_fixture:
    _print_fixture(args)
  if args.partitioned_rooms_per_side:
    _print_partitioned(args)


def _print_fixture(args: argparse.Namespace) -> None:

  print(
      "time_step_sec=30 "
      f"convergence_threshold={DEFAULT_CONVERGENCE_THRESHOLD:g} "
      f"mass_reference_tolerance={MASS_REFERENCE_TOLERANCE:g}"
  )
  print(
      "mass radiation iterative_ms direct_ms jax_ms speedup jax_speedup "
      "sweeps air_diff_K mass_diff_K jax_diff_K"
  )
  for case in (
      PhysicsCase(False, False, 5e-4),
      PhysicsCase(True, False, 1e-3),
      PhysicsCase(False, True, 5e-4),
      PhysicsCase(True, True, 1e-3),
  ):
    (
        iterative_ms,
        direct_ms,
        jax_ms,
        sweeps,
        air_difference,
        mass_difference,
        jax_difference,
    ) = _benchmark_case(case, args.steps, args.repeats, args.max_sweeps)
    mass_difference_text = (
        "-" if mass_difference is None else f"{mass_difference:.6g}"
    )
    print(
        f"{str(case.include_interior_mass):5s} "
        f"{str(case.include_radiation):9s} "
        f"{iterative_ms:12.3f} {direct_ms:9.3f} {jax_ms:6.3f} "
        f"{iterative_ms / direct_ms:7.2f} {iterative_ms / jax_ms:11.2f} "
        f"{sweeps:6.1f} {air_difference:10.6g} {mass_difference_text:>11s} "
        f"{jax_difference:10.3g}"
    )


if __name__ == "__main__":
  main()
