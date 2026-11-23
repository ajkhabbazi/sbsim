"""One-day solver agreement run on grids of partitioned radiative enclosures.

partitioned_enclosure_grid_benchmark.py scores the final state after a few
steps under constant forcing. This script instead advances every solver
through a day with time-varying forcing and records how its prediction drifts
from a float64 sparse direct reference at every step:

  * the ambient temperature follows a daily sine;
  * each room has its own HVAC cooling schedule, with staggered on/off times
    and different powers, so rooms are excited at different times and the
    switching transients are part of the comparison.

Full fields for every step would be large, so each run saves:

  * per-step reduced series of its own state: the mean and percentiles of the
    air temperature over the active CVs, the mean air temperature of each
    room, and the means for the interior mass;
  * per-step error series against the reference: the maximum and RMS error
    over the field, and the maximum error in each room;
  * the error field every --field-save-stride steps, as float32 (the error is
    small, so float32 keeps its relative precision).

The reference file additionally holds its temperature fields at the same
stride, the forcing schedule, and the room labels.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import pickle
import time

import jax
import numpy as np
import pandas as pd
import tensorflow as tf

from reference.application_grid_time_sensitivity import APPLICATIONS
from reference.application_grid_time_sensitivity import JAX_SOLVERS
from reference.application_grid_time_sensitivity import SOLVERS
from reference.application_grid_time_sensitivity import _build_benchmark_solver
from reference.application_grid_time_sensitivity import _simulator_args
from reference.partitioned_enclosure_grid_benchmark import _apply_initial_state
from reference.partitioned_enclosure_grid_benchmark import _cache_path
from reference.partitioned_enclosure_grid_benchmark import _check_initial_state
from reference.partitioned_enclosure_grid_benchmark import (
    _create_partitioned_floor_plan,
)
from reference.partitioned_enclosure_grid_benchmark import (
    _includes_interior_mass,
)
from reference.partitioned_enclosure_grid_benchmark import (
    _initial_state_digest,
)
from reference.partitioned_enclosure_grid_benchmark import (
    _load_or_create_building,
)
from reference.partitioned_enclosure_grid_benchmark import _save_fields
from reference.partitioned_enclosure_grid_benchmark import _simulator_state
from reference.partitioned_enclosure_grid_benchmark import (
    REFERENCE_CONVERGENCE_THRESHOLD,
)
from reference.partitioned_enclosure_grid_benchmark import (
    REFERENCE_ITERATION_LIMIT,
)
from smart_control.simulator import jax_simulator


CONVECTION_COEFFICIENT = 12.0
# Daily ambient sine: minimum at 03:00, maximum at 15:00.
AMBIENT_MEAN_K = 298.0
AMBIENT_AMPLITUDE_K = 7.0
AMBIENT_PEAK_HOUR = 15.0
# Room k cools from HVAC_ON_HOUR + (k % 3) to HVAC_OFF_HOUR + (k % 3) with a
# power between HVAC_MIN_POWER_W and HVAC_MAX_POWER_W, split evenly over its
# diffusers.
HVAC_ON_HOUR = 7.0
HVAC_OFF_HOUR = 18.0
HVAC_MIN_POWER_W = -150.0
HVAC_MAX_POWER_W = -400.0
# Spread of the active air temperatures saved at every step.
PERCENTILES = (0.0, 5.0, 50.0, 95.0, 100.0)
# Changing the schedule invalidates the cached reference.
FORCING_VERSION = "daily_v1"


def _room_labels(rooms_per_side: int, room_interior_size: int) -> np.ndarray:
  """Labels each room's interior air CVs 0..rooms-1 row by row; others -1."""
  floor_plan = _create_partitioned_floor_plan(
      rooms_per_side, room_interior_size
  )
  tile_size = room_interior_size + 2
  rows, columns = np.indices(floor_plan.shape)
  labels = ((rows - 1) // tile_size) * rooms_per_side + (
      (columns - 1) // tile_size
  )
  return np.where(floor_plan == 0, labels, -1)


def _forcing_schedule(
    simulation_steps: int, time_step_seconds: float, room_count: int
) -> tuple[np.ndarray, np.ndarray]:
  """Returns the ambient temperature and the room powers of every step.

  Step i advances the state from time i * dt to (i + 1) * dt with the forcing
  evaluated at the start of the step.
  """
  hours = np.arange(simulation_steps) * time_step_seconds / 3600.0
  ambient = AMBIENT_MEAN_K + AMBIENT_AMPLITUDE_K * np.cos(
      2.0 * np.pi * (hours - AMBIENT_PEAK_HOUR) / 24.0
  )
  rooms = np.arange(room_count)
  peak_power = HVAC_MIN_POWER_W + (HVAC_MAX_POWER_W - HVAC_MIN_POWER_W) * (
      rooms / max(room_count - 1, 1)
  )
  offset = (rooms % 3).astype(float)
  hour_of_day = (hours % 24.0)[:, None]
  on = (hour_of_day >= HVAC_ON_HOUR + offset) & (
      hour_of_day < HVAC_OFF_HOUR + offset
  )
  return ambient, np.where(on, peak_power, 0.0)


def _apply_room_power(
    simulator, room_labels: np.ndarray, room_power: np.ndarray
) -> None:
  input_q = simulator.building.input_q
  input_q.fill(0.0)
  diffusers = np.asarray(simulator.building.diffusers) > 0.0
  for room, power in enumerate(room_power):
    cells = diffusers & (room_labels == room)
    if not cells.any():
      raise RuntimeError(f"room {room} has no diffuser")
    input_q[cells] = power / cells.sum()


def _room_means(field: np.ndarray, room_labels: np.ndarray) -> np.ndarray:
  room_count = room_labels.max() + 1
  return np.array(
      [field[room_labels == room].mean() for room in range(room_count)]
  )


def _room_maxima(field: np.ndarray, room_labels: np.ndarray) -> np.ndarray:
  room_count = room_labels.max() + 1
  return np.array(
      [field[room_labels == room].max() for room in range(room_count)]
  )


def _run_trajectory(
    simulator, schedule, room_labels
) -> tuple[list, list, list[float], list[float], list[float]]:
  """Advances the simulator through the schedule.

  Returns the air and interior mass temperatures including the initial state,
  and the wall time, outer and CG iteration counts of every step.
  """
  ambient, room_power = schedule
  temperature, mass_temperature = _simulator_state(simulator)
  temperatures = [temperature]
  mass_temperatures = [mass_temperature]
  step_seconds, outer_iterations, cg_iterations = [], [], []
  for step, step_ambient in enumerate(ambient):
    _apply_room_power(simulator, room_labels, room_power[step])
    start = time.perf_counter()
    converged = simulator.finite_differences_timestep(
        ambient_temperature=float(step_ambient),
        convection_coefficient=CONVECTION_COEFFICIENT,
    )
    step_seconds.append(time.perf_counter() - start)
    if not converged:
      raise RuntimeError(f"solve did not converge at step {step}")
    outer_iterations.append(
        getattr(simulator, "last_outer_iteration_count", np.nan)
    )
    cg_iterations.append(getattr(simulator, "last_cg_iteration_count", np.nan))
    temperature, mass_temperature = _simulator_state(simulator)
    temperatures.append(temperature)
    mass_temperatures.append(mass_temperature)
  return (
      temperatures,
      mass_temperatures,
      step_seconds,
      outer_iterations,
      cg_iterations,
  )


def _reference_trajectory(
    args, application: str, rooms_per_side: int, building_template, schedule,
    room_labels,
) -> dict:
  """Returns the float64 sparse direct trajectory, cached per grid."""
  key = {
      "application": application,
      "rooms_per_side": rooms_per_side,
      "room_interior_size": args.room_interior_size,
      "cv_size_cm": args.cv_size_cm,
      "simulation_steps": args.simulation_steps,
      "time_step_seconds": args.time_step_seconds,
      "threshold": REFERENCE_CONVERGENCE_THRESHOLD,
      "initial_state": _initial_state_digest(building_template),
      "forcing": FORCING_VERSION,
  }
  path = _cache_path(args.cache_dir, "timeseries_reference", key)
  if path is not None and path.exists():
    with path.open("rb") as file:
      return pickle.load(file)
  # On CUDA, JAX spsolve is a different (QR) factorization, so the reference
  # is only built on the host. Run a CPU solver first to populate the cache.
  if jax.default_backend() != "cpu":
    raise RuntimeError(
        "timeseries reference is not cached; run a CPU solver first"
    )
  simulator = jax_simulator.JaxSimulator(
      *_simulator_args(
          copy.deepcopy(building_template),
          args.time_step_seconds,
          REFERENCE_CONVERGENCE_THRESHOLD,
          REFERENCE_ITERATION_LIMIT,
      ),
      relative_convergence_threshold=None,
      solver_method="spsolve",
  )
  _apply_initial_state(simulator, building_template)
  temperatures, mass_temperatures, *_ = _run_trajectory(
      simulator, schedule, room_labels
  )
  reference = {
      "temperature": temperatures,
      "interior_mass_temperature": (
          mass_temperatures
          if _includes_interior_mass(building_template)
          else None
      ),
  }
  if path is not None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("wb") as file:
      pickle.dump(reference, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(path)
  return reference


def _state_series(
    temperatures, mass_temperatures, active, room_labels, mass_mask
) -> dict[str, np.ndarray]:
  """Per-step mean, spread, and per-room mean air and mass temperatures."""
  series = {
      "mean_temperature": np.array([t[active].mean() for t in temperatures]),
      "temperature_percentiles": np.array(
          [np.percentile(t[active], PERCENTILES) for t in temperatures]
      ),
      "room_mean_temperature": np.array(
          [_room_means(t, room_labels) for t in temperatures]
      ),
  }
  if mass_mask is not None:
    mass_labels = np.where(mass_mask, room_labels, -1)
    series["mean_mass_temperature"] = np.array(
        [m[mass_mask].mean() for m in mass_temperatures]
    )
    series["room_mean_mass_temperature"] = np.array(
        [_room_means(m, mass_labels) for m in mass_temperatures]
    )
  return series


def _error_series(
    temperatures, mass_temperatures, reference, active, room_labels, mass_mask
) -> dict[str, np.ndarray]:
  """Per-step field error norms against the reference."""
  errors = [
      np.where(active, np.abs(t - r), 0.0)
      for t, r in zip(temperatures, reference["temperature"])
  ]
  series = {
      "max_abs_error": np.array([e.max() for e in errors]),
      "rms_error": np.array(
          [np.sqrt(np.mean(np.square(e[active]))) for e in errors]
      ),
      "room_max_abs_error": np.array(
          [_room_maxima(e, room_labels) for e in errors]
      ),
  }
  if mass_mask is not None:
    mass_errors = [
        np.abs(m - r)[mass_mask]
        for m, r in zip(
            mass_temperatures, reference["interior_mass_temperature"]
        )
    ]
    series["max_abs_mass_error"] = np.array([e.max() for e in mass_errors])
    series["rms_mass_error"] = np.array(
        [np.sqrt(np.mean(np.square(e))) for e in mass_errors]
    )
  return series


def _output_path(
    output_dir: Path, application: str, rooms_per_side: int, name: str
) -> Path:
  return output_dir / application / f"rooms{rooms_per_side:02d}_{name}.npz"


def _run(args) -> pd.DataFrame:
  timeseries_dir = args.output_dir / "timeseries"
  rows = []
  for application in args.applications:
    for rooms_per_side in args.rooms_per_side:
      path = _output_path(
          timeseries_dir,
          application,
          rooms_per_side,
          f"{args.device}_{args.solver}",
      )
      if path.exists() and not args.force:
        print(f"exists, skipped: {path}", flush=True)
        continue
      print(
          f"starting device={args.device} solver={args.solver} "
          f"application={application} rooms_per_side={rooms_per_side}",
          flush=True,
      )
      building_template, _ = _load_or_create_building(
          args, application, rooms_per_side
      )
      active = np.array(
          [
              [len(neighbors) > 1 for neighbors in row]
              for row in building_template.neighbors
          ]
      )
      mass_mask = (
          np.asarray(building_template.interior_mass_mask, dtype=bool)
          if _includes_interior_mass(building_template)
          else None
      )
      room_labels = _room_labels(rooms_per_side, args.room_interior_size)
      schedule = _forcing_schedule(
          args.simulation_steps, args.time_step_seconds, rooms_per_side**2
      )
      time_seconds = np.arange(args.simulation_steps + 1) * args.time_step_seconds
      field_steps = np.arange(0, args.simulation_steps + 1, args.field_save_stride)

      start = time.perf_counter()
      reference = _reference_trajectory(
          args, application, rooms_per_side, building_template, schedule,
          room_labels,
      )
      print(
          f"reference ready ms={(time.perf_counter() - start) * 1000.0:.0f}",
          flush=True,
      )
      reference_path = _output_path(
          timeseries_dir, application, rooms_per_side, "reference"
      )
      if not reference_path.exists() or args.force:
        reference_masses = reference["interior_mass_temperature"]
        _save_fields(
            reference_path,
            time_seconds=time_seconds,
            ambient_temperature=schedule[0],
            room_power=schedule[1],
            room_labels=room_labels,
            active_mask=active,
            interior_mass_mask=mass_mask,
            field_steps=field_steps,
            temperature_fields=np.stack(
                [reference["temperature"][i] for i in field_steps]
            ),
            mass_temperature_fields=(
                None
                if reference_masses is None
                else np.stack([reference_masses[i] for i in field_steps])
            ),
            **_state_series(
                reference["temperature"],
                reference_masses,
                active,
                room_labels,
                mass_mask,
            ),
        )

      simulator = _build_benchmark_solver(
          args, args.solver, copy.deepcopy(building_template)
      )
      _apply_initial_state(simulator, building_template)
      _check_initial_state(simulator, building_template)
      if hasattr(simulator, "warmup"):
        _apply_room_power(simulator, room_labels, schedule[1][0])
        simulator.warmup(
            ambient_temperature=float(schedule[0][0]),
            convection_coefficient=CONVECTION_COEFFICIENT,
        )
      (
          temperatures,
          mass_temperatures,
          step_seconds,
          outer_iterations,
          cg_iterations,
      ) = _run_trajectory(simulator, schedule, room_labels)

      errors = _error_series(
          temperatures,
          mass_temperatures,
          reference,
          active,
          room_labels,
          mass_mask,
      )
      _save_fields(
          path,
          time_seconds=time_seconds,
          field_steps=field_steps,
          error_fields=np.stack(
              [
                  np.where(
                      active,
                      temperatures[i] - reference["temperature"][i],
                      0.0,
                  )
                  for i in field_steps
              ]
          ).astype(np.float32),
          step_seconds=np.array(step_seconds),
          outer_iterations=np.array(outer_iterations, dtype=float),
          cg_iterations=np.array(cg_iterations, dtype=float),
          **_state_series(
              temperatures, mass_temperatures, active, room_labels, mass_mask
          ),
          **errors,
      )
      row = {
          "device": args.device,
          "solver": args.solver,
          "application": application,
          "rooms_per_side": rooms_per_side,
          "simulation_steps": args.simulation_steps,
          "elapsed_s": float(np.sum(step_seconds)),
          "max_abs_error_K": float(errors["max_abs_error"].max()),
          "final_max_abs_error_K": float(errors["max_abs_error"][-1]),
      }
      rows.append(row)
      print(
          f"saved {path.name} elapsed_s={row['elapsed_s']:.1f} "
          f"max_abs_error_K={row['max_abs_error_K']:.3g} "
          f"final_max_abs_error_K={row['final_max_abs_error_K']:.3g}",
          flush=True,
      )
  return pd.DataFrame(rows)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--rooms-per-side", default="3")
  parser.add_argument("--room-size-meters", type=float, default=5.0)
  parser.add_argument("--cv-size-cm", type=float, default=20.0)
  parser.add_argument("--simulation-hours", type=float, default=24.0)
  parser.add_argument("--time-step-seconds", type=float, default=300.0)
  parser.add_argument(
      "--field-save-stride",
      type=int,
      default=6,
      help="Save the error field every this many steps.",
  )
  parser.add_argument("--solver", choices=SOLVERS, required=True)
  parser.add_argument("--applications", default=",".join(APPLICATIONS))
  parser.add_argument("--convergence-threshold", type=float, default=1e-4)
  parser.add_argument("--iteration-limit", type=int, default=5000)
  parser.add_argument("--cg-tolerance", type=float, default=1e-8)
  parser.add_argument(
      "--cache-dir",
      type=Path,
      help="Directory caching building templates and reference trajectories.",
  )
  parser.add_argument("--expected-device", choices=("cpu", "gpu"))
  parser.add_argument("--force", action="store_true")
  parser.add_argument("--output-dir", type=Path, required=True)
  args = parser.parse_args()
  args.rooms_per_side = tuple(
      int(value) for value in args.rooms_per_side.split(",")
  )
  args.applications = tuple(
      value.strip() for value in args.applications.split(",")
  )
  unknown_applications = set(args.applications) - set(APPLICATIONS)
  if unknown_applications:
    parser.error(f"unknown applications: {sorted(unknown_applications)}")
  args.simulation_steps = int(
      round(args.simulation_hours * 3600.0 / args.time_step_seconds)
  )
  if args.simulation_steps < 1:
    parser.error("simulation-hours must cover at least one step")
  if args.field_save_stride < 1:
    parser.error("field-save-stride must be positive")
  room_size_cm = args.room_size_meters * 100.0
  args.room_interior_size = int(round(room_size_cm / args.cv_size_cm))
  if not np.isclose(args.room_interior_size * args.cv_size_cm, room_size_cm):
    parser.error("room-size-meters must be an integer multiple of cv-size-cm")
  # One threshold and one CG tolerance for every solver.
  args.uniform_tolerances = True
  args.resident_cg_tolerance = args.cg_tolerance

  if args.solver in JAX_SOLVERS:
    framework = "JAX"
    args.device = "gpu" if jax.default_backend() == "gpu" else "cpu"
  else:
    framework = "TensorFlow"
    args.device = "gpu" if tf.config.list_physical_devices("GPU") else "cpu"
  if args.expected_device and args.device != args.expected_device:
    raise RuntimeError(
        f"expected {args.expected_device}, but {framework} detected "
        f"{args.device}"
    )
  args.output_dir.mkdir(parents=True, exist_ok=True)
  frame = _run(args)
  if not frame.empty:
    print(frame.to_string(index=False))


if __name__ == "__main__":
  main()
