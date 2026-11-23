"""Scaling benchmark for grids made of partitioned radiative enclosures."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import pickle
import time
import warnings

import jax
import numpy as np
import pandas as pd
import tensorflow as tf

from reference.application_grid_time_sensitivity import APPLICATIONS as BASE_APPLICATIONS
from reference.application_grid_time_sensitivity import JAX_SOLVERS as BASE_JAX_SOLVERS
from reference.application_grid_time_sensitivity import RESIDENT_SOLVERS
from reference.application_grid_time_sensitivity import SOLVERS as BASE_SOLVERS
from reference.application_grid_time_sensitivity import _simulator_args
from reference.application_grid_time_sensitivity import _build_benchmark_solver
from reference.application_grid_time_sensitivity import _solver_cg_tolerance
from reference.application_grid_time_sensitivity import (
    _solver_convergence_threshold,
)
from smart_control.simulator import building as building_py
from smart_control.simulator import constants
from smart_control.simulator import grouped_jax_simulator
from smart_control.simulator import jax_simulator
from smart_control.simulator import weather_controller


DEFAULT_APPLICATIONS = ("base", "base_rad")
GROUPED_JAX_SOLVER = "grouped_jax"
JAX_SOLVERS = (*BASE_JAX_SOLVERS, GROUPED_JAX_SOLVER)
SOLVERS = (*BASE_SOLVERS, GROUPED_JAX_SOLVER)
EXTERIOR_APPLICATIONS = {
    "base_rad_exterior": (False, True),
    "base_rad_interior_mass_exterior": (True, True),
}
APPLICATIONS = {**BASE_APPLICATIONS, **EXTERIOR_APPLICATIONS}
EXTERIOR_START = pd.Timestamp("2025-06-21 11:00", tz="America/Los_Angeles")
RESULT_KEY_COLUMNS = (
    "device",
    "application",
    "rooms_per_side",
    "room_size_meters",
    "cv_size_cm",
    "solver",
    "repeat",
    "simulation_steps",
    "time_step_seconds",
    "convergence_threshold",
    "iteration_limit",
    "cg_tolerance",
)
# The reference solution every sample is scored against: a float64 sparse
# direct solve with a Picard tolerance far below any benchmarked one.
REFERENCE_CONVERGENCE_THRESHOLD = 1e-9
REFERENCE_ITERATION_LIMIT = 200
SAMPLE_ID_COLUMNS = tuple(
    column
    for column in RESULT_KEY_COLUMNS
    if column not in ("convergence_threshold", "iteration_limit")
)


def _create_partitioned_floor_plan(
    rooms_per_side: int, room_interior_size: int
) -> np.ndarray:
  """Creates disconnected square rooms separated by two wall CVs."""
  tile_size = room_interior_size + 2
  grid_size = rooms_per_side * tile_size + 2
  floor_plan = np.full((grid_size, grid_size), 2, dtype=int)
  for room_row in range(rooms_per_side):
    for room_column in range(rooms_per_side):
      row = 1 + room_row * tile_size
      column = 1 + room_column * tile_size
      floor_plan[row : row + tile_size, column : column + tile_size] = 1
      floor_plan[
          row + 1 : row + 1 + room_interior_size,
          column + 1 : column + 1 + room_interior_size,
      ] = 0
  return floor_plan


def _add_perimeter_glazing(floor_plan, rooms_per_side, room_interior_size):
  """Glazes 20% of each room facade that faces outdoors."""
  tile_size = room_interior_size + 2
  window_width = max(1, round(0.2 * room_interior_size))
  glass = constants.FENESTRATION_VALUE_IN_FILE_INPUT
  for room in range(rooms_per_side):
    start = 2 + room * tile_size + (room_interior_size - window_width) // 2
    opening = slice(start, start + window_width)
    floor_plan[1, opening] = floor_plan[-2, opening] = glass
    floor_plan[opening, 1] = floor_plan[opening, -2] = glass


def _create_building(
    rooms_per_side: int,
    room_interior_size: int,
    cv_size_cm: float,
    application: str,
    *,
    floor_plan: np.ndarray | None = None,
    include_exterior_radiation: bool = False,
):
  include_mass, include_radiation = APPLICATIONS[application]
  if floor_plan is None:
    floor_plan = _create_partitioned_floor_plan(
        rooms_per_side, room_interior_size
    )
  if application in EXTERIOR_APPLICATIONS:
    floor_plan = floor_plan.copy()
    _add_perimeter_glazing(floor_plan, rooms_per_side, room_interior_size)
    include_exterior_radiation = True
  air = building_py.MaterialProperties(0.5, 1005.0, 1.2)
  wall = building_py.MaterialProperties(0.5, 800.0, 1800.0)
  kwargs = {}
  if include_radiation:
    kwargs.update(
        inside_air_radiative_properties=building_py.RadiationProperties(
            alpha=0.0, epsilon=0.0, tau=1.0, rho=None
        ),
        inside_wall_radiative_properties=building_py.RadiationProperties(
            alpha=0.4, epsilon=0.6, tau=0.0, rho=None
        ),
        building_exterior_radiative_properties=building_py.RadiationProperties(
            alpha=0.65, epsilon=0.35, tau=0.0, rho=None
        ),
        include_radiative_heat_transfer=True,
        view_factor_method="AreaRatio",
        partition_radiative_enclosures=True,
    )
  if include_mass:
    kwargs.update(
        interior_mass_properties=building_py.MaterialProperties(
            5.0, 300.0, 2000.0
        ),
        include_interior_mass=True,
    )
  with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Connected components is showing that there are 4 or fewer.*",
        category=UserWarning,
    )
    building = building_py.FloorPlanBasedBuilding(
        cv_size_cm=cv_size_cm,
        floor_height_cm=300.0,
        initial_temp=292.0,
        inside_air_properties=air,
        inside_wall_properties=wall,
        building_exterior_properties=wall,
        floor_plan=floor_plan,
        zone_map=floor_plan.copy(),
        buffer_from_walls=0,
        include_exterior_radiative_heat_transfer=include_exterior_radiation,
        **kwargs,
    )
  if include_mass:
    building.interior_mass_temp[building.interior_mass_mask] = 291.0
  return building


def _cache_path(cache_dir: Path | None, kind: str, key: dict) -> Path | None:
  if cache_dir is None:
    return None
  digest = hashlib.sha1(repr(sorted(key.items())).encode()).hexdigest()[:16]
  return cache_dir / f"{kind}_{key['application']}_{key['rooms_per_side']}_{digest}.pkl"


def _load_or_create_building(args, application: str, rooms_per_side: int):
  """Returns the building template and the time spent constructing it.

  The view factor setup dominates large radiative grids and is identical for
  every solver and device, so with --cache-dir it is paid once and later runs
  load the pickled template. The cached construction time is reported so
  building_setup_ms stays comparable.
  """
  key = {
      "application": application,
      "rooms_per_side": rooms_per_side,
      "room_interior_size": args.room_interior_size,
      "cv_size_cm": args.cv_size_cm,
  }
  path = _cache_path(args.cache_dir, "building", key)
  if path is not None and path.exists():
    with path.open("rb") as file:
      cached = pickle.load(file)
    return cached["building"], cached["building_setup_ms"]
  start = time.perf_counter()
  building = _create_building(
      rooms_per_side, args.room_interior_size, args.cv_size_cm, application
  )
  building_setup_ms = (time.perf_counter() - start) * 1000.0
  if path is not None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("wb") as file:
      pickle.dump(
          {"building": building, "building_setup_ms": building_setup_ms},
          file,
          protocol=pickle.HIGHEST_PROTOCOL,
      )
    temporary_path.replace(path)
  return building, building_setup_ms


def _set_forcing(simulator) -> None:
  diffuser_coordinates = np.argwhere(simulator.building.diffusers > 0.0)
  if not len(diffuser_coordinates):
    raise RuntimeError("partitioned benchmark building has no diffuser")
  simulator.building.input_q[tuple(diffuser_coordinates[0])] = 50.0


def _includes_interior_mass(building) -> bool:
  return bool(getattr(building, "include_interior_mass", False))


def _initial_state_digest(building_template) -> str:
  """Identifies the initial air and interior mass temperatures."""
  digest = hashlib.sha1(
      np.ascontiguousarray(building_template.temp, dtype=np.float64).tobytes()
  )
  if _includes_interior_mass(building_template):
    digest.update(
        np.ascontiguousarray(
            building_template.interior_mass_temp, dtype=np.float64
        ).tobytes()
    )
  return digest.hexdigest()[:16]


def _apply_initial_state(simulator, building_template) -> None:
  """Starts the simulator from the template's air and mass temperatures.

  Simulator construction calls building.reset(), which refills temp and
  interior_mass_temp with the building's initial temperature and so drops the
  template's interior mass temperature. The reference and every solver sample
  restore the template state here, so they all start from the same state.
  """
  building = simulator.building
  building.temp = building_template.temp.copy()
  building.input_q.fill(0.0)
  if _includes_interior_mass(building):
    building.interior_mass_temp = building_template.interior_mass_temp.copy()
  if hasattr(simulator, "sync_state_from_building"):
    simulator.sync_state_from_building()
  if getattr(building, "include_exterior_radiative_heat_transfer", False):
    simulator._current_timestamp = EXTERIOR_START
    if not isinstance(
        simulator._weather_controller, weather_controller.WeatherController
    ) or not isinstance(
        getattr(simulator._weather_controller, "default_low_temp", None),
        (int, float),
    ):
      simulator._weather_controller = weather_controller.WeatherController(
          305.0, 305.0, convection_coefficient=12.0
      )


def _advance_timestep(simulator, step, time_step_seconds):
  if simulator.include_exterior_radiative_heat_transfer:
    simulator._current_timestamp = EXTERIOR_START + pd.Timedelta(
        step * time_step_seconds, unit="s"
    )
  return simulator.finite_differences_timestep(
      ambient_temperature=305.0, convection_coefficient=12.0
  )


def _prepare_simulator(args, solver: str, building_template, reusable=None):
  """Returns a simulator at the template's initial state and its build time.

  A resident simulator from an earlier repeat is reused, since its device
  constants depend only on the building.
  """
  if reusable is not None:
    simulator = reusable
    solver_build_ms = 0.0
  else:
    building = copy.deepcopy(building_template)
    start = time.perf_counter()
    if solver == GROUPED_JAX_SOLVER:
      simulator = grouped_jax_simulator.GroupedJaxSimulator(
          *_simulator_args(
              building,
              args.time_step_seconds,
              _solver_convergence_threshold(args, solver),
              args.iteration_limit,
          ),
          relative_convergence_threshold=None,
          grouped=True,
          linearized_radiative_heat_transfer=True,
          platform=args.device,
          solver_method="cg",
          cg_tolerance=_solver_cg_tolerance(args, solver),
          cg_iteration_limit=args.iteration_limit,
      )
    else:
      simulator = _build_benchmark_solver(args, solver, building)
    solver_build_ms = (time.perf_counter() - start) * 1000.0
  _apply_initial_state(simulator, building_template)
  _check_initial_state(simulator, building_template)
  _set_forcing(simulator)
  return simulator, solver_build_ms


def _check_initial_state(simulator, building_template) -> None:
  """Fails if the simulator does not start from the template state."""
  temperature, mass_temperature = _simulator_state(simulator)
  # float32 solvers round the state on the device.
  np.testing.assert_allclose(
      temperature, building_template.temp, rtol=0.0, atol=1e-4
  )
  if mass_temperature is not None:
    np.testing.assert_allclose(
        mass_temperature,
        building_template.interior_mass_temp,
        rtol=0.0,
        atol=1e-4,
    )


def _simulator_state(simulator) -> tuple[np.ndarray, np.ndarray | None]:
  """Returns float64 copies of the air and interior mass temperatures."""
  if hasattr(simulator, "sync_state_to_building"):
    simulator.sync_state_to_building()
  building = simulator.building
  mass_temperature = None
  if _includes_interior_mass(building):
    mass_temperature = np.array(building.interior_mass_temp, dtype=np.float64)
  return np.array(building.temp, dtype=np.float64), mass_temperature


def _reference_solution(
    args, application: str, rooms_per_side: int, building_template
) -> dict[str, list[np.ndarray] | None]:
  """Returns the reference air and mass temperatures after each step.

  Uses the JAX sparse direct solver in float64; it assembles exactly the
  system NumpyScipySimulator does, but is several times faster. The result
  depends only on the building, its initial state, and the forcing, so it is
  cached per grid.
  """
  key = {
      "application": application,
      "rooms_per_side": rooms_per_side,
      "room_interior_size": args.room_interior_size,
      "cv_size_cm": args.cv_size_cm,
      "simulation_steps": args.simulation_steps,
      "time_step_seconds": args.time_step_seconds,
      "threshold": REFERENCE_CONVERGENCE_THRESHOLD,
      "initial_state": _initial_state_digest(building_template),
  }
  path = _cache_path(args.cache_dir, "reference", key)
  if path is not None and path.exists():
    with path.open("rb") as file:
      return pickle.load(file)
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
  _set_forcing(simulator)
  temperatures = []
  mass_temperatures = []
  for step in range(args.simulation_steps):
    if not _advance_timestep(simulator, step, args.time_step_seconds):
      raise RuntimeError(
          "reference solve did not converge: "
          f"application={application}, rooms_per_side={rooms_per_side}"
      )
    temperature, mass_temperature = _simulator_state(simulator)
    temperatures.append(temperature)
    mass_temperatures.append(mass_temperature)
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


def _accuracy_metrics(
    temperature: np.ndarray,
    reference_temperature: np.ndarray,
    active: np.ndarray,
    mass_temperature: np.ndarray | None = None,
    reference_mass_temperature: np.ndarray | None = None,
    mass_mask: np.ndarray | None = None,
) -> dict[str, float]:
  """Scores air and interior mass temperatures against the reference."""
  error = np.where(
      active,
      np.abs(np.asarray(temperature, dtype=np.float64) - reference_temperature),
      0.0,
  )
  row, column = np.unravel_index(np.argmax(error), error.shape)
  metrics = {
      "max_abs_error_K": float(error[row, column]),
      "rms_error_K": float(np.sqrt(np.mean(np.square(error[active])))),
      "max_error_row": int(row),
      "max_error_column": int(column),
      "temperature_at_max_error_K": float(temperature[row, column]),
      "reference_at_max_error_K": float(reference_temperature[row, column]),
      "max_abs_mass_error_K": np.nan,
      "rms_mass_error_K": np.nan,
  }
  if mass_temperature is not None and reference_mass_temperature is not None:
    mass_error = np.abs(mass_temperature - reference_mass_temperature)[
        mass_mask
    ]
    metrics["max_abs_mass_error_K"] = float(np.max(mass_error))
    metrics["rms_mass_error_K"] = float(np.sqrt(np.mean(np.square(mass_error))))
  return metrics


def _prediction_path(
    predictions_dir: Path, application: str, rooms_per_side: int, name: str
) -> Path:
  return predictions_dir / application / f"rooms{rooms_per_side:02d}_{name}.npz"


def _save_fields(path: Path, **fields) -> None:
  """Writes compressed float64 fields, skipping the ones that are None."""
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary_path = path.with_name(path.stem + ".tmp.npz")
  np.savez_compressed(
      temporary_path,
      **{name: value for name, value in fields.items() if value is not None},
  )
  temporary_path.replace(path)


def _save_reference_fields(
    predictions_dir: Path,
    application: str,
    rooms_per_side: int,
    reference: dict,
    active: np.ndarray,
    mass_mask: np.ndarray | None,
) -> None:
  path = _prediction_path(predictions_dir, application, rooms_per_side, "reference")
  if path.exists():
    return
  mass_temperatures = reference["interior_mass_temperature"]
  _save_fields(
      path,
      temperature=reference["temperature"][-1],
      interior_mass_temperature=(
          None if mass_temperatures is None else mass_temperatures[-1]
      ),
      active_mask=active,
      interior_mass_mask=mass_mask,
  )


def _save_results(frame: pd.DataFrame, output_csv: Path) -> None:
  temporary_path = output_csv.with_suffix(".tmp")
  frame.to_csv(temporary_path, index=False)
  temporary_path.replace(output_csv)


def _write_progress(path: Path | None, rooms_per_side: int, repeat: int) -> None:
  if path is None:
    return
  temporary_path = path.with_suffix(".tmp")
  temporary_path.write_text(
      f"{rooms_per_side},{repeat}\n", encoding="utf-8"
  )
  temporary_path.replace(path)


def _radiation_metrics(building) -> dict[str, float | int]:
  enclosure_indices = getattr(building, "radiative_enclosure_indices", ())
  view_factor_blocks = getattr(building, "interior_wall_vf_by_enclosure", ())
  ifa_blocks = getattr(building, "ifa_inv_by_enclosure", ())
  block_sizes = [len(indices) for indices in enclosure_indices]
  matrix_entries = sum(block.size for block in view_factor_blocks) + sum(
      block.size for block in ifa_blocks
  )
  matrix_bytes = sum(block.nbytes for block in view_factor_blocks) + sum(
      block.nbytes for block in ifa_blocks
  )
  return {
      "radiative_enclosures": len(enclosure_indices),
      "radiating_nodes": sum(block_sizes),
      "max_enclosure_nodes": max(block_sizes, default=0),
      "radiation_matrix_entries": matrix_entries,
      "radiation_matrix_mib": matrix_bytes / (1024.0 * 1024.0),
  }


def _load_existing(args, output_csv: Path) -> tuple[pd.DataFrame, set[tuple]]:
  if output_csv.exists() and not args.force:
    frame = pd.read_csv(output_csv)
  else:
    frame = pd.DataFrame()
  if frame.empty:
    return frame, set()
  required_configuration = {
      "room_size_meters": args.room_size_meters,
      "cv_size_cm": args.cv_size_cm,
      "simulation_steps": args.simulation_steps,
      "time_step_seconds": args.time_step_seconds,
  }
  if not set(required_configuration).issubset(frame.columns):
    return pd.DataFrame(), set()
  configuration_mask = np.ones(len(frame), dtype=bool)
  for column, value in required_configuration.items():
    configuration_mask &= np.isclose(frame[column], value)
  configuration_mask &= frame["application"].isin(args.applications)
  frame = frame[configuration_mask].copy()
  if frame.empty:
    return frame, set()

  # A sample saved under a lower iteration limit already passed convergence,
  # so it remains valid when the limit is increased. Prefer that lower-limit
  # result when both the old and new runs saved the same sample.
  frame = frame.sort_values(
      ["convergence_threshold", "iteration_limit"]
  ).drop_duplicates(list(SAMPLE_ID_COLUMNS), keep="first")
  target_thresholds = frame["solver"].map(
      lambda solver: _solver_convergence_threshold(args, solver)
  )
  reusable = frame[
      (frame["convergence_threshold"] <= target_thresholds)
      & (frame["iteration_limit"] <= args.iteration_limit)
  ]
  completed = set()
  for row in reusable.loc[:, list(RESULT_KEY_COLUMNS)].itertuples(
      index=False, name=None
  ):
    values = list(row)
    solver = values[RESULT_KEY_COLUMNS.index("solver")]
    values[RESULT_KEY_COLUMNS.index("convergence_threshold")] = (
        _solver_convergence_threshold(args, solver)
    )
    values[RESULT_KEY_COLUMNS.index("iteration_limit")] = args.iteration_limit
    completed.add(tuple(values))
  print(
      f"resume: loaded_rows={len(frame)} reusable_samples={len(completed)} "
      f"target_iteration_limit={args.iteration_limit}",
      flush=True,
  )
  return frame, completed


def _run(args, output_csv: Path) -> pd.DataFrame:
  if args.progress_file is not None:
    args.progress_file.unlink(missing_ok=True)
  frame, completed = _load_existing(args, output_csv)
  rows = frame.to_dict("records")
  predictions_dir = args.predictions_dir or args.output_dir / "predictions"

  for application in args.applications:
    for rooms_per_side in args.rooms_per_side:
      pending = [
          repeat
          for repeat in range(args.repeats)
          if (
              args.device,
              application,
              rooms_per_side,
              args.room_size_meters,
              args.cv_size_cm,
              args.solver,
              repeat,
              args.simulation_steps,
              args.time_step_seconds,
              _solver_convergence_threshold(args, args.solver),
              args.iteration_limit,
              _solver_cg_tolerance(args, args.solver),
          )
          not in completed
      ]
      if not pending:
        continue

      _write_progress(args.progress_file, rooms_per_side, -1)
      print(
          f"starting device={args.device} solver={args.solver} "
          f"application={application} rooms_per_side={rooms_per_side} "
          f"room_count={rooms_per_side**2}",
          flush=True,
      )
      building_template, building_setup_ms = _load_or_create_building(
          args, application, rooms_per_side
      )
      active_mask = np.array(
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
      reference = None
      if args.measure_accuracy:
        start = time.perf_counter()
        reference = _reference_solution(
            args, application, rooms_per_side, building_template
        )
        print(
            "reference ready "
            f"ms={(time.perf_counter() - start) * 1000.0:.0f}",
            flush=True,
        )
        _save_reference_fields(
            predictions_dir,
            application,
            rooms_per_side,
            reference,
            active_mask,
            mass_mask,
        )
      expected_enclosures = rooms_per_side**2 if APPLICATIONS[application][1] else 0
      radiation_metrics = _radiation_metrics(building_template)
      if radiation_metrics["radiative_enclosures"] != expected_enclosures:
        raise RuntimeError(
            "unexpected radiative enclosure count: "
            f"expected={expected_enclosures}, "
            f"actual={radiation_metrics['radiative_enclosures']}"
        )

      resident_simulator = None
      resident_warmed = False
      for repeat in pending:
        _write_progress(args.progress_file, rooms_per_side, repeat)
        print(
            f"starting sample device={args.device} solver={args.solver} "
            f"application={application} rooms_per_side={rooms_per_side} "
            f"repeat={repeat}",
            flush=True,
        )
        simulator, solver_build_ms = _prepare_simulator(
            args, args.solver, building_template, resident_simulator
        )
        if args.solver in RESIDENT_SOLVERS:
          resident_simulator = simulator

        warmup_ms = np.nan
        if args.solver in RESIDENT_SOLVERS and not resident_warmed:
          start = time.perf_counter()
          simulator.warmup(
              ambient_temperature=305.0, convection_coefficient=12.0
          )
          warmup_ms = (time.perf_counter() - start) * 1000.0
          resident_warmed = True
        elif args.solver not in RESIDENT_SOLVERS and (
            args.solver == "tf_cg" or args.solver in JAX_SOLVERS
        ):
          # Traces the TensorFlow graph, or builds the static JAX system and
          # compiles its solve, outside the timed steps.
          start = time.perf_counter()
          if application in EXTERIOR_APPLICATIONS or args.solver == GROUPED_JAX_SOLVER:
            # A full step traces the grouped solver or exterior forcing path.
            try:
              _advance_timestep(simulator, 0, args.time_step_seconds)
            except RuntimeError as error:
              # Repeat through the measured loop so this grid's failure is
              # recorded and the following grid still runs.
              print(f"Warmup reported: {error}", flush=True)
            _apply_initial_state(simulator, building_template)
            _set_forcing(simulator)
          else:
            simulator.update_temperature_estimates(
                building_template.temp.copy(), 305.0, 12.0
            )
          warmup_ms = (time.perf_counter() - start) * 1000.0
        elif application in EXTERIOR_APPLICATIONS and args.solver not in RESIDENT_SOLVERS:
          start = time.perf_counter()
          simulator._prepare_exterior_radiation(305.0)
          warmup_ms = (time.perf_counter() - start) * 1000.0

        step_times_ms = []
        cg_iteration_counts = []
        outer_iteration_counts = []
        failure = None
        total_start = time.perf_counter()
        for simulation_step in range(args.simulation_steps):
          step_start = time.perf_counter()
          try:
            converged = _advance_timestep(
                simulator, simulation_step, args.time_step_seconds
            )
          except RuntimeError as error:
            if application not in EXTERIOR_APPLICATIONS:
              raise
            failure = str(error)
            break
          step_times_ms.append((time.perf_counter() - step_start) * 1000.0)
          if not converged:
            failure = (
                "partitioned benchmark did not converge: "
                f"application={application}, rooms_per_side={rooms_per_side}, "
                f"solver={args.solver}, simulation_step={simulation_step}, "
                f"outer_max_delta_K={simulator.last_outer_max_delta}"
            )
            if application not in EXTERIOR_APPLICATIONS:
              raise RuntimeError(failure)
            break
          cg_iteration_counts.append(
              getattr(simulator, "last_cg_iteration_count", np.nan)
          )
          outer_iteration_counts.append(
              getattr(simulator, "last_outer_iteration_count", np.nan)
          )
        elapsed_ms = (time.perf_counter() - total_start) * 1000.0
        if failure is not None:
          failure_csv = output_csv.with_name("sample_failures.csv")
          pd.DataFrame([{
              "device": args.device, "solver": args.solver,
              "application": application, "rooms_per_side": rooms_per_side,
              "repeat": repeat, "simulation_step": simulation_step,
              "reason": failure,
          }]).to_csv(
              failure_csv, mode="a", header=not failure_csv.exists(), index=False
          )
          print(f"FAILED {failure}", flush=True)
          continue

        # Scored after timing, so the device-to-host copy is not timed.
        accuracy = {"max_abs_error_K": np.nan, "rms_error_K": np.nan}
        temperature, mass_temperature = _simulator_state(simulator)
        if reference is not None:
          reference_masses = reference["interior_mass_temperature"]
          accuracy = _accuracy_metrics(
              temperature,
              reference["temperature"][-1],
              active_mask,
              mass_temperature,
              None if reference_masses is None else reference_masses[-1],
              mass_mask,
          )
        # Repeats are deterministic, so one prediction per case is enough.
        prediction_path = _prediction_path(
            predictions_dir,
            application,
            rooms_per_side,
            f"{args.device}_{args.solver}",
        )
        if not prediction_path.exists():
          _save_fields(
              prediction_path,
              temperature=temperature,
              interior_mass_temperature=mass_temperature,
          )

        active_cvs = sum(
            len(neighbors) > 1
            for row in simulator.building.neighbors
            for neighbors in row
        )
        cell_area_m2 = (args.cv_size_cm / 100.0) ** 2
        active_floor_area_m2 = active_cvs * cell_area_m2
        row = {
            "device": args.device,
            "application": application,
            "rooms_per_side": rooms_per_side,
            "room_count": rooms_per_side**2,
            "room_interior_size": args.room_interior_size,
            "room_size_meters": args.room_size_meters,
            "room_area_m2": args.room_size_meters**2,
            "room_area_sqft": args.room_size_meters**2 * 10.7639104167,
            "cv_size_cm": args.cv_size_cm,
            "reference_cell_size_cm": args.cv_size_cm,
            "grid_rows": simulator.building.temp.shape[0],
            "grid_columns": simulator.building.temp.shape[1],
            "active_cvs": active_cvs,
            "active_floor_area_m2": active_floor_area_m2,
            "active_floor_area_sqft": active_floor_area_m2 * 10.7639104167,
            "solver": args.solver,
            "thermal_node_count": getattr(simulator, "thermal_node_count", np.nan),
            "mass_node_count": getattr(simulator, "mass_node_count", np.nan),
            "repeat": repeat,
            "simulation_steps": args.simulation_steps,
            "simulated_duration_seconds": (
                args.simulation_steps * args.time_step_seconds
            ),
            "building_setup_ms": building_setup_ms,
            "solver_build_ms": solver_build_ms,
            "warmup_ms": warmup_ms,
            "elapsed_ms": elapsed_ms,
            "mean_step_ms": float(np.mean(step_times_ms)),
            "median_step_ms": float(np.median(step_times_ms)),
            "maximum_step_ms": float(np.max(step_times_ms)),
            "cg_iterations": getattr(
                simulator, "last_cg_iteration_count", np.nan
            ),
            "total_cg_iterations": float(np.nansum(cg_iteration_counts))
            if not np.all(np.isnan(cg_iteration_counts))
            else np.nan,
            "mean_cg_iterations": float(np.nanmean(cg_iteration_counts))
            if not np.all(np.isnan(cg_iteration_counts))
            else np.nan,
            "outer_iterations": getattr(
                simulator, "last_outer_iteration_count", np.nan
            ),
            "total_outer_iterations": float(
                np.nansum(outer_iteration_counts)
            ),
            "mean_outer_iterations": float(
                np.nanmean(outer_iteration_counts)
            ),
            "outer_max_delta_K": getattr(
                simulator, "last_outer_max_delta", np.nan
            ),
            "time_step_seconds": args.time_step_seconds,
            "convergence_threshold": _solver_convergence_threshold(
                args, args.solver
            ),
            "iteration_limit": args.iteration_limit,
            "cg_tolerance": _solver_cg_tolerance(args, args.solver),
            **accuracy,
            **radiation_metrics,
        }
        rows.append(row)
        completed.add(tuple(row[column] for column in RESULT_KEY_COLUMNS))
        frame = pd.DataFrame(rows)
        _save_results(frame, output_csv)
        print(
            f"saved rooms={row['room_count']} grid={row['grid_rows']}x"
            f"{row['grid_columns']} enclosures={row['radiative_enclosures']} "
            f"setup_ms={building_setup_ms:.3f} "
            f"{args.simulation_steps}_step_ms={elapsed_ms:.3f} "
            f"max_abs_error_K={accuracy['max_abs_error_K']:.3g}",
            flush=True,
        )

  if args.progress_file is not None:
    args.progress_file.unlink(missing_ok=True)
  _save_results(pd.DataFrame(rows), output_csv)
  return pd.DataFrame(rows)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--rooms-per-side",
      default="1,2,3,4,5,6,8,10,12,14,16,18,20,22,24,26,28",
  )
  parser.add_argument("--room-size-meters", type=float, default=5.0)
  parser.add_argument("--cv-size-cm", type=float, default=20.0)
  parser.add_argument("--simulation-steps", type=int, default=12)
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--solver", choices=SOLVERS, required=True)
  parser.add_argument("--applications", default=",".join(DEFAULT_APPLICATIONS))
  parser.add_argument("--time-step-seconds", type=float, default=300.0)
  parser.add_argument("--convergence-threshold", type=float, default=1e-4)
  parser.add_argument(
      "--tensorflow-convergence-threshold", type=float, default=1.5e-4
  )
  parser.add_argument(
      "--resident-convergence-threshold", type=float, default=1.5e-4
  )
  parser.add_argument("--iteration-limit", type=int, default=5000)
  parser.add_argument("--cg-tolerance", type=float, default=1e-10)
  parser.add_argument("--resident-cg-tolerance", type=float, default=1e-8)
  parser.add_argument("--jax-fp32-cg-tolerance", type=float, default=1e-6)
  parser.add_argument(
      "--uniform-tolerances",
      action="store_true",
      help=(
          "Use --convergence-threshold and --cg-tolerance for every solver,"
          " ignoring the per-solver overrides."
      ),
  )
  parser.add_argument(
      "--measure-accuracy",
      action="store_true",
      help=(
          "Score every sample against a float64 sparse direct reference and"
          " record max_abs_error_K and rms_error_K."
      ),
  )
  parser.add_argument(
      "--cache-dir",
      type=Path,
      help="Directory caching building templates and reference solutions.",
  )
  parser.add_argument(
      "--predictions-dir",
      type=Path,
      help=(
          "Directory for the final air and interior mass temperatures of the"
          " reference and of each solver, as compressed .npz files. Defaults"
          " to <output-dir>/predictions."
      ),
  )
  parser.add_argument("--expected-device", choices=("cpu", "gpu"))
  parser.add_argument("--progress-file", type=Path)
  parser.add_argument("--force", action="store_true")
  parser.add_argument(
      "--output-dir",
      type=Path,
      default=Path(
          "reference/benchmark_outputs/partitioned_enclosure_sensitivity"
      ),
  )
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
  if args.room_size_meters <= 0.0:
    parser.error("room-size-meters must be positive")
  if args.cv_size_cm <= 0.0:
    parser.error("cv-size-cm must be positive")
  if args.simulation_steps < 1:
    parser.error("simulation-steps must be positive")
  if any(value < 1 for value in args.rooms_per_side):
    parser.error("rooms-per-side values must be positive")
  room_size_cm = args.room_size_meters * 100.0
  args.room_interior_size = int(round(room_size_cm / args.cv_size_cm))
  if not np.isclose(
      args.room_interior_size * args.cv_size_cm, room_size_cm
  ):
    parser.error(
        "room-size-meters must be an integer multiple of cv-size-cm"
    )

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
  output_csv = args.output_dir / "partitioned_enclosure_grid_time.csv"
  frame = _run(args, output_csv)
  if not frame.empty:
    summary = frame.groupby(
        ["application", "room_count", "solver"]
    )["elapsed_ms"].median()
    print(f"{framework} device: {args.device}\n{summary.to_string()}")


if __name__ == "__main__":
  main()
