"""Grid-size timing sensitivity for three heat-balance applications."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import time
import warnings

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from smart_control.simulator import building as building_py
from smart_control.simulator import jax_device_resident_simulator
from smart_control.simulator import jax_simulator
from smart_control.simulator import tf_device_resident_simulator
from reference.three_solver_sensitivity_benchmark import (
    SOLVERS as BASE_SOLVERS,
    SOLVER_COLORS as BASE_SOLVER_COLORS,
    SOLVER_LABELS as BASE_SOLVER_LABELS,
    _build_solver,
    _simulator_args,
)


APPLICATIONS = {
    "base": (False, False),
    "base_rad": (False, True),
    "base_rad_interior_mass": (True, True),
}
APPLICATION_LABELS = {
    "base": "Base model",
    "base_rad": "Base model + radiation",
    "base_rad_interior_mass": "Base model + radiation + interior mass",
}
RESIDENT_SOLVER = "tf_cg_resident_fp32"
JAX_SOLVER = "jax"
JAX_CG_SOLVER = "jax_cg"
JAX_RESIDENT_SOLVER = "jax_cg_resident"
JAX_RESIDENT_FP32_SOLVER = "jax_cg_resident_fp32"
JAX_SOLVERS = (
    JAX_SOLVER,
    JAX_CG_SOLVER,
    JAX_RESIDENT_SOLVER,
    JAX_RESIDENT_FP32_SOLVER,
)
# Solvers that keep their state on the device and expose warmup() and
# sync_state_from_building().
RESIDENT_SOLVERS = (
    RESIDENT_SOLVER,
    JAX_RESIDENT_SOLVER,
    JAX_RESIDENT_FP32_SOLVER,
)
# float32 cannot reach the float64 CG tolerance.
JAX_FP32_CG_TOLERANCE = 1e-6
SOLVERS = (*BASE_SOLVERS, RESIDENT_SOLVER, *JAX_SOLVERS)
SOLVER_LABELS = {
    **BASE_SOLVER_LABELS,
    RESIDENT_SOLVER: "TensorFlow PCG\n(resident FP32)",
    JAX_SOLVER: "JAX\n(sparse direct)",
    JAX_CG_SOLVER: "JAX\n(PCG)",
    JAX_RESIDENT_SOLVER: "JAX PCG\n(resident)",
    JAX_RESIDENT_FP32_SOLVER: "JAX PCG\n(resident FP32)",
}
SOLVER_COLORS = {
    **BASE_SOLVER_COLORS,
    RESIDENT_SOLVER: "#B279A2",
    JAX_SOLVER: "#9D755D",
    JAX_CG_SOLVER: "#72B7B2",
    JAX_RESIDENT_SOLVER: "#E45756",
    JAX_RESIDENT_FP32_SOLVER: "#FF9DA6",
}
SQUARE_FEET_PER_SQUARE_METER = 10.7639104167
RESULT_KEY_COLUMNS = (
    "device",
    "application",
    "interior_size",
    "solver",
    "repeat",
    "time_step_seconds",
    "convergence_threshold",
    "iteration_limit",
    "cg_tolerance",
)


def _create_building(interior_size: int, application: str):
  include_mass, include_radiation = APPLICATIONS[application]
  floor_plan = np.full((interior_size + 4, interior_size + 4), 2, dtype=int)
  floor_plan[1:-1, 1:-1] = 1
  floor_plan[2:-2, 2:-2] = 0
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
        cv_size_cm=20.0,
        floor_height_cm=300.0,
        initial_temp=292.0,
        inside_air_properties=air,
        inside_wall_properties=wall,
        building_exterior_properties=wall,
        floor_plan=floor_plan,
        zone_map=floor_plan.copy(),
        buffer_from_walls=0,
        **kwargs,
    )
  if include_mass:
    building.interior_mass_temp[building.interior_mass_mask] = 291.0
  return building


def _save_results(frame: pd.DataFrame, output_csv: Path):
  temporary_path = output_csv.with_suffix(".tmp")
  frame.to_csv(temporary_path, index=False)
  temporary_path.replace(output_csv)


def _add_reference_floor_area_columns(
    frame: pd.DataFrame, cell_size_cm: float
) -> pd.DataFrame:
  if frame.empty:
    return frame
  cell_area_m2 = (cell_size_cm / 100.0) ** 2
  frame = frame.copy()
  frame["reference_cell_size_cm"] = cell_size_cm
  frame["active_floor_area_m2"] = frame["active_cvs"] * cell_area_m2
  frame["active_floor_area_sqft"] = (
      frame["active_floor_area_m2"] * SQUARE_FEET_PER_SQUARE_METER
  )
  frame["interior_floor_area_m2"] = frame["interior_size"] ** 2 * cell_area_m2
  frame["interior_floor_area_sqft"] = (
      frame["interior_floor_area_m2"] * SQUARE_FEET_PER_SQUARE_METER
  )
  return frame


def _solver_cg_tolerance(args, solver: str) -> float:
  if getattr(args, "uniform_tolerances", False):
    return args.cg_tolerance
  if solver == RESIDENT_SOLVER:
    return args.resident_cg_tolerance
  if solver == JAX_RESIDENT_FP32_SOLVER:
    return getattr(args, "jax_fp32_cg_tolerance", JAX_FP32_CG_TOLERANCE)
  return args.cg_tolerance


def _solver_convergence_threshold(args, solver: str) -> float:
  if getattr(args, "uniform_tolerances", False):
    return args.convergence_threshold
  if solver not in ("numpy_scipy", JAX_SOLVER) and hasattr(
      args, "tensorflow_convergence_threshold"
  ):
    return args.tensorflow_convergence_threshold
  if solver == RESIDENT_SOLVER:
    return getattr(
        args, "resident_convergence_threshold", args.convergence_threshold
    )
  return args.convergence_threshold


def _build_benchmark_solver(args, solver: str, building):
  convergence_threshold = _solver_convergence_threshold(args, solver)
  if solver in (JAX_RESIDENT_SOLVER, JAX_RESIDENT_FP32_SOLVER):
    return jax_device_resident_simulator.JaxDeviceResidentSimulator(
        *_simulator_args(
            building,
            args.time_step_seconds,
            convergence_threshold,
            args.iteration_limit,
        ),
        relative_convergence_threshold=None,
        use_float64=solver == JAX_RESIDENT_SOLVER,
        cg_tolerance=_solver_cg_tolerance(args, solver),
        cg_iteration_limit=args.iteration_limit,
        sync_to_host_each_step=False,
    )
  if solver in JAX_SOLVERS:
    return jax_simulator.JaxSimulator(
        *_simulator_args(
            building,
            args.time_step_seconds,
            convergence_threshold,
            args.iteration_limit,
        ),
        relative_convergence_threshold=None,
        solver_method="cg" if solver == JAX_CG_SOLVER else "spsolve",
        cg_tolerance=args.cg_tolerance,
        cg_iteration_limit=args.iteration_limit,
    )
  if solver == RESIDENT_SOLVER:
    return tf_device_resident_simulator.TFDeviceResidentSimulator(
        *_simulator_args(
            building,
            args.time_step_seconds,
            convergence_threshold,
            args.iteration_limit,
        ),
        relative_convergence_threshold=None,
        cg_tolerance=args.resident_cg_tolerance,
        cg_iteration_limit=args.iteration_limit,
        sync_to_host_each_step=False,
    )
  return _build_solver(
      solver,
      building,
      args.time_step_seconds,
      convergence_threshold,
      args.iteration_limit,
      args.cg_tolerance,
  )


def _run(args, output_csv: Path):
  if args.progress_file is not None:
    args.progress_file.unlink(missing_ok=True)
  if output_csv.exists() and not args.force:
    frame = pd.read_csv(output_csv)
  else:
    frame = pd.DataFrame()
  if not frame.empty:
    configuration_defaults = {
        "time_step_seconds": args.time_step_seconds,
        "convergence_threshold": args.convergence_threshold,
        "iteration_limit": args.iteration_limit,
    }
    if "cg_tolerance" not in frame:
      frame["cg_tolerance"] = args.cg_tolerance
    for column, value in configuration_defaults.items():
      if column not in frame:
        frame[column] = value
    configuration_mask = np.ones(len(frame), dtype=bool)
    for column, value in configuration_defaults.items():
      configuration_mask &= np.isclose(frame[column], value)
    frame = frame[configuration_mask].copy()
    frame = _add_reference_floor_area_columns(
        frame, args.reference_cell_size_cm
    )
    frame = frame.drop_duplicates(list(RESULT_KEY_COLUMNS), keep="last")
    completed = set(
        frame.loc[:, list(RESULT_KEY_COLUMNS)].itertuples(
            index=False, name=None
        )
    )
  else:
    completed = set()

  for application in args.applications:
    for interior_size in args.sizes:
      pending = [
          (repeat, solver)
          for repeat in range(args.repeats)
          for solver in args.solvers
          if (
              args.device,
              application,
              interior_size,
              solver,
              repeat,
              args.time_step_seconds,
              args.convergence_threshold,
              args.iteration_limit,
              _solver_cg_tolerance(args, solver),
          )
          not in completed
      ]
      if not pending:
        print(
            f"skipped device={args.device} application={application} "
            f"interior_size={interior_size}",
            flush=True,
        )
        continue
      print(
          f"starting device={args.device} application={application} "
          f"interior_size={interior_size} pending_samples={len(pending)}",
          flush=True,
      )
      building_template = _create_building(interior_size, application)
      size_rows = []
      resident_simulator = None
      resident_warmed = False
      for repeat, solver in pending:
        if args.progress_file is not None:
          temporary_progress_file = args.progress_file.with_suffix(".tmp")
          temporary_progress_file.write_text(
              f"{interior_size},{repeat}\n", encoding="utf-8"
          )
          temporary_progress_file.replace(args.progress_file)
        print(
            f"starting sample device={args.device} "
            f"application={application} interior_size={interior_size} "
            f"solver={solver} repeat={repeat}",
            flush=True,
        )
        if solver == RESIDENT_SOLVER and resident_simulator is not None:
          simulator = resident_simulator
          simulator.building.temp = building_template.temp.copy()
          simulator.building.input_q.fill(0.0)
          if simulator.include_interior_mass:
            simulator.building.interior_mass_temp = (
                building_template.interior_mass_temp.copy()
            )
          simulator.sync_state_from_building()
        else:
          building = copy.deepcopy(building_template)
          simulator = _build_benchmark_solver(args, solver, building)
          if solver == RESIDENT_SOLVER:
            resident_simulator = simulator
        source = tuple(np.argwhere(simulator.building.diffusers > 0.0)[0])
        simulator.building.input_q[source] = 50.0
        warmup_ms = np.nan
        if solver == RESIDENT_SOLVER and not resident_warmed:
          start = time.perf_counter()
          simulator.warmup(
              ambient_temperature=305.0, convection_coefficient=12.0
          )
          warmup_ms = (time.perf_counter() - start) * 1000.0
          resident_warmed = True
        elif solver == "tf_cg":
          start = time.perf_counter()
          simulator.update_temperature_estimates(
              building.temp.copy(), 305.0, 12.0
          )
          warmup_ms = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        converged = simulator.finite_differences_timestep(
            ambient_temperature=305.0,
            convection_coefficient=12.0,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if not converged:
          raise RuntimeError(
              "did not converge: "
              f"application={application}, interior_size={interior_size}, "
              f"solver={solver}, "
              "outer_iterations="
              f"{getattr(simulator, 'last_outer_iteration_count', 'unknown')}, "
              "outer_max_delta_K="
              f"{getattr(simulator, 'last_outer_max_delta', 'unknown')}, "
              f"threshold_K={args.convergence_threshold}"
          )
        active_cvs = sum(
            len(neighbors) > 1
            for row in simulator.building.neighbors
            for neighbors in row
        )
        cell_area_m2 = (args.reference_cell_size_cm / 100.0) ** 2
        active_floor_area_m2 = active_cvs * cell_area_m2
        interior_floor_area_m2 = interior_size**2 * cell_area_m2
        row = {
            "device": args.device,
            "application": application,
            "interior_size": interior_size,
            "grid_rows": simulator.building.temp.shape[0],
            "grid_columns": simulator.building.temp.shape[1],
            "active_cvs": active_cvs,
            "reference_cell_size_cm": args.reference_cell_size_cm,
            "active_floor_area_m2": active_floor_area_m2,
            "active_floor_area_sqft": (
                active_floor_area_m2 * SQUARE_FEET_PER_SQUARE_METER
            ),
            "interior_floor_area_m2": interior_floor_area_m2,
            "interior_floor_area_sqft": (
                interior_floor_area_m2 * SQUARE_FEET_PER_SQUARE_METER
            ),
            "solver": solver,
            "repeat": repeat,
            "elapsed_ms": elapsed_ms,
            "cg_iterations": (
                simulator.last_cg_iteration_count
                if solver in ("tf_cg", RESIDENT_SOLVER)
                else np.nan
            ),
            "outer_iterations": getattr(
                simulator, "last_outer_iteration_count", np.nan
            ),
            "outer_max_delta_K": getattr(
                simulator, "last_outer_max_delta", np.nan
            ),
            "cg_warmup_ms": warmup_ms,
            "time_step_seconds": args.time_step_seconds,
            "convergence_threshold": args.convergence_threshold,
            "iteration_limit": args.iteration_limit,
            "cg_tolerance": _solver_cg_tolerance(args, solver),
        }
        size_rows.append(row)
        completed.add((
            args.device,
            application,
            interior_size,
            solver,
            repeat,
            args.time_step_seconds,
            args.convergence_threshold,
            args.iteration_limit,
            _solver_cg_tolerance(args, solver),
        ))
        frame = pd.concat((frame, pd.DataFrame([row])), ignore_index=True)
        frame = frame.drop_duplicates(list(RESULT_KEY_COLUMNS), keep="last")
        _save_results(frame, output_csv)
        print(
            f"saved sample device={args.device} application={application} "
            f"interior_size={interior_size} solver={solver} "
            f"repeat={repeat} elapsed_ms={elapsed_ms:.3f} "
            f"outer_iterations={row['outer_iterations']}",
            flush=True,
        )
      median_elapsed_ms = float(
          np.median([row["elapsed_ms"] for row in size_rows])
      )
      median_outer_iterations = float(
          np.median([row["outer_iterations"] for row in size_rows])
      )
      max_outer_delta = float(
          np.max([row["outer_max_delta_K"] for row in size_rows])
      )
      print(
          f"completed device={args.device} application={application} "
          f"interior_size={interior_size} active_cvs={active_cvs} "
          f"new_samples={len(size_rows)} median_ms={median_elapsed_ms:.3f} "
          f"median_outer_iterations={median_outer_iterations:g} "
          f"max_outer_delta_K={max_outer_delta:.3e}",
          flush=True,
      )
  frame = _add_reference_floor_area_columns(frame, args.reference_cell_size_cm)
  _save_results(frame, output_csv)
  if args.progress_file is not None:
    args.progress_file.unlink(missing_ok=True)
  return frame


def _plot(frame: pd.DataFrame, output_dir: Path):
  for application, label in APPLICATION_LABELS.items():
    data = frame[frame["application"] == application]
    summary = data.groupby(["active_cvs", "solver"], as_index=False).agg(
        median_ms=("elapsed_ms", "median")
    )
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    solvers = tuple(
        solver for solver in SOLVERS if solver in data["solver"].unique()
    )
    for solver in solvers:
      solver_data = summary[summary["solver"] == solver]
      axis.plot(
          solver_data["active_cvs"],
          solver_data["median_ms"],
          marker="o",
          color=SOLVER_COLORS[solver],
          label=SOLVER_LABELS[solver].replace("\n", " "),
      )
    axis.set(xscale="log", yscale="log")
    axis.set_xlabel("Active control volumes [log scale]")
    axis.set_ylabel("One timestep [ms, log scale]")
    axis.set_title(f"Grid-time sensitivity: {label}")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(output_dir / f"grid_time_{application}.png", dpi=180)
    plt.close(figure)

    active_cvs = sorted(data["active_cvs"].unique())
    column_count = min(4, len(active_cvs))
    row_count = math.ceil(len(active_cvs) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(3.6 * column_count, 3.5 * row_count),
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.reshape(-1)
    short_labels = {
        "numpy_scipy": "Direct",
        "tf_cg": "PCG",
        RESIDENT_SOLVER: "Resident\nFP32",
        "tf_fdm": "FDM",
    }
    for axis, active_cv in zip(flat_axes, active_cvs):
      values = [
          data[(data["active_cvs"] == active_cv) & (data["solver"] == solver)][
              "elapsed_ms"
          ].to_numpy()
          for solver in solvers
      ]
      boxplot = axis.boxplot(
          values,
          tick_labels=[short_labels[solver] for solver in solvers],
          patch_artist=True,
          showmeans=True,
      )
      for box, solver in zip(boxplot["boxes"], solvers):
        box.set_facecolor(SOLVER_COLORS[solver])
      axis.set_yscale("log")
      axis.set_title(f"{active_cv:,} CVs")
      axis.grid(axis="y", alpha=0.3)
    for axis in flat_axes[len(active_cvs) :]:
      axis.set_visible(False)
    for axis in axes[:, 0]:
      axis.set_ylabel("One timestep [ms, log scale]")
    figure.suptitle(f"Grid timing distributions: {label}")
    figure.savefig(output_dir / f"grid_boxplot_{application}.png", dpi=180)
    plt.close(figure)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--sizes",
      default=(
          "8,12,16,20,24,32,40,48,64,80,96,128,160,192,256,320,384,"
          "512,640,768"
      ),
  )
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--solvers", default=",".join(SOLVERS))
  parser.add_argument("--applications", default=",".join(APPLICATIONS))
  parser.add_argument("--reference-cell-size-cm", type=float, default=10.0)
  parser.add_argument("--time-step-seconds", type=float, default=300.0)
  parser.add_argument("--convergence-threshold", type=float, default=1e-4)
  parser.add_argument("--iteration-limit", type=int, default=2000)
  parser.add_argument("--cg-tolerance", type=float, default=1e-10)
  parser.add_argument("--resident-cg-tolerance", type=float, default=1e-8)
  parser.add_argument("--expected-device", choices=("cpu", "gpu"))
  parser.add_argument("--progress-file", type=Path)
  parser.add_argument("--force", action="store_true")
  parser.add_argument("--no-plot", action="store_true")
  parser.add_argument(
      "--output-dir",
      type=Path,
      default=Path("reference/benchmark_outputs/three_solver_sensitivity"),
  )
  args = parser.parse_args()
  args.sizes = tuple(int(value) for value in args.sizes.split(","))
  args.solvers = tuple(value.strip() for value in args.solvers.split(","))
  args.applications = tuple(
      value.strip() for value in args.applications.split(",")
  )
  unknown_solvers = set(args.solvers) - set(SOLVERS)
  if unknown_solvers:
    parser.error(f"unknown solvers: {sorted(unknown_solvers)}")
  unknown_applications = set(args.applications) - set(APPLICATIONS)
  if unknown_applications:
    parser.error(f"unknown applications: {sorted(unknown_applications)}")
  if args.reference_cell_size_cm <= 0.0:
    parser.error("reference-cell-size-cm must be positive")
  args.device = "gpu" if tf.config.list_physical_devices("GPU") else "cpu"
  if args.expected_device and args.device != args.expected_device:
    raise RuntimeError(
        f"expected {args.expected_device}, but TensorFlow detected"
        f" {args.device}"
    )
  args.output_dir.mkdir(parents=True, exist_ok=True)
  output_csv = args.output_dir / "application_grid_time.csv"
  frame = _run(args, output_csv)
  if not args.no_plot:
    _plot(frame, args.output_dir)
  summary = (
      frame.groupby(["application", "active_cvs", "solver"])["elapsed_ms"]
      .median()
      .unstack()
      .to_string()
  )
  print(f"TensorFlow device: {args.device}\n{summary}")


if __name__ == "__main__":
  main()
