"""Three-way heat-balance solver benchmark and size sensitivity study.

Compares TensorFlow Jacobi FDM, NumPy/SciPy sparse direct solve, and
TensorFlow matrix-free preconditioned conjugate gradient. This is a manual
benchmark, not a unit test.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import time
from unittest import mock

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from smart_control.simulator import building as building_py
from smart_control.simulator import numpy_scipy_simulator
from smart_control.simulator import simulator_flexible_floor_plan_test
from smart_control.simulator import tf_simulator
from smart_control.simulator import weather_controller as weather_controller_py


@dataclass(frozen=True)
class PhysicsCase:
  name: str
  include_interior_mass: bool
  include_radiation: bool


CASES = (
    PhysicsCase("base", False, False),
    PhysicsCase("interior_mass", True, False),
    PhysicsCase("radiation", False, True),
    PhysicsCase("mass_and_radiation", True, True),
)
SOLVERS = ("numpy_scipy", "tf_cg", "tf_fdm")
SOLVER_LABELS = {
    "numpy_scipy": "NumPy + SciPy\n(sparse direct)",
    "tf_cg": "TensorFlow\n(PCG)",
    "tf_fdm": "TensorFlow\n(Jacobi FDM)",
}
SOLVER_COLORS = {
    "numpy_scipy": "#4C78A8",
    "tf_cg": "#54A24B",
    "tf_fdm": "#F58518",
}


def _simulator_args(
    building, time_step_seconds, convergence_threshold, iteration_limit
):
  fixture = simulator_flexible_floor_plan_test.FlexibleFloorplanSimulatorTest()
  return (
      building,
      fixture._create_small_hvac(),  # pylint: disable=protected-access
      mock.create_autospec(weather_controller_py.WeatherController),
      time_step_seconds,
      convergence_threshold,
      iteration_limit,
      iteration_limit,
      pd.Timestamp("2012-06-21 06:00:00"),
  )


def _build_solver(
    solver: str,
    building,
    time_step_seconds: float,
    convergence_threshold: float,
    iteration_limit: int,
    cg_tolerance: float,
):
  args = _simulator_args(
      building, time_step_seconds, convergence_threshold, iteration_limit
  )
  if solver == "numpy_scipy":
    return numpy_scipy_simulator.NumpyScipySimulator(
        *args, relative_convergence_threshold=None
    )
  return tf_simulator.TFSimulator(
      *args,
      relative_convergence_threshold=None,
      solver_method="cg" if solver == "tf_cg" else "fdm",
      cg_tolerance=cg_tolerance,
      cg_iteration_limit=iteration_limit,
  )


def _create_physics_solver(
    case: PhysicsCase,
    solver: str,
    time_step_seconds: float,
    convergence_threshold: float,
    iteration_limit: int,
    cg_tolerance: float,
):
  fixture = simulator_flexible_floor_plan_test.FlexibleFloorplanSimulatorTest()
  _, building = fixture._create_simulator_and_building(  # pylint: disable=protected-access
      initial_temp=292.0,
      include_interior_mass=case.include_interior_mass,
      include_radiative_heat_transfer=case.include_radiation,
      convergence_threshold=convergence_threshold,
      iteration_limit=iteration_limit,
  )
  building.conductivity[:] = 0.5
  simulator = _build_solver(
      solver,
      building,
      time_step_seconds,
      convergence_threshold,
      iteration_limit,
      cg_tolerance,
  )
  if case.include_interior_mass:
    simulator.building.interior_mass_temp[
        simulator.building.interior_mass_mask
    ] = np.linspace(
        290.0,
        294.0,
        int(np.sum(simulator.building.interior_mass_mask)),
    )
  diffuser_coordinates = np.argwhere(simulator.building.diffusers > 0.0)
  hvac_source = tuple(diffuser_coordinates[len(diffuser_coordinates) // 2])
  boundary_coordinates = sorted(
      coordinate
      for coordinate, neighbors in np.ndenumerate(
          np.asarray(simulator.building.neighbors, dtype=object)
      )
      if len(neighbors) in (2, 3)
  )
  surface_source = boundary_coordinates[len(boundary_coordinates) // 3]
  return simulator, hvac_source, surface_source


def _forcing(step: int, step_count: int) -> tuple[float, float, float, float]:
  progress = step / max(1, step_count - 1)
  phase = 2.0 * math.pi * progress
  ambient_temperature = 305.0 + 0.4 * math.sin(phase)
  convection_coefficient = 12.0 + 0.3 * math.cos(phase)
  if progress < 1.0 / 3.0:
    hvac_power = 50.0
  elif progress < 2.0 / 3.0:
    hvac_power = -30.0
  else:
    hvac_power = 15.0
  surface_gain = 25.0 * max(0.0, math.sin(math.pi * progress))
  return ambient_temperature, convection_coefficient, hvac_power, surface_gain


def _apply_forcing(simulator, hvac_source, surface_source, step, step_count):
  ambient, convection, hvac_power, surface_gain = _forcing(step, step_count)
  simulator.building.input_q.fill(0.0)
  simulator.building.input_q[hvac_source] += hvac_power
  simulator.building.input_q[surface_source] += surface_gain
  return ambient, convection


def _run_multistep_case(
    case: PhysicsCase,
    repeat: int,
    step_count: int,
    time_step_seconds: float,
    convergence_threshold: float,
    iteration_limit: int,
    cg_tolerance: float,
    maximum_air_difference: float,
    maximum_mass_difference: float,
    snapshot_steps: set[int],
):
  simulations = {
      solver: _create_physics_solver(
          case,
          solver,
          time_step_seconds,
          convergence_threshold,
          iteration_limit,
          cg_tolerance,
      )
      for solver in SOLVERS
  }
  cg_simulator, cg_hvac_source, cg_surface_source = simulations["tf_cg"]
  warmup_ambient, warmup_convection = _apply_forcing(
      cg_simulator,
      cg_hvac_source,
      cg_surface_source,
      0,
      step_count,
  )
  warmup_start = time.perf_counter()
  cg_simulator.update_temperature_estimates(
      cg_simulator.building.temp.copy(),
      ambient_temperature=warmup_ambient,
      convection_coefficient=warmup_convection,
  )
  cg_warmup_ms = (time.perf_counter() - warmup_start) * 1000.0
  timing_rows = []
  error_rows = []
  snapshots = {}
  for step in range(step_count):
    forcing = {}
    for solver, (simulator, hvac_source, surface_source) in simulations.items():
      forcing[solver] = _apply_forcing(
          simulator, hvac_source, surface_source, step, step_count
      )
    execution_order = list(SOLVERS)
    shift = repeat % len(execution_order)
    execution_order = execution_order[shift:] + execution_order[:shift]
    for solver in execution_order:
      simulator = simulations[solver][0]
      ambient, convection = forcing[solver]
      start = time.perf_counter()
      converged = simulator.finite_differences_timestep(
          ambient_temperature=ambient,
          convection_coefficient=convection,
      )
      elapsed_ms = (time.perf_counter() - start) * 1000.0
      if not converged:
        raise RuntimeError(
            f"solver did not converge: case={case.name}, solver={solver}, "
            f"repeat={repeat}, step={step}"
        )
      timing_rows.append({
          "case": case.name,
          "solver": solver,
          "repeat": repeat,
          "step": step,
          "simulated_hours": (step + 1) * time_step_seconds / 3600.0,
          "elapsed_ms": elapsed_ms,
          "cg_iterations": (
              simulator.last_cg_iteration_count if solver == "tf_cg" else None
          ),
          "cg_warmup_ms": cg_warmup_ms if solver == "tf_cg" else None,
      })

    reference = simulations["numpy_scipy"][0]
    for solver in ("tf_cg", "tf_fdm"):
      candidate = simulations[solver][0]
      air_difference = float(
          np.max(np.abs(candidate.building.temp - reference.building.temp))
      )
      mass_difference = None
      if case.include_interior_mass:
        mass_difference = float(
            np.max(
                np.abs(
                    candidate.building.interior_mass_temp
                    - reference.building.interior_mass_temp
                )
            )
        )
      if air_difference > maximum_air_difference or (
          mass_difference is not None
          and mass_difference > maximum_mass_difference
      ):
        raise RuntimeError(
            f"solution mismatch: case={case.name}, solver={solver}, "
            f"repeat={repeat}, step={step}, air={air_difference}, "
            f"mass={mass_difference}"
        )
      error_rows.append({
          "case": case.name,
          "solver": solver,
          "repeat": repeat,
          "step": step,
          "simulated_hours": (step + 1) * time_step_seconds / 3600.0,
          "air_max_difference_K": air_difference,
          "mass_max_difference_K": mass_difference,
      })
    if repeat == 0 and step in snapshot_steps:
      snapshots[step] = {
          solver: simulations[solver][0].building.temp.copy()
          for solver in SOLVERS
      }
  return timing_rows, error_rows, snapshots


def _rectangular_building(interior_size: int):
  floor_plan = np.full((interior_size + 4, interior_size + 4), 2, dtype=int)
  floor_plan[1:-1, 1:-1] = 1
  floor_plan[2:-2, 2:-2] = 0
  air = building_py.MaterialProperties(
      conductivity=0.5, heat_capacity=1005.0, density=1.2
  )
  wall = building_py.MaterialProperties(
      conductivity=0.5, heat_capacity=800.0, density=1800.0
  )
  return building_py.FloorPlanBasedBuilding(
      cv_size_cm=20.0,
      floor_height_cm=300.0,
      initial_temp=292.0,
      inside_air_properties=air,
      inside_wall_properties=wall,
      building_exterior_properties=wall,
      floor_plan=floor_plan,
      zone_map=floor_plan.copy(),
      buffer_from_walls=0,
  )


def _run_size_sensitivity(
    sizes,
    repeats,
    time_step_seconds,
    convergence_threshold,
    iteration_limit,
    cg_tolerance,
):
  rows = []
  for interior_size in sizes:
    for repeat in range(repeats):
      for solver in SOLVERS:
        building = _rectangular_building(interior_size)
        simulator = _build_solver(
            solver,
            building,
            time_step_seconds,
            convergence_threshold,
            iteration_limit,
            cg_tolerance,
        )
        source = tuple(np.argwhere(building.diffusers > 0.0)[0])
        building.input_q[source] = 50.0
        cg_warmup_ms = None
        if solver == "tf_cg":
          warmup_start = time.perf_counter()
          simulator.update_temperature_estimates(
              simulator.building.temp.copy(),
              ambient_temperature=305.0,
              convection_coefficient=12.0,
          )
          cg_warmup_ms = (time.perf_counter() - warmup_start) * 1000.0
        start = time.perf_counter()
        converged = simulator.finite_differences_timestep(
            ambient_temperature=305.0,
            convection_coefficient=12.0,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if not converged:
          raise RuntimeError(
              f"size sensitivity did not converge: size={interior_size}, "
              f"solver={solver}"
          )
        active_cvs = sum(
            len(neighbors) > 1
            for row in building.neighbors
            for neighbors in row
        )
        rows.append({
            "interior_size": interior_size,
            "grid_rows": building.temp.shape[0],
            "grid_columns": building.temp.shape[1],
            "active_cvs": active_cvs,
            "solver": solver,
            "repeat": repeat,
            "elapsed_ms": elapsed_ms,
            "cg_iterations": (
                simulator.last_cg_iteration_count if solver == "tf_cg" else None
            ),
            "cg_warmup_ms": cg_warmup_ms,
        })
  return rows


def _run_cg_tolerance_sensitivity(
    tolerances,
    repeats,
    time_step_seconds,
    convergence_threshold,
    iteration_limit,
):
  rows = []
  case = CASES[0]
  for tolerance in tolerances:
    for repeat in range(repeats):
      direct, direct_hvac, direct_surface = _create_physics_solver(
          case,
          "numpy_scipy",
          time_step_seconds,
          convergence_threshold,
          iteration_limit,
          tolerance,
      )
      cg, cg_hvac, cg_surface = _create_physics_solver(
          case,
          "tf_cg",
          time_step_seconds,
          convergence_threshold,
          iteration_limit,
          tolerance,
      )
      ambient, convection = _apply_forcing(
          direct, direct_hvac, direct_surface, 0, 2
      )
      _apply_forcing(cg, cg_hvac, cg_surface, 0, 2)
      cg.update_temperature_estimates(
          cg.building.temp.copy(),
          ambient_temperature=ambient,
          convection_coefficient=convection,
      )
      direct.finite_differences_timestep(
          ambient_temperature=ambient,
          convection_coefficient=convection,
      )
      start = time.perf_counter()
      cg.finite_differences_timestep(
          ambient_temperature=ambient,
          convection_coefficient=convection,
      )
      rows.append({
          "cg_tolerance": tolerance,
          "repeat": repeat,
          "elapsed_ms": (time.perf_counter() - start) * 1000.0,
          "cg_iterations": cg.last_cg_iteration_count,
          "residual_norm": cg.last_cg_residual_norm,
          "air_max_difference_K": float(
              np.max(np.abs(cg.building.temp - direct.building.temp))
          ),
      })
  return rows


def _plot_boxplots(timing_frame, output_path):
  figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
  for axis, case in zip(axes.reshape(-1), CASES):
    values = [
        timing_frame[
            (timing_frame["case"] == case.name)
            & (timing_frame["solver"] == solver)
        ]["elapsed_ms"].to_numpy()
        for solver in SOLVERS
    ]
    boxplot = axis.boxplot(
        values,
        tick_labels=[SOLVER_LABELS[solver] for solver in SOLVERS],
        patch_artist=True,
        showmeans=True,
    )
    for box, solver in zip(boxplot["boxes"], SOLVERS):
      box.set_facecolor(SOLVER_COLORS[solver])
    axis.set_yscale("log")
    axis.set_title(case.name.replace("_", " ").title())
    axis.set_ylabel("Time per timestep [ms, log scale]")
    axis.grid(axis="y", alpha=0.3)
  figure.suptitle("Three heat-balance solvers: repeated multistep timing")
  figure.savefig(output_path, dpi=180)
  plt.close(figure)


def _plot_size_sensitivity(size_frame, output_path):
  summary = (
      size_frame.groupby(["active_cvs", "solver"], as_index=False)["elapsed_ms"]
      .median()
      .sort_values("active_cvs")
  )
  figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
  for solver in SOLVERS:
    data = summary[summary["solver"] == solver]
    axis.plot(
        data["active_cvs"],
        data["elapsed_ms"],
        marker="o",
        label=SOLVER_LABELS[solver].replace("\n", " "),
        color=SOLVER_COLORS[solver],
    )
  axis.set_xscale("log")
  axis.set_yscale("log")
  axis.set_xlabel("Active control volumes [log scale]")
  axis.set_ylabel("One timestep [ms, log scale]")
  axis.grid(alpha=0.3)
  axis.legend()
  axis.set_title("Base-physics grid-size sensitivity")
  figure.savefig(output_path, dpi=180)
  plt.close(figure)


def _plot_cg_tolerance_sensitivity(tolerance_frame, output_path):
  summary = tolerance_frame.groupby("cg_tolerance", as_index=False).agg(
      median_ms=("elapsed_ms", "median"),
      peak_air_difference_K=("air_max_difference_K", "max"),
  )
  figure, timing_axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
  error_axis = timing_axis.twinx()
  timing_axis.plot(
      summary["cg_tolerance"],
      summary["median_ms"],
      marker="o",
      color=SOLVER_COLORS["tf_cg"],
      label="Median TF-CG time",
  )
  error_axis.plot(
      summary["cg_tolerance"],
      summary["peak_air_difference_K"] * 1000.0,
      marker="s",
      color="#E45756",
      label="Peak error vs direct",
  )
  timing_axis.set_xscale("log")
  timing_axis.invert_xaxis()
  timing_axis.set_xlabel("CG relative residual tolerance")
  timing_axis.set_ylabel("One timestep [ms]")
  error_axis.set_ylabel("Peak air difference [mK]")
  timing_axis.grid(alpha=0.3)
  timing_axis.set_title("TF-CG tolerance sensitivity, base case")
  lines = timing_axis.lines + error_axis.lines
  timing_axis.legend(lines, [line.get_label() for line in lines])
  figure.savefig(output_path, dpi=180)
  plt.close(figure)


def _plot_convergence_snapshots(
    case, snapshots, time_step_seconds, output_path
):
  snapshot_steps = sorted(snapshots)
  reference_temperatures = np.concatenate(
      [snapshots[step]["numpy_scipy"].ravel() for step in snapshot_steps]
  )
  minimum_temperature = float(np.min(reference_temperatures))
  maximum_temperature = float(np.max(reference_temperatures))
  maximum_difference = max(
      float(
          np.max(
              np.abs(snapshots[step][solver] - snapshots[step]["numpy_scipy"])
          )
      )
      for step in snapshot_steps
      for solver in ("tf_cg", "tf_fdm")
  )
  figure, axes = plt.subplots(
      len(snapshot_steps),
      5,
      figsize=(15, 3.3 * len(snapshot_steps)),
      constrained_layout=True,
      squeeze=False,
  )
  temperature_image = None
  difference_image = None
  for row, step in enumerate(snapshot_steps):
    reference = snapshots[step]["numpy_scipy"]
    cg_temperature = snapshots[step]["tf_cg"]
    fdm_temperature = snapshots[step]["tf_fdm"]
    temperature_image = axes[row, 0].imshow(
        reference,
        cmap="coolwarm",
        vmin=minimum_temperature,
        vmax=maximum_temperature,
    )
    axes[row, 1].imshow(
        cg_temperature,
        cmap="coolwarm",
        vmin=minimum_temperature,
        vmax=maximum_temperature,
    )
    axes[row, 2].imshow(
        fdm_temperature,
        cmap="coolwarm",
        vmin=minimum_temperature,
        vmax=maximum_temperature,
    )
    cg_difference = np.abs(cg_temperature - reference) * 1000.0
    fdm_difference = np.abs(fdm_temperature - reference) * 1000.0
    difference_image = axes[row, 3].imshow(
        cg_difference,
        cmap="magma",
        vmin=0.0,
        vmax=max(maximum_difference * 1000.0, 1e-6),
    )
    axes[row, 4].imshow(
        fdm_difference,
        cmap="magma",
        vmin=0.0,
        vmax=max(maximum_difference * 1000.0, 1e-6),
    )
    simulated_hours = (step + 1) * time_step_seconds / 3600.0
    axes[row, 0].set_ylabel(f"t={simulated_hours:.2f} h")
    axes[row, 3].set_title(f"|CG-ref|, max={np.max(cg_difference):.3f} mK")
    axes[row, 4].set_title(f"|FDM-ref|, max={np.max(fdm_difference):.3f} mK")
    for column in range(5):
      axes[row, column].set_xticks([])
      axes[row, column].set_yticks([])
  axes[0, 0].set_title("NumPy/SciPy direct [K]")
  axes[0, 1].set_title("TensorFlow PCG [K]")
  axes[0, 2].set_title("TensorFlow FDM [K]")
  figure.colorbar(
      temperature_image,
      ax=axes[:, :3].ravel().tolist(),
      label="Temperature [K]",
      shrink=0.85,
  )
  figure.colorbar(
      difference_image,
      ax=axes[:, 3:].ravel().tolist(),
      label="Absolute difference from direct [mK]",
      shrink=0.85,
  )
  figure.suptitle(f"Three-solver convergence: {case.name.replace('_', ' ')}")
  figure.savefig(output_path, dpi=180)
  plt.close(figure)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--hours", type=float, default=1.0)
  parser.add_argument("--time-step-seconds", type=float, default=300.0)
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--convergence-threshold", type=float, default=1e-4)
  parser.add_argument("--iteration-limit", type=int, default=2000)
  parser.add_argument("--cg-tolerance", type=float, default=1e-10)
  parser.add_argument("--max-air-difference", type=float, default=5e-3)
  parser.add_argument("--max-mass-difference", type=float, default=5e-4)
  parser.add_argument("--sensitivity-sizes", default="8,16,24,32,48,64")
  parser.add_argument("--sensitivity-repeats", type=int, default=2)
  parser.add_argument("--cg-sensitivity-tolerances", default="1e-6,1e-8,1e-10")
  parser.add_argument(
      "--output-dir",
      type=Path,
      default=Path("reference/benchmark_outputs/three_solver_sensitivity"),
  )
  args = parser.parse_args()
  sizes = tuple(int(value) for value in args.sensitivity_sizes.split(","))
  cg_sensitivity_tolerances = tuple(
      float(value) for value in args.cg_sensitivity_tolerances.split(",")
  )
  if (
      args.hours <= 0
      or args.time_step_seconds <= 0
      or args.repeats < 1
      or args.convergence_threshold <= 0
      or args.iteration_limit < 1
      or args.cg_tolerance <= 0
      or args.max_air_difference <= 0
      or args.max_mass_difference <= 0
      or args.sensitivity_repeats < 1
      or not sizes
      or min(sizes) < 2
      or not cg_sensitivity_tolerances
      or min(cg_sensitivity_tolerances) <= 0
  ):
    parser.error("all numeric arguments and sensitivity sizes must be positive")

  step_count = max(2, int(round(args.hours * 3600 / args.time_step_seconds)))
  snapshot_steps = {0, step_count // 2, step_count - 1}
  timing_rows = []
  error_rows = []
  case_snapshots = {}
  for case in CASES:
    snapshots = {}
    for repeat in range(args.repeats):
      timings, errors, repeat_snapshots = _run_multistep_case(
          case,
          repeat,
          step_count,
          args.time_step_seconds,
          args.convergence_threshold,
          args.iteration_limit,
          args.cg_tolerance,
          args.max_air_difference,
          args.max_mass_difference,
          snapshot_steps,
      )
      timing_rows.extend(timings)
      error_rows.extend(errors)
      snapshots.update(repeat_snapshots)
    case_snapshots[case.name] = snapshots

  size_rows = _run_size_sensitivity(
      sizes,
      args.sensitivity_repeats,
      args.time_step_seconds,
      args.convergence_threshold,
      args.iteration_limit,
      args.cg_tolerance,
  )
  tolerance_rows = _run_cg_tolerance_sensitivity(
      cg_sensitivity_tolerances,
      args.sensitivity_repeats,
      args.time_step_seconds,
      args.convergence_threshold,
      args.iteration_limit,
  )
  timing_frame = pd.DataFrame(timing_rows)
  error_frame = pd.DataFrame(error_rows)
  size_frame = pd.DataFrame(size_rows)
  tolerance_frame = pd.DataFrame(tolerance_rows)
  summary = (
      timing_frame.groupby(["case", "solver"], as_index=False)
      .agg(
          samples=("elapsed_ms", "size"),
          median_ms=("elapsed_ms", "median"),
          median_cg_iterations=("cg_iterations", "median"),
      )
      .merge(
          error_frame.groupby(["case", "solver"], as_index=False).agg(
              peak_air_difference_K=("air_max_difference_K", "max"),
              peak_mass_difference_K=("mass_max_difference_K", "max"),
          ),
          on=["case", "solver"],
          how="left",
      )
  )
  args.output_dir.mkdir(parents=True, exist_ok=True)
  timing_frame.to_csv(args.output_dir / "multistep_timing.csv", index=False)
  error_frame.to_csv(args.output_dir / "multistep_errors.csv", index=False)
  summary.to_csv(args.output_dir / "multistep_summary.csv", index=False)
  size_frame.to_csv(args.output_dir / "size_sensitivity.csv", index=False)
  tolerance_frame.to_csv(
      args.output_dir / "cg_tolerance_sensitivity.csv", index=False
  )
  _plot_boxplots(timing_frame, args.output_dir / "timing_boxplots.png")
  _plot_size_sensitivity(size_frame, args.output_dir / "size_sensitivity.png")
  _plot_cg_tolerance_sensitivity(
      tolerance_frame,
      args.output_dir / "cg_tolerance_sensitivity.png",
  )
  for case in CASES:
    _plot_convergence_snapshots(
        case,
        case_snapshots[case.name],
        args.time_step_seconds,
        args.output_dir / f"heatmap_{case.name}.png",
    )
  print(
      f"hours={args.hours:g} steps={step_count} repeats={args.repeats} "
      f"samples_per_case_solver={step_count * args.repeats}"
  )
  print(summary.to_string(index=False))
  print("\nSize sensitivity medians:")
  print(
      size_frame.groupby(["active_cvs", "solver"])["elapsed_ms"]
      .median()
      .unstack()
      .to_string()
  )
  print("\nCG tolerance sensitivity:")
  print(
      tolerance_frame.groupby("cg_tolerance")
      .agg(
          median_ms=("elapsed_ms", "median"),
          median_iterations=("cg_iterations", "median"),
          peak_air_difference_K=("air_max_difference_K", "max"),
      )
      .to_string()
  )
  print(f"artifacts={args.output_dir}")


if __name__ == "__main__":
  main()
