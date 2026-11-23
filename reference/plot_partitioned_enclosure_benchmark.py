"""Merges and plots CPU/GPU partitioned-enclosure benchmark results."""

from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd

APPLICATION_COLORS = {
    "base": "#4C78A8",
    "base_rad": "#F58518",
    "base_rad_interior_mass": "#E45756",
    "base_rad_exterior": "#54A24B",
    "base_rad_interior_mass_exterior": "#B279A2",
}
APPLICATION_LABELS = {
    "base": "Base model",
    "base_rad": "Base model + radiation",
    "base_rad_interior_mass": "Base model + radiation + interior mass",
    "base_rad_exterior": "Base + interior and exterior radiation",
    "base_rad_interior_mass_exterior": "Base + interior and exterior radiation + mass",
}
SOLVER_LABELS = {
    "numpy_scipy": "NumPy/SciPy direct",
    "tf_cg": "TensorFlow PCG",
    "tf_cg_resident_fp32": "Device-resident FP32 PCG",
    "tf_fdm": "TensorFlow FDM",
    "jax": "JAX sparse direct",
    "jax_cg": "JAX PCG",
    "jax_cg_resident": "JAX resident PCG",
    "jax_cg_resident_fp32": "JAX resident FP32 PCG",
    "grouped_jax": "Grouped JAX linearized PCG",
}


def _load(path: Path, device: str) -> pd.DataFrame:
  frame = pd.read_csv(path)
  frame["device"] = device
  return frame


def _plot_solver(frame: pd.DataFrame, output_dir: Path, device: str, solver: str):
  data = frame[(frame["device"] == device) & (frame["solver"] == solver)]
  if data.empty:
    return
  summary = data.groupby(
      [
          "application",
          "room_count",
          "active_cvs",
          "active_floor_area_sqft",
          "simulation_steps",
          "simulated_duration_seconds",
      ],
      as_index=False,
  ).agg(
      median_timestep_ms=("elapsed_ms", "median"),
      median_setup_ms=("building_setup_ms", "median"),
      grid_rows=("grid_rows", "max"),
      grid_columns=("grid_columns", "max"),
      radiation_matrix_mib=("radiation_matrix_mib", "max"),
  )
  figure, axes = plt.subplots(
      2, 2, figsize=(13, 9), constrained_layout=True, sharex=True
  )
  square_feet_per_active_cv = float(
      (data["active_floor_area_sqft"] / data["active_cvs"]).median()
  )
  simulation_steps = int(data["simulation_steps"].iloc[0])
  simulation_hours = float(data["simulated_duration_seconds"].iloc[0]) / 3600.0
  for application, label in APPLICATION_LABELS.items():
    application_data = summary[summary["application"] == application]
    if application_data.empty:
      continue
    color = APPLICATION_COLORS[application]
    for axis, column in (
        (axes[0, 0], "median_timestep_ms"),
        (axes[0, 1], "median_timestep_ms"),
        (axes[1, 0], "median_setup_ms"),
        (axes[1, 1], "median_setup_ms"),
    ):
      axis.plot(
          application_data["active_cvs"],
          application_data[column],
          marker="o",
          linewidth=1.8,
          color=color,
          label=label,
      )
    endpoint = application_data.iloc[-1]
    axes[0, 0].annotate(
        f"{int(endpoint.grid_rows)}x{int(endpoint.grid_columns)}",
        (endpoint.active_cvs, endpoint.median_timestep_ms),
        xytext=(4, 5),
        textcoords="offset points",
        fontsize=7,
        color=color,
    )
    axes[1, 0].annotate(
        f"{endpoint.radiation_matrix_mib:.1f} MiB",
        (endpoint.active_cvs, endpoint.median_setup_ms),
        xytext=(4, 5),
        textcoords="offset points",
        fontsize=7,
        color=color,
    )

  axes[0, 1].set_yscale("log")
  axes[1, 1].set_yscale("log")
  axes[0, 0].set_title("Multi-step simulation time - linear scale")
  axes[0, 1].set_title("Multi-step simulation time - log scale")
  axes[1, 0].set_title("Building/view-factor setup - linear scale")
  axes[1, 1].set_title("Building/view-factor setup - log scale")
  for axis in axes.flat:
    axis.set_xscale("log")
    axis.set_xlabel("Active control volumes [log scale]")
    axis.grid(alpha=0.3)
    area_axis = axis.secondary_xaxis(
        -0.18,
        functions=(
            lambda active_cvs: active_cvs * square_feet_per_active_cv,
            lambda square_feet: square_feet / square_feet_per_active_cv,
        ),
    )
    area_axis.set_xlabel("Active floor area [ft², real scale]")
  axes[0, 0].set_ylabel(
      f"Median {simulation_steps}-step ({simulation_hours:g} h) time [ms]"
  )
  axes[0, 1].set_ylabel(
      f"Median {simulation_steps}-step ({simulation_hours:g} h) time "
      "[ms, log scale]"
  )
  axes[1, 0].set_ylabel("Median setup [ms]")
  axes[1, 1].set_ylabel("Median setup [ms, log scale]")
  axes[0, 0].legend()
  figure.suptitle(
      f"{device.upper()} - {SOLVER_LABELS[solver].replace(chr(10), ' ')}: "
      f"partitioned enclosure scaling, {simulation_steps} steps"
  )
  figure.savefig(
      output_dir / f"applications_by_solver_{device}_{solver}.png", dpi=180
  )
  plt.close(figure)


def _plot_memory(frame: pd.DataFrame, output_dir: Path) -> None:
  data = frame[frame["application"] == "base_rad_interior_mass"]
  if data.empty:
    return
  summary = data.groupby(
      ["room_count", "room_interior_size", "active_cvs", "active_floor_area_sqft"],
      as_index=False,
  ).agg(
      radiation_matrix_mib=("radiation_matrix_mib", "max"),
      max_enclosure_nodes=("max_enclosure_nodes", "max"),
      radiating_nodes=("radiating_nodes", "max"),
  )
  figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
  axes[0].plot(
      summary["active_cvs"], summary["radiation_matrix_mib"], marker="o"
  )
  axes[1].plot(summary["active_cvs"], summary["radiating_nodes"], marker="o")
  square_feet_per_active_cv = float(
      (summary["active_floor_area_sqft"] / summary["active_cvs"]).median()
  )
  for axis in axes:
    axis.set(xscale="log", yscale="log")
    axis.set_xlabel("Active control volumes [log scale]")
    axis.grid(alpha=0.3)
    area_axis = axis.secondary_xaxis(
        -0.18,
        functions=(
            lambda active_cvs: active_cvs * square_feet_per_active_cv,
            lambda square_feet: square_feet / square_feet_per_active_cv,
        ),
    )
    area_axis.set_xlabel("Active floor area [ft², real scale]")
  axes[0].set_ylabel("Stored VF + IFA blocks [MiB, log scale]")
  axes[1].set_ylabel("Radiating nodes [log scale]")
  axes[0].set_title("Partitioned radiation matrix storage")
  axes[1].set_title("Total radiating nodes")
  figure.suptitle("Partitioned enclosure memory scaling")
  figure.savefig(output_dir / "partitioned_enclosure_memory_scaling.png", dpi=180)
  plt.close(figure)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--cpu-csv", type=Path, nargs="+", default=())
  parser.add_argument("--gpu-csv", type=Path, nargs="+", default=())
  parser.add_argument("--output-dir", type=Path, required=True)
  args = parser.parse_args()
  frames = []
  for device, paths in (("cpu", args.cpu_csv), ("gpu", args.gpu_csv)):
    for path in paths:
      if path.exists():
        frames.append(_load(path, device))
  if not frames:
    raise ValueError("no benchmark CSV files were found")
  frame = pd.concat(frames, ignore_index=True)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  frame.to_csv(
      args.output_dir / "cpu_gpu_partitioned_enclosure_grid_time.csv",
      index=False,
  )
  for device in sorted(frame["device"].unique()):
    for solver in sorted(frame[frame["device"] == device]["solver"].unique()):
      _plot_solver(frame, args.output_dir, device, solver)
  _plot_memory(frame, args.output_dir)


if __name__ == "__main__":
  main()
