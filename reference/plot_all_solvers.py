"""Plots every solver on CPU and GPU together from the saved benchmark CSVs.

Reads the TensorFlow/SciPy results of run_partitioned_enclosure_cpu_gpu.sh and
the JAX results of run_partitioned_enclosure_jax_cpu_gpu.sh, and draws all
solver/device combinations on the same axes: one panel per application for
the timing and for the speedup over NumPy/SciPy, and a heatmap of the median
time. When the CSVs carry max_abs_error_K (--measure-accuracy), it also plots
the error and time against error. Nothing is re-run.

Samples are only comparable when they used the same Picard threshold and CG
tolerance; pass --convergence-threshold and --cg-tolerance to keep one setting
when the CSVs mix several.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

RESULT_CSV_NAME = "partitioned_enclosure_grid_time.csv"
DEFAULT_ROOT = Path("reference/benchmark_outputs/partitioned_enclosure_sensitivity")
APPLICATION_LABELS = {
    "base": "Base model",
    "base_rad": "Base + radiation",
    "base_rad_interior_mass": "Base + radiation + interior mass",
    "base_rad_exterior": "Base + interior and exterior radiation",
    "base_rad_interior_mass_exterior": "Base + interior and exterior radiation + mass",
}
# Solver order is also the legend and heatmap row order.
SOLVER_STYLES = {
    "numpy_scipy": ("NumPy/SciPy direct", "#4C78A8"),
    "jax": ("JAX sparse direct", "#9D755D"),
    "tf_fdm": ("TensorFlow FDM", "#F58518"),
    "tf_cg": ("TensorFlow PCG", "#54A24B"),
    "tf_cg_resident_fp32": ("TF resident FP32 PCG", "#B279A2"),
    "jax_cg": ("JAX PCG", "#17BECF"),
    "jax_cg_resident": ("JAX resident PCG", "#D62728"),
    "jax_cg_resident_fp32": ("JAX resident FP32 PCG", "#FF9DA6"),
    "grouped_jax": ("Grouped JAX linearized PCG", "#7A5195"),
}
DEVICE_STYLES = {"cpu": ("-", "o"), "gpu": ("--", "s")}
REFERENCE = ("numpy_scipy", "cpu")


def _load(root: Path, sources: tuple[str, ...]) -> pd.DataFrame:
  frames = []
  for source in sources:
    for device in ("cpu", "gpu"):
      path = root / source / device / RESULT_CSV_NAME
      if not path.exists():
        print(f"missing {path}; skipped")
        continue
      frame = pd.read_csv(path)
      frame["device"] = device
      if "max_abs_error_K" not in frame:
        frame["max_abs_error_K"] = np.nan
      frames.append(frame)
  if not frames:
    raise FileNotFoundError(f"no {RESULT_CSV_NAME} found under {root}")
  frame = pd.concat(frames, ignore_index=True)
  steps = frame["simulation_steps"].unique()
  if len(steps) != 1:
    raise ValueError(f"CSVs mix simulation_steps={sorted(steps)}")
  return frame


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
  return (
      frame.groupby(
          ["application", "device", "solver", "rooms_per_side", "active_cvs"],
          as_index=False,
      )
      .agg(
          median_ms=("elapsed_ms", "median"),
          samples=("elapsed_ms", "size"),
          max_abs_error_K=("max_abs_error_K", "max"),
      )
      .sort_values("active_cvs")
  )


def _series(summary: pd.DataFrame):
  """Yields (solver, device, label, color, linestyle, marker) present."""
  for solver, (solver_label, color) in SOLVER_STYLES.items():
    for device, (linestyle, marker) in DEVICE_STYLES.items():
      if (
          (summary["solver"] == solver) & (summary["device"] == device)
      ).any():
        yield (
            solver,
            device,
            f"{solver_label} ({device.upper()})",
            color,
            linestyle,
            marker,
        )


def _plot_lines(
    summary: pd.DataFrame,
    output_path: Path,
    value_column: str,
    ylabel: str,
    title: str,
    logarithmic_y: bool,
    reference_line: float | None = None,
) -> None:
  applications = [
      app for app in APPLICATION_LABELS if app in summary["application"].unique()
  ]
  figure, axes = plt.subplots(
      1,
      len(applications),
      figsize=(7.5 * len(applications), 6.5),
      constrained_layout=True,
      squeeze=False,
  )
  handles = {}
  for axis, application in zip(axes[0], applications):
    data = summary[summary["application"] == application]
    for solver, device, label, color, linestyle, marker in _series(summary):
      series = data[(data["solver"] == solver) & (data["device"] == device)]
      series = series.dropna(subset=[value_column])
      if series.empty:
        continue
      (line,) = axis.plot(
          series["active_cvs"],
          series[value_column],
          color=color,
          linestyle=linestyle,
          marker=marker,
          markersize=4,
          linewidth=1.6,
      )
      handles[label] = line
    if reference_line is not None:
      axis.axhline(reference_line, color="black", linewidth=1, linestyle=":")
    axis.set_xscale("log")
    if logarithmic_y:
      axis.set_yscale("log")
    axis.set_title(APPLICATION_LABELS[application])
    axis.set_xlabel("Active control volumes [log scale]")
    axis.grid(alpha=0.3, which="both")
  axes[0, 0].set_ylabel(ylabel)
  figure.legend(
      handles.values(),
      handles.keys(),
      loc="outside lower center",
      ncol=min(6, len(handles)),
      fontsize=9,
  )
  figure.suptitle(title)
  figure.savefig(output_path, dpi=180)
  plt.close(figure)
  print(f"wrote {output_path}")


def _plot_heatmap(
    summary: pd.DataFrame, output_path: Path, steps: int, rooms: list[int]
) -> None:
  applications = [
      app for app in APPLICATION_LABELS if app in summary["application"].unique()
  ]
  rows = [(s, d, label) for s, d, label, *_ in _series(summary)]
  figure, axes = plt.subplots(
      1,
      len(applications),
      figsize=(1.0 * len(rooms) * len(applications) + 4, 0.45 * len(rows) + 2.5),
      constrained_layout=True,
      squeeze=False,
  )
  values = summary["median_ms"].to_numpy()
  norm = mcolors.LogNorm(vmin=np.nanmin(values), vmax=np.nanmax(values))
  image = None
  for index, (axis, application) in enumerate(zip(axes[0], applications)):
    data = summary[summary["application"] == application]
    grid = np.full((len(rows), len(rooms)), np.nan)
    for row, (solver, device, _) in enumerate(rows):
      series = data[(data["solver"] == solver) & (data["device"] == device)]
      lookup = dict(zip(series["rooms_per_side"], series["median_ms"]))
      for column, room in enumerate(rooms):
        grid[row, column] = lookup.get(room, np.nan)
    image = axis.imshow(grid, cmap="viridis_r", norm=norm, aspect="auto")
    # Marks the fastest solver of each grid size.
    for column in range(len(rooms)):
      if np.all(np.isnan(grid[:, column])):
        continue
      best = int(np.nanargmin(grid[:, column]))
      axis.add_patch(
          plt.Rectangle(
              (column - 0.5, best - 0.5), 1, 1, fill=False, ec="red", lw=2
          )
      )
    for row in range(len(rows)):
      for column in range(len(rooms)):
        value = grid[row, column]
        if np.isnan(value):
          axis.text(column, row, "–", ha="center", va="center", fontsize=7)
          continue
        text = f"{value / 1000:.1f}s" if value >= 1000 else f"{value:.0f}"
        shade = norm(value)
        axis.text(
            column,
            row,
            text,
            ha="center",
            va="center",
            fontsize=7,
            color="white" if shade > 0.55 else "black",
        )
    axis.set_xticks(range(len(rooms)))
    axis.set_xticklabels([f"{room * room}" for room in rooms], fontsize=8)
    axis.set_xlabel("Rooms")
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels(
        [label for *_, label in rows] if index == 0 else [], fontsize=8
    )
    axis.set_title(APPLICATION_LABELS[application])
  colorbar = figure.colorbar(image, ax=axes[0].tolist(), shrink=0.9)
  colorbar.set_label(f"Median {steps}-step time [ms, log scale]")
  figure.suptitle(
      f"Median {steps}-step time per solver and device (red box = fastest; "
      "– = not measured or timed out)"
  )
  figure.savefig(output_path, dpi=180)
  plt.close(figure)
  print(f"wrote {output_path}")


def _plot_time_vs_error(summary: pd.DataFrame, output_path: Path) -> None:
  """Plots time against error at the largest grid each solver completed."""
  applications = [
      app for app in APPLICATION_LABELS if app in summary["application"].unique()
  ]
  figure, axes = plt.subplots(
      1,
      len(applications),
      figsize=(7.5 * len(applications), 6.5),
      constrained_layout=True,
      squeeze=False,
  )
  handles = {}
  for axis, application in zip(axes[0], applications):
    data = summary[
        (summary["application"] == application)
        & summary["max_abs_error_K"].notna()
    ]
    if data.empty:
      continue
    # The largest grid every plotted solver reached, so points compare alike.
    common = data.groupby(["solver", "device"])["active_cvs"].max().min()
    data = data[data["active_cvs"] == common]
    for solver, device, label, color, _, marker in _series(summary):
      point = data[(data["solver"] == solver) & (data["device"] == device)]
      if point.empty:
        continue
      handle = axis.scatter(
          point["max_abs_error_K"].clip(lower=1e-12),
          point["median_ms"],
          color=color,
          marker=marker,
          s=70,
          edgecolor="black" if device == "gpu" else "none",
      )
      handles[label] = handle
    axis.set(xscale="log", yscale="log")
    axis.set_title(f"{APPLICATION_LABELS[application]} ({common:,} CVs)")
    axis.set_xlabel("Max error against reference [K, log scale]")
    axis.grid(alpha=0.3, which="both")
  axes[0, 0].set_ylabel("Median simulation time [ms, log scale]")
  figure.legend(
      handles.values(),
      handles.keys(),
      loc="outside lower center",
      ncol=min(6, len(handles)),
      fontsize=9,
  )
  figure.suptitle(
      "Time vs accuracy at the largest grid all solvers completed "
      "(lower left is better; outlined = GPU)"
  )
  figure.savefig(output_path, dpi=180)
  plt.close(figure)
  print(f"wrote {output_path}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
  parser.add_argument(
      "--sources",
      default="cpu_gpu,jax_cpu_gpu",
      help="Comma-separated result directories under --root.",
  )
  parser.add_argument("--output-dir", type=Path)
  parser.add_argument("--convergence-threshold", type=float)
  parser.add_argument("--cg-tolerance", type=float)
  parser.add_argument(
      "--heatmap-rooms-per-side",
      default="1,2,4,6,8,10,12,16,20,24,28",
  )
  args = parser.parse_args()
  output_dir = args.output_dir or args.root / "all_solvers"
  output_dir.mkdir(parents=True, exist_ok=True)

  frame = _load(args.root, tuple(args.sources.split(",")))
  for column, value in (
      ("convergence_threshold", args.convergence_threshold),
      ("cg_tolerance", args.cg_tolerance),
  ):
    if value is not None:
      frame = frame[np.isclose(frame[column], value)]
  settings = frame[["convergence_threshold", "cg_tolerance"]].drop_duplicates()
  if len(settings) > 1:
    print(
        "warning: CSVs mix solver tolerances; filter with "
        "--convergence-threshold/--cg-tolerance for a like-for-like plot:\n"
        + settings.to_string(index=False)
    )
  if frame.empty:
    raise ValueError("no samples match the requested tolerances")
  steps = int(frame["simulation_steps"].iloc[0])
  hours = steps * float(frame["time_step_seconds"].iloc[0]) / 3600.0
  summary = _summary(frame)
  summary.to_csv(output_dir / "all_solvers_summary.csv", index=False)
  print(f"wrote {output_dir / 'all_solvers_summary.csv'}")

  reference = summary[
      (summary["solver"] == REFERENCE[0]) & (summary["device"] == REFERENCE[1])
  ][["application", "active_cvs", "median_ms"]].rename(
      columns={"median_ms": "reference_ms"}
  )
  summary = summary.merge(reference, on=["application", "active_cvs"], how="left")
  summary["speedup"] = summary["reference_ms"] / summary["median_ms"]

  duration = f"{steps}-step ({hours:g} h)"
  for logarithmic in (True, False):
    suffix = "" if logarithmic else "_linear"
    scale = "log scale" if logarithmic else "linear scale"
    _plot_lines(
        summary,
        output_dir / f"all_solvers_time{suffix}.png",
        "median_ms",
        f"Median {duration} simulation time [ms, {scale}]",
        f"All solvers, CPU (solid) and GPU (dashed): {duration} simulation "
        "time",
        logarithmic,
    )
  if summary["speedup"].notna().any():
    _plot_lines(
        summary,
        output_dir / "all_solvers_speedup_vs_scipy.png",
        "speedup",
        "Speedup over NumPy/SciPy direct on CPU [log scale]",
        "All solvers: speedup over NumPy/SciPy direct (CPU); above 1 is faster",
        True,
        reference_line=1.0,
    )
  if summary["max_abs_error_K"].notna().any():
    _plot_lines(
        summary,
        output_dir / "all_solvers_accuracy.png",
        "max_abs_error_K",
        "Max |T - T_reference| after the last step [K, log scale]",
        "All solvers: error against the float64 sparse direct reference",
        True,
    )
    _plot_time_vs_error(summary, output_dir / "all_solvers_time_vs_error.png")
  rooms = [int(value) for value in args.heatmap_rooms_per_side.split(",")]
  rooms = [room for room in rooms if room in set(summary["rooms_per_side"])]
  _plot_heatmap(summary, output_dir / "all_solvers_heatmap.png", steps, rooms)


if __name__ == "__main__":
  main()
