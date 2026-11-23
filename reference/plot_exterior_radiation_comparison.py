"""Adds the paired exterior-radiation timings to the historical solver plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from reference.plot_all_solvers import APPLICATION_LABELS
from reference.plot_all_solvers import SOLVER_STYLES
from reference.plot_all_solvers import _series
from reference.plot_all_solvers import _summary


BENCHMARK_OUTPUTS = Path(__file__).resolve().parent / "benchmark_outputs"
HISTORICAL = (
    BENCHMARK_OUTPUTS
    / "partitioned_enclosure_sensitivity"
    / "all_solvers_uniform"
    / "all_solvers"
)
EXTERIOR_SAMPLES = BENCHMARK_OUTPUTS / "exterior_radiation_cpu" / "samples.csv"
EXTERIOR_PANELS = {
    "base_rad": "Glazed envelope · no interior mass",
    "base_rad_interior_mass": "Glazed envelope · with interior mass",
}


def _exterior_summary(path: Path) -> pd.DataFrame:
  samples = pd.read_csv(path)
  if (
      not samples["steps"].eq(12).all()
      or not samples["time_step_seconds"].eq(300).all()
  ):
    raise ValueError("The exterior comparison requires 12 steps of 300 s.")
  if (
      not np.isfinite(samples["trajectory_ms"]).all()
      or samples["trajectory_ms"].le(0).any()
  ):
    raise ValueError("The exterior sample times must be finite and positive.")
  return (
      samples.groupby(
          ["application", "solver", "exterior", "active_cvs"], as_index=False
      )["trajectory_ms"]
      .median()
      .rename(columns={"trajectory_ms": "median_ms"})
  )


def _style_axes(axis, title):
  axis.set_title(title)
  axis.set_xscale("log")
  axis.set_yscale("log")
  axis.set_xlabel("Active control volumes")
  axis.grid(alpha=0.28, which="both")


def _scaled_exterior_summary(path: Path, repeats: int = 3) -> pd.DataFrame:
  samples = pd.read_csv(path)
  if (
      not samples["simulation_steps"].eq(12).all()
      or not samples["time_step_seconds"].eq(300).all()
  ):
    raise ValueError("The scaling comparison requires 12 steps of 300 s.")
  samples = samples[
      samples["device"].eq("cpu")
      & samples["application"].isin(
          f"{application}_exterior" for application in EXTERIOR_PANELS
      )
  ]
  if samples.empty:
    raise ValueError("No completed exterior CPU samples have been saved.")
  summary = _summary(samples)
  summary = summary[summary["samples"].eq(repeats)].copy()
  if summary.empty:
    raise ValueError("No exterior CPU cases have completed every repeat.")
  summary["application"] = summary["application"].str.removesuffix("_exterior")
  summary["exterior"] = True
  return summary


def _plot(
    history: pd.DataFrame,
    exterior: pd.DataFrame,
    output: Path,
    *,
    scaled: bool = False,
) -> None:
  figure, axes = plt.subplots(2, 3, figsize=(20, 10), layout="constrained")
  figure.suptitle("Median 12-step (1 h) simulation time", fontsize=17)
  axes[0, 0].set_ylabel("Historical benchmark [ms, log scale]")
  axes[1, 0].set_ylabel(
      "Exterior CPU benchmark [ms, log scale]"
      if scaled
      else "Paired CPU benchmark [ms, log scale]"
  )
  axes[1, 1].sharey(axes[1, 0])

  for axis, (application, title) in zip(axes[0], APPLICATION_LABELS.items()):
    data = history[history["application"] == application]
    for solver, device, _, color, linestyle, marker in _series(history):
      points = data[
          (data["solver"] == solver) & (data["device"] == device)
      ].sort_values("active_cvs")
      if not points.empty:
        axis.plot(
            points["active_cvs"],
            points["median_ms"],
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=4,
            linewidth=1.6,
        )
    _style_axes(axis, title)

  for axis, (application, title) in zip(axes[1, :2], EXTERIOR_PANELS.items()):
    data = exterior[exterior["application"] == application]
    for solver, (_, color) in SOLVER_STYLES.items():
      for enabled in (False, True):
        points = data[
            (data["solver"] == solver) & (data["exterior"] == enabled)
        ].sort_values("active_cvs")
        if not points.empty:
          axis.plot(
              points["active_cvs"],
              points["median_ms"],
              color=color,
              linestyle="-" if enabled else ":",
              marker="o",
              markerfacecolor=color if enabled else "white",
              markersize=5,
              linewidth=1.6,
          )
    _style_axes(axis, title)
    sizes = sorted(data["active_cvs"].unique())
    if len(sizes) < 5:
      axis.set_xticks(sizes, [f"{size:,}" for size in sizes])
      axis.tick_params(axis="x", which="minor", labelbottom=False)

  legend_axis = axes[1, 2]
  legend_axis.set_axis_off()
  present_solvers = set(history["solver"]) | set(exterior["solver"])
  solver_handles = [
      Line2D([], [], color=color, linewidth=2, label=label)
      for solver, (label, color) in SOLVER_STYLES.items()
      if solver in present_solvers
  ]
  solver_legend = legend_axis.legend(
      handles=solver_handles,
      loc="upper left",
      frameon=False,
      ncol=2,
      title="Solver colors",
      borderaxespad=0,
  )
  legend_axis.add_artist(solver_legend)
  legend_axis.text(
      0,
      0.60,
      "Top row · historical CPU/GPU results",
      transform=legend_axis.transAxes,
      fontweight="bold",
  )
  legend_axis.text(
      0,
      0.30,
      (
          "Bottom row · new exterior radiation scaling (CPU)"
          if scaled
          else "Bottom row · exterior radiation ON/OFF (CPU)"
      ),
      transform=legend_axis.transAxes,
      fontweight="bold",
  )
  legend_sections = (
      (0.54, [("-", "o", "#333333", "CPU"), ("--", "s", "#333333", "GPU")]),
      (0.24, [("-", "o", "#333333", "ON"), (":", "o", "white", "OFF")]),
  )
  for y, styles in legend_sections[:1] if scaled else legend_sections:
    handles = [
        Line2D(
            [],
            [],
            color="#333333",
            linestyle=line,
            marker=marker,
            markerfacecolor=face,
            label=label,
        )
        for line, marker, face, label in styles
    ]
    legend = legend_axis.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0, y),
        frameon=False,
        ncol=2,
        borderaxespad=0,
    )
    legend_axis.add_artist(legend)
  legend_axis.text(
      0,
      0.09,
      (
          "New exterior runs use glazing on the perimeter.\n"
          "Each point requires all configured repeats."
          if scaled
          else (
              "Paired runs use identical glazing and 3 repeats per point.\n"
              "Historical and paired runs used different benchmark settings."
          )
      ),
      transform=legend_axis.transAxes,
      fontsize=9,
      va="top",
  )
  temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
  figure.savefig(temporary, dpi=180)
  temporary.replace(output)
  plt.close(figure)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--historical-summary",
      type=Path,
      default=HISTORICAL / "all_solvers_summary.csv",
  )
  parser.add_argument("--exterior-samples", type=Path, default=EXTERIOR_SAMPLES)
  parser.add_argument("--scaled-cpu-csv", type=Path)
  parser.add_argument("--completed-repeats", type=int, default=3)
  parser.add_argument(
      "--output",
      type=Path,
      default=HISTORICAL / "all_solvers_time_with_exterior_radiation.png",
  )
  args = parser.parse_args()
  history = pd.read_csv(args.historical_summary)
  scaled = args.scaled_cpu_csv is not None
  exterior = (
      _scaled_exterior_summary(args.scaled_cpu_csv, args.completed_repeats)
      if scaled
      else _exterior_summary(args.exterior_samples)
  )
  _plot(history, exterior, args.output, scaled=scaled)
  print(f"wrote {args.output}")


if __name__ == "__main__":
  main()
