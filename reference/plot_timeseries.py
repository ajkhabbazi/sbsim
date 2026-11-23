"""Plots the one-day solver agreement saved by partitioned_enclosure_timeseries.py.

The solvers agree to far below the daily temperature swing, so their
trajectories overlap exactly when drawn in kelvin. Each figure therefore shows
the temperatures once, from the reference, and compares solvers through their
deviation from it:

  * overview: forcing, the reference room-mean trajectories with the field
    5-95% band, each solver's deviation of the field-mean temperature from
    the reference mean, and the max and RMS field error over time;
  * room error heatmap: max |T - T_ref| for every room and step (time on x,
    room on y), one panel per solver, so where and when the error grows can be
    read off for every room at once;
  * error snapshots: the signed error field of every solver at a few times,
    the time series counterpart of the existing final-step error maps.

It also writes timeseries_errors.csv with the day-maximum, time-mean, and
final errors of every solver.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from reference.plot_all_solvers import APPLICATION_LABELS
from reference.plot_all_solvers import DEVICE_STYLES
from reference.plot_all_solvers import SOLVER_STYLES

FILE_PATTERN = re.compile(r"rooms(?P<rooms>\d+)_(?P<name>.+)\.npz")
# Errors below this are drawn at the floor of the log axes.
ERROR_FLOOR_K = 1e-12
SNAPSHOT_HOURS = (3.0, 9.0, 15.0, 21.0, 24.0)
GRID_COLOR = "#DDDDDD"
REFERENCE_COLOR = "#555555"


def _load(timeseries_dirs: list[Path]) -> tuple[dict, dict]:
  """Returns the reference and solver series keyed by application and grid."""
  references = {}
  runs = {}
  for timeseries_dir in timeseries_dirs:
    for path in sorted(timeseries_dir.glob("*/rooms*.npz")):
      match = FILE_PATTERN.fullmatch(path.name)
      if match is None:
        continue
      key = (path.parent.name, int(match["rooms"]))
      with np.load(path) as data:
        fields = {name: data[name] for name in data.files}
      if match["name"] == "reference":
        references[key] = fields
      else:
        device, solver = match["name"].split("_", 1)
        runs[key + (device, solver)] = fields
  return references, runs


def _series(runs: dict, application: str, rooms: int) -> list[tuple[str, str]]:
  return [
      (solver, device)
      for solver in SOLVER_STYLES
      for device in DEVICE_STYLES
      if (application, rooms, device, solver) in runs
  ]


def _label(solver: str, device: str) -> str:
  return f"{SOLVER_STYLES[solver][0]} ({device.upper()})"


def _style_axis(axis) -> None:
  axis.grid(True, which="major", color=GRID_COLOR, linewidth=0.6)
  axis.set_xlim(0.0, None)
  axis.set_xticks(np.arange(0, 25, 3))


def _plot_overview(
    application: str, rooms: int, reference: dict, runs: dict, path: Path
) -> None:
  hours = reference["time_seconds"] / 3600.0
  series = _series(runs, application, rooms)
  has_mass = "max_abs_mass_error" in runs[(application, rooms) + series[0][::-1]]
  panels = [
      "ambient", "power", "temperature", "mean_deviation", "air_error",
  ]
  if has_mass:
    panels.append("mass_error")
  heights = {"ambient": 0.8, "power": 0.8}
  figure, axes = plt.subplots(
      len(panels),
      1,
      sharex=True,
      figsize=(10.0, 1.9 * sum(heights.get(p, 1.4) for p in panels)),
      gridspec_kw={"height_ratios": [heights.get(p, 1.4) for p in panels]},
  )
  axes = dict(zip(panels, axes))
  room_count = reference["room_mean_temperature"].shape[1]
  room_colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, room_count))

  # Forcing is evaluated at the start of each step and held over it.
  axis = axes["ambient"]
  axis.step(
      hours[:-1], reference["ambient_temperature"], where="post",
      color=REFERENCE_COLOR, linewidth=1.5,
  )
  axis.set_ylabel("Ambient [K]")
  axis = axes["power"]
  for room in range(room_count):
    axis.step(
        hours[:-1], reference["room_power"][:, room], where="post",
        color=room_colors[room], linewidth=1.2,
    )
  axis.set_ylabel("HVAC [W]")

  axis = axes["temperature"]
  percentiles = reference["temperature_percentiles"]
  # Diffuser CVs run far colder than the room air, so the band excludes the
  # extreme percentiles.
  axis.fill_between(
      hours, percentiles[:, 1], percentiles[:, -2], color="#E8E8E8",
      label="Reference field 5-95%",
  )
  for room in range(room_count):
    axis.plot(
        hours, reference["room_mean_temperature"][:, room],
        color=room_colors[room], linewidth=1.2,
        label="Reference room means (colors as HVAC)" if room == 0 else None,
    )
  axis.plot(
      hours, reference["mean_temperature"], color="black", linewidth=2.0,
      label="Reference field mean",
  )
  axis.set_ylabel("Air [K]")
  axis.legend(loc="upper left", fontsize=8, frameon=False)

  for solver, device in series:
    run = runs[(application, rooms, device, solver)]
    _, color = SOLVER_STYLES[solver]
    linestyle, _ = DEVICE_STYLES[device]
    style = dict(color=color, linestyle=linestyle, linewidth=1.5)
    axes["mean_deviation"].plot(
        hours, run["mean_temperature"] - reference["mean_temperature"],
        label=_label(solver, device), **style,
    )
    axes["air_error"].plot(
        hours, np.clip(run["max_abs_error"], ERROR_FLOOR_K, None), **style
    )
    axes["air_error"].plot(
        hours, np.clip(run["rms_error"], ERROR_FLOOR_K, None),
        alpha=0.45, **{**style, "linewidth": 1.0},
    )
    if has_mass:
      axes["mass_error"].plot(
          hours, np.clip(run["max_abs_mass_error"], ERROR_FLOOR_K, None),
          **style,
      )
  axis = axes["mean_deviation"]
  axis.axhline(0.0, color=REFERENCE_COLOR, linewidth=0.8)
  axis.set_ylabel("Mean T - mean T_ref [K]")
  axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
  axis = axes["air_error"]
  axis.set_yscale("log")
  axis.set_ylabel("Air |T - T_ref| [K]")
  axis.text(
      0.01, 0.92, "solid: field max, faint: field RMS",
      transform=axis.transAxes, fontsize=8, color="#666666", va="top",
  )
  if has_mass:
    axes["mass_error"].set_yscale("log")
    axes["mass_error"].set_ylabel("Mass max |T - T_ref| [K]")
  for axis in axes.values():
    _style_axis(axis)
  axes[panels[-1]].set_xlabel("Time [h]")

  handles, labels = axes["mean_deviation"].get_legend_handles_labels()
  figure.legend(
      handles, labels, loc="lower center", ncol=min(4, len(handles)),
      frameon=False, fontsize=9,
  )
  figure.suptitle(
      f"{APPLICATION_LABELS.get(application, application)}: {rooms}x{rooms}"
      " rooms, one day against the float64 sparse direct reference"
  )
  figure.tight_layout(
      rect=(0, 0.03 + 0.025 * ((len(handles) - 1) // 4 + 1), 1, 0.98)
  )
  figure.savefig(path, dpi=140)
  plt.close(figure)
  print(f"wrote {path}")


def _plot_room_heatmaps(
    application: str, rooms: int, reference: dict, runs: dict, path: Path
) -> None:
  hours = reference["time_seconds"] / 3600.0
  series = _series(runs, application, rooms)
  errors = {
      key: np.clip(
          runs[(application, rooms) + key[::-1]]["room_max_abs_error"],
          ERROR_FLOOR_K, None,
      )
      for key in series
  }
  maximum = max(float(error.max()) for error in errors.values())
  norm = mcolors.LogNorm(
      vmin=max(maximum * 1e-6, ERROR_FLOOR_K), vmax=max(maximum, 1e-11)
  )
  room_count = reference["room_mean_temperature"].shape[1]
  columns = min(2, len(series))
  rows = int(np.ceil(len(series) / columns))
  figure, axes = plt.subplots(
      rows, columns,
      figsize=(6.4 * columns, 0.9 + max(1.6, 0.18 * room_count) * rows),
      squeeze=False, sharex=True, sharey=True, layout="constrained",
  )
  for axis in axes.flat:
    axis.set_visible(False)
  extent = (hours[0], hours[-1], room_count - 0.5, -0.5)
  for index, key in enumerate(series):
    axis = axes[index // columns][index % columns]
    axis.set_visible(True)
    image = axis.imshow(
        errors[key].T, aspect="auto", cmap="magma_r", norm=norm,
        extent=extent, interpolation="nearest",
    )
    axis.set_title(
        f"{_label(*key)}: max {float(errors[key].max()):.2e} K", fontsize=9
    )
    axis.set_xticks(np.arange(0, 25, 3))
    axis.set_yticks(np.arange(room_count))
    if index % columns == 0:
      axis.set_ylabel("Room (row-major)")
    if index // columns == rows - 1 or index + columns >= len(series):
      axis.set_xlabel("Time [h]")
  figure.colorbar(
      image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02,
      label="Room max |T - T_ref| [K] (log)",
  )
  figure.suptitle(
      f"{APPLICATION_LABELS.get(application, application)}: {rooms}x{rooms}"
      " rooms, per-room air error over the day"
  )
  figure.savefig(path, dpi=140)
  plt.close(figure)
  print(f"wrote {path}")


def _plot_snapshots(
    application: str, rooms: int, reference: dict, runs: dict, path: Path
) -> None:
  field_hours = reference["field_steps"] * (
      reference["time_seconds"][1] / 3600.0
  )
  snapshot_indices = sorted(
      {int(np.argmin(np.abs(field_hours - hour))) for hour in SNAPSHOT_HOURS}
  )
  series = _series(runs, application, rooms)
  active = reference["active_mask"]
  columns = len(snapshot_indices) + 1
  figure, axes = plt.subplots(
      len(series) + 1, columns,
      figsize=(2.4 * columns, 2.4 * (len(series) + 1)),
      squeeze=False,
      layout="constrained",
      gridspec_kw={"width_ratios": [0.7] + [1.0] * (columns - 1)},
  )
  for axis in axes.flat:
    axis.set_xticks([])
    axis.set_yticks([])
  axes[0][0].set_axis_off()

  # Top row: the reference temperature field at the snapshot times.
  temperature_fields = reference["temperature_fields"][snapshot_indices]
  temperature_norm = mcolors.Normalize(
      vmin=float(temperature_fields[:, active].min()),
      vmax=float(temperature_fields[:, active].max()),
  )
  for column, index in enumerate(snapshot_indices, start=1):
    axis = axes[0][column]
    image = axis.imshow(
        np.ma.masked_where(~active, reference["temperature_fields"][index]),
        cmap="viridis", norm=temperature_norm,
    )
    axis.set_title(f"t = {field_hours[index]:.1f} h", fontsize=9)
  figure.colorbar(image, ax=axes[0].tolist(), fraction=0.03, pad=0.01,
                  label="T_ref [K]")

  # One row per solver; signed error with a symmetric scale per row, since
  # float32 and float64 solvers differ by orders of magnitude.
  cmap = plt.get_cmap("RdBu_r").copy()
  cmap.set_bad("#F4F4F4")
  for row, (solver, device) in enumerate(series, start=1):
    run = runs[(application, rooms, device, solver)]
    error_fields = run["error_fields"][snapshot_indices].astype(np.float64)
    limit = max(float(np.abs(error_fields).max()), ERROR_FLOOR_K)
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    axis = axes[row][0]
    axis.set_axis_off()
    axis.text(
        1.0, 0.5, _label(solver, device).replace(" (", "\n("),
        ha="right", va="center", fontsize=9, transform=axis.transAxes,
    )
    for column, index in enumerate(snapshot_indices, start=1):
      image = axes[row][column].imshow(
          np.ma.masked_where(~active, run["error_fields"][index]),
          cmap=cmap, norm=norm,
      )
    colorbar = figure.colorbar(
        image, ax=axes[row].tolist(), fraction=0.03, pad=0.01,
        label="T - T_ref [K]",
    )
    colorbar.formatter.set_powerlimits((-2, 2))
    colorbar.update_ticks()
  figure.suptitle(
      f"{APPLICATION_LABELS.get(application, application)}: {rooms}x{rooms}"
      " rooms, signed air error at selected times"
  )
  figure.savefig(path, dpi=120)
  plt.close(figure)
  print(f"wrote {path}")


def _plot_room_trajectories(
    application: str, rooms: int, reference: dict, runs: dict, path: Path
) -> None:
  """One row per room: every solver's room-mean air temperature.

  The left column overlays the trajectories in kelvin; they coincide at that
  scale, so the right column shows each solver minus the reference.
  """
  hours = reference["time_seconds"] / 3600.0
  series = _series(runs, application, rooms)
  room_count = reference["room_mean_temperature"].shape[1]
  figure, axes = plt.subplots(
      room_count, 2,
      figsize=(12.0, 1.7 * room_count + 1.2),
      sharex=True, squeeze=False, layout="constrained",
      gridspec_kw={"width_ratios": [1.0, 1.0]},
  )
  for room in range(room_count):
    power = reference["room_power"][:, room]
    # Steps where this room's HVAC switches on or off.
    edges = np.flatnonzero(np.diff(np.r_[0, power != 0.0, 0]))
    for axis in axes[room]:
      # Shades the hours this room's HVAC is on; the same for every solver.
      for start, stop in zip(edges[::2], edges[1::2]):
        axis.axvspan(
            hours[start], hours[stop], color="#EEF3FA", linewidth=0,
            zorder=0,
        )
      _style_axis(axis)
    axis_temperature, axis_deviation = axes[room]
    axis_temperature.plot(
        hours, reference["room_mean_temperature"][:, room],
        color="black", linewidth=3.0, alpha=0.35, label="Reference",
    )
    for solver, device in series:
      run = runs[(application, rooms, device, solver)]
      _, color = SOLVER_STYLES[solver]
      linestyle, _ = DEVICE_STYLES[device]
      style = dict(color=color, linestyle=linestyle, linewidth=1.2)
      axis_temperature.plot(
          hours, run["room_mean_temperature"][:, room],
          label=_label(solver, device), **style,
      )
      axis_deviation.plot(
          hours,
          run["room_mean_temperature"][:, room]
          - reference["room_mean_temperature"][:, room],
          **style,
      )
    axis_deviation.axhline(0.0, color=REFERENCE_COLOR, linewidth=0.8)
    axis_deviation.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    peak = power.min()
    axis_temperature.set_ylabel(
        f"Room {room}\n{peak:.0f} W" if peak else f"Room {room}\nno HVAC",
        fontsize=9,
    )
  axes[0][0].set_title("Room-mean air temperature [K]", fontsize=10)
  axes[0][1].set_title("Room mean - reference room mean [K]", fontsize=10)
  axes[-1][0].set_xlabel("Time [h]")
  axes[-1][1].set_xlabel("Time [h]")
  handles, labels = axes[0][0].get_legend_handles_labels()
  figure.legend(
      handles, labels, loc="outside lower center",
      ncol=min(5, len(handles)), frameon=False, fontsize=9,
  )
  figure.suptitle(
      f"{APPLICATION_LABELS.get(application, application)}: {rooms}x{rooms}"
      " rooms, room-mean air temperature by solver (same forcing for all;"
      " shaded = this room's HVAC on)",
      fontsize=11,
  )
  figure.savefig(path, dpi=130)
  plt.close(figure)
  print(f"wrote {path}")


def _summary(references: dict, runs: dict) -> pd.DataFrame:
  rows = []
  for (application, rooms, device, solver), run in runs.items():
    if (application, rooms) not in references:
      print(f"no reference for {application} rooms={rooms}; skipped")
      continue
    row = {
        "application": application,
        "rooms_per_side": rooms,
        "device": device,
        "solver": solver,
        "simulation_steps": len(run["step_seconds"]),
        "elapsed_s": float(np.sum(run["step_seconds"])),
        "mean_outer_iterations": float(np.nanmean(run["outer_iterations"])),
        "day_max_abs_error_K": float(run["max_abs_error"].max()),
        "time_mean_rms_error_K": float(run["rms_error"][1:].mean()),
        "final_max_abs_error_K": float(run["max_abs_error"][-1]),
        "day_max_mean_deviation_K": float(
            np.abs(
                run["mean_temperature"]
                - references[(application, rooms)]["mean_temperature"]
            ).max()
        ),
        "day_max_abs_mass_error_K": np.nan,
    }
    if "max_abs_mass_error" in run:
      row["day_max_abs_mass_error_K"] = float(run["max_abs_mass_error"].max())
    rows.append(row)
  return pd.DataFrame(rows)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--timeseries-dir",
      type=Path,
      action="append",
      required=True,
      help="Repeatable; usually <output-dir>/cpu/timeseries and .../gpu/...",
  )
  parser.add_argument("--output-dir", type=Path, required=True)
  args = parser.parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)

  references, runs = _load(
      [path for path in args.timeseries_dir if path.exists()]
  )
  frame = _summary(references, runs)
  if frame.empty:
    print("no saved time series; nothing to plot")
    return
  frame = frame.sort_values(
      ["application", "rooms_per_side", "solver", "device"]
  )
  csv_path = args.output_dir / "timeseries_errors.csv"
  frame.to_csv(csv_path, index=False)
  print(f"wrote {csv_path}")

  for application, rooms in sorted(set(zip(frame.application, frame.rooms_per_side))):
    reference = references[(application, rooms)]
    suffix = f"{application}_rooms{rooms:02d}.png"
    _plot_overview(
        application, rooms, reference, runs,
        args.output_dir / f"overview_{suffix}",
    )
    _plot_room_trajectories(
        application, rooms, reference, runs,
        args.output_dir / f"room_trajectories_{suffix}",
    )
    _plot_room_heatmaps(
        application, rooms, reference, runs,
        args.output_dir / f"room_error_heatmap_{suffix}",
    )
    _plot_snapshots(
        application, rooms, reference, runs,
        args.output_dir / f"error_snapshots_{suffix}",
    )

  with pd.option_context("display.float_format", "{:.3g}".format):
    print(frame.drop(columns=["simulation_steps"]).to_string(index=False))


if __name__ == "__main__":
  main()
