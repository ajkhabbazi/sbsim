"""Compares the saved final temperature fields of every solver.

Reads the .npz files that partitioned_enclosure_grid_benchmark.py writes under
<output-dir>/predictions (one reference file and one file per device and
solver for each application and grid) and, without re-running anything:

  * writes prediction_errors.csv with the air and interior mass errors of every
    solver against the float64 sparse direct reference, and the cell of the
    largest air error;
  * plots the maximum air and mass error of every solver per application;
  * draws, for each application, the reference air temperature and the
    absolute air error map of every solver on the largest saved grid.
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


def _load(predictions_dirs: list[Path]) -> tuple[dict, dict]:
  """Returns the reference and solver fields keyed by application and grid."""
  references = {}
  predictions = {}
  for predictions_dir in predictions_dirs:
    for path in sorted(predictions_dir.glob("*/rooms*.npz")):
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
        predictions[key + (device, solver)] = fields
  return references, predictions


def _errors(references: dict, predictions: dict) -> pd.DataFrame:
  rows = []
  for (application, rooms, device, solver), fields in predictions.items():
    reference = references.get((application, rooms))
    if reference is None:
      print(f"no reference for {application} rooms={rooms}; skipped")
      continue
    active = reference["active_mask"]
    error = np.where(
        active, np.abs(fields["temperature"] - reference["temperature"]), 0.0
    )
    row, column = np.unravel_index(np.argmax(error), error.shape)
    result = {
        "application": application,
        "rooms_per_side": rooms,
        "active_cvs": int(active.sum()),
        "device": device,
        "solver": solver,
        "max_abs_error_K": float(error[row, column]),
        "rms_error_K": float(np.sqrt(np.mean(np.square(error[active])))),
        "max_error_row": int(row),
        "max_error_column": int(column),
        "max_abs_mass_error_K": np.nan,
    }
    if "interior_mass_temperature" in reference:
      mass_mask = reference["interior_mass_mask"]
      result["max_abs_mass_error_K"] = float(
          np.max(
              np.abs(
                  fields["interior_mass_temperature"]
                  - reference["interior_mass_temperature"]
              )[mass_mask]
          )
      )
    rows.append(result)
  return pd.DataFrame(rows)


def _ordered_series(frame: pd.DataFrame) -> list[tuple[str, str]]:
  present = set(zip(frame["solver"], frame["device"]))
  return [
      (solver, device)
      for solver in SOLVER_STYLES
      for device in DEVICE_STYLES
      if (solver, device) in present
  ]


def _plot_errors(frame: pd.DataFrame, output_path: Path) -> None:
  applications = [a for a in APPLICATION_LABELS if a in set(frame.application)]
  metrics = [("max_abs_error_K", "Air: max |T - T_ref| [K]")]
  if frame["max_abs_mass_error_K"].notna().any():
    metrics.append(("max_abs_mass_error_K", "Interior mass: max |T - T_ref| [K]"))
  figure, axes = plt.subplots(
      len(metrics),
      len(applications),
      figsize=(5.2 * len(applications), 3.8 * len(metrics)),
      squeeze=False,
  )
  for column, application in enumerate(applications):
    subset = frame[frame.application == application]
    for row, (metric, label) in enumerate(metrics):
      axis = axes[row][column]
      if subset[metric].isna().all():
        axis.text(
            0.5, 0.5, "no interior mass", ha="center", va="center",
            transform=axis.transAxes, color="#666666",
        )
        axis.set_axis_off()
        continue
      for solver, device in _ordered_series(subset):
        series = subset[
            (subset.solver == solver) & (subset.device == device)
        ].sort_values("active_cvs")
        name, color = SOLVER_STYLES[solver]
        linestyle, marker = DEVICE_STYLES[device]
        axis.plot(
            series["active_cvs"],
            series[metric].clip(lower=ERROR_FLOOR_K),
            linestyle=linestyle,
            marker=marker,
            markersize=6,
            linewidth=1.5,
            color=color,
            label=f"{name} ({device.upper()})",
        )
      axis.set_xscale("log")
      axis.set_yscale("log")
      axis.grid(True, which="major", color="#DDDDDD", linewidth=0.6)
      axis.set_xlabel("Active control volumes [log scale]")
      if column == 0 or axes[row][0].get_ylabel() == "":
        axis.set_ylabel(label)
      if row == 0:
        axis.set_title(APPLICATION_LABELS[application])
  handles, labels = {}, {}
  for axis in axes.flat:
    for handle, label in zip(*axis.get_legend_handles_labels()):
      handles.setdefault(label, handle)
  figure.legend(
      handles.values(), handles.keys(), loc="lower center",
      ncol=min(5, len(handles)), frameon=False,
  )
  figure.suptitle(
      "Final-step error against the float64 sparse direct reference"
      " (same initial state for every solver)"
  )
  figure.tight_layout(rect=(0, 0.04 + 0.03 * (len(handles) // 5), 1, 0.97))
  figure.savefig(output_path, dpi=150)
  plt.close(figure)
  print(f"wrote {output_path}")


def _plot_maps(
    application: str,
    rooms: int,
    reference: dict,
    predictions: dict,
    output_path: Path,
) -> None:
  series = [
      (solver, device)
      for solver in SOLVER_STYLES
      for device in DEVICE_STYLES
      if (application, rooms, device, solver) in predictions
  ]
  active = reference["active_mask"]
  errors = {
      (solver, device): np.ma.masked_where(
          ~active,
          np.abs(
              predictions[(application, rooms, device, solver)]["temperature"]
              - reference["temperature"]
          ),
      )
      for solver, device in series
  }
  maximum = max(float(error.max()) for error in errors.values())
  norm = mcolors.LogNorm(
      vmin=max(maximum * 1e-6, ERROR_FLOOR_K), vmax=max(maximum, 1e-11)
  )
  panels = len(series) + 1
  columns = min(4, panels)
  rows = int(np.ceil(panels / columns))
  figure, axes = plt.subplots(
      rows, columns, figsize=(3.6 * columns, 3.4 * rows), squeeze=False
  )
  for axis in axes.flat:
    axis.set_axis_off()

  axis = axes[0][0]
  image = axis.imshow(
      np.ma.masked_where(~active, reference["temperature"]), cmap="viridis"
  )
  axis.set_title("Reference air temperature [K]", fontsize=9)
  figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)

  cmap = plt.get_cmap("Greys").copy()
  cmap.set_bad("#F4F4F4")
  for index, (solver, device) in enumerate(series, start=1):
    axis = axes[index // columns][index % columns]
    error = errors[(solver, device)]
    image = axis.imshow(
        np.ma.masked_less_equal(error, 0.0).filled(norm.vmin),
        cmap="magma_r",
        norm=norm,
    )
    row, column = np.unravel_index(np.argmax(error.filled(0.0)), error.shape)
    axis.plot(column, row, marker="x", color="#1F77B4", markersize=7)
    axis.set_title(
        f"{SOLVER_STYLES[solver][0]} ({device.upper()})\n"
        f"max {float(error.max()):.2e} K",
        fontsize=9,
    )
  figure.colorbar(
      image, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02,
      label="|T - T_ref| [K] (log)",
  )
  figure.suptitle(
      f"{APPLICATION_LABELS.get(application, application)}:"
      f" {rooms}x{rooms} rooms, final step (x = largest error)"
  )
  figure.savefig(output_path, dpi=130)
  plt.close(figure)
  print(f"wrote {output_path}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--predictions-dir",
      type=Path,
      action="append",
      required=True,
      help="Repeatable; usually <output-dir>/cpu/predictions and .../gpu/...",
  )
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
      "--map-rooms-per-side",
      type=int,
      help="Grid for the error maps; defaults to the largest saved grid.",
  )
  args = parser.parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)

  references, predictions = _load(
      [path for path in args.predictions_dir if path.exists()]
  )
  frame = _errors(references, predictions)
  if frame.empty:
    print("no saved predictions; nothing to plot")
    return
  frame = frame.sort_values(
      ["application", "rooms_per_side", "solver", "device"]
  )
  csv_path = args.output_dir / "prediction_errors.csv"
  frame.to_csv(csv_path, index=False)
  print(f"wrote {csv_path}")
  _plot_errors(frame, args.output_dir / "prediction_errors.png")

  for application in frame.application.unique():
    saved_rooms = sorted(
        rooms
        for app, rooms in references
        if app == application
        and any(key[:2] == (app, rooms) for key in predictions)
    )
    rooms = args.map_rooms_per_side or saved_rooms[-1]
    if (application, rooms) not in references:
      continue
    _plot_maps(
        application,
        rooms,
        references[(application, rooms)],
        predictions,
        args.output_dir / f"error_maps_{application}_rooms{rooms:02d}.png",
    )

  summary = frame.groupby(["application", "solver", "device"])[
      ["max_abs_error_K", "max_abs_mass_error_K"]
  ].max()
  with pd.option_context("display.float_format", "{:.3g}".format):
    print(summary.to_string())


if __name__ == "__main__":
  main()
