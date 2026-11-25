"""Reproducible diagnostics for the saved full/grouped 100-zone physics replay.

This reads, but never modifies, the source benchmark directory. The fair thermal
comparison is grouped-linear versus full-linear with identical recorded HVAC
power. Independent closed-loop timing rollouts are reported separately.
"""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


COMPARISON_LABELS = (
    'full_linear_minus_full_nonlinear',
    'grouped_linear_minus_full_nonlinear',
    'grouped_linear_minus_full_linear',
    'grouped_nonlinear_minus_grouped_linear',
  'grouped_nonlinear_minus_full_nonlinear',
)


def temperature_metrics(difference):
  """Summarize a nonempty time × zone array without losing its sign."""
  delta = np.asarray(difference, dtype=np.float64)
  if delta.ndim != 2 or not delta.size:
    raise ValueError('Temperature comparison requires a nonempty time × zone array')
  if not np.isfinite(delta).all():
    raise ValueError('Temperature comparison contains a nonfinite value')
  absolute = np.abs(delta)
  maximum_time, maximum_zone = np.unravel_index(absolute.argmax(), delta.shape)
  metrics = dict(
      bias_k=float(delta.mean()),
      mae_k=float(absolute.mean()),
      rmse_k=float(np.sqrt(np.mean(delta**2))),
      p50_absolute_k=float(np.percentile(absolute, 50)),
      p90_absolute_k=float(np.percentile(absolute, 90)),
      p95_absolute_k=float(np.percentile(absolute, 95)),
      p99_absolute_k=float(np.percentile(absolute, 99)),
      max_absolute_k=float(absolute[maximum_time, maximum_zone]),
      maximum_time_index=int(maximum_time),
      maximum_zone_index=int(maximum_zone),
      negative_percentage=float(100 * np.mean(delta < 0)),
      sample_count=int(delta.size),
  )
  for threshold in (0.05, 0.1, 0.2, 0.4):
    key = str(threshold).replace('.', '_')
    metrics[f'absolute_over_{key}_percentage'] = float(
        100 * np.mean(absolute > threshold)
    )
  return metrics


def _digest(path):
  return sha256(path.read_bytes()).hexdigest()


def _time_rows(difference, forcing, times, group):
  rows = []
  for label in dict.fromkeys(group):
    selected = np.asarray(group) == label
    delta = difference[selected]
    absolute = np.abs(delta)
    rows.append(dict(
        interval=label,
        samples=int(delta.size),
        mean_time_mae_k=float(absolute.mean(axis=1).mean()),
        mean_time_cross_zone_p95_k=float(
            np.percentile(absolute, 95, axis=1).mean()
        ),
        maximum_time_cross_zone_p95_k=float(
            np.percentile(absolute, 95, axis=1).max()
        ),
        mean_hvac_w=float(forcing[selected].mean()),
        **temperature_metrics(delta),
    ))
  return rows


def _matched_physics_timing(summary):
  metrics = {row['variant']: row for row in summary['physics']['metrics']}
  full = metrics['full_linear']
  grouped = metrics['grouped_jax']
  if full['method'] != 'sparse_lu' or grouped['method'] != 'sparse_lu':
    raise ValueError('Matched timing requires sparse_lu for both linear replays')
  return dict(
      full_linear_ms_per_step=full['mean_step_ms'],
      grouped_linear_ms_per_step=grouped['mean_step_ms'],
      matched_sparse_lu_speedup=full['mean_step_ms'] / grouped['mean_step_ms'],
      full_linear_solve_counts=full['linear_solves_per_step'],
      grouped_linear_solve_counts=grouped['linear_solves_per_step'],
  )


def _closed_loop_feedback(root, times, zones, full_physics, grouped_physics):
  paths = [root / 'timing' / f'100-{name}' / 'rollout.npz'
           for name in ('full_jax', 'grouped_jax')]
  if not all(path.is_file() for path in paths):
    return None
  with np.load(paths[0], allow_pickle=False) as data:
    full = data['zone_c'].astype(np.float64)
  with np.load(paths[1], allow_pickle=False) as data:
    grouped = data['zone_c'].astype(np.float64)
  if full.shape != grouped.shape or full.shape != (len(times), len(zones)):
    raise ValueError('Closed-loop and physics arrays have different shapes')
  delta = grouped - full
  time_metrics = _time_rows(delta, np.zeros_like(delta), times, times.strftime('%Y-%m-%d'))
  for row in time_metrics:
    del row['mean_hvac_w']  # These rollout files do not store HVAC heat.
  return dict(
      description=(
          'Independent native-VAV timing rollouts: full nonlinear CG versus '
          'grouped linear dense. This is not an identical-HVAC replay.'
      ),
      comparison=temperature_metrics(delta),
      timing_rollout_minus_same_model_physics=dict(
          full=temperature_metrics(full - full_physics),
          grouped=temperature_metrics(grouped - grouped_physics),
      ),
      days=time_metrics,
  )


def analyze_saved(root, nonlinear_path=None):
  """Calculate every report statistic from source artifacts."""
  root = Path(root)
  summary_path = root / 'summary.json'
  physics_path = root / 'physics.npz'
  manifest_path = root / 'models' / 'zone100' / 'floorplan.json'
  summary = json.loads(summary_path.read_text(encoding='utf-8'))
  manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
  auxiliary = (
      dict(
          relative_path=str(Path(nonlinear_path).resolve().relative_to(root.parent.resolve()))
          if _is_relative_to(Path(nonlinear_path).resolve(), root.parent.resolve())
          else str(Path(nonlinear_path).resolve()),
          replay_npz_sha256=_digest(nonlinear_path),
      )
      if nonlinear_path is not None else None
  )
  with np.load(physics_path, allow_pickle=False) as data:
    required = ('full_jax', 'full_linear', 'grouped_jax', 'interval_end',
                'zones', 'hvac_diffuser_w')
    arrays = {name: data[name] for name in required}
  times = pd.DatetimeIndex(arrays['interval_end'])
  names = arrays['zones'].astype(str).tolist()
  full = arrays['full_jax']
  linear = arrays['full_linear']
  grouped = arrays['grouped_jax']
  forcing = arrays['hvac_diffuser_w']
  if not times.is_monotonic_increasing or not times.is_unique:
    raise ValueError('Physics timestamps must be strictly ordered')
  if any(array.shape != full.shape for array in (linear, grouped, forcing)):
    raise ValueError('Temperature and HVAC arrays must have the same shape')
  if full.shape != (len(times), len(names)) or full.ndim != 2 or not full.size:
    raise ValueError('Physics arrays do not agree with timestamps and zone names')
  for array in (full, linear, grouped, forcing):
    if not np.isfinite(array).all():
      raise ValueError('Physics arrays contain a nonfinite value')
  rooms = manifest['rooms']
  if [room['name'] for room in rooms] != names:
    raise ValueError('The 100-zone model does not match the saved zone order')
  exterior_walls = np.array([
      sum(wall.get('window') is not None for wall in room['walls'])
      for room in rooms
  ], dtype=int)
  differences = {
      'full_linear_minus_full_nonlinear': linear - full,
      'grouped_linear_minus_full_nonlinear': grouped - full,
      'grouped_linear_minus_full_linear': grouped - linear,
  }
  identity = (
      differences['grouped_linear_minus_full_nonlinear']
      - differences['full_linear_minus_full_nonlinear']
      - differences['grouped_linear_minus_full_linear']
  )
  comparisons = {}
  for name, delta in differences.items():
    metrics = temperature_metrics(delta)
    metrics['maximum_zone'] = names[metrics.pop('maximum_zone_index')]
    metrics['maximum_interval_end'] = times[
        metrics.pop('maximum_time_index')
    ].isoformat()
    comparisons[name] = metrics
  zones = []
  total = differences['grouped_linear_minus_full_nonlinear']
  for index, name in enumerate(names):
    metrics = temperature_metrics(total[:, [index]])
    metrics.pop('maximum_zone_index')
    metrics['maximum_interval_end'] = times[
        metrics.pop('maximum_time_index')
    ].isoformat()
    metrics.pop('maximum_time_index', None)
    zones.append(dict(
        zone=name,
        exterior_walls=int(exterior_walls[index]),
        mean_hvac_w=float(forcing[:, index].mean()),
        min_hvac_w=float(forcing[:, index].min()),
        max_hvac_w=float(forcing[:, index].max()),
        **metrics,
    ))
  exterior = []
  for count in np.unique(exterior_walls):
    selected = exterior_walls == count
    delta = total[:, selected]
    metrics = temperature_metrics(delta)
    metrics.pop('maximum_time_index')
    index = metrics.pop('maximum_zone_index')
    metrics['maximum_zone'] = names[np.flatnonzero(selected)[index]]
    exterior.append(dict(
        exterior_walls=int(count),
        zone_count=int(selected.sum()),
        mean_hvac_w=float(forcing[:, selected].mean()),
        **metrics,
    ))
  hvac_loads = []
  load_masks = (
      ('cooling_over_750_w', forcing < -750),
      ('cooling_250_to_750_w', (forcing <= -250) & (forcing > -750)),
      ('cooling_10_to_250_w', (forcing <= -10) & (forcing > -250)),
      ('near_zero_abs_le_10_w', np.abs(forcing) <= 10),
      ('heating_over_10_w', forcing > 10),
  )
  for label, selected in load_masks:
    if not selected.any():
      continue
    delta = total[selected][:, None]
    metrics = temperature_metrics(delta)
    row_index, column_index = np.argwhere(selected)[
        metrics.pop('maximum_time_index')
    ]
    metrics.pop('maximum_zone_index')
    metrics['maximum_zone'] = names[column_index]
    metrics['maximum_interval_end'] = times[row_index].isoformat()
    hvac_loads.append(dict(
        category=label, mean_hvac_w=float(forcing[selected].mean()),
        **metrics,
    ))
  mean_mae = np.abs(total).mean(axis=1)
  worst_indices = np.argsort(mean_mae, kind='stable')[::-1][:8]
  worst_intervals = []
  for index in worst_indices:
    absolute = np.abs(total[index])
    worst_intervals.append(dict(
        interval_end=times[index].isoformat(),
        mean_absolute_error_k=float(absolute.mean()),
        cross_zone_p95_k=float(np.percentile(absolute, 95)),
        max_absolute_k=float(absolute.max()),
        maximum_zone=names[int(absolute.argmax())],
        mean_hvac_w=float(forcing[index].mean()),
        min_hvac_w=float(forcing[index].min()),
        max_hvac_w=float(forcing[index].max()),
    ))
  with (root / 'timing.csv').open(newline='', encoding='utf-8') as stream:
    timing_rows = list(csv.DictReader(stream))
  scale = []
  for zones_count in (16, 36, 100):
    rows = [row for row in timing_rows if int(row['zones']) == zones_count]
    by_engine = {row['engine']: row for row in rows}
    full_row, grouped_row = by_engine['full_jax'], by_engine['grouped_jax']
    # thermal_nodes is the complete storage vector and already includes mass.
    full_states = int(full_row['thermal_nodes'])
    grouped_states = int(grouped_row['thermal_nodes'])
    scale.append(dict(
        zones=zones_count,
        full_states=full_states,
        grouped_states=grouped_states,
        state_reduction=full_states / grouped_states,
        full_nonlinear_cg_ms=float(full_row['warm_ms']),
        grouped_linear_dense_ms=float(grouped_row['warm_ms']),
        combined_speedup=float(full_row['warm_ms']) / float(grouped_row['warm_ms']),
        full_core_states=int(full_row['thermal_nodes']) - int(full_row['mass_nodes']),
        grouped_core_states=int(grouped_row['thermal_nodes']) - int(grouped_row['mass_nodes']),
    ))
  absolute = np.abs(total)
  per_time_p95 = np.percentile(absolute, 95, axis=1)
  result = dict(
      source=dict(
          directory=str(root.resolve()),
          physics_npz_sha256=_digest(physics_path),
          summary_json_sha256=_digest(summary_path),
          floorplan_json_sha256=_digest(manifest_path),
          auxiliary_replay=auxiliary,
      ),
      zone_count=len(names),
      steps=len(times),
      sample_count=total.size,
      comparisons=comparisons,
      maximum_identity_error_k=float(np.abs(identity).max()),
      total_error_time_series=dict(
          mean_cross_zone_p95_k=float(per_time_p95.mean()),
          minimum_cross_zone_p95_k=float(per_time_p95.min()),
          maximum_cross_zone_p95_k=float(per_time_p95.max()),
      ),
      zones=sorted(zones, key=lambda row: row['zone']),
      days=_time_rows(total, forcing, times, times.strftime('%Y-%m-%d')),
      hours=_time_rows(total, forcing, times, times.strftime('%H:00')),
      exterior_walls=exterior,
      hvac_loads=hvac_loads,
      absolute_error_hvac_correlation=float(np.corrcoef(
          np.abs(total).ravel(), np.abs(forcing).ravel()
      )[0, 1]),
      worst_intervals=worst_intervals,
      timing=dict(rows=scale, **_matched_physics_timing(summary)),
      independent_closed_loop=_closed_loop_feedback(
          root, times, names, full, grouped
      ),
  )
  if nonlinear_path is not None:
    with np.load(nonlinear_path, allow_pickle=False) as data:
      nonlinear = data['grouped_nonlinear_c']
      durations = data['step_s']
      solves = data['linear_solves']
      if list(data['interval_end']) != list(arrays['interval_end']):
        raise ValueError('Supplementary nonlinear timestamps do not match the source')
      if list(data['zones']) != list(arrays['zones']):
        raise ValueError('Supplementary nonlinear zones do not match the source')
      if (str(data['solver_method']) != 'sparse_lu' or not bool(data['nonlinear'])
          or not bool(data['prepared_nonlinear'])):
        raise ValueError('Supplement must be a prepared grouped nonlinear sparse-LU replay')
      source_grouped = next(
          row for row in summary['physics']['metrics']
          if row['variant'] == 'grouped_jax'
      )
      if (int(data['thermal_node_count']) != source_grouped['thermal_nodes']
          or int(data['mass_node_count']) != source_grouped['mass_nodes']):
        raise ValueError('Supplementary state counts do not match the source')
      source_hashes = dict(
          source_physics_sha256=_digest(physics_path),
          source_summary_sha256=_digest(summary_path),
      )
      for key, digest in source_hashes.items():
        if str(data[key]) != digest:
          raise ValueError(f'Supplementary {key} does not match the source')
    if nonlinear.shape != full.shape or not np.isfinite(nonlinear).all():
      raise ValueError('Supplementary nonlinear temperatures have an invalid shape')
    if (durations.shape != (len(times),) or solves.shape != (len(times),)
        or not np.isfinite(durations).all() or np.any(durations < 0)
        or np.any(solves < 2)):
      raise ValueError('Supplementary nonlinear replay has invalid timing or solves')
    for label, delta in (
        ('grouped_nonlinear_minus_grouped_linear', nonlinear - grouped),
        ('grouped_nonlinear_minus_full_nonlinear', nonlinear - full),
    ):
      nonlinear_comparison = temperature_metrics(delta)
      nonlinear_comparison['maximum_zone'] = names[
          nonlinear_comparison.pop('maximum_zone_index')
      ]
      nonlinear_comparison['maximum_interval_end'] = times[
          nonlinear_comparison.pop('maximum_time_index')
      ].isoformat()
      result['comparisons'][label] = nonlinear_comparison
    result['source']['auxiliary_replay'].update(source_hashes)
    result['auxiliary_timing'] = dict(
        warm_mean_step_ms=float(1000 * durations[1:].mean()),
        linear_solves_per_step={
            str(int(count)): int(np.sum(solves == count))
            for count in np.unique(solves)
        },
    )
  return result


def _is_relative_to(path, other):
  try:
    path.relative_to(other)
    return True
  except ValueError:
    return False


def write_outputs(result, output):
  """Write derived CSV/JSON files into a newly reserved analysis directory."""
  output.mkdir(parents=True, exist_ok=False)
  attributes = ('zones', 'days', 'hours', 'exterior_walls', 'hvac_loads',
                'worst_intervals')
  file_names = ('zone_errors.csv', 'daily_errors.csv', 'hourly_errors.csv',
                'exterior_wall_errors.csv', 'hvac_load_errors.csv',
                'worst_intervals.csv')
  for attribute, name in zip(attributes, file_names, strict=True):
    rows = result[attribute]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / name).open('w', newline='', encoding='utf-8') as stream:
      writer = csv.DictWriter(stream, fieldnames=fields)
      writer.writeheader()
      writer.writerows(rows)
  with (output / 'comparisons.csv').open('w', newline='', encoding='utf-8') as stream:
    rows = [dict(comparison=name, **row)
            for name, row in result['comparisons'].items()]
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  with (output / 'timing_scale.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(result['timing']['rows'][0]))
    writer.writeheader()
    writer.writerows(result['timing']['rows'])
  closed_loop = result['independent_closed_loop']
  if closed_loop is not None:
    with (output / 'closed_loop_daily_errors.csv').open(
        'w', newline='', encoding='utf-8'
    ) as stream:
      writer = csv.DictWriter(stream, fieldnames=list(closed_loop['days'][0]))
      writer.writeheader()
      writer.writerows(closed_loop['days'])
    with (output / 'closed_loop_vs_physics.csv').open(
        'w', newline='', encoding='utf-8'
    ) as stream:
      rows = [dict(comparison=name, **row) for name, row in
              closed_loop['timing_rollout_minus_same_model_physics'].items()]
      writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
      writer.writeheader()
      writer.writerows(rows)
  (output / 'analysis_summary.json').write_text(
      json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8'
  )
  from . import analysis_plots
  paths = analysis_plots.make_plots(
      Path(result['source']['directory']), output, result
  )
  return [output / name for name in (*file_names, 'comparisons.csv',
                                   'timing_scale.csv', 'analysis_summary.json')] + paths


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--source', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--grouped-nonlinear', type=Path)
  args = parser.parse_args()
  if args.output.exists():
    parser.error('Analysis output already exists; choose a new directory')
  result = analyze_saved(args.source, args.grouped_nonlinear)
  paths = write_outputs(result, args.output.resolve())
  for path in paths:
    print(path)


if __name__ == '__main__':
  main()
