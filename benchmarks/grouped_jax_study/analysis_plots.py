"""Static diagnostic figures and machine-readable companion tables.

English labels make one image usable from both language guides. Each plotted
quantity is also written to CSV or analysis_summary.json by analyze_grouped_full.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


BLUE = '#2a78d6'
ORANGE = '#eb6834'
GREEN = '#1baf7a'
INK = '#25344a'
MUTED = '#596579'
GRID = '#dbe2ec'
SEQUENTIAL = LinearSegmentedColormap.from_list(
    'analysis_blue', ['#f3f7fd', '#8fb9eb', BLUE, '#19477e']
)


def _style(ax):
  ax.set_facecolor('white')
  ax.grid(axis='y', color=GRID, linewidth=0.55, zorder=0)
  ax.tick_params(colors=MUTED, labelsize=9)
  for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
  for side in ('left', 'bottom'):
    ax.spines[side].set_color(GRID)
  ax.xaxis.label.set_color(MUTED)
  ax.yaxis.label.set_color(MUTED)
  ax.title.set_color(INK)


def _time(source, output, result):
  """Compare stored Full/Grouped zone averages, without tail quantiles."""
  from . import report as benchmark_report

  with np.load(source / 'physics.npz', allow_pickle=False) as data:
    times = pd.DatetimeIndex(data['interval_end'])
    names = data['zones'].astype(str)
    full = data['full_jax']
    grouped = data['grouped_jax']
  absolute = np.abs(grouped - full)
  mean = absolute.mean(axis=1)
  room = int(absolute.mean(axis=0).argmax())
  pd.DataFrame({
      'interval_end': times.astype(str),
      'zone': names[room],
      'full_nonlinear_c': full[:, room],
      'grouped_linear_c': grouped[:, room],
      'absolute_discrepancy_k': absolute[:, room],
      'mean_absolute_discrepancy_k': mean,
  }).to_csv(output / 'zone_average_timeseries.csv', index=False)
  fig = benchmark_report._physics_plot(
      {'full_jax': full, 'grouped_jax': grouped}, times, names, room
  )
  return _save(fig, output / 'error_timeseries.png')


def _zones(source, output, result):
  manifest = json.loads(
      (source / 'models' / 'zone100' / 'floorplan.json').read_text(
          encoding='utf-8'
      )
  )
  by_name = {row['zone']: row for row in result['zones']}
  values = np.array([by_name[room['name']]['mae_k'] for room in manifest['rooms']])
  cmap = SEQUENTIAL
  norm = plt.Normalize(0, values.max())
  fig, ax = plt.subplots(figsize=(10.5, 5.6), layout='constrained')
  for room, value in zip(manifest['rooms'], values, strict=True):
    left, bottom, right, top = room['bounds_m']
    ax.add_patch(Rectangle(
        (left + 0.03, bottom + 0.03), right - left - 0.06,
        top - bottom - 0.06, facecolor=cmap(norm(value)),
        edgecolor='white', linewidth=1.1,
    ))
  for room in sorted(manifest['rooms'], key=lambda room: by_name[room['name']]['mae_k'])[-4:]:
    left, bottom, right, top = room['bounds_m']
    ax.text((left + right) / 2, (bottom + top) / 2,
            f"{room['name']}\n{by_name[room['name']]['mae_k']:.3f} K",
            fontsize=7.5, ha='center', va='center', color='white',
            bbox=dict(facecolor='#25344a', edgecolor='none', alpha=0.80, pad=1.4))
  ax.set_xlim(-1, 51.5)
  ax.set_ylim(-1, 46.5)
  ax.set_aspect('equal')
  ax.set_xlabel('Floor-plan x [m]')
  ax.set_ylabel('Floor-plan y [m]')
  ax.set_title('Mean absolute zone-temperature discrepancy by room', loc='left',
               fontsize=12)
  _style(ax)
  ax.grid(False)
  colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                          fraction=0.03, pad=0.02)
  colorbar.set_label('Time-averaged |grouped linear − full nonlinear| [K]',
                     color=MUTED, fontsize=9)
  colorbar.ax.tick_params(labelsize=9, colors=MUTED)
  path = output / 'zone_mae_map.png'
  fig.savefig(path, dpi=160, facecolor='white')
  plt.close(fig)
  return path


def _dot_panel(ax, rows, labels, details):
  scores = np.array([float(row['mae_k']) for row in rows])
  positions = np.arange(len(rows))[::-1]
  for position, value in zip(positions, scores, strict=True):
    ax.plot([0, value], [position, position], color=BLUE, linewidth=2.0,
            solid_capstyle='round', zorder=2)
    ax.plot(value, position, 'o', color=BLUE, markersize=8,
            markeredgecolor='white', markeredgewidth=1.5, zorder=3)
    ax.text(value + 0.004, position, f'{value:.3f} K', va='center', fontsize=9,
            color=INK)
  ax.set_yticks(positions)
  ax.set_yticklabels([f'{label}\n{detail}' for label, detail in
                      zip(labels, details, strict=True)], fontsize=9)
  ax.set_xlim(0, max(scores.max() * 1.26, 0.01))
  ax.set_xlabel('Mean |grouped linear − full nonlinear| [K]')
  _style(ax)
  ax.grid(axis='x', color=GRID, linewidth=0.55)
  ax.grid(axis='y', visible=False)


def _strata(output, result):
  fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.8), layout='constrained')
  groups = (
      (axes[0], result['exterior_walls'], 'Exterior exposure',
       [str(row['exterior_walls']) for row in result['exterior_walls']],
       [f"{row['zone_count']} zones; HVAC mean {row['mean_hvac_w']:.0f} W"
        for row in result['exterior_walls']],
       'Number of exterior walls with windows'),
      (axes[1], result['hvac_loads'], 'HVAC heat input',
       ['>750 W cooling', '250–750 W cooling', '10–250 W cooling',
        '|HVAC| ≤ 10 W', '>10 W heating'][:len(result['hvac_loads'])],
       [f"{row['sample_count']:,} samples; mean {row['mean_hvac_w']:.0f} W"
        for row in result['hvac_loads']],
       'Recorded full-reference diffuser power'),
  )
  for ax, rows, title, labels, details, xlabel in groups:
    _dot_panel(ax, rows, labels, details)
    ax.set_title(title, loc='left', pad=18, fontsize=11)
    ax.set_xlabel(xlabel + '\nMean absolute zone-temperature error [K]', fontsize=9)
  fig.suptitle('Error strata describe associations, not causal attribution',
               color=INK, fontsize=12, x=0.03, ha='left')
  return _save(fig, output / 'error_strata.png')


def _timing(source, output, result):
  summary = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
  metrics = {row['variant']: row for row in summary['physics']['metrics']}
  auxiliary = result.get('auxiliary_timing')
  fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.4), layout='constrained',
                           sharey=True)
  panels = (
      ('Full grid\n18,101 states (8,000 mass)',
       [('Nonlinear\nCG', metrics['full_jax']['mean_step_ms']),
        ('Linear\nsparse LU', metrics['full_linear']['mean_step_ms'])]),
      ('Grouped grid\n621 states (100 mass)',
       ([('Nonlinear\nsparse LU', auxiliary['warm_mean_step_ms'])]
        if auxiliary else [])
       + [('Linear\nsparse LU', metrics['grouped_jax']['mean_step_ms'])]),
  )
  for ax, (title, rows) in zip(axes, panels, strict=True):
    values = [value for _, value in rows]
    bars = ax.bar(range(len(rows)), values,
                  color=[ORANGE if name.startswith('Linear') else BLUE
                         for name, _ in rows], width=0.52, zorder=3)
    ax.set_yscale('log')
    ax.set_ylim(0.2, 48)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([name for name, _ in rows], fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, pad=17)
    for bar, value in zip(bars, values, strict=True):
      ax.text(bar.get_x() + bar.get_width() / 2, value * 1.15,
              f'{value:.3f} ms', ha='center', va='bottom', color=INK,
              fontsize=10)
    _style(ax)
    if ax is axes[0]:
      ax.set_ylabel('Warm wall time [ms/step] (log scale)')
  fig.suptitle('State reduction and nonlinear outer solves change runtime together',
               fontsize=12, color=INK, x=0.035, ha='left')
  fig.supxlabel(
      'Fixed-HVAC physics replays; the full nonlinear reference also records native VAV heat. '
      'First step excluded.',
      color=MUTED, fontsize=9
  )
  return _save(fig, output / 'timing_decomposition.png')


def _save(fig, path):
  fig.savefig(path, dpi=160, facecolor='white')
  plt.close(fig)
  return path


def make_plots(source, output, result):
  """Create the four report figures from immutable benchmark artifacts."""
  source = Path(source)
  output = Path(output)
  return [
      _time(source, output, result),
      _zones(source, output, result),
      _strata(output, result),
      _timing(source, output, result),
  ]
