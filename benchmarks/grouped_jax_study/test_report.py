"""Verify direct Full/Grouped zone-average temperature comparison plots."""

import numpy as np
import pandas as pd

from benchmarks.grouped_jax_study import report


def test_physics_plot_compares_zone_averages_without_quantiles(
    tmp_path, monkeypatch,
):
  full = np.array([[20.0, 22.0], [21.0, 23.0]])
  traces = {
      'full_jax': full,
      'full_linear': full + 0.0001,
      'grouped_jax': np.array([[19.0, 22.0], [24.0, 24.0]]),
  }
  figures = []

  monkeypatch.setattr(
      report.plt.Figure, 'savefig',
      lambda figure, *args, **kwargs: figures.append(figure)
  )
  report.temperature(
      tmp_path,
      traces,
      metrics=[],
      times=pd.date_range('2024-07-15', periods=2, freq='5min', tz='UTC'),
      names=['room_1', 'room_2'],
      forcing={},
  )

  top, bottom = figures[0].axes
  # room_1's mean absolute discrepancy is 2 K versus 0.5 K in room_2.
  assert 'room_1' in top.get_title()
  assert top.get_ylabel() == 'Zone-average temperature [°C]'
  top_traces = {line.get_label(): line.get_ydata() for line in top.lines}
  assert set(top_traces) == {'Full JAX', 'Grouped JAX'}
  np.testing.assert_allclose(top_traces['Full JAX'], [20.0, 21.0])
  np.testing.assert_allclose(top_traces['Grouped JAX'], [19.0, 24.0])
  bottom_traces = {line.get_label(): line.get_ydata() for line in bottom.lines}
  assert set(bottom_traces) == {'Grouped JAX MAE'}
  np.testing.assert_allclose(bottom_traces['Grouped JAX MAE'], [0.5, 2.0])
  for axis in (top, bottom):
    _, labels = axis.get_legend_handles_labels()
    assert all('p95' not in label.lower() for label in labels)
    assert all('quantile' not in label.lower() for label in labels)
