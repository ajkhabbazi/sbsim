from pathlib import Path

from benchmarks.grouped_jax_study import analyze_grouped_full as analysis


ROOT = Path(__file__).resolve().parent / 'results' / 'restructure_cpu'


def test_analysis_plots_are_generated_from_saved_zone_average_errors(tmp_path):
  if not (ROOT / 'physics.npz').is_file():
    import pytest
    pytest.skip('The saved CPU physics result is not available')
  supplement = ROOT.parent / 'grouped_full_nonlinear_replay.npz'
  result = analysis.analyze_saved(
      ROOT, supplement if supplement.is_file() else None
  )
  from benchmarks.grouped_jax_study import analysis_plots
  paths = analysis_plots.make_plots(ROOT, tmp_path, result)
  assert {path.name for path in paths} == {
      'error_timeseries.png', 'zone_mae_map.png',
      'error_strata.png', 'timing_decomposition.png',
  }
  import matplotlib.image as mpimg
  for path in paths:
    assert path.stat().st_size > 1000
    image = mpimg.imread(path)
    assert image.shape[0] >= 600
    assert image.shape[1] >= 1000


def test_temperature_comparison_csv_matches_saved_zone_averages(tmp_path):
  if not (ROOT / 'physics.npz').is_file():
    import pytest
    pytest.skip('The saved CPU physics result is not available')
  import numpy as np
  import pandas as pd

  supplement = ROOT.parent / 'grouped_full_nonlinear_replay.npz'
  result = analysis.analyze_saved(
      ROOT, supplement if supplement.is_file() else None
  )
  from benchmarks.grouped_jax_study import analysis_plots
  analysis_plots._time(ROOT, tmp_path, result)
  with np.load(ROOT / 'physics.npz', allow_pickle=False) as data:
    full = data['full_jax']
    grouped = data['grouped_jax']
    names = data['zones'].astype(str)
    times = pd.DatetimeIndex(data['interval_end'])
  room = np.abs(grouped - full).mean(axis=0).argmax()
  table = pd.read_csv(tmp_path / 'zone_average_timeseries.csv')

  assert list(table.columns) == [
      'interval_end', 'zone', 'full_nonlinear_c', 'grouped_linear_c',
      'absolute_discrepancy_k', 'mean_absolute_discrepancy_k',
  ]
  assert len(table) == len(times) == 2016
  assert table['zone'].unique().tolist() == [names[room]]
  pd.testing.assert_index_equal(
      pd.DatetimeIndex(table['interval_end']), times, check_names=False,
  )
  np.testing.assert_allclose(table['full_nonlinear_c'], full[:, room])
  np.testing.assert_allclose(table['grouped_linear_c'], grouped[:, room])
  np.testing.assert_allclose(
      table['absolute_discrepancy_k'], np.abs(grouped[:, room] - full[:, room])
  )
  np.testing.assert_allclose(
      table['mean_absolute_discrepancy_k'], np.abs(grouped - full).mean(axis=1)
  )
