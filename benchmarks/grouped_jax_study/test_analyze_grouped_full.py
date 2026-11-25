from pathlib import Path
from hashlib import sha256

import numpy as np
import pytest

from benchmarks.grouped_jax_study import analyze_grouped_full as analysis


ROOT = Path(__file__).resolve().parent / 'results' / 'restructure_cpu'


def test_metrics_preserve_signed_error_and_positive_bias():
  delta = np.array([[-0.1, 0.3], [0.5, -0.2]])
  metrics = analysis.temperature_metrics(delta)
  assert metrics['bias_k'] == pytest.approx(0.125)
  assert metrics['mae_k'] == pytest.approx(0.275)
  assert metrics['negative_percentage'] == pytest.approx(50.0)
  assert metrics['absolute_over_0_2_percentage'] == pytest.approx(50.0)
  assert metrics['maximum_zone_index'] == 0  # Column index for the 0.5 K error.
  assert metrics['maximum_time_index'] == 1
  np.testing.assert_allclose(
      delta, (delta - delta) + delta, rtol=0, atol=0
  )


def test_empty_time_zone_slice_is_rejected():
  with pytest.raises(ValueError, match='nonempty'):
    analysis.temperature_metrics(np.empty((0, 4)))


def test_saved_physics_statistics_and_decomposition():
  if not (ROOT / 'physics.npz').is_file():
    pytest.skip('The saved CPU physics result is not available')
  result = analysis.analyze_saved(ROOT)
  summary = result['comparisons']
  total = summary['grouped_linear_minus_full_nonlinear']
  linear = summary['full_linear_minus_full_nonlinear']
  grouped = summary['grouped_linear_minus_full_linear']
  assert total['mae_k'] == pytest.approx(0.10218534095890404)
  assert linear['mae_k'] == pytest.approx(0.00012475962958059998)
  assert grouped['mae_k'] == pytest.approx(0.10218892495638444)
  assert total['bias_k'] == pytest.approx(0.09981509599923435)
  assert result['maximum_identity_error_k'] == 0.0
  assert result['zone_count'] == 100
  assert len(result['zones']) == 100
  assert len(result['days']) == 8  # Seven days include the final midnight sample.
  assert [row['exterior_walls'] for row in result['exterior_walls']] == [0, 1, 2]
  assert result['timing']['matched_sparse_lu_speedup'] == pytest.approx(
      3.741974115669459 / 0.41067789784787456
  )
  assert [(row['full_states'], row['grouped_states'])
          for row in result['timing']['rows']] == [
              (2945, 129), (6565, 253), (18101, 621)
          ]
  assert result['timing']['rows'][-1]['full_core_states'] == 10101
  assert result['timing']['rows'][-1]['grouped_core_states'] == 521
  assert result['hvac_loads'][0]['sample_count'] == 42896
  assert result['hvac_loads'][0]['mae_k'] == pytest.approx(0.25362783409668926)
  assert result['absolute_error_hvac_correlation'] == pytest.approx(
      0.837657908288528
  )


def test_optional_nonlinear_replay_validates_provenance_and_warm_timing(tmp_path):
  if not (ROOT / 'physics.npz').is_file():
    pytest.skip('The saved CPU physics result is not available')
  with np.load(ROOT / 'physics.npz', allow_pickle=False) as source:
    payload = dict(
        grouped_nonlinear_c=source['grouped_jax'] + 0.1,
        interval_end=source['interval_end'], zones=source['zones'],
        step_s=np.full(len(source['interval_end']), 0.001),
        linear_solves=np.full(len(source['interval_end']), 2),
    )
  payload.update(
      source_physics_sha256=np.array(
          sha256((ROOT / 'physics.npz').read_bytes()).hexdigest()),
      source_summary_sha256=np.array(
          sha256((ROOT / 'summary.json').read_bytes()).hexdigest()),
      solver_method=np.array('sparse_lu'), nonlinear=np.array(True),
      prepared_nonlinear=np.array(True), thermal_node_count=np.array(621),
      mass_node_count=np.array(100),
  )
  nonlinear_path = tmp_path / 'nonlinear.npz'
  np.savez_compressed(nonlinear_path, **payload)
  result = analysis.analyze_saved(ROOT, nonlinear_path)
  row = result['comparisons']['grouped_nonlinear_minus_grouped_linear']
  assert row['mae_k'] == pytest.approx(0.1)
  assert result['auxiliary_timing'] == dict(
      warm_mean_step_ms=pytest.approx(1.0),
      linear_solves_per_step={'2': len(payload['step_s'])},
  )
  assert 'replay_npz_sha256' in result['source']['auxiliary_replay']
  payload['source_summary_sha256'] = np.array('not-the-source-hash')
  mismatched_path = tmp_path / 'mismatched.npz'
  np.savez_compressed(mismatched_path, **payload)
  with pytest.raises(ValueError, match='source'):
    analysis.analyze_saved(ROOT, mismatched_path)
