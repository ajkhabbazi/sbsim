"""Focused tests for the supplementary grouped nonlinear HVAC replay."""

from types import SimpleNamespace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.grouped_jax_study import replay_grouped_nonlinear as replay


def test_replay_arguments_apply_the_physics_sparse_lu_override():

  setup = dict(
      days=7, start_day='2024-07-15', energyplus_input_version='24.2',
      seed=42, grouped_method='dense', platform='cpu',
  )
  args = replay._replay_arguments(setup, Path('/weather.epw'))
  assert args.grouped_method == 'sparse_lu'
  assert args.physics_grouped_method == 'sparse_lu'
  assert args.run_day == date(2024, 7, 15)
  assert args.platform == 'cpu'


def test_nonlinear_is_selected_before_the_environment_prepares_its_matrix():
  from benchmarks.grouped_jax_study import run

  args = replay._replay_arguments(
      dict(days=7, start_day='2024-07-15', energyplus_input_version='24.2',
           seed=0, grouped_method='dense'), Path('/weather.epw')
  )
  options = run._solver_options(
      args, 'grouped_jax', linearized_radiative_heat_transfer=False
  )
  assert options['grouped']
  assert options['solver_method'] == 'sparse_lu'
  assert options['linearized_radiative_heat_transfer'] is False


class _FakeWeather:
  def get_current_temp(self, stamp):
    return 280.0 + stamp.minute / 5

  def get_air_convection_coefficient(self, stamp):
    del stamp
    return 12.0


class _FakeSimulator:
  def __init__(self):
    self.building = SimpleNamespace(
        input_q=np.zeros(4),
        diffusers=np.array([False, True, False, True]),
        temp=np.full((2, 2), 295.0),
    )
    self._current_timestamp = None
    self.last_solve_count = 0
    self.calls = []
    self.building.get_zone_average_temps = self.get_zone_average_temps

  @property
  def current_timestamp(self):
    return self._current_timestamp

  @current_timestamp.setter
  def current_timestamp(self, value):
    self._current_timestamp = value

  def get_zone_average_temps(self):
    return {'room': float(self.building.temp.mean())}

  def finite_differences_timestep(self, *, ambient_temperature,
                                  convection_coefficient):
    self.calls.append(dict(
        timestamp=self.current_timestamp,
        ambient_temperature=ambient_temperature,
        convection_coefficient=convection_coefficient,
        input_q=self.building.input_q.copy(),
    ))
    self.building.temp += self.building.input_q.mean() / 10000
    self.last_solve_count = 2
    # The real finite_differences_timestep solves heat only; it does not advance
    # the simulator timestamp. The benchmark replay gets weather by explicit time.


def test_replay_applies_the_recorded_heat_at_the_recorded_timestamp():
  sim = _FakeSimulator()
  start = pd.Timestamp('2024-07-15 00:00', tz='UTC-06:00')
  forcing = np.array([[100.0, 200.0], [300.0, 400.0]])
  result = replay.replay_saved_hvac(
      sim, _FakeWeather(), start, forcing, np.array([1, 3]), steps=2
  )
  np.testing.assert_array_equal(sim.calls[0]['input_q'], [0, 100, 0, 200])
  np.testing.assert_array_equal(sim.calls[1]['input_q'], [0, 300, 0, 400])
  assert sim.calls[0]['timestamp'] == start
  assert sim.calls[1]['timestamp'] == start + pd.Timedelta(minutes=5)
  assert result['linear_solves'].tolist() == [2, 2]
  assert result['interval_end'][0] == (
      start + pd.Timedelta(minutes=5)
  ).isoformat()

  assert result['temperatures_k'].shape == (2, 1)
  assert (result['step_s'] >= 0).all()


def test_replay_rejects_a_truncated_forcing_array():
  with np.testing.assert_raises_regex(ValueError, 'steps'):
    replay.replay_saved_hvac(
        _FakeSimulator(), _FakeWeather(),
        pd.Timestamp('2024-07-15', tz='UTC'), np.zeros((1, 2)),
        np.array([1, 3]), steps=2
    )
