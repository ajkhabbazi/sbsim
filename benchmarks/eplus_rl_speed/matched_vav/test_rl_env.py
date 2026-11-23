"""Matched RL environment episode and action checks."""

from datetime import date, timedelta
import json
from pathlib import Path
import os

import numpy as np
import pytest


@pytest.mark.parametrize('engine', ['eplus', 'sbsim'])
def test_real_model_uses_full_episode_and_auto_resets(engine, tmp_path):
  from benchmarks.eplus_rl_speed.matched_vav import rl_env

  ep_home = os.environ.get('ENERGYPLUS_DIR')
  if not ep_home:
    pytest.skip('Set ENERGYPLUS_DIR to run against the real model and EPW')
  ep_home = Path(ep_home)
  weather = next((ep_home / 'WeatherData').glob('*Chicago*epw'))
  model = Path(__file__).parent / 'models' / 'zone05'
  env = rl_env.make_env(engine, model, weather, ep_home, tmp_path)
  action = np.array([1 / 3, -1 / 3], np.float32)
  try:
    initial = env.reset()
    assert initial.is_first()
    assert initial.observation.shape == (6,)
    early_observations = []
    for step in range(1, 289):
      previous = env.current_time_step().observation
      result = env.step(action)
      if step <= 24:
        early_observations.append(result.observation)
      assert np.isfinite(result.observation).all()
      assert bool(result.is_last()) == (step == 288)
    assert not np.array_equal(previous, result.observation)
    assert env.episode_steps == env.total_steps == 288
    restart = env.step(action)
    assert restart.is_first()
    np.testing.assert_allclose(
        restart.observation, initial.observation, atol=1e-5
    )
    assert env.total_steps == 288
    assert env.episode_steps == 0
    assert len(env.reset_times) == 2
    changed = [
        env.step(np.array([1, -1], np.float32)).observation for _ in range(24)
    ]
    assert np.max(np.abs(np.array(early_observations) - changed)) > 0.01
  finally:
    env.close()


@pytest.mark.parametrize('engine', ['eplus', 'sbsim'])
@pytest.mark.parametrize('start', [date(2024, 7, 15), date(2024, 12, 29)])
def test_real_model_runs_one_week_without_daily_resets(engine, start, tmp_path):
  from benchmarks.eplus_rl_speed.matched_vav import generate
  from benchmarks.eplus_rl_speed.matched_vav import rl_env

  ep_home = os.environ.get('ENERGYPLUS_DIR')
  if not ep_home:
    pytest.skip('Set ENERGYPLUS_DIR to run against the real model and EPW')
  ep_home = Path(ep_home)
  weather = next((ep_home / 'WeatherData').glob('*Chicago*epw'))
  model = tmp_path / 'model'
  generate.write_models(
      model, generate.PlanSpec(5, window_width_m=2), start, '24.2'
  )
  kwargs = {'sbsim_solver': 'fast-lu'} if engine == 'sbsim' else {}
  env = rl_env.make_env(
      engine, model, weather, ep_home, tmp_path, episode_days=7, **kwargs
  )
  try:
    first = env.reset()
    assert first.is_first() and env.steps_per_episode == 2016
    daily_weather = []
    for step in range(1, 2017):
      result = env.step(np.zeros(2, dtype=np.float32))
      assert np.isfinite(result.observation).all()
      assert bool(result.is_last()) == (step == 2016)
      if step % 288 == 0:
        daily_weather.append(result.observation[-1])
    assert len(set(daily_weather)) > 1
    assert env.total_steps == 2016
    assert env.completed_episodes == len(env.reset_times) == 1
    last = start + timedelta(days=6)
    if engine == 'eplus':
      native = json.loads(env.backend.model_path.read_text())
      period = native['RunPeriod']['Smoke Day']
      assert (period['end_month'], period['end_day_of_month']) == (
          last.month, last.day
      )
      assert env.backend.steps == 2016
    else:
      assert env.backend.sim.current_timestamp.date() == last + timedelta(days=1)
  finally:
    env.close()


def test_energyplus_zone_energy_control_reads_positive_system_energy(tmp_path):
  from benchmarks.eplus_rl_speed.matched_vav import rl_env

  ep_home = os.environ.get('ENERGYPLUS_DIR')
  if not ep_home:
    pytest.skip('Set ENERGYPLUS_DIR to run against the real model and EPW')
  ep_home = Path(ep_home)
  weather = next((ep_home / 'WeatherData').glob('*Chicago*epw'))
  model = Path(__file__).parent / 'models' / 'zone05'
  env = rl_env.make_env(
      'eplus', model, weather, ep_home, tmp_path, control='zone-energy'
  )
  try:
    first = env.reset()
    assert first.observation.shape == (7,)
    assert first.observation[-1] == 0.
    assert env.action_spec().shape == (10,)
    for _ in range(12):
      step = env.step(np.zeros(10, dtype=np.float32))
      assert step.observation[-1] >= 0.
    assert len(env.backend.hvac_kwh) == 12
    assert sum(env.backend.hvac_kwh) > 0.
    assert len(set(env.backend.actuator_handles)) == 10
  finally:
    env.close()
