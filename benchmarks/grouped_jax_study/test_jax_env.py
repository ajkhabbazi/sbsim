"""Check the batched JAX environment against the native thermal and VAV path."""

from datetime import date

import jax
import numpy as np

from benchmarks.grouped_jax_study.matched_vav import floorplan
from benchmarks.grouped_jax_study.matched_vav import generate
from benchmarks.grouped_jax_study.matched_vav import sbsim_model
from benchmarks.grouped_jax_study.matched_vav import settings


def _fixture(tmp_path):
  plan_dir = tmp_path / 'model'
  spec = floorplan.PlanSpec(5, window_width_m=2)
  day = date(2024, 7, 19)  # The following day starts eco mode.
  generate.write_models(plan_dir, spec, day, '24.2', office=True)
  epw = tmp_path / 'test.epw'
  rows = ['LOCATION,Test,IL,USA,TMY3,1234,41.98,-87.9,-6,200'] + ['TEST'] * 7
  for day_number in range(18, 22):
    for hour in range(1, 25):
      daylight = 7 <= hour <= 19
      values = [
          2024,
          7,
          day_number,
          hour,
          60,
          '?',
          20 + hour / 3,
          10,
          50,
          100000,
          0,
          0,
          300,
          200 * daylight,
          100 * daylight,
          100 * daylight,
      ] + [0] * 19
      rows.append(','.join(map(str, values)))
  epw.write_text('\n'.join(rows) + '\n')
  return plan_dir, epw


def test_batch_fixed_heat_and_native_vav_trajectory(tmp_path):
  from benchmarks.grouped_jax_study import jax_env

  plan_dir, epw = _fixture(tmp_path)
  env = jax_env.JaxOffice(plan_dir, epw, 2, 2, platform='cpu')
  plan, day = generate.load_plan(plan_dir)
  weather = sbsim_model.EpwWeather(epw, day, 2)
  native = sbsim_model.make_simulator(
      plan,
      weather,
      weather.start,
      office=True,
      solver='linear-radiation-envelope',
  )
  for terminal in native.hvac.vavs.values():
    terminal.thermostat._setpoint_schedule = (
        sbsim_model._SharedSetpointSchedule(
            0,
            24,
            (294.15, 297.15),
            (294.15, 297.15),
            time_zone=weather.start.tz,
        )
    )
  initial = env.reset(13)
  assert isinstance(initial, jax.Array)
  assert initial.shape == (2, env.observation_size)
  assert env.action_size == 10
  np.testing.assert_array_equal(initial[0], initial[1])

  static = native._build_static(
      weather.get_air_convection_coefficient(weather.start)
  )
  groups, active = static['groups'], static['active']
  zone_power = np.arange(len(plan.rooms), dtype=float) * 40
  for name, power in zip(native.hvac.vavs, zone_power):
    native.building.apply_thermal_power_zone(name, power)
  native.finite_differences_timestep(
      ambient_temperature=weather.get_current_temp(weather.start),
      convection_coefficient=weather.get_air_convection_coefficient(
          weather.start
      ),
  )
  fixed_step = jax_env._temperature_update(
      initial_temperature=env._initial_state.temperature,
      hvac_heat=np.stack([zone_power, zone_power]),
      source=env._operators.sources[0],
      transition=env._operators.transition,
  )
  np.testing.assert_allclose(
      np.asarray(fixed_step[0])[groups],
      native.building.temp.ravel()[active],
      atol=1e-9,
      rtol=0,
  )
  native.reset()

  rng = np.random.default_rng(13)
  count = 24 * settings.STEPS_PER_HOUR + 10
  for idx in range(count):
    actions = rng.uniform(-1, 1, (2, env.action_size)).astype(np.float32)
    heat, cool = actions[0].astype(float).reshape(-1, 2).T
    for terminal, low, high in zip(native.hvac.vavs.values(), heat, cool):
      schedule = terminal.thermostat.get_setpoint_schedule()
      bounds = (20.5 + 1.5 * low + 273.15, 24.5 + 1.5 * high + 273.15)
      schedule.comfort_temp_window = schedule.eco_temp_window = bounds
    stamp = native.current_timestamp
    native.setup_step_sim()
    native.execute_step_sim(video_filename=None)
    obs, reward = env.step(actions)
    expected = (
        np.array([
            *native.building.get_zone_average_temps().values(),
            weather.get_current_temp(stamp),
        ])
        - 273.15
    )
    np.testing.assert_allclose(
        np.asarray(obs[0]), (expected - 22.5) / 10, rtol=0, atol=3e-7
    )
    penalty = np.maximum(21.0 - expected[:-1], 0) + np.maximum(
        expected[:-1] - 24.0, 0
    )
    np.testing.assert_allclose(reward[0], -penalty.mean(), rtol=0, atol=3e-7)
    states = [
        v.thermostat._current_mode.value for v in native.hvac.vavs.values()
    ]
    np.testing.assert_array_equal(env._state.mode[0], states)
    np.testing.assert_allclose(
        np.asarray(env._state.temperature[0])[groups],
        native.building.temp.ravel()[active],
        rtol=0,
        atol=1e-8,
        err_msg=f'timestep {idx}',
    )
  assert not np.array_equal(obs[0], obs[1])
  np.testing.assert_array_equal(env.reset(13), initial)


def test_nonlinear_grouped_radiation_matches_native(tmp_path):
  from benchmarks.grouped_jax_study import jax_env

  plan_dir, epw = _fixture(tmp_path)
  env = jax_env.JaxOffice(plan_dir, epw, 2, 1, nonlinear=True)
  plan, day = generate.load_plan(plan_dir)
  weather = sbsim_model.EpwWeather(epw, day)
  native = sbsim_model.make_simulator(
      plan,
      weather,
      weather.start,
      office=True,
      solver='linear-radiation-envelope',
  )
  # Use the native nonlinear iteration with the same merged envelope groups.
  native.nonlinear = True
  native.eliminate_radiosity = False
  native.hvac._schedule.comfort_temp_window = (293.65, 297.65)
  native.hvac._schedule.eco_temp_window = (293.65, 297.65)
  env.reset(0)
  actions = np.zeros((2, env.action_size), dtype=np.float32)
  actions[1, ::2] = 1
  actions[1, 1::2] = -1
  for idx in range(150):
    native.setup_step_sim()
    native.execute_step_sim(video_filename=None)
    obs, _ = env.step(actions)
    groups = native._static['groups']
    np.testing.assert_allclose(
        np.asarray(env._state.temperature[0])[groups],
        native.building.temp.ravel()[native._static['active']],
        atol=1e-7,
        rtol=0,
        err_msg=f'timestep {idx}',
    )
  assert not np.array_equal(obs[0], obs[1])
  assert env.nonlinear_converged
  assert 1 < env.worst_nonlinear_iterations < 20
  env.reset(0)
  assert env.nonlinear_converged
  assert env.worst_nonlinear_iterations == 0
