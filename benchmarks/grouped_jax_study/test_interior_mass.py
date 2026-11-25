"""Energy and integration checks for the zone interior mass model."""

import jax
import numpy as np
import pytest

from benchmarks.grouped_jax_study.matched_vav import generate, sbsim_model
from benchmarks.grouped_jax_study import interior_mass, jax_env
from benchmarks.grouped_jax_study.test_jax_env import _fixture


def test_two_surface_enclosure_conserves_exchange():
  weights = np.array([4.0, 9.0])
  exchange = interior_mass.enclosure_operator(weights)
  np.testing.assert_allclose(exchange.sum(axis=1), 0, atol=1e-20)
  np.testing.assert_allclose(exchange, exchange.T, atol=1e-20)
  hot, cold = 302.0, 293.0
  expected = interior_mass.SIGMA * (hot**4 - cold**4) * 36.0 / 13.0
  np.testing.assert_allclose(exchange @ np.array([hot, cold])**4, [expected, -expected])


def test_mass_capacity_and_jax_match_direct_heat_balance(tmp_path):
  model_dir, epw = _fixture(tmp_path)
  config = interior_mass.InteriorMass()
  env = jax_env.JaxOffice(model_dir, epw, 2, 1, mass_config=config)
  plan, day = generate.load_plan(model_dir)
  weather = sbsim_model.EpwWeather(epw, day)
  native = sbsim_model.make_simulator(
      plan, weather, weather.start, office=True, solver='linear-radiation-envelope'
  )
  h = weather.get_air_convection_coefficient(weather.start)
  static = native._build_grouped_conduction(h)
  air_nodes = np.asarray(env._operators.zone_nodes)
  mass = interior_mass.assemble(native.building, static, air_nodes, config)
  base_count = native._build_static(h)['conduction'].shape[0]
  assert env.thermal_nodes == base_count + len(plan.rooms)
  assert env.interior_mass_nodes == len(plan.rooms)
  np.testing.assert_allclose(mass.storage[-len(plan.rooms):], 2000.0)
  np.testing.assert_allclose(
      np.asarray(mass.conduction.sum(axis=1)).ravel()[air_nodes],
      np.asarray(static['conduction'].sum(axis=1)).ravel()[air_nodes],
  )
  for zone, air in enumerate(air_nodes):
    assert mass.conduction[base_count + zone, air] == pytest.approx(-60.0)
    assert mass.conduction[air, base_count + zone] == pytest.approx(-60.0)
  radiation = mass.radiation.toarray()
  np.testing.assert_allclose(radiation, radiation.T, atol=1e-18)
  np.testing.assert_allclose(
      radiation.sum(axis=0), np.pad(static['geometry'].emission, (0, len(plan.rooms))),
      atol=1e-18,
  )
  env.reset(0)
  initial = np.asarray(env._state.temperature[0]).copy()
  initial[base_count:] += np.linspace(2.0, 7.0, len(plan.rooms))
  heat = np.linspace(-100.0, 50.0, len(plan.rooms))
  correction = initial**4 - interior_mass.SLOPE * initial
  matrix = (mass.conduction + interior_mass.SLOPE * mass.radiation).toarray()
  geometry = static['geometry']
  outdoor = weather.get_current_temp(weather.start)
  exterior = geometry.forcing(weather.get_exterior_radiation(weather.start, outdoor))
  projected_office = np.bincount(
      static['groups'], weights=native._office_heat[weather.start.hour].ravel()[static['active']]
  )
  source = np.pad(
      static['ambient'] * outdoor + projected_office + exterior.shortwave_air_gain
      + exterior.emission * exterior.radiant_temperature**4,
      (0, len(plan.rooms)),
  )
  rhs = mass.storage * initial - radiation @ correction + source
  rhs[air_nodes] += heat
  expected = np.linalg.solve(matrix, rhs)
  result = jax_env._temperature_update(
      initial, heat, env._operators.sources[0], env._operators.transition
  )
  np.testing.assert_allclose(result, expected, rtol=0, atol=1e-9)

  initial_obs = env.reset(0)
  action = np.tile([[-1.0, -1.0], [1.0, 1.0]], (1, len(plan.rooms))).astype(np.float32)
  for _ in range(30):
    obs, reward = jax.block_until_ready(env.step(action))
  assert np.isfinite(reward).all()
  assert not np.array_equal(obs[0], obs[1])
  assert not np.array_equal(env.interior_mass_temperature_c[0], env.interior_mass_temperature_c[1])
  np.testing.assert_array_equal(env.reset(0), initial_obs)
  np.testing.assert_allclose(env.interior_mass_temperature_c, 22.5)


def test_mass_properties_reject_nonphysical_values():
  with pytest.raises(ValueError):
    interior_mass.InteriorMass(capacity_per_m2=0)
  with pytest.raises(ValueError):
    interior_mass.InteriorMass(emissivity=1.1)
