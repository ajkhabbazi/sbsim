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
  # Check a daytime step (12:00) so transmitted window solar is nonzero and
  # verified to enter interior mass nodes rather than room air nodes.
  daytime_step = 12 * 12
  stamp = weather._data.index[daytime_step]
  outdoor = weather.get_current_temp(stamp)
  poa = weather.facade_flux(stamp, mass.geometry.azimuths)
  exterior = mass.geometry.forcing(
      weather.get_exterior_radiation(stamp, outdoor), poa_irradiance=poa
  )
  base_exterior = static['geometry'].forcing(
      weather.get_exterior_radiation(stamp, outdoor), poa_irradiance=poa
  )
  np.testing.assert_allclose(
      exterior.shortwave_air_gain[air_nodes], 0.0, atol=1e-12
  )
  assert exterior.shortwave_air_gain[base_count:].sum() > 0.0
  np.testing.assert_allclose(
      exterior.shortwave_air_gain[base_count:],
      base_exterior.shortwave_air_gain[air_nodes],
  )
  projected_office = np.bincount(
      static['groups'],
      weights=native._office_heat[stamp.hour].ravel()[static['active']],
  )
  source = (
      np.pad(static['ambient'] * outdoor + projected_office, (0, len(plan.rooms)))
      + exterior.shortwave_air_gain
      + exterior.mass_gain
      + exterior.emission * exterior.radiant_temperature**4
  )
  rhs = mass.storage * initial - radiation @ correction + source
  rhs[air_nodes] += heat
  expected = np.linalg.solve(matrix, rhs)
  result = jax_env._temperature_update(
      initial, heat, env._operators.sources[daytime_step], env._operators.transition
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
