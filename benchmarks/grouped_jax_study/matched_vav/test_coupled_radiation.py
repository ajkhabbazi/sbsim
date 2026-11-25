"""Checks for native-equivalent flux and the grouped one-solve experiment."""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg

from smart_control.simulator import building_radiation_utils as radiation
from smart_control.simulator import constants
from smart_control.simulator import weather_controller

from . import coupled_radiation
from . import floorplan
from . import sbsim_model


def test_view_factors_match_native_for_shared_walls_and_glazing():
  plan = floorplan.make_plan(floorplan.PlanSpec(5, window_width_m=2))
  floor = plan.grid.astype(np.int8)
  floor[floor == constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT] = (
      constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION
  )
  floor[np.isin(floor, (
      constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT,
      constants.FENESTRATION_VALUE_IN_FILE_INPUT,
  ))] = constants.INTERIOR_WALL_VALUE_IN_FUNCTION
  walls = radiation.mark_interior_wall_adjacent_to_air(floor)
  indices, factors = coupled_radiation._interior_view_factors(
      plan.grid.shape, plan.grid.astype(np.int8).tobytes()
  )
  np.testing.assert_array_equal(indices, np.flatnonzero(walls))
  np.testing.assert_array_equal(factors.toarray(), radiation.get_vf(floor, walls))


def test_augmented_radiosity_matches_native_flux_with_nonclosed_view_rows():
  f = np.array([[0., .45, .45], [.45, 0., .48], [.45, .48, 0.]])
  eps = np.array([.8, .9, .93])
  temp = np.array([285., 298., 310.])
  ifa = radiation.calculate_ifa_inv(f, radiation.calculate_a_tilde_inv(eps, f))
  expected = radiation.net_radiative_heatflux_function_of_t(temp, ifa)
  a, outward, correction = coupled_radiation._radiosity_operators(sparse.csr_matrix(f), eps)
  blackbody = coupled_radiation.SIGMA * temp**4
  j = linalg.spsolve(a.tocsc(), blackbody)
  np.testing.assert_allclose(correction * blackbody - outward @ j, expected, atol=1e-12)


def test_a_grouped_radiation_step_preserves_nodes_and_solves_once():
  plan = floorplan.make_plan(floorplan.PlanSpec(1, window_width_m=2))
  start = pd.Timestamp('2024-07-15 12:00', tz='UTC')
  weather = weather_controller.WeatherController(304., 304.)
  models = [
      sbsim_model.make_simulator(plan, weather, start, solver=solver)
      for solver in ('coupled-radiation', 'linear-radiation')
  ]
  for model in models:
    assert model.finite_differences_timestep(
        ambient_temperature=304., convection_coefficient=12.
    )
    assert model.radiosity_nodes > 0
    assert model.active_heat_nodes < np.count_nonzero(model.building.len_neighbors >= 2)
  assert models[1].last_outer_iteration_count == 1
  np.testing.assert_allclose(models[1].building.temp, models[0].building.temp, atol=.01)


def test_eliminating_radiosity_keeps_the_grouped_linear_solution():
  plan = floorplan.make_plan(floorplan.PlanSpec(5, window_width_m=2))
  start = pd.Timestamp('2024-07-15 12:00', tz='UTC')
  weather = weather_controller.WeatherController(309., 309.)
  a, b = [
      sbsim_model.make_simulator(plan, weather, start, solver='linear-radiation-envelope')
      for _ in range(2)
  ]
  b.eliminate_radiosity = False
  for _ in range(5):
    for sim in (a, b):
      sim.step_sim()
    np.testing.assert_allclose(a.building.temp, b.building.temp, atol=1e-8, rtol=0)
  assert a.radiosity_unknowns_per_solve == 0 < b.radiosity_unknowns_per_solve
