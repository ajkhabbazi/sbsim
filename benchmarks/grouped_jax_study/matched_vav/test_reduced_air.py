"""Checks for a geometry-preserving reduction of interior air cells."""

import numpy as np
import pytest
from scipy import ndimage
from scipy import sparse

from . import reduced_air
from . import reduced_envelope


def test_l_shaped_air_and_disconnected_pockets_keep_their_geometry():
  air = np.zeros((8, 9), dtype=bool)
  air[2:6, 2:4] = True
  air[4:6, 4:7] = True
  air[2, 6] = True
  shell = ndimage.binary_dilation(air, structure=np.ones((3, 3), bool))
  solved = np.flatnonzero(shell)
  positions = reduced_air.group_cells(solved, air, [np.argwhere(air)])
  mapping = np.full(air.size, -1)
  mapping[solved] = positions
  mapping = mapping.reshape(air.shape)
  assert mapping[2, 2] == mapping[5, 6]
  assert mapping[2, 6] != mapping[5, 6]
  assert len(np.unique(mapping[shell & ~air])) == np.count_nonzero(shell & ~air)
  tiled = reduced_air.group_cells(
      solved, air, [np.argwhere(air)], block_cells=2
  )
  assert len(np.unique(tiled)) > len(np.unique(positions))


def test_projection_sums_air_capacity_and_cancels_internal_wall_free_faces():
  # Two air CVs connected by 4 W/K, sharing a single wall neighbor via two paths.
  full = sparse.csr_matrix(
      [[11.0, -4.0, -2.0], [-4.0, 16.0, -3.0], [-2.0, -3.0, 16.0]]
  )
  projector = reduced_air.projection(np.array([0, 0, 1]))
  contracted = (projector.T @ full @ projector).toarray()
  np.testing.assert_array_equal(contracted, [[19.0, -5.0], [-5.0, 16.0]])
  np.testing.assert_array_equal(projector.T @ [5.0, 9.0, 11.0], [14.0, 11.0])


def test_envelope_segments_stop_at_glazing_and_at_zone_boundaries():
  floor = np.full((4, 12), 2)
  floor[1, 1:11] = [1, 1, 1, 4, 4, 1, 1, 1, 1, 1]
  floor[2, 1:11] = 0
  rooms = [
      np.argwhere((floor == 0) & (np.arange(12) < 8)),
      np.argwhere((floor == 0) & (np.arange(12) >= 8)),
  ]
  active = np.flatnonzero(floor != 2)
  groups = reduced_envelope.group_envelope_cells(
      active, floor, rooms, np.arange(len(active))
  )
  facade = groups[1:9].reshape(-1, 2)
  np.testing.assert_array_equal(facade[:, 0], facade[:, 1])
  assert len(np.unique(facade)) == 4
  assert groups[0] != groups[1] and groups[9] != groups[8]


def test_thick_facade_keeps_layers_and_stops_at_window():
  floor = np.full((5, 14), 2)
  floor[1:3, 1:13] = [1] * 5 + [4] * 2 + [1] * 5
  floor[3, 1:13] = 0
  active = np.flatnonzero(floor != 2)
  groups = reduced_envelope.group_envelope_cells(
      active, floor, [np.argwhere(floor == 0)], np.arange(len(active)),
      wall_depth_cells=2,
  ).reshape(3, 12)
  for layer in groups[:2]:
    assert layer[2] == layer[3] == layer[4]
    assert layer[5] == layer[6]
    assert layer[7] == layer[8] == layer[9]
    assert len(set(layer[2:10])) == 3
    assert layer[0] != layer[2] and layer[-1] != layer[-3]
  assert not (groups[0] == groups[1]).any()


def test_solver_runs_an_l_shaped_room_and_one_cell_groups_match_the_full_grid():
  import pandas as pd
  from . import floorplan
  from . import sbsim_model
  from smart_control.simulator import weather_controller

  air = np.zeros((8, 9), dtype=bool)
  air[2:6, 2:4] = True
  air[4:6, 4:7] = True
  shell = ndimage.binary_dilation(air, structure=np.ones((3, 3), bool))
  grid = np.full(air.shape, 2, dtype=np.uint8)
  grid[shell] = 1
  grid[air] = 0
  area = np.count_nonzero(air) * 0.5**2
  room = floorplan.Room('room_1', (0.5, 0.5, 3.0, 2.5), area, area * 3, ())
  plan = floorplan.FloorPlan(floorplan.PlanSpec(1), (room,), grid, grid.copy())
  stamp = pd.Timestamp('2024-07-15', tz='UTC')
  weather = weather_controller.WeatherController(300.0, 300.0)
  reference = sbsim_model.make_simulator(
      plan, weather, stamp, solver='cached-lu'
  )
  one_cell = sbsim_model.make_simulator(
      plan, weather, stamp, solver='reduced-air', air_block_cells=1
  )
  one_zone = sbsim_model.make_simulator(
      plan, weather, stamp, solver='reduced-air'
  )

  for model in (reference, one_cell, one_zone):
    model.building.input_q[2, 2] = 100.0
    model.step_sim()
    assert model.last_outer_max_delta <= sbsim_model.OUTER_TOLERANCE
  np.testing.assert_allclose(
      one_cell.building.temp, reference.building.temp, atol=1e-8, rtol=0
  )
  assert (
      one_zone.active_heat_nodes
      == one_cell.active_heat_nodes - np.count_nonzero(air) + 1
  )
  assert np.ptp(one_zone.building.temp[air]) == 0
  assert np.ptp(reference.building.temp[air]) > 0


@pytest.mark.parametrize(
    'block_cells', [0, -1, 1.5, float('nan'), float('inf'), True]
)
def test_air_block_size_requires_a_positive_integer(block_cells):
  air = np.ones((2, 2), dtype=bool)
  with pytest.raises(ValueError, match='positive integer'):
    reduced_air.group_cells(
        np.arange(4), air, [np.argwhere(air)], block_cells=block_cells
    )


def test_native_unpadded_floor_plan_works_without_exterior_radiation():
  import pandas as pd
  from smart_control.simulator import weather_controller
  from . import floorplan, sbsim_model

  stamp = pd.Timestamp('2024-07-15', tz='UTC')
  weather = weather_controller.WeatherController(300.0, 300.0)
  fixture = sbsim_model.make_simulator(
      floorplan.make_plan(floorplan.PlanSpec(1)),
      weather,
      stamp,
      solver='cached-lu',
  )
  grid = np.ones((5, 5), dtype=np.uint8)
  grid[1:-1, 1:-1] = 0
  building = reduced_air.VectorizedBuilding(
      cv_size_cm=50,
      floor_height_cm=300,
      initial_temp=294.0,
      floor_plan=grid,
      zone_map=grid.copy(),
      buffer_from_walls=0,
  )
  model = reduced_air.ReducedAirSimulator(
      building=building,
      hvac=fixture.hvac,
      weather_controller=weather,
      time_step_sec=300,
      convergence_threshold=1e-4,
      iteration_limit=100,
      iteration_warning=100,
      start_timestamp=stamp,
  )
  assert model.finite_differences_timestep(
      ambient_temperature=300.0, convection_coefficient=12.0
  )
  assert building.temp.shape == (7, 7)
  assert (
      model.active_heat_nodes
      == np.count_nonzero(building.len_neighbors >= 2) - 8
  )


@pytest.mark.parametrize('initial_dtype', [np.float64, np.float32, np.int32])
def test_fast_lu_keeps_full_grid_temperature_and_vav_state(initial_dtype):
  import pandas as pd
  from . import floorplan, sbsim_model
  from smart_control.simulator import weather_controller

  stamp = pd.Timestamp('2024-07-19 23:40', tz='America/Chicago')
  weather = weather_controller.WeatherController(304.0, 304.0)
  plan = floorplan.make_plan(floorplan.PlanSpec(5, window_width_m=2.0))
  models = [
      sbsim_model.make_simulator(plan, weather, stamp, office=True, solver=solver)
      for solver in ('cached-lu', 'fast-lu', 'optimized-lu', 'tuned-lu')
  ]
  for model in models:
    model.building.temp = model.building.temp.astype(initial_dtype)
  reference, *optimized = models
  for step in range(30):
    for model in models:
      model.hvac._schedule.comfort_temp_window = (288.0, 297.0 - step % 5)
      model.hvac._schedule.eco_temp_window = (289.0, 296.0 + step % 5)
      model.step_sim()
    for model in optimized:
      np.testing.assert_array_equal(model.building.temp, reference.building.temp)
      np.testing.assert_array_equal(model.building.input_q, reference.building.input_q)
  for model in optimized:
    assert model.factorization_count == reference.factorization_count == 1
    assert model.triangular_solve_count == reference.triangular_solve_count


@pytest.mark.parametrize('zone', ['America/Chicago', 'dateutil/America/Chicago'])
def test_fast_schedule_keeps_local_calendar_and_live_settings(zone):
  import pandas as pd
  from . import sbsim_model
  from smart_control.simulator import setpoint_schedule

  settings = dict(
      morning_start_hour=0,
      evening_start_hour=24,
      comfort_temp_window=(290, 295),
      eco_temp_window=(285, 300),
      time_zone=pd.Timestamp('2024-07-01', tz=zone).tz,
  )
  reference = setpoint_schedule.SetpointSchedule(**settings)
  optimized = sbsim_model._SharedSetpointSchedule(**settings)
  for stamp in map(pd.Timestamp, ['2024-07-15 01:00Z', '2024-07-15 01:00']):
    assert reference.get_temperature_window(
        stamp
    ) == optimized.get_temperature_window(stamp)
  stamp = pd.Timestamp('2024-07-15 13:00Z')
  assert optimized.get_temperature_window(stamp) == (290, 295)
  optimized.comfort_temp_window = (291, 296)
  assert optimized.get_temperature_window(stamp) == (291, 296)
  optimized.holidays.add(stamp.dayofyear)
  assert optimized.get_temperature_window(stamp) == (285, 300)


@pytest.mark.parametrize('solver', ['fast-lu', 'tuned-lu'])
def test_solar_only_keeps_solar_power_and_solves_once(solver):
  import pandas as pd
  from . import floorplan, sbsim_model
  from smart_control.simulator import building_radiation_utils
  from smart_control.simulator import weather_controller

  stamp = pd.Timestamp('2024-07-15 12:00', tz='UTC')
  plan = floorplan.make_plan(floorplan.PlanSpec(1, window_width_m=2))
  weather = weather_controller.WeatherController(300.0, 300.0)
  solar, manual = [
      sbsim_model.make_simulator(
          plan, weather, stamp, solver=solver, exterior_radiation=mode
      )
      for mode in ('solar-only', 'off')
  ]
  power = np.zeros_like(solar.building.temp)
  row, col = solar.building._room_dict['room_1'][0]
  power[row, col] = 500.0
  forcing = building_radiation_utils.ExteriorRadiationForcing(
      power, np.ones_like(power), np.zeros_like(power), 275.0
  )
  solar._prepare_exterior_radiation = lambda _: forcing
  original_q = solar.building.input_q
  manual.building.input_q = manual.building.input_q + power

  parameters = dict(ambient_temperature=300.0, convection_coefficient=12.0)
  for model in (solar, manual):
    assert model.finite_differences_timestep(**parameters)
    assert model.last_outer_iteration_count == 1
    assert model.triangular_solve_count == 1
  assert solar.building.input_q is original_q
  np.testing.assert_array_equal(solar.building.temp, manual.building.temp)


def test_batched_windows_preserve_power_and_order_at_shared_receivers():
  from . import sbsim_model

  targets = [
      (np.array([0, 1, 2]), np.array([12, 13, 14])),
      (np.array([3, 4]), np.array([13, 14])),
      (np.array([5, 6, 7]), np.array([13, 15])),
  ]
  incident = np.random.default_rng(0).uniform(0, 900, size=(4, 4))
  transmittance = 0.34
  expected = np.zeros_like(incident)
  for windows, air in targets:
    expected.flat[air] += transmittance * incident.flat[windows].sum() / len(air)
  batched = sbsim_model._WindowSolar(targets, transmittance)
  np.testing.assert_array_equal(batched(incident), expected)


@pytest.mark.parametrize('initial_dtype', [np.float64, np.int32])
def test_linear_longwave_keeps_incoming_radiation_and_uses_one_solve(initial_dtype):
  import pandas as pd
  from . import floorplan, sbsim_model
  from smart_control.simulator import building_radiation_utils
  from smart_control.simulator import weather_controller

  plan = floorplan.make_plan(floorplan.PlanSpec(1, window_width_m=2))
  stamp = pd.Timestamp('2024-07-15 12:00', tz='UTC')
  sim = sbsim_model.make_simulator(
      plan, weather_controller.WeatherController(300.0, 300.0), stamp,
      solver='tuned-lu', exterior_radiation='linear-longwave',
  )
  sim.building.temp = sim.building.temp.astype(initial_dtype)
  previous = sim.building.temp.astype(np.float64)
  shape = previous.shape
  emission = np.zeros(shape)
  emission[1, 3] = 8e-8
  solar = np.zeros(shape)
  solar[1, 3] = 20.0
  radiant, ambient = 280.0, 300.0
  sim._prepare_exterior_radiation = (
      lambda _: building_radiation_utils.ExteriorRadiationForcing(
          solar, emission, np.zeros(shape), radiant,
      )
  )
  if initial_dtype == np.int32:
    # Exercise adding the longwave diagonal after an ordinary LU exists.
    sim.update_temperature_estimates(
        sim.building.temp, ambient_temperature=ambient, convection_coefficient=12.0
    )
  prior_solves = sim.triangular_solve_count
  assert sim.finite_differences_timestep(
      ambient_temperature=ambient, convection_coefficient=12.0
  )
  static = sim._static
  active = static['solved']
  temperature = sim.building.temp.ravel()[active]
  expected = (
      static['storage'] * previous.ravel()[active]
      + static['boundary_conductance'] * ambient + solar.ravel()[active]
      + emission.ravel()[active] * (radiant**4 - previous.ravel()[active]**4)
  )
  actual = static['a_cond'] @ temperature + (
      4 * emission.ravel()[active] * sbsim_model.LINEAR_LONGWAVE_REFERENCE_K**3
      * (temperature - previous.ravel()[active])
  )
  np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
  assert sim.last_outer_iteration_count == 1
  assert sim.triangular_solve_count - prior_solves == 1


def test_condensed_envelope_matches_grouped_walls_with_exterior_radiation():
  import pandas as pd
  from . import floorplan, sbsim_model
  from smart_control.simulator import solar_radiation
  from smart_control.simulator import weather_controller

  stamp = pd.Timestamp('2024-07-15 12:00', tz='UTC')
  plan = floorplan.make_plan(floorplan.PlanSpec(5, window_width_m=2))

  class SunnyWeather(weather_controller.WeatherController):
    def get_exterior_radiation(self, timestamp, ambient_temperature):
      return solar_radiation.ExteriorRadiationData(
          timestamp=timestamp,
          ambient_temp_k=ambient_temperature,
          sky_temp_k=275.0,
          irradiance=solar_radiation.IrradianceComponents(
              ghi=550.0, dni=400.0, dhi=150.0, solar_zenith=50, solar_azimuth=150
          ),
      )

  weather = SunnyWeather(304.0, 304.0)
  grouped, condensed = [
      sbsim_model.make_simulator(plan, weather, stamp, office=True, solver=solver)
      for solver in ('reduced-envelope', 'condensed-envelope')
  ]
  for _ in range(24):
    for model in (grouped, condensed):
      model.step_sim()
      assert model.last_outer_max_delta <= sbsim_model.OUTER_TOLERANCE
    np.testing.assert_allclose(
        grouped.building.temp, condensed.building.temp, atol=2e-6, rtol=0
    )
  assert grouped.active_heat_nodes == condensed.active_heat_nodes
  assert 0 < condensed.iterated_surface_nodes < condensed.active_heat_nodes
