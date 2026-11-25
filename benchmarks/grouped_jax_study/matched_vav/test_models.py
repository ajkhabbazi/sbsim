"""Regression checks for the paired model inputs."""

import numpy as np
import pytest

from . import floorplan


def test_refining_cells_preserves_five_zone_geometry_and_windows():
  coarse = floorplan.make_plan(floorplan.PlanSpec(5, window_width_m=2))
  fine = floorplan.make_plan(
      floorplan.PlanSpec(5, cv_size_m=0.25, window_width_m=2)
  )

  assert coarse.rooms == fine.rooms
  assert [room.name for room in coarse.rooms] == [
      f'room_{i}' for i in range(1, 6)
  ]
  assert (
      sum(
          wall.neighbor is not None
          for room in coarse.rooms
          for wall in room.walls
      )
      == 10
  )
  assert (
      sum(
          wall.window is not None
          for room in coarse.rooms
          for wall in room.walls
      )
      == 10
  )
  assert coarse.rooms[4].bounds_m == (6.0, 0.5, 11.0, 4.5)
  assert coarse.rooms[4].walls[1].neighbor is None
  assert coarse.grid.shape == (21, 36)
  assert np.count_nonzero(coarse.zone_map == 0) == 400
  assert np.count_nonzero(fine.zone_map == 0) == 1600
  assert np.count_nonzero(coarse.grid == 4) == 40
  assert np.count_nonzero(fine.grid == 4) == 160
  assert np.count_nonzero(coarse.zone_map == 4) == 0


def test_dimensions_must_match_grid():
  with pytest.raises(ValueError, match='wall_thickness_m'):
    floorplan.make_plan(floorplan.PlanSpec(5, cv_size_m=0.4))


def test_energyplus_uses_exact_room_geometry_and_one_vav_per_room():
  import datetime
  from . import energyplus_model

  plan = floorplan.make_plan(floorplan.PlanSpec(5, window_width_m=2))
  model = energyplus_model.make_model(plan, datetime.date(2024, 7, 15), '24.2')
  assert len(model['AirLoopHVAC']) == 1
  assert len(model['AirTerminal:SingleDuct:VAV:Reheat']) == 5
  assert len(model['FenestrationSurface:Detailed']) == 10
  assert model['Zone']['room_5']['floor_area'] == 20
  assert model['Zone']['room_5']['volume'] == 60
  walls = model['BuildingSurface:Detailed']
  internal = {
      name: wall
      for name, wall in walls.items()
      if wall['outside_boundary_condition'] == 'Surface'
  }
  assert len(internal) == 10
  for name, wall in internal.items():
    opposite = internal[wall['outside_boundary_condition_object']]
    assert opposite['outside_boundary_condition_object'] == name
  north = walls['room_5 north']
  south = walls['room_2 south']
  assert north['outside_boundary_condition_object'] == 'room_2 south'
  assert south['outside_boundary_condition_object'] == 'room_5 north'
  assert {v['vertex_y_coordinate'] for v in north['vertices']} == {4.5}
  assert {v['vertex_y_coordinate'] for v in south['vertices']} == {5.0}
  assert walls['room_5 east']['outside_boundary_condition'] == 'Outdoors'
  assert (
      model['ZoneHVAC:EquipmentConnections']['room_5 Connections'][
          'zone_air_inlet_node_or_nodelist_name'
      ]
      == model['AirTerminal:SingleDuct:VAV:Reheat']['room_5 VAV'][
          'air_outlet_node_name'
      ]
  )


def test_sbsim_resolves_one_vav_per_room_and_advances_normal_step():
  import pandas as pd
  from . import sbsim_model
  from smart_control.simulator import weather_controller

  plan = floorplan.make_plan(floorplan.PlanSpec(5, window_width_m=2))
  start = pd.Timestamp('2024-07-15', tz='UTC')
  simulator = sbsim_model.make_simulator(
      plan, weather_controller.WeatherController(305, 305), start
  )
  radiation = simulator._build_exterior_radiation_geometry()
  faces = radiation.face_areas.reshape(4, -1)
  from smart_control.simulator import building_radiation_utils

  compass = building_radiation_utils.EXTERIOR_FACE_AZIMUTHS
  room5 = np.ravel_multi_index(
      tuple(zip(*simulator.building._room_dict['room_5'])), plan.grid.shape
  )
  room5_windows = [
      glazing
      for glazing, room in radiation.window_targets
      if np.array_equal(np.sort(room), np.sort(room5))
  ]
  assert len(radiation.window_targets) == 10
  assert len(room5_windows) == 2
  directions = {}
  for glazing in room5_windows:
    area = faces[:, glazing].sum(axis=1)
    active = np.flatnonzero(area)
    assert len(active) == 1
    i = active[0]
    directions[compass[i]] = area[i] / plan.spec.height_m
  assert directions == {90: 2, 180: 2}
  before = simulator.building.temp.copy()
  simulator.step_sim()
  assert list(simulator.hvac.vavs) == [f'room_{i}' for i in range(1, 6)]
  assert simulator.current_timestamp == start + pd.Timedelta(minutes=5)
  assert all(vav.flow_rate_demand > 0 for vav in simulator.hvac.vavs.values())
  assert np.isfinite(simulator.building.temp).all()
  assert np.max(np.abs(simulator.building.temp - before)) > 0
  for _ in range(36):
    simulator.step_sim()
    indoor = simulator.building.temp[plan.zone_map == 0] - 273.15
    assert np.isfinite(indoor).all()
    assert 0 < indoor.min() <= indoor.max() < 50


def test_generation_round_trip_and_rejects_mismatched_arrays(tmp_path):
  import datetime
  from . import generate

  generate.write_models(
      tmp_path,
      floorplan.PlanSpec(5, window_width_m=2),
      datetime.date(2024, 7, 15),
      '24.2',
  )
  plan, run_day = generate.load_plan(tmp_path)
  assert run_day == datetime.date(2024, 7, 15)
  assert len(plan.rooms) == 5
  assert np.count_nonzero(plan.grid == 4) == 40
  assert (tmp_path / 'model.epJSON').is_file()
  plan.grid[4, 4] = 2
  np.save(tmp_path / 'floor_plan.npy', plan.grid)
  with pytest.raises(ValueError, match='floor_plan.npy'):
    generate.load_plan(tmp_path)


def test_epw_rejects_missing_hourly_solar_before_interpolating(tmp_path):
  from datetime import date
  from . import sbsim_model

  epw = tmp_path / 'missing_solar.epw'
  headers = ['LOCATION,Test,IL,USA,TMY3,1234,41.98,-87.9,-6,200']
  rows = []
  for day in (14, 15, 16):
    for hour in range(1, 25):
      ghi = 9999 if (day, hour) == (15, 12) else 200
      values = [
          2024,
          7,
          day,
          hour,
          60,
          '?',
          25,
          10,
          50,
          100000,
          0,
          0,
          300,
          ghi,
          100,
          100,
      ] + [0] * 19
      rows.append(','.join(map(str, values)))
  epw.write_text('\n'.join(headers + ['TEST'] * 7 + rows) + '\n')
  with pytest.raises(ValueError, match='missing or invalid samples'):
    sbsim_model.EpwWeather(epw, date(2024, 7, 15))


def test_office_model_keeps_geometry_and_adds_active_load_and_outdoor_air():
  from datetime import date
  from . import energyplus_model
  from . import office_loads

  plan = floorplan.make_plan(floorplan.PlanSpec(16, window_width_m=2))
  day = date(2024, 7, 15)
  baseline = energyplus_model.make_model(plan, day, '24.2')
  office = energyplus_model.make_model(plan, day, '24.2', office=True)
  assert (
      office['BuildingSurface:Detailed'] == baseline['BuildingSurface:Detailed']
  )
  assert (
      office['FenestrationSurface:Detailed']
      == baseline['FenestrationSurface:Detailed']
  )
  assert (
      len(office['People'])
      == len(office['Lights'])
      == len(office['ElectricEquipment'])
      == 16
  )
  assert len(office['AirLoopHVAC:OutdoorAirSystem']) == 1
  assert office_loads.FRESH_AIR_FRACTION[12] == 0.20
  assert office_loads.FRESH_AIR_FRACTION[0] == 0.05
  room = plan.rooms[0]
  assert (
      office_loads.sensible_watts(room, 10)
      > office_loads.sensible_watts(room, 0)
      > 0
  )
  assert office_loads.sensible_watts(
      plan.rooms[6], 10
  ) != office_loads.sensible_watts(room, 10)


def test_office_gains_heat_air_and_do_not_accumulate_in_hvac_forcing():
  import pandas as pd
  from . import sbsim_model
  from smart_control.simulator import weather_controller

  plan = floorplan.make_plan(floorplan.PlanSpec(1))
  stamp = pd.Timestamp('2024-07-15 10:00', tz='UTC')
  weather = weather_controller.WeatherController(295.65, 295.65)
  baseline = sbsim_model.make_simulator(plan, weather, stamp)
  office = sbsim_model.make_simulator(plan, weather, stamp, office=True)

  baseline.step_sim()
  office.step_sim()

  assert office.hvac.air_handler.recirculation == 0.8
  assert office.building.get_zone_average_temps()['room_1'] > (
      baseline.building.get_zone_average_temps()['room_1'] + 0.1
  )
  np.testing.assert_array_equal(
      office.building.input_q, baseline.building.input_q
  )
