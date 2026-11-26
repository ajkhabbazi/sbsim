"""Checks for the public grouped heat balance and its solver options."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from smart_control.simulator import air_handler
from smart_control.simulator import boiler
from smart_control.simulator import building
from smart_control.simulator import constants
from smart_control.simulator import grouped_jax_simulator
from smart_control.simulator import hvac_floorplan_based
from smart_control.simulator import setpoint_schedule
from smart_control.simulator import solar_radiation
from smart_control.simulator import weather_controller


class GroupedJaxSimulatorTest(parameterized.TestCase):

  def make_sim(
      self, *, native_mass_with_glazing=False, with_glazing=False, **kwargs
  ):
    plan = np.full((12, 14), 2)
    plan[1:-1, 1:-1] = 1
    plan[2:-2, 2:-2] = 0
    building_options = {}
    if with_glazing or native_mass_with_glazing:
      plan[1, 5] = constants.FENESTRATION_VALUE_IN_FILE_INPUT
    if native_mass_with_glazing:
      building_options.update(
          include_interior_mass=True,
          interior_mass_properties=building.MaterialProperties(
              5.0, 300.0, 2000.0
          ),
      )
    floor = building.FloorPlanBasedBuilding(
        cv_size_cm=50,
        floor_height_cm=300,
        initial_temp=295.65,
        floor_plan=plan,
        zone_map=plan.copy(),
        buffer_from_walls=0,
        include_exterior_radiative_heat_transfer=True,
        **building_options,
    )
    start = pd.Timestamp('2024-07-15 12:00', tz='UTC')
    hvac = hvac_floorplan_based.FloorPlanBasedHvac(
        air_handler.AirHandler(0.85, 288.15, 289.15, 600.0, 0.7, 1.0),
        boiler.Boiler(350.0, 3.0, 0.6),
        setpoint_schedule.SetpointSchedule(
            6, 22, (294.15, 297.15), (292.15, 299.15), time_zone=start.tz
        ),
        0.1,
        0.005,
        list(floor.get_zone_average_temps()),
    )
    return grouped_jax_simulator.GroupedJaxSimulator(
        floor,
        hvac,
        weather_controller.WeatherController(293.15, 300.15),
        300,
        1e-6,
        100,
        100,
        start,
        **kwargs,
    )

  def rollout(self, sim):
    temperature = []
    name = next(iter(sim.hvac.vavs))
    for watts in (1200.0, 300.0, -850.0, 20.0, 0.0, 600.0):
      sim.building.apply_thermal_power_zone(name, watts)
      self.assertTrue(
          sim.finite_differences_timestep(
              ambient_temperature=300.15,
              convection_coefficient=12.0,
          )
      )
      temperature.append(sim.state_temperatures.copy())
    return np.stack(temperature)

  @parameterized.parameters('sparse_lu', 'cg', 'dense', 'spsolve')
  def test_linear_mass_is_joint_and_single_outer_solve(self, method):
    options = dict(
        include_interior_mass=True,
        include_interior_radiative_heat_transfer=True,
        include_exterior_radiative_heat_transfer=True,
    )
    ref = self.make_sim(solver_method='sparse_lu', **options)
    sim = self.make_sim(solver_method=method, **options)
    np.testing.assert_allclose(
        self.rollout(sim), self.rollout(ref), atol=2e-7, rtol=0
    )
    self.assertEqual(sim.last_solve_count, 1)
    self.assertEqual(sim.mass_node_count, 1)
    self.assertGreater(np.ptp(sim.state_temperatures), 0.0)
    sim.reset()
    self.assertIsNone(sim.state_temperatures)
    ref.reset()
    np.testing.assert_allclose(
        self.rollout(sim), self.rollout(ref), atol=2e-7, rtol=0
    )

  def test_optional_physics_and_full_size(self):
    grouped = self.make_sim(
        include_interior_mass=False,
        include_interior_radiative_heat_transfer=False,
        include_exterior_radiative_heat_transfer=False,
    )
    full = self.make_sim(
        grouped=False,
        include_interior_mass=True,
        include_interior_radiative_heat_transfer=True,
        include_exterior_radiative_heat_transfer=True,
        linearized_radiative_heat_transfer=False,
        solver_method='cg',
    )
    self.rollout(grouped)
    self.rollout(full)
    self.assertEqual(grouped.mass_node_count, 0)
    self.assertIsNone(grouped.mass_temperature)
    self.assertGreater(full.mass_node_count, 1)
    self.assertGreater(full.thermal_node_count, grouped.thermal_node_count)
    self.assertGreater(full.last_solve_count, 1)

  def test_runtime_boundary_changes_trigger_new_factorization(self):
    sim = self.make_sim(
        include_interior_mass=True,
        include_interior_radiative_heat_transfer=False,
        include_exterior_radiative_heat_transfer=False,
    )
    sim.prepare_thermal_system(12)
    first_shape = sim.group_map.shape
    self.assertEqual(sim.factorization_count, 1)
    sim.prepare_thermal_system(12)
    self.assertEqual(sim.factorization_count, 1)
    sim.prepare_thermal_system(6)
    self.assertEqual(sim.factorization_count, 2)
    self.assertEqual(sim.group_map.shape, first_shape)

  def test_native_mass_receives_projected_window_solar_heat(self):
    sim = self.make_sim(native_mass_with_glazing=True)
    sim.prepare_thermal_system(12.0)
    system = sim._grouped_system
    weather = solar_radiation.ExteriorRadiationData(
        sim.current_timestamp,
        305.0,
        290.0,
        solar_radiation.IrradianceComponents(600.0, 500.0, 100.0, 45.0, 180.0),
    )
    poa = (350.0, 200.0, 500.0, 400.0)
    native = sim._build_exterior_radiation_geometry().forcing(
        weather,
        poa_irradiance=poa,
    )
    self.assertGreater(native.mass_gain.sum(), 0.0)
    expected = np.zeros(system.storage.size)
    np.add.at(
        expected,
        system.groups,
        native.shortwave_air_gain.ravel()[system.active],
    )
    np.add.at(
        expected,
        system.mass_nodes,
        native.mass_gain.ravel()[system.mass_active],
    )
    actual = system.geometry.forcing(weather, poa_irradiance=poa)
    np.testing.assert_allclose(
        actual.shortwave_air_gain + actual.mass_gain, expected
    )

  @parameterized.parameters(False, True)
  def test_effective_mass_receives_transmitted_window_solar_by_default(
      self, grouped
  ):
    sim = self.make_sim(
        with_glazing=True,
        grouped=grouped,
        include_interior_mass=True,
        interior_mass=grouped_jax_simulator.InteriorMass(),
        include_exterior_radiative_heat_transfer=True,
    )
    sim.prepare_thermal_system(12.0)
    system = sim._grouped_system
    weather = solar_radiation.ExteriorRadiationData(
        sim.current_timestamp,
        305.0,
        290.0,
        solar_radiation.IrradianceComponents(600.0, 500.0, 100.0, 45.0, 180.0),
    )
    poa = (350.0, 200.0, 500.0, 400.0)
    native = sim._build_exterior_radiation_geometry().forcing(
        weather, poa_irradiance=poa
    )
    actual = system.geometry.forcing(weather, poa_irradiance=poa)
    total_gain = actual.shortwave_air_gain + actual.mass_gain
    air_mask = (
        sim.building.floor_plan.ravel()[system.active]
        == constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT
    )
    air_nodes = np.unique(system.groups[air_mask])
    mass_nodes = np.unique(system.mass_nodes)
    np.testing.assert_allclose(total_gain[air_nodes], 0.0, atol=1e-12)
    expected_transmitted = native.shortwave_air_gain.ravel()[
        system.mass_active
    ].sum()
    self.assertGreater(expected_transmitted, 0.0)
    np.testing.assert_allclose(
        total_gain[mass_nodes].sum(), expected_transmitted
    )

    sim_to_air = self.make_sim(
        with_glazing=True,
        grouped=grouped,
        include_interior_mass=True,
        interior_mass=grouped_jax_simulator.InteriorMass(),
        include_exterior_radiative_heat_transfer=True,
        solar_to_interior_mass=False,
    )
    sim_to_air.prepare_thermal_system(12.0)
    air_forcing = sim_to_air._grouped_system.geometry.forcing(
        weather, poa_irradiance=poa
    )
    air_total = air_forcing.shortwave_air_gain + air_forcing.mass_gain
    np.testing.assert_allclose(air_total[mass_nodes], 0.0, atol=1e-12)
    np.testing.assert_allclose(air_total[air_nodes].sum(), expected_transmitted)

  def test_standard_vav_interface_advances_one_interval(self):
    sim = self.make_sim(include_exterior_radiative_heat_transfer=False)
    start = sim.current_timestamp
    sim.step_sim()
    self.assertEqual(sim.current_timestamp, start + pd.Timedelta(minutes=5))
    self.assertEqual(
        set(sim.building.get_zone_average_temps()), set(sim.hvac.vavs)
    )
    self.assertTrue(np.isfinite(sim.building.temp).all())


if __name__ == '__main__':
  absltest.main()
