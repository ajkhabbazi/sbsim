"""Tests for the coupled air/interior-mass simulators."""

import copy
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from smart_control.simulator import numpy_scipy_simulator
from smart_control.simulator import simulator_flexible_floor_plan_test
from smart_control.simulator import tf_coupled_interior_mass_simulator as coupled_simulator
from smart_control.simulator import tf_device_resident_simulator
from smart_control.simulator import weather_controller as weather_controller_py


class CoupledInteriorMassSimulatorTest(parameterized.TestCase):

  def _building(self, *, mass: bool, radiation: bool):
    fixture = (
        simulator_flexible_floor_plan_test.FlexibleFloorplanSimulatorTest()
    )
    _, building = fixture._create_simulator_and_building(  # pylint: disable=protected-access
        initial_temp=292.0,
        include_interior_mass=mass,
        include_radiative_heat_transfer=radiation,
        convergence_threshold=1e-4,
        iteration_limit=2000,
    )
    building.conductivity[:] = 0.5
    if mass:
      building.interior_mass_temp[building.interior_mass_mask] = 291.0
    return building

  def _args(self, building, *, iteration_limit=2000):
    fixture = (
        simulator_flexible_floor_plan_test.FlexibleFloorplanSimulatorTest()
    )
    return (
        building,
        fixture._create_small_hvac(),  # pylint: disable=protected-access
        mock.create_autospec(weather_controller_py.WeatherController),
        300.0,
        1e-4,
        iteration_limit,
        iteration_limit,
        pd.Timestamp("2012-06-21 06:00:00"),
    )

  @parameterized.parameters(False, True)
  def test_no_mass_delegates_to_device_resident_parent(self, radiation):
    building = self._building(mass=False, radiation=radiation)
    parent = tf_device_resident_simulator.TFDeviceResidentSimulator(
        *self._args(copy.deepcopy(building)),
        relative_convergence_threshold=None,
        cg_tolerance=1e-8,
    )
    coupled = coupled_simulator.TFCoupledInteriorMassSimulator(
        *self._args(copy.deepcopy(building)),
        relative_convergence_threshold=None,
        cg_tolerance=1e-8,
    )
    for simulator in (parent, coupled):
      simulator.warmup(ambient_temperature=305.0, convection_coefficient=12.0)
      self.assertTrue(
          simulator.finite_differences_timestep(
              ambient_temperature=305.0, convection_coefficient=12.0
          )
      )
    np.testing.assert_array_equal(coupled.building.temp, parent.building.temp)

  @parameterized.parameters(False, True)
  def test_coupled_scipy_matches_legacy_physical_solution(self, radiation):
    building = self._building(mass=True, radiation=radiation)
    legacy = numpy_scipy_simulator.NumpyScipySimulator(
        *self._args(copy.deepcopy(building)),
        relative_convergence_threshold=None,
    )
    coupled = coupled_simulator.NumpyScipyCoupledInteriorMassSimulator(
        *self._args(copy.deepcopy(building)),
        relative_convergence_threshold=None,
    )
    for simulator in (legacy, coupled):
      self.assertTrue(
          simulator.finite_differences_timestep(
              ambient_temperature=305.0, convection_coefficient=12.0
          )
      )
    np.testing.assert_allclose(
        coupled.building.temp, legacy.building.temp, atol=2e-4, rtol=0.0
    )
    np.testing.assert_allclose(
        coupled.building.interior_mass_temp,
        legacy.building.interior_mass_temp,
        atol=2e-4,
        rtol=0.0,
    )
    self.assertLessEqual(
        coupled.last_outer_iteration_count, legacy.last_outer_iteration_count
    )

  @parameterized.parameters(False, True)
  def test_pcg_and_scipy_use_equivalent_coupled_system(self, radiation):
    building = self._building(mass=True, radiation=radiation)
    pcg = coupled_simulator.TFCoupledInteriorMassSimulator(
        *self._args(copy.deepcopy(building)),
        relative_convergence_threshold=None,
        cg_tolerance=1e-8,
    )
    direct = coupled_simulator.NumpyScipyCoupledInteriorMassSimulator(
        *self._args(copy.deepcopy(building)),
        relative_convergence_threshold=None,
    )
    pcg.warmup(ambient_temperature=305.0, convection_coefficient=12.0)
    for simulator in (pcg, direct):
      self.assertTrue(
          simulator.finite_differences_timestep(
              ambient_temperature=305.0, convection_coefficient=12.0
          )
      )
    np.testing.assert_allclose(
        pcg.building.temp, direct.building.temp, atol=1e-3, rtol=0.0
    )
    np.testing.assert_allclose(
        pcg.building.interior_mass_temp,
        direct.building.interior_mass_temp,
        atol=1e-4,
        rtol=0.0,
    )

  def test_mass_storage_rhs_is_fixed_and_exchange_conserves_energy(self):
    building = self._building(mass=True, radiation=True)
    simulator = coupled_simulator.NumpyScipyCoupledInteriorMassSimulator(
        *self._args(building), relative_convergence_threshold=None
    )
    old_air = building.temp.copy()
    old_mass = building.interior_mass_temp.copy()
    first = coupled_simulator._assemble_coupled_scipy_system(  # pylint: disable=protected-access
        simulator,
        old_air,
        old_mass,
        old_air,
        old_mass,
        305.0,
        12.0,
    )
    changed_air = old_air + 4.0
    changed_mass = old_mass.copy()
    changed_mass[building.interior_mass_mask] += 7.0
    second = coupled_simulator._assemble_coupled_scipy_system(  # pylint: disable=protected-access
        simulator,
        changed_air,
        changed_mass,
        old_air,
        old_mass,
        305.0,
        12.0,
    )
    mass_count = int(np.sum(building.interior_mass_mask))
    np.testing.assert_array_equal(
        first[1][-mass_count:], second[1][-mass_count:]
    )

    no_radiation = self._building(mass=True, radiation=False)
    direct = coupled_simulator.NumpyScipyCoupledInteriorMassSimulator(
        *self._args(no_radiation), relative_convergence_threshold=None
    )
    previous_mass = no_radiation.interior_mass_temp.copy()
    self.assertTrue(
        direct.finite_differences_timestep(
            ambient_temperature=305.0, convection_coefficient=12.0
        )
    )
    mask = no_radiation.interior_mass_mask
    floor_height = no_radiation.floor_height_cm / 100.0
    u = no_radiation.cv_size_cm / 100.0
    conductance = no_radiation.interior_mass_conductivity * u * u / floor_height
    storage = (
        no_radiation.interior_mass_density
        * no_radiation.interior_mass_heat_capacity
        * u
        * u
        * floor_height
        / 300.0
    )
    exchange_to_mass = conductance * (
        no_radiation.temp - no_radiation.interior_mass_temp
    )
    stored_by_mass = storage * (no_radiation.interior_mass_temp - previous_mass)
    np.testing.assert_allclose(
        exchange_to_mass[mask], stored_by_mass[mask], atol=1e-9, rtol=0.0
    )

  def test_multiple_steps_commit_mass_state_once_per_step(self):
    building = self._building(mass=True, radiation=False)
    simulator = coupled_simulator.NumpyScipyCoupledInteriorMassSimulator(
        *self._args(building), relative_convergence_threshold=None
    )
    states = [building.interior_mass_temp.copy()]
    for ambient in (305.0, 306.0, 307.0):
      self.assertTrue(
          simulator.finite_differences_timestep(
              ambient_temperature=ambient, convection_coefficient=12.0
          )
      )
      states.append(building.interior_mass_temp.copy())
    mask = building.interior_mass_mask
    for previous, current in zip(states, states[1:]):
      self.assertGreater(
          float(np.max(np.abs(current[mask] - previous[mask]))), 0.0
      )

  def test_failed_outer_iteration_does_not_commit_state(self):
    building = self._building(mass=True, radiation=True)
    simulator = coupled_simulator.NumpyScipyCoupledInteriorMassSimulator(
        *self._args(building, iteration_limit=1),
        relative_convergence_threshold=None,
    )
    old_air = building.temp.copy()
    old_mass = building.interior_mass_temp.copy()
    self.assertFalse(
        simulator.finite_differences_timestep(
            ambient_temperature=305.0, convection_coefficient=12.0
        )
    )
    np.testing.assert_array_equal(building.temp, old_air)
    np.testing.assert_array_equal(building.interior_mass_temp, old_mass)


if __name__ == "__main__":
  absltest.main()
