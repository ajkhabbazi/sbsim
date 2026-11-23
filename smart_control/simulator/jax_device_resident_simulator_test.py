"""Tests for the device-resident JAX heat-balance simulator."""

import copy
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from smart_control.simulator import jax_device_resident_simulator
from smart_control.simulator import numpy_scipy_simulator
from smart_control.simulator import simulator_flexible_floor_plan_test
from smart_control.simulator import weather_controller as weather_controller_py


class JaxDeviceResidentSimulatorTest(parameterized.TestCase):

  def _simulator_args(self, building):
    fixture = (
        simulator_flexible_floor_plan_test.FlexibleFloorplanSimulatorTest()
    )
    return (
        building,
        fixture._create_small_hvac(),  # pylint: disable=protected-access
        mock.create_autospec(weather_controller_py.WeatherController),
        300.0,
        1e-4,
        2000,
        2000,
        pd.Timestamp("2012-06-21 06:00:00"),
    )

  def _building(self, include_interior_mass, include_radiation):
    fixture = (
        simulator_flexible_floor_plan_test.FlexibleFloorplanSimulatorTest()
    )
    _, building = fixture._create_simulator_and_building(  # pylint: disable=protected-access
        initial_temp=292.0,
        include_interior_mass=include_interior_mass,
        include_radiative_heat_transfer=include_radiation,
        convergence_threshold=1e-4,
        iteration_limit=2000,
    )
    building.conductivity[:] = 0.5
    if include_interior_mass:
      building.interior_mass_temp[building.interior_mass_mask] = 291.0
    return building

  @parameterized.product(
      include_interior_mass=(False, True),
      include_radiation=(False, True),
  )
  def test_matches_scipy_direct_solver(
      self, include_interior_mass, include_radiation
  ):
    building = self._building(include_interior_mass, include_radiation)
    direct = numpy_scipy_simulator.NumpyScipySimulator(
        *self._simulator_args(copy.deepcopy(building)),
        relative_convergence_threshold=None,
    )
    resident = jax_device_resident_simulator.JaxDeviceResidentSimulator(
        *self._simulator_args(copy.deepcopy(building)),
        relative_convergence_threshold=None,
        cg_tolerance=1e-12,
    )
    for simulator in (direct, resident):
      source = tuple(np.argwhere(simulator.building.diffusers > 0.0)[0])
      simulator.building.input_q[source] = 50.0
    resident.warmup(ambient_temperature=305.0, convection_coefficient=12.0)

    # Both iterate to the same Picard tolerance, so they agree to about that
    # tolerance rather than to round-off.
    for _ in range(3):
      for simulator in (direct, resident):
        self.assertTrue(
            simulator.finite_differences_timestep(
                ambient_temperature=305.0, convection_coefficient=12.0
            )
        )
      np.testing.assert_allclose(
          resident.building.temp, direct.building.temp, atol=1e-4, rtol=0.0
      )
      if include_interior_mass:
        np.testing.assert_allclose(
            resident.building.interior_mass_temp,
            direct.building.interior_mass_temp,
            atol=1e-4,
            rtol=0.0,
        )
    self.assertGreater(resident.last_cg_iteration_count, 0)
    self.assertGreater(resident.last_outer_iteration_count, 0)

  def test_keeps_temperature_on_device_until_explicit_sync(self):
    building = self._building(True, True)
    resident = jax_device_resident_simulator.JaxDeviceResidentSimulator(
        *self._simulator_args(building),
        relative_convergence_threshold=None,
        sync_to_host_each_step=False,
    )
    building.input_q[tuple(np.argwhere(building.diffusers > 0.0)[0])] = 50.0
    initial_temperature = building.temp.copy()
    resident.warmup(ambient_temperature=305.0, convection_coefficient=12.0)
    resident.finite_differences_timestep(
        ambient_temperature=305.0, convection_coefficient=12.0
    )
    np.testing.assert_array_equal(building.temp, initial_temperature)
    resident.sync_state_to_building()
    self.assertGreater(
        float(np.max(np.abs(building.temp - initial_temperature))), 0.0
    )

  def test_warmup_does_not_advance_state(self):
    building = self._building(False, True)
    resident = jax_device_resident_simulator.JaxDeviceResidentSimulator(
        *self._simulator_args(building),
        relative_convergence_threshold=None,
    )
    building.input_q[tuple(np.argwhere(building.diffusers > 0.0)[0])] = 50.0
    initial_temperature = building.temp.copy()
    resident.warmup(ambient_temperature=305.0, convection_coefficient=12.0)
    resident.sync_state_to_building()
    np.testing.assert_array_equal(building.temp, initial_temperature)


if __name__ == "__main__":
  absltest.main()
