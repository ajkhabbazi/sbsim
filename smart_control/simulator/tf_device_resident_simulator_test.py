"""Tests for the device-resident TensorFlow heat-balance simulator."""

import copy
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from smart_control.simulator import numpy_scipy_simulator
from smart_control.simulator import simulator_flexible_floor_plan_test
from smart_control.simulator import tf_device_resident_simulator
from smart_control.simulator import weather_controller as weather_controller_py


class TFDeviceResidentSimulatorTest(parameterized.TestCase):

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

  @parameterized.parameters(
      (False, False),
      (False, True),
      (True, True),
  )
  def test_matches_sparse_direct_solver(
      self, include_interior_mass, include_radiation
  ):
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
    direct = numpy_scipy_simulator.NumpyScipySimulator(
        *self._simulator_args(copy.deepcopy(building)),
        relative_convergence_threshold=None,
    )
    resident = tf_device_resident_simulator.TFDeviceResidentSimulator(
        *self._simulator_args(copy.deepcopy(building)),
        relative_convergence_threshold=None,
        cg_tolerance=1e-8,
    )
    for simulator in (direct, resident):
      source = tuple(np.argwhere(simulator.building.diffusers > 0.0)[0])
      simulator.building.input_q[source] = 50.0
    resident.warmup(ambient_temperature=305.0, convection_coefficient=12.0)

    self.assertTrue(
        direct.finite_differences_timestep(
            ambient_temperature=305.0, convection_coefficient=12.0
        )
    )
    self.assertTrue(
        resident.finite_differences_timestep(
            ambient_temperature=305.0, convection_coefficient=12.0
        )
    )
    np.testing.assert_allclose(
        resident.building.temp, direct.building.temp, atol=1e-3, rtol=0.0
    )
    if include_interior_mass:
      np.testing.assert_allclose(
          resident.building.interior_mass_temp,
          direct.building.interior_mass_temp,
          atol=1e-4,
          rtol=0.0,
      )

  def test_keeps_temperature_on_device_until_explicit_sync(self):
    fixture = (
        simulator_flexible_floor_plan_test.FlexibleFloorplanSimulatorTest()
    )
    _, building = fixture._create_simulator_and_building(  # pylint: disable=protected-access
        initial_temp=292.0,
        convergence_threshold=1e-4,
        iteration_limit=2000,
    )
    building.conductivity[:] = 0.5
    resident = tf_device_resident_simulator.TFDeviceResidentSimulator(
        *self._simulator_args(building),
        relative_convergence_threshold=None,
        cg_tolerance=1e-8,
        sync_to_host_each_step=False,
    )
    source = tuple(np.argwhere(building.diffusers > 0.0)[0])
    building.input_q[source] = 50.0
    initial_temperature = building.temp.copy()
    resident.warmup(ambient_temperature=305.0, convection_coefficient=12.0)

    self.assertTrue(
        resident.finite_differences_timestep(
            ambient_temperature=305.0, convection_coefficient=12.0
        )
    )
    np.testing.assert_array_equal(building.temp, initial_temperature)
    resident.sync_state_to_building()
    self.assertGreater(
        float(np.max(np.abs(building.temp - initial_temperature))), 0.0
    )


if __name__ == "__main__":
  absltest.main()
