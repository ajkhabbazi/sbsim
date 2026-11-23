"""Tests for the JAX sparse direct heat-balance simulator."""

import copy
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from smart_control.simulator import jax_simulator
from smart_control.simulator import numpy_scipy_simulator
from smart_control.simulator import simulator_flexible_floor_plan_test
from smart_control.simulator import weather_controller as weather_controller_py


class JaxSimulatorTest(parameterized.TestCase):

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

  @parameterized.product(
      include_interior_mass=(False, True),
      include_radiation=(False, True),
      solver_method=("spsolve", "cg"),
  )
  def test_matches_scipy_direct_solver(
      self, include_interior_mass, include_radiation, solver_method
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
    jax_direct = jax_simulator.JaxSimulator(
        *self._simulator_args(copy.deepcopy(building)),
        relative_convergence_threshold=None,
        solver_method=solver_method,
        cg_tolerance=1e-12,
    )
    for simulator in (direct, jax_direct):
      source = tuple(np.argwhere(simulator.building.diffusers > 0.0)[0])
      simulator.building.input_q[source] = 50.0

    # CG stops at a relative residual, which leaves a small absolute error
    # where the storage term makes the smallest eigenvalue small.
    atol = 1e-6 if solver_method == "cg" else 1e-7
    # Several steps so that the cached static system is reused.
    for _ in range(3):
      for simulator in (direct, jax_direct):
        self.assertTrue(
            simulator.finite_differences_timestep(
                ambient_temperature=305.0, convection_coefficient=12.0
            )
        )
      np.testing.assert_allclose(
          jax_direct.building.temp, direct.building.temp, atol=atol, rtol=0.0
      )
      if include_interior_mass:
        np.testing.assert_allclose(
            jax_direct.building.interior_mass_temp,
            direct.building.interior_mass_temp,
            atol=atol,
            rtol=0.0,
        )

  def test_cg_records_iterations(self):
    fixture = (
        simulator_flexible_floor_plan_test.FlexibleFloorplanSimulatorTest()
    )
    _, building = fixture._create_simulator_and_building(  # pylint: disable=protected-access
        initial_temp=292.0,
        include_interior_mass=False,
        include_radiative_heat_transfer=False,
        convergence_threshold=1e-4,
        iteration_limit=2000,
    )
    simulator = jax_simulator.JaxSimulator(
        *self._simulator_args(building),
        relative_convergence_threshold=None,
        solver_method="cg",
    )
    simulator.building.input_q[
        tuple(np.argwhere(simulator.building.diffusers > 0.0)[0])
    ] = 50.0
    self.assertTrue(
        simulator.finite_differences_timestep(
            ambient_temperature=305.0, convection_coefficient=12.0
        )
    )
    self.assertGreater(simulator.last_cg_iteration_count, 0)

  def test_cg_raises_when_iteration_limit_is_too_small(self):
    fixture = (
        simulator_flexible_floor_plan_test.FlexibleFloorplanSimulatorTest()
    )
    _, building = fixture._create_simulator_and_building(  # pylint: disable=protected-access
        initial_temp=292.0,
        include_interior_mass=False,
        include_radiative_heat_transfer=False,
        convergence_threshold=1e-4,
        iteration_limit=2000,
    )
    simulator = jax_simulator.JaxSimulator(
        *self._simulator_args(building),
        relative_convergence_threshold=None,
        solver_method="cg",
        cg_iteration_limit=1,
    )
    with self.assertRaisesRegex(RuntimeError, "did not converge"):
      simulator.finite_differences_timestep(
          ambient_temperature=305.0, convection_coefficient=12.0
      )

  def test_rejects_unknown_solver_method(self):
    fixture = (
        simulator_flexible_floor_plan_test.FlexibleFloorplanSimulatorTest()
    )
    _, building = fixture._create_simulator_and_building(  # pylint: disable=protected-access
        initial_temp=292.0
    )
    with self.assertRaisesRegex(ValueError, "solver_method"):
      jax_simulator.JaxSimulator(
          *self._simulator_args(building), solver_method="lu"
      )


if __name__ == "__main__":
  absltest.main()
