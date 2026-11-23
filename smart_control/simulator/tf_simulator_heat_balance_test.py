"""Tests for the face based heat balance of the TF finite difference solver.

Every control volume exchanges heat with a neighbor through the face the two
of them share, so the conductance of that face is the series combination of the
half cell on either side of it,

  Gamma_f = [l_P / (2 k_P) + l_N / (2 k_N)]^-1,

and a face that opens onto the ambient puts the convective film in series with
the outer half cell of the boundary CV,

  Gamma_inf,f = [1 / h_ext + l_P / (2 k_P)]^-1.

Taking the owner CV's own conductivity across its own full width instead makes
the two CVs sharing a face disagree about the heat crossing it, and using
h_ext on its own drops the conduction resistance of the outer half cell. The
first group of tests pins down both conductances, and the second group checks
whole-field solutions of the implicit Euler system against the finite
difference relaxation the simulator runs by default.
"""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from smart_control.simulator import numpy_scipy_simulator
from smart_control.simulator import simulator as simulator_py
from smart_control.simulator import simulator_flexible_floor_plan as flexible_py
from smart_control.simulator import simulator_flexible_floor_plan_test
from smart_control.simulator import tf_simulator as tf_simulator_py
from smart_control.simulator import tf_simulator_test
from smart_control.simulator import weather_controller as weather_controller_py


def _mask_neighbors(mask: np.ndarray) -> list[list[list[tuple[int, int]]]]:
  """Returns the neighbor table of the CVs marked in a boolean mask.

  Args:
    mask: True on the CVs that make up the building.

  Returns:
    For every CV of the grid, the coordinates of its marked neighbors. CVs
    that are not marked have none, which makes them exterior CVs.
  """
  rows, cols = mask.shape

  def _inside(i: int, j: int) -> bool:
    return 0 <= i < rows and 0 <= j < cols and bool(mask[i][j])

  return [
      [
          [
              (a, b)
              for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
              if _inside(a, b)
          ]
          if _inside(i, j)
          else []
          for j in range(cols)
      ]
      for i in range(rows)
  ]


def _solid_block_neighbors(
    shape: tuple[int, int],
    rows: tuple[int, int],
    cols: tuple[int, int],
) -> list[list[list[tuple[int, int]]]]:
  """Returns a neighbor table whose CVs fill one solid rectangular block.

  Args:
    shape: shape of the grid.
    rows: inclusive first and last row of the block.
    cols: inclusive first and last column of the block.

  Returns:
    For every CV of the grid, the coordinates of its in-block neighbors. CVs
    outside the block have none, which makes them exterior CVs.
  """
  mask = np.full(shape, False)
  mask[rows[0] : rows[1] + 1, cols[0] : cols[1] + 1] = True
  return _mask_neighbors(mask)


class FaceConductanceTest(absltest.TestCase):
  """Tests the conductances that carry heat across a face."""

  # A 4 x 5 block of CVs inside a 6 x 7 grid, so that the block has a ring of
  # boundary CVs around a 2 x 3 patch of interior CVs.
  SHAPE = (6, 7)
  BLOCK_ROWS = (1, 4)
  BLOCK_COLS = (1, 5)
  CV_SIZE_M = 0.2
  CONVECTION_COEFFICIENT = 12.0

  def _two_material_conductivity(self) -> np.ndarray:
    """Returns a conductivity field with a vertical material interface."""
    conductivity = np.full(self.SHAPE, 0.05, dtype=np.float32)
    conductivity[:, 3:] = 50.0
    return conductivity

  def _conductances(self, conductivity: np.ndarray):
    """Returns the geometry, the face and the ambient conductance tensors."""
    boundary_cv_mapping = tf_simulator_py.get_cv_mapping(
        _solid_block_neighbors(self.SHAPE, self.BLOCK_ROWS, self.BLOCK_COLS)
    )
    t_u, t_v = tf_simulator_py.get_cv_dimension_tensors(
        self.CV_SIZE_M, boundary_cv_mapping, self.SHAPE
    )
    face = tf_simulator_py.get_oriented_face_conductance_tensors(
        conductivity, t_u, t_v, boundary_cv_mapping
    )
    ambient = tf_simulator_py.get_oriented_ambient_conductance_tensors(
        self.CONVECTION_COEFFICIENT,
        conductivity,
        t_u,
        t_v,
        boundary_cv_mapping,
    )
    return (
        t_u.numpy(),
        t_v.numpy(),
        [tensor.numpy() for tensor in face],
        [tensor.numpy() for tensor in ambient],
    )

  def test_half_cell_resistance_is_length_over_twice_the_conductivity(self):
    conductivity = self._two_material_conductivity()
    boundary_cv_mapping = tf_simulator_py.get_cv_mapping(
        _solid_block_neighbors(self.SHAPE, self.BLOCK_ROWS, self.BLOCK_COLS)
    )
    t_u, t_v = tf_simulator_py.get_cv_dimension_tensors(
        self.CV_SIZE_M, boundary_cv_mapping, self.SHAPE
    )

    t_r_u, t_r_v = tf_simulator_py.get_half_cell_resistance_tensors(
        conductivity, t_u, t_v
    )

    np.testing.assert_allclose(
        t_r_u.numpy(), t_u.numpy() / (2.0 * conductivity), rtol=1e-6
    )
    np.testing.assert_allclose(
        t_r_v.numpy(), t_v.numpy() / (2.0 * conductivity), rtol=1e-6
    )

  def test_interior_face_conductance_is_the_two_half_cells_in_series(self):
    conductivity = self._two_material_conductivity()
    u, v, (left, right, top, bottom), _ = self._conductances(conductivity)

    # (2, 2) and (2, 3) are interior CVs on either side of the material
    # interface: insulating exterior wall on the left, well mixed air on the
    # right.
    expected = 1.0 / (
        u[2][2] / (2.0 * conductivity[2][2])
        + u[2][3] / (2.0 * conductivity[2][3])
    )
    self.assertAlmostEqual(right[2][2], expected, places=5)
    self.assertAlmostEqual(left[2][3], expected, places=5)

    # Vertically the material is the same on both sides of the face.
    expected_vertical = 1.0 / (
        v[2][3] / (2.0 * conductivity[2][3])
        + v[3][3] / (2.0 * conductivity[3][3])
    )
    np.testing.assert_allclose(bottom[2][3], expected_vertical, rtol=1e-5)
    np.testing.assert_allclose(top[3][3], expected_vertical, rtol=1e-5)

  def test_interior_face_conductance_is_dominated_by_the_insulator(self):
    """The series form is bounded by twice the smaller half-cell conductance."""
    conductivity = self._two_material_conductivity()
    u, _, (_, right, _, _), _ = self._conductances(conductivity)

    insulating_side = 2.0 * conductivity[2][2] / u[2][2]
    owner_cell_form = conductivity[2][3] / u[2][3]

    self.assertLess(right[2][2], insulating_side)
    # The shipped form took the owner CV's own conductivity, which for air
    # against an exterior wall is off by three orders of magnitude.
    self.assertLess(right[2][2] * 100.0, owner_cell_form)

  def test_interior_face_conductance_is_symmetric_across_every_face(self):
    """Both CVs sharing a face have to see the same conductance."""
    conductivity = self._two_material_conductivity()
    u, v, (left, right, top, bottom), _ = self._conductances(conductivity)

    # Only the CVs of the block take part in the balance; the ones around it
    # are exterior CVs that are held at the ambient temperature.
    inside = np.full(self.SHAPE, False)
    inside[
        self.BLOCK_ROWS[0] : self.BLOCK_ROWS[1] + 1,
        self.BLOCK_COLS[0] : self.BLOCK_COLS[1] + 1,
    ] = True
    horizontal = inside[:, :-1] & inside[:, 1:]
    vertical = inside[:-1, :] & inside[1:, :]

    # The balance uses the product of the face area and the conductance, and
    # a left or right face has area v * z while a top or bottom face has u * z.
    np.testing.assert_allclose(
        (v * right)[:, :-1][horizontal],
        (v * left)[:, 1:][horizontal],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        (u * bottom)[:-1, :][vertical], (u * top)[1:, :][vertical], rtol=1e-5
    )

  def test_shared_face_area_is_the_narrower_of_the_two_cvs(self):
    """Where a half width CV meets a full width one, both see one area."""
    # Cutting the top right corner off the block leaves (2, 4) without a top
    # neighbor, so it is half as tall as the interior CV (2, 3) beside it and
    # the two of them only touch over the shorter of their two faces.
    mask = np.full(self.SHAPE, False)
    mask[1:5, 1:6] = True
    mask[1, 4:6] = False
    boundary_cv_mapping = tf_simulator_py.get_cv_mapping(_mask_neighbors(mask))
    conductivity = np.full(self.SHAPE, 2.0, dtype=np.float32)
    t_u, t_v = tf_simulator_py.get_cv_dimension_tensors(
        self.CV_SIZE_M, boundary_cv_mapping, self.SHAPE
    )
    u, v = t_u.numpy(), t_v.numpy()
    left, right, _, _ = (
        tensor.numpy()
        for tensor in tf_simulator_py.get_oriented_face_conductance_tensors(
            conductivity, t_u, t_v, boundary_cv_mapping
        )
    )

    self.assertAlmostEqual(v[2][3], self.CV_SIZE_M, places=6)
    self.assertAlmostEqual(v[2][4], self.CV_SIZE_M / 2.0, places=6)
    series = 1.0 / (
        u[2][3] / (2.0 * conductivity[2][3])
        + u[2][4] / (2.0 * conductivity[2][4])
    )
    # The balance multiplies by the owner's own face area, so the CV that
    # overhangs the shared face carries the halved conductance.
    self.assertAlmostEqual(right[2][3], series / 2.0, places=4)
    self.assertAlmostEqual(left[2][4], series, places=4)
    self.assertAlmostEqual(
        v[2][3] * right[2][3], v[2][4] * left[2][4], places=5
    )

  def test_uniform_material_keeps_the_owner_cell_conductance(self):
    """On a uniform grid of a single material both forms agree."""
    conductivity = np.full(self.SHAPE, 2.0, dtype=np.float32)
    u, v, (left, right, top, bottom), _ = self._conductances(conductivity)

    # (2, 2), (2, 3) and (3, 3) are interior CVs, so all of these faces are
    # between two full width CVs.
    self.assertAlmostEqual(right[2][2], conductivity[2][2] / u[2][2], places=4)
    self.assertAlmostEqual(left[2][3], conductivity[2][3] / u[2][3], places=4)
    self.assertAlmostEqual(bottom[2][3], conductivity[2][3] / v[2][3], places=4)
    self.assertAlmostEqual(top[3][3], conductivity[3][3] / v[3][3], places=4)

  def test_boundary_cv_face_conductance_accounts_for_the_half_width(self):
    """A boundary CV is half as wide, so its half cell is half as resistive."""
    conductivity = np.full(self.SHAPE, 2.0, dtype=np.float32)
    u, _, (_, right, _, _), _ = self._conductances(conductivity)

    # (2, 1) is a left edge boundary CV, half as wide as the interior CV
    # (2, 2) next to it.
    self.assertAlmostEqual(u[2][1], self.CV_SIZE_M / 2.0, places=6)
    expected = 1.0 / (
        u[2][1] / (2.0 * conductivity[2][1])
        + u[2][2] / (2.0 * conductivity[2][2])
    )
    self.assertAlmostEqual(right[2][1], expected, places=4)

  def test_no_interior_conductance_on_faces_that_open_onto_the_ambient(self):
    conductivity = self._two_material_conductivity()
    (
        _,
        _,
        (left, right, top, bottom),
        (
            ambient_left,
            ambient_right,
            ambient_top,
            ambient_bottom,
        ),
    ) = self._conductances(conductivity)

    # (2, 1) is a left edge CV and (1, 1) is a top left corner CV.
    self.assertEqual(left[2][1], 0.0)
    self.assertGreater(ambient_left[2][1], 0.0)
    self.assertEqual(left[1][1], 0.0)
    self.assertEqual(top[1][1], 0.0)
    self.assertGreater(ambient_left[1][1], 0.0)
    self.assertGreater(ambient_top[1][1], 0.0)

    # A face is either an interior face or an ambient face, never both.
    for interior, ambient in (
        (left, ambient_left),
        (right, ambient_right),
        (top, ambient_top),
        (bottom, ambient_bottom),
    ):
      np.testing.assert_array_equal(interior * ambient, np.zeros(self.SHAPE))

  def test_ambient_conductance_adds_the_outer_half_cell(self):
    conductivity = self._two_material_conductivity()
    (
        u,
        v,
        _,
        (
            ambient_left,
            _,
            ambient_top,
            _,
        ),
    ) = self._conductances(conductivity)

    expected_left = 1.0 / (
        1.0 / self.CONVECTION_COEFFICIENT + u[2][1] / (2.0 * conductivity[2][1])
    )
    self.assertAlmostEqual(ambient_left[2][1], expected_left, places=5)

    expected_top = 1.0 / (
        1.0 / self.CONVECTION_COEFFICIENT + v[1][2] / (2.0 * conductivity[1][2])
    )
    self.assertAlmostEqual(ambient_top[1][2], expected_top, places=5)

  def test_ambient_conductance_stays_below_the_film_coefficient(self):
    """Two resistances in series conduct less than either one alone."""
    conductivity = self._two_material_conductivity()
    _, _, _, ambient = self._conductances(conductivity)

    for tensor in ambient:
      active = tensor > 0.0
      self.assertTrue(np.any(active))
      self.assertTrue(np.all(tensor[active] < self.CONVECTION_COEFFICIENT))

    # On the insulating side the half cell dominates the film by a wide
    # margin, so the exchange with the ambient is nothing like h_ext.
    ambient_left = ambient[0]
    self.assertLess(ambient_left[2][1], self.CONVECTION_COEFFICIENT / 10.0)

  def test_ambient_conductance_approaches_the_film_coefficient(self):
    """A conductive boundary CV leaves the film as the only resistance."""
    conductivity = np.full(self.SHAPE, 5.0e5, dtype=np.float32)
    _, _, _, (ambient_left, _, _, _) = self._conductances(conductivity)

    self.assertAlmostEqual(
        ambient_left[2][1], self.CONVECTION_COEFFICIENT, places=3
    )


class WholeFieldSolutionTest(parameterized.TestCase):
  """Tests the direct and CG solutions of the implicit Euler balance."""

  AMBIENT_TEMPERATURE = 285.0
  CONVECTION_COEFFICIENT = 12.0
  TIME_STEP_SEC = 300.0
  FEATURE_TIME_STEP_SEC = 30.0
  FEATURE_CONVERGENCE_THRESHOLD = 1e-4
  FEATURE_ITERATION_LIMIT = 2000
  # Long enough that a couple of hundred steps reach the steady state.
  LONG_STEP_SEC = 86400.0

  def _create_simulator(
      self,
      backend,
      iteration_limit=200,
      time_step_sec=TIME_STEP_SEC,
  ):
    """Returns the requested solver over the small test building."""
    fixtures = tf_simulator_test.TFSimulatorTest()
    simulator_class = (
        numpy_scipy_simulator.NumpyScipySimulator
        if backend == 'numpy_scipy'
        else tf_simulator_py.TFSimulator
    )
    kwargs = {}
    if backend != 'numpy_scipy':
      kwargs['solver_method'] = backend
    return simulator_class(
        fixtures._create_test_building(),  # pylint: disable=protected-access
        fixtures._create_small_hvac(),  # pylint: disable=protected-access
        mock.create_autospec(weather_controller_py.WeatherController),
        time_step_sec,
        1e-6,
        iteration_limit,
        iteration_limit,
        pd.Timestamp('2012-12-21'),
        **kwargs,
    )

  def _create_feature_simulator(
      self,
      backend: str,
      include_interior_mass: bool,
      include_radiative_heat_transfer: bool,
  ):
    """Returns a controlled solver-comparison case with real thermal forcing."""
    fixture = (
        simulator_flexible_floor_plan_test.FlexibleFloorplanSimulatorTest()
    )
    _, building = fixture._create_simulator_and_building(  # pylint: disable=protected-access
        initial_temp=292.0,
        include_interior_mass=include_interior_mass,
        include_radiative_heat_transfer=include_radiative_heat_transfer,
        convergence_threshold=self.FEATURE_CONVERGENCE_THRESHOLD,
        iteration_limit=self.FEATURE_ITERATION_LIMIT,
    )
    building.conductivity[:] = 0.5

    simulator_class = (
        numpy_scipy_simulator.NumpyScipySimulator
        if backend == 'numpy_scipy'
        else tf_simulator_py.TFSimulator
    )
    kwargs = {}
    if backend != 'numpy_scipy':
      kwargs['solver_method'] = backend
    simulator = simulator_class(
        building,
        fixture._create_small_hvac(),  # pylint: disable=protected-access
        mock.create_autospec(weather_controller_py.WeatherController),
        self.FEATURE_TIME_STEP_SEC,
        self.FEATURE_CONVERGENCE_THRESHOLD,
        self.FEATURE_ITERATION_LIMIT,
        self.FEATURE_ITERATION_LIMIT,
        pd.Timestamp('2012-12-21'),
        relative_convergence_threshold=None,
        **kwargs,
    )
    source_coordinates = np.argwhere(simulator.building.diffusers > 0.0)
    source = tuple(source_coordinates[len(source_coordinates) // 2])
    simulator.building.input_q[source] = 50.0
    if include_interior_mass:
      simulator.building.interior_mass_temp[
          simulator.building.interior_mass_mask
      ] = np.linspace(
          286.0,
          298.0,
          int(np.sum(simulator.building.interior_mass_mask)),
      )
    return simulator

  def _ambient_conductances(self, simulator):
    """Returns the four ambient conductance tensors as numpy arrays."""
    tensors = tf_simulator_py.get_oriented_ambient_conductance_tensors(
        self.CONVECTION_COEFFICIENT,
        simulator.building.conductivity,
        simulator._t_u,  # pylint: disable=protected-access
        simulator._t_v,  # pylint: disable=protected-access
        simulator._boundary_cv_mapping,  # pylint: disable=protected-access
    )
    return [tensor.numpy().astype(np.float64) for tensor in tensors]

  def _face_areas(self, simulator):
    """Returns the area of the left, right, top and bottom face of every CV."""
    z = simulator.building.floor_height_cm / 100.0
    u = simulator._t_u.numpy().astype(np.float64)  # pylint: disable=protected-access
    v = simulator._t_v.numpy().astype(np.float64)  # pylint: disable=protected-access
    return v * z, v * z, u * z, u * z

  def _storage(self, simulator):
    """Returns the storage conductance rho c u v z / dt of every CV, in W/K."""
    building = simulator.building
    return (
        building.density
        * building.heat_capacity
        * simulator._t_u.numpy()  # pylint: disable=protected-access
        * simulator._t_v.numpy()  # pylint: disable=protected-access
        * (building.floor_height_cm / 100.0)
    ) / simulator._time_step_sec  # pylint: disable=protected-access

  def _balance(self, simulator, temperatures, previous_temperatures):
    """Returns the energy imbalance of every CV in W, and its scale in W/K.

    The balance is assembled here from the conductance tensors alone, so it
    repeats none of the arithmetic of the solvers under test.

    Args:
      simulator: the simulator holding the geometry and the conductances.
      temperatures: the temperature field to test.
      previous_temperatures: the field at the start of the time step.

    Returns:
      The net power into every CV, which is zero where the balance closes, and
      the total conductance of the CV, which turns that power into the
      temperature error it corresponds to.
    """
    conductances = (
        simulator._t_face_conductance_left,  # pylint: disable=protected-access
        simulator._t_face_conductance_right,  # pylint: disable=protected-access
        simulator._t_face_conductance_top,  # pylint: disable=protected-access
        simulator._t_face_conductance_bottom,  # pylint: disable=protected-access
    )
    # The neighbor across each face, as a roll of the temperature field.
    shifts = ((0, 1), (0, -1), (1, 0), (-1, 0))
    ambient = self._ambient_conductances(simulator)
    areas = self._face_areas(simulator)

    residual = simulator.building.input_q.astype(np.float64).copy()
    scale = self._storage(simulator).astype(np.float64)
    for conductance, area, shift, ambient_conductance in zip(
        conductances, areas, shifts, ambient
    ):
      conductance = conductance.numpy().astype(np.float64)
      neighbor = np.roll(temperatures, shift, axis=(0, 1))
      residual += area * conductance * (neighbor - temperatures)
      residual += (
          area * ambient_conductance * (self.AMBIENT_TEMPERATURE - temperatures)
      )
      scale += area * (conductance + ambient_conductance)

    residual += self._storage(simulator) * (
        previous_temperatures - temperatures
    )
    return residual, scale

  def test_cg_solution_closes_the_energy_balance(self):
    """The converged CG solve leaves negligible CV imbalance."""
    simulator = self._create_simulator('cg')
    previous_temperatures = simulator.building.temp.copy().astype(np.float64)

    temperatures, _ = simulator.update_temperature_estimates(
        simulator.building.temp.copy(),
        ambient_temperature=self.AMBIENT_TEMPERATURE,
        convection_coefficient=self.CONVECTION_COEFFICIENT,
    )

    residual, scale = self._balance(
        simulator, temperatures.astype(np.float64), previous_temperatures
    )
    solved = ~simulator._t_exerior_temps_mask.numpy()  # pylint: disable=protected-access
    # Reading the imbalance as the temperature error it corresponds to.
    self.assertLess(np.max(np.abs(residual[solved] / scale[solved])), 5e-5)

  def test_cg_solution_is_a_fixed_point_of_the_relaxation(self):
    """One relaxation sweep over the CG solve must not move it."""
    cg = self._create_simulator('cg')
    relaxation = self._create_simulator('fdm')

    temperatures, _ = cg.update_temperature_estimates(
        cg.building.temp.copy(),
        ambient_temperature=self.AMBIENT_TEMPERATURE,
        convection_coefficient=self.CONVECTION_COEFFICIENT,
    )
    _, max_delta = relaxation.update_temperature_estimates(
        temperatures,
        ambient_temperature=self.AMBIENT_TEMPERATURE,
        convection_coefficient=self.CONVECTION_COEFFICIENT,
    )

    self.assertLess(max_delta, 5e-5)

  def test_all_solvers_match_the_converged_relaxation(self):
    """FDM, TF-CG and NumPy/SciPy converge to the same field."""
    direct = self._create_simulator('numpy_scipy')
    cg = self._create_simulator('cg')
    relaxation = self._create_simulator('fdm')

    direct_converged = direct.finite_differences_timestep(
        ambient_temperature=self.AMBIENT_TEMPERATURE,
        convection_coefficient=self.CONVECTION_COEFFICIENT,
    )
    cg_converged = cg.finite_differences_timestep(
        ambient_temperature=self.AMBIENT_TEMPERATURE,
        convection_coefficient=self.CONVECTION_COEFFICIENT,
    )
    relaxation_converged = relaxation.finite_differences_timestep(
        ambient_temperature=self.AMBIENT_TEMPERATURE,
        convection_coefficient=self.CONVECTION_COEFFICIENT,
    )

    self.assertTrue(direct_converged)
    self.assertTrue(cg_converged)
    self.assertTrue(relaxation_converged)
    np.testing.assert_allclose(
        direct.building.temp,
        cg.building.temp,
        rtol=0.0,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        direct.building.temp,
        relaxation.building.temp,
        rtol=0.0,
        atol=5e-3,
    )

  @parameterized.parameters('cg', 'numpy_scipy')
  def test_complete_linear_solve_needs_one_outer_iteration(self, backend):
    """Without nonlinear features, whole-field solvers finish in one pass."""
    simulator = self._create_simulator(backend, iteration_limit=1)
    self.assertTrue(
        simulator.finite_differences_timestep(
            ambient_temperature=self.AMBIENT_TEMPERATURE,
            convection_coefficient=self.CONVECTION_COEFFICIENT,
        )
    )

  @parameterized.named_parameters(
      ('base', False, False, 5e-4),
      ('interior_mass', True, False, 1e-3),
      ('radiation', False, True, 5e-4),
      ('interior_mass_and_radiation', True, True, 1e-3),
  )
  def test_all_solvers_converge_with_optional_physics(
      self,
      include_interior_mass,
      include_radiation,
      tolerance,
  ):
    """All three backends converge in every optional-physics case."""
    direct = self._create_feature_simulator(
        'numpy_scipy', include_interior_mass, include_radiation
    )
    cg = self._create_feature_simulator(
        'cg', include_interior_mass, include_radiation
    )
    iterative = self._create_feature_simulator(
        'fdm', include_interior_mass, include_radiation
    )

    direct_converged = direct.finite_differences_timestep(
        ambient_temperature=305.0,
        convection_coefficient=self.CONVECTION_COEFFICIENT,
    )
    cg_converged = cg.finite_differences_timestep(
        ambient_temperature=305.0,
        convection_coefficient=self.CONVECTION_COEFFICIENT,
    )
    iterative_converged = iterative.finite_differences_timestep(
        ambient_temperature=305.0,
        convection_coefficient=self.CONVECTION_COEFFICIENT,
    )
    self.assertTrue(direct_converged)
    self.assertTrue(cg_converged)
    self.assertTrue(iterative_converged)
    np.testing.assert_allclose(
        cg.building.temp,
        direct.building.temp,
        rtol=0.0,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        iterative.building.temp,
        direct.building.temp,
        rtol=0.0,
        atol=tolerance,
    )

    if include_interior_mass:
      np.testing.assert_allclose(
          cg.building.interior_mass_temp,
          direct.building.interior_mass_temp,
          rtol=0.0,
          atol=5e-5,
      )
      np.testing.assert_allclose(
          iterative.building.interior_mass_temp,
          direct.building.interior_mass_temp,
          rtol=0.0,
          atol=5e-5,
      )

  def test_steady_state_relaxes_to_the_ambient_temperature(self):
    """Without a source, every CV ends up at the ambient temperature."""
    # The exterior wall of this building is insulating, k = 0.05 W/(m K), so
    # its time constant runs to days. Implicit Euler is unconditionally
    # stable, so long steps get to the steady state at no cost in accuracy.
    simulator = self._create_simulator('cg', time_step_sec=self.LONG_STEP_SEC)
    self.assertEqual(np.max(np.abs(simulator.building.input_q)), 0.0)

    for _ in range(200):
      simulator.finite_differences_timestep(
          ambient_temperature=self.AMBIENT_TEMPERATURE,
          convection_coefficient=self.CONVECTION_COEFFICIENT,
      )

    # The tolerance is loose enough to absorb the float32 rounding of the
    # assembled coefficients, which leaves the discrete steady state a small
    # fraction of a degree away from a perfectly uniform field.
    np.testing.assert_allclose(
        simulator.building.temp,
        np.full_like(simulator.building.temp, self.AMBIENT_TEMPERATURE),
        atol=5e-2,
    )

  def test_steady_state_loses_exactly_what_is_put_in(self):
    """At steady state the ambient carries away the whole source."""
    simulator = self._create_simulator('cg', time_step_sec=self.LONG_STEP_SEC)
    interior = ~simulator._t_exerior_temps_mask.numpy()  # pylint: disable=protected-access
    interior_cvs = np.argwhere(interior)
    source_cv = tuple(interior_cvs[len(interior_cvs) // 2])
    source_watts = 500.0
    simulator.building.input_q[source_cv] = source_watts

    for _ in range(200):
      simulator.finite_differences_timestep(
          ambient_temperature=self.AMBIENT_TEMPERATURE,
          convection_coefficient=self.CONVECTION_COEFFICIENT,
      )

    temperatures = simulator.building.temp.astype(np.float64)
    excess = temperatures - self.AMBIENT_TEMPERATURE
    conductances = (
        simulator._t_face_conductance_left,  # pylint: disable=protected-access
        simulator._t_face_conductance_right,  # pylint: disable=protected-access
        simulator._t_face_conductance_top,  # pylint: disable=protected-access
        simulator._t_face_conductance_bottom,  # pylint: disable=protected-access
    )
    shifts = ((0, 1), (0, -1), (1, 0), (-1, 0))
    loss = 0.0
    for conductance, area, shift, ambient_conductance in zip(
        conductances,
        self._face_areas(simulator),
        shifts,
        self._ambient_conductances(simulator),
    ):
      loss += np.sum(area * ambient_conductance * excess * interior)
      # A CV with a single neighbor is an exterior CV, held at the ambient
      # temperature, and the face into it conducts without a film.
      pinned_neighbor = np.roll(~interior, shift, axis=(0, 1))
      loss += np.sum(
          area
          * conductance.numpy().astype(np.float64)
          * excess
          * interior
          * pinned_neighbor
      )

    self.assertAlmostEqual(loss / source_watts, 1.0, places=2)


class NumpyAgreementTest(absltest.TestCase):
  """Checks the tensorized balance against the iterative one it replaces.

  TFSimulator and Simulator solve the same finite volume energy balance, one
  over the whole grid at once and one CV at a time, so the two have to agree
  to within the float32 the tensor path runs in. The conductances are compared
  face by face first, since a disagreement there is much easier to read than
  the temperature difference it eventually produces.
  """

  AMBIENT_TEMPERATURE = 285.0
  CONVECTION_COEFFICIENT = 12.0

  def _create_pair(self):
    """Returns a TFSimulator and a Simulator over the same building."""
    fixture = tf_simulator_test.TFSimulatorTest()
    arguments = (
        fixture._create_small_hvac(),  # pylint: disable=protected-access
        mock.create_autospec(weather_controller_py.WeatherController),
        300.0,
        0.1,
        20,
        15,
        pd.Timestamp('2012-12-21'),
    )
    tensor_simulator = tf_simulator_py.TFSimulator(
        fixture._create_test_building(), *arguments  # pylint: disable=protected-access
    )
    iterative_simulator = simulator_py.Simulator(
        fixture._create_test_building(), *arguments  # pylint: disable=protected-access
    )
    return tensor_simulator, iterative_simulator

  def test_face_conductances_match_the_iterative_simulator(self):
    """Every shared face carries the same conductance in both solvers.

    Only the faces that actually join two building CVs are compared. A cell
    outside the building has no neighbors at all, and the tensor path leaves
    conductances sitting on its faces rather than zeroing them, but nothing
    reads them: apply_exterior_temps overwrites those cells every step, and no
    building CV is ever their neighbor, so every face joining the two is a
    boundary face carrying a convective film instead.
    """
    tensor_simulator, iterative_simulator = self._create_pair()
    building = iterative_simulator.building
    floor_height = building.floor_height_cm / 100.0
    widths_u = tensor_simulator._t_u.numpy()  # pylint: disable=protected-access
    widths_v = tensor_simulator._t_v.numpy()  # pylint: disable=protected-access

    # The owner CV's own face area, which the tensor path multiplies the
    # conductance by. The narrowing to the shared area is already folded into
    # the conductance itself.
    faces = {
        (0, -1): ('left', tensor_simulator._t_face_conductance_left, widths_v),  # pylint: disable=protected-access
        (0, 1): ('right', tensor_simulator._t_face_conductance_right, widths_v),  # pylint: disable=protected-access
        (-1, 0): ('top', tensor_simulator._t_face_conductance_top, widths_u),  # pylint: disable=protected-access
        (1, 0): (
            'bottom',
            tensor_simulator._t_face_conductance_bottom,  # pylint: disable=protected-access
            widths_u,
        ),
    }
    conductances = {
        offset: (name, tensor.numpy().astype(np.float64), width)
        for offset, (name, tensor, width) in faces.items()
    }

    compared = 0
    rows, columns = building.temp.shape
    for i in range(rows):
      for j in range(columns):
        for neighbor in building.neighbors[i][j]:
          ni, nj = int(neighbor[0]), int(neighbor[1])
          name, conductance, width = conductances[(ni - i, nj - j)]
          expected = iterative_simulator._get_face_conductance(  # pylint: disable=protected-access
              (i, j), (ni, nj)
          )
          self.assertAlmostEqual(
              conductance[i, j] * width[i, j] * floor_height,
              expected,
              delta=1e-4 * max(expected, 1.0),
              msg=f'{name} face of CV {(i, j)}',
          )
          compared += 1

    # Guard against the loop silently finding nothing to compare.
    self.assertGreater(compared, 0)

  def test_relaxation_converges_to_the_same_temperatures(self):
    """Both solvers reach the same fixed point from the same initial field."""
    tensor_simulator, iterative_simulator = self._create_pair()

    converged = []
    for simulator in (tensor_simulator, iterative_simulator):
      temperatures = simulator.building.temp.copy()
      for _ in range(60):
        temperatures, _ = simulator.update_temperature_estimates(
            temperatures,
            ambient_temperature=self.AMBIENT_TEMPERATURE,
            convection_coefficient=self.CONVECTION_COEFFICIENT,
        )
      converged.append(np.asarray(temperatures, dtype=np.float64))

    # Loose enough to absorb the float32 the tensor path assembles its
    # coefficients in, tight enough to catch any real difference in the
    # balance: the measured gap is about 1e-3 K.
    np.testing.assert_allclose(converged[0], converged[1], atol=5e-2)


class RadiationLinearizationTest(absltest.TestCase):
  r"""Checks the exact split of the interior longwave exchange.

  Both solvers rewrite $\sum_j M_{ij} \sigma (T_j^4 - T_i^4)$ as
  $\sum_j M_{ij} h_{ij} (T_j - T_i)$ with $h_{ij}$ the exact secant of
  $\sigma T^4$, and move the part proportional to the CV's own temperature onto
  the denominator. That is an identity, so the fixed point is whatever the
  explicit form had; what changes is that the sweep can no longer run away. Left
  fully explicit the same balance reaches NaN at a day-long step, which is what
  the large time step case below guards against.
  """

  AMBIENT_TEMPERATURE = 315.0
  CONVECTION_COEFFICIENT = 12.0

  def _create_simulator(self, simulator_class, time_step_sec, iteration_limit):
    """Returns a simulator of the given class over a radiating building."""
    fixture = (
        simulator_flexible_floor_plan_test.FlexibleFloorplanSimulatorTest()
    )
    _, building = fixture._create_simulator_and_building(  # pylint: disable=protected-access
        initial_temp=292.0,
        include_interior_mass=True,
        include_radiative_heat_transfer=True,
        convergence_threshold=0.001,
        iteration_limit=100,
    )
    return simulator_class(
        building,
        fixture._create_small_hvac(),  # pylint: disable=protected-access
        mock.create_autospec(weather_controller_py.WeatherController),
        time_step_sec,
        0.0001,
        iteration_limit,
        iteration_limit,
        pd.Timestamp('2012-12-21'),
    )

  def _relax(self, simulator, sweeps):
    """Runs a fixed number of sweeps and returns the field and the movement."""
    temperatures = simulator.building.temp.copy()
    delta = np.inf
    for _ in range(sweeps):
      temperatures, delta = simulator.update_temperature_estimates(
          temperatures,
          ambient_temperature=self.AMBIENT_TEMPERATURE,
          convection_coefficient=self.CONVECTION_COEFFICIENT,
      )
    return np.asarray(temperatures, dtype=np.float64), float(delta)

  def test_a_huge_time_step_stays_finite_in_both_solvers(self):
    """A step far past the explicit stability limit does not run away.

    With the radiative self term left in the numerator this case reached NaN
    within a handful of sweeps, because a CV's update could overshoot its
    neighbors with nothing pulling it back. Moving that term onto the
    denominator makes the update a convex combination of temperatures, so the
    field can only stay inside the envelope it started in. Convergence at this
    step size is not the point and is not asserted - staying finite is.
    """
    for simulator_class in (
        flexible_py.SimulatorFlexibleGeometries,
        tf_simulator_py.TFSimulator,
    ):
      with self.subTest(simulator_class.__name__):
        simulator = self._create_simulator(simulator_class, 864000.0, 60)
        temperatures, _ = self._relax(simulator, 60)

        self.assertTrue(np.all(np.isfinite(temperatures)))
        # The zone still has a heat source in it, so the field is not bounded
        # by the ambient. What the split does guarantee is that radiation
        # cannot be what pushes it out of a physical range.
        self.assertGreaterEqual(temperatures.min(), 250.0)
        self.assertLessEqual(temperatures.max(), 400.0)

  def test_the_converged_field_is_a_fixed_point_of_the_sweep(self):
    """Sweeping the converged field again leaves it where it is.

    The split only moves terms between the two sides of the update, so the
    field it settles on has to satisfy the balance the explicit form was
    written for. If the diagonal and the source did not come from the same
    identity, this is where the mismatch would show up as a residual movement.
    """
    for simulator_class in (
        flexible_py.SimulatorFlexibleGeometries,
        tf_simulator_py.TFSimulator,
    ):
      with self.subTest(simulator_class.__name__):
        simulator = self._create_simulator(simulator_class, 300.0, 200)
        _, delta = self._relax(simulator, 60)
        self.assertLess(delta, 1e-4)

  def test_both_solvers_agree_with_radiation_switched_on(self):
    """The tensor and iterative radiation paths reach the same field.

    The two assemble the same split in different places - one CV at a time
    against the whole grid at once - so agreeing here covers the scatter back
    onto the grid, the area weighting, and the wall/mass branch all at once.
    """
    converged = []
    for simulator_class in (
        flexible_py.SimulatorFlexibleGeometries,
        tf_simulator_py.TFSimulator,
    ):
      simulator = self._create_simulator(simulator_class, 300.0, 200)
      temperatures, _ = self._relax(simulator, 40)
      converged.append(temperatures)

    # Same tolerance as the non-radiating comparison: wide enough for the
    # float32 the tensor path assembles its coefficients in.
    np.testing.assert_allclose(converged[0], converged[1], atol=5e-2)


if __name__ == '__main__':
  absltest.main()
