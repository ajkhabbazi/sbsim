"""NumPy geometry with a JAX coefficient assembly and sparse solve."""

from __future__ import annotations

import jax
from jax import lax
from jax.experimental.sparse import linalg as jax_sparse_linalg
import jax.numpy as jnp
import numpy as np

from smart_control.simulator import building_radiation_utils
from smart_control.simulator import simulator_flexible_floor_plan

_SOLVER_METHODS = ('spsolve', 'cg')
# A CV exchanges heat with at most its four face neighbors.
_MAX_NEIGHBORS = 4


class JaxSimulator(simulator_flexible_floor_plan.SimulatorFlexibleGeometries):
  """Solves each linearized heat balance with JAX `spsolve` or PCG.

  The system is the same one `NumpyScipySimulator` assembles, but its
  structure is resolved only once. The sparsity pattern, the face
  conductances, the CV volumes and the boundary face geometry depend only on
  the building, so they are computed in NumPy on the first solve and kept as
  JAX arrays. Each call then only moves the state that changes between
  iterations - the previous step temperature, the heat input, the interior
  mass temperature and the radiative exchange terms - into JAX, assembles the
  diagonal and right-hand side in one jitted function, and solves with
  `jax.experimental.sparse.linalg.spsolve` or with Jacobi-preconditioned
  conjugate gradients. The result is copied back to NumPy, so the rest of the
  simulator never sees a JAX array.

  `spsolve` has a native implementation only on CUDA GPUs, where it is a
  sparse QR factorization that parallelizes poorly; on CPU it calls back into
  `scipy.sparse.linalg.spsolve`. The face conductances are symmetric and every
  active row is strictly diagonally dominant through its storage term, so the
  system is symmetric positive definite and CG applies. CG needs only
  neighbor gathers and reductions, which is what a GPU does well. Its
  matrix-vector product uses a padded neighbor table rather than CSR, so it
  gathers without scattering.

  Radiation and interior mass still require the inherited outer fixed-point
  iteration.
  """

  def __init__(
      self,
      *args,
      solver_method: str = 'spsolve',
      use_float64: bool = True,
      spsolve_tolerance: float = 1e-12,
      spsolve_reorder: int = 1,
      cg_tolerance: float = 1e-10,
      cg_iteration_limit: int = 5000,
      **kwargs,
  ):
    """Simulator init.

    Args:
      *args: Positional arguments of `SimulatorFlexibleGeometries`.
      solver_method: `spsolve` for the sparse direct solve or `cg` for
        Jacobi-preconditioned conjugate gradients.
      use_float64: Whether to assemble and solve in float64. FP32 is not
        accurate enough to resolve the default convergence thresholds. This
        sets the process-wide `jax_enable_x64` flag: the thread-local
        `enable_x64` context does not reach the CPU `spsolve`, which runs as a
        SciPy callback on an XLA runtime thread and would then return FP32.
      spsolve_tolerance: Singularity tolerance passed to `spsolve`.
      spsolve_reorder: Fill-reducing reordering passed to `spsolve`; 0 for
        none, 1 symrcm, 2 symamd, 3 csrmetisnd.
      cg_tolerance: CG stops once the residual norm is below this times
        max(norm(rhs), 1). The right-hand side is that of the change over the
        step, so the tolerance is relative to the step's net heat flows rather
        than to the stored energy.
      cg_iteration_limit: Maximum CG iterations per solve.
      **kwargs: Keyword arguments of `SimulatorFlexibleGeometries`.
    """
    if solver_method not in _SOLVER_METHODS:
      raise ValueError(
          f'solver_method must be one of {_SOLVER_METHODS}, got'
          f' {solver_method!r}'
      )
    self._solver_method = solver_method
    self._use_float64 = use_float64
    if use_float64:
      jax.config.update('jax_enable_x64', True)
    self._spsolve_tolerance = spsolve_tolerance
    self._spsolve_reorder = spsolve_reorder
    self.cg_tolerance = cg_tolerance
    self.cg_iteration_limit = cg_iteration_limit
    if solver_method == 'cg':
      self.last_cg_iteration_count = 0
      self.last_cg_residual_norm = np.nan
    self._static_system = None
    super().__init__(*args, **kwargs)
    self._solves_entire_linear_system = True
    self._jit_solve = jax.jit(self._solve)

  @property
  def _dtype(self):
    return jnp.float64 if self._use_float64 else jnp.float32

  def _build_static_system(self) -> dict[str, jax.Array]:
    """Resolves every coefficient that does not change between solves."""
    building = self.building
    rows_count, cols_count = building.temp.shape
    cv_count = rows_count * cols_count
    flat_index = np.arange(cv_count).reshape(rows_count, cols_count)
    exterior = np.fromiter(
        (
            len(building.neighbors[row][column]) <= 1
            for row in range(rows_count)
            for column in range(cols_count)
        ),
        dtype=bool,
        count=cv_count,
    ).reshape(rows_count, cols_count)
    floor_height = building.floor_height_cm / 100.0

    volume = np.zeros((rows_count, cols_count))
    face_conductance_sum = np.zeros((rows_count, cols_count))
    exterior_neighbor_conductance = np.zeros((rows_count, cols_count))
    # Up to four ambient faces per CV; absent faces keep zero area.
    ambient_area = np.zeros((4, rows_count, cols_count))
    ambient_half_resistance = np.zeros((4, rows_count, cols_count))
    mass_conductance = np.zeros((rows_count, cols_count))
    include_interior_mass = getattr(building, 'include_interior_mass', False)

    matrix_rows = []
    matrix_columns = []
    matrix_values = []
    for row in range(rows_count):
      for column in range(cols_count):
        index = flat_index[row, column]
        # Every row carries its diagonal; its value is filled in per solve.
        matrix_rows.append(index)
        matrix_columns.append(index)
        matrix_values.append(0.0)
        if exterior[row, column]:
          continue

        u, v = self._get_cv_dimensions((row, column))
        volume[row, column] = u * v * floor_height
        for neighbor_row, neighbor_column in building.neighbors[row][column]:
          conductance = self._get_face_conductance(
              (row, column), (neighbor_row, neighbor_column)
          )
          face_conductance_sum[row, column] += conductance
          if exterior[neighbor_row, neighbor_column]:
            exterior_neighbor_conductance[row, column] += conductance
          else:
            matrix_rows.append(index)
            matrix_columns.append(flat_index[neighbor_row, neighbor_column])
            matrix_values.append(-conductance)

        # Populates the per-CV ambient face cache shared with the base class.
        self._get_ambient_conductance((row, column), 1.0)
        for face, (area, half_resistance) in enumerate(
            self._ambient_face_cache[(row, column)]
        ):
          ambient_area[face, row, column] = area
          ambient_half_resistance[face, row, column] = half_resistance

        if include_interior_mass and building.interior_mass_mask[row, column]:
          mass_conductance[row, column] = (
              building.interior_mass_conductivity[row, column]
              * u
              * v
              / floor_height
          )

    matrix_rows = np.asarray(matrix_rows)
    matrix_columns = np.asarray(matrix_columns)
    order = np.lexsort((matrix_columns, matrix_rows))
    matrix_rows = matrix_rows[order]
    matrix_columns = matrix_columns[order]
    indptr = np.concatenate(
        ([0], np.cumsum(np.bincount(matrix_rows, minlength=cv_count)))
    )

    # Padded neighbor table for the CG matrix-vector product. Unused slots
    # point at the row itself with a zero coefficient.
    off_diagonal = matrix_rows != matrix_columns
    off_rows = matrix_rows[off_diagonal]
    off_row_start = np.concatenate(
        ([0], np.cumsum(np.bincount(off_rows, minlength=cv_count)))
    )
    slot = np.arange(len(off_rows)) - off_row_start[off_rows]
    neighbor_index = np.tile(np.arange(cv_count)[:, None], (1, _MAX_NEIGHBORS))
    neighbor_coupling = np.zeros((cv_count, _MAX_NEIGHBORS))
    neighbor_index[off_rows, slot] = matrix_columns[off_diagonal]
    # The stored values are -conductance; CG uses A x = D x - sum(G x_n).
    neighbor_coupling[off_rows, slot] = -np.asarray(matrix_values)[order][
        off_diagonal
    ]

    radiation_index = np.full((rows_count, cols_count), -1)
    if getattr(building, 'include_radiative_heat_transfer', False):
      radiation_index = np.where(exterior, -1, building.lwx_index)
    radiation_area = building.cv_size_cm / 100.0 * floor_height

    dtype = self._dtype
    return {
        'exterior': jnp.asarray(exterior.ravel()),
        'volume': jnp.asarray(volume.ravel(), dtype=dtype),
        'face_conductance_sum': jnp.asarray(
            face_conductance_sum.ravel(), dtype=dtype
        ),
        'exterior_neighbor_conductance': jnp.asarray(
            exterior_neighbor_conductance.ravel(), dtype=dtype
        ),
        'ambient_area': jnp.asarray(
            ambient_area.reshape(4, cv_count), dtype=dtype
        ),
        'ambient_half_resistance': jnp.asarray(
            ambient_half_resistance.reshape(4, cv_count), dtype=dtype
        ),
        'mass_conductance': jnp.asarray(mass_conductance.ravel(), dtype=dtype),
        'radiation_index': jnp.asarray(
            radiation_index.ravel(), dtype=jnp.int32
        ),
        'radiation_area': jnp.asarray(radiation_area, dtype=dtype),
        'off_diagonal_data': jnp.asarray(
            np.asarray(matrix_values)[order], dtype=dtype
        ),
        'diagonal_position': jnp.asarray(
            np.flatnonzero(matrix_rows == matrix_columns), dtype=jnp.int32
        ),
        'indices': jnp.asarray(matrix_columns, dtype=jnp.int32),
        'indptr': jnp.asarray(indptr, dtype=jnp.int32),
        'neighbor_index': jnp.asarray(neighbor_index, dtype=jnp.int32),
        'neighbor_coupling': jnp.asarray(neighbor_coupling, dtype=dtype),
    }

  def _solve(
      self,
      static: dict[str, jax.Array],
      temperature_estimates: jax.Array,
      previous_temperature: jax.Array,
      input_q: jax.Array,
      storage_per_volume: jax.Array,
      mass_temperature: jax.Array,
      radiation_coefficient: jax.Array,
      radiation_driving: jax.Array,
      radiation_is_wall: jax.Array,
      ambient_temperature: jax.Array,
      convection_coefficient: jax.Array,
      exterior_forcing: (
          building_radiation_utils.ExteriorRadiationForcing | None
      ),
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Assembles the diagonal and right-hand side, then solves the system."""
    # The system is solved for the change over the step, y = T - T^(-), not
    # for T itself. Storage enters A T = b as rho c V / dt T on the diagonal
    # and the same times T^(-) in b, which for a wall CV is five orders of
    # magnitude above the conduction terms. Forming b and A T^(-) separately
    # and subtracting them loses those digits, which in float32 leaves errors
    # of 0.1 K in air CVs. Writing b - A T^(-) directly, with every term a
    # conductance times a temperature difference, cancels storage exactly.
    storage = storage_per_volume * static['volume']
    ambient_conductance = jnp.sum(
        static['ambient_area']
        / (1.0 / convection_coefficient + static['ambient_half_resistance']),
        axis=0,
    )
    mass_conductance = static['mass_conductance']
    diagonal = (
        storage
        + static['face_conductance_sum']
        + ambient_conductance
        + mass_conductance
    )
    neighbor_flow = jnp.sum(
        static['neighbor_coupling']
        * (
            previous_temperature[static['neighbor_index']]
            - previous_temperature[:, None]
        ),
        axis=1,
    )
    rhs = (
        input_q
        + (static['exterior_neighbor_conductance'] + ambient_conductance)
        * (ambient_temperature - previous_temperature)
        + mass_conductance * (mass_temperature - previous_temperature)
        + neighbor_flow
    )
    if exterior_forcing is not None:
      coefficient, driving = exterior_forcing.linearized_air_terms(
          temperature_estimates, previous_temperature
      )
      diagonal += coefficient
      rhs += driving

    # Interior walls take the exchange coefficient on their own diagonal; an
    # interior mass node radiates at its mass temperature, so it stays
    # explicit in the air CV sharing its coordinates.
    radiation_index = static['radiation_index']
    has_radiation = radiation_index >= 0
    gather_index = jnp.maximum(radiation_index, 0)
    coefficient = radiation_coefficient[gather_index]
    driving = radiation_driving[gather_index]
    is_wall = radiation_is_wall[gather_index]
    radiation_area = static['radiation_area']
    implicit = has_radiation & is_wall
    explicit = has_radiation & ~is_wall
    diagonal += jnp.where(implicit, coefficient * radiation_area, 0.0)
    rhs += jnp.where(
        implicit,
        (driving - coefficient * previous_temperature) * radiation_area,
        0.0,
    )
    rhs += jnp.where(
        explicit,
        (driving - coefficient * mass_temperature) * radiation_area,
        0.0,
    )

    exterior = static['exterior']
    diagonal = jnp.where(exterior, 1.0, diagonal)
    rhs = jnp.where(exterior, ambient_temperature - previous_temperature, rhs)

    if self._solver_method == 'cg':
      initial = jnp.where(
          exterior,
          ambient_temperature - previous_temperature,
          temperature_estimates - previous_temperature,
      )
      change, iterations, residual_norm, target_norm = self._solve_cg(
          static, diagonal, rhs, initial
      )
    else:
      data = (
          static['off_diagonal_data']
          .at[static['diagonal_position']]
          .set(diagonal)
      )
      change = jax_sparse_linalg.spsolve(
          data,
          static['indices'],
          static['indptr'],
          rhs,
          tol=self._spsolve_tolerance,
          reorder=self._spsolve_reorder,
      )
      iterations = jnp.array(0, dtype=jnp.int32)
      residual_norm = jnp.array(0.0, dtype=rhs.dtype)
      target_norm = jnp.array(0.0, dtype=rhs.dtype)
    solution = previous_temperature + change
    max_delta = jnp.max(jnp.abs(solution - temperature_estimates))
    return solution, max_delta, iterations, residual_norm, target_norm

  def _solve_cg(
      self,
      static: dict[str, jax.Array],
      diagonal: jax.Array,
      rhs: jax.Array,
      initial: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Solves the SPD system with Jacobi-preconditioned CG."""
    neighbor_index = static['neighbor_index']
    neighbor_coupling = static['neighbor_coupling']

    def _matvec(vector: jax.Array) -> jax.Array:
      return diagonal * vector - jnp.sum(
          neighbor_coupling * vector[neighbor_index], axis=1
      )

    inverse_diagonal = 1.0 / diagonal
    residual = rhs - _matvec(initial)
    preconditioned = inverse_diagonal * residual
    target_norm = self.cg_tolerance * jnp.maximum(jnp.linalg.norm(rhs), 1.0)

    def _condition(state):
      iteration, _, residual, _, _ = state
      return (iteration < self.cg_iteration_limit) & (
          jnp.linalg.norm(residual) > target_norm
      )

    def _body(state):
      iteration, solution, residual, direction, residual_dot = state
      matrix_direction = _matvec(direction)
      step = residual_dot / jnp.dot(direction, matrix_direction)
      solution = solution + step * direction
      residual = residual - step * matrix_direction
      preconditioned = inverse_diagonal * residual
      next_residual_dot = jnp.dot(residual, preconditioned)
      direction = (
          preconditioned + (next_residual_dot / residual_dot) * direction
      )
      return iteration + 1, solution, residual, direction, next_residual_dot

    iteration, solution, residual, _, _ = lax.while_loop(
        _condition,
        _body,
        (
            jnp.array(0, dtype=jnp.int32),
            initial,
            residual,
            preconditioned,
            jnp.dot(residual, preconditioned),
        ),
    )
    return solution, iteration, jnp.linalg.norm(residual), target_norm

  def _jax_exterior_forcing(
      self, forcing: building_radiation_utils.ExteriorRadiationForcing | None
  ) -> building_radiation_utils.ExteriorRadiationForcing | None:
    if forcing is None:
      return None
    air_gain, emission, mass_gain, radiant_temperature = forcing
    return building_radiation_utils.ExteriorRadiationForcing(
        shortwave_air_gain=jnp.asarray(air_gain.ravel(), dtype=self._dtype),
        emission=jnp.asarray(emission.ravel(), dtype=self._dtype),
        mass_gain=jnp.asarray(mass_gain.ravel(), dtype=self._dtype),
        radiant_temperature=jnp.asarray(radiant_temperature, dtype=self._dtype),
    )

  def update_temperature_estimates(
      self,
      temperature_estimates: np.ndarray,
      ambient_temperature: float,
      convection_coefficient: float,
  ) -> tuple[np.ndarray, float]:
    """Assembles and directly solves the current linearized heat balance."""
    building = self.building
    shape = building.temp.shape
    if self._static_system is None:
      self._static_system = self._build_static_system()
    dtype = self._dtype

    if getattr(building, 'include_radiative_heat_transfer', False):
      (
          radiation_coefficient,
          radiation_driving,
          radiation_is_wall,
      ) = building.get_longwave_interior_radiative_exchange_terms(
          temperature_estimates
      )
    else:
      # A single unused entry keeps the gather in `_solve` well defined.
      radiation_coefficient = np.zeros(1)
      radiation_driving = np.zeros(1)
      radiation_is_wall = np.zeros(1, dtype=bool)

    if getattr(building, 'include_interior_mass', False):
      mass_temperature = building.interior_mass_temp
    else:
      mass_temperature = np.zeros(shape)

    def _flat(array: np.ndarray) -> jax.Array:
      return jnp.asarray(np.ravel(array), dtype=dtype)

    solution, max_delta, iterations, residual_norm, target_norm = (
        self._jit_solve(  # pylint: disable=not-callable
            self._static_system,
            _flat(temperature_estimates),
            _flat(building.temp),
            _flat(building.input_q),
            _flat(
                building.density * building.heat_capacity / self._time_step_sec
            ),
            _flat(mass_temperature),
            jnp.asarray(radiation_coefficient, dtype=dtype),
            jnp.asarray(radiation_driving, dtype=dtype),
            jnp.asarray(radiation_is_wall, dtype=bool),
            jnp.asarray(ambient_temperature, dtype=dtype),
            jnp.asarray(convection_coefficient, dtype=dtype),
            self._jax_exterior_forcing(self._exterior_forcing),
        )
    )
    if self._solver_method == 'cg':
      self.last_cg_iteration_count = int(iterations)
      self.last_cg_residual_norm = float(residual_norm)
      if self.last_cg_residual_norm > float(target_norm):
        raise RuntimeError(
            'JAX CG did not converge: '
            f'iterations={self.last_cg_iteration_count}, '
            f'residual_norm={self.last_cg_residual_norm}, '
            f'target_norm={float(target_norm)}'
        )
    return (
        np.asarray(solution, dtype=np.float64).reshape(shape),
        float(max_delta),
    )
