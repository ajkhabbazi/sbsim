"""Whole-system heat balance on a full or grouped floor plan.

Linear radiation and interior mass share one system. Sparse direct solvers
solve it once per step. JAX CG and dense response are alternative backends for
the same assembled equations.
"""

from __future__ import annotations

import gin
import jax
from jax.experimental import sparse as jax_sparse
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jax_linalg
import numpy as np
from scipy import sparse
from scipy.sparse import linalg

from smart_control.simulator import cached_lu_simulator
from smart_control.simulator import grouped_heat_balance

InteriorMass = grouped_heat_balance.InteriorMass
_REFERENCE_K = 295.65
_SLOPE = 4.0 * _REFERENCE_K**3
_METHODS = ('sparse_lu', 'cg', 'dense', 'spsolve')


def _radiation_remainder(previous, change):
  """Compute (T + dT)**4 - T**4 - slope*dT without cancellation."""
  current = previous + change
  return change * ((current + previous) * (current**2 + previous**2) - _SLOPE)


def _make_device_step(
    device,
    method,
    matrix,
    transfer,
    radiation,
    conduction,
    linear,
    tolerance,
    limit,
    cg_rtol,
    cg_maxiter,
    radiation_diagonal=None,
    room_factors=None,
    system_storage=None,
):
  """Compile a single linear step or an on-device nonlinear iteration."""

  def device_csr(value):
    arrays = jax.device_put((value.data, value.indices, value.indptr), device)
    return jax_sparse.CSR(arrays, shape=value.shape)

  def sparse_matvec(value):
    # Floor-plan rows are short and their structure never changes. Pad those
    # rows once so a CG iteration only does gathers and dot products.
    value = value.tocsr()
    counts = np.diff(value.indptr)
    width = int(counts.max(initial=0))
    if value.shape[0] * width > 4 * value.nnz:
      csr = device_csr(value)
      return lambda x: csr @ x
    rows = np.repeat(np.arange(value.shape[0]), counts)
    columns = np.arange(value.nnz) - np.repeat(value.indptr[:-1], counts)
    gather = np.zeros((value.shape[0], width), dtype=np.int32)
    weight = np.zeros_like(gather, dtype=np.float64)
    gather[rows, columns] = value.indices
    weight[rows, columns] = value.data
    gather, weight = jax.device_put((gather, weight), device)
    return lambda x: jnp.sum(weight * x[gather], axis=1)

  conductance_vector = sparse_matvec(conduction)
  if room_factors is not None and method == 'cg':
    room_vector = sparse_matvec(room_factors)
    room_transpose_vector = sparse_matvec(room_factors.T)
    radiative_diag = jax.device_put(radiation_diagonal, device)
    storage = jax.device_put(system_storage, device)

    def radiation_vector(x):
      return radiative_diag * x - room_transpose_vector(room_vector(x))

    def matrix_vector(x):
      return conductance_vector(x) + storage * x + _SLOPE * radiation_vector(x)

  else:
    system = device_csr(matrix)
    radiative = device_csr(radiation)

    def radiation_vector(x):
      return radiative @ x

    def matrix_vector(x):
      return system @ x

  diagonal_inv = jax.device_put(1 / matrix.diagonal(), device)
  if transfer is not None:
    transfer = jax.device_put(transfer, device)

  def linear_solve(rhs, initial):
    if method == 'dense':
      result = transfer @ rhs
      return result, jnp.all(jnp.isfinite(result))
    if method == 'spsolve':
      result = jax_sparse.linalg.spsolve(
          system.data, system.indices, system.indptr, rhs, tol=cg_rtol
      )
    else:
      result, _ = jax_linalg.cg(
          matrix_vector,
          rhs,
          x0=initial,
          M=lambda x: diagonal_inv * x,
          tol=cg_rtol,
          atol=cg_rtol,
          maxiter=cg_maxiter,
      )
    error = jnp.linalg.norm(matrix_vector(result) - rhs)
    scale = jnp.maximum(jnp.linalg.norm(rhs), 1)
    return result, jnp.isfinite(error) & (error <= 10 * cg_rtol * scale)

  @jax.jit
  def step(previous, forcing):
    rhs = forcing - conductance_vector(previous) - radiation_vector(previous**4)
    initial, ok = linear_solve(rhs, jnp.zeros_like(previous))
    if linear:
      return initial, jnp.int32(1), jnp.float64(0), ok

    def condition(state):
      _, count, error, valid = state
      return (count < limit) & (error > tolerance) & valid

    def update(state):
      old, count, _, valid = state
      current, solved = linear_solve(
          rhs - radiation_vector(_radiation_remainder(previous, old)), old
      )
      error = jnp.max(jnp.abs(current - old))
      return current, count + 1, error, valid & solved & jnp.isfinite(error)

    change, count, error, ok = jax.lax.while_loop(
        condition, update, (initial, jnp.int32(1), jnp.float64(jnp.inf), ok)
    )
    return change, count, error, ok & (error <= tolerance)

  example = jax.device_put(np.zeros(matrix.shape[0], dtype=np.float64), device)
  return step.lower(example, example).compile()


@gin.configurable
class GroupedJaxSimulator(cached_lu_simulator.CachedLUSimulator):
  """A full or grouped floor-plan solver with optional coupled heat storage.

  Construction takes the normal floor-plan simulator arguments. Physics flags
  default to the corresponding building settings. Apply an explicit
  `InteriorMass` to select effective properties per square metre of floor.
  With grouped=False each active CV and each mass CV has its own temperature.

  sparse_lu uses a cached CPU sparse factorization. cg and spsolve solve the
  same sparse system with JAX on the requested device. dense uses a response
  precomputed by sparse LU with a JAX matrix product per solve. Only nonlinear
  radiation uses an outer iteration. cg uses its own inner iterations.
  """

  def __init__(
      self,
      *args,
      grouped: bool = True,
      include_interior_radiative_heat_transfer: bool | None = None,
      include_exterior_radiative_heat_transfer: bool | None = None,
      include_interior_mass: bool | None = None,
      linearized_radiative_heat_transfer: bool = True,
      interior_mass: InteriorMass | None = None,
      platform: str = 'cpu',
      solver_method: str = 'sparse_lu',
      wall_depth_cells: int = 1,
      air_block_cells: int | None = None,
      cg_tolerance: float = 1e-9,
      cg_iteration_limit: int = 3000,
      **kwargs,
  ):
    if solver_method not in _METHODS:
      raise ValueError(f'Choose a solver_method from {_METHODS}')
    if platform not in ('cpu', 'gpu'):
      raise ValueError("platform must be 'cpu' or 'gpu'")
    if solver_method == 'sparse_lu' and platform != 'cpu':
      raise ValueError(
          "The sparse_lu backend uses CPU; choose 'spsolve' for GPU"
      )
    if (
        not np.isfinite(cg_tolerance)
        or cg_tolerance <= 0
        or cg_iteration_limit < 1
    ):
      raise ValueError('CG tolerance and iteration limit must be positive')
    if (
        isinstance(wall_depth_cells, bool)
        or not isinstance(wall_depth_cells, (int, np.integer))
        or wall_depth_cells < 1
    ):
      raise ValueError('wall_depth_cells must be a positive integer')
    if interior_mass is not None and not isinstance(
        interior_mass, InteriorMass
    ):
      raise TypeError('interior_mass must be an InteriorMass configuration')
    self._grouped_system = None
    self._grouped_h = None
    self._interior_override = include_interior_radiative_heat_transfer
    self._exterior_override = include_exterior_radiative_heat_transfer
    self._mass_override = include_interior_mass
    self._mass_config = interior_mass
    self.grouped = grouped
    self.wall_depth_cells = wall_depth_cells
    self.air_block_cells = air_block_cells
    self.linearized_radiative_heat_transfer = linearized_radiative_heat_transfer
    self.solver_method = solver_method
    self.platform = platform
    self._cg_tolerance = cg_tolerance
    self._cg_iteration_limit = cg_iteration_limit
    self.device = None
    if solver_method != 'sparse_lu':
      jax.config.update('jax_enable_x64', True)
      try:
        self.device = jax.devices(platform)[0]
      except (RuntimeError, IndexError) as error:
        raise ValueError(
            f'JAX device unavailable for platform {platform!r}'
        ) from error
    super().__init__(*args, **kwargs)

  @property
  def include_interior_radiative_heat_transfer(self):
    if self._interior_override is not None:
      return self._interior_override
    return self.building.include_radiative_heat_transfer

  @property
  def include_exterior_radiative_heat_transfer(self):
    if self._exterior_override is not None:
      return self._exterior_override
    return super().include_exterior_radiative_heat_transfer

  @property
  def include_interior_mass(self):
    if self._mass_override is not None:
      return self._mass_override
    return self._mass_config is not None or self.building.include_interior_mass

  def reset(self):
    super().reset()
    self._mass_temperature = None
    self._state_temperatures = None
    self.last_solve_count = 0
    self.last_outer_iteration_count = 0
    self.last_outer_max_delta = np.nan
    self.last_heat_balance_converged = False

  @property
  def thermal_node_count(self):
    return (
        0 if self._grouped_system is None else self._grouped_system.storage.size
    )

  @property
  def mass_node_count(self):
    if self._grouped_system is None or not self._grouped_system.mass_nodes.size:
      return 0
    return self.thermal_node_count - int(self._grouped_system.groups.max()) - 1

  @property
  def group_map(self):
    """Physical-cell-to-temperature index, with -1 for the ambient boundary."""
    if self._grouped_system is None:
      return None
    positions = np.full(self.building.temp.shape, -1, dtype=int)
    positions.flat[self._grouped_system.active] = self._grouped_system.groups
    return positions

  @property
  def state_temperatures(self):
    """Last solved node temperatures in kelvin, including any mass nodes."""
    return self._state_temperatures

  @property
  def mass_temperature(self):
    """Last solved mass temperature grid in kelvin, or None before solving."""
    return self._mass_temperature

  def prepare_thermal_system(self, convection_coefficient):
    """Prepare or reuse a solve for this exterior convection coefficient."""
    h = float(convection_coefficient)
    if not np.isfinite(h) or h <= 0:
      raise ValueError(
          'Exterior convection coefficient must be finite and positive'
      )
    if self._grouped_system is not None and self._grouped_h == h:
      return
    system = grouped_heat_balance.assemble(
        self,
        h,
        grouped=self.grouped,
        wall_depth_cells=self.wall_depth_cells,
        air_block_cells=self.air_block_cells,
        include_interior_radiative_heat_transfer=(
            self.include_interior_radiative_heat_transfer
        ),
        include_exterior_radiative_heat_transfer=(
            self.include_exterior_radiative_heat_transfer
        ),
        include_interior_mass=self.include_interior_mass,
        interior_mass=self._mass_config,
    )
    matrix = (system.conduction + _SLOPE * system.radiation).tocsr()
    matrix.sort_indices()
    conductance = (system.conduction - sparse.diags(system.storage)).tocsr()
    conductance.eliminate_zeros()
    linear = self.linearized_radiative_heat_transfer or not system.radiation.nnz
    if self.solver_method in ('sparse_lu', 'dense'):
      factor = linalg.splu(matrix.tocsc(), permc_spec='MMD_AT_PLUS_A')
    else:
      factor = None
    step = None
    if self.device is not None:
      transfer = (
          factor.solve(np.eye(matrix.shape[0])) if factor is not None else None
      )
      step = _make_device_step(
          self.device,
          self.solver_method,
          matrix,
          transfer,
          system.radiation,
          conductance,
          linear,
          self._convergence_threshold,
          self._iteration_limit,
          self._cg_tolerance,
          self._cg_iteration_limit,
          radiation_diagonal=getattr(system, 'radiation_diagonal', None),
          room_factors=getattr(system, 'room_factors', None),
          system_storage=system.storage,
      )
    self._factor = factor if step is None else None
    self._device_step = step
    self._conductance = conductance
    self._linear_heat_balance = linear
    self._grouped_system = system
    self._grouped_h = h
    self.factorization_count += int(factor is not None)

  def _previous_temperature(self, system):
    grid = self.building.temp.ravel()[system.active]
    summed = np.bincount(
        system.groups,
        weights=system.cv_storage * grid,
        minlength=system.storage.size,
    )
    if system.mass_nodes.size:
      native_mass = (
          self.building.interior_mass_temp
          if self.building.include_interior_mass
          else self._mass_temperature
      )
      if native_mass is None:
        native_mass = self.building.temp
      summed += np.bincount(
          system.mass_nodes,
          weights=system.mass_cell_storage
          * native_mass.ravel()[system.mass_active],
          minlength=system.storage.size,
      )
    return summed / system.storage

  def _forcing(self, system, outdoor):
    power = system.ambient * outdoor + np.bincount(
        system.groups,
        weights=self.building.input_q.ravel()[system.active],
        minlength=system.storage.size,
    )
    if system.geometry is None:
      return power
    weather = self._weather_controller.get_exterior_radiation(
        self.current_timestamp, outdoor
    )
    facade = getattr(self._weather_controller, 'facade_flux', None)
    poa = (
        facade(self.current_timestamp, system.geometry.azimuths)
        if callable(facade)
        else None
    )
    forcing = system.geometry.forcing(weather, poa_irradiance=poa)
    return (
        power
        + forcing.shortwave_air_gain
        + forcing.mass_gain
        + (forcing.emission * forcing.radiant_temperature**4)
    )

  def _solve_on_host(self, system, previous, forcing):
    radiation = system.radiation
    rhs = forcing - self._conductance @ previous - radiation @ previous**4
    change = self._factor.solve(rhs)
    if self._linear_heat_balance:
      return change, 1, 0.0, np.isfinite(change).all()
    error = np.inf
    for count in range(2, self._iteration_limit + 1):
      candidate = self._factor.solve(
          rhs - radiation @ _radiation_remainder(previous, change)
      )
      error = np.max(np.abs(candidate - change))
      change = candidate
      if not np.isfinite(error) or error <= self._convergence_threshold:
        return change, count, error, error <= self._convergence_threshold
    return change, self._iteration_limit, error, False

  def step_sim(self, video_filename=None):
    """Advance the native HVAC and heat-balance API, with optional rendering."""
    self.setup_step_sim()
    self.execute_step_sim(video_filename=video_filename)

  def finite_differences_timestep(
      self, *, ambient_temperature, convection_coefficient
  ):
    """Advance air, envelope, and optional thermal mass in the same solve."""
    self.prepare_thermal_system(convection_coefficient)
    system = self._grouped_system
    previous = self._previous_temperature(system)
    forcing = self._forcing(system, ambient_temperature)
    if self._device_step is None:
      change, count, error, ok = self._solve_on_host(system, previous, forcing)
      self.triangular_solve_count += count
    else:
      change, count, error, ok = jax.device_get(
          self._device_step(
              jax.device_put(previous, self.device),
              jax.device_put(forcing, self.device),
          )
      )
    self.last_solve_count = int(count)
    self.last_outer_iteration_count = int(count)
    self.last_outer_max_delta = float(error)
    self.last_heat_balance_converged = bool(ok) and bool(
        np.isfinite(change).all()
    )
    if not self.last_heat_balance_converged:
      raise RuntimeError(
          f'Heat balance did not converge with {self.solver_method} at'
          f' {self.current_timestamp} after {count} linear solves'
          f' (update={error:g})'
      )
    temperature = previous + change
    grid = np.full(
        self.building.temp.shape, ambient_temperature, dtype=np.float64
    )
    grid.ravel()[system.active] = temperature[system.groups]
    self.building.temp = grid
    if system.mass_nodes.size:
      self._mass_temperature = grid.copy()
      self._mass_temperature.ravel()[system.mass_active] = temperature[
          system.mass_nodes
      ]
      if self.building.include_interior_mass:
        self.building.interior_mass_temp = self._mass_temperature
    self._state_temperatures = temperature
    return True
