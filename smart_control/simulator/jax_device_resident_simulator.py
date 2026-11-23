"""Device-resident JAX heat-balance simulator."""

from __future__ import annotations

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from smart_control.simulator import constants
from smart_control.simulator import jax_simulator


class JaxDeviceResidentSimulator(jax_simulator.JaxSimulator):
  """Runs the complete Picard timestep and PCG solve in one jitted function.

  `JaxSimulator` solves on the device but leaves the outer iteration on the
  host: every Picard iteration computes the longwave exchange terms in NumPy,
  updates the interior mass nodes in a Python loop, and copies the state to
  and from the device. This class moves all of it into a single
  `jax.lax.while_loop`:

  * the longwave exchange terms are evaluated on the device. Enclosures of
    equal size are stacked and handled with one batched matrix product, so a
    grid of identical rooms costs one `einsum` per iteration rather than one
    per room;
  * the interior mass update is a masked vector expression;
  * air and mass temperatures stay on the device between timesteps. Only the
    heat input goes in and a few scalars come out per step.

  Like `TFDeviceResidentSimulator`, the building arrays are only refreshed
  when `sync_state_to_building` is called, unless `sync_to_host_each_step` is
  set. Changes made directly to `building.temp` or
  `building.interior_mass_temp` must be pushed with `sync_state_from_building`.
  """

  def __init__(
      self,
      *args,
      use_float64: bool = True,
      cg_tolerance: float = 1e-10,
      cg_iteration_limit: int = 5000,
      sync_to_host_each_step: bool = True,
      **kwargs,
  ):
    """Simulator init.

    Args:
      *args: Positional arguments of `SimulatorFlexibleGeometries`.
      use_float64: Whether to run in float64. Consumer GPUs run float64 at a
        small fraction of their float32 rate, so float32 can be much faster
        there at the cost of a looser attainable convergence threshold.
      cg_tolerance: CG stops once the residual norm is below this times
        max(norm(rhs), 1).
      cg_iteration_limit: Maximum CG iterations per solve.
      sync_to_host_each_step: Whether to copy the device state back into the
        building arrays after every timestep.
      **kwargs: Keyword arguments of `SimulatorFlexibleGeometries`.
    """
    kwargs.pop('solver_method', None)
    super().__init__(
        *args,
        solver_method='cg',
        use_float64=use_float64,
        cg_tolerance=cg_tolerance,
        cg_iteration_limit=cg_iteration_limit,
        **kwargs,
    )
    self._sync_to_host_each_step = sync_to_host_each_step
    self.last_outer_iteration_count = 0
    self.last_outer_max_delta = np.nan
    self._static_system = self._build_static_system()
    self._resident = self._build_resident_constants()
    self._jit_timestep = jax.jit(self._device_timestep)
    self.sync_state_from_building()

  @property
  def include_radiative_heat_transfer(self) -> bool:
    return bool(
        getattr(self.building, 'include_radiative_heat_transfer', False)
    )

  @property
  def include_interior_mass(self) -> bool:
    return bool(getattr(self.building, 'include_interior_mass', False))

  def reset(self):
    super().reset()
    if hasattr(self, '_resident'):
      self.sync_state_from_building()

  def sync_state_from_building(self) -> None:
    """Copies externally modified building state to the device."""
    dtype = self._dtype
    self._device_temperature = jnp.asarray(
        np.ravel(self.building.temp), dtype=dtype
    )
    if self.include_interior_mass:
      mass_temperature = self.building.interior_mass_temp
    else:
      mass_temperature = np.zeros_like(self.building.temp)
    self._device_mass_temperature = jnp.asarray(
        np.ravel(mass_temperature), dtype=dtype
    )

  def sync_state_to_building(self) -> None:
    """Copies the device state to the NumPy-backed building arrays."""
    shape = self.building.temp.shape
    self.building.temp = np.asarray(
        self._device_temperature, dtype=np.float64
    ).reshape(shape)
    if self.include_interior_mass:
      self.building.interior_mass_temp = np.asarray(
          self._device_mass_temperature, dtype=np.float64
      ).reshape(shape)

  def _build_resident_constants(self) -> dict:
    """Resolves the radiation and interior mass data used on the device."""
    building = self.building
    dtype = self._dtype
    resident = {
        'storage_per_volume': jnp.asarray(
            np.ravel(
                building.density * building.heat_capacity / self._time_step_sec
            ),
            dtype=dtype,
        ),
    }

    if self.include_interior_mass:
      mask = np.asarray(building.interior_mass_mask, dtype=bool)
      floor_height = building.floor_height_cm / 100.0
      conductivity = np.where(mask, building.interior_mass_conductivity, 1.0)
      # Same update as Simulator.update_interior_mass_temperatures:
      # T_m = (T_air + t0 T_m^(-)) / (1 + t0), t0 = rho c z^2 / (k dt).
      t0_mass = np.where(
          mask,
          building.interior_mass_density
          * building.interior_mass_heat_capacity
          * floor_height**2
          / (conductivity * self._time_step_sec),
          0.0,
      )
      resident['mass_mask'] = jnp.asarray(mask.ravel())
      resident['t0_mass'] = jnp.asarray(t0_mass.ravel(), dtype=dtype)

    if self.include_radiative_heat_transfer:
      lwx_index = np.ravel(building.lwx_index)
      node_count = int(np.max(lwx_index)) + 1
      node_cell = np.empty(node_count, dtype=np.int64)
      node_cell[lwx_index[lwx_index >= 0]] = np.flatnonzero(lwx_index >= 0)
      if self.include_interior_mass:
        node_is_wall = ~np.ravel(building.interior_mass_mask)[node_cell]
      else:
        node_is_wall = np.ones(node_count, dtype=bool)

      # Enclosures of equal size share one batched matrix product.
      groups = {}
      for indices, ifa_inv in zip(
          building.radiative_enclosure_indices, building.ifa_inv_by_enclosure
      ):
        exchange = -np.array(ifa_inv, dtype=float)
        np.fill_diagonal(exchange, 0.0)
        groups.setdefault(len(indices), []).append(
            (np.asarray(indices), exchange)
        )
      resident['node_cell'] = jnp.asarray(node_cell, dtype=jnp.int32)
      resident['node_is_wall'] = jnp.asarray(node_is_wall)
      resident['enclosure_groups'] = tuple(
          (
              jnp.asarray(
                  np.stack([indices for indices, _ in members]),
                  dtype=jnp.int32,
              ),
              jnp.asarray(
                  np.stack([exchange for _, exchange in members]), dtype=dtype
              ),
          )
          for members in groups.values()
      )
    return resident

  def _radiation_terms(
      self,
      resident: dict,
      temperature: jax.Array,
      mass_temperature: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluates calculate_radiative_exchange_terms for every enclosure."""
    if not self.include_radiative_heat_transfer:
      zeros = jnp.zeros(1, dtype=temperature.dtype)
      return zeros, zeros, jnp.zeros(1, dtype=bool)

    node_cell = resident['node_cell']
    node_is_wall = resident['node_is_wall']
    # A wall node radiates at the air temperature on it, a mass node at its
    # own mass temperature.
    node_temperature = jnp.where(
        node_is_wall, temperature[node_cell], mass_temperature[node_cell]
    )
    coefficient = jnp.zeros_like(node_temperature)
    driving = jnp.zeros_like(node_temperature)
    sigma = constants.STEFAN_BOLTZMANN_CONSTANT
    for indices, exchange in resident['enclosure_groups']:
      t1 = node_temperature[indices]
      t2 = t1 * t1
      t3 = t2 * t1
      powers = jnp.stack((jnp.ones_like(t1), t1, t2, t3, t2 * t2), axis=-1)
      # Ampere and newer GPUs default float32 matmuls to TF32, whose 10-bit
      # mantissa cannot carry T^4 ~ 1e10; it left 0.2 K errors in mass nodes.
      moments = jnp.einsum(
          'gij,gjk->gik',
          exchange,
          powers,
          precision=lax.Precision.HIGHEST,
      )
      group_coefficient = sigma * (
          t3 * moments[..., 0]
          + t2 * moments[..., 1]
          + t1 * moments[..., 2]
          + moments[..., 3]
      )
      group_driving = sigma * (
          t3 * moments[..., 1]
          + t2 * moments[..., 2]
          + t1 * moments[..., 3]
          + moments[..., 4]
      )
      flat_indices = indices.ravel()
      coefficient = coefficient.at[flat_indices].set(group_coefficient.ravel())
      driving = driving.at[flat_indices].set(group_driving.ravel())
    return coefficient, driving, node_is_wall

  def _device_timestep(
      self,
      static: dict,
      resident: dict,
      previous_temperature: jax.Array,
      previous_mass_temperature: jax.Array,
      input_q: jax.Array,
      ambient_temperature: jax.Array,
      convection_coefficient: jax.Array,
      exterior_forcing,
  ):
    """Runs the Picard iteration of one timestep on the device."""
    include_mass = self.include_interior_mass
    complete_linear_solve = not (
        self.include_radiative_heat_transfer
        or include_mass
        or exterior_forcing is not None
    )
    outer_limit = 1 if complete_linear_solve else self._iteration_limit
    dtype = previous_temperature.dtype

    def _condition(state):
      iteration, _, _, max_delta, *_ = state
      return (iteration < outer_limit) & (
          max_delta > self._convergence_threshold
      )

    def _body(state):
      iteration, temperature, mass_temperature, *_ = state
      coefficient, driving, is_wall = self._radiation_terms(
          resident, temperature, mass_temperature
      )
      (
          next_temperature,
          air_delta,
          cg_iterations,
          residual_norm,
          target_norm,
      ) = self._solve(
          static,
          temperature,
          previous_temperature,
          input_q,
          resident['storage_per_volume'],
          mass_temperature,
          coefficient,
          driving,
          is_wall,
          ambient_temperature,
          convection_coefficient,
          exterior_forcing,
      )
      if include_mass:
        # Every iteration re-solves the mass from the previous step's value.
        t0_mass = resident['t0_mass']
        numerator = next_temperature + t0_mass * previous_mass_temperature
        if exterior_forcing is not None:
          mass_conductance = jnp.where(
              resident['mass_mask'], static['mass_conductance'], 1.0
          )
          numerator += exterior_forcing.mass_gain / mass_conductance
        next_mass_temperature = jnp.where(
            resident['mass_mask'],
            numerator / (1.0 + t0_mass),
            previous_mass_temperature,
        )
        max_delta = jnp.maximum(
            air_delta,
            jnp.max(jnp.abs(next_mass_temperature - mass_temperature)),
        )
      else:
        next_mass_temperature = mass_temperature
        max_delta = air_delta
      return (
          iteration + 1,
          next_temperature,
          next_mass_temperature,
          max_delta,
          cg_iterations,
          residual_norm,
          target_norm,
      )

    initial = (
        jnp.array(0, dtype=jnp.int32),
        previous_temperature,
        previous_mass_temperature,
        jnp.array(jnp.inf, dtype=dtype),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(jnp.inf, dtype=dtype),
        jnp.array(0.0, dtype=dtype),
    )
    return lax.while_loop(_condition, _body, initial)

  def _run_device_timestep(
      self, ambient_temperature: float, convection_coefficient: float
  ):
    dtype = self._dtype
    return self._jit_timestep(  # pylint: disable=not-callable
        self._static_system,
        self._resident,
        self._device_temperature,
        self._device_mass_temperature,
        jnp.asarray(np.ravel(self.building.input_q), dtype=dtype),
        jnp.asarray(ambient_temperature, dtype=dtype),
        jnp.asarray(convection_coefficient, dtype=dtype),
        self._jax_exterior_forcing(
            self._prepare_exterior_radiation(ambient_temperature)
        ),
    )

  def warmup(
      self, *, ambient_temperature: float, convection_coefficient: float
  ) -> None:
    """Compiles the complete timestep without advancing the device state."""
    result = self._run_device_timestep(
        ambient_temperature, convection_coefficient
    )
    jax.block_until_ready(result)

  def finite_differences_timestep(
      self, *, ambient_temperature: float, convection_coefficient: float
  ) -> bool:
    """Runs one complete device-resident timestep."""
    (
        outer_iterations,
        temperature,
        mass_temperature,
        max_delta,
        cg_iterations,
        residual_norm,
        target_norm,
    ) = self._run_device_timestep(ambient_temperature, convection_coefficient)
    self._device_temperature = temperature
    self._device_mass_temperature = mass_temperature
    self.last_outer_iteration_count = int(outer_iterations)
    self.last_outer_max_delta = float(max_delta)
    self.last_cg_iteration_count = int(cg_iterations)
    self.last_cg_residual_norm = float(residual_norm)
    if self.last_cg_residual_norm > float(target_norm):
      raise RuntimeError(
          'Device-resident JAX CG did not converge: '
          f'iterations={self.last_cg_iteration_count}, '
          f'residual_norm={self.last_cg_residual_norm}, '
          f'target_norm={float(target_norm)}'
      )
    if self._sync_to_host_each_step:
      self.sync_state_to_building()
    complete_linear_solve = not (
        self.include_radiative_heat_transfer
        or self.include_interior_mass
        or self.include_exterior_radiative_heat_transfer
    )
    return (
        complete_linear_solve
        or self.last_outer_max_delta <= self._convergence_threshold
    )
