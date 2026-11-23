"""Coupled air/interior-mass heat-balance simulators.

The existing simulators lag interior-mass temperature in the air solve and
update it after each solve.  The classes in this module instead solve the air
CVs and all active mass nodes in one symmetric block system.  Radiation stays
in the existing nonlinear outer (Picard) iteration.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
import tensorflow as tf

from smart_control.simulator import building_radiation_utils
from smart_control.simulator import numpy_scipy_simulator
from smart_control.simulator import tf_device_resident_simulator
from smart_control.simulator import tf_simulator


@tf.function(reduce_retracing=True)
def _solve_coupled_pcg(
    air_diagonal: tf.Tensor,
    air_rhs: tf.Tensor,
    coupling_left: tf.Tensor,
    coupling_right: tf.Tensor,
    coupling_top: tf.Tensor,
    coupling_bottom: tf.Tensor,
    exterior_mask: tf.Tensor,
    mass_mask: tf.Tensor,
    mass_conductance: tf.Tensor,
    mass_storage: tf.Tensor,
    previous_mass_temperature: tf.Tensor,
    initial_air_temperature: tf.Tensor,
    initial_mass_temperature: tf.Tensor,
    tolerance: tf.Tensor,
    iteration_limit: tf.Tensor,
    mass_gain: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Solves the symmetric air/mass block system with Jacobi PCG."""
  dtype = air_diagonal.dtype
  active_air_mask = tf.math.logical_not(exterior_mask)
  mass_indices = tf.cast(tf.where(mass_mask), tf.int32)
  mass_g = tf.boolean_mask(mass_conductance, mass_mask)
  mass_c = tf.boolean_mask(mass_storage, mass_mask)
  mass_diagonal = mass_c + mass_g
  air_size = tf.size(air_diagonal)

  def _split(vector: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    return (
        tf.reshape(vector[:air_size], tf.shape(air_diagonal)),
        vector[air_size:],
    )

  def _matvec(vector: tf.Tensor) -> tf.Tensor:
    air, mass = _split(vector)
    active_air = tf.where(active_air_mask, air, tf.zeros_like(air))
    zero_column = tf.zeros_like(active_air[:, :1])
    zero_row = tf.zeros_like(active_air[:1, :])
    left = tf.concat((zero_column, active_air[:, :-1]), axis=1)
    right = tf.concat((active_air[:, 1:], zero_column), axis=1)
    top = tf.concat((zero_row, active_air[:-1, :]), axis=0)
    bottom = tf.concat((active_air[1:, :], zero_row), axis=0)
    mass_grid = tf.scatter_nd(mass_indices, mass, tf.shape(air))
    air_result = air_diagonal * air
    air_result -= coupling_left * left
    air_result -= coupling_right * right
    air_result -= coupling_top * top
    air_result -= coupling_bottom * bottom
    air_result -= mass_conductance * mass_grid
    air_result = tf.where(exterior_mask, air, air_result)
    mass_result = mass_diagonal * mass - mass_g * tf.boolean_mask(
        air, mass_mask
    )
    return tf.concat((tf.reshape(air_result, [-1]), mass_result), axis=0)

  mass_rhs = mass_c * tf.boolean_mask(previous_mass_temperature, mass_mask)
  if mass_gain is not None:
    mass_rhs += tf.boolean_mask(mass_gain, mass_mask)
  rhs = tf.concat((tf.reshape(air_rhs, [-1]), mass_rhs), axis=0)
  diagonal = tf.concat((tf.reshape(air_diagonal, [-1]), mass_diagonal), axis=0)
  solution = tf.concat(
      (
          tf.reshape(initial_air_temperature, [-1]),
          tf.boolean_mask(initial_mass_temperature, mass_mask),
      ),
      axis=0,
  )
  residual = rhs - _matvec(solution)
  preconditioned_residual = residual / diagonal
  direction = preconditioned_residual
  residual_dot = tf.tensordot(residual, preconditioned_residual, axes=1)
  # Use the air RHS norm, as the legacy PCG does.  The mass storage RHS can be
  # much larger and would otherwise silently loosen the absolute tolerance
  # when mass nodes are appended to the vector.
  target_norm = tolerance * tf.maximum(
      tf.linalg.norm(tf.reshape(air_rhs, [-1])),
      tf.constant(1.0, dtype=dtype),
  )
  iteration = tf.constant(0, dtype=tf.int32)

  def _condition(
      iteration,
      solution,
      residual,
      preconditioned_residual,
      direction,
      residual_dot,
  ):
    del solution, preconditioned_residual, direction, residual_dot
    return tf.logical_and(
        iteration < iteration_limit,
        tf.linalg.norm(residual) > target_norm,
    )

  def _body(
      iteration,
      solution,
      residual,
      preconditioned_residual,
      direction,
      residual_dot,
  ):
    del preconditioned_residual
    matrix_direction = _matvec(direction)
    step = residual_dot / tf.tensordot(direction, matrix_direction, axes=1)
    next_solution = solution + step * direction
    restart = tf.equal(tf.math.floormod(iteration + 1, 5), 0)
    recursive_residual = residual - step * matrix_direction
    next_residual = tf.cond(
        restart,
        lambda: rhs - _matvec(next_solution),
        lambda: recursive_residual,
    )
    next_preconditioned = next_residual / diagonal
    next_residual_dot = tf.tensordot(next_residual, next_preconditioned, axes=1)
    beta = next_residual_dot / residual_dot
    next_direction = tf.cond(
        restart,
        lambda: next_preconditioned,
        lambda: next_preconditioned + beta * direction,
    )
    return (
        iteration + 1,
        next_solution,
        next_residual,
        next_preconditioned,
        next_direction,
        next_residual_dot,
    )

  iteration, solution, residual, _, _, _ = tf.while_loop(
      _condition,
      _body,
      (
          iteration,
          solution,
          residual,
          preconditioned_residual,
          direction,
          residual_dot,
      ),
  )
  air_solution, mass_solution = _split(solution)
  residual = rhs - _matvec(solution)
  mass_grid = tf.tensor_scatter_nd_update(
      previous_mass_temperature, mass_indices, mass_solution
  )
  return air_solution, mass_grid, tf.linalg.norm(residual), iteration


class TFCoupledInteriorMassSimulator(
    tf_device_resident_simulator.TFDeviceResidentSimulator
):
  """Device-resident PCG simulator with mass nodes in the linear solve."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.last_linear_solve_count = 0
    if self.include_interior_mass:
      self._t_mass_conductance = (
          self._t_interior_mass_conductivity
          * self._t_u
          * self._t_v
          / self._t_floor_height
      )
      self._t_mass_storage = (
          self._t_interior_mass_density
          * self._t_interior_mass_heat_capacity
          * self._t_u
          * self._t_v
          * self._t_floor_height
          / self._t_time_step
      )

  def _solve_coupled_on_device(
      self,
      temperature_estimate: tf.Tensor,
      mass_temperature_estimate: tf.Tensor,
      previous_temperature: tf.Tensor,
      previous_mass_temperature: tf.Tensor,
      input_q: tf.Tensor,
      ambient_temperature: tf.Tensor,
      convection_coefficient: tf.Tensor,
      exterior_forcing,
  ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    (
        gamma_ambient_left,
        gamma_ambient_right,
        gamma_ambient_top,
        gamma_ambient_bottom,
    ) = self._ambient_conductances(convection_coefficient)
    uz = self._t_floor_height * self._t_u
    vz = self._t_floor_height * self._t_v
    radiative_temperatures = self._radiative_surface_temperatures(
        temperature_estimate, mass_temperature_estimate
    )
    lwx_coefficient, lwx_driving = self._get_longwave_grid_terms(
        radiative_temperatures, self._t_floor_height
    )
    mass_gain = None
    if exterior_forcing is not None:
      coefficient, driving = exterior_forcing.linearized_air_terms(
          temperature_estimate
      )
      lwx_coefficient += coefficient
      input_q += driving
      mass_gain = exterior_forcing.mass_gain
    diagonal = self._get_denominator(
        self._t_face_conductance_left,
        self._t_face_conductance_right,
        gamma_ambient_left,
        gamma_ambient_right,
        gamma_ambient_top,
        gamma_ambient_bottom,
        vz,
        self._t_face_conductance_bottom,
        self._t_face_conductance_top,
        uz,
        self._t_density_static,
        self._t_heat_capacity_static,
        self._t_floor_height,
        self._t_time_step,
        lwx_coefficient,
    )
    zeros = tf.zeros_like(temperature_estimate)
    rhs = self._get_numerator(
        self._t_face_conductance_left,
        self._t_face_conductance_right,
        gamma_ambient_left,
        gamma_ambient_right,
        gamma_ambient_top,
        gamma_ambient_bottom,
        vz,
        self._t_face_conductance_bottom,
        self._t_face_conductance_top,
        uz,
        self._t_density_static,
        self._t_heat_capacity_static,
        self._t_floor_height,
        self._t_time_step,
        zeros,
        zeros,
        zeros,
        zeros,
        ambient_temperature,
        input_q,
        previous_temperature,
        lwx_driving,
        zeros,
    )
    coupling_left = vz * self._t_face_conductance_left
    coupling_right = vz * self._t_face_conductance_right
    coupling_top = uz * self._t_face_conductance_top
    coupling_bottom = uz * self._t_face_conductance_bottom
    exterior_float = tf.cast(self._t_exerior_temps_mask, tf.float32)
    rhs += ambient_temperature * (
        coupling_left * tf_simulator.shift_tensor_right(exterior_float)
        + coupling_right * tf_simulator.shift_tensor_left(exterior_float)
        + coupling_top * tf_simulator.shift_tensor_down(exterior_float)
        + coupling_bottom * tf_simulator.shift_tensor_up(exterior_float)
    )
    ambient_field = tf.fill(tf.shape(rhs), ambient_temperature)
    rhs = tf.where(self._t_exerior_temps_mask, ambient_field, rhs)
    diagonal = tf.where(
        self._t_exerior_temps_mask, tf.ones_like(diagonal), diagonal
    )
    initial_air = tf.where(
        self._t_exerior_temps_mask, ambient_field, temperature_estimate
    )
    # The augmented system has mass-storage and air rows at very different
    # scales.  Accumulate PCG in FP64, then return FP32 resident state.  This
    # mirrors TFSimulator's host PCG and prevents recursive-residual drift from
    # reporting convergence before the mass block is balanced.
    air, mass, residual_norm, iterations = _solve_coupled_pcg(
        tf.cast(diagonal, tf.float64),
        tf.cast(rhs, tf.float64),
        tf.cast(coupling_left, tf.float64),
        tf.cast(coupling_right, tf.float64),
        tf.cast(coupling_top, tf.float64),
        tf.cast(coupling_bottom, tf.float64),
        self._t_exerior_temps_mask,
        self._t_interior_mass_mask,
        tf.cast(self._t_mass_conductance, tf.float64),
        tf.cast(self._t_mass_storage, tf.float64),
        tf.cast(previous_mass_temperature, tf.float64),
        tf.cast(initial_air, tf.float64),
        tf.cast(mass_temperature_estimate, tf.float64),
        tf.cast(self._t_cg_tolerance, tf.float64),
        self._t_cg_iteration_limit,
        tf.cast(mass_gain, tf.float64) if mass_gain is not None else None,
    )
    air = tf.cast(air, tf.float32)
    mass = tf.cast(mass, tf.float32)
    residual_norm = tf.cast(residual_norm, tf.float32)
    air = tf.where(self._t_exerior_temps_mask, ambient_field, air)
    residual_target = self._t_cg_tolerance * tf.maximum(
        tf.linalg.norm(rhs), tf.constant(1.0, dtype=tf.float32)
    )
    return air, mass, residual_norm, residual_target, iterations

  @tf.function(reduce_retracing=True)
  def _coupled_device_timestep(
      self,
      ambient_temperature: tf.Tensor,
      convection_coefficient: tf.Tensor,
      input_q: tf.Tensor,
      exterior_forcing,
  ):
    previous_temperature = self._device_temperature.read_value()
    previous_mass_temperature = self._device_mass_temperature.read_value()
    nonlinear = (
        self.include_radiative_heat_transfer or exterior_forcing is not None
    )
    outer_limit = tf.constant(
        self._iteration_limit if nonlinear else 1, tf.int32
    )
    iteration = tf.constant(0, dtype=tf.int32)
    max_delta = tf.constant(np.inf, dtype=tf.float32)
    residual_norm = tf.constant(np.inf, dtype=tf.float32)
    residual_target = tf.constant(0.0, dtype=tf.float32)
    total_cg_iterations = tf.constant(0, dtype=tf.int32)

    def _condition(
        iteration,
        temperature,
        mass_temperature,
        max_delta,
        residual_norm,
        residual_target,
        total_cg_iterations,
    ):
      del temperature, mass_temperature, residual_norm, residual_target
      del total_cg_iterations
      needs_iteration = tf.logical_or(
          iteration == 0, max_delta > self._t_convergence_threshold
      )
      return tf.logical_and(iteration < outer_limit, needs_iteration)

    def _body(
        iteration,
        temperature,
        mass_temperature,
        max_delta,
        residual_norm,
        residual_target,
        total_cg_iterations,
    ):
      del max_delta, residual_norm, residual_target
      (
          next_temperature,
          next_mass_temperature,
          next_residual_norm,
          next_residual_target,
          cg_iterations,
      ) = self._solve_coupled_on_device(
          temperature,
          mass_temperature,
          previous_temperature,
          previous_mass_temperature,
          input_q,
          ambient_temperature,
          convection_coefficient,
          exterior_forcing,
      )
      next_delta = tf.maximum(
          tf.reduce_max(tf.abs(next_temperature - temperature)),
          tf.reduce_max(tf.abs(next_mass_temperature - mass_temperature)),
      )
      return (
          iteration + 1,
          next_temperature,
          next_mass_temperature,
          next_delta,
          next_residual_norm,
          next_residual_target,
          total_cg_iterations + cg_iterations,
      )

    result = tf.while_loop(
        _condition,
        _body,
        (
            iteration,
            previous_temperature,
            previous_mass_temperature,
            max_delta,
            residual_norm,
            residual_target,
            total_cg_iterations,
        ),
    )
    (
        outer_iterations,
        temperature,
        mass_temperature,
        max_delta,
        residual_norm,
        residual_target,
        total_cg_iterations,
    ) = result
    outer_converged = tf.logical_or(
        tf.constant(not nonlinear),
        max_delta <= self._t_convergence_threshold,
    )
    return (
        outer_converged,
        residual_norm <= residual_target,
        outer_iterations,
        total_cg_iterations,
        residual_norm,
        max_delta,
        temperature,
        mass_temperature,
    )

  def warmup(
      self, *, ambient_temperature: float, convection_coefficient: float
  ) -> None:
    if not self.include_interior_mass:
      super().warmup(
          ambient_temperature=ambient_temperature,
          convection_coefficient=convection_coefficient,
      )
      return
    self._coupled_device_timestep(
        tf.constant(ambient_temperature, dtype=tf.float32),
        tf.constant(convection_coefficient, dtype=tf.float32),
        tf.convert_to_tensor(self.building.input_q, dtype=tf.float32),
        self._prepare_tf_exterior_radiation(ambient_temperature),
    )

  def finite_differences_timestep(
      self, *, ambient_temperature: float, convection_coefficient: float
  ) -> bool:
    if not self.include_interior_mass:
      result = super().finite_differences_timestep(
          ambient_temperature=ambient_temperature,
          convection_coefficient=convection_coefficient,
      )
      self.last_linear_solve_count = self.last_outer_iteration_count
      return result
    (
        outer_converged,
        cg_converged,
        outer_iterations,
        cg_iterations,
        residual_norm,
        max_delta,
        temperature,
        mass_temperature,
    ) = self._coupled_device_timestep(
        tf.constant(ambient_temperature, dtype=tf.float32),
        tf.constant(convection_coefficient, dtype=tf.float32),
        tf.convert_to_tensor(self.building.input_q, dtype=tf.float32),
        self._prepare_tf_exterior_radiation(ambient_temperature),
    )
    self.last_outer_iteration_count = int(outer_iterations.numpy())
    self.last_linear_solve_count = self.last_outer_iteration_count
    self.last_outer_max_delta = float(max_delta.numpy())
    self.last_cg_iteration_count = int(cg_iterations.numpy())
    self.last_cg_residual_norm = float(residual_norm.numpy())
    if not bool(cg_converged.numpy()):
      raise RuntimeError(
          "Coupled TensorFlow CG did not converge: "
          f"iterations={self.last_cg_iteration_count}, "
          f"residual_norm={self.last_cg_residual_norm}"
      )
    if bool(outer_converged.numpy()):
      self._device_temperature.assign(temperature)
      self._device_mass_temperature.assign(mass_temperature)
      if self._sync_to_host_each_step:
        self.sync_state_to_building()
      return True
    return False


def _radiation_terms(
    building, air_temperature: np.ndarray, mass_temperature: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
  if not building.include_radiative_heat_transfer:
    return None, None, None
  surface_mask = building.interior_wall_mask | building.interior_mass_mask
  surface_temperature = np.zeros_like(air_temperature)
  surface_temperature[building.interior_wall_mask] = air_temperature[
      building.interior_wall_mask
  ]
  surface_temperature[building.interior_mass_mask] = mass_temperature[
      building.interior_mass_mask
  ]
  temperatures = surface_temperature[surface_mask]
  coefficient = np.zeros_like(temperatures, dtype=float)
  driving = np.zeros_like(temperatures, dtype=float)
  for enclosure_indices, ifa_inv in zip(
      building.radiative_enclosure_indices, building.ifa_inv_by_enclosure
  ):
    enclosure_coefficient, enclosure_driving = (
        building_radiation_utils.calculate_radiative_exchange_terms(
            temperatures[enclosure_indices], ifa_inv
        )
    )
    coefficient[enclosure_indices] = enclosure_coefficient
    driving[enclosure_indices] = enclosure_driving
  return (
      coefficient,
      driving,
      building.interior_wall_mask[surface_mask],
  )


def _assemble_coupled_scipy_system(
    simulator,
    air_temperature: np.ndarray,
    mass_temperature: np.ndarray,
    previous_air_temperature: np.ndarray,
    previous_mass_temperature: np.ndarray,
    ambient_temperature: float,
    convection_coefficient: float,
    exterior_forcing=None,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
  """Builds the coupled block matrix and RHS used by the SciPy backend."""
  building = simulator.building
  rows_count, columns_count = building.temp.shape
  air_count = rows_count * columns_count
  flat_index = np.arange(air_count).reshape(rows_count, columns_count)
  exterior = np.fromiter(
      (
          len(building.neighbors[row][column]) <= 1
          for row in range(rows_count)
          for column in range(columns_count)
      ),
      dtype=bool,
      count=air_count,
  ).reshape(rows_count, columns_count)
  mass_coordinates = np.argwhere(building.interior_mass_mask)
  mass_index = np.full(building.temp.shape, -1, dtype=int)
  for offset, (row, column) in enumerate(mass_coordinates):
    mass_index[row, column] = air_count + offset
  system_size = air_count + len(mass_coordinates)
  matrix_rows: list[int] = []
  matrix_columns: list[int] = []
  matrix_values: list[float] = []
  rhs = np.zeros(system_size, dtype=np.float64)
  floor_height = building.floor_height_cm / 100.0
  radiation_area = building.cv_size_cm / 100.0 * floor_height
  radiation_coefficient, radiation_driving, radiation_is_wall = (
      _radiation_terms(building, air_temperature, mass_temperature)
  )
  exterior_terms = (
      exterior_forcing.linearized_air_terms(air_temperature)
      if exterior_forcing is not None
      else None
  )

  for row in range(rows_count):
    for column in range(columns_count):
      index = flat_index[row, column]
      if exterior[row, column]:
        matrix_rows.append(index)
        matrix_columns.append(index)
        matrix_values.append(1.0)
        rhs[index] = ambient_temperature
        continue
      u, v = simulator._get_cv_dimensions((row, column))  # pylint: disable=protected-access
      storage = (
          building.density[row, column]
          * building.heat_capacity[row, column]
          * u
          * v
          * floor_height
          / simulator._time_step_sec  # pylint: disable=protected-access
      )
      diagonal = storage
      rhs_value = (
          storage * previous_air_temperature[row, column]
          + building.input_q[row, column]
      )
      if exterior_terms is not None:
        exterior_coefficient, exterior_driving = exterior_terms
        diagonal += exterior_coefficient[row, column]
        rhs_value += exterior_driving[row, column]
      for neighbor_row, neighbor_column in building.neighbors[row][column]:
        conductance = simulator._get_face_conductance(  # pylint: disable=protected-access
            (row, column), (neighbor_row, neighbor_column)
        )
        diagonal += conductance
        if exterior[neighbor_row, neighbor_column]:
          rhs_value += conductance * ambient_temperature
        else:
          matrix_rows.append(index)
          matrix_columns.append(flat_index[neighbor_row, neighbor_column])
          matrix_values.append(-conductance)
      ambient_conductance = simulator._get_ambient_conductance(  # pylint: disable=protected-access
          (row, column), convection_coefficient
      )
      diagonal += ambient_conductance
      rhs_value += ambient_conductance * ambient_temperature
      if building.interior_mass_mask[row, column]:
        mass_conductance = (
            building.interior_mass_conductivity[row, column]
            * u
            * v
            / floor_height
        )
        diagonal += mass_conductance
        matrix_rows.append(index)
        matrix_columns.append(mass_index[row, column])
        matrix_values.append(-mass_conductance)
      if radiation_coefficient is not None:
        radiation_index = building.lwx_index[row, column]
        if radiation_index >= 0:
          coefficient = radiation_coefficient[radiation_index]
          driving = radiation_driving[radiation_index]
          if radiation_is_wall[radiation_index]:
            diagonal += coefficient * radiation_area
            rhs_value += driving * radiation_area
          else:
            rhs_value += (
                driving - coefficient * mass_temperature[row, column]
            ) * radiation_area
      matrix_rows.append(index)
      matrix_columns.append(index)
      matrix_values.append(diagonal)
      rhs[index] = rhs_value

  for row, column in mass_coordinates:
    u, v = simulator._get_cv_dimensions((row, column))  # pylint: disable=protected-access
    index = mass_index[row, column]
    conductance = (
        building.interior_mass_conductivity[row, column] * u * v / floor_height
    )
    storage = (
        building.interior_mass_density[row, column]
        * building.interior_mass_heat_capacity[row, column]
        * u
        * v
        * floor_height
        / simulator._time_step_sec  # pylint: disable=protected-access
    )
    matrix_rows.extend((index, index))
    matrix_columns.extend((flat_index[row, column], index))
    matrix_values.extend((-conductance, storage + conductance))
    rhs[index] = storage * previous_mass_temperature[row, column]
    if exterior_forcing is not None:
      rhs[index] += exterior_forcing.mass_gain[row, column]
  system = sparse.csr_matrix(
      (matrix_values, (matrix_rows, matrix_columns)),
      shape=(system_size, system_size),
  )
  return system, rhs, mass_coordinates, mass_index


class NumpyScipyCoupledInteriorMassSimulator(
    numpy_scipy_simulator.NumpyScipySimulator
):
  """SciPy direct simulator using the same coupled air/mass equations."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.last_linear_solve_count = 0
    self.last_linear_residual_norm = np.nan

  def finite_differences_timestep(
      self, *, ambient_temperature: float, convection_coefficient: float
  ) -> bool:
    if not self.building.include_interior_mass:
      result = super().finite_differences_timestep(
          ambient_temperature=ambient_temperature,
          convection_coefficient=convection_coefficient,
      )
      self.last_linear_solve_count = self.last_outer_iteration_count
      return result
    previous_air = self.building.temp.copy()
    previous_mass = self.building.interior_mass_temp.copy()
    air_estimate = previous_air.copy()
    mass_estimate = previous_mass.copy()
    exterior_forcing = self._prepare_exterior_radiation(ambient_temperature)
    nonlinear = (
        self.building.include_radiative_heat_transfer
        or exterior_forcing is not None
    )
    outer_limit = self._iteration_limit if nonlinear else 1
    converged = False
    max_delta = np.inf
    for iteration in range(outer_limit):
      system, rhs, mass_coordinates, _ = _assemble_coupled_scipy_system(
          self,
          air_estimate,
          mass_estimate,
          previous_air,
          previous_mass,
          ambient_temperature,
          convection_coefficient,
          exterior_forcing,
      )
      solution = sparse_linalg.spsolve(system, rhs)
      next_air = solution[: previous_air.size].reshape(previous_air.shape)
      next_mass = previous_mass.copy()
      for offset, (row, column) in enumerate(mass_coordinates):
        next_mass[row, column] = solution[previous_air.size + offset]
      max_delta = max(
          float(np.max(np.abs(next_air - air_estimate))),
          float(np.max(np.abs(next_mass - mass_estimate))),
      )
      air_estimate = next_air
      mass_estimate = next_mass
      self.last_linear_residual_norm = float(
          np.linalg.norm(system @ solution - rhs)
      )
      if not nonlinear:
        converged = True
        break
      if max_delta <= self._convergence_threshold:
        converged = True
        break
    self.last_outer_iteration_count = iteration + 1
    self.last_linear_solve_count = iteration + 1
    self.last_outer_max_delta = max_delta
    if converged:
      self.building.temp = air_estimate
      self.building.interior_mass_temp = mass_estimate
    else:
      logging.warning(
          "Coupled SciPy solve did not converge in %d outer iterations; "
          "max_delta=%g",
          self.last_outer_iteration_count,
          max_delta,
      )
    return converged
