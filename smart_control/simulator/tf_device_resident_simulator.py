"""Device-resident TensorFlow heat-balance simulator."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from smart_control.simulator import building_radiation_utils
from smart_control.simulator import tf_simulator


class TFDeviceResidentSimulator(tf_simulator.TFSimulator):
  """Runs the complete Picard timestep and FP32 PCG solve on one device."""

  def __init__(
      self,
      *args,
      cg_tolerance: float = 1e-8,
      cg_iteration_limit: int = 2000,
      sync_to_host_each_step: bool = True,
      **kwargs,
  ):
    kwargs.pop("solver_method", None)
    super().__init__(
        *args,
        solver_method="cg",
        cg_tolerance=cg_tolerance,
        cg_iteration_limit=cg_iteration_limit,
        **kwargs,
    )
    self._sync_to_host_each_step = sync_to_host_each_step
    self._t_density_static = tf.constant(
        self.building.density, dtype=tf.float32
    )
    self._t_heat_capacity_static = tf.constant(
        self.building.heat_capacity, dtype=tf.float32
    )
    self._t_floor_height = tf.constant(
        self.building.floor_height_cm / 100.0, dtype=tf.float32
    )
    self._t_time_step = tf.constant(self._time_step_sec, dtype=tf.float32)
    self._t_convergence_threshold = tf.constant(
        self._convergence_threshold, dtype=tf.float32
    )
    self._t_iteration_limit = tf.constant(self._iteration_limit, dtype=tf.int32)
    self._t_cg_tolerance = tf.constant(self.cg_tolerance, dtype=tf.float32)
    self._t_cg_iteration_limit = tf.constant(
        self.cg_iteration_limit, dtype=tf.int32
    )
    (
        ambient_left,
        ambient_right,
        ambient_top,
        ambient_bottom,
    ) = tf_simulator.get_oriented_convection_coefficient_tensors(
        1.0,
        self.building.conductivity.shape,
        self._boundary_cv_mapping,
    )
    self._t_ambient_face_masks = (
        ambient_left > 0.0,
        ambient_right > 0.0,
        ambient_top > 0.0,
        ambient_bottom > 0.0,
    )
    (
        self._t_half_resistance_u,
        self._t_half_resistance_v,
    ) = tf_simulator.get_half_cell_resistance_tensors(
        self.building.conductivity, self._t_u, self._t_v
    )
    interior_wall_mask = self.building.interior_wall_mask
    if interior_wall_mask is None:
      interior_wall_mask = np.zeros_like(self.building.temp, dtype=bool)
    self._t_interior_wall_mask = tf.constant(interior_wall_mask, dtype=tf.bool)
    if self.include_interior_mass:
      self._t_radiative_surface_mask = tf.math.logical_or(
          self._t_interior_wall_mask, self._t_interior_mass_mask
      )
      initial_mass_temperature = self.building.interior_mass_temp
    else:
      self._t_radiative_surface_mask = self._t_interior_wall_mask
      initial_mass_temperature = np.zeros_like(self.building.temp)
    self._device_temperature = tf.Variable(
        self.building.temp, dtype=tf.float32, trainable=False
    )
    self._device_mass_temperature = tf.Variable(
        initial_mass_temperature, dtype=tf.float32, trainable=False
    )

  def reset(self):
    super().reset()
    if hasattr(self, "_device_temperature"):
      self.sync_state_from_building()

  def sync_state_from_building(self) -> None:
    """Copies externally modified building state to the resident variables."""
    self._device_temperature.assign(
        tf.convert_to_tensor(self.building.temp, dtype=tf.float32)
    )
    if self.include_interior_mass:
      self._device_mass_temperature.assign(
          tf.convert_to_tensor(
              self.building.interior_mass_temp, dtype=tf.float32
          )
      )

  def sync_state_to_building(self) -> None:
    """Copies resident state to NumPy-backed building arrays."""
    self.building.temp = self._device_temperature.numpy()
    if self.include_interior_mass:
      self.building.interior_mass_temp = self._device_mass_temperature.numpy()

  def _ambient_conductances(
      self, convection_coefficient: tf.Tensor
  ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    def _face(mask: tf.Tensor, resistance: tf.Tensor) -> tf.Tensor:
      coefficient = tf.fill(tf.shape(resistance), convection_coefficient)
      conductance = tf.math.reciprocal(
          tf.math.reciprocal(coefficient) + resistance
      )
      return tf.where(mask, conductance, tf.zeros_like(conductance))

    left_mask, right_mask, top_mask, bottom_mask = self._t_ambient_face_masks
    return (
        _face(left_mask, self._t_half_resistance_u),
        _face(right_mask, self._t_half_resistance_u),
        _face(top_mask, self._t_half_resistance_v),
        _face(bottom_mask, self._t_half_resistance_v),
    )

  def _radiative_surface_temperatures(
      self, air_temperature: tf.Tensor, mass_temperature: tf.Tensor
  ) -> tf.Tensor:
    if not self.include_radiative_heat_transfer:
      return tf.zeros((1, 1), dtype=tf.float32)
    surface_temperature = air_temperature
    if self.include_interior_mass:
      surface_temperature = tf.where(
          self._t_interior_mass_mask,
          mass_temperature,
          surface_temperature,
      )
    return tf.reshape(
        tf.boolean_mask(surface_temperature, self._t_radiative_surface_mask),
        [-1, 1],
    )

  def _update_mass_on_device(
      self,
      air_temperature: tf.Tensor,
      previous_step_mass_temperature: tf.Tensor,
      exterior_forcing: (
          building_radiation_utils.ExteriorRadiationForcing | None
      ),
  ) -> tf.Tensor:
    if not self.include_interior_mass:
      return previous_step_mass_temperature
    numerator = (
        air_temperature + self._t_0_mass * previous_step_mass_temperature
    )
    if exterior_forcing is not None:
      conductance = (
          self._t_interior_mass_conductivity
          * self._t_u
          * self._t_v
          / self._t_floor_height
      )
      numerator += tf.math.divide_no_nan(
          exterior_forcing.mass_gain, conductance
      )
    updated = numerator / (1.0 + self._t_0_mass)
    return tf.where(
        self._t_interior_mass_mask,
        updated,
        previous_step_mass_temperature,
    )

  def _solve_air_on_device(
      self,
      temperature_estimate: tf.Tensor,
      previous_step_temperature: tf.Tensor,
      mass_temperature: tf.Tensor,
      input_q: tf.Tensor,
      ambient_temperature: tf.Tensor,
      convection_coefficient: tf.Tensor,
      exterior_forcing: (
          building_radiation_utils.ExteriorRadiationForcing | None
      ),
  ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    (
        gamma_ambient_left,
        gamma_ambient_right,
        gamma_ambient_top,
        gamma_ambient_bottom,
    ) = self._ambient_conductances(convection_coefficient)
    uz = self._t_floor_height * self._t_u
    vz = self._t_floor_height * self._t_v
    radiative_temperatures = self._radiative_surface_temperatures(
        temperature_estimate, mass_temperature
    )
    lwx_coefficient, lwx_driving = self._get_longwave_grid_terms(
        radiative_temperatures, self._t_floor_height
    )
    if exterior_forcing is not None:
      coefficient, driving = exterior_forcing.linearized_air_terms(
          temperature_estimate
      )
      lwx_coefficient += coefficient
      input_q += driving
    denominator = self._get_denominator(
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
        previous_step_temperature,
        lwx_driving,
        mass_temperature,
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
        self._t_exerior_temps_mask, tf.ones_like(denominator), denominator
    )
    initial = tf.where(
        self._t_exerior_temps_mask,
        ambient_field,
        temperature_estimate,
    )
    temperature, residual_norm, iteration_count = (
        tf_simulator._solve_with_preconditioned_conjugate_gradient(  # pylint: disable=protected-access
            diagonal,
            rhs,
            coupling_left,
            coupling_right,
            coupling_top,
            coupling_bottom,
            self._t_exerior_temps_mask,
            initial,
            self._t_cg_tolerance,
            self._t_cg_iteration_limit,
        )
    )
    temperature = tf.where(
        self._t_exerior_temps_mask, ambient_field, temperature
    )
    residual_target = self._t_cg_tolerance * tf.maximum(
        tf.linalg.norm(rhs), tf.constant(1.0, dtype=tf.float32)
    )
    return temperature, residual_norm, residual_target, iteration_count

  @tf.function(reduce_retracing=True)
  def _device_timestep(
      self,
      ambient_temperature: tf.Tensor,
      convection_coefficient: tf.Tensor,
      input_q: tf.Tensor,
      exterior_forcing: (
          building_radiation_utils.ExteriorRadiationForcing | None
      ),
  ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    previous_temperature = self._device_temperature.read_value()
    previous_mass_temperature = self._device_mass_temperature.read_value()
    complete_linear_solve = not (
        self.include_radiative_heat_transfer
        or self.include_interior_mass
        or exterior_forcing is not None
    )
    outer_limit = tf.constant(
        1 if complete_linear_solve else self._iteration_limit, dtype=tf.int32
    )
    iteration = tf.constant(0, dtype=tf.int32)
    max_delta = tf.constant(np.inf, dtype=tf.float32)
    residual_norm = tf.constant(np.inf, dtype=tf.float32)
    residual_target = tf.constant(0.0, dtype=tf.float32)
    cg_iterations = tf.constant(0, dtype=tf.int32)

    def _condition(
        iteration,
        temperature,
        mass_temperature,
        max_delta,
        residual_norm,
        residual_target,
        cg_iterations,
    ):
      del temperature, mass_temperature, residual_norm, residual_target
      del cg_iterations
      return tf.logical_and(
          iteration < outer_limit,
          max_delta > self._t_convergence_threshold,
      )

    def _body(
        iteration,
        temperature,
        mass_temperature,
        max_delta,
        residual_norm,
        residual_target,
        cg_iterations,
    ):
      del max_delta, residual_norm, residual_target, cg_iterations
      (
          next_temperature,
          next_residual_norm,
          next_residual_target,
          next_cg_iterations,
      ) = self._solve_air_on_device(
          temperature,
          previous_temperature,
          mass_temperature,
          input_q,
          ambient_temperature,
          convection_coefficient,
          exterior_forcing,
      )
      next_mass_temperature = self._update_mass_on_device(
          next_temperature, previous_mass_temperature, exterior_forcing
      )
      air_delta = tf.reduce_max(tf.abs(next_temperature - temperature))
      if self.include_interior_mass:
        mass_delta = tf.reduce_max(
            tf.abs(next_mass_temperature - mass_temperature)
        )
        next_max_delta = tf.maximum(air_delta, mass_delta)
      else:
        next_max_delta = air_delta
      return (
          iteration + 1,
          next_temperature,
          next_mass_temperature,
          next_max_delta,
          next_residual_norm,
          next_residual_target,
          next_cg_iterations,
      )

    (
        outer_iterations,
        temperature,
        mass_temperature,
        max_delta,
        residual_norm,
        residual_target,
        cg_iterations,
    ) = tf.while_loop(
        _condition,
        _body,
        (
            iteration,
            previous_temperature,
            previous_mass_temperature,
            max_delta,
            residual_norm,
            residual_target,
            cg_iterations,
        ),
    )
    self._device_temperature.assign(temperature)
    self._device_mass_temperature.assign(mass_temperature)
    outer_converged = tf.logical_or(
        tf.constant(complete_linear_solve),
        max_delta <= self._t_convergence_threshold,
    )
    cg_converged = residual_norm <= residual_target
    return (
        outer_converged,
        cg_converged,
        outer_iterations,
        cg_iterations,
        residual_norm,
        max_delta,
    )

  def _prepare_tf_exterior_radiation(self, ambient_temperature):
    return self._tf_exterior_forcing(
        self._prepare_exterior_radiation(ambient_temperature)
    )

  def warmup(
      self, *, ambient_temperature: float, convection_coefficient: float
  ) -> None:
    """Traces the complete timestep without advancing resident state."""
    temperature = self._device_temperature.read_value()
    mass_temperature = self._device_mass_temperature.read_value()
    self._device_timestep(
        tf.constant(ambient_temperature, dtype=tf.float32),
        tf.constant(convection_coefficient, dtype=tf.float32),
        tf.convert_to_tensor(self.building.input_q, dtype=tf.float32),
        self._prepare_tf_exterior_radiation(ambient_temperature),
    )
    self._device_temperature.assign(temperature)
    self._device_mass_temperature.assign(mass_temperature)

  def finite_differences_timestep(
      self, *, ambient_temperature: float, convection_coefficient: float
  ) -> bool:
    """Runs one complete device-resident FP32 PCG timestep."""
    (
        outer_converged,
        cg_converged,
        outer_iterations,
        cg_iterations,
        residual_norm,
        max_delta,
    ) = self._device_timestep(
        tf.constant(ambient_temperature, dtype=tf.float32),
        tf.constant(convection_coefficient, dtype=tf.float32),
        tf.convert_to_tensor(self.building.input_q, dtype=tf.float32),
        self._prepare_tf_exterior_radiation(ambient_temperature),
    )
    self.last_outer_iteration_count = int(outer_iterations.numpy())
    self.last_outer_max_delta = float(max_delta.numpy())
    self.last_cg_iteration_count = int(cg_iterations.numpy())
    self.last_cg_residual_norm = float(residual_norm.numpy())
    if not bool(cg_converged.numpy()):
      raise RuntimeError(
          "Device-resident TensorFlow CG did not converge: "
          f"iterations={self.last_cg_iteration_count}, "
          f"residual_norm={self.last_cg_residual_norm}"
      )
    if self._sync_to_host_each_step:
      self.sync_state_to_building()
    return bool(outer_converged.numpy())
