"""Group straight envelope segments and condense the linear heat-balance rows."""

import numpy as np
from scipy import linalg
from scipy.sparse import linalg as sparse_linalg

from smart_control.simulator import building_radiation_utils
from smart_control.simulator import constants
from smart_control.simulator import grouped_heat_balance

from . import reduced_air


group_envelope_cells = grouped_heat_balance.group_envelope_cells


class ReducedEnvelopeSimulator(reduced_air.ReducedAirSimulator):
  """Contract air, straight wall runs, and glazing into thermal nodes."""

  def _group_positions(self, solved, floor_plan, rooms):
    groups = super()._group_positions(solved, floor_plan, rooms)
    return group_envelope_cells(solved, floor_plan, rooms, groups)


class _SurfaceSystem:
  """Eliminate temperature-independent rows from a symmetric heat matrix."""

  def __init__(self, matrix, emission):
    self.surface = np.flatnonzero(emission)
    self.interior = np.flatnonzero(emission == 0)
    s, i = self.surface, self.interior
    interior_matrix = matrix[i][:, i].tocsc()
    self.interior_lu = sparse_linalg.splu(interior_matrix) if len(i) else None
    self.interior_to_surface = matrix[s][:, i]
    coupling = self.interior_to_surface.T.toarray()
    self.response = (
        self.interior_lu.solve(coupling) if len(i) else coupling
    )
    self.surface_matrix = (
        matrix[s][:, s].toarray() - self.interior_to_surface @ self.response
    )
    self.diagonal = np.diag_indices(len(s))

  def eliminate(self, rhs):
    s, i = self.surface, self.interior
    interior = self.interior_lu.solve(rhs[i]) if len(i) else rhs[i]
    surface = rhs[s] - self.interior_to_surface @ interior
    return surface, interior

  def solve(self, surface_rhs, interior_rhs, coefficient, radiant_temperature):
    matrix = self.surface_matrix.copy()
    matrix[self.diagonal] += coefficient
    factor = linalg.cho_factor(matrix, check_finite=False)
    surface = linalg.cho_solve(
        factor, surface_rhs + coefficient * radiant_temperature,
        check_finite=False,
    )
    temperature = np.empty(len(self.surface) + len(self.interior))
    temperature[self.surface] = surface
    temperature[self.interior] = interior_rhs - self.response @ surface
    return temperature


def _project_radiation(geometry, active, groups, count):
  """Precompute group heat gains per unit incident solar flux and direction."""
  shortwave = []
  for area in geometry.face_areas:
    watts = geometry.absorptivity * area
    for glass, air in geometry.window_targets:
      watts.flat[air] += geometry.transmittance * area.flat[glass].sum() / air.size
    shortwave.append(np.bincount(
        groups, weights=watts.ravel()[active], minlength=count
    ))
  emission = np.bincount(
      groups, weights=geometry.emission.ravel()[active], minlength=count
  )
  # Solar absorptivity and transmission are already present in each
  # direction's projected coefficients.
  return building_radiation_utils.ExteriorRadiationGeometry(
      face_areas=np.stack(shortwave),
      azimuths=geometry.azimuths,
      absorptivity=np.ones(count),
      emission=emission,
      transmittance=0.0,
      window_targets=(),
      mass_mask=np.zeros(count, dtype=bool),
  )


class CondensedEnvelopeSimulator(ReducedEnvelopeSimulator):
  """Solve the same grouped model with nonlinear work at exterior surfaces."""

  @property
  def iterated_surface_nodes(self):
    return None if self._static is None else len(self._static['system'].surface)

  def _build_static(self, convection_coefficient):
    static = super()._build_static(convection_coefficient)
    groups, active = static['groups'], static['solved']
    count = static['matrix'].shape[0]
    if self.include_exterior_radiative_heat_transfer:
      geometry = self._build_exterior_radiation_geometry()
      static['radiation'] = _project_radiation(geometry, active, groups, count)
      emission = static['radiation'].emission
    else:
      emission = np.zeros(count)
    static['system'] = _SurfaceSystem(static['matrix'], emission)
    static['group_ambient'] = np.bincount(
        groups, weights=static['ambient_conductance']
    )
    outside = np.ones(self.building.temp.size, dtype=bool)
    outside[active] = False
    static['outside'] = outside
    self.factorization_count += int(static['system'].interior_lu is not None)
    return static

  def finite_differences_timestep(
      self, *, ambient_temperature, convection_coefficient
  ):
    if self._static is None or self._static_h != convection_coefficient:
      self._static = self._build_static(convection_coefficient)
      self._static_h = convection_coefficient
    return super().finite_differences_timestep(
        ambient_temperature=ambient_temperature,
        convection_coefficient=convection_coefficient,
    )

  def _prepare_exterior_radiation(self, ambient_temperature):
    geometry = self._static.get('radiation')
    if geometry is None:
      return None
    weather = self._weather_controller.get_exterior_radiation(
        self.current_timestamp, ambient_temperature
    )
    return geometry.forcing(weather)

  def _iterate_temperature_timestep(self, ambient_temperature, convection_coefficient):
    del convection_coefficient
    s = self._static
    system, active, groups = s['system'], s['solved'], s['groups']
    previous = self.building.temp.ravel()
    rhs = np.bincount(
        groups, weights=s['storage'] * previous[active]
        + self.building.input_q.ravel()[active],
    ) + s['group_ambient'] * ambient_temperature
    forcing = self._exterior_forcing
    if forcing is not None:
      rhs += forcing.shortwave_air_gain
    surface_rhs, interior_rhs = system.eliminate(rhs)
    if len(system.surface):
      estimate = np.bincount(groups, weights=previous[active]) / s['group_sizes']
      initial_ambient_delta = np.max(
          np.abs(previous[s['outside']] - ambient_temperature), initial=0.0
      )
      for step in range(self._iteration_limit):
        coefficient, _ = forcing.linearized_air_terms(estimate)
        temperature = system.solve(
            surface_rhs, interior_rhs, coefficient[system.surface],
            forcing.radiant_temperature,
        )
        if step == 0:
          delta = max(
              np.max(np.abs(temperature[groups] - previous[active])),
              initial_ambient_delta,
          )
        else:
          delta = np.max(np.abs(temperature - estimate))
        estimate = temperature
        if delta <= self._convergence_threshold:
          break
      drive = rhs + coefficient * forcing.radiant_temperature
    else:
      step, temperature, coefficient = 0, interior_rhs, 0.0
      delta = np.max(np.abs(temperature[groups] - previous[active]))
      drive = rhs
    residual = np.linalg.norm(s['matrix'] @ temperature + coefficient * temperature - drive)
    if not np.isfinite(residual) or residual > self._pcg_rtol * max(
        np.linalg.norm(drive), 1.0
    ):
      raise RuntimeError(f'Condensed heat balance did not converge ({residual:g})')
    expanded = np.full(previous.shape, ambient_temperature)
    expanded[active] = temperature[groups]
    self.building.temp = expanded.reshape(self.building.temp.shape)
    self.last_outer_iteration_count = step + 1
    self.last_outer_max_delta = float(delta)
    return not len(system.surface) or delta <= self._convergence_threshold
