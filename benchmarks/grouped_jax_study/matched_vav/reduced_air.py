"""CPU heat-balance prototype with fine walls and grouped interior air."""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from smart_control.simulator import building as building_lib
from smart_control.simulator import building_utils
from smart_control.simulator import cached_lu_simulator
from smart_control.simulator import constants
from smart_control.simulator import grouped_heat_balance


# The core solver owns the geometry mapping. Keep existing study imports.
_validate_block_size = grouped_heat_balance._validate_block_size
group_cells = grouped_heat_balance.group_cells
projection = grouped_heat_balance.projection


class TimestepMatrix:
  """Reuse a CSR matrix whose radiation coefficients affect its diagonal."""

  def __init__(self, conduction):
    self._conduction = conduction
    self._matrix = conduction.copy()
    rows = np.repeat(np.arange(conduction.shape[0]), np.diff(conduction.indptr))
    self._slots = np.flatnonzero(rows == conduction.indices)
    self._diagonal = conduction.data[self._slots]

  def with_radiation(self, diagonal):
    if len(self._slots) != self._matrix.shape[0]:
      return (self._conduction + sparse.diags(diagonal)).tocsr()
    self._matrix.data[self._slots] = self._diagonal + diagonal
    return self._matrix

  def with_surface_radiation(self, surface, radial):
    if len(self._slots) != self._matrix.shape[0]:
      diagonal = np.zeros(self._matrix.shape[0])
      diagonal[surface] = radial
      return self.with_radiation(diagonal)
    self._matrix.data[self._slots[surface]] = self._diagonal[surface] + radial
    return self._matrix


class VectorizedBuilding(building_lib.FloorPlanBasedBuilding):
  """Use flat CV indices for repeated HVAC operations on any room shape."""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    names = [
        name
        for name in self._room_dict
        if name.startswith(constants.ROOM_STRING_DESIGNATOR)
    ]
    cells = [
        np.ravel_multi_index(
            tuple(np.asarray(self._room_dict[name]).T), self.temp.shape
        )
        for name in names
    ]
    self._zone_names = names
    self._zone_cells = np.concatenate(cells)
    self._zone_sizes = np.array([len(zone) for zone in cells])
    self._zone_codes = np.repeat(np.arange(len(names)), self._zone_sizes)
    weights = self.diffusers.ravel()
    self._diffuser_indices = {
        name: group[weights[group] > 0.0] for name, group in zip(names, cells)
    }

  def get_zone_average_temps(self):
    total = np.bincount(
        self._zone_codes,
        weights=self.temp.ravel()[self._zone_cells],
        minlength=len(self._zone_names),
    )
    return dict(zip(self._zone_names, total / self._zone_sizes))

  def apply_thermal_power_zone(self, zone_name, power):
    if zone_name not in self._diffuser_indices:
      return super().apply_thermal_power_zone(zone_name, power)
    cells = self._diffuser_indices[zone_name]
    self.input_q.ravel()[cells] = power * self.diffusers.ravel()[cells]


class ReducedAirSimulator(cached_lu_simulator.CachedLUSimulator):
  """Solve the fine heat balance under an equal-temperature constraint per group."""

  def __init__(self, *, building, air_block_cells=None, **kwargs):
    if (
        building.include_interior_mass
        or building.include_radiative_heat_transfer
    ):
      raise ValueError('Reduced air prototype supports exterior radiation only')
    if building._convection_simulator is not None:
      raise ValueError(
          'Reduced air prototype uses finite-volume face heat exchange'
      )
    _validate_block_size(air_block_cells)
    self._air_block_cells = air_block_cells
    super().__init__(building=building, **kwargs)

  @property
  def active_heat_nodes(self):
    if self._static is None:
      return None
    return self._static['matrix'].shape[0]

  def _group_positions(self, solved, floor_plan, rooms):
    air = floor_plan == constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT
    return group_cells(solved, air, rooms, block_cells=self._air_block_cells)

  def _build_static(self, convection_coefficient):
    full = super()._build_static(convection_coefficient)
    floor_plan = building_utils.guarantee_air_padding_in_frame(
        self.building.floor_plan
    )
    if floor_plan.shape != self.building.temp.shape:
      raise ValueError(
          'Floor plan and SBSim temperature grid have different shapes'
      )
    rooms = [
        np.asarray(cells)
        for name, cells in self.building._room_dict.items()
        if name.startswith(constants.ROOM_STRING_DESIGNATOR)
    ]
    groups = self._group_positions(full['solved'], floor_plan, rooms)
    mapping = projection(groups)
    matrix = (mapping.T @ full['a_cond'] @ mapping).tocsr()
    return dict(
        solved=full['solved'],
        storage=full['storage'],
        ambient_conductance=full['boundary_conductance'],
        matrix=matrix,
        timestep_matrix=TimestepMatrix(matrix),
        groups=groups,
        group_sizes=np.bincount(groups),
    )

  def update_temperature_estimates(
      self, temperature_estimates, ambient_temperature, convection_coefficient
  ):
    if self._static is None or self._static_h != convection_coefficient:
      self._static = self._build_static(convection_coefficient)
      self._static_h = convection_coefficient
      self._lu = None
    s = self._static
    active = s['solved']
    groups = s['groups']
    estimate = temperature_estimates.ravel()
    rhs_fine = (
        s['storage'] * self.building.temp.ravel()[active]
        + self.building.input_q.ravel()[active]
        + s['ambient_conductance'] * ambient_temperature
    )
    matrix = s['matrix']
    if self._exterior_forcing is not None:
      coefficient, driving = self._exterior_forcing.linearized_air_terms(
          temperature_estimates
      )
      rhs_fine += driving.ravel()[active]
      diagonal = np.bincount(groups, weights=coefficient.ravel()[active])
      matrix = s['timestep_matrix'].with_radiation(diagonal)
    rhs = np.bincount(groups, weights=rhs_fine)
    solution = np.bincount(groups, weights=estimate[active]) / s['group_sizes']
    if self._lu is None or self._stale:
      self._lu = sparse_linalg.splu(matrix.tocsc(), permc_spec='MMD_AT_PLUS_A')
      self.factorization_count += 1
      self._stale = False
    solves = self._pcg(matrix, rhs, solution)
    self.triangular_solve_count += solves
    self._stale = solves > self._refactor_after
    residual = np.linalg.norm(rhs - matrix @ solution)
    if not np.isfinite(residual) or residual > self._pcg_rtol * max(
        np.linalg.norm(rhs), 1.0
    ):
      raise RuntimeError(
          f'Reduced heat balance did not converge (residual {residual:g})'
      )
    expanded = np.full(estimate.size, ambient_temperature)
    expanded[active] = solution[groups]
    return expanded.reshape(temperature_estimates.shape), float(
        np.max(np.abs(expanded - estimate))
    )
