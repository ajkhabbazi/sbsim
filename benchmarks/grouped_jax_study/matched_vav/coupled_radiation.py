"""Sparse SBSim heat balance with both interior and exterior longwave."""

from functools import lru_cache

import numpy as np
from scipy import ndimage
from scipy import sparse
from scipy.sparse import linalg

from smart_control.simulator import building as building_lib
from smart_control.simulator import building_radiation_utils as radiation
from smart_control.simulator import building_utils
from smart_control.simulator import cached_lu_simulator
from smart_control.simulator import constants

from . import reduced_air
from . import reduced_envelope

REFERENCE_K = 295.65
SLOPE = 4 * REFERENCE_K**3
SIGMA = constants.STEFAN_BOLTZMANN_CONSTANT


@lru_cache(maxsize=2)
def _interior_view_factors(shape, labels):
  floor = np.frombuffer(labels, dtype=np.int8).reshape(shape)
  indexed = floor.copy()
  exterior = floor == constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT
  solid = np.isin(floor, (
      constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT,
      constants.FENESTRATION_VALUE_IN_FILE_INPUT,
  ))
  indexed[exterior] = constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION
  indexed[solid] = constants.INTERIOR_WALL_VALUE_IN_FUNCTION
  walls = radiation.mark_interior_wall_adjacent_to_air(indexed)
  # A wall's native AreaRatio enclosure is the union of boundary walls for
  # its adjacent four-connected air regions, including both sides if shared.
  air_labels, _ = ndimage.label(indexed == constants.INTERIOR_SPACE_VALUE_IN_FUNCTION)
  padded = np.pad(air_labels, 1)
  row, col = np.nonzero(walls)
  adjacent = np.stack([
      padded[row + 1 + dr, col + 1 + dc]
      for dr, dc in radiation.FOUR_CONNECTED_DIRECTIONS
  ], axis=1)
  regions, inverse = np.unique(adjacent, axis=0, return_inverse=True)
  labels_by_region = [set(region[region != 0]) for region in regions]
  wall_by_label = {
      label: np.flatnonzero(np.any(adjacent == label, axis=1))
      for label in np.unique(adjacent) if label
  }
  direct = np.zeros((row.size, row.size))
  for i, labels in enumerate(labels_by_region):
    connected = np.unique(np.concatenate([wall_by_label[label] for label in labels]))
    if connected.size > 1:
      wall_rows = np.flatnonzero(inverse == i)
      direct[np.ix_(wall_rows, connected)] = 1.0 / (connected.size - 1)
  np.fill_diagonal(direct, 0.0)
  return np.flatnonzero(walls), sparse.csr_matrix(radiation.fix_view_factors(direct))


def _emissivity(building, walls):
  floor = building.floor_plan
  glass = floor == constants.FENESTRATION_VALUE_IN_FILE_INPUT
  solid = np.where(glass, constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT, floor)
  _, exterior, _, _ = building_utils.construct_building_data_types(
      floor_plan=solid, zone_map=building._zone_map
  )
  exterior_eps = building.exterior_wall_radiative_properties.epsilon
  inside_eps = building_lib.DefaultInsideWallRadiationProperties().epsilon
  epsilon = np.where(
      exterior == constants.EXTERIOR_WALL_VALUE_IN_FUNCTION, exterior_eps,
      inside_eps,
  )
  epsilon[glass] = building.fenestration_radiative_properties.epsilon
  return epsilon.ravel()[walls]


def _radiosity_operators(view, epsilon):
  """Return the native SBSim radiosity equations and flux normalization."""
  eye = sparse.eye(len(epsilon), format='csr')
  radiosity = sparse.diags(1 / epsilon) @ (
      eye - sparse.diags(1 - epsilon) @ view
  )
  outward = eye - view
  # The native net-flux helper treats the diagonal as minus the sum of
  # off-diagonal terms. This preserves that behavior even with imperfect
  # view-factor row closure.
  uniform = linalg.splu(radiosity.tocsc()).solve(np.ones(len(epsilon)))
  return radiosity.tocsr(), outward, outward @ uniform


class CoupledRadiationSimulator(cached_lu_simulator.CachedLUSimulator):
  """Full or grouped heat balance with sparse radiosity variables."""

  nonlinear = True
  merge_envelope = False
  eliminate_radiosity = False
  has_interior_longwave = True

  def __init__(self, *, building, air_block_cells=None, **kwargs):
    if building.include_interior_mass or building.include_radiative_heat_transfer:
      raise ValueError('This solver constructs its own wall radiosity equations')
    if not building.include_exterior_radiative_heat_transfer:
      raise ValueError('This solver requires exterior radiation geometry')
    if building._convection_simulator is not None:
      raise ValueError('This solver uses finite-volume face heat exchange')
    reduced_air._validate_block_size(air_block_cells)
    self._air_block_cells = air_block_cells
    super().__init__(building=building, **kwargs)

  @property
  def active_heat_nodes(self):
    return self._static['conduction'].shape[0]

  @property
  def radiosity_nodes(self):
    return self._static['radiosity'].shape[0]

  @property
  def radiosity_unknowns_per_solve(self):
    return 0 if self.eliminate_radiosity else self.radiosity_nodes

  def _build_grouped_conduction(self, convection_coefficient, *, wall_depth_cells=1):
    """Project the finite-volume balance and external forcing onto groups."""
    native = super()._build_static(convection_coefficient)
    floor = self.building.floor_plan
    active = native['solved']
    air = floor == constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT
    rooms = [
        np.asarray(cells)
        for name, cells in self.building._room_dict.items()
        if name.startswith(constants.ROOM_STRING_DESIGNATOR)
    ]
    groups = reduced_air.group_cells(
        active, air, rooms, block_cells=self._air_block_cells
    )
    if self.merge_envelope:
      groups = reduced_envelope.group_envelope_cells(
          active, floor, rooms, groups, wall_depth_cells=wall_depth_cells
      )
    projector = reduced_air.projection(groups)
    conduction = (projector.T @ native['a_cond'] @ projector).tocsr()
    count = conduction.shape[0]
    geometry = reduced_envelope._project_radiation(
        self._build_exterior_radiation_geometry(), active, groups, count
    )
    return dict(
        active=active, groups=groups, sizes=np.bincount(groups), geometry=geometry,
        ambient=np.bincount(groups, weights=native['boundary_conductance']),
        storage=native['storage'], conduction=conduction,
        position=native['position'], radiation_area=native['radiation_area'],
    )

  def _build_static(self, convection_coefficient):
    static = self._build_grouped_conduction(convection_coefficient)
    groups, geometry = static['groups'], static['geometry']
    conduction = static['conduction']
    count = conduction.shape[0]
    floor = self.building.floor_plan
    walls, view = _interior_view_factors(floor.shape, floor.astype(np.int8).tobytes())
    positions = static['position'][walls]
    if np.any(positions < 0):
      raise ValueError('Expected every interior radiating CV to be active')
    wall_groups = groups[positions]
    picker = sparse.csr_matrix(
        (np.ones(len(walls)), (np.arange(len(walls)), wall_groups)),
        shape=(len(walls), count),
    )
    radiosity, outward, row_correction = _radiosity_operators(
        view, _emissivity(self.building, walls)
    )
    area = static['radiation_area']
    coupling = (area * (picker.T @ outward)).tocsr()
    local_emission = geometry.emission - area * SIGMA * np.bincount(
        wall_groups, weights=row_correction, minlength=count
    )
    static.update(
        wall_groups=wall_groups, coupling=coupling, radiosity=radiosity,
        local_emission=local_emission,
    )
    if self.eliminate_radiosity:
      radiating = np.unique(wall_groups)
      response = linalg.splu(radiosity.tocsc()).solve(picker[:, radiating].toarray())
      projected = (SIGMA * (coupling[radiating] @ response))
      rows, cols = np.meshgrid(radiating, radiating, indexing='ij')
      operator = sparse.diags(local_emission, format='csr') + sparse.csr_matrix(
          (projected.ravel(), (rows.ravel(), cols.ravel())), shape=(count, count)
      )
      static['operator'] = operator
      matrix = conduction + SLOPE * operator
    else:
      matrix = sparse.bmat((
          (conduction + sparse.diags(SLOPE * local_emission), coupling),
          (-SLOPE * SIGMA * picker, radiosity),
      ), format='csr')
    static['factor'] = linalg.splu(matrix.tocsc(), permc_spec='MMD_AT_PLUS_A')
    if not self.nonlinear:
      static['linear_matrix'] = matrix
    return static

  def finite_differences_timestep(
      self, *, ambient_temperature, convection_coefficient
  ):
    if self._static is None or self._static_h != convection_coefficient:
      self._static = self._build_static(convection_coefficient)
      self._static_h = convection_coefficient
      self.factorization_count += 1
    s = self._static
    active, groups = s['active'], s['groups']
    prior_grid = np.asarray(self.building.temp, dtype=np.float64).ravel()
    previous_active = prior_grid[active]
    previous = np.bincount(groups, weights=previous_active) / s['sizes']
    weather = self._weather_controller.get_exterior_radiation(
        self.current_timestamp, ambient_temperature
    )
    forcing = s['geometry'].forcing(weather)
    incoming = (
        np.bincount(
            groups, weights=s['storage'] * previous_active
            + self.building.input_q.ravel()[active],
        ) + s['ambient'] * ambient_temperature
        + forcing.shortwave_air_gain
        + forcing.emission * forcing.radiant_temperature**4
    )
    shape = len(previous)
    correction = previous**4 - SLOPE * previous
    if self.eliminate_radiosity:
      rhs = incoming - s['operator'] @ correction
    else:
      rhs = np.concatenate((
          incoming - s['local_emission'] * correction,
          SIGMA * correction[s['wall_groups']],
      ))
    solution = s['factor'].solve(rhs)
    solves = 1
    if self.nonlinear:
      tolerance = self._pcg_rtol * max(np.linalg.norm(incoming), 1.0)
      for _ in range(self._iteration_limit):
        temperature, surface = solution[:shape], solution[shape:]
        t4 = temperature**4
        residual = np.concatenate((
            incoming - s['conduction'] @ temperature - s['coupling'] @ surface
            - s['local_emission'] * t4,
            SIGMA * t4[s['wall_groups']] - s['radiosity'] @ surface,
        ))
        if np.linalg.norm(residual) <= tolerance:
          break
        solution += s['factor'].solve(residual)
        solves += 1
      else:
        raise RuntimeError(f'Coupled longwave solve did not converge at {self.current_timestamp}')
    else:
      residual = np.linalg.norm(s['linear_matrix'] @ solution - rhs)
      if residual > self._pcg_rtol * max(np.linalg.norm(rhs), 1.0):
        raise RuntimeError(f'Linear coupled radiation residual {residual:g}')
    if not np.isfinite(solution).all():
      raise RuntimeError('Nonfinite coupled heat-balance temperature')
    expanded = np.full(prior_grid.shape, ambient_temperature)
    expanded[active] = solution[:shape][groups]
    self.last_outer_max_delta = float(np.max(np.abs(expanded - prior_grid)))
    self.building.temp = expanded.reshape(self.building.temp.shape)
    self.last_outer_iteration_count = solves
    self.triangular_solve_count += solves
    self.last_heat_balance_converged = True
    return True


class LinearCoupledRadiationSimulator(CoupledRadiationSimulator):
  nonlinear = False


class LinearCoupledEnvelopeSimulator(LinearCoupledRadiationSimulator):
  merge_envelope = True
  eliminate_radiosity = True
