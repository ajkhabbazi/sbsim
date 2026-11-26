"""Assemble finite-volume heat balances with optional grouped temperatures."""

# This assembler intentionally reuses the base simulators' thermal internals.
# pylint: disable=protected-access

from collections import defaultdict
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy import ndimage
from scipy import sparse

from smart_control.simulator import building as building_lib
from smart_control.simulator import building_radiation_utils
from smart_control.simulator import building_utils
from smart_control.simulator import cached_lu_simulator
from smart_control.simulator import constants


@dataclass(frozen=True)
class InteriorMass:
  """Effective thermal properties per square metre of floor."""

  capacity_per_m2: float = 30_000.0
  air_conductance_per_m2: float = 3.0
  exposed_area_per_m2: float = 1.0
  emissivity: float = 0.9

  def __post_init__(self):
    if not np.isfinite((
        self.capacity_per_m2,
        self.air_conductance_per_m2,
        self.exposed_area_per_m2,
        self.emissivity,
    )).all():
      raise ValueError('Interior mass properties must be finite')
    if self.capacity_per_m2 <= 0 or self.air_conductance_per_m2 < 0:
      raise ValueError(
          'Mass capacity must be positive and conductance nonnegative'
      )
    if self.exposed_area_per_m2 < 0 or not 0 <= self.emissivity <= 1:
      raise ValueError(
          'Exposed area must be nonnegative and emissivity in [0, 1]'
      )


@dataclass(frozen=True)
class HeatBalance:
  """Temperature mappings and SI coefficients of one implicit heat balance."""

  active: np.ndarray
  groups: np.ndarray
  storage: np.ndarray
  conduction: sparse.csr_matrix
  radiation: sparse.csr_matrix
  ambient: np.ndarray
  geometry: building_radiation_utils.ExteriorRadiationGeometry | None
  mass_active: np.ndarray
  mass_nodes: np.ndarray
  cv_storage: np.ndarray
  mass_cell_storage: np.ndarray
  radiation_diagonal: np.ndarray
  room_factors: sparse.csr_matrix


def _validate_block_size(block_cells):
  if block_cells is not None and (
      isinstance(block_cells, bool)
      or not isinstance(block_cells, Integral)
      or block_cells < 1
  ):
    raise ValueError('air block size must be a positive integer')


def _tile_cells(cells, block_cells):
  if block_cells is None:
    return (cells,)
  tiles = (cells - cells.min(axis=0)) // block_cells
  _, tile_index = np.unique(tiles, axis=0, return_inverse=True)
  order = np.argsort(tile_index, kind='stable')
  edges = np.cumsum(np.bincount(tile_index))[:-1]
  return np.split(cells[order], edges)


def _label_cells(cells):
  """Label four-connected components of a nonempty cell list."""
  local = cells - cells.min(axis=0)
  mask = np.zeros(tuple(local.max(axis=0) + 1), dtype=bool)
  mask[tuple(local.T)] = True
  labels, count = ndimage.label(mask)
  return labels[tuple(local.T)], count


def group_cells(solved, air, rooms, *, block_cells=None):
  """Return one group ID per active CV, merging only connected room air."""
  _validate_block_size(block_cells)
  active = np.zeros(air.shape, dtype=bool)
  active.ravel()[solved] = True
  active_air = active & air
  groups = np.full(air.shape, -1, dtype=np.int64)
  next_group = 0
  for cells in rooms:
    cells = cells[active_air[tuple(cells.T)]]
    if cells.size == 0:
      continue
    for tile in _tile_cells(cells, block_cells):
      labels, count = _label_cells(tile)
      groups[tuple(tile.T)] = next_group + labels - 1
      next_group += count
  positions = groups.ravel()[solved]
  individual = positions < 0
  positions[individual] = next_group + np.arange(np.count_nonzero(individual))
  return positions


def projection(positions):
  """Expand one temperature per group back into the active fine cells."""
  return sparse.csr_matrix(
      (np.ones(len(positions)), (np.arange(len(positions)), positions)),
      shape=(len(positions), int(positions.max()) + 1),
  )


def group_envelope_cells(
    solved, floor_plan, rooms, groups, *, wall_depth_cells=1
):
  """Join straight wall runs while retaining each layer across wall depth."""
  pos = np.full(floor_plan.shape, -1, dtype=int)
  pos.ravel()[solved] = np.arange(len(solved))
  contacts = np.full(floor_plan.shape, -1, dtype=int)
  contacts[floor_plan == constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT] = -2
  for index, cells in enumerate(rooms):
    air = (
        floor_plan[tuple(cells.T)]
        == constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT
    )
    contacts[tuple(cells[air].T)] = index
  solid = np.isin(
      floor_plan,
      (
          constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT,
          constants.FENESTRATION_VALUE_IN_FILE_INPUT,
      ),
  )
  directions = building_radiation_utils.FOUR_CONNECTED_DIRECTIONS
  faces = defaultdict(list)
  height, width = floor_plan.shape
  for row, col in np.argwhere(solid & (pos >= 0)):
    material = floor_plan[row, col]
    signature = []
    for side, (dr, dc) in enumerate(directions):
      for distance in range(1, wall_depth_cells + 1):
        r, c = row + dr * distance, col + dc * distance
        if not (0 <= r < height and 0 <= c < width):
          break
        if contacts[r, c] != -1:
          signature.append((side, contacts[r, c], distance))
          break
        if floor_plan[r, c] != material:
          break
    if len(signature) == 2 and signature[0][0] // 2 == signature[1][0] // 2:
      faces[(material, tuple(signature))].append((row, col))
  result = groups.copy()
  for cells in faces.values():
    coords = np.asarray(cells)
    labels, count = _label_cells(coords)
    positions = pos[tuple(coords.T)]
    for label in range(1, count + 1):
      members = positions[labels == label]
      result[members] = result[members[0]]
  return np.unique(result, return_inverse=True)[1]


def _room_enclosures(building, air):
  """Find connected physical air regions within each logical HVAC zone."""
  rooms = []
  for name, indices in building._room_dict.items():
    if not name.startswith(constants.ROOM_STRING_DESIGNATOR):
      continue
    cells = np.asarray(indices)
    cells = cells[air[tuple(cells.T)]]
    if cells.size:
      labels, count = _label_cells(cells)
      if count == 1:
        rooms.append(cells)
      else:
        rooms.extend(cells[labels == label] for label in range(1, count + 1))
  return rooms


def _exterior_geometry(sim, floor, active, group_at, size, solar_mass_at):
  """Project facade absorption and transmission to the solved CV groups."""
  building = sim.building
  glass = floor == constants.FENESTRATION_VALUE_IN_FILE_INPUT
  if glass.any() and not building.include_exterior_radiative_heat_transfer:
    raise ValueError(
        'Enable exterior geometry on the building when using glazing'
    )
  opaque = floor == constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT
  wall = getattr(building, 'exterior_wall_radiative_properties', None)
  if wall is None:
    wall = building_lib.DefaultExteriorWallRadiationProperties()
  glazing = getattr(building, 'fenestration_radiative_properties', wall)
  areas = np.zeros((4, *floor.shape))
  active_mask = np.zeros(floor.shape, dtype=bool)
  active_mask.ravel()[active] = True
  height = building.floor_height_cm / 100
  for r, c in np.argwhere(active_mask & (glass | opaque)):
    neighbors = building.neighbors[r][c]
    if len(neighbors) == 4:
      continue
    u, v = sim._get_cv_dimensions((r, c))
    for side, (dr, dc) in enumerate(
        building_radiation_utils.FOUR_CONNECTED_DIRECTIONS
    ):
      if (r + dr, c + dc) not in neighbors:
        areas[side, r, c] = height * (u if dr else v)
  exposed = areas.sum(axis=0)
  epsilon = np.where(glass, glazing.epsilon, wall.epsilon)
  alpha = np.where(glass, glazing.alpha, wall.alpha)
  if glass.any() and glazing.tau > 0.0:
    air = active_mask & (floor == constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT)
    windows = building_radiation_utils.exterior_window_targets(
        floor, air, exposed > 0
    )
  else:
    windows = ()
  if solar_mass_at is None:
    solar_targets = group_at.ravel()
  else:
    solar_targets = np.where(
        solar_mass_at >= 0, solar_mass_at, group_at
    ).ravel()
  groups = group_at.ravel()[active]
  projected_areas = []
  for area in areas:
    absorption = (alpha * area).ravel()[active]
    projected = np.bincount(groups, weights=absorption, minlength=size)
    for window, recipients in windows:
      np.add.at(
          projected,
          solar_targets[recipients],
          glazing.tau * area.flat[window].sum() / recipients.size,
      )
    projected_areas.append(projected)
  emission = (
      np.bincount(
          groups, weights=(exposed * epsilon).ravel()[active], minlength=size
      )
      * constants.STEFAN_BOLTZMANN_CONSTANT
  )
  orientation = getattr(building, 'floor_plan_orientation', 0.0)
  if not 0 <= orientation <= 360:
    raise ValueError('Floor plan orientation must be between 0 and 360')
  return building_radiation_utils.ExteriorRadiationGeometry(
      face_areas=np.stack(projected_areas),
      azimuths=tuple(
          (direction + orientation) % 360
          for direction in building_radiation_utils.EXTERIOR_FACE_AZIMUTHS
      ),
      absorptivity=np.ones(size),
      emission=emission,
      transmittance=0.0,
      window_targets=(),
      mass_mask=np.zeros(size, dtype=bool),
  )


def _mass_properties(sim, floor, active, native, config):
  """Return fine air indices, mass capacity, coupling and exposed gray area."""
  building = sim.building
  mass_positions = np.flatnonzero(
      floor.ravel()[active] == constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT
  )
  mass_active = active[mass_positions]
  cells = np.unravel_index(mass_active, floor.shape)
  area = np.array(
      [np.prod(sim._get_cv_dimensions((r, c))) for r, c in zip(*cells)],
      dtype=float,
  )
  if config is None and building.include_interior_mass:
    height = building.floor_height_cm / 100
    storage = (
        building.interior_mass_density[cells]
        * building.interior_mass_heat_capacity[cells]
        * area
        * height
        / sim._time_step_sec
    )
    conductance = native['mass_conductance'][mass_positions]
    epsilon = getattr(building, '_epsilon_interior_mass', None)
    if epsilon is None:
      epsilon = InteriorMass().emissivity
    else:
      epsilon = epsilon[cells]
    gray_area = epsilon * area
  else:
    if config is None:
      config = InteriorMass()
    storage = config.capacity_per_m2 * area / sim._time_step_sec
    conductance = config.air_conductance_per_m2 * area
    gray_area = (config.emissivity * config.exposed_area_per_m2) * area
  if np.any(~np.isfinite(storage)) or np.any(storage <= 0):
    raise ValueError('Interior mass cells require positive finite capacity')
  if np.any(~np.isfinite(conductance)) or np.any(conductance < 0):
    raise ValueError(
        'Interior mass cells require nonnegative finite conductance'
    )
  return mass_positions, mass_active, storage, conductance, gray_area


def _wall_emissivity(building, floor, solid):
  native = getattr(building, '_epsilon', None)
  if native is not None and native.shape == floor.shape:
    return native
  wall = getattr(building, 'exterior_wall_radiative_properties', None)
  if wall is None:
    wall = building_lib.DefaultExteriorWallRadiationProperties()
  glazing = getattr(building, 'fenestration_radiative_properties', wall)
  inside = building_lib.DefaultInsideWallRadiationProperties()
  outdoor = floor == constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT
  exposed = ndimage.binary_dilation(outdoor) & solid
  glass = floor == constants.FENESTRATION_VALUE_IN_FILE_INPUT
  return np.where(
      glass, glazing.epsilon, np.where(exposed, wall.epsilon, inside.epsilon)
  )


def _interior_radiation(
    building, floor, rooms, group_at, mass_at, mass_area_at, size
):
  """Build one gray-area rank-one irradiance factor per room enclosure."""
  solid = np.isin(
      floor,
      (
          constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT,
          constants.FENESTRATION_VALUE_IN_FILE_INPUT,
      ),
  )
  weight_at = (
      _wall_emissivity(building, floor, solid)
      * building.cv_size_cm
      / 100
      * building.floor_height_cm
      / 100
  )
  sigma = constants.STEFAN_BOLTZMANN_CONSTANT
  diagonal = np.zeros(size)
  row_indices, columns, values = [], [], []
  for room_index, cells in enumerate(rooms):
    room = tuple(cells.T)
    faces = np.array(
        [
            (nr, nc)
            for r, c in cells
            for nr, nc in building.neighbors[r][c]
            if solid[nr, nc] and group_at[nr, nc] >= 0
        ],
        dtype=int,
    ).reshape(-1, 2)
    face_coords = tuple(faces.T)
    room_mass = mass_at[room] >= 0
    nodes = np.concatenate((group_at[face_coords], mass_at[room][room_mass]))
    weights = np.concatenate(
        (weight_at[face_coords], mass_area_at[room][room_mass])
    )
    keep = weights > 0
    if not keep.any():
      continue
    nodes, inverse = np.unique(nodes[keep], return_inverse=True)
    weights = np.bincount(inverse, weights=weights[keep])
    diagonal[nodes] += sigma * weights
    row_indices.extend([room_index] * len(nodes))
    columns.extend(nodes)
    values.extend(np.sqrt(sigma / weights.sum()) * weights)
  factors = sparse.csr_matrix(
      (values, (row_indices, columns)), shape=(len(rooms), size)
  )
  return diagonal, factors


def assemble(
    sim,
    convection_coefficient,
    *,
    grouped,
    wall_depth_cells,
    air_block_cells,
    include_interior_radiative_heat_transfer,
    include_exterior_radiative_heat_transfer,
    include_interior_mass,
    interior_mass,
    solar_to_interior_mass=True,
):
  """Assemble the same conductance and gray area-ratio radiation on either grid.

  Effective mass properties are per floor area. Without an explicit
  configuration, native mass uses the building's volumetric capacity and
  exchange coefficient, plus its emissivity when available and unit exposed
  area per floor area.
  Glazing transmits shortwave power to interior mass nodes when interior mass
  is enabled (by default), otherwise to the connected air groups.
  """
  building = sim.building
  floor = building_utils.guarantee_air_padding_in_frame(building.floor_plan)
  if floor.shape != building.temp.shape:
    raise ValueError('Floor plan and temperature grid have different shapes')
  if (
      building.include_interior_mass
      and building.interior_mass_mask.shape != floor.shape
  ):
    raise ValueError(
        'The native interior mass grid must match the padded floor plan'
    )
  native = cached_lu_simulator.CachedLUSimulator._build_static(
      sim, convection_coefficient
  )
  active = native['solved']
  if not active.size:
    raise ValueError('Floor plan has no active control volumes')
  air = floor == constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT
  rooms = _room_enclosures(building, air)
  if grouped:
    groups = group_cells(active, air, rooms, block_cells=air_block_cells)
    groups = group_envelope_cells(
        active, floor, rooms, groups, wall_depth_cells=wall_depth_cells
    )
    mapping = projection(groups)
  else:
    groups = np.arange(len(active))
    mapping = None
  cv_storage = native['storage']
  core = native['a_cond']
  if building.include_interior_mass:
    core = core - sparse.diags(native['mass_conductance'])
  if mapping is not None:
    core = (mapping.T @ core @ mapping).tocsr()
  node_count = core.shape[0]
  storage = np.bincount(groups, weights=cv_storage, minlength=node_count)
  ambient = np.bincount(
      groups, weights=native['boundary_conductance'], minlength=node_count
  )
  group_at = np.full(floor.shape, -1, dtype=int)
  group_at.ravel()[active] = groups
  mass_at = np.full(floor.shape, -1, dtype=int)
  mass_area_at = np.zeros(floor.shape)
  mass_active = np.empty(0, dtype=int)
  mass_nodes = np.empty(0, dtype=int)
  mass_cell_storage = np.empty(0)

  if include_interior_mass:
    mass_positions, mass_active, mass_cell_storage, conductance, gray_area = (
        _mass_properties(sim, floor, active, native, interior_mass)
    )
    mass_air = groups[mass_positions]
    _, mass_group = np.unique(mass_air, return_inverse=True)
    mass_count = np.max(mass_group, initial=-1) + 1
    mass_nodes = node_count + mass_group
    mass_at.ravel()[mass_active] = mass_nodes
    mass_area_at.ravel()[mass_active] = gray_area
    mass_storage = np.bincount(
        mass_group, weights=mass_cell_storage, minlength=mass_count
    )
    if mass_count:
      links = sparse.csr_matrix(
          (-conductance, (mass_air, mass_group)), shape=(node_count, mass_count)
      )
      air_exchange = np.bincount(
          mass_air, weights=conductance, minlength=node_count
      )
      mass_exchange = np.bincount(
          mass_group, weights=conductance, minlength=mass_count
      )
      core = sparse.bmat(
          (
              (core + sparse.diags(air_exchange), links),
              (links.T, sparse.diags(mass_storage + mass_exchange)),
          ),
          format='csr',
      )
    storage = np.concatenate((storage, mass_storage))
    ambient = np.pad(ambient, (0, mass_count))

  size = storage.size
  if include_interior_radiative_heat_transfer:
    radiative_diag, room_factors = _interior_radiation(
        building, floor, rooms, group_at, mass_at, mass_area_at, size
    )
  else:
    radiative_diag = np.zeros(size)
    room_factors = sparse.csr_matrix((0, size))
  geometry = None
  if include_exterior_radiative_heat_transfer:
    solar_mass_at = (
        mass_at if include_interior_mass and solar_to_interior_mass else None
    )
    geometry = _exterior_geometry(
        sim, floor, active, group_at, size, solar_mass_at
    )
    radiative_diag += geometry.emission
  radiation = (
      sparse.diags(radiative_diag) - room_factors.T @ room_factors
  ).tocsr()
  radiation.eliminate_zeros()
  return HeatBalance(
      active,
      groups,
      storage,
      core.tocsr(),
      radiation,
      ambient,
      geometry,
      mass_active,
      mass_nodes,
      cv_storage,
      mass_cell_storage,
      radiative_diag,
      room_factors,
  )
