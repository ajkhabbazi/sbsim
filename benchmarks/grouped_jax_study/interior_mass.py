"""Implicit grouped air, envelope, and interior mass heat balance."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .matched_vav import coupled_radiation
from .matched_vav import settings
from smart_control.simulator import building_radiation_utils
from smart_control.simulator import constants
from smart_control.simulator.grouped_heat_balance import InteriorMass as InteriorMass

SLOPE = coupled_radiation.SLOPE
SIGMA = coupled_radiation.SIGMA


@dataclass(frozen=True)
class HeatBalance:
  conduction: sparse.csr_matrix
  radiation: sparse.csr_matrix
  storage: np.ndarray
  geometry: building_radiation_utils.ExteriorRadiationGeometry


def enclosure_operator(weights):
  """Map surface T**4 to watts leaving each diffuse-gray enclosure surface.

  For area-ratio F_ij = A_j / sum(A), all surfaces see a common irradiation.
  Eliminating radiosities gives a symmetric, energy-conserving operator even
  when the active surfaces have different emissivities.
  """
  weights = np.asarray(weights, dtype=np.float64)
  total = weights.sum()
  if total == 0:
    return np.zeros((len(weights), len(weights)))
  return SIGMA * (np.diag(weights) - np.outer(weights, weights) / total)


def assemble(
    building, static, air_nodes, config, *, solar_to_interior_mass=True
):
  """Append one mass storage state per room and assemble room longwave."""
  count = static['conduction'].shape[0]
  n_rooms = len(air_nodes)
  size = count + n_rooms
  mass_nodes = np.arange(count, size)
  cell_side = building.cv_size_cm / 100
  cell_face_area = cell_side * building.floor_height_cm / 100
  floor_area = building._zone_sizes * cell_side**2
  mass_storage = config.capacity_per_m2 * floor_area / settings.STEP_SECONDS
  g = config.air_conductance_per_m2 * floor_area
  air_storage = np.bincount(
      static['groups'], weights=static['storage'], minlength=count
  )
  air_diagonal = sparse.diags(np.bincount(air_nodes, weights=g, minlength=count))
  links = sparse.csr_matrix((-g, (air_nodes, np.arange(n_rooms))), (count, n_rooms))
  conduction = sparse.bmat((
      (static['conduction'] + air_diagonal, links),
      (links.T, sparse.diags(mass_storage + g)),
  ), format='csr')

  # Count each room-facing wall/glass face. A shared wall has two faces in
  # different room enclosures but retains the same thermal wall node.
  floor = building.floor_plan
  cols = floor.shape[1]
  solid = np.isin(floor, (
      constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT,
      constants.FENESTRATION_VALUE_IN_FILE_INPUT,
  )).ravel()
  group_at = np.full(floor.size, -1, dtype=int)
  group_at[static['active']] = static['groups']
  wall_flats = np.flatnonzero(solid & (group_at >= 0))
  emissivity = np.zeros(floor.size)
  emissivity[wall_flats] = coupled_radiation._emissivity(building, wall_flats)
  rad_rows, rad_cols, rad_values = [], [], []
  for zone, (cells, mass) in enumerate(zip(
      (building._room_dict[name] for name in building._zone_names), mass_nodes
  )):
    faces = np.array([
        nr * cols + nc
        for row, col in cells
        for nr, nc in building.neighbors[row][col]
        if solid[nr * cols + nc] and group_at[nr * cols + nc] >= 0
    ], dtype=int)
    wall_groups, inverse = np.unique(group_at[faces], return_inverse=True)
    wall_weights = np.bincount(inverse, weights=emissivity[faces] * cell_face_area)
    mass_weight = config.emissivity * config.exposed_area_per_m2 * floor_area[zone]
    nodes = np.append(wall_groups, mass)
    exchange = enclosure_operator(np.append(wall_weights, mass_weight))
    rad_rows.extend(np.repeat(nodes, len(nodes)))
    rad_cols.extend(np.tile(nodes, len(nodes)))
    rad_values.extend(exchange.ravel())

  radiation = sparse.csr_matrix((rad_values, (rad_rows, rad_cols)), shape=(size, size))
  radiation += sparse.diags(np.pad(static['geometry'].emission, (0, n_rooms)))
  storage = np.concatenate((air_storage, mass_storage))
  base_geometry = static['geometry']
  face_areas = np.pad(base_geometry.face_areas, ((0, 0), (0, n_rooms)))
  if solar_to_interior_mass:
    face_areas[:, mass_nodes] = face_areas[:, air_nodes]
    face_areas[:, air_nodes] = 0.0
  geometry = building_radiation_utils.ExteriorRadiationGeometry(
      face_areas=face_areas,
      azimuths=base_geometry.azimuths,
      absorptivity=np.pad(
          base_geometry.absorptivity, (0, n_rooms), constant_values=1.0
      ),
      emission=np.pad(base_geometry.emission, (0, n_rooms)),
      transmittance=0.0,
      window_targets=(),
      mass_mask=np.zeros(size, dtype=bool),
  )
  return HeatBalance(conduction, radiation, storage, geometry)
