"""Build one floor plan in both metric coordinates and SBSim cells."""

from dataclasses import dataclass
import math

import numpy as np

from smart_control.simulator import constants

from .settings import WINDOW_HEIGHT_FRACTION


@dataclass(frozen=True)
class PlanSpec:
  zones: int
  columns: int | None = None
  room_width_m: float = 5.0
  room_depth_m: float = 4.0
  height_m: float = 3.0
  wall_thickness_m: float = 0.5
  cv_size_m: float = 0.5
  window_width_m: float = 0.0
  exterior_wall_thickness_m: float | None = None


@dataclass(frozen=True)
class Window:
  offset_m: float  # From the first endpoint of the counterclockwise wall.
  width_m: float
  sill_m: float
  head_m: float


@dataclass(frozen=True)
class Wall:
  side: str
  start_m: tuple[float, float]
  end_m: tuple[float, float]
  neighbor: str | None
  opposite_side: str
  window: Window | None


@dataclass(frozen=True)
class Room:
  name: str
  bounds_m: tuple[float, float, float, float]  # west, south, east, north
  area_m2: float
  volume_m3: float
  walls: tuple[Wall, ...]


@dataclass(frozen=True)
class FloorPlan:
  spec: PlanSpec
  rooms: tuple[Room, ...]
  grid: np.ndarray
  zone_map: np.ndarray


def _cells(value: float, size: float, name: str) -> int:
  if not math.isfinite(value) or value <= 0:
    raise ValueError(f'{name} must be positive and finite')
  count = round(value / size)
  if count < 1 or not math.isclose(value, count * size, abs_tol=1e-9):
    raise ValueError(f'{name} must be a positive multiple of cv_size_m')
  return count


def make_plan(spec: PlanSpec) -> FloorPlan:
  """Lay out rooms north to south, with physical wall strips between them."""
  if not isinstance(spec.zones, int) or spec.zones < 1:
    raise ValueError('zones must be a positive integer')
  cols = (
      math.ceil(math.sqrt(spec.zones)) if spec.columns is None else spec.columns
  )
  if not isinstance(cols, int) or not 1 <= cols <= spec.zones:
    raise ValueError('columns must be an integer between 1 and zones')
  for name in ('cv_size_m', 'height_m'):
    value = getattr(spec, name)
    if not math.isfinite(value) or value <= 0:
      raise ValueError(f'{name} must be positive and finite')
  cv = spec.cv_size_m
  thickness = _cells(spec.wall_thickness_m, cv, 'wall_thickness_m')
  ext_thickness = (
      _cells(
          spec.exterior_wall_thickness_m, cv, 'exterior_wall_thickness_m'
      )
      if spec.exterior_wall_thickness_m is not None
      else thickness
  )
  if ext_thickness < thickness:
    raise ValueError('exterior_wall_thickness_m must be >= wall_thickness_m')
  width = _cells(spec.room_width_m, cv, 'room_width_m')
  depth = _cells(spec.room_depth_m, cv, 'room_depth_m')
  window = 0
  if spec.window_width_m != 0:
    window = _cells(spec.window_width_m, cv, 'window_width_m')
    if any(n - window < 2 or (n - window) % 2 for n in (width, depth)):
      raise ValueError(
          'window_width_m needs centered grid alignment and at least one '
          'opaque cell at each end on both wall lengths'
      )

  rows = math.ceil(spec.zones / cols)
  step_x, step_y = width + thickness, depth + thickness
  grid = np.full(
      (
          rows * step_y - thickness + 2 * ext_thickness + 2,
          cols * step_x - thickness + 2 * ext_thickness + 2,
      ),
      constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT,
      dtype=np.uint8,
  )
  zone_map = grid.copy()
  names = {divmod(i, cols): f'room_{i + 1}' for i in range(spec.zones)}
  rooms = []
  # SBSim row zero faces north. EnergyPlus uses y increasing northward.
  for (row, col), name in names.items():
    r0, c0 = 1 + ext_thickness + row * step_y, 1 + ext_thickness + col * step_x
    r1, c1 = r0 + depth, c0 + width
    locations = ((row + 1, col), (row, col + 1), (row - 1, col), (row, col - 1))
    sides = ('south', 'east', 'north', 'west')
    pad_s = thickness if locations[0] in names else ext_thickness
    pad_e = thickness if locations[1] in names else ext_thickness
    pad_n = thickness if locations[2] in names else ext_thickness
    pad_w = thickness if locations[3] in names else ext_thickness
    shell = np.s_[
        r0 - pad_n : r1 + pad_s, c0 - pad_w : c1 + pad_e
    ]
    air = np.s_[r0:r1, c0:c1]
    for array in (grid, zone_map):
      array[shell] = constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT
      array[air] = constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT

    west, east = (c0 - 1) * cv, (c1 - 1) * cv
    south, north = (grid.shape[0] - 1 - r1) * cv, (grid.shape[0] - 1 - r0) * cv
    corners = ((west, south), (east, south), (east, north), (west, north))
    dx, dy = (width - window) // 2, (depth - window) // 2
    glazing = (
        np.s_[r1 : r1 + ext_thickness, c0 + dx : c0 + dx + window],
        np.s_[r0 + dy : r0 + dy + window, c1 : c1 + ext_thickness],
        np.s_[r0 - ext_thickness : r0, c0 + dx : c0 + dx + window],
        np.s_[r0 + dy : r0 + dy + window, c0 - ext_thickness : c0],
    )
    walls = []
    for i, side in enumerate(sides):
      neighbor = names.get(locations[i])
      opening = None
      if window and neighbor is None:
        grid[glazing[i]] = constants.FENESTRATION_VALUE_IN_FILE_INPUT
        opening = Window(
            (dx if i % 2 == 0 else dy) * cv,
            spec.window_width_m,
            spec.height_m * (1 - WINDOW_HEIGHT_FRACTION) / 2,
            spec.height_m * (1 + WINDOW_HEIGHT_FRACTION) / 2,
        )
      walls.append(
          Wall(
              side,
              corners[i],
              corners[(i + 1) % 4],
              neighbor,
              sides[(i + 2) % 4],
              opening,
          )
      )
    area = spec.room_width_m * spec.room_depth_m
    rooms.append(
        Room(
            name,
            (west, south, east, north),
            area,
            area * spec.height_m,
            tuple(walls),
        )
    )

  return FloorPlan(spec, tuple(rooms), grid, zone_map)
