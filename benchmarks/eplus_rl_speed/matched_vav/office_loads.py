"""Shared hourly ventilation and design loads for the office benchmark."""

from dataclasses import dataclass

from .floorplan import Room

ACTIVITY_W_PER_PERSON = 120.0
SENSIBLE_FRACTION = 0.60


@dataclass(frozen=True)
class RoomUse:
  label: str
  people_per_m2: float
  lights_w_per_m2: float
  plugs_w_per_m2: float
  occupancy: tuple[float, ...]


def _day(*periods: tuple[int, float]) -> tuple[float, ...]:
  """Expand (end hour, value) pairs into local standard time hourly values."""
  hours = []
  for end, value in periods:
    hours.extend([value] * (end - len(hours)))
  return tuple(hours)


OFFICE = RoomUse(
    'Open Office',
    0.08,
    7.0,
    8.0,
    _day(
        (7, 0),
        (8, 0.25),
        (12, 0.90),
        (13, 0.55),
        (17, 0.90),
        (19, 0.25),
        (24, 0),
    ),
)
MEETING = RoomUse(
    'Meeting',
    0.25,
    9.0,
    5.0,
    _day(
        (8, 0),
        (9, 0.15),
        (11, 0.85),
        (13, 0.20),
        (16, 0.65),
        (18, 0.15),
        (24, 0),
    ),
)
SUPPORT = RoomUse(
    'Support',
    0.03,
    5.0,
    12.0,
    _day(
        (7, 0),
        (9, 0.20),
        (17, 0.35),
        (20, 0.15),
        (24, 0),
    ),
)
ROOM_USES = (OFFICE, MEETING, SUPPORT)
_ROOM_SEQUENCE = (OFFICE,) * 6 + (MEETING,) * 2 + (SUPPORT,) * 2
FRESH_AIR_FRACTION = _day((7, 0.05), (19, 0.20), (24, 0.05))
LIGHTS = _day((7, 0.05), (8, 0.45), (18, 0.95), (21, 0.35), (24, 0.05))
PLUGS = _day(
    (7, 0.25),
    (8, 0.50),
    (12, 1.00),
    (13, 0.75),
    (18, 1.0),
    (21, 0.50),
    (24, 0.25),
)


def room_use(room: Room) -> RoomUse:
  index = int(room.name.removeprefix('room_')) - 1
  return _ROOM_SEQUENCE[index % len(_ROOM_SEQUENCE)]


def sensible_watts(room: Room, hour: int) -> float:
  use = room_use(room)
  people = (
      ACTIVITY_W_PER_PERSON
      * SENSIBLE_FRACTION
      * use.people_per_m2
      * use.occupancy[hour]
  )
  return room.area_m2 * (
      people
      + use.lights_w_per_m2 * LIGHTS[hour]
      + use.plugs_w_per_m2 * PLUGS[hour]
  )
