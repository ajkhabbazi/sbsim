"""Write the common manifest, EnergyPlus model, and SBSim input arrays."""

import argparse
from dataclasses import asdict
from dataclasses import replace
from datetime import date
import json
from pathlib import Path
import re

import numpy as np

from .energyplus_model import make_model
from .floorplan import FloorPlan
from .floorplan import PlanSpec
from .floorplan import make_plan
from .settings import STEP_SECONDS


def energyplus_from_manifest(plan, run_day, manifest):
  version = manifest['energyplus_version']
  if manifest.get('detailed_office', False):
    from . import detailed_office
    return detailed_office.make_model(plan, run_day, version)
  return make_model(plan, run_day, version, office=manifest.get('office', False))


def write_models(
    output: Path,
    spec: PlanSpec,
    run_day: date,
    version: str,
    *,
    office: bool = False,
    detailed_office: bool = False,
) -> None:
  if not re.fullmatch(r'\d+\.\d+(\.\d+)?', version):
    raise ValueError('EnergyPlus version must be major.minor[.patch]')
  plan = make_plan(spec)
  manifest = dict(
      spec=asdict(spec),
      run_day=run_day.isoformat(),
      energyplus_version=version,
      timestep_seconds=STEP_SECONDS,
      grid_shape=list(plan.grid.shape),
      grid_cells=plan.grid.size,
      rooms=[asdict(room) for room in plan.rooms],
  )
  if office or detailed_office:
    manifest['office'] = True
  if detailed_office:
    manifest['detailed_office'] = True
  model = energyplus_from_manifest(plan, run_day, manifest)
  output.mkdir(parents=True, exist_ok=True)
  for name, data in [('floorplan.json', manifest), ('model.epJSON', model)]:
    (output / name).write_text(
        json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8'
    )
  np.save(output / 'floor_plan.npy', plan.grid, allow_pickle=False)
  np.save(output / 'zone_map.npy', plan.zone_map, allow_pickle=False)


def read_manifest(directory: Path) -> dict:
  return json.loads((directory / 'floorplan.json').read_text(encoding='utf-8'))


def load_plan(directory: Path) -> tuple[FloorPlan, date]:
  manifest = read_manifest(directory)
  plan = make_plan(PlanSpec(**manifest['spec']))
  arrays = {}
  for name, field in [('floor_plan.npy', 'grid'), ('zone_map.npy', 'zone_map')]:
    array = np.load(directory / name, allow_pickle=False)
    expected = getattr(plan, field)
    if array.dtype != expected.dtype or not np.array_equal(array, expected):
      raise ValueError(f'{name} does not match floorplan.json')
    arrays[field] = array
  rooms = json.loads(json.dumps([asdict(room) for room in plan.rooms]))
  if (
      manifest['rooms'] != rooms
      or manifest['grid_shape'] != list(plan.grid.shape)
      or manifest['grid_cells'] != plan.grid.size
      or manifest['timestep_seconds'] != STEP_SECONDS
  ):
    raise ValueError(
        'floorplan.json contains inconsistent derived geometry or timing'
    )
  return replace(plan, **arrays), date.fromisoformat(manifest['run_day'])


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--out', type=Path, required=True)
  parser.add_argument('--zones', type=int, required=True)
  parser.add_argument('--columns', type=int)
  parser.add_argument('--room-width-m', type=float, default=5.0)
  parser.add_argument('--room-depth-m', type=float, default=4.0)
  parser.add_argument('--height-m', type=float, default=3.0)
  parser.add_argument('--wall-thickness-m', type=float, default=0.5)
  parser.add_argument('--cv-size-m', type=float, default=0.5)
  parser.add_argument('--window-width-m', type=float, default=0.0)
  parser.add_argument(
      '--run-day', type=date.fromisoformat, default=date(2024, 7, 15)
  )
  parser.add_argument('--energyplus-version', default='24.2')
  parser.add_argument(
      '--office',
      action='store_true',
      help='Add scheduled office loads and outdoor air',
  )
  parser.add_argument(
      '--detailed-office', action='store_true',
      help='Use the office schedules, detailed envelope and autosized VAV air loops',
  )
  args = parser.parse_args()
  spec = PlanSpec(
      **{name: getattr(args, name) for name in PlanSpec.__dataclass_fields__}
  )
  try:
    write_models(
        args.out,
        spec,
        args.run_day,
        args.energyplus_version,
        office=args.office,
        detailed_office=args.detailed_office,
    )
  except ValueError as error:
    parser.error(str(error))
  print(args.out.resolve())


if __name__ == '__main__':
  main()
