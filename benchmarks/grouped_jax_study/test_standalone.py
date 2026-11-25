"""Regression checks for independence from the two source studies."""

import builtins
import importlib
from pathlib import Path
import sys

import pytest


STUDY = Path(__file__).resolve().parent
OLD_PREFIXES = (
    'benchmarks.eplus_rl_speed',
    'benchmarks.sbsim_sac_study',
)
ENTRY_POINTS = (
    'run',
    'physics',
    'eplus_env',
    'train',
    'interior_mass',
    'jax_env',
)
MODEL_FILES = (
    'floorplan.json',
    'model.epJSON',
    'floor_plan.npy',
    'zone_map.npy',
)


def test_runtime_sources_do_not_reference_either_source_study():
  violations = []
  for path in STUDY.rglob('*.py'):
    if path.name.startswith('test_'):
      continue
    source = path.read_text(encoding='utf-8')
    for prefix in OLD_PREFIXES:
      if prefix in source:
        violations.append(f'{path.relative_to(STUDY)}: {prefix}')
  assert not violations, '\n'.join(violations)


def test_default_models_are_inside_the_study():
  from benchmarks.grouped_jax_study import physics
  from benchmarks.grouped_jax_study import train

  models = (
      physics.DEFAULT_MODEL,
      train.MODELS / 'zone100_office',
      train.MODELS / 'zone100_detailed_office',
  )
  for model in models:
    assert model.is_relative_to(STUDY)
    missing = [name for name in MODEL_FILES if not (model / name).is_file()]
    assert not missing, f'{model}: missing {missing}'


def test_default_models_match_regenerated_geometry_and_energyplus_inputs():
  import json
  from datetime import date

  from benchmarks.grouped_jax_study import train
  from benchmarks.grouped_jax_study.matched_vav import generate

  for name, detailed_office in (
      ('zone100_office', False),
      ('zone100_detailed_office', True),
  ):
    directory = train.MODELS / name
    manifest = generate.read_manifest(directory)
    plan, run_day = generate.load_plan(directory)
    assert run_day == date(2024, 7, 15)
    assert len(plan.rooms) == len(manifest['rooms']) == 100
    assert manifest['office'] is True
    assert manifest.get('detailed_office', False) is detailed_office
    assert json.loads((directory / 'model.epJSON').read_text(encoding='utf-8')) == (
        generate.energyplus_from_manifest(plan, run_day, manifest)
    )


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_entry_points_import_with_source_studies_blocked(monkeypatch):
  original_import = builtins.__import__

  def guarded_import(name, *args, **kwargs):
    if any(name == prefix or name.startswith(prefix + '.') for prefix in OLD_PREFIXES):
      raise AssertionError(f'Attempted to import source-study module {name}')
    return original_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, '__import__', guarded_import)
  for module in ENTRY_POINTS:
    qualified = f'benchmarks.grouped_jax_study.{module}'
    sys.modules.pop(qualified, None)
    importlib.import_module(qualified)
