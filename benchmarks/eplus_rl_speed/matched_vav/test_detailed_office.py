"""Model-generation and in-process EnergyPlus checks for the detailed office."""

from datetime import date
import json
import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks.eplus_rl_speed.matched_vav import generate


def test_detailed_office_keeps_the_shared_floor_plan(tmp_path):
  spec = generate.PlanSpec(25, window_width_m=2)
  generate.write_models(tmp_path, spec, date(2024, 7, 15), '24.2', detailed_office=True)
  manifest = generate.read_manifest(tmp_path)
  plan, _ = generate.load_plan(tmp_path)
  model = json.loads((tmp_path / 'model.epJSON').read_text())
  assert manifest['detailed_office'] and manifest['office']
  assert len(plan.rooms) == len(model['Zone']) == 25
  assert len(model['AirLoopHVAC']) >= 2
  assert len(model['AirTerminal:SingleDuct:VAV:Reheat']) == 25
  assert len(model['Sizing:Zone']) == 25
  assert model['BuildingSurface:Detailed']['room_1 ceiling']['outside_boundary_condition'] == 'Outdoors'
  assert model['BuildingSurface:Detailed']['room_1 floor']['outside_boundary_condition'] == 'Ground'


def test_detailed_office_episode_meters_power_with_zone_control(tmp_path):
  from benchmarks.eplus_rl_speed.matched_vav import rl_env

  installation = os.environ.get('ENERGYPLUS_DIR')
  if not installation:
    pytest.skip('Set ENERGYPLUS_DIR to run the EnergyPlus API')
  installation = Path(installation)
  weather = next((installation / 'WeatherData').glob('*Chicago*epw'))
  model_dir = tmp_path / 'model'
  spec = generate.PlanSpec(5, window_width_m=2)
  generate.write_models(model_dir, spec, date(2024, 7, 15), '24.2', detailed_office=True)
  env = rl_env.make_env('eplus', model_dir, weather, installation, tmp_path, control='zone-energy')
  try:
    assert env.reset().observation.shape == (7,)
    assert env.action_spec().shape == (10,)
    for _ in range(env.steps_per_episode):
      step = env.step(np.zeros(10, dtype=np.float32))
      assert np.isfinite(step.observation).all()
    assert step.is_last()
    assert len(env.backend.hvac_kwh) == 288
    assert sum(env.backend.hvac_kwh) > 0
  finally:
    env.close()
