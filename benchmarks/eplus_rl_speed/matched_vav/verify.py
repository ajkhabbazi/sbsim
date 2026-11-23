"""Run a paired model and check geometry, VAV activity and finite temperatures."""

import argparse
import csv
from datetime import date, timedelta
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from . import energyplus_model as eplus
from . import office_loads
from .generate import energyplus_from_manifest
from .generate import load_plan
from .generate import read_manifest
from .settings import STEP_SECONDS
from .settings import STEPS_PER_HOUR

DAY_STEPS = 24 * STEPS_PER_HOUR


def _check_eplus_geometry(eio: Path, plan) -> None:
  columns = (
      'Minimum X {m}',
      'Minimum Y {m}',
      'Maximum X {m}',
      'Maximum Y {m}',
      'Ceiling Height {m}',
      'Floor Area {m2}',
      'Volume {m3}',
      'Exterior Window Area {m2}',
  )
  names = []
  actual = {}
  with eio.open(encoding='utf-8', newline='') as stream:
    for row in csv.reader(stream, skipinitialspace=True):
      if not row:
        continue
      if row[0].strip() == '! <Zone Information>':
        names = [item.strip() for item in row]
      elif row[0].strip() == 'Zone Information':
        actual[row[1].casefold()] = dict(zip(names, row))
  if set(actual) != {room.name for room in plan.rooms}:
    raise ValueError('EnergyPlus did not report the expected zones')
  for room in plan.rooms:
    window_area = sum(
        w.window.width_m * (w.window.head_m - w.window.sill_m)
        for w in room.walls
        if w.window is not None
    )
    expected = (
        *room.bounds_m,
        plan.spec.height_m,
        room.area_m2,
        room.volume_m3,
        window_area,
    )
    measured = [float(actual[room.name][column]) for column in columns]
    # EIO rounds these values to two decimal places.
    if not np.allclose(expected, measured, rtol=0, atol=0.005001):
      raise ValueError(
          f'EnergyPlus geometry disagrees for {room.name}: {measured}'
      )


def _check_eplus_outputs(
    path: Path, plan, run_day: date, office: bool, *, detailed=None, days=1,
) -> dict:
  with path.open(encoding='utf-8', newline='') as stream:
    reader = csv.DictReader(stream)
    fields = reader.fieldnames or []
    rows = list(reader)
  if len(rows) != days * DAY_STEPS:
    raise ValueError(
        f'Expected {days * DAY_STEPS} EnergyPlus timesteps, got {len(rows)}'
    )

  for step, row in enumerate(rows, start=1):
    day_index, day_step = divmod(step - 1, DAY_STEPS)
    day = run_day + timedelta(days=day_index)
    hour, remainder = divmod((day_step + 1) * STEP_SECONDS, 3600)
    minute, second = divmod(remainder, 60)
    expected = [day.strftime('%m/%d'), f'{hour:02}:{minute:02}:{second:02}']
    if row['Date/Time'].split() != expected:
      raise ValueError(f'Unexpected EnergyPlus timestamp on step {step}')

  def values(key, variable):
    prefix = f'{key.upper()}:{variable} ['
    field = next((item for item in fields if item.startswith(prefix)), None)
    if field is None:
      raise ValueError(f'EnergyPlus did not report {key}: {variable}')
    data = np.array([float(row[field]) for row in rows])
    if not np.isfinite(data).all():
      raise ValueError(f'Non-finite EnergyPlus output for {key}: {variable}')
    return data

  temperatures, flow, reheat = [], [], []
  active_terminals = 0
  names = [room.name for room in plan.rooms]
  for room in names:
    temperatures.extend(values(room, eplus.ZONE_TEMP))
    damper = values(f'{room} VAV', eplus.DAMPER)
    supply = values(f'{room} Supply', eplus.NODE_FLOW)
    reheat.extend(values(f'{room} Reheat', eplus.COIL_HEAT))
    supplied = damper.max() > 0 and supply.max() > 0
    active_terminals += int(supplied)
    if detailed is None and not supplied:
      raise ValueError(f'EnergyPlus terminal did not supply air to {room}')
    flow.extend(supply)
  report = dict(
      steps=len(rows),
      zone_temp_c=[min(temperatures), max(temperatures)],
      vav_flow_m3_s=[min(flow), max(flow)],
      peak_reheat_w=max(reheat),
  )
  if detailed is not None:
    report.update(_detailed_eplus_outputs(values, plan, detailed))
    report['zones_with_supply_air'] = active_terminals
    return report
  report['peak_dx_cooling_w'] = float(values('Central DX Coil', eplus.COIL_COOL).max())
  if office:
    outdoor = values('Central VAV', eplus.OFFICE_OUTDOOR_AIR)
    expected_outdoor = np.repeat(
        office_loads.FRESH_AIR_FRACTION, STEPS_PER_HOUR
    )
    if not np.allclose(outdoor, expected_outdoor, atol=1e-6):
      raise ValueError('EnergyPlus outdoor-air flow fraction is off schedule')
    per_room = []
    for room in plan.rooms:
      heat = sum(
          values(room.name, variable)
          for variable in eplus.OFFICE_SENSIBLE_LOADS
      )
      expected = np.repeat(
          [office_loads.sensible_watts(room, hour) for hour in range(24)],
          STEPS_PER_HOUR,
      )
      if not np.allclose(heat, expected, atol=1e-5):
        raise ValueError(
            f'EnergyPlus sensible gains are off schedule for {room.name}'
        )
      per_room.append(heat)
    heat = np.sum(per_room, axis=0)
    report['outdoor_air_fraction'] = [
        float(outdoor.min()),
        float(outdoor.max()),
    ]
    report['office_sensible_gain_w'] = [float(heat.min()), float(heat.max())]
  return report


def _detailed_eplus_outputs(values, plan, model):
  names = [room.name for room in plan.rooms]
  fans = list(model.get(eplus.FAN, ())) + list(model.get('Fan:SystemModel', ()))
  coils = list(model.get(eplus.DX_COIL, ())) + list(model.get('Coil:Cooling:DX:TwoSpeed', ()))
  central_heaters = set(model[eplus.REHEAT]) - {f'{name} Reheat' for name in names}

  def totals(keys, variable):
    return np.sum([values(key, variable) for key in keys], axis=0)

  def as_kwh(power):
    return float(np.sum(power) * STEP_SECONDS / 3_600_000)

  peak_coil = totals(coils, eplus.COIL_COOL)
  delivered_outdoor = np.stack([
      values(f'{name} VAV', 'Zone Air Terminal Outdoor Air Volume Flow Rate')
      for name in names
  ])
  degree_hours = []
  ventilation_ratios = []
  for index, name in enumerate(names):
    zone = values(name, eplus.ZONE_TEMP)
    occupied = values(name, 'Zone People Occupant Count') > 0
    low = values(name, 'Zone Thermostat Heating Setpoint Temperature')
    high = values(name, 'Zone Thermostat Cooling Setpoint Temperature')
    error = np.maximum(low - zone, 0) + np.maximum(zone - high, 0)
    degree_hours.append(float(np.sum(error[occupied]) / STEPS_PER_HOUR))
    target = values(name, 'Zone Target Voz Ventilation Flow Rate')
    needed = occupied & (target > 0)
    ventilation_ratios.extend((delivered_outdoor[index, needed] / target[needed]).tolist())
  ratios = np.asarray(ventilation_ratios)
  return dict(
      peak_dx_cooling_w=float(np.max(peak_coil)),
      cooling_steps=int(np.count_nonzero(peak_coil > 1.0)),
      hvac_electric_kwh=dict(
          fan=as_kwh(totals(fans, 'Fan Electricity Rate')),
          dx=as_kwh(totals(coils, 'Cooling Coil Electricity Rate')),
          central_heat=as_kwh(totals(sorted(central_heaters), 'Heating Coil Electricity Rate')),
          reheat=as_kwh(totals([f'{name} Reheat' for name in names], 'Heating Coil Electricity Rate')),
      ),
      zones_with_outdoor_air=int(np.count_nonzero(np.any(delivered_outdoor > 0, axis=1))),
      occupied_zone_steps_with_ventilation_target=len(ratios),
      occupied_zone_steps_with_at_least_99pct_of_ventilation_target=(
          float(np.mean(ratios >= 0.99)) if ratios.size else None
      ),
      occupied_outdoor_air_to_target_p05=(
          float(np.percentile(ratios, 5)) if ratios.size else None
      ),
      total_occupied_setpoint_discomfort_zone_degree_hours=float(sum(degree_hours)),
      worst_occupied_zone_degree_hours=max(degree_hours),
      air_loops={
          name: dict(
              supply_air_flow_peak_m3_s=float(values(
                  loop['supply_side_outlet_node_names'], eplus.NODE_FLOW).max()),
              supply_temperature_c=np.percentile(values(
                  loop['supply_side_outlet_node_names'], 'System Node Temperature'),
                  [5, 50, 95]).tolist(),
          ) for name, loop in model['AirLoopHVAC'].items()
      },
  )


def verify_energyplus(binary, epw, model_dir, output, plan, run_day, *, days=1):
  # Confirm that the epJSON and arrays are from the same generator inputs.
  manifest = read_manifest(model_dir)
  office = manifest.get('office', False)
  detailed = manifest.get('detailed_office', False)
  model_path = model_dir / 'model.epJSON'
  model = json.loads(model_path.read_text(encoding='utf-8'))
  if model != energyplus_from_manifest(plan, run_day, manifest):
    raise ValueError('model.epJSON does not match floorplan.json')
  output.mkdir(parents=True, exist_ok=False)
  if days != 1:
    last = run_day + timedelta(days=days - 1)
    model['RunPeriod']['Smoke Day'].update(
        end_year=last.year, end_month=last.month, end_day_of_month=last.day,
    )
    model_path = output / 'audit.epJSON'
    model_path.write_text(json.dumps(model), encoding='utf-8')
  with (output / 'energyplus.log').open('w', encoding='utf-8') as log:
    process = subprocess.run(
        [
            binary,
            '-w',
            str(epw.resolve()),
            '-d',
            str(output.resolve()),
            str(model_path.resolve()),
        ],
        cwd=output,
        stdout=log,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=300,
    )
  end_file = output / 'eplusout.end'
  end = end_file.read_text(encoding='utf-8') if end_file.is_file() else ''
  status = re.search(
      r'Completed Successfully--\s*(\d+) Warning;\s*(\d+) Severe Errors', end
  )
  if process.returncode or not status or int(status[2]):
    raise RuntimeError(
        f'EnergyPlus did not complete successfully. See {output}/energyplus.log'
        ' and eplusout.err'
    )
  err = (output / 'eplusout.err').read_text(encoding='utf-8')
  if 'No node connection errors were found.' not in err:
    raise RuntimeError(
        f'EnergyPlus node checks were unsuccessful. See {output}/eplusout.err'
    )
  _check_eplus_geometry(output / 'eplusout.eio', plan)
  details = {}
  if detailed:
    with (output / 'eplusout.eio').open(encoding='utf-8', newline='') as stream:
      sizing = [
          dict(air_loop=row[1], mode=row[2], design_capacity_w=float(row[4]),
               design_supply_m3_s=float(row[6]))
          for row in csv.reader(stream, skipinitialspace=True)
          if row and row[0].strip() == 'System Sizing Information'
      ]
    if len(sizing) != 2 * len(model['AirLoopHVAC']):
      raise ValueError('Missing EnergyPlus heating or cooling sizing for an air loop')
    details['air_loop_sizing'] = sizing
    details['warning_lines'] = [
        line.strip() for line in err.splitlines() if '** Warning **' in line
    ]
  return dict(
      version=err.splitlines()[0].split(',')[2].strip(),
      warnings=int(status[1]),
      severe_errors=int(status[2]),
      **details,
      **_check_eplus_outputs(
          output / 'eplusout.csv', plan, run_day, office,
          detailed=model if detailed else None, days=days,
      ),
  )


def verify_sbsim(epw, plan, run_day, steps, *, office=False):
  import jax
  import pandas as pd
  from . import sbsim_model

  weather = sbsim_model.EpwWeather(epw, run_day)
  simulator = sbsim_model.make_simulator(
      plan, weather, weather.start, office=office
  )
  temperatures, flow = [], []
  active_vavs = set()
  reheat_max = 0.0
  max_delta = 0.0
  indoor_mask = plan.zone_map == 0
  air_min, air_max = float('inf'), float('-inf')
  outdoor_air, sensible_gain = [], []
  for step in range(steps):
    hour = simulator.current_timestamp.hour
    simulator.step_sim()
    if office:
      outdoor_air.append(1.0 - simulator.hvac.air_handler.recirculation)
      sensible_gain.append(simulator.office_heat_w[hour])
      if not np.isclose(outdoor_air[-1], office_loads.FRESH_AIR_FRACTION[hour]):
        raise ValueError('SBSim outdoor-air fraction is off schedule')
      expected = sum(
          office_loads.sensible_watts(room, hour) for room in plan.rooms
      )
      if not np.isclose(sensible_gain[-1], expected):
        raise ValueError(
            'SBSim sensible gains disagree with the grid-based forcing'
        )
    delta = simulator.last_outer_max_delta
    if not np.isfinite(delta) or delta > sbsim_model.OUTER_TOLERANCE:
      raise RuntimeError(
          f'SBSim heat balance did not converge on step {step + 1}: {delta}'
      )
    max_delta = max(max_delta, delta)
    air = simulator.building.temp[indoor_mask] - sbsim_model.KELVIN_OFFSET
    low, high = float(air.min()), float(air.max())
    if not (-20.0 <= low <= high <= 65.0):
      raise RuntimeError(
          f'SBSim indoor air outside the smoke-test range on step {step + 1}: '
          f'{low:.2f} to {high:.2f} C'
      )
    air_min, air_max = min(air_min, low), max(air_max, high)
    temperatures.extend(simulator.building.get_zone_average_temps().values())
    for name, vav in simulator.hvac.vavs.items():
      demand = vav.flow_rate_demand
      flow.append(demand)
      if demand > 0:
        active_vavs.add(name)
      reheat_max = max(reheat_max, vav.reheat_demand)
  if not np.isfinite(temperatures).all() or not np.isfinite(flow).all():
    raise RuntimeError('SBSim reported non-finite zone temperature or flow')
  if active_vavs != set(simulator.hvac.vavs):
    raise RuntimeError('One or more SBSim VAVs did not deliver air')
  expected_time = weather.start + pd.Timedelta(
      seconds=steps * simulator.time_step_sec
  )
  if simulator.current_timestamp != expected_time:
    raise RuntimeError(
        'SBSim did not advance by the requested number of timesteps'
    )
  offset = sbsim_model.KELVIN_OFFSET
  report = dict(
      steps=steps,
      backend=jax.default_backend(),
      jax_version=jax.__version__,
      zone_temp_c=[min(temperatures) - offset, max(temperatures) - offset],
      vav_flow_m3_s=[min(flow), max(flow)],
      indoor_cv_temp_c=[air_min, air_max],
      peak_reheat_water_flow=reheat_max,
      max_outer_delta_k=max_delta,
  )
  if office:
    report['outdoor_air_fraction'] = [min(outdoor_air), max(outdoor_air)]
    report['office_sensible_gain_w'] = [min(sensible_gain), max(sensible_gain)]
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('model_dir', type=Path)
  parser.add_argument('--epw', type=Path, required=True)
  parser.add_argument(
      '--energyplus',
      default='energyplus',
      help='Executable path or name on PATH',
  )
  parser.add_argument(
      '--output',
      type=Path,
      required=True,
      help='New directory for EnergyPlus logs and validation.json',
  )
  parser.add_argument(
      '--steps',
      type=int,
      default=DAY_STEPS,
      help=f'SBSim smoke-test steps (1 to {DAY_STEPS})',
  )
  parser.add_argument('--days', type=int, default=1, help='Weather days for a detailed office')
  args = parser.parse_args()
  if not 1 <= args.steps <= DAY_STEPS:
    parser.error(f'--steps must be between 1 and {DAY_STEPS}')
  if not args.epw.is_file():
    parser.error(f'EPW file does not exist: {args.epw}')
  binary = shutil.which(args.energyplus)
  if binary is None:
    parser.error(f'EnergyPlus executable not found: {args.energyplus}')
  if args.output.exists():
    parser.error(f'--output already exists: {args.output}')
  manifest = read_manifest(args.model_dir)
  detailed = manifest.get('detailed_office', False)
  if args.days < 1 or (args.days != 1 and not detailed):
    parser.error('--days must be positive and multiple days require --detailed-office')

  plan, run_day = load_plan(args.model_dir)
  results = dict(
      run_day=run_day.isoformat(),
      epw=args.epw.name,
      zones=len(plan.rooms),
      grid_shape=list(plan.grid.shape),
      grid_cells=plan.grid.size,
      windows=sum(w.window is not None for r in plan.rooms for w in r.walls),
  )
  results['energyplus'] = verify_energyplus(
      str(Path(binary).resolve()),
      args.epw,
      args.model_dir,
      args.output,
      plan,
      run_day,
      days=args.days,
  )
  office = manifest.get('office', False)
  if detailed:
    results['detailed_office'] = True
  else:
    results['sbsim'] = verify_sbsim(args.epw, plan, run_day, args.steps, office=office)
  if office:
    results['office'] = True
  report = json.dumps(results, indent=2, allow_nan=False)
  (args.output / 'validation.json').write_text(report + '\n', encoding='utf-8')
  print(report)


if __name__ == '__main__':
  main()
