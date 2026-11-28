"""Compare native grouped/full SBSim and detailed EnergyPlus for one week."""

import argparse
from collections import Counter
from collections import defaultdict
from contextlib import contextmanager
from copy import copy
from dataclasses import asdict
from datetime import date
import faulthandler
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from tempfile import mkdtemp
from time import perf_counter
import traceback

import numpy as np
import pandas as pd

from .matched_vav import generate
from .matched_vav import sbsim_model
from .matched_vav import settings
from . import eplus_env
from . import physics
from smart_control.simulator.grouped_jax_simulator import GroupedJaxSimulator
from smart_control.simulator.grouped_jax_simulator import InteriorMass


_SECTIONS = ('timing', 'physics', 'sac')
_ENGINES = ('full_jax', 'grouped_jax', 'energyplus')
_TICK = pd.Timedelta(seconds=settings.STEP_SECONDS)
_DAY_STEPS = 24 * settings.STEPS_PER_HOUR
_KELVIN = sbsim_model.KELVIN_OFFSET
_CENTER = np.array([20.5, 24.5])
_ACTION_RANGE = 1.5


_DETAILED_INTERIOR_MASS = InteriorMass(
    capacity_per_m2=settings.DETAILED_MASS_CAPACITY_PER_M2,
    air_conductance_per_m2=settings.DETAILED_MASS_AIR_CONDUCTANCE_PER_M2,
    exposed_area_per_m2=settings.DETAILED_MASS_EXPOSED_AREA_PER_M2,
    emissivity=0.9,
)


class _OfficeSimulator(sbsim_model._OfficeForcing, GroupedJaxSimulator):
  """Use the office heat schedule with the public thermal solver."""


def _solver_options(args, variant, *, linearized_radiative_heat_transfer=None):
  grouped = variant == 'grouped_jax'
  linearized = variant != 'full_jax'
  if linearized_radiative_heat_transfer is not None:
    linearized = linearized_radiative_heat_transfer
  method = (
      args.grouped_method
      if grouped
      else ('sparse_lu' if linearized and args.platform == 'cpu' else 'cg')
  )
  return dict(
      grouped=grouped,
      solver_method=method,
      platform=args.platform,
      linearized_radiative_heat_transfer=linearized,
      include_interior_radiative_heat_transfer=True,
      include_exterior_radiative_heat_transfer=True,
      include_interior_mass=True,
      interior_mass=_DETAILED_INTERIOR_MASS,
      wall_depth_cells=2,
  )


class _SbsimOffice:
  """One native SBSim HVAC instance with independently actuated zones."""

  def __init__(self, plan, weather, args, variant,
               *, linearized_radiative_heat_transfer=None):
    self.num_envs = 1
    self.episode_steps = args.days * _DAY_STEPS
    self.names = [room.name for room in plan.rooms]
    self.observation_size = len(self.names) + 1
    self.action_size = 2 * len(self.names)
    self.weather = weather
    initial = sbsim_model.make_simulator(
        plan,
        weather,
        weather.start,
        office=True,
        detailed_office=True,
        solver='linear-radiation',
    )
    self.sim = _OfficeSimulator(
        building=initial.building,
        hvac=initial.hvac,
        office_heat=initial._office_heat,
        weekend_office_heat=initial._weekend_office_heat,
        weather_controller=weather,
        start_timestamp=weather.start,
        time_step_sec=settings.STEP_SECONDS,
        convergence_threshold=sbsim_model.OUTER_TOLERANCE,
        iteration_limit=100,
        iteration_warning=100,
        relative_convergence_threshold=None,
        **_solver_options(
            args, variant,
            linearized_radiative_heat_transfer=linearized_radiative_heat_transfer
        ),
    )
    start = perf_counter()
    self.sim.prepare_thermal_system(
        weather.get_air_convection_coefficient(weather.start)
    )
    self.prepare_s = perf_counter() - start
    self._steps = None

  def reset(self, seed):
    del seed
    self.sim.hvac.initialize_zone_identifier(self.names)
    self.sim.reset()
    self._schedules = []
    for vav in self.sim.hvac.vavs.values():
      schedule = copy(vav.thermostat.get_setpoint_schedule())
      vav.thermostat._setpoint_schedule = schedule
      self._schedules.append(schedule)
    self._steps = 0
    return self._observe(self.weather.start)

  def _observe(self, timestamp):
    temperatures = list(self.sim.building.get_zone_average_temps().values())
    raw = np.asarray(temperatures + [self.weather.get_current_temp(timestamp)])
    values = (raw - _KELVIN).astype(np.float32)[None, :]
    if not np.isfinite(values).all():
      raise RuntimeError(
          f'SBSim returned nonfinite temperatures at {timestamp}'
      )
    return (values - 22.5) / 10.0

  def step(self, actions):
    if self._steps is None or self._steps >= self.episode_steps:
      raise RuntimeError('Reset SBSim before starting an episode')
    pairs = np.asarray(actions, dtype=np.float64)
    if pairs.shape != (1, self.action_size) or not np.isfinite(pairs).all():
      raise ValueError(
          f'Expected finite actions with shape (1, {self.action_size})'
      )
    bounds = (
        _CENTER + _KELVIN + _ACTION_RANGE * np.clip(pairs, -1, 1).reshape(-1, 2)
    )
    for schedule, pair in zip(self._schedules, bounds, strict=True):
      schedule.comfort_temp_window = tuple(pair)
      schedule.eco_temp_window = tuple(pair)
    timestamp = self.sim.current_timestamp
    self.sim.step_sim(video_filename=None)
    self._steps += 1
    obs = self._observe(timestamp)
    zones_c = 10 * obs[:, :-1] + 22.5
    penalty = np.maximum(settings.HEATING_SETPOINT_C - zones_c, 0) + np.maximum(
        zones_c - settings.COOLING_SETPOINT_C, 0
    )
    return obs, -penalty.mean(axis=1)

  def close(self):
    pass


def _model(args, root, zones, *, needs_energyplus=True):
  spec = generate.PlanSpec(
      zones,
      window_width_m=2,
      exterior_wall_thickness_m=settings.EXTERIOR_WALL_THICKNESS_M,
  )
  plan = generate.make_plan(spec)
  location = root / 'models' / f'zone{zones}'
  if needs_energyplus and not all(
      (location / name).is_file() for name in ('floorplan.json', 'model.epJSON')
  ):
    generate.write_models(
        location,
        spec,
        args.run_day,
        args.energyplus_version,
        detailed_office=True,
    )
  weather = sbsim_model.EpwWeather(args.epw, args.run_day, args.days)
  return plan, weather, location


def _make_environment(args, variant, plan, weather, model_dir, output):
  if variant == 'energyplus':
    return eplus_env.EnergyPlusOffice(
        model_dir, args.epw, args.energyplus_dir, output, 1, args.days
    )
  return _SbsimOffice(plan, weather, args, variant)


def _actions(weather, zones, steps):
  pairs = (
      physics.setpoint_schedule(weather.start, steps) - _CENTER
  ) / _ACTION_RANGE
  return np.tile(pairs, (1, zones)).astype(np.float32)[:, None, :]


def _diagnostics(env, output):
  if isinstance(env, _SbsimOffice):
    return dict(
        thermal_nodes=int(env.sim.thermal_node_count),
        mass_nodes=int(env.sim.mass_node_count),
        thermal_prepare_s=env.prepare_s,
        method=env.sim.solver_method,
    )
  logs = sorted(output.glob('worker-*/episode-*/eplusout.end'))
  return dict(energyplus_completion=[log.read_text().strip() for log in logs])


def _completed_case(root, section, zones, variant, artifact, *, matches=None):
  output = root / section / f'{zones}-{variant}'
  summary = output / 'summary.json'
  if summary.is_file() and (output / artifact).is_file():
    saved = json.loads(summary.read_text(encoding='utf-8'))
    if saved.get('zones') != zones or saved.get('engine') != variant:
      raise ValueError(f'Case metadata does not match {output}')
    if matches is None or matches(saved):
      print(f'Skipping completed {section}: {variant}, {zones} zones', flush=True)
      return saved
  if output.exists():
    archive = root / 'interrupted' / section
    archive.mkdir(parents=True, exist_ok=True)
    backup = Path(mkdtemp(prefix=f'{output.name}-', dir=archive)) / output.name
    output.rename(backup)
    print(f'Preserved previous case at {backup}', flush=True)
  return None


def _timing(args, root):
  rows = []
  for zones in args.zones:
    plan, weather, location = _model(args, root, zones)
    actions = _actions(weather, zones, args.days * _DAY_STEPS)
    for variant in args.engines:
      saved = _completed_case(root, 'timing', zones, variant, 'rollout.npz')
      if saved is not None:
        rows.append(saved)
        continue
      print(f'Timing {variant}, {zones} zones', flush=True)
      output = root / 'timing' / f'{zones}-{variant}'
      start = perf_counter()
      env = _make_environment(args, variant, plan, weather, location, output)
      setup = perf_counter() - start
      try:
        start = perf_counter()
        env.reset(args.seed)
        reset = perf_counter() - start
        durations = np.empty(len(actions))
        zone_c = np.empty((len(actions), zones), dtype=np.float32)
        for index, action in enumerate(actions):
          start = perf_counter()
          obs, _ = env.step(action)
          durations[index] = perf_counter() - start
          zone_c[index] = obs[0, :-1] * 10.0 + 22.5
      finally:
        start = perf_counter()
        env.close()
        shutdown = perf_counter() - start
      output.mkdir(parents=True, exist_ok=True)
      np.savez_compressed(
          output / 'rollout.npz', step_s=durations, zone_c=zone_c
      )
      warm = durations[1:]
      row = dict(
          zones=zones,
          engine=variant,
          steps=len(actions),
          setup_s=setup,
          reset_s=reset,
          first_step_s=float(durations[0]),
          shutdown_s=shutdown,
          step_sum_s=float(durations.sum()),
          warm_ms=float(1000 * warm.mean()),
          warm_p95_ms=float(1000 * np.percentile(warm, 95)),
          min_zone_c=float(zone_c.min()),
          max_zone_c=float(zone_c.max()),
          **_diagnostics(env, output),
      )
      rows.append(row)
      _write_json(output / 'summary.json', row)
  return rows


def _heat_balance_trace(args, root):
  plan, weather, _ = _model(
      args, root, args.physics_zones, needs_energyplus=False
  )
  steps = args.days * _DAY_STEPS
  zones = len(plan.rooms)
  actions = _actions(weather, zones, steps)
  traces = {}
  metrics = []
  forcing = None
  for variant in ('full_jax', 'full_linear', 'grouped_jax'):
    print(f'Physics {variant}, {zones} zones', flush=True)
    start = perf_counter()
    env = _SbsimOffice(plan, weather, args, variant)
    setup = perf_counter() - start
    env.reset(args.seed)
    sim = env.sim
    diffuser = np.flatnonzero(sim.building.diffusers)
    is_reference = forcing is None
    if is_reference:
      forcing = np.empty((steps, diffuser.size))
    temperatures = np.empty((steps, zones))
    elapsed = np.empty(steps)
    solves = np.empty(steps, dtype=int)
    for i in range(steps):
      stamp = weather.start + i * _TICK
      start = perf_counter()
      if is_reference:
        # Native HVAC heating/cooling takes effect in the following heat step.
        forcing[i] = sim.building.input_q.flat[diffuser]
        env.step(actions[i])
      else:
        sim._current_timestamp = stamp
        sim.building.input_q.flat[diffuser] = forcing[i]
        sim.finite_differences_timestep(
            ambient_temperature=weather.get_current_temp(stamp),
            convection_coefficient=weather.get_air_convection_coefficient(
                stamp
            ),
        )
      temperatures[i] = list(sim.building.get_zone_average_temps().values())
      elapsed[i] = perf_counter() - start
      solves[i] = sim.last_solve_count
    traces[variant] = temperatures - _KELVIN
    row = dict(
        variant=variant,
        zones=zones,
        steps=steps,
        setup_s=setup,
        first_step_s=float(elapsed[0]),
        mean_step_ms=float(1000 * elapsed[1:].mean()),
        control='native_vav' if is_reference else 'hvac_replay',
        linear_solves_per_step=dict(sorted(Counter(solves.tolist()).items())),
        **_diagnostics(env, root),
    )
    metrics.append(row)
  timestamps = pd.date_range(weather.start + _TICK, periods=steps, freq=_TICK)
  return (
      traces,
      metrics,
      timestamps,
      [room.name for room in plan.rooms],
      dict(
          hvac_diffuser_w=forcing,
          diffuser_flat_indices=diffuser,
          grid_shape=plan.grid.shape,
          scheduled_setpoints_c=_CENTER + _ACTION_RANGE * actions[:, 0, :2],
      ),
  )


@contextmanager
def _gpu_telemetry(output, enabled, log):
  if not enabled:
    yield
    return
  command = shutil.which('nvidia-smi')
  wsl_command = Path('/usr/lib/wsl/lib/nvidia-smi')
  if command is None and wsl_command.is_file():
    command = str(wsl_command)
  if command is None:
    print('GPU telemetry unavailable: nvidia-smi was not found', file=log)
    yield
    return
  query = [
      command,
      '--query-gpu=timestamp,index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu',
      '--format=csv,nounits',
      '--loop=2',
  ]
  line_buffer = shutil.which('stdbuf')
  if line_buffer:
    query[:0] = [line_buffer, '-oL', '-eL']
  path = output / 'gpu_telemetry.csv'
  print(f'  GPU telemetry: {path}', flush=True)
  with path.open('w', encoding='utf-8') as stream:
    try:
      monitor = subprocess.Popen(query, stdout=stream, stderr=subprocess.STDOUT)
    except OSError as error:
      print(f'GPU telemetry unavailable: {error}', file=log)
      yield
      return
    try:
      yield
    finally:
      if monitor.poll() is None:
        monitor.terminate()
        try:
          monitor.wait(timeout=3)
        except subprocess.TimeoutExpired:
          monitor.kill()
          monitor.wait()


@contextmanager
def _sac_debug_log(output, enabled, *, monitor_gpu=False):
  if not enabled:
    yield None
    return
  path = output / 'sac_debug.log'
  print(f'  SAC diagnostic log: {path}', flush=True)
  with path.open('w', encoding='utf-8', buffering=1) as log, _gpu_telemetry(output, monitor_gpu, log):
    print(f'Benchmark PID: {os.getpid()}', file=log)
    for key in ('XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_PYTHON_CLIENT_MEM_FRACTION',
                'XLA_PYTHON_CLIENT_ALLOCATOR', 'XLA_FLAGS'):
      print(f'{key}={os.environ.get(key, "<default>")}', file=log)
    faulthandler.dump_traceback_later(45, repeat=True, file=log)
    try:
      yield log
    except Exception:
      traceback.print_exc(file=log)
      raise
    finally:
      faulthandler.cancel_dump_traceback_later()


def _sac(args, root):
  import jax
  from .train import train

  learner = jax.devices(args.sac_learner_platform)[0]
  rows = []
  for zones in args.zones:
    plan, weather, location = _model(args, root, zones)
    for variant in args.engines:
      saved = _completed_case(
          root, 'sac', zones, variant, 'actor.npz',
          matches=lambda row: row.get('learner_platform', args.platform) == learner.platform,
      )
      if saved is not None:
        rows.append(saved)
        continue
      print(f'SAC {variant}, {zones} zones', flush=True)
      output = root / 'sac' / f'{zones}-{variant}'
      output.mkdir(parents=True)
      timing = defaultdict(float)
      start = perf_counter()
      env = _make_environment(args, variant, plan, weather, location, output)
      timing['environment_setup_s'] = perf_counter() - start
      print(f"  Environment ready in {timing['environment_setup_s']:.2f}s", flush=True)
      config = argparse.Namespace(
          engine='eplus' if variant == 'energyplus' else 'sbsim',
          transitions=args.transitions or env.episode_steps,
          seed=args.seed,
          batch_size=args.batch_size,
          collect_every=args.collect_every,
          updates=args.updates,
      )
      try:
        monitor_gpu = learner.platform == 'gpu' or (
            args.platform == 'gpu' and variant != 'energyplus'
        )
        with jax.default_device(learner), _sac_debug_log(
            output, args.sac_debug, monitor_gpu=monitor_gpu,
        ) as debug_log:
          state, stats = train(
              env, config, learner, timing, log_progress=True, debug_log=debug_log,
          )
      finally:
        close_start = perf_counter()
        env.close()
        timing['shutdown_s'] = perf_counter() - close_start
      elapsed = perf_counter() - start
      timing['other_s'] = elapsed - sum(timing.values())
      actor = {}
      for layer, (weight, bias) in enumerate(state.actor):
        actor[f'weight_{layer}'] = np.asarray(weight)
        actor[f'bias_{layer}'] = np.asarray(bias)
      np.savez_compressed(output / 'actor.npz', **actor)
      row = dict(
          zones=zones,
          engine=variant,
          learner_platform=learner.platform,
          learner_device=learner.device_kind,
          elapsed_s=elapsed,
          transitions_per_s=stats['transitions'] / elapsed,
          timing_s=dict(timing),
          **stats,
          **_diagnostics(env, output),
      )
      if args.sac_debug:
        row['sac_debug'] = True
      if os.environ.get('XLA_FLAGS'):
        row['xla_flags'] = os.environ['XLA_FLAGS']
      rows.append(row)
      _write_json(output / 'summary.json', row)
  return rows


def _write_json(path, doc):
  temporary = path.with_suffix('.json.tmp')
  temporary.write_text(
      json.dumps(doc, indent=2, allow_nan=False) + '\n', encoding='utf-8'
  )
  temporary.replace(path)


def _parser():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--epw', type=Path, required=True)
  parser.add_argument('--energyplus-dir', type=Path)
  parser.add_argument('--energyplus-version', default='24.2')
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument(
      '--resume', action='store_true',
      help='Reuse completed sections and cases in --output with the same configuration',
  )
  parser.add_argument('--section', choices=_SECTIONS, action='append')
  parser.add_argument(
      '--engines', choices=_ENGINES, nargs='+', default=list(_ENGINES),
      help='Engines to run; one per process avoids sharing JAX compilation',
  )
  parser.add_argument('--zones', type=int, nargs='+', default=[16, 36, 100])
  parser.add_argument('--physics-zones', type=int, default=100)
  parser.add_argument('--days', type=int, default=7)
  parser.add_argument(
      '--transitions', type=int,
      help='Total SAC transitions across episodes (defaults to one episode)',
  )
  parser.add_argument(
      '--run-day', type=date.fromisoformat, default=date(2024, 7, 15)
  )
  parser.add_argument('--platform', choices=['cpu', 'gpu'], default='cpu')
  parser.add_argument(
      '--sac-learner-platform', choices=('cpu', 'gpu'),
      help='Device for the SAC policy and updates (defaults to --platform)',
  )
  parser.add_argument(
      '--grouped-method',
      choices=['dense', 'cg', 'spsolve', 'sparse_lu'],
      default='dense',
  )
  parser.add_argument(
      '--physics-grouped-method',
      choices=['dense', 'cg', 'spsolve', 'sparse_lu'],
      help='Override the grouped backend for the fixed-HVAC physics replay',
  )
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--collect-every', type=int, default=96)
  parser.add_argument('--updates', type=int, default=8)
  parser.add_argument('--batch-size', type=int, default=256)
  parser.add_argument(
      '--sac-debug', action='store_true',
      help='Record update traces, GPU memory telemetry and Python stacks per SAC case',
  )
  return parser


def main():
  import jax
  from . import report

  parser = _parser()
  args = parser.parse_args()
  sections = tuple(dict.fromkeys(args.section or _SECTIONS))
  args.sac_learner_platform = args.sac_learner_platform or args.platform
  args.zones = list(dict.fromkeys(args.zones))
  if not args.epw.is_file():
    parser.error('--epw must name an existing file')
  if (
      min(
          *args.zones,
          args.physics_zones,
          args.days,
          args.updates,
          args.batch_size,
      )
      < 1
  ):
    parser.error('Zone, day, update and batch counts must be positive')
  if not 1 <= args.collect_every <= args.days * _DAY_STEPS:
    parser.error('--collect-every must be within the episode length')
  methods = {
      section: (
          (args.physics_grouped_method or args.grouped_method)
          if section == 'physics' else args.grouped_method
      )
      for section in sections
  }
  if args.platform == 'gpu' and 'sparse_lu' in methods.values():
    parser.error('sparse_lu runs on CPU; use a JAX grouped method for GPU')
  if ('timing' in sections or 'sac' in sections) and (
      args.energyplus_dir is None
      or not (args.energyplus_dir / 'pyenergyplus/api.py').is_file()
  ):
    parser.error('Timing and SAC need --energyplus-dir containing pyenergyplus')
  try:
    device = jax.devices(args.platform)[0]
    if 'sac' in sections:
      jax.devices(args.sac_learner_platform)[0]
  except (RuntimeError, IndexError) as error:
    parser.error(f'JAX device unavailable: {error}')

  root = args.output.resolve()
  if args.resume:
    if not (root / 'summary.json').is_file():
      parser.error('--resume requires an existing study summary.json in --output')
  else:
    try:
      root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
      parser.error(f'--output already exists: {root}. Use --resume to continue.')
  summary = dict(
      sections=sections,
      section_grouped_method=methods,
      physics_zones=args.physics_zones,
      setup=dict(
          start_day=args.run_day.isoformat(),
          days=args.days,
          zones=args.zones,
          episode_steps=args.days * _DAY_STEPS,
          cv_size_m=0.5,
          window_width_m=2.0,
          epw=args.epw.name,
          epw_sha256=sha256(args.epw.read_bytes()).hexdigest(),
          platform=device.platform,
          jax_device=device.device_kind,
          jax=jax.__version__,
          host=platform.platform(),
          energyplus_input_version=args.energyplus_version,
          cpu_affinity=sorted(os.sched_getaffinity(0))
          if hasattr(os, 'sched_getaffinity')
          else None,
          seed=args.seed,
          collect_every=args.collect_every,
          updates=args.updates,
          batch_size=args.batch_size,
          grouped_method=args.grouped_method,
          exterior_wall_thickness_m=settings.EXTERIOR_WALL_THICKNESS_M,
          effective_interior_mass=asdict(_DETAILED_INTERIOR_MASS),
      ),
  )
  with (root / '.run.lock').open('a') as lock:
    try:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      parser.error(f'A benchmark process already holds the run lock in {root}')
    if args.resume:
      saved = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
      if saved.get('section_artifact_dir'):
        parser.error('--resume requires a run directory, not a merged report')
      mismatches = [
          key for key, value in summary['setup'].items()
          if saved['setup'].get(key) != value
      ]
      saved_methods = saved['section_grouped_method']
      mismatches.extend(
          f'{section} solver' for section, method in methods.items()
          if section in saved_methods and saved_methods[section] != method
      )
      saved_physics_zones = saved.get('physics_zones', saved.get('physics', {}).get('zones'))
      if 'physics' in sections and saved_physics_zones not in (None, args.physics_zones):
        mismatches.append('physics_zones')
      if 'sac' in sections and 'sac' in saved:
        saved_platform = saved['sac'][0].get('learner_platform', saved['setup']['platform'])
        if saved_platform != args.sac_learner_platform:
          mismatches.append('SAC learner platform for completed section')
      if mismatches:
        parser.error('Resume configuration differs: ' + ', '.join(mismatches))
      saved['sections'] = list(dict.fromkeys((*saved['sections'], *sections)))
      saved_methods.update(methods)
      saved['physics_zones'] = saved_physics_zones or args.physics_zones
      summary = saved
    _run_sections(args, root, sections, methods, summary, report)
  print(root, flush=True)


def _run_sections(args, root, sections, methods, summary, report):
  _write_json(root / 'summary.json', summary)
  for section in sections:
    if section in summary:
      print(f'Skipping completed section: {section}', flush=True)
      continue
    if section == 'timing':
      summary[section] = _timing(args, root)
      report.timing(root, summary[section])
    elif section == 'physics':
      physics_args = copy(args)
      physics_args.grouped_method = methods[section]
      summary[section] = report.temperature(
          root, *_heat_balance_trace(physics_args, root)
      )
    elif section == 'sac':
      summary[section] = _sac(args, root)
      report.sac(root, summary[section])
    _write_json(root / 'summary.json', summary)
    report.markdown(root, summary)


if __name__ == '__main__':
  main()
