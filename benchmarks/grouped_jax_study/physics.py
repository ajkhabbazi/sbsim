"""Time and compare full and grouped SBSim heat balances on one office week."""

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import scipy

from benchmarks.eplus_rl_speed.matched_vav import coupled_radiation
from benchmarks.eplus_rl_speed.matched_vav import generate
from benchmarks.eplus_rl_speed.matched_vav import sbsim_model
from benchmarks.eplus_rl_speed.matched_vav import settings
from smart_control.simulator import constants

DAY_STEPS = 24 * settings.STEPS_PER_HOUR
TICK = pd.Timedelta(seconds=settings.STEP_SECONDS)
REFERENCE = 'full-nonlinear'
CONDENSED = 'grouped-linear-condensed-scipy'
VARIANTS = {
    REFERENCE: (False, True, False),
    'full-linear': (False, False, False),
    'grouped-nonlinear': (True, True, False),
    'grouped-linear': (True, False, False),
    CONDENSED: (True, False, True),
}
DEFAULT_MODEL = (
    Path(__file__).resolve().parents[1] / 'eplus_rl_speed' / 'matched_vav'
    / 'models' / 'zone100_office'
)


class PhysicsSimulator(sbsim_model.OfficeCoupledRadiationSimulator):
  """Use the same office and radiosity system in each factorial variant."""

  def __init__(self, *, grouped, nonlinear, condensed, **kwargs):
    self.merge_envelope = grouped
    self.nonlinear = nonlinear
    self.eliminate_radiosity = condensed
    self.matrix_setup_s = 0.0
    self.last_kernel_s = 0.0
    super().__init__(air_block_cells=None if grouped else 1, **kwargs)

  def _build_static(self, convection_coefficient):
    begin = perf_counter()
    static = super()._build_static(convection_coefficient)
    self.matrix_setup_s += perf_counter() - begin
    return static

  def finite_differences_timestep(self, **kwargs):
    begin = perf_counter()
    prior_setup = self.matrix_setup_s
    converged = super().finite_differences_timestep(**kwargs)
    self.last_kernel_s = perf_counter() - begin - (self.matrix_setup_s - prior_setup)
    if not converged:
      raise RuntimeError(f'Heat balance did not converge at {self.current_timestamp}')
    return converged


def make_simulator(plan, weather, start, variant):
  # The shared factory configures physical properties, diffusers, office
  # schedules, and VAVs. The comparison changes only its heat-balance class.
  office = sbsim_model.make_simulator(
      plan, weather, start, office=True, solver='coupled-radiation'
  )
  grouped, nonlinear, condensed = VARIANTS[variant]
  return PhysicsSimulator(
      grouped=grouped,
      nonlinear=nonlinear,
      condensed=condensed,
      office_heat=office._office_heat,
      building=office.building,
      hvac=office.hvac,
      weather_controller=weather,
      start_timestamp=start,
      time_step_sec=settings.STEP_SECONDS,
      convergence_threshold=sbsim_model.OUTER_TOLERANCE,
      iteration_limit=100,
      iteration_warning=100,
      relative_convergence_threshold=None,
  )


def setpoint_schedule(start, steps):
  """Keep the same daytime and overnight setpoints for all zones."""
  times = pd.date_range(start, periods=steps, freq=TICK)
  occupied = (times.dayofweek < 5) & (times.hour >= 7) & (times.hour < 19)
  return np.where(occupied[:, None], (21.0, 24.0), (19.0, 26.0))


@dataclass
class HeatForcing:
  flat_indices: np.ndarray
  watts: np.ndarray


@dataclass
class Trace:
  zones_c: np.ndarray
  kernel_s: np.ndarray
  step_s: np.ndarray
  lu_solves: np.ndarray
  snapshots_c: np.ndarray

  @classmethod
  def empty(cls, sim, steps):
    return cls(
        zones_c=np.empty((steps, len(sim.hvac.vavs))),
        kernel_s=np.empty(steps),
        step_s=np.empty(steps),
        lu_solves=np.empty(steps, dtype=int),
        snapshots_c=np.empty(
            (_snapshot_steps(steps).size, *sim.building.temp.shape),
            dtype=np.float32,
        ),
    )

  def record_step(self, sim, step, begin):
    self.zones_c[step] = np.fromiter(
        sim.building.get_zone_average_temps().values(),
        dtype=np.float64,
        count=self.zones_c.shape[1],
    ) - sbsim_model.KELVIN_OFFSET
    self.step_s[step] = perf_counter() - begin
    self.kernel_s[step] = sim.last_kernel_s
    self.lu_solves[step] = sim.last_outer_iteration_count
    if (step + 1) % DAY_STEPS == 15 * settings.STEPS_PER_HOUR:
      self.snapshots_c[step // DAY_STEPS] = (
          sim.building.temp - sbsim_model.KELVIN_OFFSET
      )


@dataclass
class ClosedLoopRun:
  trace: Trace
  forcing: HeatForcing
  reset_s: float


def _snapshot_steps(steps):
  return np.arange(15 * settings.STEPS_PER_HOUR - 1, steps, DAY_STEPS)


def run_closed_loop(sim, start, actions):
  """Advance native thermostat, heat balance, and VAVs with scheduled setpoints."""
  begin = perf_counter()
  sim.reset()
  reset_s = perf_counter() - begin
  schedule = next(iter(sim.hvac.vavs.values())).thermostat.get_setpoint_schedule()
  trace = Trace.empty(sim, len(actions))
  diffusers = np.flatnonzero(sim.building.diffusers)
  forcing = HeatForcing(diffusers, np.empty((len(actions), diffusers.size)))
  for step, setpoints_c in enumerate(actions):
    # The thermal solve consumes the previous step's VAV demand. Capture the
    # grid before the current thermostat/VAV step overwrites that demand.
    forcing.watts[step] = sim.building.input_q.flat[diffusers]
    begin = perf_counter()
    bounds = tuple(setpoints_c + sbsim_model.KELVIN_OFFSET)
    schedule.comfort_temp_window = bounds
    schedule.eco_temp_window = bounds
    sim.setup_step_sim()
    sim.execute_step_sim(video_filename=None)
    trace.record_step(sim, step, begin)
  if sim.current_timestamp != start + len(actions) * TICK:
    raise RuntimeError('Closed-loop simulator did not cover the requested period')
  return ClosedLoopRun(trace, forcing, reset_s)


def run_replay(sim, weather, start, forcing):
  """Replay per-diffuser forcing with the same office loads and weather."""
  sim.reset()
  trace = Trace.empty(sim, len(forcing.watts))
  for step, watts in enumerate(forcing.watts):
    begin = perf_counter()
    stamp = start + step * TICK
    sim._current_timestamp = stamp
    sim.building.input_q.flat[forcing.flat_indices] = watts
    sim.finite_differences_timestep(
        ambient_temperature=weather.get_current_temp(stamp),
        convection_coefficient=weather.get_air_convection_coefficient(stamp),
    )
    trace.record_step(sim, step, begin)
  return trace


def _timings(trace, *, exclude_setup_step):
  sample = slice(1, None) if exclude_setup_step else slice(None)
  return dict(
      first_step_s=float(trace.step_s[0]),
      thermal_kernel_sum_s=float(trace.kernel_s.sum()),
      step_sum_s=float(trace.step_s.sum()),
      warm_kernel_ms=float(1000 * trace.kernel_s[sample].mean()),
      warm_end_to_end_ms=float(1000 * trace.step_s[sample].mean()),
      warm_end_to_end_p90_ms=float(1000 * np.percentile(trace.step_s[sample], 90)),
      lu_solves_per_timestep=dict(sorted(Counter(trace.lu_solves.tolist()).items())),
  )


def _errors(candidate, reference, names, timestamps):
  delta = candidate.zones_c - reference.zones_c
  absolute = np.abs(delta)
  step, zone = np.unravel_index(absolute.argmax(), absolute.shape)
  return dict(
      mean_absolute_c=float(absolute.mean()),
      rmse_c=float(np.sqrt(np.mean(delta**2))),
      p95_absolute_c=float(np.percentile(absolute, 95)),
      maximum_absolute_c=float(absolute[step, zone]),
      maximum_zone=names[zone],
      maximum_interval_end=timestamps[step].isoformat(),
      zone_temperature_range_c=[float(candidate.zones_c.min()), float(candidate.zones_c.max())],
  )


def _save_forcing(output, forcing, plan, start, actions):
  flat_indices = forcing.flat_indices
  zones = np.full(plan.grid.size, -1, dtype=int)
  for index, room in enumerate(plan.rooms):
    west, south, east, north = room.bounds_m
    cv = plan.spec.cv_size_m
    y0 = plan.grid.shape[0] - 1 - round(north / cv)
    y1 = plan.grid.shape[0] - 1 - round(south / cv)
    x0, x1 = round(west / cv) + 1, round(east / cv) + 1
    zones.reshape(plan.grid.shape)[y0:y1, x0:x1] = index
  zone_indices = zones[flat_indices]
  if np.any(zone_indices < 0):
    raise RuntimeError('HVAC replay has a diffuser outside the mapped zones')
  np.savez_compressed(
      output / 'forcing.npz',
      hvac_flat_indices=flat_indices,
      hvac_w=forcing.watts,
      zone_hvac_w=forcing.watts @ (zone_indices[:, None] == np.arange(len(plan.rooms))),
      zone_names=np.array([room.name for room in plan.rooms]),
      grid_shape=np.array(plan.grid.shape),
      zone_setpoints_c=actions,
      initial_time_iso=np.array(start.isoformat()),
  )


def _save_traces(output, plan, times, closed, replay):
  traces = {
      f'{name.replace("-", "_")}_{mode}':
          trace.zones_c
      for mode, runs in (('closed_loop', closed), ('fixed_hvac', replay))
      for name, trace in runs.items()
  }
  np.savez_compressed(
      output / 'zone_temperatures.npz',
      **traces,
      zone_names=np.array([room.name for room in plan.rooms]),
      interval_end_iso=np.array([stamp.isoformat() for stamp in times]),
  )
  snapshots = {
      f'{name.replace("-", "_")}_{mode}': trace.snapshots_c
      for mode, runs in (('closed_loop', closed), ('fixed_hvac', replay))
      for name, trace in runs.items()
  }
  np.savez_compressed(
      output / 'temperature_snapshots.npz',
      **snapshots,
      floor_plan=plan.grid,
      interval_end_iso=np.array([
          times[step].isoformat() for step in _snapshot_steps(len(times))
      ]),
  )


def _plot(output, plan, times, closed, replay):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.dates as mdates
  import matplotlib.pyplot as plt

  colors = ('#1a2833', '#157e88', '#c74d26', '#6b3192')
  styles = ('-', '--', '-', '--')
  four = tuple(VARIANTS)[:4]
  fig, axes = plt.subplots(2, 2, figsize=(12, 6.6), sharex=True, layout='constrained')
  for row, (mode, runs) in enumerate((('Closed loop', closed), ('Fixed HVAC', replay))):
    abs_error = np.abs(runs['grouped-linear'].zones_c - runs[REFERENCE].zones_c)
    worst_zone = np.unravel_index(abs_error.argmax(), abs_error.shape)[1]
    for col, zone in enumerate((worst_zone, len(plan.rooms) // 2)):
      ax = axes[row, col]
      for name, color, style in zip(four, colors, styles):
        ax.plot(
            times, runs[name].zones_c[:, zone], label=name,
            color=color, ls=style, lw=0.9,
        )
      room = plan.rooms[zone].name
      ax.set_title(f'{mode} · {room}')
      ax.set_ylabel('Zone air temperature (°C)')
      ax.grid(alpha=0.22)
      ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d', tz=times.tz))
  axes[0, 0].legend(ncol=2, fontsize=8)
  fig.suptitle('SBSim zone temperature comparison')
  fig.savefig(output / 'zone_temperature_traces.png', dpi=150)
  plt.close(fig)

  exterior = plan.grid == constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT
  field_limits = np.concatenate([
      replay[name].snapshots_c[-1][~exterior] for name in four
  ])
  fig, axes = plt.subplots(2, 2, figsize=(9, 8), layout='constrained')
  for ax, name in zip(axes.flat, four):
    field = np.ma.masked_where(exterior, replay[name].snapshots_c[-1])
    panel = ax.imshow(
        field, cmap='inferno', interpolation='none',
        vmin=float(field_limits.min()), vmax=float(field_limits.max()),
    )
    ax.set_title(name)
    ax.set_axis_off()
  fig.colorbar(panel, ax=axes, shrink=0.75, label='Temperature (°C)')
  last = times[_snapshot_steps(len(times))[-1]].strftime('%b %d %H:%M')
  fig.suptitle(f'Fixed HVAC · {last} local standard time')
  fig.savefig(output / 'temperature_snapshots.png', dpi=150)
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL)
  parser.add_argument('--epw', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--days', type=int, default=7)
  args = parser.parse_args()
  if args.days < 1 or not args.epw.is_file():
    parser.error('Provide a positive day count and an existing EPW file')
  plan, day = generate.load_plan(args.model_dir)
  if not generate.read_manifest(args.model_dir).get('office'):
    parser.error('The model must include the matched office gains')
  args.output.mkdir(parents=True, exist_ok=False)

  start = perf_counter()
  weather = sbsim_model.EpwWeather(args.epw, day, args.days)
  weather_setup_s = perf_counter() - start
  steps = args.days * DAY_STEPS
  actions = setpoint_schedule(weather.start, steps)
  timestamps = pd.date_range(weather.start + TICK, periods=steps, freq=TICK)
  names = np.array([room.name for room in plan.rooms])
  closed, replay, measurements = {}, {}, {}
  fixed = None
  for name, (grouped, nonlinear, condensed) in VARIANTS.items():
    # Include equally cold geometry/view-factor setup in each variant.
    coupled_radiation._interior_view_factors.cache_clear()
    start = perf_counter()
    sim = make_simulator(plan, weather, weather.start, name)
    model_setup_s = perf_counter() - start
    run = run_closed_loop(sim, weather.start, actions)
    if name == REFERENCE:
      fixed = run.forcing
      _save_forcing(args.output, fixed, plan, weather.start, actions)
    closed[name] = run.trace
    matrix_setup_s = sim.matrix_setup_s
    replay[name] = run_replay(sim, weather, weather.start, fixed)
    if sim.factorization_count != 1 or sim.matrix_setup_s != matrix_setup_s:
      raise RuntimeError(f'Expected one static matrix setup for {name}')
    for trace in (closed[name], replay[name]):
      if not np.isfinite(trace.zones_c).all():
        raise RuntimeError(f'Nonfinite zone temperature in {name}')
    measurements[name] = dict(
        grouped=grouped,
        nonlinear=nonlinear,
        condensed_radiosity=condensed,
        heat_nodes=sim.active_heat_nodes,
        radiosity_nodes=sim.radiosity_nodes,
        radiosity_unknowns=sim.radiosity_unknowns_per_solve,
        model_setup_s=model_setup_s,
        matrix_setup_s=matrix_setup_s,
        closed_loop_reset_s=run.reset_s,
        closed_loop=_timings(closed[name], exclude_setup_step=True),
        fixed_hvac=_timings(replay[name], exclude_setup_step=False),
        closed_loop_error=_errors(closed[name], closed[REFERENCE], names, timestamps),
        fixed_hvac_error=_errors(replay[name], replay[REFERENCE], names, timestamps),
    )
    print(name, json.dumps(measurements[name]), flush=True)

  reference_replay_error = _errors(replay[REFERENCE], closed[REFERENCE], names, timestamps)
  condensed_replay_error = _errors(replay[CONDENSED], replay['grouped-linear'], names, timestamps)
  if reference_replay_error['maximum_absolute_c'] > 1e-9:
    raise RuntimeError('Full nonlinear fixed forcing did not reproduce its source run')
  if condensed_replay_error['maximum_absolute_c'] > 1e-7:
    raise RuntimeError('Eliminating radiosity changed the grouped linear model')
  _save_traces(args.output, plan, timestamps, closed, replay)
  _plot(args.output, plan, timestamps, closed, replay)
  summary = dict(
      platform=platform.platform(),
      backend='SciPy sparse LU, CPU float64',
      scipy=scipy.__version__,
      numpy=np.__version__,
      cpu_affinity=sorted(os.sched_getaffinity(0)),
      local_start=weather.start.isoformat(),
      steps=steps,
      zones=len(plan.rooms),
      grid_shape=list(plan.grid.shape),
      timestep_seconds=settings.STEP_SECONDS,
      epw=args.epw.name,
      epw_sha256=sha256(args.epw.read_bytes()).hexdigest(),
      floorplan_sha256=sha256((args.model_dir / 'floor_plan.npy').read_bytes()).hexdigest(),
      floorplan_manifest_sha256=sha256(
          (args.model_dir / 'floorplan.json').read_bytes()
      ).hexdigest(),
      weather_setup_s=weather_setup_s,
      setpoints='Weekdays 07:00–19:00 local standard time 21/24 °C, otherwise 19/26 °C',
      linear_method='One cached solve with a 22.5 °C derivative and the previous state T^4 correction',
      fixed_hvac_scope='Exact HVAC watts at each diffuser from full nonlinear; office heat and weather retained',
      reference_replay_error=reference_replay_error,
      condensed_vs_grouped_linear_fixed_error=condensed_replay_error,
      variants=measurements,
  )
  (args.output / 'summary.json').write_text(
      json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8'
  )


if __name__ == '__main__':
  main()
