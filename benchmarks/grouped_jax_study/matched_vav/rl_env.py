"""Matching TF-Agents interfaces for the two floor-plan simulators."""

from datetime import timedelta
import json
from numbers import Integral
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from . import energyplus_model as eplus
from . import settings
from .generate import energyplus_from_manifest
from .generate import load_plan
from .generate import read_manifest

EPISODE_STEPS = 24 * settings.STEPS_PER_HOUR
KELVIN_OFFSET = 273.15
OUTDOOR_TEMP = 'Site Outdoor Air Drybulb Temperature'
HEAT_SCHEDULE = 'Heating Setpoint'
COOL_SCHEDULE = 'Cooling Setpoint'
WAIT_SECONDS = 60
_STOP = object()
WEEKDAYS = (
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
)


def setpoints(action):
  """Map normalized actions to heating [19, 22] and cooling [23, 26] °C."""
  heating, cooling = np.clip(action, -1.0, 1.0)
  return 20.5 + 1.5 * float(heating), 24.5 + 1.5 * float(cooling)


class MatchedEnv(py_environment.PyEnvironment):
  """Shared observation, reward, episode boundary and step timer."""

  def __init__(self, backend, zones, steps_per_episode=EPISODE_STEPS):
    super().__init__(handle_auto_reset=True)
    self.backend = backend
    self.steps_per_episode = steps_per_episode
    self._expected_shape = (zones + 1,)
    self._obs_spec = array_spec.ArraySpec(
        self._expected_shape, np.float32, 'temperatures'
    )
    self._act_spec = array_spec.BoundedArraySpec(
        (2,), np.float32, -1.0, 1.0, 'setpoints'
    )
    self.step_times = []
    self.reset_times = []
    self.episode_steps = 0
    self.completed_episodes = 0

  def action_spec(self):
    return self._act_spec

  def observation_spec(self):
    return self._obs_spec

  @property
  def total_steps(self):
    return len(self.step_times)

  def _observation(self, raw):
    values = np.asarray(raw, dtype=np.float32)
    if values.shape != self._expected_shape or not np.isfinite(values).all():
      raise RuntimeError(f'Invalid temperature observations: {values}')
    return (values - 22.5) / 10.0

  def _reset(self):
    start = time.perf_counter()
    raw = self.backend.reset()
    self.reset_times.append(time.perf_counter() - start)
    self.episode_steps = 0
    return ts.restart(self._observation(raw))

  def _step(self, action):
    pair = self._map_action(action)
    start = time.perf_counter()
    raw = self.backend.step(pair)
    self.step_times.append(time.perf_counter() - start)
    observation = self._observation(raw)
    self.episode_steps += 1
    reward = self._reward(raw)
    if self.episode_steps == self.steps_per_episode:
      self.completed_episodes += 1
      return ts.termination(observation, reward)
    return ts.transition(observation, reward)

  def _map_action(self, action):
    return setpoints(action)

  def _reward(self, raw):
    zones = raw[:-1]
    penalty = np.maximum(settings.HEATING_SETPOINT_C - zones, 0) + np.maximum(
        zones - settings.COOLING_SETPOINT_C, 0
    )
    return np.float32(-np.mean(penalty))

  def close(self):
    self.backend.close()


class _EnergyPlus:
  """Give one API worker exclusive ownership of a simulation state."""

  setpoint_schedule_names = (HEAT_SCHEDULE, COOL_SCHEDULE)

  def __init__(self, model, epw, ep_home, output, names, steps_per_episode):
    if not (ep_home / 'pyenergyplus' / 'api.py').is_file():
      raise ValueError(f'pyenergyplus is missing from {ep_home}')
    sys.path.insert(0, str(ep_home))
    try:
      from pyenergyplus import api as api_module
    finally:
      sys.path.remove(str(ep_home))
    if Path(api_module.__file__).resolve().parent.parent != ep_home:
      raise ValueError(
          'pyenergyplus was already imported from another installation'
      )
    self.api = api_module.EnergyPlusAPI()
    self.runtime = self.api.runtime
    self.exchange = self.api.exchange
    self.epw = epw
    self.output = output
    self.names = names
    self.steps_per_episode = steps_per_episode
    self.episodes = 0
    self.worker = None
    self.run_dirs = []
    model['Output:Variable'] = {
        'Run Period Weather': {
            'key_value': 'Environment',
            'variable_name': OUTDOOR_TEMP,
            'reporting_frequency': 'RunPeriod',
        }
    }
    # Sensor values still come from the Data Exchange API. Keep the error log.
    model['OutputControl:Files'] = {
        'Training Output': {
            f'output_{kind}': 'No'
            for kind in (
                'csv',
                'mtr',
                'eso',
                'eio',
                'tabular',
                'audit',
                'bnd',
                'rdd',
                'mdd',
                'mtd',
                'shd',
            )
        }
    }
    self.model_path = output / 'training.epJSON'
    self.model_path.write_text(json.dumps(model), encoding='utf-8')

  def reset(self):
    self.close()
    self.action_queue = queue.Queue()
    self.observation_queue = queue.Queue()
    self.release_terminal = threading.Event()
    self.sensor_handles = None
    self.actuator_handles = None
    self.active_step = False
    self.next_action = None
    self.stopping = False
    self.steps = 0
    self.run_code = None
    self.worker_error = None
    self.episode_dir = self.output / f'episode-{self.episodes:03d}'
    self.episode_dir.mkdir()
    self.run_dirs.append(self.episode_dir)
    self.episodes += 1
    self.state = self.api.state_manager.new_state()
    self.runtime.set_console_output_status(self.state, False)
    for name in self.names:
      self.exchange.request_variable(self.state, eplus.ZONE_TEMP, name)
    self.exchange.request_variable(self.state, OUTDOOR_TEMP, 'Environment')
    self.runtime.callback_begin_zone_timestep_after_init_heat_balance(
        self.state, self._begin
    )
    self.runtime.callback_end_zone_timestep_after_zone_reporting(
        self.state, self._end
    )
    self._register_extra_callbacks()
    self.worker = threading.Thread(target=self._run, daemon=True)
    self.worker.start()
    return self._receive()

  def _register_extra_callbacks(self):
    pass

  def step(self, pair):
    self.action_queue.put(pair)
    return self._receive()

  def close(self):
    if self.worker is None:
      return
    self.stopping = True
    self.action_queue.put(_STOP)
    self.release_terminal.set()
    self.worker.join(WAIT_SECONDS)
    if self.worker.is_alive():
      raise RuntimeError(f'EnergyPlus did not stop. See {self.episode_dir}')
    self.worker = None
    self.api.state_manager.delete_state(self.state)
    self.runtime.clear_callbacks()
    if self.worker_error is not None:
      raise RuntimeError(
          f'EnergyPlus worker failed. See {self.episode_dir}'
      ) from self.worker_error
    if self.run_code not in (0, None):
      raise RuntimeError(
          f'EnergyPlus exit {self.run_code}. See {self.episode_dir}'
      )

  def _run(self):
    try:
      self.run_code = self.runtime.run_energyplus(
          self.state,
          [
              '-j',
              '1',
              '-w',
              str(self.epw),
              '-d',
              str(self.episode_dir),
              str(self.model_path),
          ],
      )
      self.observation_queue.put(('done', self.run_code))
    except Exception as error:
      self.worker_error = error
      self.observation_queue.put(('error', error))

  def _receive(self):
    try:
      kind, value = self.observation_queue.get(timeout=WAIT_SECONDS)
    except queue.Empty as error:
      raise RuntimeError(
          f'EnergyPlus callback timed out. See {self.episode_dir}'
      ) from error
    if kind == 'error':
      raise RuntimeError(
          f'EnergyPlus callback failed. See {self.episode_dir}'
      ) from value
    if kind != 'observation':
      raise RuntimeError(
          f'EnergyPlus ended early ({value}). See {self.episode_dir}'
      )
    return value

  def _begin(self, state):
    try:
      if self.stopping:
        self.runtime.stop_simulation(state)
        return
      ex = self.exchange
      if (
          ex.warmup_flag(state)
          or ex.kind_of_sim(state) != 3
          or not ex.api_data_fully_ready(state)
      ):
        return
      if self.sensor_handles is None:
        self.sensor_handles = [
            ex.get_variable_handle(state, eplus.ZONE_TEMP, name)
            for name in self.names
        ]
        self.sensor_handles.append(
            ex.get_variable_handle(state, OUTDOOR_TEMP, 'Environment')
        )
        self.actuator_handles = [
            ex.get_actuator_handle(
                state, 'Schedule:Constant', 'Schedule Value', name
            )
            for name in self.setpoint_schedule_names
        ]
        if min(self.sensor_handles + self.actuator_handles) < 0:
          raise RuntimeError(
              'EnergyPlus did not expose all matched sensors and actuators'
          )
        if not np.isclose(
            ex.zone_time_step(state) * 3600, settings.STEP_SECONDS
        ):
          raise RuntimeError(
              'EnergyPlus zone timestep does not match the benchmark'
          )
        self._publish(state)
        self.next_action = self._await_action(state)
      if self.next_action is None:
        return
      for handle, value in zip(self.actuator_handles, self.next_action):
        ex.set_actuator_value(state, handle, value)
      self.active_step = True
    except Exception as error:
      self._fail(state, error)

  def _end(self, state):
    if self.active_step:
      self.active_step = False
      try:
        self.steps += 1
        self._publish(state)
        if self.steps < self.steps_per_episode:
          # Pause before the engine prepares the next timestep so that work
          # happens inside the timed step, even when SAC trains between steps.
          self.next_action = self._await_action(state)
        elif not self.release_terminal.wait(WAIT_SECONDS):
          raise RuntimeError('Timed out waiting for EnergyPlus episode cleanup')
      except Exception as error:
        self._fail(state, error)

  def _await_action(self, state):
    command = self.action_queue.get(timeout=WAIT_SECONDS)
    if command is _STOP:
      self.runtime.stop_simulation(state)
      return None
    return command

  def _publish(self, state):
    values = [
        self.exchange.get_variable_value(state, handle)
        for handle in self.sensor_handles
    ]
    values.extend(self._extra_observations())
    self.observation_queue.put(('observation', np.asarray(values)))

  def _extra_observations(self):
    return ()

  def _fail(self, state, error):
    self.stopping = True
    self.worker_error = error
    self.observation_queue.put(('error', error))
    self.runtime.stop_simulation(state)


class _Sbsim:

  def __init__(
      self,
      plan,
      weather_file,
      run_day,
      names,
      office=False,
      *,
      solver='jax',
      air_block_cells=None,
      episode_days=1,
      exterior_radiation='full',
  ):
    import jax
    import scipy
    from . import sbsim_model

    self.names = names
    self.weather = sbsim_model.EpwWeather(weather_file, run_day, episode_days)
    self.sim = sbsim_model.make_simulator(
        plan,
        self.weather,
        self.weather.start,
        office=office,
        solver=solver,
        air_block_cells=air_block_cells,
        exterior_radiation=exterior_radiation,
    )
    self.version = jax.__version__ if solver == 'jax' else scipy.__version__
    self.solver = solver
    self.air_block_cells = air_block_cells
    self.exterior_radiation = exterior_radiation
    self.tolerance = sbsim_model.OUTER_TOLERANCE
    self.schedule = self.sim.hvac.vavs[
        names[0]
    ].thermostat.get_setpoint_schedule()

  def _apply(self, pair):
    bounds = tuple(value + KELVIN_OFFSET for value in pair)
    self.schedule.comfort_temp_window = bounds
    self.schedule.eco_temp_window = bounds

  def reset(self):
    self._apply((settings.HEATING_SETPOINT_C, settings.COOLING_SETPOINT_C))
    # Native VAV.reset() preserves its thermostat object, including mode history.
    self.sim.hvac.initialize_zone_identifier(self.names)
    self.sim.reset()
    return self._read(self.weather.start)

  def step(self, pair):
    timestamp = self.sim.current_timestamp
    self._apply(pair)
    self.sim.setup_step_sim()
    self.sim.execute_step_sim(video_filename=None)
    delta = self.sim.last_outer_max_delta
    if hasattr(self.sim, 'last_heat_balance_converged'):
      converged = self.sim.last_heat_balance_converged
    elif self.exterior_radiation == 'full':
      converged = delta <= self.tolerance
    else:
      converged = self.sim.last_linear_converged
    if not np.isfinite(delta) or not converged:
      raise RuntimeError(f'SBSim failed to converge at {timestamp}: {delta}')
    return self._read(timestamp)

  def _read(self, timestamp):
    temperatures = self.sim.building.get_zone_average_temps()
    values = [temperatures[name] for name in self.names]
    values.append(self.weather.get_current_temp(timestamp))
    return np.asarray(values) - KELVIN_OFFSET

  def close(self):
    pass


def make_env(
    engine,
    model_dir,
    epw,
    ep_home,
    output,
    *,
    sbsim_solver='jax',
    air_block_cells=None,
    episode_days=1,
    exterior_radiation='full',
    control='shared',
):
  if control not in ('shared', 'zone-energy') or (
      control != 'shared' and engine != 'eplus'
  ):
    raise ValueError('Zone-energy control is an optional EnergyPlus workload')
  if (
      isinstance(episode_days, bool)
      or not isinstance(episode_days, Integral)
      or episode_days < 1
  ):
    raise ValueError('episode_days must be a positive integer')
  episode_days = int(episode_days)
  steps_per_episode = episode_days * EPISODE_STEPS
  plan, run_day = load_plan(model_dir)
  names = [room.name for room in plan.rooms]
  manifest = read_manifest(model_dir)
  office = manifest.get('office', False)
  detailed = manifest.get('detailed_office', False)
  if engine == 'eplus':
    if exterior_radiation != 'full':
      raise ValueError('Exterior radiation ablations apply only to SBSim')
    actual = json.loads(
        (model_dir / 'model.epJSON').read_text(encoding='utf-8')
    )
    if actual != energyplus_from_manifest(plan, run_day, manifest):
      raise ValueError('model.epJSON does not match floorplan.json')
    last_day = run_day + timedelta(days=episode_days - 1)
    period = actual['RunPeriod']['Smoke Day']
    period.update(
        end_year=last_day.year,
        end_month=last_day.month,
        end_day_of_month=last_day.day,
    )
    if (
        last_day.year == run_day.year + 1
        and (last_day.month, last_day.day) < (run_day.month, run_day.day)
    ):
      # EnergyPlus 24.2 with a TMY file needs month/day ranges to wrap Jan 1.
      del period['begin_year'], period['end_year']
      period['day_of_week_for_start_day'] = WEEKDAYS[run_day.weekday()]
    if control == 'zone-energy':
      from . import energyplus_zone_control
      backend_type = energyplus_zone_control.ZoneEnergyPlus
      env_type = energyplus_zone_control.ZoneEnergyEnv
    else:
      backend_type = _EnergyPlus
      env_type = MatchedEnv
    backend = backend_type(
        actual, epw.resolve(), ep_home.resolve(), output.resolve(), names,
        steps_per_episode,
    )
  elif engine == 'sbsim':
    if detailed:
      raise ValueError('The detailed office uses EnergyPlus HVAC and envelope inputs')
    env_type = MatchedEnv
    backend = _Sbsim(
        plan,
        epw,
        run_day,
        names,
        office=office,
        solver=sbsim_solver,
        air_block_cells=air_block_cells,
        episode_days=episode_days,
        exterior_radiation=exterior_radiation,
    )
  else:
    raise ValueError(f'Unknown simulator {engine}')
  env = env_type(backend, len(names), steps_per_episode)
  env.control = control
  return env
