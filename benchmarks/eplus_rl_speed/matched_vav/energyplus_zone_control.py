"""EnergyPlus-only workload with individual zone setpoints and HVAC energy."""

import numpy as np
from tf_agents.specs import array_spec

from . import energyplus_model
from . import rl_env

JOULES_PER_KWH = 3_600_000.0
ENERGY_PENALTY = 0.1


def _separate_setpoints(model, names):
  schedules = model['Schedule:Constant']
  thermostats = model[energyplus_model.THERMOSTAT]
  devices = model['ZoneControl:Thermostat']
  originals = (rl_env.HEAT_SCHEDULE, rl_env.COOL_SCHEDULE)
  actuator_names = []
  for room in names:
    heat, cool = (f'{name} {room}' for name in originals)
    for old, new in zip(originals, (heat, cool)):
      schedules[new] = schedules[old].copy()
    key = f'Setpoints {room}'
    thermostats[key] = dict(
        heating_setpoint_temperature_schedule_name=heat,
        cooling_setpoint_temperature_schedule_name=cool,
    )
    devices[f'{room} Thermostat']['control_1_name'] = key
    actuator_names.extend((heat, cool))
  return tuple(actuator_names)


class ZoneEnergyPlus(rl_env._EnergyPlus):
  """Collect HVAC electric power at each internal system step."""

  def __init__(self, model, epw, ep_home, output, names, steps_per_episode):
    self.setpoint_schedule_names = _separate_setpoints(model, names)
    self._power_sensors = [
        (variable, name)
        for kind, variable in (
            (energyplus_model.FAN, 'Fan Electricity Rate'),
            ('Fan:SystemModel', 'Fan Electricity Rate'),
            (energyplus_model.DX_COIL, 'Cooling Coil Electricity Rate'),
            ('Coil:Cooling:DX:TwoSpeed', 'Cooling Coil Electricity Rate'),
            (energyplus_model.REHEAT, 'Heating Coil Electricity Rate'),
        )
        for name in model.get(kind, ())
    ]
    if not self._power_sensors:
      raise ValueError('No supported HVAC electricity meters in the model')
    super().__init__(model, epw, ep_home, output, names, steps_per_episode)

  def _register_extra_callbacks(self):
    self._power_handles = None
    self._hvac_joules = 0.0
    self.hvac_kwh = []
    for variable, key in self._power_sensors:
      self.exchange.request_variable(self.state, variable, key)
    self.runtime.callback_end_system_timestep_after_hvac_reporting(
        self.state, self._read_power
    )

  def _read_power(self, state):
    if not self.active_step or self.stopping:
      return
    try:
      exchange = self.exchange
      if self._power_handles is None:
        self._power_handles = [
            exchange.get_variable_handle(state, variable, key)
            for variable, key in self._power_sensors
        ]
        if min(self._power_handles) < 0:
          missing = [
              sensor for sensor, h in zip(self._power_sensors, self._power_handles)
              if h < 0
          ]
          raise RuntimeError(f'EnergyPlus HVAC power sensors missing: {missing}')
      watts = sum(exchange.get_variable_value(state, h) for h in self._power_handles)
      self._hvac_joules += watts * exchange.system_time_step(state) * 3600
    except Exception as error:
      self._fail(state, error)

  def _extra_observations(self):
    energy = self._hvac_joules / JOULES_PER_KWH
    if self.steps:
      self.hvac_kwh.append(energy)
    self._hvac_joules = 0.0
    return (energy,)


class ZoneEnergyEnv(rl_env.MatchedEnv):
  """Independent zone cooling/heating commands and an energy penalty."""

  def __init__(self, backend, zones, steps_per_episode):
    super().__init__(backend, zones, steps_per_episode)
    self._expected_shape = (zones + 2,)
    self._obs_spec = array_spec.ArraySpec(
        self._expected_shape, np.float32, 'temperatures_and_hvac_energy'
    )
    self._act_spec = array_spec.BoundedArraySpec(
        (2 * zones,), np.float32, -1.0, 1.0, 'zone_setpoints'
    )

  def _map_action(self, action):
    pairs = np.clip(action, -1.0, 1.0).astype(np.float64).reshape(-1, 2)
    return ([20.5, 24.5] + 1.5 * pairs).ravel()

  def _observation(self, raw):
    values = super()._observation(raw)
    values[-1] = raw[-1] / 10.0
    return values

  def _reward(self, raw):
    return np.float32(super()._reward(raw[:-1]) - ENERGY_PENALTY * raw[-1])
