"""Batched device-resident SBSim using the native grouped heat balance."""

from numbers import Integral
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import linalg

from .matched_vav import coupled_radiation
from .matched_vav import generate
from .matched_vav import office_loads
from .matched_vav import sbsim_model
from .matched_vav import settings
from smart_control.simulator import thermostat
from smart_control.utils import constants

from . import interior_mass


_OFF = thermostat.Thermostat.Mode.OFF.value
_HEAT = thermostat.Thermostat.Mode.HEAT.value
_COOL = thermostat.Thermostat.Mode.COOL.value
_PASSIVE = thermostat.Thermostat.Mode.PASSIVE_COOL.value
_KELVIN = sbsim_model.KELVIN_OFFSET
_NONLINEAR_TOLERANCE_K = 1e-10
_NONLINEAR_LIMIT = 100


class _State(NamedTuple):
  temperature: jax.Array
  hvac_heat: jax.Array
  mode: jax.Array
  previous_outdoor: jax.Array
  step: jax.Array


class _NonlinearStats(NamedTuple):
  worst_iterations: jax.Array
  converged: jax.Array


class _Operators(NamedTuple):
  transition: np.ndarray
  sources: np.ndarray
  outdoor: np.ndarray
  comfort: np.ndarray
  zone_nodes: np.ndarray
  grid_weights: np.ndarray
  outdoor_weight: float
  recirculation: np.ndarray
  air_flow: np.ndarray
  water_flow: np.ndarray
  water_temp: float
  heating_supply: float
  cooling_supply: float


def _build_operators(
    model_dir, epw, episode_days, mass_config=None, *, solar_to_interior_mass=True
):
  """Contract the native equations once and precompute episode forcing."""
  plan, run_day = generate.load_plan(model_dir)
  office = generate.read_manifest(model_dir).get('office', False)
  weather = sbsim_model.EpwWeather(epw, run_day, episode_days)
  native = sbsim_model.make_simulator(
      plan,
      weather,
      weather.start,
      office=office,
      solver='linear-radiation-envelope',
  )
  building = native.building
  h = weather.get_air_convection_coefficient(weather.start)
  if mass_config is None:
    static = native._build_static(h)
  else:
    wall_depth_cells = round(plan.spec.wall_thickness_m / plan.spec.cv_size_m)
    static = native._build_grouped_conduction(h, wall_depth_cells=wall_depth_cells)
  active, groups = static['active'], static['groups']
  count = static['conduction'].shape[0]
  positions = np.full(building.temp.size, -1, dtype=np.int64)
  positions[active] = groups
  names = list(native.hvac.vavs)
  hvac_projection = np.zeros((count, len(names)))
  zone_nodes = []
  for index, name in enumerate(names):
    cells = np.ravel_multi_index(
        np.asarray(building._room_dict[name]).T, building.temp.shape
    )
    room_nodes = np.unique(positions[cells])
    if len(room_nodes) != 1 or room_nodes[0] < 0:
      raise ValueError(f'Expected one connected air group for {name}')
    zone_nodes.append(room_nodes[0])
    np.add.at(
        hvac_projection[:, index],
        positions[cells],
        building.diffusers.ravel()[cells],
    )

  mass_count = len(zone_nodes) if mass_config is not None else 0
  if mass_config is None:
    storage = np.bincount(groups, weights=static['storage'], minlength=count)
    radiation = static['operator']
    factor = static['factor']
    geometry = static['geometry']
  else:
    balance = interior_mass.assemble(
        building,
        static,
        zone_nodes,
        mass_config,
        solar_to_interior_mass=solar_to_interior_mass,
    )
    storage, radiation, geometry = (
        balance.storage,
        balance.radiation,
        balance.geometry,
    )
    factor = linalg.splu(
        (balance.conduction + coupled_radiation.SLOPE * radiation).tocsc(),
        permc_spec='MMD_AT_PLUS_A',
    )
    hvac_projection = np.pad(hvac_projection, ((0, mass_count), (0, 0)))
  timestamps = weather._data.index
  outdoor = weather._data['temp_k'].to_numpy()
  if office:
    grouped_office = np.stack([
        np.bincount(groups, weights=power.ravel()[active], minlength=count)
        for power in native._office_heat
    ])
    recirculation = (
        1 - np.asarray(office_loads.FRESH_AIR_FRACTION)[timestamps.hour]
    )
  else:
    grouped_office = np.zeros((24, count))
    recirculation = np.full(
        len(timestamps), native.hvac.air_handler.recirculation
    )

  forcing_table = np.zeros((len(timestamps), count + mass_count))
  comfort = np.empty(len(timestamps), dtype=bool)
  for step, stamp in enumerate(timestamps):
    ambient = float(outdoor[step])
    forcing = geometry.forcing(
        weather.get_exterior_radiation(stamp, ambient),
        poa_irradiance=weather.facade_flux(stamp, geometry.azimuths),
    )
    forcing_table[step, :count] = (
        static['ambient'] * ambient
        + grouped_office[stamp.hour]
    )
    forcing_table[step] += (
        forcing.shortwave_air_gain
        + forcing.mass_gain
        + forcing.emission * forcing.radiant_temperature**4
    )
    comfort[step] = native.hvac.is_comfort_mode(stamp)

  # Solve M against all fixed RHS channels in one pass. The device kernel
  # evaluates the same equation as the native sparse solver:
  # M T_new = C T_previous - R (T_previous**4 - slope * T_previous) + Q.
  inputs = np.column_stack(
      (np.diag(storage), -radiation.toarray(), hvac_projection)
  )
  transition = np.ascontiguousarray(factor.solve(inputs).T)
  sources = np.ascontiguousarray(factor.solve(forcing_table.T).T)
  initial = (
      np.bincount(groups, weights=building.temp.ravel()[active])
      / static['sizes']
  )
  if mass_count:
    initial = np.concatenate((initial, initial[zone_nodes]))
  grid_weights = np.pad(static['sizes'] / building.temp.size, (0, mass_count))
  air_handler = native.hvac.air_handler
  operators = _Operators(
      transition,
      sources,
      outdoor,
      comfort,
      np.asarray(zone_nodes, dtype=np.int32),
      grid_weights,
      1 - len(active) / building.temp.size,
      recirculation,
      np.array(
          [terminal.max_air_flow_rate for terminal in native.hvac.vavs.values()]
      ),
      np.array([
          terminal._reheat_max_water_flow_rate
          for terminal in native.hvac.vavs.values()
      ]),
      native.hvac.boiler.reheat_water_setpoint,
      air_handler.heating_air_temp_setpoint,
      air_handler.cooling_air_temp_setpoint,
  )
  return operators, initial, float(building.temp.ravel()[0])


def _temperature_update(initial_temperature, hvac_heat, source, transition):
  correction = (
      initial_temperature**4 - coupled_radiation.SLOPE * initial_temperature
  )
  channels = jnp.concatenate(
      (initial_temperature, correction, hvac_heat), axis=-1
  )
  return channels @ transition + source


def _nonlinear_temperature_update(previous, hvac_heat, source, transition):
  """Solve the nonlinear balance with the same fixed linear preconditioner."""
  first = _temperature_update(previous, hvac_heat, source, transition)
  nodes = previous.shape[-1]
  radiation_response = transition[nodes : 2 * nodes]

  def fixed_point(temperature):
    # This is (T**4 - slope*T) - (previous**4 - slope*previous),
    # factored to avoid subtracting large fourth-power terms.
    change = (temperature - previous) * (
        (temperature + previous) * (temperature**2 + previous**2)
        - coupled_radiation.SLOPE
    )
    return first + change @ radiation_response

  def residual(current, candidate):
    # F(T)-T is the heat-balance residual preconditioned by the fixed LU,
    # so its infinity norm measures the remaining correction in kelvin.
    return jnp.max(jnp.abs(candidate - current))

  def needs_iteration(state):
    _, _, count, error = state
    return (
        (count < _NONLINEAR_LIMIT)
        & jnp.isfinite(error)
        & (error > _NONLINEAR_TOLERANCE_K)
    )

  def iterate(state):
    _, current, count, _ = state
    candidate = fixed_point(current)
    return current, candidate, count + 1, residual(current, candidate)

  candidate = fixed_point(first)
  initial = (first, candidate, jnp.int32(2), residual(first, candidate))
  temperature, _, count, error = jax.lax.while_loop(
      needs_iteration, iterate, initial
  )
  converged = jnp.isfinite(error) & (error <= _NONLINEAR_TOLERANCE_K)
  return temperature, count, converged


def _observation(zone_temp, outdoor):
  all_temps = jnp.concatenate(
      (
          zone_temp,
          jnp.broadcast_to(outdoor, (*zone_temp.shape[:-1], 1)),
      ),
      axis=-1,
  )
  return ((all_temps - _KELVIN - 22.5) / 10).astype(jnp.float32)


def _advance(state, actions, operators, nonlinear_stats=None):
  index = state.step
  zone_temp = state.temperature[:, operators.zone_nodes]
  action = (
      jnp.clip(actions, -1, 1)
      .astype(jnp.float64)
      .reshape(zone_temp.shape[0], zone_temp.shape[1], 2)
  )
  heat = 20.5 + 1.5 * action[..., 0] + _KELVIN
  cool = 24.5 + 1.5 * action[..., 1] + _KELVIN
  mid = 0.5 * (cool - heat) + heat
  heat_now = (zone_temp < heat) | ((zone_temp < mid) & (state.mode == _HEAT))
  cool_now = (zone_temp > cool) | ((zone_temp > mid) & (state.mode == _COOL))
  normal_mode = jnp.where(heat_now, _HEAT, jnp.where(cool_now, _COOL, _OFF))
  # Eco transition takes priority over the normal hysteresis for this step
  # unless the zone is already above the eco cooling setpoint.
  entered_eco = (index > 0) & operators.comfort[jnp.maximum(index - 1, 0)]
  passive = (
      ~operators.comfort[index]
      & (zone_temp <= cool)
      & (entered_eco | ((state.mode == _PASSIVE) & (zone_temp > heat)))
  )
  mode = jnp.where(passive, _PASSIVE, normal_mode).astype(jnp.int32)

  outdoor = operators.outdoor[index]
  recirculated = (
      state.temperature @ operators.grid_weights
      + operators.outdoor_weight * state.previous_outdoor
  )
  recirculation = operators.recirculation[index]
  mixed_air = recirculation * recirculated + (1 - recirculation) * outdoor
  supply = jnp.clip(
      mixed_air, operators.heating_supply, operators.cooling_supply
  )[:, None]
  flow = operators.air_flow * jnp.where(
      (mode == _HEAT) | (mode == _COOL), 1.0, 0.1
  )
  water = operators.water_flow * (mode == _HEAT)
  water_capacity_rate = constants.WATER_HEAT_CAPACITY * water
  air_capacity_rate = constants.AIR_HEAT_CAPACITY * flow
  zone_supply = (
      (
          supply * (air_capacity_rate - water_capacity_rate)
          + operators.water_temp * constants.WATER_HEAT_CAPACITY * water
      )
      / flow
      / constants.AIR_HEAT_CAPACITY
  )
  hvac_heat = air_capacity_rate * (zone_supply - zone_temp)

  # The VAV output computed from this step's action is applied next step.
  thermal_inputs = (
      state.temperature,
      state.hvac_heat,
      operators.sources[index],
      operators.transition,
  )
  if nonlinear_stats is None:
    temperature = _temperature_update(*thermal_inputs)
  else:
    temperature, count, converged = _nonlinear_temperature_update(
        *thermal_inputs
    )
    nonlinear_stats = _NonlinearStats(
        jnp.maximum(nonlinear_stats.worst_iterations, count),
        nonlinear_stats.converged & converged,
    )
    # Surface failures to on-device callers without forcing a transfer every step.
    temperature = jnp.where(converged, temperature, jnp.nan)
  updated_zone_temp = temperature[:, operators.zone_nodes]
  penalty = jnp.maximum(
      settings.HEATING_SETPOINT_C + _KELVIN - updated_zone_temp, 0
  )
  penalty += jnp.maximum(
      updated_zone_temp - (_KELVIN + settings.COOLING_SETPOINT_C), 0
  )
  reward = (-jnp.mean(penalty, axis=-1)).astype(jnp.float32)
  result = (
      _State(temperature, hvac_heat, mode, outdoor, index + 1),
      _observation(updated_zone_temp, outdoor),
      reward,
  )
  return result if nonlinear_stats is None else (*result, nonlinear_stats)


class JaxOffice:
  """Batched office environments sharing one immutable weather/model sequence.

  Reset is deterministic. Actions are zone-major normalized heating/cooling
  pairs. The caller controls the episode horizon and resets all lanes together.
  Set nonlinear=True to iterate current-step longwave exchange. A failed
  nonlinear solve produces NaN observations and sets nonlinear_converged=False.
  Pass an InteriorMass configuration to add one effective mass state per room.
  """

  def __init__(
      self,
      model_dir,
      epw,
      num_envs,
      episode_days,
      *,
      platform='cpu',
      nonlinear=False,
      mass_config=None,
      solar_to_interior_mass=True,
  ):
    if platform not in ('cpu', 'gpu'):
      raise ValueError("platform must be 'cpu' or 'gpu'")
    for name, value in (('num_envs', num_envs), ('episode_days', episode_days)):
      if (
          isinstance(value, bool)
          or not isinstance(value, Integral)
          or value < 1
      ):
        raise ValueError(f'{name} must be a positive integer')
    start = perf_counter()
    jax.config.update('jax_enable_x64', True)
    try:
      self.device = jax.devices(platform)[0]
    except (RuntimeError, IndexError) as error:
      raise RuntimeError(
          f'JAX {platform.upper()} backend is unavailable: {error}'
      ) from error
    self.num_envs = int(num_envs)
    self.episode_steps = int(episode_days) * 24 * settings.STEPS_PER_HOUR
    operators, initial, initial_outdoor = _build_operators(
        Path(model_dir),
        Path(epw),
        int(episode_days),
        mass_config,
        solar_to_interior_mass=solar_to_interior_mass,
    )
    zone_count = len(operators.zone_nodes)
    self.observation_size = zone_count + 1
    self.action_size = 2 * zone_count
    self.thermal_nodes = initial.size
    self.interior_mass_nodes = zone_count if mass_config is not None else 0
    self._operators = jax.device_put(operators, self.device)
    self._initial_state = jax.device_put(
        _State(
            np.broadcast_to(
                initial, (self.num_envs, self.thermal_nodes)
            ).copy(),
            np.zeros((self.num_envs, zone_count)),
            np.zeros((self.num_envs, zone_count), dtype=np.int32),
            np.array(initial_outdoor),
            np.array(0, dtype=np.int32),
        ),
        self.device,
    )
    self._initial_observation = jax.device_put(
        np.broadcast_to(
            (
                (
                    np.concatenate(
                        (initial[operators.zone_nodes], operators.outdoor[:1])
                    )
                    - _KELVIN
                    - 22.5
                )
                / 10
            ).astype(np.float32),
            (self.num_envs, self.observation_size),
        ).copy(),
        self.device,
    )
    self.initialization_s = perf_counter() - start
    start = perf_counter()
    input_example = jax.device_put(
        np.zeros((self.num_envs, self.action_size), dtype=np.float32),
        self.device,
    )
    self._initial_nonlinear_stats = None
    arguments = (self._initial_state, input_example, self._operators)
    if nonlinear:
      self._initial_nonlinear_stats = jax.device_put(
          _NonlinearStats(np.int32(0), np.bool_(True)), self.device
      )
      arguments += (self._initial_nonlinear_stats,)
    self._advance = jax.jit(_advance).lower(*arguments).compile()
    self.compilation_s = perf_counter() - start
    self._nonlinear_stats = self._initial_nonlinear_stats
    self._state = None
    self._steps = 0

  def reset(self, seed: int):
    """Return batched initial observations on the selected JAX device."""
    del seed  # The initial conditions and weather are deterministic.
    self._state = self._initial_state
    self._nonlinear_stats = self._initial_nonlinear_stats
    self._steps = 0
    return self._initial_observation

  def step(self, actions):
    """Advance all lanes and return observations and rewards on-device."""
    if self._state is None or self._steps >= self.episode_steps:
      raise RuntimeError('Reset JaxOffice before starting a new episode')
    if not hasattr(actions, 'shape'):
      actions = np.asarray(actions, dtype=np.float32)
    actions = jax.device_put(actions, self.device)
    if actions.shape != (self.num_envs, self.action_size):
      raise ValueError(
          f'Expected actions of shape {(self.num_envs, self.action_size)}, got'
          f' {actions.shape}'
      )
    arguments = (self._state, actions.astype(jnp.float32), self._operators)
    if self._nonlinear_stats is None:
      self._state, observations, rewards = self._advance(*arguments)
    else:
      self._state, observations, rewards, self._nonlinear_stats = self._advance(
          *arguments, self._nonlinear_stats
      )
    self._steps += 1
    return observations, rewards

  @property
  def interior_mass_temperature_c(self):
    """Return mass temperatures in room order for each initialized replica."""
    if self._state is None or not self.interior_mass_nodes:
      return None
    return self._state.temperature[:, -self.interior_mass_nodes:] - _KELVIN

  @property
  def worst_nonlinear_iterations(self):
    """Maximum evaluations of the fixed point on a step in this episode."""
    if self._nonlinear_stats is None:
      return None
    return int(self._nonlinear_stats.worst_iterations)

  @property
  def nonlinear_converged(self):
    """Whether every nonlinear step met the temperature-correction tolerance."""
    if self._nonlinear_stats is None:
      return None
    return bool(self._nonlinear_stats.converged)

  def close(self):
    """Match the shared driver lifecycle."""
