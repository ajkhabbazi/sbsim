"""Create a native SBSim HVAC model and read its weather from an EPW file."""

from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from pvlib import irradiance
from pvlib.iotools import read_epw
from pvlib.solarposition import get_solarposition
from scipy.sparse import linalg as sparse_linalg

from smart_control.simulator import air_handler
from smart_control.simulator import boiler
from smart_control.simulator import building
from smart_control.simulator import cached_lu_simulator
from smart_control.simulator import hvac_floorplan_based
from smart_control.simulator import jax_device_resident_simulator
from smart_control.simulator import setpoint_schedule
from smart_control.simulator import solar_radiation
from smart_control.simulator import vav
from smart_control.simulator import weather_controller

from . import office_loads
from . import coupled_radiation
from . import reduced_air
from . import reduced_envelope
from . import settings
from .floorplan import FloorPlan

KELVIN_OFFSET = 273.15
OUTER_TOLERANCE = 1e-4
EXTERIOR_RADIATION_MODES = ('full', 'linear-longwave', 'solar-only', 'off')
LINEAR_LONGWAVE_REFERENCE_K = 22.5 + KELVIN_OFFSET


class EpwWeather(weather_controller.BaseWeatherController):
  """Prepare five-minute SBSim weather over a local-standard-time period."""

  def __init__(self, epw: Path, run_day: date, days: int = 1):
    data, site = read_epw(epw, coerce_year=run_day.year)
    if not data.index.is_unique or not data.index.is_monotonic_increasing:
      raise ValueError('EPW needs chronological hourly records')
    self.start = pd.Timestamp(run_day, tz=data.index.tz)
    ticks = pd.date_range(
        self.start,
        periods=days * 24 * settings.STEPS_PER_HOUR,
        freq=f'{settings.STEP_SECONDS}s',
    )
    if len(data) in (8760, 8784):
      # A TMY file is cyclic, including the interpolation across Jan 1.
      end = self.start + pd.Timedelta(days=days)
      starts_on_jan_1 = self.start.month == 1 and self.start.day == 1
      if starts_on_jan_1 or end.year > run_day.year:
        prev = data.iloc[[-1]].set_axis(
            data.index[[-1]] - pd.DateOffset(years=1)
        )
        parts = [prev, data]
        for offset in range(1, end.year - run_day.year + 1):
          parts.append(data.set_axis(data.index + pd.DateOffset(years=offset)))
        data = pd.concat(parts)
        data = data.loc[~data.index.duplicated()]
    required_hours = pd.date_range(self.start, periods=days * 24, freq='h')
    if not required_hours.isin(data.index).all():
      raise ValueError(
          f'EPW needs all hourly records for {days} days from {run_day}'
      )

    step = pd.Timedelta(seconds=settings.STEP_SECONDS)

    def sample(hourly, source_offset, target_offset):
      return np.interp(
          (ticks + target_offset).asi8,
          (hourly.index + source_offset).asi8,
          hourly.to_numpy(),
          left=np.nan,
          right=np.nan,
      )

    solar = data[['ghi', 'dni', 'dhi', 'ghi_infrared']]
    solar = solar.where((solar >= 0) & (solar < 9999))
    radiation = pd.DataFrame(
        {
            column: sample(solar[column], pd.Timedelta(minutes=30), step / 2)
            for column in solar
        },
        index=ticks,
    )
    dry_bulb = data['temp_air']
    dry_bulb = dry_bulb.where((dry_bulb > -100) & (dry_bulb < 99.9))
    temp_c = sample(dry_bulb, pd.Timedelta(hours=1), step)
    if (
        not np.isfinite(radiation.to_numpy()).all()
        or not np.isfinite(temp_c).all()
    ):
      raise ValueError(f'EPW has missing or invalid samples near {run_day}')

    sun = get_solarposition(
        ticks + step / 2,
        site['latitude'],
        site['longitude'],
        altitude=site['altitude'],
    )
    self._data = radiation.assign(
        temp_k=temp_c + KELVIN_OFFSET,
        sky_k=(radiation['ghi_infrared'] / 5.670374419e-8) ** 0.25,
        zenith=sun['apparent_zenith'].to_numpy(),
        azimuth=sun['azimuth'].to_numpy(),
    )
    self._facade_flux = {}

  def facade_flux(self, timestamp, azimuths):
    """Use the same pvlib irradiance model for an entire weather series."""
    if azimuths not in self._facade_flux:
      data = self._data
      self._facade_flux[azimuths] = np.column_stack([
          irradiance.get_total_irradiance(
              surface_tilt=90.0, surface_azimuth=azimuth,
              dni=data.dni.to_numpy(), ghi=data.ghi.to_numpy(),
              dhi=data.dhi.to_numpy(), solar_zenith=data.zenith.to_numpy(),
              solar_azimuth=data.azimuth.to_numpy(),
          )['poa_global']
          for azimuth in azimuths
      ])
    index = self._data.index.get_loc(timestamp)
    return tuple(self._facade_flux[azimuths][index])

  def get_current_temp(self, timestamp: pd.Timestamp) -> float:
    return float(self._data.at[timestamp, 'temp_k'])

  def get_air_convection_coefficient(self, timestamp: pd.Timestamp) -> float:
    return 12.0

  def get_exterior_radiation(self, timestamp, ambient_temperature):
    row = self._data.loc[timestamp]
    return solar_radiation.ExteriorRadiationData(
        timestamp=timestamp,
        ambient_temp_k=ambient_temperature,
        sky_temp_k=float(row.sky_k),
        irradiance=solar_radiation.IrradianceComponents(
            ghi=float(row.ghi),
            dni=float(row.dni),
            dhi=float(row.dhi),
            solar_zenith=float(row.zenith),
            solar_azimuth=float(row.azimuth),
            timestamp=timestamp,
        ),
    )


class _OfficeForcing:
  """Apply the scheduled sensible gains and outdoor-air ratio at each step."""

  def __init__(self, *, office_heat: np.ndarray, **kwargs):
    super().__init__(**kwargs)
    self._office_heat = office_heat
    self.office_heat_w = office_heat.sum(axis=(1, 2))

  def execute_step_sim(self, video_filename='sample.mp4'):
    hour = self.current_timestamp.hour
    self.hvac.air_handler.recirculation = (
        1.0 - office_loads.FRESH_AIR_FRACTION[hour]
    )
    super().execute_step_sim(video_filename)

  def finite_differences_timestep(
      self, *, ambient_temperature, convection_coefficient
  ):
    # Keep input_q as the native HVAC forcing outside the heat-balance call.
    building = self.building
    hvac_heat = building.input_q
    building.input_q = (
        hvac_heat + self._office_heat[self.current_timestamp.hour]
    )
    try:
      return super().finite_differences_timestep(
          ambient_temperature=ambient_temperature,
          convection_coefficient=convection_coefficient,
      )
    finally:
      building.input_q = hvac_heat


class OfficeJaxSimulator(
    _OfficeForcing, jax_device_resident_simulator.JaxDeviceResidentSimulator
):
  pass


class OfficeCachedLUSimulator(
    _OfficeForcing, cached_lu_simulator.CachedLUSimulator
):
  pass


class _SharedSetpointSchedule(setpoint_schedule.SetpointSchedule):
  """Share calendar lookups across VAVs while reading live setpoints."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._calendar_zone = self._time_zone
    self._calendar_cache = []

  def _calendar_fields(self, timestamp):
    if self._calendar_zone is not self._time_zone:
      self._calendar_zone = self._time_zone
      self._calendar_cache.clear()
    for cached_timestamp, fields in self._calendar_cache:
      if cached_timestamp is timestamp:
        return fields
    local = self._localize_or_convert_timezone(timestamp)
    fields = local.hour, local.dayofyear, super().is_weekend(local)
    # All VAVs share the current timestamp and may also check the previous one.
    self._calendar_cache.insert(0, (timestamp, fields))
    del self._calendar_cache[2:]
    return fields

  def is_comfort_mode(self, current_timestamp):
    hour, day, weekend = self._calendar_fields(current_timestamp)
    return (
        self.morning_start_hour <= hour < self.evening_start_hour
        and day not in self.holidays
        and not weekend
    )


class FastLUSimulator(cached_lu_simulator.CachedLUSimulator):
  """Keep the full grid and reuse its CSR sparsity during radiation updates."""

  def __init__(self, *args, exterior_radiation='full', **kwargs):
    if exterior_radiation not in EXTERIOR_RADIATION_MODES:
      raise ValueError(f'Unknown exterior radiation mode: {exterior_radiation}')
    super().__init__(*args, **kwargs)
    self._exterior_radiation_mode = exterior_radiation
    self._linear_longwave_conductance = None
    if exterior_radiation != 'full' and (
        self.building.include_radiative_heat_transfer
        or self.building.include_interior_mass
    ):
      raise ValueError('Radiation ablations require linear interior heat balance')
    if exterior_radiation in ('solar-only', 'linear-longwave') and (
        not self.include_exterior_radiative_heat_transfer
    ):
      raise ValueError('This radiation mode requires exterior geometry')

  def finite_differences_timestep(
      self, *, ambient_temperature, convection_coefficient
  ):
    if self._exterior_radiation_mode == 'full':
      return super().finite_differences_timestep(
          ambient_temperature=ambient_temperature,
          convection_coefficient=convection_coefficient,
      )
    building = self.building
    input_q = building.input_q
    if self._exterior_radiation_mode in ('solar-only', 'linear-longwave'):
      forcing = self._prepare_exterior_radiation(ambient_temperature)
      driving = forcing.shortwave_air_gain
      if self._exterior_radiation_mode == 'linear-longwave':
        if self._linear_longwave_conductance is None:
          self._linear_longwave_conductance = (
              4 * forcing.emission * LINEAR_LONGWAVE_REFERENCE_K**3
          )
          self._lu = None
        previous = np.asarray(building.temp, dtype=np.float64)
        # Retain the exact exchange at the previous surface temperature.
        # The implicit slope stays fixed so its matrix can be factored once.
        exchange = forcing.emission * (
            forcing.radiant_temperature**4 - previous**4
        )
        driving = (
            driving + exchange + self._linear_longwave_conductance * previous
        )
      building.input_q = input_q + driving
    try:
      # The assembled radiation terms are linear in the new temperatures.
      self.last_linear_converged = self._iterate_temperature_timestep(
          ambient_temperature, convection_coefficient
      )
      return self.last_linear_converged
    finally:
      building.input_q = input_q

  def _build_static(self, convection_coefficient):
    static = super()._build_static(convection_coefficient)
    static['timestep_matrix'] = reduced_air.TimestepMatrix(static['a_cond'])
    return static

  def _assemble_timestep_matrix(self, static, radiation_diagonal):
    if self._linear_longwave_conductance is not None:
      if 'linear_longwave_matrix' not in static:
        conductance = self._linear_longwave_conductance.ravel()[static['solved']]
        static['linear_longwave_matrix'] = static['timestep_matrix'].with_radiation(
            conductance
        ).copy()
      return static['linear_longwave_matrix']
    return static['timestep_matrix'].with_radiation(radiation_diagonal)

  def _pcg(self, a, b, x):
    if self._exterior_radiation_mode == 'full':
      return super()._pcg(a, b, x)
    # The cached factorization is for this exact, constant matrix.
    x[:] = self._lu.solve(b)
    residual = np.linalg.norm(a @ x - b)
    if not np.isfinite(residual) or residual > self._pcg_rtol * max(
        np.linalg.norm(b), 1.0
    ):
      raise RuntimeError(f'Linear heat balance did not converge ({residual:g})')
    return 1


class OfficeFastLUSimulator(_OfficeForcing, FastLUSimulator):
  pass


class OptimizedLUSimulator(FastLUSimulator):
  """Keep every CV and evaluate the nonlinear terms only on exposed faces."""

  def _prepare_exterior_radiation(self, ambient_temperature):
    weather = self._weather_controller
    if (
        not self.include_exterior_radiative_heat_transfer
        or type(weather) is not EpwWeather
    ):
      return super()._prepare_exterior_radiation(ambient_temperature)
    if self._exterior_radiation_geometry is None:
      self._exterior_radiation_geometry = self._build_exterior_radiation_geometry()
    geometry = self._exterior_radiation_geometry
    return geometry.forcing(
        weather.get_exterior_radiation(self.current_timestamp, ambient_temperature),
        poa_irradiance=weather.facade_flux(self.current_timestamp, geometry.azimuths),
        window_air_gain=getattr(self, '_window_solar', None),
    )

  def _iterate_temperature_timestep(self, ambient_temperature, convection_coefficient):
    building = self.building
    forcing = self._exterior_forcing
    if (
        forcing is None or building.include_interior_mass
        or building.include_radiative_heat_transfer
        or building.temp.dtype != np.float64
    ):
      return super()._iterate_temperature_timestep(
          ambient_temperature, convection_coefficient
      )
    if self._static is None or self._static_h != convection_coefficient:
      self._static = self._build_static(convection_coefficient)
      self._static_h = convection_coefficient
      self._lu = None
    if 'surface' not in self._static:
      active = self._static['solved']
      emission = forcing.emission.ravel()[active]
      surface = np.flatnonzero(emission)
      outside = np.ones(building.temp.size, dtype=bool)
      outside[active] = False
      self._static.update(
          surface=surface, emission=emission[surface], outside=outside
      )

    s = self._static
    active, surface = s['solved'], s['surface']
    previous = np.asarray(building.temp, dtype=np.float64).ravel()
    estimate = previous[active]
    base_rhs = (
        s['storage'] * estimate + building.input_q.ravel()[active]
        + s['boundary_conductance'] * ambient_temperature
    )
    shortwave = forcing.shortwave_air_gain.ravel()[active]
    rhs = base_rhs + shortwave
    surface_base, surface_solar = base_rhs[surface], shortwave[surface]
    ambient_delta = np.max(
        np.abs(previous[s['outside']] - ambient_temperature), initial=0.0
    )
    radiant = forcing.radiant_temperature
    for step in range(self._iteration_limit):
      surface_temp = estimate[surface]
      radial = s['emission'] * (radiant + surface_temp) * (
          radiant**2 + surface_temp**2
      )
      rhs[surface] = surface_base + (surface_solar + radial * radiant)
      matrix = s['timestep_matrix'].with_surface_radiation(surface, radial)
      if self._lu is None or self._stale:
        self._lu = sparse_linalg.splu(matrix.tocsc(), permc_spec='MMD_AT_PLUS_A')
        self.factorization_count += 1
        self._stale = False
      solution = estimate.copy()
      solves = self._pcg(matrix, rhs, solution)
      self.triangular_solve_count += solves
      self._stale = solves > self._refactor_after
      delta = max(np.max(np.abs(solution - estimate)), ambient_delta)
      ambient_delta = 0.0
      estimate = solution
      if delta <= self._convergence_threshold:
        break
    updated = np.full(previous.shape, ambient_temperature)
    updated[active] = estimate
    building.temp = updated.reshape(building.temp.shape)
    self.last_outer_iteration_count = step + 1
    self.last_outer_max_delta = float(delta)
    return delta <= self._convergence_threshold


class OfficeOptimizedLUSimulator(_OfficeForcing, OptimizedLUSimulator):
  pass


class _WindowSolar:
  """Sum equal-sized window faces in batches and preserve per-cell add order."""

  def __init__(self, targets, transmittance):
    by_count = defaultdict(list)
    for index, (window, _) in enumerate(targets):
      by_count[len(window)].append(index)
    self._windows = [
        (indices, np.stack([targets[i][0] for i in indices]))
        for indices in by_count.values()
    ]
    self._transmittance = transmittance
    self._air_sizes = np.array([len(air) for _, air in targets], dtype=int)
    self._recipients = np.concatenate([air for _, air in targets])
    self._window_for_recipient = np.repeat(
        np.arange(len(targets)), self._air_sizes
    )

  def __call__(self, incident):
    watts = np.empty(len(self._air_sizes))
    flat = incident.ravel()
    for indices, cells in self._windows:
      watts[indices] = flat[cells].sum(axis=1)
    powers = (self._transmittance * watts) / self._air_sizes
    transmitted = np.zeros(incident.shape, dtype=incident.dtype)
    np.add.at(
        transmitted.ravel(), self._recipients,
        powers[self._window_for_recipient],
    )
    return transmitted


class _SharedSupplyVav(vav.Vav):
  def output(self, zone_temp, supply_air_temp):
    self._zone_air_temperature = zone_temp
    zone_supply = self.compute_zone_supply_temp(
        supply_air_temp, self.boiler.reheat_water_setpoint
    )
    return self._heat_from_supply_temp(zone_temp, zone_supply), zone_supply


class _SharedSupplyHvac(hvac_floorplan_based.FloorPlanBasedHvac):
  _vav_type = _SharedSupplyVav


class TunedLUSimulator(OptimizedLUSimulator):
  """Cache the static window routing in the full-grid solver."""

  def _build_exterior_radiation_geometry(self):
    geometry = super()._build_exterior_radiation_geometry()
    self._window_solar = (
        _WindowSolar(geometry.window_targets, geometry.transmittance)
        if type(self._weather_controller) is EpwWeather and geometry.window_targets
        else None
    )
    return geometry


class OfficeTunedLUSimulator(_OfficeForcing, TunedLUSimulator):
  pass


class OfficeReducedAirSimulator(
    _OfficeForcing, reduced_air.ReducedAirSimulator
):
  pass


class OfficeReducedEnvelopeSimulator(
    _OfficeForcing, reduced_envelope.ReducedEnvelopeSimulator
):
  pass


class OfficeCondensedEnvelopeSimulator(
    _OfficeForcing, reduced_envelope.CondensedEnvelopeSimulator
):
  pass


class OfficeCoupledRadiationSimulator(
    _OfficeForcing, coupled_radiation.CoupledRadiationSimulator
):
  pass


class OfficeLinearRadiationSimulator(
    _OfficeForcing, coupled_radiation.LinearCoupledRadiationSimulator
):
  pass


class OfficeLinearRadiationEnvelopeSimulator(
    _OfficeForcing, coupled_radiation.LinearCoupledEnvelopeSimulator
):
  pass


_SOLVERS = {
    'jax': (
        jax_device_resident_simulator.JaxDeviceResidentSimulator,
        OfficeJaxSimulator,
    ),
    'cached-lu': (
        cached_lu_simulator.CachedLUSimulator,
        OfficeCachedLUSimulator,
    ),
    'fast-lu': (FastLUSimulator, OfficeFastLUSimulator),
    'optimized-lu': (OptimizedLUSimulator, OfficeOptimizedLUSimulator),
    'tuned-lu': (TunedLUSimulator, OfficeTunedLUSimulator),
    'reduced-air': (reduced_air.ReducedAirSimulator, OfficeReducedAirSimulator),
    'reduced-envelope': (
        reduced_envelope.ReducedEnvelopeSimulator, OfficeReducedEnvelopeSimulator,
    ),
    'condensed-envelope': (
        reduced_envelope.CondensedEnvelopeSimulator, OfficeCondensedEnvelopeSimulator,
    ),
    'coupled-radiation': (
        coupled_radiation.CoupledRadiationSimulator, OfficeCoupledRadiationSimulator,
    ),
    'linear-radiation': (
        coupled_radiation.LinearCoupledRadiationSimulator, OfficeLinearRadiationSimulator,
    ),
    'linear-radiation-envelope': (
        coupled_radiation.LinearCoupledEnvelopeSimulator, OfficeLinearRadiationEnvelopeSimulator,
    ),
}


def make_simulator(
    plan: FloorPlan,
    weather,
    start: pd.Timestamp,
    *,
    office: bool = False,
    solver: str = 'jax',
    air_block_cells: int | None = None,
    exterior_radiation: str = 'full',
):
  """Build floor-plan HVAC devices and the selected thermal solver."""
  if solver not in _SOLVERS:
    raise ValueError(f'Unknown SBSim solver: {solver}')
  coupled = solver in ('coupled-radiation', 'linear-radiation', 'linear-radiation-envelope')
  grouped = coupled or solver in ('reduced-air', 'reduced-envelope', 'condensed-envelope')
  if air_block_cells is not None and not grouped:
    raise ValueError('air_block_cells requires a grouped solver')
  if exterior_radiation != 'full' and solver not in ('fast-lu', 'tuned-lu'):
    raise ValueError('Exterior radiation ablations require fast-lu or tuned-lu')
  spec = plan.spec
  wall = building.MaterialProperties(
      settings.WALL_CONDUCTIVITY,
      settings.WALL_SPECIFIC_HEAT,
      settings.WALL_DENSITY,
  )
  # SBSim window CVs use a full-height facade area. Use an effective solar
  # transmission over that area to represent the common shorter window.
  glazing = building.RadiationProperties(
      alpha=0.1 * settings.WINDOW_HEIGHT_FRACTION,
      epsilon=0.9,
      tau=settings.WINDOW_SHGC * settings.WINDOW_HEIGHT_FRACTION,
  )
  building_type = (
      building.FloorPlanBasedBuilding
      if solver == 'jax'
      else reduced_air.VectorizedBuilding
  )
  built = building_type(
      cv_size_cm=spec.cv_size_m * 100,
      floor_height_cm=spec.height_m * 100,
      initial_temp=(settings.HEATING_SETPOINT_C + settings.COOLING_SETPOINT_C)
      / 2
      + KELVIN_OFFSET,
      inside_air_properties=building.MaterialProperties(50.0, 1005.0, 1.2),
      inside_wall_properties=wall,
      building_exterior_properties=wall,
      floor_plan=plan.grid,
      zone_map=plan.zone_map,
      buffer_from_walls=0,
      include_exterior_radiative_heat_transfer=True,
      fenestration_radiative_properties=glazing,
  )
  names = [room.name for room in plan.rooms]
  room_cells = built.get_zone_average_temps()
  if list(room_cells) != names:
    raise ValueError(
        f'SBSim room order differs from the floor plan: {list(room_cells)}'
    )
  office_heat = np.zeros((24, *plan.grid.shape)) if office else None
  for room in plan.rooms:
    rows, cols = np.array(built._room_dict[room.name]).T  # pylint: disable=protected-access
    west, south, east, north = room.bounds_m
    cv = spec.cv_size_m
    expected = (
        round(west / cv) + 1,
        round(east / cv),
        plan.grid.shape[0] - 1 - round(north / cv),
        plan.grid.shape[0] - 2 - round(south / cv),
    )
    if (
        (cols.min(), cols.max(), rows.min(), rows.max()) != expected
        or not np.isclose(len(rows) * cv**2, room.area_m2)
        or not np.isclose(built.diffusers[rows, cols].sum(), 1.0)
    ):
      raise ValueError(
          f'SBSim coordinates, area, or diffusers disagree for {room.name}'
      )
    if office_heat is not None:
      watts = np.array(
          [office_loads.sensible_watts(room, hour) for hour in range(24)]
      )
      office_heat[:, rows, cols] = watts[:, None] / len(rows)

  flow = plan.rooms[0].area_m2 * settings.FLOW_PER_M2
  ahu = air_handler.AirHandler(
      recirculation=1.0,
      heating_air_temp_setpoint=settings.SUPPLY_AIR_C - 1 + KELVIN_OFFSET,
      cooling_air_temp_setpoint=settings.SUPPLY_AIR_C + KELVIN_OFFSET,
      fan_differential_pressure=settings.FAN_PRESSURE_PA,
      fan_efficiency=settings.FAN_EFFICIENCY,
      max_air_flow_rate=flow * len(names),
      device_id='matched_vav_ahu',
  )
  heat_source = boiler.Boiler(350.0, 3.0, 0.6, 'matched_vav_boiler')
  bounds = (
      settings.HEATING_SETPOINT_C + KELVIN_OFFSET,
      settings.COOLING_SETPOINT_C + KELVIN_OFFSET,
  )
  schedule_type = (
      _SharedSetpointSchedule
      if solver in ('fast-lu', 'optimized-lu', 'tuned-lu') or grouped
      else setpoint_schedule.SetpointSchedule
  )
  schedule = schedule_type(
      0, 24, bounds, bounds, time_zone=start.tz
  )
  hvac_type = (
      _SharedSupplyHvac
      if solver == 'tuned-lu' or coupled
      else hvac_floorplan_based.FloorPlanBasedHvac
  )
  hvac = hvac_type(
      air_handler=ahu,
      boiler=heat_source,
      schedule=schedule,
      vav_max_air_flow_rate=flow,
      vav_reheat_max_water_flow_rate=0.00036 * plan.rooms[0].area_m2,
      zone_identifier=names,
  )
  simulator = _SOLVERS[solver][int(office)]
  extra = {'office_heat': office_heat} if office else {}
  if solver == 'jax':
    extra['sync_to_host_each_step'] = True
  elif grouped:
    extra['air_block_cells'] = air_block_cells
  elif solver in ('fast-lu', 'tuned-lu'):
    extra['exterior_radiation'] = exterior_radiation
  return simulator(
      building=built,
      hvac=hvac,
      weather_controller=weather,
      time_step_sec=settings.STEP_SECONDS,
      convergence_threshold=OUTER_TOLERANCE,
      iteration_limit=100,
      iteration_warning=100,
      start_timestamp=start,
      relative_convergence_threshold=None,
      **extra,
  )
