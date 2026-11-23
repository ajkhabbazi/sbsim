"""Tests for the exterior radiation energy balance across solver backends."""

import io

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from smart_control.simulator import building
from smart_control.simulator import cached_lu_simulator
from smart_control.simulator import jax_device_resident_simulator
from smart_control.simulator import jax_simulator
from smart_control.simulator import numpy_scipy_simulator
from smart_control.simulator import simulator_flexible_floor_plan
from smart_control.simulator import simulator_flexible_floor_plan_test
from smart_control.simulator import solar_radiation
from smart_control.simulator import tf_coupled_interior_mass_simulator
from smart_control.simulator import tf_device_resident_simulator
from smart_control.simulator import tf_simulator
from smart_control.simulator import weather_controller

_TIMESTAMP = pd.Timestamp('2025-06-21 08:00:00')
_AMBIENT = 300.0
_SKY = 275.0


def _net_air_power(forcing, temperature):
  return forcing.linearized_air_terms(temperature, temperature)[1]


class FixedRadiationWeather(weather_controller.WeatherController):
  """Weather with an east-facing beam and an independently specified sky."""

  def __init__(self, *, sky=_SKY, dni=200.0, ground=None):
    super().__init__(default_low_temp=_AMBIENT, default_high_temp=_AMBIENT)
    self.sky = sky
    self.dni = dni
    self.ground = ground
    self.radiation_calls = 0

  def get_exterior_radiation(self, timestamp, ambient_temperature):
    self.radiation_calls += 1
    return solar_radiation.ExteriorRadiationData(
        timestamp=timestamp,
        ambient_temp_k=ambient_temperature,
        sky_temp_k=self.sky,
        ground_temp_k=self.ground,
        irradiance=solar_radiation.IrradianceComponents(
            ghi=0.0,
            dni=self.dni,
            dhi=0.0,
            solar_zenith=60.0,
            solar_azimuth=90.0,
        ),
    )


class ExteriorRadiationTest(parameterized.TestCase):

  def _make_simulator(
      self,
      simulator_class,
      *,
      with_mass=False,
      weather=None,
      enable_radiation=True,
      orientation=0.0,
      floor_plan=None,
      exterior_material=None,
      timestep=300.0,
      with_interior_radiation=False,
      **kwargs,
  ):
    if floor_plan is None:
      floor_plan = np.full((7, 7), 2)
      floor_plan[1:6, 1:6] = 1
      floor_plan[2:5, 2:5] = 0
      floor_plan[3, 4:6] = 4
    floor = building.FloorPlanBasedBuilding(
        cv_size_cm=20.0,
        floor_height_cm=300.0,
        initial_temp=_AMBIENT,
        inside_air_properties=building.MaterialProperties(0.2, 700.0, 1.2),
        building_exterior_properties=exterior_material,
        floor_plan=floor_plan,
        zone_map=np.where(floor_plan == 4, 1, floor_plan),
        buffer_from_walls=0,
        include_exterior_radiative_heat_transfer=enable_radiation,
        include_radiative_heat_transfer=with_interior_radiation,
        floor_plan_orientation=orientation,
        include_interior_mass=with_mass,
        interior_mass_properties=building.MaterialProperties(
            conductivity=5.0, heat_capacity=300.0, density=2000.0
        )
        if with_mass
        else None,
    )
    fixture = (
        simulator_flexible_floor_plan_test.FlexibleFloorplanSimulatorTest()
    )
    return simulator_class(
        floor,
        fixture._create_small_hvac(),
        weather or FixedRadiationWeather(),
        timestep,
        1e-4,
        2000,
        2000,
        _TIMESTAMP,
        relative_convergence_threshold=None,
        **kwargs,
    )

  def test_corner_uses_each_faces_area_and_orientation(self):
    weather = FixedRadiationWeather(sky=_AMBIENT)
    sim = self._make_simulator(
        simulator_flexible_floor_plan.SimulatorFlexibleGeometries,
        weather=weather,
    )
    forcing = sim._prepare_exterior_radiation(_AMBIENT)
    power = _net_air_power(forcing, sim.building.temp)
    east_beam = 200.0 * np.sin(np.pi / 3)
    # 0.3 m² for each corner face and 0.6 m² for an edge face.
    self.assertAlmostEqual(power[1, 5], 0.65 * 0.3 * east_beam)
    self.assertAlmostEqual(power[2, 5], 0.65 * 0.6 * east_beam)
    self.assertAlmostEqual(power[1, 3], 0.0)
    self.assertAlmostEqual(power[3, 5], 0.1 * 0.6 * east_beam)

  def test_orientation_rotates_which_facade_receives_beam(self):
    sim = self._make_simulator(
        simulator_flexible_floor_plan.SimulatorFlexibleGeometries,
        weather=FixedRadiationWeather(sky=_AMBIENT),
        orientation=90.0,
    )
    power = _net_air_power(
        sim._prepare_exterior_radiation(_AMBIENT), sim.building.temp
    )
    self.assertGreater(power[1, 3], 60.0)
    self.assertAlmostEqual(power[2, 5], 0.0)

  @parameterized.parameters(None, 295.0)
  def test_longwave_matches_fourth_power_law(self, ground):
    sim = self._make_simulator(
        simulator_flexible_floor_plan.SimulatorFlexibleGeometries,
        weather=FixedRadiationWeather(dni=0.0, ground=ground),
    )
    forcing = sim._prepare_exterior_radiation(_AMBIENT)
    surface_temperature = sim.building.temp.copy()
    surface_temperature[1, 3] = 305.0
    ground_temp = _AMBIENT if ground is None else ground
    expected = (
        0.6
        * 0.93
        * 5.67e-8
        * (
            (_SKY**4 - 305.0**4) / np.sqrt(8.0)
            + (ground_temp**4 - 305.0**4) * 0.5
            + (_AMBIENT**4 - 305.0**4) * (0.5 - 1.0 / np.sqrt(8.0))
        )
    )
    self.assertAlmostEqual(
        _net_air_power(forcing, surface_temperature)[1, 3], expected
    )

  def test_weather_controller_supplies_sky_and_day_night_irradiance(self):
    controller = weather_controller.WeatherController(_AMBIENT, _AMBIENT)
    noon = pd.Timestamp('2025-06-21 12:00:00', tz='US/Pacific')
    day = controller.get_exterior_radiation(
        noon, controller.get_current_temp(noon)
    )
    midnight = noon.normalize()
    night = controller.get_exterior_radiation(
        midnight, controller.get_current_temp(midnight)
    )
    self.assertLess(day.sky_temp_k, _AMBIENT)
    self.assertGreater(day.irradiance.ghi, 0.0)
    self.assertEqual(night.irradiance.ghi, 0.0)

  def test_replay_uses_observed_dew_point_and_interprets_naive_time_as_utc(
      self,
  ):
    weather_csv = io.StringIO(
        'Time,TempF,DewPointC,Humidity\n'
        '2025-06-21T19:00:00Z,80.33,10,50\n'
        '2025-06-21T21:00:00Z,80.33,20,50\n'
    )
    controller = weather_controller.ReplayWeatherController(weather_csv)
    timestamp = pd.Timestamp('2025-06-21 20:00:00')
    forcing = controller.get_exterior_radiation(timestamp, _AMBIENT)
    expected = _AMBIENT * (0.787 + 0.764 * np.log(288.15 / 273.0)) ** 0.25
    self.assertAlmostEqual(forcing.sky_temp_k, expected)
    self.assertEqual(forcing.timestamp, timestamp.tz_localize('UTC'))

  def test_replay_uses_observed_cloud_cover_for_solar_irradiance(self):
    timestamp = pd.Timestamp('2025-06-21 20:00:00')

    def ghi(cloud_cover):
      weather_csv = io.StringIO(
          'Time,TempF,Humidity,SkyCoverage\n'
          f'2025-06-21T20:00:00Z,80.33,50,{cloud_cover}\n'
      )
      replay = weather_controller.ReplayWeatherController(weather_csv)
      return replay.get_exterior_radiation(timestamp, _AMBIENT).irradiance.ghi

    self.assertGreater(ghi(0), 0.0)
    self.assertAlmostEqual(ghi(100), 0.2 * ghi(0))

  def test_transmitting_window_requires_an_indoor_destination(self):
    floor_plan = np.full((7, 7), 2)
    floor_plan[1:6, 1:6] = 1
    floor_plan[2:5, 2:5] = 0
    floor_plan[3, 4] = 1  # An opaque wall separates the window from the room.
    floor_plan[3, 5] = 4
    sim = self._make_simulator(
        simulator_flexible_floor_plan.SimulatorFlexibleGeometries,
        floor_plan=floor_plan,
    )
    with self.assertRaisesRegex(ValueError, 'glazing.*indoor air'):
      sim.finite_differences_timestep(
          ambient_temperature=_AMBIENT, convection_coefficient=12.0
      )

  def test_window_cells_stay_aligned_when_floor_plan_needs_padding(self):
    solver = simulator_flexible_floor_plan.SimulatorFlexibleGeometries
    padded = self._make_simulator(solver)
    unpadded = self._make_simulator(
        solver, floor_plan=padded.building.floor_plan[1:-1, 1:-1]
    )
    np.testing.assert_array_equal(
        unpadded.building.floor_plan, padded.building.floor_plan
    )
    np.testing.assert_array_equal(
        unpadded._prepare_exterior_radiation(_AMBIENT).shortwave_air_gain,
        padded._prepare_exterior_radiation(_AMBIENT).shortwave_air_gain,
    )

  def test_window_shared_by_air_regions_transmits_its_power_once(self):
    floor_plan = np.array([
        [2, 2, 2, 2, 2, 2, 2],
        [2, 1, 4, 4, 4, 1, 2],
        [2, 1, 0, 1, 0, 1, 2],
        [2, 1, 0, 1, 0, 1, 2],
        [2, 1, 1, 1, 1, 1, 2],
        [2, 2, 2, 2, 2, 2, 2],
    ])
    sim = self._make_simulator(
        simulator_flexible_floor_plan.SimulatorFlexibleGeometries,
        floor_plan=floor_plan,
        orientation=90.0,
        weather=FixedRadiationWeather(sky=_AMBIENT),
    )
    forcing = sim._prepare_exterior_radiation(_AMBIENT)
    expected = 3 * 0.8 * 0.6 * 200.0 * np.sin(np.pi / 3)
    np.testing.assert_allclose(
        _net_air_power(forcing, sim.building.temp)[floor_plan == 0],
        expected / 4,
    )

  @parameterized.parameters(False, True)
  def test_window_transmitted_power_goes_to_air_or_mass(self, with_mass):
    sim = self._make_simulator(
        simulator_flexible_floor_plan.SimulatorFlexibleGeometries,
        with_mass=with_mass,
        weather=FixedRadiationWeather(sky=_AMBIENT),
    )
    forcing = sim._prepare_exterior_radiation(_AMBIENT)
    receiving_air = sim.building.floor_plan == 0
    expected_watts = 0.8 * 0.6 * 200.0 * np.sin(np.pi / 3)
    air_power = _net_air_power(forcing, sim.building.temp)
    self.assertAlmostEqual(
        air_power[receiving_air].sum(), 0.0 if with_mass else expected_watts
    )
    self.assertAlmostEqual(
        forcing.mass_gain.sum(), expected_watts if with_mass else 0.0
    )
    self.assertAlmostEqual(air_power[3, 4], 0.0)

  def test_disabled_does_not_request_radiation(self):
    weather = FixedRadiationWeather()
    sim = self._make_simulator(
        jax_device_resident_simulator.JaxDeviceResidentSimulator,
        enable_radiation=False,
        weather=weather,
    )
    self.assertTrue(
        sim.finite_differences_timestep(
            ambient_temperature=_AMBIENT,
            convection_coefficient=12.0,
        )
    )
    self.assertEqual(weather.radiation_calls, 0)
    np.testing.assert_allclose(sim.building.temp, _AMBIENT)

  def test_cold_sky_converges_for_lightweight_insulated_wall(self):
    sim = self._make_simulator(
        jax_device_resident_simulator.JaxDeviceResidentSimulator,
        weather=FixedRadiationWeather(sky=250.0, dni=0.0),
        exterior_material=building.MaterialProperties(0.015, 1000.0, 30.0),
        timestep=3600.0,
    )
    self.assertTrue(
        sim.finite_differences_timestep(
            ambient_temperature=_AMBIENT,
            convection_coefficient=12.0,
        )
    )
    self.assertTrue(np.isfinite(sim.building.temp).all())
    self.assertBetween(sim.building.temp[1, 3], 250.0, _AMBIENT)

  def test_jax_resident_combines_exterior_and_interior_radiation(self):
    options = dict(with_mass=True, with_interior_radiation=True)
    reference = self._make_simulator(jax_simulator.JaxSimulator, **options)
    resident = self._make_simulator(
        jax_device_resident_simulator.JaxDeviceResidentSimulator,
        sync_to_host_each_step=False,
        **options,
    )
    initial = resident.building.temp.copy()
    for sim in (reference, resident):
      self.assertTrue(
          sim.finite_differences_timestep(
              ambient_temperature=_AMBIENT,
              convection_coefficient=12.0,
          )
      )
    np.testing.assert_array_equal(resident.building.temp, initial)
    resident.sync_state_to_building()
    np.testing.assert_allclose(
        resident.building.temp, reference.building.temp, atol=1e-4, rtol=0.0
    )
    np.testing.assert_allclose(
        resident.building.interior_mass_temp,
        reference.building.interior_mass_temp,
        atol=1e-4,
        rtol=0.0,
    )

  @parameterized.named_parameters(
      ('tf_fdm', tf_simulator.TFSimulator, {}, False),
      ('tf_cg_mass', tf_simulator.TFSimulator, {'solver_method': 'cg'}, True),
      ('jax_sparse', jax_simulator.JaxSimulator, {}, False),
      (
          'jax_cg_mass',
          jax_simulator.JaxSimulator,
          {'solver_method': 'cg'},
          True,
      ),
      (
          'jax_resident',
          jax_device_resident_simulator.JaxDeviceResidentSimulator,
          {},
          False,
      ),
      (
          'jax_resident_mass',
          jax_device_resident_simulator.JaxDeviceResidentSimulator,
          {},
          True,
      ),
      (
          'tf_resident',
          tf_device_resident_simulator.TFDeviceResidentSimulator,
          {},
          True,
      ),
      (
          'tf_coupled',
          tf_coupled_interior_mass_simulator.TFCoupledInteriorMassSimulator,
          {},
          True,
      ),
      ('scipy', numpy_scipy_simulator.NumpyScipySimulator, {}, False),
      ('cached_lu', cached_lu_simulator.CachedLUSimulator, {}, False),
      (
          'scipy_coupled',
          (
              tf_coupled_interior_mass_simulator.NumpyScipyCoupledInteriorMassSimulator  # pylint: disable=line-too-long
          ),
          {},
          True,
      ),
  )
  def test_solvers_match_scalar_energy_balance(self, solver, kwargs, with_mass):
    sims = [
        self._make_simulator(solver_class, with_mass=with_mass, **options)
        for solver_class, options in (
            (simulator_flexible_floor_plan.SimulatorFlexibleGeometries, {}),
            (solver, kwargs),
        )
    ]
    for sim in sims:
      for _ in range(2):
        self.assertTrue(
            sim.finite_differences_timestep(
                ambient_temperature=_AMBIENT,
                convection_coefficient=12.0,
            )
        )
      self.assertEqual(sim._weather_controller.radiation_calls, 2)
      self.assertLess(sim.building.temp[1, 3], _AMBIENT - 0.01)
      self.assertGreater(sim.building.temp[2, 5], _AMBIENT + 0.1)
      np.testing.assert_array_equal(sim.building.input_q, 0.0)
      if with_mass:
        self.assertGreater(sim.building.interior_mass_temp[3, 3], _AMBIENT)
    np.testing.assert_allclose(
        sims[0].building.temp,
        sims[1].building.temp,
        atol=3e-4,
        rtol=0.0,
    )
    if with_mass:
      np.testing.assert_allclose(
          sims[0].building.interior_mass_temp,
          sims[1].building.interior_mass_temp,
          atol=1e-4,
          rtol=0.0,
      )


if __name__ == '__main__':
  absltest.main()
