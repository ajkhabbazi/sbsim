"""Generate a native EnergyPlus epJSON model with VAV reheat."""

from datetime import date
from math import dist

from . import office_loads
from . import settings
from .floorplan import FloorPlan

VAV = 'AirTerminal:SingleDuct:VAV:Reheat'
ADU = 'ZoneHVAC:AirDistributionUnit'
REHEAT = 'Coil:Heating:Electric'
DX_SYSTEM = 'CoilSystem:Cooling:DX'
DX_COIL = 'Coil:Cooling:DX:SingleSpeed'
FAN = 'Fan:VariableVolume'
THERMOSTAT = 'ThermostatSetpoint:DualSetpoint'

ZONE_TEMP = 'Zone Mean Air Temperature'
DAMPER = 'Zone Air Terminal VAV Damper Position'
NODE_FLOW = 'System Node Standard Density Volume Flow Rate'
COIL_HEAT = 'Heating Coil Heating Rate'
COIL_COOL = 'Cooling Coil Total Cooling Rate'
OFFICE_OUTDOOR_AIR = 'Air System Outdoor Air Flow Fraction'
OFFICE_SENSIBLE_LOADS = (
    'Zone People Sensible Heating Rate',
    'Zone Lights Total Heating Rate',
    'Zone Electric Equipment Total Heating Rate',
)


def _geometry(add, plan):
  height = plan.spec.height_m
  add(
      'GlobalGeometryRules',
      'Geometry',
      starting_vertex_position='UpperLeftCorner',
      vertex_entry_direction='Counterclockwise',
      coordinate_system='World',
  )
  add(
      'Material',
      'Solid Wall',
      roughness='MediumRough',
      thickness=plan.spec.wall_thickness_m,
      conductivity=settings.WALL_CONDUCTIVITY,
      density=settings.WALL_DENSITY,
      specific_heat=settings.WALL_SPECIFIC_HEAT,
  )
  add(
      'Material:NoMass',
      'Slab Resistance',
      roughness='MediumRough',
      thermal_resistance=1.0,
  )
  add('Construction', 'Wall', outside_layer='Solid Wall')
  add('Construction', 'Slab', outside_layer='Slab Resistance')
  if plan.spec.window_width_m:
    add(
        'WindowMaterial:SimpleGlazingSystem',
        'Glazing',
        u_factor=2.8,
        solar_heat_gain_coefficient=settings.WINDOW_SHGC,
        visible_transmittance=0.6,
    )
    add('Construction', 'Window', outside_layer='Glazing')

  def surface(room_name, label, kind, construction, boundary, coords, **kwargs):
    outdoor = boundary == 'Outdoors'
    axes = ('vertex_x_coordinate', 'vertex_y_coordinate', 'vertex_z_coordinate')
    add(
        'BuildingSurface:Detailed',
        f'{room_name} {label}',
        surface_type=kind,
        construction_name=construction,
        zone_name=room_name,
        outside_boundary_condition=boundary,
        sun_exposure='SunExposed' if outdoor else 'NoSun',
        wind_exposure='WindExposed' if outdoor else 'NoWind',
        vertices=[dict(zip(axes, point)) for point in coords],
        **kwargs,
    )

  for room in plan.rooms:
    name = room.name
    add(
        'Zone',
        name,
        ceiling_height=height,
        volume=room.volume_m3,
        floor_area=room.area_m2,
    )
    w, s, e, n = room.bounds_m
    surface(
        name,
        'floor',
        'Floor',
        'Slab',
        'Adiabatic',
        [(w, s, 0), (w, n, 0), (e, n, 0), (e, s, 0)],
    )
    surface(
        name,
        'ceiling',
        'Ceiling',
        'Slab',
        'Adiabatic',
        [(w, s, height), (e, s, height), (e, n, height), (w, n, height)],
    )
    for wall in room.walls:
      start, end = wall.start_m, wall.end_m
      boundary = 'Outdoors'
      kwargs = {}
      if wall.neighbor is not None:
        boundary = 'Surface'
        kwargs['outside_boundary_condition_object'] = (
            f'{wall.neighbor} {wall.opposite_side}'
        )
      surface(
          name,
          wall.side,
          'Wall',
          'Wall',
          boundary,
          [(*start, height), (*start, 0), (*end, 0), (*end, height)],
          **kwargs,
      )
      if wall.window is not None:
        window = wall.window
        length = dist(start, end)
        points = [
            tuple(a + d / length * (b - a) for a, b in zip(start, end))
            for d in (window.offset_m, window.offset_m + window.width_m)
        ]
        first, last = points
        vertices = [
            (*first, window.head_m),
            (*first, window.sill_m),
            (*last, window.sill_m),
            (*last, window.head_m),
        ]
        add(
            'FenestrationSurface:Detailed',
            f'{name} {wall.side} window',
            surface_type='Window',
            construction_name='Window',
            building_surface_name=f'{name} {wall.side}',
            number_of_vertices=4,
            **{
                f'vertex_{i}_{axis}_coordinate': value
                for i, vertex in enumerate(vertices, start=1)
                for axis, value in zip('xyz', vertex)
            },
        )


def _zone_hvac(add, room):
  name = room.name
  box_in, coil_in, zone_in, zone_return = (
      f'{name} Box In',
      f'{name} Reheat In',
      f'{name} Supply',
      f'{name} Return',
  )
  terminal, coil, adu, equipment = (
      f'{name} VAV',
      f'{name} Reheat',
      f'{name} ADU',
      f'{name} Equipment',
  )
  add(
      'ZoneControl:Thermostat',
      f'{name} Thermostat',
      zone_or_zonelist_name=name,
      control_type_schedule_name='Control Type',
      control_1_object_type=THERMOSTAT,
      control_1_name='Zone Setpoints',
  )
  add(
      'ZoneHVAC:EquipmentConnections',
      f'{name} Connections',
      zone_name=name,
      zone_conditioning_equipment_list_name=equipment,
      zone_air_inlet_node_or_nodelist_name=zone_in,
      zone_air_node_name=f'{name} Air',
      zone_return_air_node_or_nodelist_name=zone_return,
  )
  add(
      'ZoneHVAC:EquipmentList',
      equipment,
      load_distribution_scheme='SequentialLoad',
      equipment=[
          dict(
              zone_equipment_object_type=ADU,
              zone_equipment_name=adu,
              zone_equipment_cooling_sequence=1,
              zone_equipment_heating_or_no_load_sequence=1,
          )
      ],
  )
  add(
      ADU,
      adu,
      air_distribution_unit_outlet_node_name=zone_in,
      air_terminal_object_type=VAV,
      air_terminal_name=terminal,
  )
  add(
      VAV,
      terminal,
      availability_schedule_name='On',
      damper_air_outlet_node_name=coil_in,
      air_inlet_node_name=box_in,
      maximum_air_flow_rate=room.area_m2 * settings.FLOW_PER_M2,
      zone_minimum_air_flow_input_method='Constant',
      constant_minimum_air_flow_fraction=0.5,
      reheat_coil_object_type=REHEAT,
      reheat_coil_name=coil,
      air_outlet_node_name=zone_in,
      damper_heating_action='Reverse',
      maximum_reheat_air_temperature=45.0,
  )
  add(
      REHEAT,
      coil,
      availability_schedule_name='On',
      efficiency=1.0,
      nominal_capacity=room.area_m2 * 200.0,
      air_inlet_node_name=coil_in,
      air_outlet_node_name=zone_in,
  )


def _office_schedules_and_loads(add, plan):
  def schedule(name, hourly):
    fields = ['Through: 12/31', 'For: AllDays']
    for hour, value in enumerate(hourly, start=1):
      if hour == 24 or value != hourly[hour]:
        fields.extend((f'Until: {hour:02}:00', value))
    add(
        'Schedule:Compact',
        name,
        schedule_type_limits_name='Fraction',
        data=[{'field': field} for field in fields],
    )

  schedule('Office Outdoor Air Fraction', office_loads.FRESH_AIR_FRACTION)
  schedule('Office Lights', office_loads.LIGHTS)
  schedule('Office Plug Loads', office_loads.PLUGS)
  for use in office_loads.ROOM_USES:
    schedule(f'{use.label} Occupancy', use.occupancy)
  add(
      'ScheduleTypeLimits',
      'Occupant Activity',
      lower_limit_value=0.0,
      numeric_type='Continuous',
  )
  add(
      'Schedule:Constant',
      'Office Activity',
      schedule_type_limits_name='Occupant Activity',
      hourly_value=office_loads.ACTIVITY_W_PER_PERSON,
  )

  for room in plan.rooms:
    use = office_loads.room_use(room)
    zone = dict(zone_or_zonelist_or_space_or_spacelist_name=room.name)
    add(
        'People',
        f'{room.name} People',
        **zone,
        number_of_people_schedule_name=f'{use.label} Occupancy',
        activity_level_schedule_name='Office Activity',
        number_of_people_calculation_method='People',
        number_of_people=room.area_m2 * use.people_per_m2,
        fraction_radiant=0.3,
        sensible_heat_fraction=office_loads.SENSIBLE_FRACTION,
    )
    add(
        'Lights',
        f'{room.name} Lights',
        **zone,
        schedule_name='Office Lights',
        design_level_calculation_method='Watts/Area',
        watts_per_floor_area=use.lights_w_per_m2,
        fraction_radiant=0.3,
        fraction_visible=0.2,
    )
    add(
        'ElectricEquipment',
        f'{room.name} Plug Loads',
        **zone,
        schedule_name='Office Plug Loads',
        design_level_calculation_method='Watts/Area',
        watts_per_floor_area=use.plugs_w_per_m2,
        fraction_radiant=0.2,
    )


def _outdoor_air(add, design_flow):
  add('OutdoorAir:Node', 'Outdoor Intake')
  add(
      'OutdoorAir:Mixer',
      'Outdoor Air Mixer',
      mixed_air_node_name='Mixed Air',
      outdoor_air_stream_node_name='Outdoor Intake',
      relief_air_stream_node_name='Relief Air',
      return_air_stream_node_name='Loop Inlet',
  )
  add(
      'Controller:OutdoorAir',
      'Outdoor Air Controller',
      relief_air_outlet_node_name='Relief Air',
      return_air_node_name='Loop Inlet',
      mixed_air_node_name='Mixed Air',
      actuator_node_name='Outdoor Intake',
      minimum_outdoor_air_flow_rate=0.0,
      maximum_outdoor_air_flow_rate=design_flow,
      economizer_control_type='NoEconomizer',
      minimum_fraction_of_outdoor_air_schedule_name=(
          'Office Outdoor Air Fraction'
      ),
      maximum_fraction_of_outdoor_air_schedule_name=(
          'Office Outdoor Air Fraction'
      ),
  )
  add(
      'AirLoopHVAC:ControllerList',
      'Outdoor Air Controllers',
      controller_1_object_type='Controller:OutdoorAir',
      controller_1_name='Outdoor Air Controller',
  )
  add(
      'AirLoopHVAC:OutdoorAirSystem:EquipmentList',
      'Outdoor Air Equipment',
      component_1_object_type='OutdoorAir:Mixer',
      component_1_name='Outdoor Air Mixer',
  )
  add(
      'AirLoopHVAC:OutdoorAirSystem',
      'Outdoor Air System',
      controller_list_name='Outdoor Air Controllers',
      outdoor_air_equipment_list_name='Outdoor Air Equipment',
  )


def _air_loop(add, plan, office):
  area = sum(room.area_m2 for room in plan.rooms)
  flow = area * settings.FLOW_PER_M2
  add(
      'AirLoopHVAC',
      'Central VAV',
      design_supply_air_flow_rate=flow,
      branch_list_name='Supply Branches',
      supply_side_inlet_node_name='Loop Inlet',
      demand_side_outlet_node_name='Demand Return',
      demand_side_inlet_node_names='Demand Supply',
      supply_side_outlet_node_names='Loop Outlet',
  )
  add(
      'BranchList',
      'Supply Branches',
      branches=[dict(branch_name='Supply Branch')],
  )
  fan_inlet = 'Mixed Air' if office else 'Loop Inlet'
  components = [
      (FAN, 'Supply Fan', fan_inlet, 'Fan Outlet'),
      (DX_SYSTEM, 'DX System', 'Fan Outlet', 'Loop Outlet'),
  ]
  if office:
    _outdoor_air(add, flow)
    components.insert(
        0,
        (
            'AirLoopHVAC:OutdoorAirSystem',
            'Outdoor Air System',
            'Loop Inlet',
            'Mixed Air',
        ),
    )
  add(
      'Branch',
      'Supply Branch',
      components=[
          dict(
              component_object_type=kind,
              component_name=name,
              component_inlet_node_name=inlet,
              component_outlet_node_name=outlet,
          )
          for kind, name, inlet, outlet in components
      ],
  )
  add(
      'AirLoopHVAC:SupplyPath',
      'Supply Path',
      supply_air_path_inlet_node_name='Demand Supply',
      components=[
          dict(
              component_object_type='AirLoopHVAC:ZoneSplitter',
              component_name='Zone Splitter',
          )
      ],
  )
  add(
      'AirLoopHVAC:ZoneSplitter',
      'Zone Splitter',
      inlet_node_name='Demand Supply',
      nodes=[
          dict(outlet_node_name=f'{room.name} Box In') for room in plan.rooms
      ],
  )
  add(
      'AirLoopHVAC:ReturnPath',
      'Return Path',
      return_air_path_outlet_node_name='Demand Return',
      components=[
          dict(
              component_object_type='AirLoopHVAC:ZoneMixer',
              component_name='Zone Mixer',
          )
      ],
  )
  add(
      'AirLoopHVAC:ZoneMixer',
      'Zone Mixer',
      outlet_node_name='Demand Return',
      nodes=[
          dict(inlet_node_name=f'{room.name} Return') for room in plan.rooms
      ],
  )
  add(
      FAN,
      'Supply Fan',
      availability_schedule_name='On',
      fan_total_efficiency=settings.FAN_EFFICIENCY,
      pressure_rise=settings.FAN_PRESSURE_PA,
      maximum_flow_rate=flow,
      fan_power_minimum_flow_rate_input_method='Fraction',
      fan_power_minimum_flow_fraction=0.5,
      motor_efficiency=0.9,
      motor_in_airstream_fraction=1.0,
      fan_power_coefficient_1=0.0,
      fan_power_coefficient_2=0.0,
      fan_power_coefficient_3=0.0,
      fan_power_coefficient_4=1.0,
      fan_power_coefficient_5=0.0,
      air_inlet_node_name=fan_inlet,
      air_outlet_node_name='Fan Outlet',
  )
  add(
      DX_SYSTEM,
      'DX System',
      availability_schedule_name='On',
      dx_cooling_coil_system_inlet_node_name='Fan Outlet',
      dx_cooling_coil_system_outlet_node_name='Loop Outlet',
      dx_cooling_coil_system_sensor_node_name='Loop Outlet',
      cooling_coil_object_type=DX_COIL,
      cooling_coil_name='Central DX Coil',
  )
  # Synthetic fixed-performance curves keep the benchmark independent of vendor data.
  add(
      'Curve:Biquadratic',
      'DX Temperature Modifier',
      coefficient1_constant=1.0,
      coefficient2_x=0.0,
      coefficient3_x_2=0.0,
      coefficient4_y=0.0,
      coefficient5_y_2=0.0,
      coefficient6_x_y=0.0,
      minimum_value_of_x=0.0,
      maximum_value_of_x=40.0,
      minimum_value_of_y=-30.0,
      maximum_value_of_y=60.0,
  )
  add(
      'Curve:Quadratic',
      'DX Capacity Flow Modifier',
      coefficient1_constant=0.0,
      coefficient2_x=1.0,
      coefficient3_x_2=0.0,
      minimum_value_of_x=0.0,
      maximum_value_of_x=1.5,
  )
  add(
      'Curve:Quadratic',
      'DX EIR Flow Modifier',
      coefficient1_constant=1.0,
      coefficient2_x=0.0,
      coefficient3_x_2=0.0,
      minimum_value_of_x=0.0,
      maximum_value_of_x=1.5,
  )
  add(
      'Curve:Quadratic',
      'DX Part Load',
      coefficient1_constant=0.85,
      coefficient2_x=0.15,
      coefficient3_x_2=0.0,
      minimum_value_of_x=0.0,
      maximum_value_of_x=1.0,
  )
  add(
      DX_COIL,
      'Central DX Coil',
      availability_schedule_name='On',
      gross_rated_total_cooling_capacity=17_500.0 * flow,
      gross_rated_sensible_heat_ratio=0.72,
      gross_rated_cooling_cop=3.0,
      rated_air_flow_rate=flow,
      minimum_outdoor_dry_bulb_temperature_for_compressor_operation=0.0,
      air_inlet_node_name='Fan Outlet',
      air_outlet_node_name='Loop Outlet',
      total_cooling_capacity_function_of_temperature_curve_name=(
          'DX Temperature Modifier'
      ),
      total_cooling_capacity_function_of_flow_fraction_curve_name=(
          'DX Capacity Flow Modifier'
      ),
      energy_input_ratio_function_of_temperature_curve_name=(
          'DX Temperature Modifier'
      ),
      energy_input_ratio_function_of_flow_fraction_curve_name=(
          'DX EIR Flow Modifier'
      ),
      part_load_fraction_correlation_curve_name='DX Part Load',
  )
  add(
      'SetpointManager:Scheduled',
      'Supply Setpoint Manager',
      control_variable='Temperature',
      schedule_name='Supply Setpoint',
      setpoint_node_or_nodelist_name='Loop Outlet',
  )


def make_model(
    plan: FloorPlan, run_day: date, version: str, *, office: bool = False
) -> dict:
  """Return the complete model, including one EPW day and smoke-test outputs."""
  model = {}

  def add(kind, name, **fields):
    model.setdefault(kind, {})[name] = fields

  add('Version', 'Version', version_identifier=version)
  add('Building', 'Matched VAV', solar_distribution='FullExterior')
  add(
      'Timestep',
      'Timestep',
      number_of_timesteps_per_hour=settings.STEPS_PER_HOUR,
  )
  add(
      'SimulationControl',
      'Simulation',
      do_zone_sizing_calculation='No',
      do_system_sizing_calculation='No',
      do_plant_sizing_calculation='No',
      run_simulation_for_sizing_periods='No',
      run_simulation_for_weather_file_run_periods='Yes',
  )
  add(
      'RunPeriod',
      'Smoke Day',
      begin_month=run_day.month,
      begin_day_of_month=run_day.day,
      end_month=run_day.month,
      end_day_of_month=run_day.day,
      begin_year=run_day.year,
      end_year=run_day.year,
      use_weather_file_holidays_and_special_days='No',
      use_weather_file_daylight_saving_period='No',
  )
  add(
      'ScheduleTypeLimits',
      'Fraction',
      lower_limit_value=0,
      upper_limit_value=1,
      numeric_type='Continuous',
  )
  add(
      'ScheduleTypeLimits',
      'Thermostat Mode',
      lower_limit_value=0,
      upper_limit_value=4,
      numeric_type='Discrete',
  )
  add('ScheduleTypeLimits', 'Temperature', unit_type='Temperature')
  schedules = [
      ('On', 'Fraction', 1),
      ('Control Type', 'Thermostat Mode', 4),
      ('Heating Setpoint', 'Temperature', settings.HEATING_SETPOINT_C),
      ('Cooling Setpoint', 'Temperature', settings.COOLING_SETPOINT_C),
      ('Supply Setpoint', 'Temperature', settings.SUPPLY_AIR_C),
  ]
  for name, limits, value in schedules:
    add(
        'Schedule:Constant',
        name,
        schedule_type_limits_name=limits,
        hourly_value=value,
    )
  add(
      THERMOSTAT,
      'Zone Setpoints',
      heating_setpoint_temperature_schedule_name='Heating Setpoint',
      cooling_setpoint_temperature_schedule_name='Cooling Setpoint',
  )
  _geometry(add, plan)
  _air_loop(add, plan, office)
  if office:
    _office_schedules_and_loads(add, plan)
  for room in plan.rooms:
    _zone_hvac(add, room)

  add('OutputControl:Files', 'Smoke Output', output_csv='Yes')
  outputs = [
      (ZONE_TEMP, '*'),
      (DAMPER, '*'),
      (COIL_HEAT, '*'),
      (COIL_COOL, 'Central DX Coil'),
  ]
  outputs.extend((NODE_FLOW, f'{room.name} Supply') for room in plan.rooms)
  if office:
    outputs.append((OFFICE_OUTDOOR_AIR, 'Central VAV'))
    outputs.extend((variable, '*') for variable in OFFICE_SENSIBLE_LOADS)
  for i, (variable, key) in enumerate(outputs, start=1):
    add(
        'Output:Variable',
        f'Smoke Output {i}',
        key_value=key,
        variable_name=variable,
        reporting_frequency='Timestep',
    )
  return model
