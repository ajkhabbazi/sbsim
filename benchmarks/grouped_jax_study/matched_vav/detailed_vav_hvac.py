"""Autosized DX VAV systems for the detailed EnergyPlus office model."""

from math import ceil

from . import energyplus_model as eplus
from .floorplan import FloorPlan


FAN = 'Fan:SystemModel'
DX_COIL = 'Coil:Cooling:DX:TwoSpeed'
LOOP_OUTDOOR_AIR = 'Air System Outdoor Air Mass Flow Rate'
ZONE_OUTDOOR_AIR = 'Zone Air Terminal Outdoor Air Volume Flow Rate'
_AUTOSIZE = 'Autosize'
_MAX_ZONES_PER_LOOP = 20
_COOLING_DESIGN_C = 14.0
_ZONE_OUTDOOR_AIR_SPEC = 'Office Ventilation Rates'
_ZONE_AIR_DISTRIBUTION = 'Office Air Distribution'
_FAN_AVAILABILITY = 'Office HVAC Operation'


def replace_hvac(model: dict, plan: FloorPlan) -> None:
  """Replace the office model's central loop and autosize its room terminals."""
  for kind in (
      'AirLoopHVAC',
      'AirLoopHVAC:ControllerList',
      'AirLoopHVAC:OutdoorAirSystem',
      'AirLoopHVAC:OutdoorAirSystem:EquipmentList',
      'AirLoopHVAC:ReturnPath',
      'AirLoopHVAC:SupplyPath',
      'AirLoopHVAC:ZoneMixer',
      'AirLoopHVAC:ZoneSplitter',
      'Branch',
      'BranchList',
      'Controller:OutdoorAir',
      'Curve:Biquadratic',
      'Curve:Quadratic',
      'OutdoorAir:Mixer',
      'OutdoorAir:Node',
      'SetpointManager:Scheduled',
      eplus.DX_COIL,
      eplus.DX_SYSTEM,
      eplus.FAN,
  ):
    model.pop(kind, None)
  model['Schedule:Compact'].pop('Office Outdoor Air Fraction', None)
  model['Schedule:Constant'].pop('Supply Setpoint', None)
  outputs = model['Output:Variable']
  for name, output in list(outputs.items()):
    if output['key_value'] in ('Central VAV', 'Central DX Coil'):
      del outputs[name]

  def add(kind, name, **fields):
    model.setdefault(kind, {})[name] = fields

  for control in model['SimulationControl'].values():
    control.update(
        do_zone_sizing_calculation='Yes',
        do_system_sizing_calculation='Yes',
    )
  _add_design_days(add)
  _add_curves(add)
  _add_availability_schedule(add)
  add(
      'DesignSpecification:OutdoorAir',
      _ZONE_OUTDOOR_AIR_SPEC,
      outdoor_air_method='Sum',
      outdoor_air_flow_per_person=0.0025,
      outdoor_air_flow_per_zone_floor_area=0.0003,
  )
  add(
      'DesignSpecification:ZoneAirDistribution',
      _ZONE_AIR_DISTRIBUTION,
      zone_air_distribution_effectiveness_in_cooling_mode=1.0,
      zone_air_distribution_effectiveness_in_heating_mode=1.0,
  )
  for room in plan.rooms:
    name = room.name
    _add_zone_sizing(add, name)
    model[eplus.VAV][f'{name} VAV'].update(
        maximum_air_flow_rate=_AUTOSIZE,
        constant_minimum_air_flow_fraction=0.30,
        damper_heating_action='ReverseWithLimits',
        maximum_flow_per_zone_floor_area_during_reheat=_AUTOSIZE,
        maximum_flow_fraction_during_reheat=_AUTOSIZE,
        design_specification_outdoor_air_object_name=_ZONE_OUTDOOR_AIR_SPEC,
    )
    model[eplus.REHEAT][f'{name} Reheat']['nominal_capacity'] = _AUTOSIZE

  # Split row-major rooms into contiguous groups with similar zone counts.
  loop_count = ceil(len(plan.rooms) / _MAX_ZONES_PER_LOOP)
  for i in range(loop_count):
    start = i * len(plan.rooms) // loop_count
    stop = (i + 1) * len(plan.rooms) // loop_count
    names = [room.name for room in plan.rooms[start:stop]]
    _add_air_loop(add, f'AHU {i + 1}', names)

  for variable in (
      eplus.COIL_COOL,
      eplus.OFFICE_OUTDOOR_AIR,
      LOOP_OUTDOOR_AIR,
      ZONE_OUTDOOR_AIR,
  ):
    # The wildcard keys follow the actual EnergyPlus component naming.
    add(
        'Output:Variable',
        variable,
        key_value='*',
        variable_name=variable,
        reporting_frequency='Timestep',
    )


def _add_design_days(add):
  # Chicago 0.4% cooling wet-bulb and 99.6% heating dry-bulb design conditions
  # from EnergyPlus 24.2's RefBldgMediumOfficeNew2004_Chicago_epJSON example.
  common = dict(
      day_of_month=21,
      barometric_pressure=99063.0,
      humidity_condition_type='WetBulb',
      solar_model_indicator='ASHRAEClearSky',
  )
  add(
      'SizingPeriod:DesignDay',
      'Chicago 0.4% Summer Design Day',
      **common,
      month=7,
      day_type='SummerDesignDay',
      maximum_dry_bulb_temperature=31.2,
      daily_dry_bulb_temperature_range=10.7,
      wetbulb_or_dewpoint_at_maximum_dry_bulb=25.5,
      wind_direction=230,
      wind_speed=5.3,
      sky_clearness=1.0,
  )
  add(
      'SizingPeriod:DesignDay',
      'Chicago 99.6% Winter Design Day',
      **common,
      month=1,
      day_type='WinterDesignDay',
      maximum_dry_bulb_temperature=-20.6,
      daily_dry_bulb_temperature_range=0.0,
      wetbulb_or_dewpoint_at_maximum_dry_bulb=-20.6,
      wind_direction=270,
      wind_speed=4.9,
      sky_clearness=0.0,
  )
  add(
      'Sizing:Parameters',
      'Office Sizing Margins',
      heating_sizing_factor=1.15,
      cooling_sizing_factor=1.15,
  )


def _add_curves(add):
  # Measured 10-ton and low-speed DX performance coefficients from the same
  # official MediumOffice example. All units share these normalized curves.
  high_capacity = (0.52357, 0.03478, 0.0, -0.001915, -0.00010838, 0.0)
  high_eir = (0.9847, -0.04285, 0.0013562, 0.009934, 0.0006398, -0.001169)
  low_capacity = (0.4136, 0.03105, 0.0, 0.006952, -0.0002128, 0.0)
  low_eir = (1.1389, -0.04518, 0.0014298, 0.006044, 0.0006745, -0.0012325)
  biquadratic_fields = (
      'coefficient1_constant',
      'coefficient2_x',
      'coefficient3_x_2',
      'coefficient4_y',
      'coefficient5_y_2',
      'coefficient6_x_y',
  )
  for name, coefficients in (
      ('DX High Capacity Temperature', high_capacity),
      ('DX High EIR Temperature', high_eir),
      ('DX Low Capacity Temperature', low_capacity),
      ('DX Low EIR Temperature', low_eir),
  ):
    add(
        'Curve:Biquadratic',
        name,
        **dict(zip(biquadratic_fields, coefficients)),
        minimum_value_of_x=11.1,
        maximum_value_of_x=29.4,
        minimum_value_of_y=10.0,
        maximum_value_of_y=50.3,
    )
  for name, intercept, slope, low, high in (
      ('DX Capacity Flow', 0.7685, 0.2315, 0.776, 1.197),
      ('DX EIR Flow', 1.192, -0.1917, 0.776, 1.197),
      ('DX Part Load Fraction', 0.85, 0.15, 0.0, 1.0),
  ):
    add(
        'Curve:Quadratic',
        name,
        coefficient1_constant=intercept,
        coefficient2_x=slope,
        coefficient3_x_2=0.0,
        minimum_value_of_x=low,
        maximum_value_of_x=high,
    )
  add(
      'Curve:Cubic',
      'VAV Fan Power',
      coefficient1_constant=0.040759894,
      coefficient2_x=0.08804497,
      coefficient3_x_2=-0.07292612,
      coefficient4_x_3=0.943739823,
      minimum_value_of_x=0.0,
      maximum_value_of_x=1.0,
  )


def _add_zone_sizing(add, name):
  add(
      'Sizing:Zone',
      f'{name} Sizing',
      zone_or_zonelist_name=name,
      design_specification_outdoor_air_object_name=_ZONE_OUTDOOR_AIR_SPEC,
      design_specification_zone_air_distribution_object_name=_ZONE_AIR_DISTRIBUTION,
      zone_cooling_design_supply_air_temperature_input_method=(
          'SupplyAirTemperature'
      ),
      zone_cooling_design_supply_air_temperature=_COOLING_DESIGN_C,
      zone_cooling_design_supply_air_humidity_ratio=0.0085,
      zone_heating_design_supply_air_temperature_input_method=(
          'SupplyAirTemperature'
      ),
      zone_heating_design_supply_air_temperature=40.0,
      zone_heating_design_supply_air_humidity_ratio=0.008,
      cooling_design_air_flow_method='DesignDay',
      heating_design_air_flow_method='DesignDay',
  )


def _add_availability_schedule(add):
  add(
      'Schedule:Compact',
      _FAN_AVAILABILITY,
      schedule_type_limits_name='Fraction',
      data=[
          {'field': field}
          for field in (
              'Through: 12/31',
              'For: SummerDesignDay WinterDesignDay',
              'Until: 24:00',
              1.0,
              'For: Weekdays',
              'Until: 06:00',
              0.0,
              'Until: 22:00',
              1.0,
              'Until: 24:00',
              0.0,
              'For: AllOtherDays',
              'Until: 24:00',
              0.0,
          )
      ],
  )


def _add_air_loop(add, loop, zones):
  loop_in = f'{loop} Loop Inlet'
  mixed = f'{loop} Mixed Air'
  dx_out = f'{loop} DX Outlet'
  fan_in = f'{loop} Heating Outlet'
  loop_out = f'{loop} Loop Outlet'
  demand_supply = f'{loop} Demand Supply'
  demand_return = f'{loop} Demand Return'
  outdoor = f'{loop} Outdoor Air System'
  dx = f'{loop} DX System'
  heating = f'{loop} Heating Coil'
  fan = f'{loop} Supply Fan'
  branch = f'{loop} Supply Branch'
  splitter, mixer = f'{loop} Splitter', f'{loop} Mixer'
  availability = f'{loop} Availability'
  night_cycle = f'{loop} Night Cycle'

  add(
      'AirLoopHVAC',
      loop,
      design_supply_air_flow_rate=_AUTOSIZE,
      availability_manager_list_name=availability,
      branch_list_name=f'{loop} Branches',
      supply_side_inlet_node_name=loop_in,
      demand_side_outlet_node_name=demand_return,
      demand_side_inlet_node_names=demand_supply,
      supply_side_outlet_node_names=loop_out,
  )
  add(
      'AvailabilityManagerAssignmentList',
      availability,
      managers=[
          dict(
              availability_manager_object_type='AvailabilityManager:NightCycle',
              availability_manager_name=night_cycle,
          )
      ],
  )
  add(
      'AvailabilityManager:NightCycle',
      night_cycle,
      applicability_schedule_name='On',
      fan_schedule_name=_FAN_AVAILABILITY,
      control_type='CycleOnAny',
      thermostat_tolerance=1.0,
      cycling_run_time_control_type='FixedRunTime',
      cycling_run_time=1800,
  )
  add('BranchList', f'{loop} Branches', branches=[dict(branch_name=branch)])
  add(
      'Branch',
      branch,
      components=[
          dict(
              component_object_type=kind,
              component_name=name,
              component_inlet_node_name=inlet,
              component_outlet_node_name=outlet,
          )
          for kind, name, inlet, outlet in (
              ('AirLoopHVAC:OutdoorAirSystem', outdoor, loop_in, mixed),
              (eplus.DX_SYSTEM, dx, mixed, dx_out),
              (eplus.REHEAT, heating, dx_out, fan_in),
              (FAN, fan, fan_in, loop_out),
          )
      ],
  )
  add(
      'AirLoopHVAC:SupplyPath',
      f'{loop} Supply Path',
      supply_air_path_inlet_node_name=demand_supply,
      components=[
          dict(
              component_object_type='AirLoopHVAC:ZoneSplitter',
              component_name=splitter,
          )
      ],
  )
  add(
      'AirLoopHVAC:ZoneSplitter',
      splitter,
      inlet_node_name=demand_supply,
      nodes=[dict(outlet_node_name=f'{name} Box In') for name in zones],
  )
  add(
      'AirLoopHVAC:ReturnPath',
      f'{loop} Return Path',
      return_air_path_outlet_node_name=demand_return,
      components=[
          dict(
              component_object_type='AirLoopHVAC:ZoneMixer',
              component_name=mixer,
          )
      ],
  )
  add(
      'AirLoopHVAC:ZoneMixer',
      mixer,
      outlet_node_name=demand_return,
      nodes=[dict(inlet_node_name=f'{name} Return') for name in zones],
  )
  _add_ventilation(add, loop, loop_in, mixed, outdoor, zones)
  _add_conditioning(
      add, loop, dx, heating, fan, mixed, dx_out, fan_in, loop_out
  )
  _add_setpoints(add, loop, mixed, dx_out, fan_in, loop_out)

  add(
      'Sizing:System',
      f'{loop} Sizing',
      airloop_name=loop,
      type_of_load_to_size_on='Sensible',
      type_of_zone_sum_to_use='Coincident',
      preheat_design_temperature=7.0,
      preheat_design_humidity_ratio=0.008,
      precool_design_temperature=_COOLING_DESIGN_C,
      precool_design_humidity_ratio=0.0085,
      central_cooling_design_supply_air_temperature=_COOLING_DESIGN_C,
      central_cooling_design_supply_air_humidity_ratio=0.0085,
      central_heating_design_supply_air_temperature=18.0,
      central_heating_design_supply_air_humidity_ratio=0.008,
      design_outdoor_air_flow_rate=_AUTOSIZE,
      central_heating_maximum_system_air_flow_ratio=0.30,
      system_outdoor_air_method='ZoneSum',
      central_cooling_capacity_control_method='VAV',
  )


def _add_ventilation(add, loop, loop_in, mixed, outdoor, zones):
  intake, relief = f'{loop} Outdoor Intake', f'{loop} Relief Air'
  mixer = f'{loop} Outdoor Air Mixer'
  controllers, controller = (
      f'{loop} Outdoor Air Controllers',
      f'{loop} OA Control',
  )
  equipment, ventilation = (
      f'{loop} Outdoor Air Equipment',
      f'{loop} Ventilation',
  )
  add('OutdoorAir:Node', intake)
  add(
      'OutdoorAir:Mixer',
      mixer,
      mixed_air_node_name=mixed,
      outdoor_air_stream_node_name=intake,
      relief_air_stream_node_name=relief,
      return_air_stream_node_name=loop_in,
  )
  add(
      'Controller:OutdoorAir',
      controller,
      relief_air_outlet_node_name=relief,
      return_air_node_name=loop_in,
      mixed_air_node_name=mixed,
      actuator_node_name=intake,
      minimum_outdoor_air_flow_rate=0.0,
      maximum_outdoor_air_flow_rate=_AUTOSIZE,
      economizer_control_type='DifferentialDryBulb',
      economizer_control_action_type='ModulateFlow',
      economizer_maximum_limit_dry_bulb_temperature=24.0,
      economizer_minimum_limit_dry_bulb_temperature=-12.0,
      minimum_limit_type='FixedMinimum',
      mechanical_ventilation_controller_name=ventilation,
  )
  # A ZoneList expands to all rooms without the controller's extensible
  # fields truncating standalone room entries in EnergyPlus 24.2.
  zone_list = f'{loop} Zones'
  add('ZoneList', zone_list, zones=[dict(zone_name=name) for name in zones])
  add(
      'Controller:MechanicalVentilation',
      ventilation,
      availability_schedule_name='On',
      demand_controlled_ventilation='Yes',
      system_outdoor_air_method='ZoneSum',
      zone_specifications=[
          dict(
              zone_or_zonelist_name=zone_list,
              design_specification_outdoor_air_object_name=_ZONE_OUTDOOR_AIR_SPEC,
              design_specification_zone_air_distribution_object_name=_ZONE_AIR_DISTRIBUTION,
          )
      ],
  )
  add(
      'AirLoopHVAC:ControllerList',
      controllers,
      controller_1_object_type='Controller:OutdoorAir',
      controller_1_name=controller,
  )
  add(
      'AirLoopHVAC:OutdoorAirSystem:EquipmentList',
      equipment,
      component_1_object_type='OutdoorAir:Mixer',
      component_1_name=mixer,
  )
  add(
      'AirLoopHVAC:OutdoorAirSystem',
      outdoor,
      controller_list_name=controllers,
      outdoor_air_equipment_list_name=equipment,
      availability_manager_list_name=f'{loop} Availability',
  )


def _add_conditioning(
    add, loop, dx, heating, fan, mixed, dx_out, fan_in, loop_out
):
  coil = f'{loop} DX Coil'
  add('OutdoorAir:Node', f'{loop} Condenser Intake')
  add(
      'CoilSystem:Cooling:DX',
      dx,
      availability_schedule_name='On',
      dx_cooling_coil_system_inlet_node_name=mixed,
      dx_cooling_coil_system_outlet_node_name=dx_out,
      dx_cooling_coil_system_sensor_node_name=dx_out,
      cooling_coil_object_type=DX_COIL,
      cooling_coil_name=coil,
  )
  add(
      DX_COIL,
      coil,
      availability_schedule_name='On',
      air_inlet_node_name=mixed,
      air_outlet_node_name=dx_out,
      condenser_air_inlet_node_name=f'{loop} Condenser Intake',
      minimum_outdoor_dry_bulb_temperature_for_compressor_operation=10.0,
      high_speed_gross_rated_total_cooling_capacity=_AUTOSIZE,
      high_speed_rated_sensible_heat_ratio=_AUTOSIZE,
      high_speed_gross_rated_cooling_cop=3.23372055845678,
      high_speed_rated_air_flow_rate=_AUTOSIZE,
      low_speed_gross_rated_total_cooling_capacity=_AUTOSIZE,
      low_speed_gross_rated_sensible_heat_ratio=_AUTOSIZE,
      low_speed_gross_rated_cooling_cop=3.28186528141147,
      low_speed_rated_air_flow_rate=_AUTOSIZE,
      total_cooling_capacity_function_of_temperature_curve_name=(
          'DX High Capacity Temperature'
      ),
      total_cooling_capacity_function_of_flow_fraction_curve_name=(
          'DX Capacity Flow'
      ),
      energy_input_ratio_function_of_temperature_curve_name=(
          'DX High EIR Temperature'
      ),
      energy_input_ratio_function_of_flow_fraction_curve_name='DX EIR Flow',
      low_speed_total_cooling_capacity_function_of_temperature_curve_name=(
          'DX Low Capacity Temperature'
      ),
      low_speed_energy_input_ratio_function_of_temperature_curve_name=(
          'DX Low EIR Temperature'
      ),
      part_load_fraction_correlation_curve_name='DX Part Load Fraction',
      unit_internal_static_air_pressure=450.0,
  )
  add(
      eplus.REHEAT,
      heating,
      availability_schedule_name='On',
      efficiency=1.0,
      nominal_capacity=_AUTOSIZE,
      air_inlet_node_name=dx_out,
      air_outlet_node_name=fan_in,
      temperature_setpoint_node_name=fan_in,
  )
  add(
      FAN,
      fan,
      availability_schedule_name=_FAN_AVAILABILITY,
      air_inlet_node_name=fan_in,
      air_outlet_node_name=loop_out,
      design_maximum_air_flow_rate=_AUTOSIZE,
      speed_control_method='Continuous',
      design_pressure_rise=750.0,
      motor_efficiency=0.91,
      design_electric_power_consumption=_AUTOSIZE,
      design_power_sizing_method='TotalEfficiencyAndPressure',
      fan_total_efficiency=0.5915,
      electric_power_function_of_flow_fraction_curve_name='VAV Fan Power',
  )


def _add_setpoints(add, loop, mixed, dx_out, fan_in, loop_out):
  reset_range = dict(
      outdoor_low_temperature=10.0, outdoor_high_temperature=26.0
  )
  add(
      'SetpointManager:OutdoorAirReset',
      f'{loop} Supply Reset',
      control_variable='Temperature',
      setpoint_at_outdoor_low_temperature=19.0,
      setpoint_at_outdoor_high_temperature=_COOLING_DESIGN_C,
      setpoint_node_or_nodelist_name=loop_out,
      **reset_range,
  )
  add(
      'SetpointManager:OutdoorAirReset',
      f'{loop} Heating Reset',
      control_variable='Temperature',
      setpoint_at_outdoor_low_temperature=17.0,
      setpoint_at_outdoor_high_temperature=12.0,
      setpoint_node_or_nodelist_name=fan_in,
      **reset_range,
  )
  for label, node in (('Cooling', dx_out), ('Economizer', mixed)):
    freeze_protection = {}
    if node == mixed:
      freeze_protection = dict(
          cooling_coil_inlet_node_name=mixed,
          cooling_coil_outlet_node_name=dx_out,
          minimum_temperature_at_cooling_coil_outlet_node=7.2,
      )
    add(
        'SetpointManager:MixedAir',
        f'{loop} {label} Setpoint',
        control_variable='Temperature',
        reference_setpoint_node_name=loop_out,
        fan_inlet_node_name=fan_in,
        fan_outlet_node_name=loop_out,
        setpoint_node_or_nodelist_name=node,
        **freeze_protection,
    )
