"""Office envelope and schedules for the example-based EnergyPlus model."""

from calendar import month_name

from . import energyplus_model


def _envelope(model, plan):
  del model['Construction']['Slab'], model['Material:NoMass']['Slab Resistance']
  if not any(wall.neighbor is not None for room in plan.rooms for wall in room.walls):
    del model['Construction']['Wall']
  model['Material:NoMass'].update({
      'Facade Insulation': dict(roughness='MediumRough', thermal_resistance=1.75),
      'Roof Insulation': dict(roughness='MediumRough', thermal_resistance=3.5),
      'Slab Insulation': dict(roughness='MediumRough', thermal_resistance=0.75),
  })
  model['Material']['Concrete Deck'] = dict(
      roughness='MediumRough', thickness=0.10, conductivity=1.4,
      density=2300.0, specific_heat=900.0,
  )
  model['Construction'].update({
      'Insulated Facade': dict(outside_layer='Facade Insulation', layer_2='Solid Wall'),
      'Insulated Roof': dict(outside_layer='Roof Insulation', layer_2='Concrete Deck'),
      'Ground Slab': dict(outside_layer='Slab Insulation', layer_2='Concrete Deck'),
  })
  # Monthly under-building boundary temperatures from the Chicago MediumOffice.
  temperatures = (19.527, 19.502, 19.536, 19.598, 20.002, 21.640,
                  22.225, 22.375, 21.449, 20.121, 19.802, 19.633)
  model['Site:GroundTemperature:BuildingSurface'] = {
      'Chicago Building Surface': {
          f'{month.lower()}_ground_temperature': value
          for month, value in zip(month_name[1:], temperatures, strict=True)
      },
  }
  surfaces = model['BuildingSurface:Detailed']
  model['ZoneInfiltration:DesignFlowRate'] = infiltration = {}
  model['Schedule:Compact']['Office Infiltration'] = dict(
      schedule_type_limits_name='Fraction', data=[
          {'field': field} for field in (
              'Through: 12/31', 'For: Weekdays SummerDesignDay',
              'Until: 07:00', 1.0, 'Until: 19:00', 0.25, 'Until: 24:00', 1.0,
              'For: AllOtherDays', 'Until: 24:00', 1.0,
          )
      ],
  )
  for room in plan.rooms:
    name = room.name
    surfaces[f'{name} floor'].update(
        construction_name='Ground Slab', outside_boundary_condition='Ground',
    )
    surfaces[f'{name} ceiling'].update(
        surface_type='Roof', construction_name='Insulated Roof',
        outside_boundary_condition='Outdoors', sun_exposure='SunExposed',
        wind_exposure='WindExposed',
    )
    exterior_walls = [wall for wall in room.walls if wall.neighbor is None]
    for wall in exterior_walls:
      surfaces[f'{name} {wall.side}']['construction_name'] = 'Insulated Facade'
    if exterior_walls:
      infiltration[f'{name} Infiltration'] = dict(
          zone_or_zonelist_or_space_or_spacelist_name=name,
          schedule_name='Office Infiltration',
          design_flow_rate_calculation_method='Flow/ExteriorWallArea',
          flow_rate_per_exterior_surface_area=0.000302,
          constant_term_coefficient=1.0,
      )


def _activity_schedules(model):
  weekend_fractions = {
      'Office Lights': 0.2,
      'Office Plug Loads': 0.35,
      'Open Office Occupancy': 0.0,
      'Meeting Occupancy': 0.0,
      'Support Occupancy': 0.0,
  }
  for name, fraction in weekend_fractions.items():
    data = model['Schedule:Compact'][name]['data']
    data[1]['field'] = 'For: Weekdays SummerDesignDay'
    data.extend([
        {'field': 'For: AllOtherDays'}, {'field': 'Until: 24:00'},
        {'field': fraction},
    ])


def make_model(plan, run_day, version):
  """Create the detailed office on the same geometry as the matched case."""
  from . import detailed_vav_hvac

  model = energyplus_model.make_model(plan, run_day, version, office=True)
  model['Building']['Matched VAV']['solar_distribution'] = 'FullInteriorAndExterior'
  _envelope(model, plan)
  _activity_schedules(model)
  detailed_vav_hvac.replace_hvac(model, plan)

  out = model['Output:Variable']
  for index, variable in enumerate((
      'Fan Electricity Rate', 'Cooling Coil Electricity Rate', 'Heating Coil Electricity Rate',
      'Zone Target Voz Ventilation Flow Rate', 'Zone Thermostat Heating Setpoint Temperature',
      'Zone Thermostat Cooling Setpoint Temperature', 'Site Outdoor Air Drybulb Temperature',
      'Zone People Occupant Count',
  ), 1):
    out[f'Detailed Office {index}'] = dict(
        key_value='*', variable_name=variable, reporting_frequency='Timestep',
    )
  for loop, air_loop in model['AirLoopHVAC'].items():
    for variable in (energyplus_model.NODE_FLOW, 'System Node Temperature'):
      out[f'{loop} {variable}'] = dict(
          key_value=air_loop['supply_side_outlet_node_names'], variable_name=variable,
          reporting_frequency='Timestep',
      )
  return model
