"""Check detailed HVAC loop sizing and connections on the matched floor plan."""

from datetime import date

from benchmarks.eplus_rl_speed.matched_vav import detailed_vav_hvac
from benchmarks.eplus_rl_speed.matched_vav import energyplus_model
from benchmarks.eplus_rl_speed.matched_vav.floorplan import PlanSpec
from benchmarks.eplus_rl_speed.matched_vav.floorplan import make_plan


def test_each_detailed_air_loop_sizes_and_ventilates_its_own_rooms():
  plan = make_plan(PlanSpec(41, columns=10, window_width_m=2))
  model = energyplus_model.make_model(
      plan, date(2024, 7, 15), '24.2', office=True
  )
  surfaces = model['BuildingSurface:Detailed'].copy()
  thermostats = model['ZoneControl:Thermostat'].copy()

  detailed_vav_hvac.replace_hvac(model, plan)

  assert model['BuildingSurface:Detailed'] == surfaces
  assert model['ZoneControl:Thermostat'] == thermostats
  assert set(model['AirLoopHVAC']) == {'AHU 1', 'AHU 2', 'AHU 3'}
  assert len(model['SizingPeriod:DesignDay']) == 2
  assert {
      obj['airloop_name'] for obj in model['Sizing:System'].values()
  } == set(model['AirLoopHVAC'])
  served = []
  for loop in model['AirLoopHVAC']:
    assert (
        model['AirLoopHVAC'][loop]['design_supply_air_flow_rate'] == 'Autosize'
    )
    assert (
        model['Fan:SystemModel'][f'{loop} Supply Fan'][
            'design_maximum_air_flow_rate'
        ]
        == 'Autosize'
    )
    assert (
        model['Coil:Cooling:DX:TwoSpeed'][f'{loop} DX Coil'][
            'low_speed_gross_rated_total_cooling_capacity'
        ]
        == 'Autosize'
    )
    zones = model['Controller:MechanicalVentilation'][f'{loop} Ventilation'][
        'zone_specifications'
    ]
    assert len(zones) == 1
    room_list = model['ZoneList'][zones[0]['zone_or_zonelist_name']]['zones']
    assert len(room_list) <= 20
    split = model['AirLoopHVAC:ZoneSplitter'][f'{loop} Splitter']['nodes']
    assert [node['outlet_node_name'] for node in split] == [
        f"{room['zone_name']} Box In" for room in room_list
    ]
    for room in room_list:
      name = room['zone_name']
      terminal = model[energyplus_model.VAV][f'{name} VAV']
      assert terminal['maximum_air_flow_rate'] == 'Autosize'
      assert (
          terminal['design_specification_outdoor_air_object_name']
          == zones[0]['design_specification_outdoor_air_object_name']
      )
      served.append(name)
  assert served == [room.name for room in plan.rooms]


def test_detailed_air_loop_can_cycle_outside_fan_schedule():
  plan = make_plan(PlanSpec(1))
  model = energyplus_model.make_model(
      plan, date(2024, 7, 15), '24.2', office=True
  )
  detailed_vav_hvac.replace_hvac(model, plan)

  loop = model['AirLoopHVAC']['AHU 1']
  managers = model['AvailabilityManagerAssignmentList'][
      loop['availability_manager_list_name']
  ]['managers']
  assert len(managers) == 1
  assert managers[0]['availability_manager_object_type'] == (
      'AvailabilityManager:NightCycle'
  )
  cycle = model['AvailabilityManager:NightCycle'][
      managers[0]['availability_manager_name']
  ]
  fan_schedule = model['Fan:SystemModel']['AHU 1 Supply Fan'][
      'availability_schedule_name'
  ]
  assert cycle['fan_schedule_name'] == fan_schedule
  assert cycle['control_type'] == 'CycleOnAny'
  assert cycle['cycling_run_time'] == 1800
  fields = [f['field'] for f in model['Schedule:Compact'][fan_schedule]['data']]
  assert fields[fields.index('For: SummerDesignDay WinterDesignDay') + 2] == 1.0
  assert fields[fields.index('For: AllOtherDays') + 2] == 0.0
