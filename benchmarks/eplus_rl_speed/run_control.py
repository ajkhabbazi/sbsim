import sys, time
from collections import Counter
from bench_paths import ENERGYPLUS_DIR, WEATHER
sys.path.insert(0, str(ENERGYPLUS_DIR))
from pyenergyplus.api import EnergyPlusAPI

ZONES = ["SPACE1-1", "SPACE2-1", "SPACE3-1", "SPACE4-1", "SPACE5-1"]
IDEAL = [f"ZONE {i} Ideal Loads" for i in range(1, 6)]
HEAT_SP, COOL_SP = 21.0, 24.0   # fixed for now; the agent sets these in step 4

api = EnergyPlusAPI()
ex = api.exchange
state = api.state_manager.new_state()
api.runtime.set_console_output_status(state, False)

# readings EnergyPlus must keep available during the run
for z in ZONES:
    ex.request_variable(state, "Zone Mean Air Temperature", z)
ex.request_variable(state, "Site Outdoor Air Drybulb Temperature", "Environment")
for i in IDEAL:
    ex.request_variable(state, "Zone Ideal Loads Zone Total Heating Rate", i)
    ex.request_variable(state, "Zone Ideal Loads Zone Total Cooling Rate", i)

h = {}              # handles, looked up on the first call
calls = Counter()   # control steps by simulation kind, warm-up excluded

def control(s):
    if not ex.api_data_fully_ready(s):
        return
    if not h:
        h["zone"] = [ex.get_variable_handle(s, "Zone Mean Air Temperature", z) for z in ZONES]
        h["out"] = ex.get_variable_handle(s, "Site Outdoor Air Drybulb Temperature", "Environment")
        h["heat"] = [ex.get_variable_handle(s, "Zone Ideal Loads Zone Total Heating Rate", i) for i in IDEAL]
        h["cool"] = [ex.get_variable_handle(s, "Zone Ideal Loads Zone Total Cooling Rate", i) for i in IDEAL]
        h["heat_sp"] = ex.get_actuator_handle(s, "Schedule:Compact", "Schedule Value", "Htg-SetP-Sch")
        h["cool_sp"] = ex.get_actuator_handle(s, "Schedule:Compact", "Schedule Value", "Clg-SetP-Sch")
        if -1 in h["zone"] + h["heat"] + h["cool"] + [h["out"], h["heat_sp"], h["cool_sp"]]:
            print("handle not found:", h)
            api.runtime.stop_simulation(s)
            return
    # read (not used yet; the agent uses these in step 4)
    zone_t = [ex.get_variable_value(s, k) for k in h["zone"]]
    out_t = ex.get_variable_value(s, h["out"])
    heat_w = sum(ex.get_variable_value(s, k) for k in h["heat"])
    cool_w = sum(ex.get_variable_value(s, k) for k in h["cool"])
    # write
    ex.set_actuator_value(s, h["heat_sp"], HEAT_SP)
    ex.set_actuator_value(s, h["cool_sp"], COOL_SP)
    if not ex.warmup_flag(s):
        calls[ex.kind_of_sim(s)] += 1

api.runtime.callback_begin_zone_timestep_after_init_heat_balance(state, control)

t0 = time.perf_counter()
code = api.runtime.run_energyplus(state, [
    "-w", str(WEATHER),
    "-d", "out-control",
    "5Zone_IdealLoadsAirSystems_ReturnPlenum.idf",
])
print("exit code:", code)
print("time:", round(time.perf_counter() - t0, 2), "s")
print("control steps by kind of sim:", dict(calls))
