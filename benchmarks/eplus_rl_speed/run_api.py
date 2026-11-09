import sys, time
from bench_paths import ENERGYPLUS_DIR, WEATHER
sys.path.insert(0, str(ENERGYPLUS_DIR))
from pyenergyplus.api import EnergyPlusAPI

api = EnergyPlusAPI()
state = api.state_manager.new_state()
api.runtime.set_console_output_status(state, False)

t0 = time.perf_counter()
code = api.runtime.run_energyplus(state, [
    "-w", str(WEATHER),
    "-d", "out-api",
    "5Zone_IdealLoadsAirSystems_ReturnPlenum.idf",
])
print("exit code:", code)
print("time:", round(time.perf_counter() - t0, 2), "s")
