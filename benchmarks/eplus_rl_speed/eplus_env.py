"""EnergyPlus as an RL environment: reset() and step(action).

EnergyPlus runs in a background thread. At the start of every 5-minute step of
the run period it sends the readings to the agent and waits for the two
setpoints. Warm-up and sizing days run without the agent, at DEFAULT_ACTION.
"""
import sys, queue, threading
from pathlib import Path
import numpy as np

from bench_paths import ENERGYPLUS_DIR, WEATHER
sys.path.insert(0, str(ENERGYPLUS_DIR))
from pyenergyplus.api import EnergyPlusAPI

HERE = Path(__file__).resolve().parent
IDF = HERE / "5Zone_IdealLoadsAirSystems_ReturnPlenum.idf"
EPW = WEATHER

ZONES = ["SPACE1-1", "SPACE2-1", "SPACE3-1", "SPACE4-1", "SPACE5-1"]
IDEAL = [f"ZONE {i} Ideal Loads" for i in range(1, 6)]
OBS_NAMES = [f"T_{z} [C]" for z in ZONES] + ["T_out [C]", "heating [W]", "cooling [W]"]

HEAT_RANGE = (15.0, 22.0)        # heating setpoint limits [C]
COOL_RANGE = (23.0, 30.0)        # cooling setpoint limits [C]; never below heating
DEFAULT_ACTION = (21.0, 24.0)    # used on warm-up and sizing days
STEP_S = 300                     # 5-minute steps, matches Timestep,12 in the IDF
RUN_PERIOD = 3                   # kind_of_sim value of the weather-file run period
STOP = None                      # sent in place of an action to end a run early


class EnergyPlusEnv:
    def __init__(self, out_dir="out-env"):
        self.api = EnergyPlusAPI()
        self.ex = self.api.exchange
        self.out_dir = str(HERE / out_dir)
        self._thread = None
        self._state = None

    # ---- agent side -------------------------------------------------------
    def reset(self):
        """Start a new run and return the first observation."""
        self.close()
        st = self.api.state_manager.new_state()
        self._state = st
        self.api.runtime.set_console_output_status(st, False)
        for z in ZONES:
            self.ex.request_variable(st, "Zone Mean Air Temperature", z)
        self.ex.request_variable(st, "Site Outdoor Air Drybulb Temperature", "Environment")
        for i in IDEAL:
            self.ex.request_variable(st, "Zone Ideal Loads Zone Total Heating Rate", i)
            self.ex.request_variable(st, "Zone Ideal Loads Zone Total Cooling Rate", i)
        self.api.runtime.callback_begin_zone_timestep_after_init_heat_balance(st, self._callback)

        self._h = {}
        self._stopping = False
        self._done = False
        self._obs_q = queue.Queue()
        self._act_q = queue.Queue()
        self._thread = threading.Thread(target=self._run, args=(st,), daemon=True)
        self._thread.start()

        kind, payload = self._obs_q.get()
        if kind != "obs":
            self._finish(kind, payload)
            raise RuntimeError("EnergyPlus ended before the first step")
        self._obs = payload
        return payload

    def step(self, action):
        """Apply (heating, cooling) setpoints for 5 minutes.

        Returns (observation, reward, done, info). Reward is minus the heating
        plus cooling energy of the step [kWh]. The last step of a run returns
        done=True with the previous observation and zero reward, because
        EnergyPlus reports nothing after its final step.
        """
        if self._done:
            raise RuntimeError("episode is over; call reset()")
        self._act_q.put(self._clip(action))
        kind, payload = self._obs_q.get()
        if kind == "obs":
            self._obs = payload
            reward = -float(payload[-2] + payload[-1]) * STEP_S / 3.6e6
            return payload, reward, False, {}
        self._finish(kind, payload)
        return self._obs, 0.0, True, {}

    def close(self):
        """Stop a run that is still going and free its memory."""
        if self._thread is not None and self._thread.is_alive():
            self._stopping = True
            self._act_q.put(STOP)          # wakes the callback if it is waiting
            self._thread.join()
        self._thread = None
        if self._state is not None:
            self.api.state_manager.delete_state(self._state)
            self._state = None

    # ---- EnergyPlus side (background thread) ------------------------------
    def _run(self, st):
        code = self.api.runtime.run_energyplus(
            st, ["-w", str(EPW), "-d", self.out_dir, str(IDF)])
        self._obs_q.put(("done", code))

    def _callback(self, st):
        if self._stopping:
            return
        try:
            if not self.ex.api_data_fully_ready(st):
                return
            if not self._h:
                self._get_handles(st)
            if self.ex.warmup_flag(st) or self.ex.kind_of_sim(st) != RUN_PERIOD:
                self._apply(st, DEFAULT_ACTION)
                return
            self._obs_q.put(("obs", self._read(st)))
            action = self._act_q.get()
            if action is STOP:
                self._stopping = True
                self.api.runtime.stop_simulation(st)
                return
            self._apply(st, action)
        except Exception as e:
            self._stopping = True
            self._obs_q.put(("error", e))
            self.api.runtime.stop_simulation(st)

    def _get_handles(self, st):
        ex = self.ex
        h = {
            "zone": [ex.get_variable_handle(st, "Zone Mean Air Temperature", z) for z in ZONES],
            "out": [ex.get_variable_handle(st, "Site Outdoor Air Drybulb Temperature", "Environment")],
            "heat": [ex.get_variable_handle(st, "Zone Ideal Loads Zone Total Heating Rate", i) for i in IDEAL],
            "cool": [ex.get_variable_handle(st, "Zone Ideal Loads Zone Total Cooling Rate", i) for i in IDEAL],
            "heat_sp": ex.get_actuator_handle(st, "Schedule:Compact", "Schedule Value", "Htg-SetP-Sch"),
            "cool_sp": ex.get_actuator_handle(st, "Schedule:Compact", "Schedule Value", "Clg-SetP-Sch"),
        }
        flat = h["zone"] + h["out"] + h["heat"] + h["cool"] + [h["heat_sp"], h["cool_sp"]]
        if -1 in flat:
            raise RuntimeError(f"handle not found: {h}")
        step_s = ex.zone_time_step(st) * 3600
        if abs(step_s - STEP_S) > 1e-6:
            raise RuntimeError(f"zone time step is {step_s} s, expected {STEP_S} s")
        self._h = h

    def _read(self, st):
        v = lambda k: self.ex.get_variable_value(st, k)
        h = self._h
        temps = [v(k) for k in h["zone"] + h["out"]]
        heat = sum(v(k) for k in h["heat"])
        cool = sum(v(k) for k in h["cool"])
        return np.array(temps + [heat, cool], dtype=np.float32)

    def _apply(self, st, action):
        self.ex.set_actuator_value(st, self._h["heat_sp"], action[0])
        self.ex.set_actuator_value(st, self._h["cool_sp"], action[1])

    @staticmethod
    def _clip(action):
        heat = float(np.clip(action[0], *HEAT_RANGE))
        cool = float(np.clip(action[1], *COOL_RANGE))
        return heat, cool

    def _finish(self, kind, payload):
        """Handle the end of a run: join the thread and raise on failure."""
        self._done = True
        self._thread.join()
        if kind == "error":
            raise payload
        if kind == "done" and payload != 0:
            raise RuntimeError(f"EnergyPlus exited with code {payload}; see {self.out_dir}\\eplusout.err")
