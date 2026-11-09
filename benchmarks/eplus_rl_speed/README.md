# EnergyPlus vs SBSim: RL training speed

Scripts that train the same SAC agent on EnergyPlus and on SBSim and time where
the training time goes: the simulator, the agent and replay buffer, or the
gradient updates.

This is a working branch. The two buildings are not matched yet (SBSim runs
`sb1` in 1-day episodes, EnergyPlus runs a 5-zone example in 1-year episodes),
so timings from these scripts are not a fair comparison yet. No results are kept
in this folder.

## Requirements

- Windows. The scripts were written and run on Windows 11.
  `run_all_timings.py` uses Windows calls and has not been run elsewhere.
- EnergyPlus 26.2.0, which also provides `pyenergyplus` and the weather file.
- A Python 3.11 environment with SBSim's own dependencies. The runs so far used
  TensorFlow 2.15.1 and tf-agents 0.18.0.
- `psutil`, used by `run_all_timings.py`.

## Setup

`bench_paths.py` holds the two folders every script needs:

- `ENERGYPLUS_DIR`, the EnergyPlus install folder. The default is
  `C:\EnergyPlusV26-2-0`.
- `SBSIM_DIR`, the SBSim source folder. The default is this repository.

Change the defaults in that file, or set environment variables of the same
names.

## Steps

| Step | Script                         | What it does                                                           |
| ---- | ------------------------------ | ---------------------------------------------------------------------- |
| 2    | none (the EnergyPlus program)  | Runs the building for a full year from the command line, no control    |
| 3    | `run_api.py`                   | The same run through the EnergyPlus Python API                         |
| 4    | `run_control.py`               | Sets the two setpoints and reads the sensors at every 5-minute step    |
| 5    | `test_env.py`                  | Tests `eplus_env.py`, EnergyPlus as an environment with reset and step |
| 6b   | `test_tf_env.py`               | Tests `eplus_tf_env.py`, the same environment in SBSim's format        |
| 7a   | `check_sbsim_env.py`           | Loads SBSim's environment and times 12 steps at a fixed action         |
| 7b   | `train_bench.py --sim eplus`   | Trains the agent on EnergyPlus and times it                            |
| 7c   | `train_bench.py --sim sbsim`   | Trains the agent on SBSim and times it                                 |

The annual site energy must be the same on every run: 225.20 GJ without control
(steps 2 and 3) and 271.51 GJ at fixed setpoints of 21 and 24 C (steps 4, 5 and
6b).

## Running

Run from this folder, in the Python environment above.

```
python run_all_timings.py --quick
python run_all_timings.py
```

The first line is a short test of the script itself, with a few training steps.
The second is the full run.

`run_all_timings.py` runs every step as a fresh process and stops at the first
step that fails. It writes one folder per run to `out-all/`, with a log per step
and a `results.md` table. Output folders are not tracked by git.

Timing needs a quiet machine, so the script:

- waits until the whole machine is below 20% CPU before each step
- samples the CPU used by other programs during each step, and marks the step
  as busy if they average over 2 cores or pass 3 cores in any 10 seconds
- names the busiest other programs while they are above 2 cores
- refuses a full run on battery

Each script can also be run alone, for example
`python train_bench.py --sim eplus`.

## How the training differs from SBSim's `train.py`

`train_bench.py` follows
`smart_control/reinforcement_learning/scripts/train.py`, with these changes:

- A TF-Agents uniform replay buffer replaces Reverb, which has no Windows build.
- There is no starter buffer, evaluation, metrics or summaries. Each loop
  collects steps first, so the buffer has data before the first update.
- The actor network is rebuilt with the wrapper from the `SAC_Demo` notebook,
  because the one in `sac_networks.py` raises a `TypeError` on `network_state`
  with tf-agents 0.18.
- The time spent inside the simulator is measured separately.

## The EnergyPlus building

`5Zone_IdealLoadsAirSystems_ReturnPlenum.idf` comes from the example files of
EnergyPlus 26.2.0, with two changes:

- 12 timesteps per hour in place of 4, so one step is 5 minutes.
- One run period for the full year, 1 January to 31 December, in place of the
  two single-day run periods.

The weather file is Chicago O'Hare (TMY3), also shipped with EnergyPlus.
EnergyPlus's license is in `LICENSE-EnergyPlus.txt`.

The agent sets one heating and one cooling setpoint for all five zones, each
mapped from an action in [-1, 1]. It observes the five zone temperatures, the
outdoor temperature, and the total heating and cooling rates.
