# SBSim Full-Grid and Grouped-Grid Heat Balances

**English** | [한국어](README_ko.md)

`GroupedJaxSimulator` uses existing SBSim floor plans, HVAC, weather, and
`step_sim()`. It applies the same finite-volume heat balance and per-zone indoor
longwave equations to full and grouped grids. Indoor longwave radiation,
exterior radiation, interior thermal mass, longwave linearization, and the
solver can be selected to compare computational cost and temperature
differences.

In this directory, [`run.py`](run.py) runs the measurements and
[`report.py`](report.py) creates tables, CSV files, and figures. `results/`
contains raw data and reports generated through this API. The office floor-plan,
SBSim, and EnergyPlus model builders are also copied into the local
[`matched_vav/`](matched_vav/) package. The EnergyPlus and SAC adapters use the
local [`eplus_env.py`](eplus_env.py), [`physics.py`](physics.py),
[`train.py`](train.py), and [`sac.py`](sac.py).

## Standalone Study Layout

**This benchmark runs without `benchmarks/eplus_rl_speed` or
`benchmarks/sbsim_sac_study`.** Both originals remain in place; only the
required runtime modules, model inputs, and trainer code were copied here.
`test_standalone.py` checks source references to those paths, the location and
presence of default model inputs, and imports while those packages are blocked.
It also loads both default 100-zone manifests, geometry/NPY inputs, and run
date, and verifies that the stored `model.epJSON` exactly matches the local
generator's output. This regression test does not launch an EnergyPlus
subprocess.

```text
grouped_jax_study/
├── matched_vav/             # Office, SBSim, and EnergyPlus model/runtime modules
│   ├── models/zone05/       # Input for migrated focused tests
│   ├── models/zone16/
│   ├── models/zone36/
│   ├── models/zone100/
│   ├── models/zone100_office/            # Default SBSim input for train.py
│   └── models/zone100_detailed_office/   # Detailed EnergyPlus input
├── run.py / report.py       # Full benchmark and report generation
├── physics.py / eplus_env.py
├── train.py / sac.py / sac_probe.py
├── interior_mass.py / jax_env.py  # Local modules for the migrated batched trainer
├── figures/                 # Shared English/Korean guide figures
└── results/                 # Existing measurements; use a new subdirectory for new runs
```

Standalone here means no import or path dependency on the two sibling studies.
The heat-balance implementation remains in
`smart_control.simulator.grouped_heat_balance` and `grouped_jax_simulator`, and
the project's Python dependencies are still required. Running EnergyPlus also
requires an external EnergyPlus 24.2 installation with `pyenergyplus` and an EPW
weather file. `run.py` stores generated models under the new output directory;
`train.py` searches the local `matched_vav/models/` directory by default.

## Using with Existing SBSim Models

Create `FloorPlanBasedBuilding`, `FloorPlanBasedHvac`, and the weather
controller in existing code, then change only the simulator class. In the
example below, `floorplan_building`, `hvac`, `weather`, and `start_timestamp`
are the objects and start time from that code.

```python
from smart_control.simulator.grouped_heat_balance import InteriorMass
from smart_control.simulator.grouped_jax_simulator import GroupedJaxSimulator

sim = GroupedJaxSimulator(
    building=floorplan_building,
    hvac=hvac,
    weather_controller=weather,
    start_timestamp=start_timestamp,
    time_step_sec=300,
    convergence_threshold=1e-6,
    iteration_limit=100,
    iteration_warning=100,
    grouped=True,
    include_interior_radiative_heat_transfer=True,
    include_exterior_radiative_heat_transfer=True,
    include_interior_mass=True,
    interior_mass=InteriorMass(),
    linearized_radiative_heat_transfer=True,
    platform='cpu',
    solver_method='sparse_lu',
)
sim.reset()
h = weather.get_air_convection_coefficient(start_timestamp)
sim.prepare_thermal_system(h)
sim.step_sim()
zone_temperatures_k = sim.building.get_zone_average_temps()
```

`prepare_thermal_system(h)` is optional. If omitted, preparation occurs during
the first heat-balance step. Assembly and preparation results are reused while
the outdoor-air convection coefficient $h$ remains unchanged. Set the start time
and `time_step_sec` within the range supported by the weather data.

`TFSimulator` and `GroupedJaxSimulator` share the calling convention of
`SimulatorFlexibleGeometries`. In existing floor-plan-based RL code, pass the
`sim` above to `SimulatorBuilding(simulator=sim, occupancy=occupancy)`. Existing
observations, actions, and VAV calls can be used unchanged. When running the two
models side by side, create separate Building and HVAC instances for each solver
so that their mutable state does not mix. Set `solver_method` to a value
supported by each solver.

| Option                                     | Default                 | Meaning                                                                                                                                                                                                                     |
| ------------------------------------------ | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `grouped`                                  | `True`                  | Groups connected air and wall/window segments with uniform boundaries. `False` keeps a temperature state for every active cell.                                                                                             |
| `include_interior_radiative_heat_transfer` | `None`                  | Indoor longwave radiation. `None` follows `Building.include_radiative_heat_transfer`.                                                                                                                                       |
| `include_exterior_radiative_heat_transfer` | `None`                  | Exterior longwave radiation and absorbed/transmitted solar radiation. `None` follows `Building.include_exterior_radiative_heat_transfer`.                                                                                   |
| `include_interior_mass`                    | `None`                  | Interior thermal mass. `None` chooses based on whether `InteriorMass` is specified or on `Building.include_interior_mass`. `False` disables mass.                                                                           |
| `interior_mass`                            | `None`                  | If specified, uses effective properties per unit floor area. If omitted, uses the existing Building mass properties when enabled. If mass is enabled and the Building has no properties, uses the default `InteriorMass()`. |
| `linearized_radiative_heat_transfer`       | `True`                  | Uses a fixed slope at 22.5 °C and a correction based on the previous temperature. `False` iteratively applies longwave radiation at the updated temperature.                                                                |
| `platform` / `solver_method`               | `'cpu'` / `'sparse_lu'` | Selects the computing device and linear-system solver.                                                                                                                                                                      |

`wall_depth_cells` takes wall thickness in cells and defaults to 1. For example,
use two layers to represent a thickness of 0.50 m with 0.25 m cells. In grouped
mode, `air_block_cells` can restrict connected air to smaller grid blocks.

The default effective properties of `InteriorMass()` are a heat capacity of 30
kJ/(m²·K), an air exchange coefficient of 3 W/(m²·K), a radiation-exposed area
equal to 1 times the floor area, and an emissivity of 0.90. Change them with
`capacity_per_m2`, `air_conductance_per_m2`, `exposed_area_per_m2`, and
`emissivity`. These values represent the effects of furniture and other interior
objects and should be treated as model inputs when calibration against
measurements is needed. Enabling exterior radiation uses the weather
controller's radiation inputs and the Building's exterior-wall and window
optical properties.

For inputs with windows, first set
`include_exterior_radiative_heat_transfer=True` on the Building so that windows
are constructed as boundary surfaces. The solver's exterior-radiation switch can
then enable or disable the calculation. When using the Building's existing
volumetric mass properties, pad the perimeter of the floor plan with outdoor-air
cells. An `InteriorMass()` supplied as a solver option uses its own properties
per unit floor area.

After preparation, `thermal_node_count` and `mass_node_count` report the state
counts, and `group_map` maps each cell to its temperature-state index. A
`group_map` value of `-1` denotes an outdoor-air boundary. After a step,
`state_temperatures` contains the Kelvin temperatures of all states, while
`mass_temperature` displays temperatures at mass-enabled locations on the
original grid. Use `sim.building.temp` for existing visualizations.
`last_solve_count` is the number of linear-system solves in one step and is
distinct from the CG solver's internal iteration count.

## Solvers

Grouped mode sums the heat balances of cells that share a temperature to form
`Pᵀ A P`. Collecting the indoor longwave area-ratio terms and exterior longwave
terms in `R`, a linear step solves air, walls, windows, and mass together in one
sparse heat balance, `A₀ + sR`. The fixed slope is `s = 4 × 295.65³`. Previous
temperatures, weather, shortwave radiation, and HVAC heat enter the right-hand
side.

| Solver      | CPU           | GPU      | Processing                                                                                                    |
| ----------- | ------------- | -------- | ------------------------------------------------------------------------------------------------------------- |
| `sparse_lu` | SciPy         |          | Reuses sparse LU. Linear models use one direct solve per step.                                                |
| `cg`        | JAX           | JAX      | Sparse conjugate gradient. Linear models perform internal CG iterations without outer longwave iterations.    |
| `dense`     | JAX           | JAX      | Prepares a dense response with sparse LU on the host, then applies it by matrix multiplication on the device. |
| `spsolve`   | SciPy via JAX | JAX CUDA | Sparse direct solve. CUDA uses sparse QR.                                                                     |

For GPU execution, set `platform='gpu'` and select a JAX solver in an
environment where JAX recognizes a supported device. A single-model
`GroupedJaxSimulator` runs the existing host-side VAV and writes results back to
the host-side Building grid after each step. Earlier `JaxOffice` results that
ran SAC and multi-building batches on the device are measurements from a
separate implementation.

For matrix derivations and examples, see
[`benchmarks/grouped_jax_study/HEAT_BALANCE_EQUATIONS.md`](HEAT_BALANCE_EQUATIONS.md).
For group maps of the original floor plan and earlier SAC results, see
[`benchmarks/grouped_jax_study/LUMPED_MODEL.md`](LUMPED_MODEL.md). For the
measured spatial/temporal error tail, reduced-state mechanics, matched
linear/nonlinear sparse-LU supplement, and the distinction between fixed-HVAC
physics and independent closed-loop rollouts, see the detailed
[full-versus-grouped analysis](GROUPED_FULL_ANALYSIS.md) and its
[CSV/JSON/PNG artifacts](results/grouped_full_analysis/analysis_summary.json).
In the saved 100-zone case, restoring grouped nonlinear radiation leaves the
0.102 K-scale discrepancy nearly unchanged; separate air, envelope, and mass
contributions remain unassigned.

## Running on a Small Synthetic Floor Plan

From the repository root, the following command applies four solvers to a small
synthetic floor plan without a weather file. First-step timing includes deferred
matrix preparation and any required JAX compilation. It also prints later-step
timings and the maximum absolute grid-temperature difference from the first
solver.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m smart_control.simulator.benchmark_solvers --tiles 1 2 --steps 3 \
  --solvers grouped_jax grouped_jax_cg grouped_jax_dense grouped_jax_spsolve
```

For GPU execution, specify only JAX solvers and add `--grouped-platform gpu`.

## Reproducing the Office Benchmark

The weekly comparison uses a common office floor plan and the Chicago O’Hare
TMY3 EPW. The default is 7 days starting July 15, 2024, with 5-minute steps and
a 0.50 m SBSim grid. The three SBSim variants use the same area-ratio indoor
longwave model, exterior radiation, effective interior thermal mass, and VAV
system. EnergyPlus applies a detailed office HVAC and envelope model to the
common zone layout. Set `EPW` to the weather-file path and `EPLUS_DIR` to the
EnergyPlus installation path containing `pyenergyplus`.

Set the installation paths in the terminal where the CPU and GPU commands will
run. The paths below are for the EnergyPlus 24.2 installation restored on this
machine and include the same EPW used in the CPU report. On another machine, set
the installation and EPW paths for that environment.

```bash
EPLUS_DIR="$HOME/.local/share/sbsim/energyplus-24.2" EPW="$EPLUS_DIR/WeatherData/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"
test -r "$EPW" && test -r "$EPLUS_DIR/pyenergyplus/api.py"
```

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 8 .venv/bin/python -m benchmarks.grouped_jax_study.run \
  --epw "$EPW" --energyplus-dir "$EPLUS_DIR" \
  --zones 16 36 100 --physics-zones 100 --days 7 \
  --platform cpu --grouped-method dense --physics-grouped-method sparse_lu \
  --output /tmp/sbsim-grouped-jax-study
```

`taskset -c 8` is an example of pinning a run to one logical CPU on Linux.
Replace it with a core number available to the current process. Use a new
`--output` path for a new run. Select individual sections with
`--section timing`, `--section physics`, or `--section sac`; repeat the option
to run multiple sections. If omitted, all three sections run. `--energyplus-dir`
can be omitted when running only `physics`. The default grouped solver for study
runs is `dense`. In the command above, `--physics-grouped-method sparse_lu` uses
the CPU sparse direct solver only for the physics replay.

After stopping the previous process, resume an interrupted run by adding
`--resume` with the same configuration and `--output`. Completed measurement
sections and saved individual timing and SAC cases are reused.
`--section sac --resume` runs only SAC while preserving existing timing and
temperature results. Intermediate training state is not saved, so interrupted
episodes restart from the beginning and any remaining logs are preserved under
`interrupted/`. SAC reports environment-setup completion, learner-initialization
start and completion, the start of the first update, and subsequent collection
progress.

If SAC training stalls for a long time, add `--sac-debug` to the same command.
`sac/<zones>-<engine>/sac_debug.log` records batch transfers and JAX calls for
the first batch of updates, JAX memory-allocation statistics for a GPU learner,
and Python errors. During training, Python call stacks are also appended to the
same file every 45 seconds. For SAC cases using a GPU, enabling `--sac-debug`
records total VRAM, used and free memory, and GPU utilization every 2 seconds in
`sac/<zones>-<engine>/gpu_telemetry.csv` through a separate `nvidia-smi`
process. Entries unsupported under WSL may be recorded as `N/A`. JAX allocations
and total device usage describe the process allocator and the entire device,
respectively.

[`sac_probe.py`](sac_probe.py) performs 16 SAC updates with the same 17
observations, 32 actions, and batch size of 256 as a 16-zone run. It uses a
synthetic trajectory and opens neither a simulator nor a weather file. If it
does not finish within 60 seconds, it prints the most recent call stack and
exits.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/jax_cuda/bin/python -m benchmarks.grouped_jax_study.sac_probe
```

`--platform gpu --sac-learner-platform cpu` uses the GPU for SBSim heat balances
while performing SAC policy execution and updates on the CPU. The same CPU
learner is used for Full JAX, Grouped JAX, and EnergyPlus. Result tables record
both the heat-balance device and the learning device. A run stalled during GPU
learning can continue with these settings together with
`--section sac --resume`. If completed SAC results used a different learning
device, specify a new `--output` to retain both measurements.

| Section   | Comparison                                                                                                                                                                                                                                       |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `timing`  | Runs full-grid nonlinear JAX CG, grouped-grid linear SBSim, and detailed EnergyPlus at multiple zone sizes. Applies the same setpoint sequence and records preparation, reset, first-step, and subsequent-step timings.                          |
| `physics` | Records temperatures and actual per-diffuser HVAC heat from full nonlinear SBSim. Replays the same heat through full linear and grouped linear models to calculate temperature differences. Full linear uses sparse LU on CPU and JAX CG on GPU. |
| `sac`     | Applies the same JAX SAC training configuration to full nonlinear SBSim, grouped linear SBSim, and detailed EnergyPlus. Records environment steps, policy execution, and learning updates separately.                                            |

By default, each SAC run collects 2,016 transitions in one episode per model and
performs 8 learning updates every 96 transitions with batch size 256, for 168
updates in total. Observations are per-zone air temperatures and outdoor
temperature, actions are per-zone heating and cooling setpoints, and the reward
is the temperature deviation outside a fixed comfort range. The experiment
measures throughput through run completion; policy convergence and savings are
outside its scope.

Fixed-heat replay isolates the effects of the grid and longwave approximation.
Reference temperatures are recorded during a run with responsive VAV control,
while the two candidates reuse the heat inputs from that run. Therefore, the
absolute runtimes of `physics` and the closed-loop step timings of `timing` must
be interpreted with their respective measurement scopes. In closed-loop
operation, even small temperature differences can change VAV control switching
and the heat input at the next step. Read throughput together with the seed,
zone sizes, solvers, platforms, and episode length used.

## Reading the Result Files

The combined timing, physics, and SAC report for the current CPU measurements is
[`results/cpu/REPORT.md`](results/cpu/REPORT.md), with three figures embedded in
the document. Raw data from the individual runs are in
[`results/timing_cpu/`](results/timing_cpu/),
[`results/physics_cpu/`](results/physics_cpu/), and
[`results/sac_cpu/`](results/sac_cpu/). Rebuild the combined report from these
three directories with
`python -m benchmarks.grouped_jax_study.report --output <new-directory> <three-directories>`.

The following paths are relative to the directory passed as `--output`. Only
results for the sections run are generated.

| Path                                  | Contents                                                                                                                           |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `REPORT.md`, `REPORT_ko.md`           | English and Korean reports with run conditions, result tables, and figures                                                         |
| `summary.json`                        | EPW SHA-256, devices and versions, mass properties, timing and control settings, and results by section                            |
| `timing.csv`, `timing.png`            | Preparation, reset, and step timings by zone count and engine. Step means and 95th percentiles exclude the first step.             |
| `timing/<zones>-<engine>/rollout.npz` | Runtime and zone temperatures for each 5-minute step                                                                               |
| `physics.csv`, `physics.png`          | End time of each 5-minute interval, per-zone temperatures, and plots of temperatures in a representative zone and errors over time |
| `physics_errors.csv`, `physics.npz`   | Temperature-error summaries by comparison, original temperature time series, and diffuser heat for replay                          |
| `sac.csv`, `sac.png`                  | Transition and update counts, environment and learning-stage timings, and total runtime                                            |
| `sac/<zones>-<engine>/actor.npz`      | Policy neural-network weights from each SAC run                                                                                    |

In `physics`, let $e_{t,z}=|T_{\mathrm{candidate},t,z}-T_{\mathrm{base},t,z}|$,
where `base` is the full nonlinear model and each $T_{t,z}$ is a **zone-average
temperature**, not an individual finite-volume cell temperature. The top panel
in `physics.png` directly overlays **Full JAX and Grouped JAX** in one
representative zone, chosen by its largest time-averaged absolute grouped-model
error. Both curves are model-computed zone averages in °C. The bottom panel
shows the mean absolute Full/Grouped difference over all zones at each time,
$\mathrm{MAE}(t)=\frac{1}{Z}\sum_z e_{t,z}$, in K. No quantile curve is plotted.
The scalar MAE, RMSE, p95, and maximum in the error table still use all time ×
zone samples together. CSV and NPZ temperature series retain the same units and
zone ordering. The stored Full-linear statistics remain separate numerical
evidence about longwave linearization; Full linear is not an additional curve in
this direct Full/Grouped temperature comparison.

For SBSim, `setup_s` includes the `prepare_thermal_system()` call made in
advance, while `thermal_prepare_s` separately measures that portion. EnergyPlus
initialization and weather warmup are included in reset. EnergyPlus runs native
HVAC and heat balances on the CPU, so its compute device remains CPU even when
`--platform gpu` is selected. In that case, platform selection applies to the
JAX solvers and SAC training.

### Reproducing on GPU

`JAX_GPU_PYTHON` is a Python installation with CUDA-enabled JAX. In an
environment with the CUDA overlay, use `.venv/jax_cuda/bin/python` from the
repository root. `EPW` and `EPLUS_DIR` are the same inputs as in the CPU example
above.

```bash
unset JAX_PLATFORMS
JAX_GPU_PYTHON="${JAX_GPU_PYTHON:-$PWD/.venv/jax_cuda/bin/python}"
"$JAX_GPU_PYTHON" -c 'import jax; print(jax.devices("gpu"))' && \ OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$JAX_GPU_PYTHON" \
  -m benchmarks.grouped_jax_study.run \
  --epw "$EPW" --energyplus-dir "$EPLUS_DIR" \
  --zones 16 36 100 --physics-zones 100 --days 7 --platform gpu \
  --grouped-method dense --physics-grouped-method cg \
  --output benchmarks/grouped_jax_study/results/gpu
```

After each section completes, `summary.json` and both `REPORT.md` and
`REPORT_ko.md` are updated. To measure sparse direct solves on GPU, specify
`--physics-grouped-method spsolve` in an environment with CUDA `spsolve`
support. The currently saved
[`results/gpu/summary.json`](results/gpu/summary.json) and
[`results/gpu/REPORT.md`](results/gpu/REPORT.md) contain timing and physics
results using JAX 0.4.34 on an NVIDIA GeForce RTX 3080. No completed GPU SAC
results have been added to this directory yet.
