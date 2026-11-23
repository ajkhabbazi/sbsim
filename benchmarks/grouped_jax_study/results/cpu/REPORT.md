[한국어](REPORT_ko.md)

# Full and Grouped SBSim Compared with Detailed EnergyPlus

The same weather from USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw was used for
7 days starting on 2024-07-15. Zones measure 5 × 4 m, the grid spacing is 0.5 m,
and the exterior-window width is 2.0 m. The time interval is 5 minutes, with
2,016 steps per episode.

JAX 0.4.34 ran on cpu. The detailed EnergyPlus input version is `24.2`.

The full-node and grouped-node SBSim models use the same envelope, air, and
thermal-mass properties and VAV system. Detailed EnergyPlus has separate
envelope, load, and HVAC implementations, so temperature differences between
engines also include these modeling differences.

The logical CPUs used for the separate runs were timing `[4]`, physics `[8]`,
sac `[6]`.

The grouped solver setting for each section was timing `dense`, physics
`sparse_lu`, sac `dense`.

The raw-data directories for the individual runs were timing `../timing_cpu`,
physics `../physics_cpu`, sac `../sac_cpu`.

## Step Time at Fixed Setpoints

A 21–24 °C setpoint range was applied on weekdays from 07:00–19:00 and 19–26 °C
at all other times. The average excluding the first step includes setpoint
application, simulation, observation, and EnergyPlus process communication.

| Zones | Engine      | Initialization s | Reset s | First step s | Subsequent ms/step | States |
| ----: | ----------- | ---------------: | ------: | -----------: | -----------------: | -----: |
|    16 | Full JAX    |            0.762 |   0.002 |        0.008 |              9.060 |   2945 |
|    16 | Grouped JAX |            0.285 |   0.002 |        0.004 |              1.838 |    129 |
|    16 | EnergyPlus  |            3.725 |   2.317 |        0.003 |              1.564 |      — |
|    36 | Full JAX    |            0.969 |   0.002 |        0.015 |             16.919 |   6565 |
|    36 | Grouped JAX |            0.435 |   0.001 |        0.003 |              2.505 |    253 |
|    36 | EnergyPlus  |            3.832 |   5.079 |        0.005 |              3.101 |      — |
|   100 | Full JAX    |            1.645 |   0.003 |        0.037 |             34.624 |  18101 |
|   100 | Grouped JAX |            0.503 |   0.001 |        0.003 |              2.400 |    621 |
|   100 | EnergyPlus  |            2.591 |  19.924 |        0.010 |              7.493 |      — |

At 100 zones, the full JAX and EnergyPlus step times were 14.42 and 3.12 times
the grouped JAX step time, respectively.

![Step time by grid and engine](timing.png)

Measurements: `timing.csv`

## Temperature Differences from Longwave Linearization and Grid Grouping

HVAC heat by diffuser was recorded from the 100-zone full nonlinear SBSim run.
The same heat was replayed in full linear and grouped linear SBSim to compare
the effects of longwave linearization and grid grouping. All three variants use
indoor and outdoor radiation and interior thermal mass.

| Comparison                        |  MAE K | RMSE K | 95th percentile K | Maximum K |
| --------------------------------- | -----: | -----: | ----------------: | --------: |
| Full linear vs. full nonlinear    | 0.0001 | 0.0002 |            0.0006 |    0.0022 |
| Grouped linear vs. full nonlinear | 0.1022 | 0.1505 |            0.3500 |    0.5597 |
| Grouped linear vs. full linear    | 0.1022 | 0.1505 |            0.3497 |    0.5586 |

| Heat balance | Solver      | States | Steps by LU/linear-solve count               | Subsequent ms/step |
| ------------ | ----------- | -----: | -------------------------------------------- | -----------------: |
| Full JAX     | `cg`        | 18,101 | 2 solves × 478 steps, 3 solves × 1,538 steps |             42.561 |
| Full linear  | `sparse_lu` | 18,101 | 1 solve × 2,016 steps                        |              8.672 |
| Grouped JAX  | `sparse_lu` |    621 | 1 solve × 2,016 steps                        |              0.607 |

The full nonlinear baseline timing includes HVAC feedback and heat recording;
timing for the two linear models includes fixed-heat replay and zone-temperature
reads.

![Representative-zone temperature and temperature error over time](physics.png)

Five-minute temperature time series: `physics.csv`; error statistics:
`physics_errors.csv`; temperatures and HVAC heat: `physics.npz`

## Actual SAC Training

JAX SAC was trained for one episode per environment with seed 0. 8 updates were
performed every 96 transitions, with batch size 256. Heating and cooling
setpoints were controlled independently in each zone. The reward was the
negative zone-average amount by which temperatures fell outside 21–24 °C.

| Zones | Engine      | Simulator device | Learner device | Transitions | Updates | Total s | Simulator ms/transition |
| ----: | ----------- | ---------------- | -------------- | ----------: | ------: | ------: | ----------------------: |
|    16 | Full JAX    | CPU              | CPU            |       2,016 |     168 |   30.45 |                   9.626 |
|    16 | Grouped JAX | CPU              | CPU            |       2,016 |     168 |    8.66 |                   2.170 |
|    16 | EnergyPlus  | CPU              | CPU            |       2,016 |     168 |   19.65 |                   3.125 |
|    36 | Full JAX    | CPU              | CPU            |       2,016 |     168 |   44.64 |                  16.666 |
|    36 | Grouped JAX | CPU              | CPU            |       2,016 |     168 |   10.07 |                   2.644 |
|    36 | EnergyPlus  | CPU              | CPU            |       2,016 |     168 |   14.66 |                   2.275 |
|   100 | Full JAX    | CPU              | CPU            |       2,016 |     168 |   89.46 |                  39.390 |
|   100 | Grouped JAX | CPU              | CPU            |       2,016 |     168 |   14.84 |                   4.128 |
|   100 | EnergyPlus  | CPU              | CPU            |       2,016 |     168 |   37.94 |                   8.472 |

![JAX SAC runtime by engine and zone count](sac.png)

SAC timing and training statistics: `sac.csv`

Environment and aggregate values: `summary.json`
