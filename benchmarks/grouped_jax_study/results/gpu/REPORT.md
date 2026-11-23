[한국어](REPORT_ko.md)

# Full and Grouped SBSim Compared with Detailed EnergyPlus

The same weather from USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw was used for
7 days starting on 2024-07-15. Zones measure 5 × 4 m, the grid spacing is 0.5 m,
and the exterior-window width is 2.0 m. The time interval is 5 minutes, with
2,016 steps per episode.

JAX 0.4.34 ran on NVIDIA GeForce RTX 3080. The detailed EnergyPlus input version
is `24.2`.

The full-node and grouped-node SBSim models use the same envelope, air, and
thermal-mass properties and VAV system. Detailed EnergyPlus has separate
envelope, load, and HVAC implementations, so temperature differences between
engines also include these modeling differences.

The logical CPUs used for the run were
`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`.

The grouped solver setting for each section was timing `dense`, physics `cg`,
sac `dense`.

## Step Time at Fixed Setpoints

A 21–24 °C setpoint range was applied on weekdays from 07:00–19:00 and 19–26 °C
at all other times. The average excluding the first step includes setpoint
application, simulation, observation, and EnergyPlus process communication.

| Zones | Engine      | Initialization s | Reset s | First step s | Subsequent ms/step | States |
| ----: | ----------- | ---------------: | ------: | -----------: | -----------------: | -----: |
|    16 | Full JAX    |            1.463 |   0.001 |        0.017 |             14.138 |   2945 |
|    16 | Grouped JAX |            0.165 |   0.001 |        0.007 |              1.749 |    129 |
|    16 | EnergyPlus  |            2.317 |   1.683 |        0.002 |              0.764 |      — |
|    36 | Full JAX    |            1.733 |   0.001 |        0.029 |             14.280 |   6565 |
|    36 | Grouped JAX |            0.235 |   0.001 |        0.004 |              1.984 |    253 |
|    36 | EnergyPlus  |            2.030 |   3.008 |        0.002 |              1.194 |      — |
|   100 | Full JAX    |            9.276 |   0.002 |        0.029 |             15.663 |  18101 |
|   100 | Grouped JAX |            0.487 |   0.002 |        0.006 |              2.551 |    621 |
|   100 | EnergyPlus  |            1.974 |   6.541 |        0.003 |              2.853 |      — |

At 100 zones, the full JAX and EnergyPlus step times were 6.14 and 1.12 times
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

| Heat balance | Solver | States | Steps by LU/linear-solve count               | Subsequent ms/step |
| ------------ | ------ | -----: | -------------------------------------------- | -----------------: |
| Full JAX     | `cg`   | 18,101 | 2 solves × 478 steps, 3 solves × 1,538 steps |             15.878 |
| Full linear  | `cg`   | 18,101 | 1 solve × 2,016 steps                        |              9.703 |
| Grouped JAX  | `cg`   |    621 | 1 solve × 2,016 steps                        |              2.432 |

The full nonlinear baseline timing includes HVAC feedback and heat recording;
timing for the two linear models includes fixed-heat replay and zone-temperature
reads.

![Representative-zone temperature and temperature error over time](physics.png)

Five-minute temperature time series: `physics.csv`; error statistics:
`physics_errors.csv`; temperatures and HVAC heat: `physics.npz`

Environment and aggregate values: `summary.json`
