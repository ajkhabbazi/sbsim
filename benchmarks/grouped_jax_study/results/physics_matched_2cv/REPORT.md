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

The logical CPUs used for the run were
`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`.

The grouped solver setting for each section was physics `sparse_lu`.

## Temperature Differences from Longwave Linearization and Grid Grouping

HVAC heat by diffuser was recorded from the 100-zone full nonlinear SBSim run.
The same heat was replayed in full linear and grouped linear SBSim to compare
the effects of longwave linearization and grid grouping. All three variants use
indoor and outdoor radiation and interior thermal mass.

| Comparison                        |  MAE K | RMSE K | 95th percentile K | Maximum K |
| --------------------------------- | -----: | -----: | ----------------: | --------: |
| Full linear vs. full nonlinear    | 0.0000 | 0.0001 |            0.0002 |    0.0008 |
| Grouped linear vs. full nonlinear | 0.0292 | 0.0391 |            0.0885 |    0.2269 |
| Grouped linear vs. full linear    | 0.0292 | 0.0390 |            0.0884 |    0.2267 |

| Heat balance | Solver      | States | Steps by LU/linear-solve count               | Subsequent ms/step |
| ------------ | ----------- | -----: | -------------------------------------------- | -----------------: |
| Full JAX     | `cg`        | 18,509 | 2 solves × 388 steps, 3 solves × 1,628 steps |             26.735 |
| Full linear  | `sparse_lu` | 18,509 | 1 solve × 2,016 steps                        |             16.606 |
| Grouped JAX  | `sparse_lu` |    789 | 1 solve × 2,016 steps                        |              1.591 |

The full nonlinear baseline timing includes HVAC feedback and heat recording;
timing for the two linear models includes fixed-heat replay and zone-temperature
reads.

![Representative-zone average temperature and zone-average error over time](physics.png)

The top panel directly overlays Full JAX and Grouped JAX in one representative
zone, selected by its largest time-average absolute error. Every temperature is
a model-computed zone average in °C, not an individual cell temperature. The
bottom panel shows the mean absolute Full/Grouped difference over all zones at
each time in K. No quantile curve is plotted.

Five-minute temperature time series: `physics.csv`; error statistics:
`physics_errors.csv`; temperatures and HVAC heat: `physics.npz`

Environment and aggregate values: `summary.json`
