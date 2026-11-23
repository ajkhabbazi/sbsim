"""Scaling benchmark: Picard + radiation + interior mass, three linear solvers.

  python -m smart_control.simulator.benchmark_solvers --tiles 2 6 --steps 3 \
      --solvers numpy_scipy cached_lu jax_cg

The floor plan of the small test building is tiled (rows x cols) to scale the
number of control volumes. Every solver advances the same building through the
same steps; the first is the reference for the temperature differences.
"""

from __future__ import annotations

import argparse
import copy
import time
from unittest import mock

import numpy as np
import pandas as pd

from smart_control.simulator import building as building_py
from smart_control.simulator import cached_lu_simulator
from smart_control.simulator import numpy_scipy_simulator
from smart_control.simulator import simulator_flexible_floor_plan_test as fixture_mod
from smart_control.simulator import weather_controller as weather_controller_py


def make_building(tile_rows: int, tile_cols: int, radiation: bool, mass: bool):
  fx = fixture_mod.FlexibleFloorplanSimulatorTest()
  plan = np.tile(fx._create_dummy_floor_plan_small(), (tile_rows, tile_cols))  # pylint: disable=protected-access
  inside_air = building_py.MaterialProperties(
      conductivity=50.0, heat_capacity=1.0, density=1.2
  )
  inside_wall = building_py.MaterialProperties(
      conductivity=2.0, heat_capacity=500.0, density=1800.0
  )
  exterior = building_py.MaterialProperties(
      conductivity=0.05, heat_capacity=500.0, density=3000.0
  )
  interior_mass = building_py.MaterialProperties(
      conductivity=0.5, heat_capacity=1000.0, density=2000.0
  )
  return building_py.FloorPlanBasedBuilding(
      cv_size_cm=20.0,
      floor_height_cm=300.0,
      initial_temp=292.0,
      inside_air_properties=inside_air,
      inside_wall_properties=inside_wall,
      building_exterior_properties=exterior,
      floor_plan=plan,
      zone_map=copy.deepcopy(plan),
      buffer_from_walls=0,
      interior_mass_properties=interior_mass,
      include_interior_mass=mass,
      include_radiative_heat_transfer=radiation,
      view_factor_method='ScriptF' if radiation else None,
  )


_GROUPED_METHODS = {
    'grouped_jax': 'sparse_lu',
    'grouped_jax_cg': 'cg',
    'grouped_jax_dense': 'dense',
    'grouped_jax_spsolve': 'spsolve',
}


def make_sim(kind: str, building, *, grouped_platform='cpu'):
  fx = fixture_mod.FlexibleFloorplanSimulatorTest()
  args = (
      building,
      fx._create_small_hvac(),  # pylint: disable=protected-access
      mock.create_autospec(weather_controller_py.WeatherController),
      300.0,
      1e-4,
      2000,
      2000,
      pd.Timestamp('2012-06-21 06:00:00'),
  )
  if kind == 'numpy_scipy':
    return numpy_scipy_simulator.NumpyScipySimulator(
        *args, relative_convergence_threshold=None
    )
  if kind == 'cached_lu':
    return cached_lu_simulator.CachedLUSimulator(
        *args, relative_convergence_threshold=None
    )
  # Local simulator imports keep JAX optional for NumPy-only comparisons.
  # pylint: disable=import-outside-toplevel
  if kind == 'jax_cg':
    from smart_control.simulator import jax_device_resident_simulator as j  # pylint: disable=g-import-not-at-top

    return j.JaxDeviceResidentSimulator(
        *args, relative_convergence_threshold=None, cg_tolerance=1e-10
    )
  if kind in _GROUPED_METHODS:
    from smart_control.simulator import grouped_jax_simulator

    return grouped_jax_simulator.GroupedJaxSimulator(
        *args,
        relative_convergence_threshold=None,
        solver_method=_GROUPED_METHODS[kind],
        platform=grouped_platform,
    )
  raise ValueError(kind)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--tiles', nargs=2, type=int, default=[2, 4])
  p.add_argument('--steps', type=int, default=3)
  p.add_argument(
      '--solvers', nargs='+', default=['numpy_scipy', 'cached_lu', 'jax_cg']
  )
  p.add_argument('--no-radiation', action='store_true')
  p.add_argument('--no-mass', action='store_true')
  p.add_argument('--grouped-platform', choices=('cpu', 'gpu'), default='cpu')
  a = p.parse_args()

  t0 = time.time()
  base = make_building(
      a.tiles[0], a.tiles[1], not a.no_radiation, not a.no_mass
  )
  n = base.temp.size
  print(
      f'tiles {a.tiles}: grid {base.temp.shape} = {n} CVs, build'
      f' {time.time() - t0:.1f}s',
      flush=True,
  )
  ref_temp = None
  for kind in a.solvers:
    b = copy.deepcopy(base)
    t0 = time.time()
    sim = make_sim(kind, b, grouped_platform=a.grouped_platform)
    t_init = time.time() - t0
    for src in np.argwhere(b.diffusers > 0.0)[
        :: max(1, int((b.diffusers > 0).sum()) // 20)
    ]:
      b.input_q[tuple(src)] = 50.0
    if kind == 'jax_cg':
      sim.warmup(ambient_temperature=273.0, convection_coefficient=12.0)
    times, outer = [], []
    for _ in range(a.steps):
      t0 = time.time()
      ok = sim.finite_differences_timestep(
          ambient_temperature=273.0, convection_coefficient=12.0
      )
      if kind == 'jax_cg':
        sim.sync_state_to_building()
      times.append(time.time() - t0)
      outer.append(getattr(sim, 'last_outer_iteration_count', None))
      if not ok:
        print('   (did not converge)')
    extra = ''
    if kind == 'cached_lu':
      extra = (
          f'  LUs {sim.factorization_count}  triangular solves'
          f' {sim.triangular_solve_count}'
      )
    if kind == 'jax_cg':
      extra = (
          f'  CG iters/solve {sim.last_cg_iteration_count}  outer'
          f' {sim.last_outer_iteration_count}'
      )
    if kind in _GROUPED_METHODS:
      extra = f'  nodes {sim.thermal_node_count}  solves per step {outer}'
    if ref_temp is None:
      ref_temp, diff = b.temp.copy(), 0.0
    else:
      diff = float(np.max(np.abs(b.temp - ref_temp)))
    print(
        f'{kind:12s} init {t_init:6.1f}s  step times'
        f' {[round(x, 2) for x in times]}  max|dT| vs first {diff:.2e}{extra}',
        flush=True,
    )


if __name__ == '__main__':
  main()
