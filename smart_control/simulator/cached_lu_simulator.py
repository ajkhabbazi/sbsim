"""Heat-balance simulator that factorizes the timestep matrix once.

`NumpyScipySimulator` rebuilds the sparse system in Python and calls `spsolve`
(assembly + LU + solve) on every Picard iteration of every timestep. But the
matrix is almost constant:

  * the storage term C/dt, conduction, ambient film and interior-mass coupling
    depend only on geometry, materials, dt and the convection coefficient;
  * the linearized longwave exchange adds `coefficient * area` to the diagonal
    of the interior wall nodes. The coefficient is ~4*sigma*eps*T^3, so it moves
    by about one percent per Kelvin; everything else of the radiation (the
    driving flux) is on the right-hand side.

So A_k = A_cond + diag(D_k) with D_k small and slowly varying. This class
assembles A_cond once, factorizes A_ref = A_cond + diag(D_ref) once, and solves
every Picard iteration with preconditioned conjugate gradients on the exact
A_k, preconditioned by the (stale) LU of A_ref. When D_k stays near D_ref the
preconditioned system is nearly the identity and CG needs one or two triangular
solves; if it needs more than `refactor_after`, the next call refactorizes at
the current D. Without radiation D = 0, the LU is exact, and a step costs one
triangular solve.

The exterior CVs are pinned to ambient, so they are eliminated from the
unknowns (their conduction goes to the right-hand side). That keeps A symmetric
positive definite, which PCG requires.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from smart_control.simulator import simulator_flexible_floor_plan


class CachedLUSimulator(
    simulator_flexible_floor_plan.SimulatorFlexibleGeometries
):
  """Same equations as `NumpyScipySimulator`, one LU for many solves."""

  def __init__(
      self,
      *args,
      refactor_after: int = 3,
      pcg_rtol: float = 1e-11,
      pcg_iteration_limit: int = 50,
      **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self._solves_entire_linear_system = True
    self._refactor_after = refactor_after
    self._pcg_rtol = pcg_rtol
    self._pcg_iteration_limit = pcg_iteration_limit
    self._static = None
    self._static_h = None
    self._lu = None
    self._stale = True
    # Diagnostics.
    self.factorization_count = 0
    self.triangular_solve_count = 0

  # ------------------------------------------------------------ assembly
  def _build_static(self, convection_coefficient: float) -> dict:
    """Assembles everything that does not depend on temperature (once)."""
    building = self.building
    rows_count, cols_count = building.temp.shape
    cv_count = rows_count * cols_count
    exterior = np.fromiter(
        (
            len(building.neighbors[r][c]) <= 1
            for r in range(rows_count)
            for c in range(cols_count)
        ),
        dtype=bool,
        count=cv_count,
    )
    solved = np.flatnonzero(~exterior)
    position = np.full(cv_count, -1, dtype=np.int64)
    position[solved] = np.arange(len(solved))
    n = len(solved)
    floor_height = building.floor_height_cm / 100.0
    include_mass = bool(getattr(building, 'include_interior_mass', False))

    storage = np.zeros(n)
    boundary_conductance = np.zeros(n)  # to pinned exterior CVs and ambient
    mass_conductance = np.zeros(n)
    diagonal = np.zeros(n)
    rows, cols, vals = [], [], []
    for i, flat in enumerate(solved):
      r, c = divmod(int(flat), cols_count)
      u, v = self._get_cv_dimensions((r, c))
      storage[i] = (
          building.density[r, c]
          * building.heat_capacity[r, c]
          * u
          * v
          * floor_height
          / self._time_step_sec
      )
      d = storage[i]
      for nr, nc in building.neighbors[r][c]:
        g = self._get_face_conductance((r, c), (nr, nc))
        d += g
        j = position[nr * cols_count + nc]
        if j < 0:
          boundary_conductance[i] += g
        else:
          rows.append(i)
          cols.append(j)
          vals.append(-g)
      g_amb = self._get_ambient_conductance((r, c), convection_coefficient)
      d += g_amb
      boundary_conductance[i] += g_amb
      if include_mass and building.interior_mass_mask[r, c]:
        mass_conductance[i] = (
            building.interior_mass_conductivity[r, c] * u * v / floor_height
        )
        d += mass_conductance[i]
      diagonal[i] = d
    a_cond = (
        sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
        + sparse.diags(diagonal)
    ).tocsr()

    static = dict(
        solved=solved,
        position=position,
        n=n,
        cv_count=cv_count,
        storage=storage,
        boundary_conductance=boundary_conductance,
        mass_conductance=mass_conductance,
        a_cond=a_cond,
        radiation_area=building.cv_size_cm / 100.0 * floor_height,
    )
    if getattr(building, 'include_radiative_heat_transfer', False):
      lwx = np.asarray(building.lwx_index).ravel()
      cv = np.flatnonzero((lwx >= 0) & (position >= 0))
      static['lwx_cv'] = cv
      static['lwx_pos'] = lwx[cv]
      static['lwx_sol'] = position[cv]
    return static

  # --------------------------------------------------------------- solve
  def update_temperature_estimates(
      self,
      temperature_estimates: np.ndarray,
      ambient_temperature: float,
      convection_coefficient: float,
  ) -> tuple[np.ndarray, float]:
    building = self.building
    if self._static is None or self._static_h != convection_coefficient:
      self._static = self._build_static(convection_coefficient)
      self._static_h = convection_coefficient
      self._lu = None
    s = self._static
    sol, n = s['solved'], s['n']
    t_flat = np.asarray(temperature_estimates, dtype=np.float64).reshape(-1)

    rhs = (
        s['storage']
        * np.asarray(building.temp, dtype=np.float64).reshape(-1)[sol]
        + np.asarray(building.input_q, dtype=np.float64).reshape(-1)[sol]
        + s['boundary_conductance'] * ambient_temperature
    )
    mass_flat = None
    if getattr(building, 'include_interior_mass', False):
      mass_flat = np.asarray(
          building.interior_mass_temp, dtype=np.float64
      ).reshape(-1)
      rhs += s['mass_conductance'] * mass_flat[sol]

    radiation_diagonal = np.zeros(n)
    if 'lwx_cv' in s:
      coefficient, driving, is_wall = (
          building.get_longwave_interior_radiative_exchange_terms(
              temperature_estimates
          )
      )
      pos, sidx, cv = s['lwx_pos'], s['lwx_sol'], s['lwx_cv']
      area = s['radiation_area']
      wall = np.asarray(is_wall, dtype=bool)[pos]
      c, g = np.asarray(coefficient)[pos], np.asarray(driving)[pos]
      radiation_diagonal[sidx[wall]] = c[wall] * area
      rhs[sidx[wall]] += g[wall] * area
      if (~wall).any():
        rhs[sidx[~wall]] += (g[~wall] - c[~wall] * mass_flat[cv[~wall]]) * area

    if self._exterior_forcing is not None:
      coefficient, driving = self._exterior_forcing.linearized_air_terms(
          temperature_estimates
      )
      radiation_diagonal += coefficient.ravel()[sol]
      rhs += driving.ravel()[sol]
    a_k = self._assemble_timestep_matrix(s, radiation_diagonal)
    if self._lu is None or self._stale:
      self._lu = sparse_linalg.splu(a_k.tocsc(), permc_spec='MMD_AT_PLUS_A')
      self.factorization_count += 1
      self._stale = False

    x = t_flat[sol].copy()
    solves = self._pcg(a_k, rhs, x)
    self.triangular_solve_count += solves
    if solves > self._refactor_after:
      self._stale = True  # next call refactorizes at the then-current diagonal

    solution = np.full(s['cv_count'], ambient_temperature)
    solution[sol] = x
    max_delta = float(np.max(np.abs(solution - t_flat)))
    return solution.reshape(building.temp.shape), max_delta

  def _assemble_timestep_matrix(self, static, radiation_diagonal):
    return (static['a_cond'] + sparse.diags(radiation_diagonal)).tocsr()

  def _pcg(self, a: sparse.csr_matrix, b: np.ndarray, x: np.ndarray) -> int:
    """PCG with the cached LU as preconditioner. Updates x; returns #solves."""
    tol = self._pcg_rtol * max(np.linalg.norm(b), 1.0)
    r = b - a @ x
    if np.linalg.norm(r) <= tol:
      return 0
    z = self._lu.solve(r)
    p = z.copy()
    rz = r @ z
    solves = 1
    for _ in range(self._pcg_iteration_limit):
      ap = a @ p
      alpha = rz / (p @ ap)
      x += alpha * p
      r -= alpha * ap
      if np.linalg.norm(r) <= tol:
        break
      z = self._lu.solve(r)
      solves += 1
      rz_new = r @ z
      p = z + (rz_new / rz) * p
      rz = rz_new
    return solves
