"""Building Radiation Utility Functions

For computing the physical and thermal characteristics of buildings.
"""

from collections import deque
import dataclasses
import math
from typing import Any, Callable, Mapping, NamedTuple, Sequence

import numpy as np

from smart_control.simulator import constants
from smart_control.simulator import solar_radiation

TEMPORARY_MARKED_VALUE = -33
TEMPORARY_BLOCKED_VALUE = -34
AIR_IN_LINE_OF_SIGHT = 9  # Air nodes along line of sight between wall nodes
FOUR_CONNECTED_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
# Compass bearings of FOUR_CONNECTED_DIRECTIONS.
EXTERIOR_FACE_AZIMUTHS = (0.0, 180.0, 270.0, 90.0)


class ExteriorRadiationForcing(NamedTuple):
  """Solar power in W, emission in W/K⁴, and radiant temperature in K.

  Fields are NumPy or backend-native arrays. The radiant temperature is scalar.
  """

  shortwave_air_gain: Any
  emission: Any
  mass_gain: Any
  radiant_temperature: Any

  def linearized_air_terms(self, temperature, reference_temperature=0.0):
    """Returns diagonal conductance and driving power for the chosen reference.

    Factoring T_rad⁴ - T_surface⁴ gives a positive secant conductance for
    positive temperatures. Freeze that conductance for the current linear
    solve and keep the unknown surface temperature implicit. Solvers for
    temperature change use the previous temperature as their reference.
    """
    radiant = self.radiant_temperature
    conductance = (
        self.emission * (radiant + temperature) * (radiant**2 + temperature**2)
    )
    driving = self.shortwave_air_gain + conductance * (
        radiant - reference_temperature
    )
    return conductance, driving


@dataclasses.dataclass(frozen=True)
class ExteriorRadiationGeometry:
  """Static exterior-face geometry and shortwave routing for a floor plan."""

  face_areas: np.ndarray
  azimuths: tuple[float, ...]
  absorptivity: np.ndarray
  emission: np.ndarray
  transmittance: float
  window_targets: tuple[tuple[np.ndarray, np.ndarray], ...]
  mass_mask: np.ndarray

  def forcing(
      self,
      weather: solar_radiation.ExteriorRadiationData,
      *,
      poa_irradiance: tuple[float, ...] | None = None,
      window_air_gain: Callable[[np.ndarray], np.ndarray] | None = None,
  ) -> ExteriorRadiationForcing:
    """Converts incident radiation to power once per weather sample."""
    ambient = weather.ambient_temp_k
    ground = ambient if weather.ground_temp_k is None else weather.ground_temp_k
    # The model has vertical facades. F_ground = F_sky = 1/2, with the
    # sky-facing fraction split between the sky dome and surrounding air.
    sky_weight = 0.5 * math.sqrt(0.5)
    surroundings_t4 = (
        0.5 * ground**4
        + sky_weight * weather.sky_temp_k**4
        + (0.5 - sky_weight) * ambient**4
    )

    irradiance = weather.irradiance
    absorbed = np.zeros_like(self.emission)
    transmitted = np.zeros_like(self.emission)
    if irradiance.ghi > 0.0 or irradiance.dni > 0.0 or irradiance.dhi > 0.0:
      if poa_irradiance is None:
        poa_irradiance = tuple(
            solar_radiation.calculate_poa_irradiance(irradiance, 90.0, azimuth)
            for azimuth in self.azimuths
        )
      incident = sum(
          area * flux
          for area, flux in zip(self.face_areas, poa_irradiance, strict=True)
      )
      absorbed = self.absorptivity * incident
      if window_air_gain is None:
        for window_indices, air_indices in self.window_targets:
          watts = self.transmittance * incident.flat[window_indices].sum()
          transmitted.flat[air_indices] += watts / air_indices.size
      else:
        transmitted = window_air_gain(incident)

    return ExteriorRadiationForcing(
        shortwave_air_gain=absorbed
        + np.where(self.mass_mask, 0.0, transmitted),
        emission=self.emission,
        mass_gain=np.where(self.mass_mask, transmitted, 0.0),
        radiant_temperature=surroundings_t4**0.25,
    )


def exterior_window_targets(
    floor_plan: np.ndarray, active_air: np.ndarray, exposed: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
  """Maps each exposed glazing group to the union of adjacent air regions."""
  window_value = constants.FENESTRATION_VALUE_IN_FILE_INPUT
  if not np.any((floor_plan == window_value) & exposed):
    return ()

  # Label connected air independently of HVAC zone boundaries. Only recipients
  # whose temperatures are solved can take a share of the transmitted power.
  air_labels = np.zeros(floor_plan.shape, dtype=int)
  air_indices_by_label = {}
  for label, group in enumerate(
      _find_connected_groups(active_air, True).values(), 1
  ):
    rows, cols = zip(*group['indices'])
    air_labels[rows, cols] = label
    air_indices_by_label[label] = np.ravel_multi_index(
        (rows, cols), floor_plan.shape
    )

  targets = []
  rows_count, cols_count = floor_plan.shape
  for group in _find_connected_groups(floor_plan, window_value).values():
    window = group['indices']
    exposed_window = [(r, c) for r, c in window if exposed[r, c]]
    if not exposed_window:
      continue
    adjacent_air = set()
    for r, c in window:
      for dr, dc in FOUR_CONNECTED_DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows_count and 0 <= nc < cols_count:
          if air_labels[nr, nc]:
            adjacent_air.add(air_labels[nr, nc])
    if not adjacent_air:
      raise ValueError(
          f'Exterior glazing at {exposed_window[0]} transmits solar power but '
          'has no adjacent indoor air region.'
      )
    targets.append((
        np.ravel_multi_index(tuple(zip(*exposed_window)), floor_plan.shape),
        np.concatenate([air_indices_by_label[i] for i in sorted(adjacent_air)]),
    ))
  return tuple(targets)


# pylint: disable=invalid-name


def calculate_a_tilde_inv(epsilon: np.ndarray, F: np.ndarray) -> np.ndarray:
  r"""Calculates the inverse A-tilde matrix for radiosity calculations.

  Starting from the gray, diffuse-surface radiosity relation

  $$J_i = \epsilon_i E_{b,i} + (1 - \epsilon_i) G_i,$$

  and using $G_i = \sum_j F_{ij} J_j$, the equations become

  $$J_i - (1 - \epsilon_i) \sum_j F_{ij} J_j = \epsilon_i E_{b,i}.$$

  Dividing each row by $\epsilon_i$ gives

  $$\tilde{\mathbf{A}}\,\mathbf{J} = \mathbf{E}_b,$$

  where

  $$\tilde{A}_{ij} =
  \frac{\delta_{ij} - (1 - \epsilon_i) F_{ij}}{\epsilon_i}.$$

  This function returns $\tilde{\mathbf{A}}^{-1}$, which maps blackbody
  emissive power to radiosity: $\mathbf{J} =
  \tilde{\mathbf{A}}^{-1}\mathbf{E}_b$.

  Args:
    epsilon: Surface emissivities.
    F: View-factor matrix in the convention used by this calculation, where
      ``F[i, j]`` contributes radiosity from surface ``j`` to irradiation on
      surface ``i``.

  Returns:
    The inverse A-tilde matrix, $\tilde{\mathbf{A}}^{-1}$.

  Raises:
    ValueError: If inputs have incompatible dimensions, non-finite values, or
      emissivities outside the interval $(0, 1]$.
  """
  epsilon = np.asarray(epsilon, dtype=float)
  F = np.asarray(F, dtype=float)
  n = epsilon.shape[0]

  if epsilon.ndim != 1 or F.shape != (n, n):
    raise ValueError('epsilon and F must have compatible square dimensions.')
  if not np.all(np.isfinite(epsilon)) or np.any((epsilon <= 0) | (epsilon > 1)):
    raise ValueError(
        'epsilon values must be finite and in the interval (0, 1].'
    )
  if not np.all(np.isfinite(F)):
    raise ValueError('F must contain only finite values.')

  a_tilde = (np.eye(n) - (1.0 - epsilon)[:, np.newaxis] * F) / epsilon[
      :, np.newaxis
  ]
  return np.linalg.solve(a_tilde, np.eye(n))


def calculate_ifa_inv(F: np.ndarray, A_inv: np.ndarray) -> np.ndarray:
  r"""Builds the operator mapping emissive power to net outward heat flux.

  With $\mathbf{J} = \tilde{\mathbf{A}}^{-1}\mathbf{E}_b$ and
  $\mathbf{G} = \mathbf{F}\mathbf{J}$, the net radiative heat flux leaving
  each surface is

  $$\mathbf{q}^{out} = \mathbf{J} - \mathbf{G}
    = (\mathbf{I} - \mathbf{F})\tilde{\mathbf{A}}^{-1}\mathbf{E}_b.$$

  This function returns the operator

  $$\mathbf{IFA}_{inv} =
    (\mathbf{I} - \mathbf{F})\tilde{\mathbf{A}}^{-1}.$$

  Despite its historical name, $\mathbf{IFA}_{inv}$ is not itself a matrix
  inverse. The input named ``A_inv`` must specifically be
  $\tilde{\mathbf{A}}^{-1}$ from `calculate_a_tilde_inv`. ``F[i, j]`` follows
  the same convention used there: it weights radiosity from surface $j$ in
  irradiation on surface $i$.

  For a complete enclosure, $\sum_j F_{ij}=1$, this operator has zero row sums
  and therefore gives zero net flux at uniform temperature. Physical energy
  conservation additionally requires area-weighted reciprocity,
  $A_iF_{ij}=A_jF_{ji}$. The default equal-area ``AreaRatio`` factors satisfy
  both conditions within each connected enclosure.

  See `net_radiative_heatflux_function_of_t` for the pairwise form used to
  return heat flux gained by a surface rather than heat flux leaving it.

  Args:
    F: Square view-factor matrix.
    A_inv: $\tilde{\mathbf{A}}^{-1}$ from `calculate_a_tilde_inv`.

  Returns:
    The matrix $(\mathbf{I}-\mathbf{F})\tilde{\mathbf{A}}^{-1}$, which maps
    blackbody emissive power to net outward radiative heat flux.
  """

  n = F.shape[0]

  I = np.eye(n)
  ifa_inv = (I - F) @ A_inv
  return ifa_inv


def net_radiative_heatflux_function_of_t(
    T: np.ndarray, ifa_inv: np.ndarray
) -> np.array:
  r"""
  Calculates the net radiative heat flux gained by all surfaces given
    surface temperatures.

  Equations:
  --------------------
  The net radiative heat flux leaving surface $i$ is:

  $$q^{out}_i = J_i - G_i$$

  where:
  $J_i$ is the radiosity (total outgoing radiative flux) from surface $i$,
  $G_i$ is the irradiation (total incoming radiative flux) onto surface $i$.

  The radiosity is given by:

  $$J_i = \epsilon_i E_{b,i} + \rho_i G_i$$

  where $\epsilon_i$ is the emissivity, $\rho_i = 1 - \epsilon_i$ is the
    reflectivity (for opaque surfaces), and $E_{b,i}$ is the blackbody
    emission from surface $i$.

  The irradiation for the $i$ surface is:

  $$G_i A_i = \sum_{j=1,\, j\neq i}^n J_j A_j F_{ji}$$

  where $F_{ji}$ is the view factor from surface $j$ to $i$.

  Combining these, the radiosity equation for all surfaces can be written in
    vector-matrix form as:

  $$\tilde{\mathbf{A}}\, \mathbf{J} = \mathbf{E}_b$$

  where $\tilde{A}_{ij} =
    \frac{\delta_{ij} - (1-\epsilon_i) F_{ij}}{\epsilon_i}$.

  Solving for $\mathbf{J}$:

  $$\mathbf{J} = \tilde{\mathbf{A}}^{-1} \mathbf{E}_b$$

  The net heat flux vector for all surfaces is:

  $$\mathbf{q}^{out}=
  (\mathbf{I}-\tilde{\mathbf{F}})\tilde{\mathbf{A}}^{-1}\mathbf{E}_b
  = \mathbf{IFA}_{inv}\mathbf{E}_b$$

  where $\tilde{\mathbf{F}}$ is the matrix of view factors,
    $F_{ij}$ and $\mathbf{E}_b$ is $\sigma \mathbf{T}^4$.

  Pairwise (difference) form:
  ---------------------------
  The operator above returns flux leaving each surface. The energy balance uses
  flux gained by each surface, so with

  $$\mathbf{R} = \mathbf{IFA}_{inv}
    = (\mathbf{I}-\mathbf{F})\tilde{\mathbf{A}}^{-1},$$

  the required sign convention is

  $$\mathbf{q}^{gain} = -\mathbf{q}^{out}
    = -\mathbf{R}\mathbf{E}_b.$$

  To convert this matrix product to a pairwise temperature-difference form,
  assume the view factors satisfy enclosure closure,

  $$\mathbf{F}\mathbf{1}=\mathbf{1}.$$

  The definition of $\tilde{\mathbf{A}}$ then gives

  $$\left[\tilde{\mathbf{A}}\mathbf{1}\right]_i
    = \frac{1-(1-\epsilon_i)\sum_jF_{ij}}{\epsilon_i} = 1,$$

  and therefore

  $$\tilde{\mathbf{A}}^{-1}\mathbf{1}=\mathbf{1}, \qquad
    \mathbf{R}\mathbf{1}
    = (\mathbf{I}-\mathbf{F})\mathbf{1}=\mathbf{0}.$$

  Thus every row of $\mathbf{R}$ sums to zero. Define the off-diagonal
  exchange matrix by

  $$M_{ij} = -R_{ij} \ge 0 \quad (i \neq j), \qquad
    M_{ii} = 0,$$

  so that the diagonal entry is

  $$R_{ii} = -\sum_{j\neq i}R_{ij} = \sum_{j\neq i}M_{ij}.$$

  Expanding the gained flux component by component gives

  $$\begin{aligned}
    q_i^{gain}
    &= -\sum_jR_{ij}E_{b,j} \\
    &= -R_{ii}E_{b,i} + \sum_{j\neq i}M_{ij}E_{b,j} \\
    &= \sum_{j\neq i}M_{ij}(E_{b,j}-E_{b,i}) \\
    &= \sum_{j\neq i}M_{ij}\,\sigma(T_j^4-T_i^4).
  \end{aligned}$$

  This is the pairwise form evaluated by the implementation. It is identically
  zero at uniform temperature and preserves the hot-to-cold sign. It is exactly
  equal to $-\mathbf{R}\mathbf{E}_b$ when the view-factor rows close to one.
  If closure is not satisfied, the two forms differ by the residual row-sum
  term; the pairwise form deliberately removes that artificial isothermal heat
  source but cannot repair inaccurate exchange magnitudes.

  This is also the form EnergyPlus uses in
  ``CalcInteriorRadExchange``, where the self term of the exchange matrix is
  zeroed and the sum is taken over $T_j^4 - T_i^4$.

  The default ``AreaRatio`` method produces complete, reciprocal factors within
  each connected enclosure under this model's equal-area surface assumption,
  so the pairwise and absolute radiosity forms agree there.

  Nomenclature and Units:
  -----------------------
  - $q_i$        : Net radiative heat flux gained by surface $i$
                    [$\mathrm{W/m^2}$]
  - $\mathbf{q}$ : Vector of $q_i$ for all $i=1..n$ [$\mathrm{W/m^2}$]
  - $M_{ij}$     : Pairwise radiative exchange coefficient [dimensionless]
  - $J_i$        : Radiosity of surface $i$ [$\mathrm{W/m^2}$]
  - $\mathbf{J}$ : Vector of $J_i$ for all $i=1..n$ [$\mathrm{W/m^2}$]
  - $G_i$        : Irradiation on surface $i$ [$\mathrm{W/m^2}$]
  - $E_{b,i}$    : Blackbody emissive power of surface $i$ [$\mathrm{W/m^2}$]
  - $\mathbf{E}_b$: Vector of $E_{b,i}$ for all $i=1..n$ [$\mathrm{W/m^2}$]
  - $\epsilon_i$ : Emissivity of surface $i$ [dimensionless]
  - $\rho_i$     : Reflectivity of surface $i$ [dimensionless],
                    ($\rho_i=1-\epsilon_i$)
  - $A_i$        : Area of surface $i$ [$\mathrm{m^2}$]
  - $F_{ij}$     : View factor from surface $i$ to $j$ [dimensionless]
  - $\tilde{\mathbf{A}}$: Matrix with elements
    ($\tilde{A}_{ij} =
    \frac{\delta_{ij} - (1-\epsilon_i) F_{ij}}{\epsilon_i}$)
  - $\mathbf{I}$ : $n \times n$ identity matrix
  - $\tilde{\mathbf{F}}$: Matrix of $F_{ij}$ (view factors)
  - $\delta_{ij}$: Kronecker delta ($=1$ if $i=j$, $=0$ otherwise)
  - $\sigma$: Stefan-Boltzmann constant [$\mathrm{W/m^2K^4}$]
  - $\mathbf{T}$: Vector of surface temperatures [K]

  References:
  -----------
  - Incropera, F.P., DeWitt, D.P., "Fundamentals of Heat and Mass Transfer"

  Args:
    T (np.ndarray): Surface temperatures in Kelvin.
    ifa_inv (np.ndarray): (I - F) @ A_inv.

  Returns:
      np.ndarray : q, net radiative heat flux gained by each surface [W/m^2].
        Positive means the surface is being heated by the rest of the enclosure.

  """
  sigma = (
      constants.STEFAN_BOLTZMANN_CONSTANT
  )  # [W/m^2K^4] Stefan-Boltzmann constant

  exchange = -np.array(ifa_inv, dtype=float)
  np.fill_diagonal(exchange, 0.0)
  emissive_power = sigma * np.power(np.asarray(T, dtype=float), 4)
  return exchange @ emissive_power - exchange.sum(axis=1) * emissive_power


def calculate_radiative_exchange_terms(
    T: np.ndarray, ifa_inv: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  r"""Calculates exact conductance-form radiative exchange terms.

  Equations:
  --------------------
  The pairwise exchange that net_radiative_heatflux_function_of_t evaluates,

  $$q_i = \sum_{j \neq i} M_{ij}\,\sigma\left(T_j^4 - T_i^4\right),$$

  factors exactly. For any two temperatures,

  $$\sigma\left(T_j^4 - T_i^4\right) =
    \underbrace{\sigma\left(T_i^2 + T_j^2\right)\left(T_i + T_j\right)}_{h_{ij}}
    \left(T_j - T_i\right),$$

  which is an identity, not a linearization: $h_{ij}$ is the radiative
  conductance [W/m^2/K] between the pair. The textbook $4\sigma\bar{T}^3$ is
  what this collapses to as $T_i \to T_j$. Writing the exchange as

  $$q_i = g_i-c_iT_i,$$

  with

  $$c_i = \sum_{j\neq i}M_{ij}h_{ij}, \qquad
    g_i = \sum_{j\neq i}M_{ij}h_{ij}T_j.$$

  Here $c_i$ is the exchange coefficient [W/m^2/K] belonging on the solver
  diagonal, and $g_i$ is the driving flux [W/m^2] belonging in the source.

  lets a solver carry the second term on the diagonal instead of lumping the
  whole of $q_i$ into the source.

  Why that matters:
  -----------------
  Write the non-radiative part of a control-volume balance as

  $$D_i T_i = N_i + A_i q_i,$$

  where $D_i > 0$ contains storage and transport terms multiplying the CV's
  unknown temperature, $N_i$ contains their driving terms, and $A_i$ is the
  radiating area. A fully explicit radiation update evaluates the complete
  nonlinear flux at the previous iterate,

  $$T_i^{(k+1)} = \frac{N_i + A_i q_i(\mathbf{T}^{(k)})}{D_i}.$$

  For a hot surface, $q_i$ contains a large negative term proportional to
  $-(T_i^{(k)})^4$, but that temperature does not appear in the denominator.
  If the storage contribution to $D_i$ becomes small, as it does for a long
  time step, the explicit correction can cross far past the surrounding
  temperatures. The next iteration raises that overshoot to the fourth power,
  which can alternate in sign, grow, and eventually overflow.

  Substituting the exact split $q_i = g_i-c_iT_i$ instead gives the
  semi-implicit Picard update

  $$T_i^{(k+1)} =
    \frac{N_i + A_i g_i(\mathbf{T}^{(k)})}
         {D_i + A_i c_i(\mathbf{T}^{(k)})}.$$

  Since $M_{ij} \ge 0$ and $h_{ij} > 0$, $c_i = \sum_jM_{ij}h_{ij}$ is
  nonnegative and the radiation term adds positive diagonal damping. Its
  numerator weights are the same $M_{ij}h_{ij}$ whose sum forms that diagonal,
  so the radiation-only contribution is bounded by the temperatures driving
  it instead of acting as an undamped fourth-power source. Other heat sources
  may still move the complete CV update outside the neighboring-temperature
  range, but radiation no longer causes the sign-flipping runaway described
  above.

  Because $q_i = g_i-c_iT_i$ is an exact identity, moving $c_iT_i$ to the
  diagonal does not change the converged heat balance. Only the
  temperature-dependent coefficients are lagged during iteration.

  The function itself performs no approximation. A solver may use the returned
  coefficient semi-implicitly while lagging its temperature dependence by one
  iteration; that outer iteration is a Picard linearization of the nonlinear
  balance and recovers the nonlinear answer at convergence.

  Implementation:
  ---------------
  This function operates on one enclosure at a time. For enclosure $e$, let
  $\mathbf{T}_e$ contain its $n_e$ surface temperatures and let
  $\mathbf{M}_e$ be ``-ifa_inv`` with a zero diagonal. The returned vectors are
  $\mathbf{c}_e$ and $\mathbf{g}_e$, whose entries are the $c_i$ and $g_i$
  defined above for nodes in that enclosure.

  Expanding $h_{ij}$ avoids materializing an $n_e \times n_e$
  pair-temperature array. Define

  $$\mathbf{s}_{e,k} = \mathbf{M}_e\mathbf{T}_e^k,
    \qquad k=0,1,2,3,4,$$

  where $\mathbf{T}_e^0$ is the all-ones vector. Then

  $$\mathbf{c}_e = \sigma(\mathbf{T}_e^3\odot\mathbf{s}_{e,0}
    + \mathbf{T}_e^2\odot\mathbf{s}_{e,1}
    + \mathbf{T}_e\odot\mathbf{s}_{e,2} + \mathbf{s}_{e,3}),$$

  $$\mathbf{g}_e = \sigma(\mathbf{T}_e^3\odot\mathbf{s}_{e,1}
    + \mathbf{T}_e^2\odot\mathbf{s}_{e,2}
    + \mathbf{T}_e\odot\mathbf{s}_{e,3} + \mathbf{s}_{e,4}).$$

  Thus one matrix multiplication with five right-hand sides produces both
  outputs. This keeps intermediate storage linear in enclosure size and works
  directly with variable-size enclosure blocks; no padded 3-D batch is needed.

  Args:
    T: Surface temperatures in Kelvin, one per node of the enclosure.
    ifa_inv: (I - F) @ A_inv, as built by calculate_ifa_inv.

  Returns:
    The enclosure vectors $(\mathbf{c}_e, \mathbf{g}_e)$, both shaped like
    ``T``. $\mathbf{c}_e$ is the exchange coefficient [W/m^2/K] belonging on
    the diagonal, and $\mathbf{g}_e$ is the driving flux [W/m^2] belonging in
    the source. Their combination
    $\mathbf{g}_e-\mathbf{c}_e\odot\mathbf{T}_e$ reproduces
    `net_radiative_heatflux_function_of_t` exactly.
  """
  sigma = (
      constants.STEFAN_BOLTZMANN_CONSTANT
  )  # [W/m^2K^4] Stefan-Boltzmann constant

  exchange = -np.array(ifa_inv, dtype=float)
  np.fill_diagonal(exchange, 0.0)

  temperatures = np.asarray(T, dtype=float)
  squared = np.square(temperatures)
  cubed = squared * temperatures
  fourth_power = squared * squared
  powers = np.column_stack(
      (np.ones_like(temperatures), temperatures, squared, cubed, fourth_power)
  )
  moment_0, moment_1, moment_2, moment_3, moment_4 = (exchange @ powers).T

  coefficient = sigma * (
      cubed * moment_0 + squared * moment_1 + temperatures * moment_2 + moment_3
  )
  driving = sigma * (
      cubed * moment_1 + squared * moment_2 + temperatures * moment_3 + moment_4
  )
  return coefficient, driving


def mark_air_connected_interior_walls(
    indexed_floor_plan: np.ndarray,
    start_pos: tuple[int, int],
    interior_wall_value: int = constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
    marked_value: int = TEMPORARY_MARKED_VALUE,
    air_value: int = constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
    also_mark_air: bool = False,
    air_marked_value: int = AIR_IN_LINE_OF_SIGHT,
) -> tuple[np.ndarray | None, np.ndarray | None]:
  """
  Mark all interior wall nodes that are connected to the same air space as the
      starting position (interior wall or air cell).
  Uses 4-directional connectivity to check wall-air adjacency.
  All connected walls are marked.

  Args:
    indexed_floor_plan (np.ndarray): 2D numpy array representing the floor plan
        where different values represent different types of cells (walls, air,
        etc.).
    start_pos (Tuple[int, int]): Starting position (row, col). Can be either an
        interior wall or an air cell. If it's an interior wall, finds all walls
        connected to the same air space. If it's an air cell, finds all walls
        connected to that air space.
    interior_wall_value (int, optional): Value used to represent interior walls
        in the floor plan. Defaults to -3 (from "constants.py").
    marked_value (int, optional): Value used to mark connected interior walls.
        Only used internally. Defaults to -33.
    air_value (int, optional): Value used to represent air spaces in the floor
        plan. Defaults to 0 (from "constants.py").
    also_mark_air (bool, optional): If True, the connected air cells are also
        marked, with `air_marked_value`, in `modified_floor_plan`. The extracted
        `interior_space_array` is unaffected. Callers that place nodes on air
        cells, such as interior mass, use this to recover the whole enclosure
        without a line-of-sight sweep. Defaults to False.
    air_marked_value (int, optional): Value used to mark the connected air cells
        when `also_mark_air` is True. Defaults to 9 (`AIR_IN_LINE_OF_SIGHT`).

  Returns:
    A tuple containing:

      - `modified_floor_plan`: Copy of input floor plan with connected walls
          marked with marked_value. `None` if `start_pos` is invalid.

      - `interior_space_array`: Extracted interior space containing only air and
          marked walls, cropped to the bounding box of the connected region.
          `None` if `start_pos` is invalid or no interior space is found.

  Raises:
      ValueError: If the starting position is out of bounds of the floor plan.

  Note:
      This function is used as the first step in radiative heat transfer
      calculations to identify all interior wall nodes that are connected to the
      same air space. The marked_value (-33) indicates walls that can
      potentially participate in radiative heat transfer with each other.
  """
  # Make a copy to avoid modifying the original
  floor_plan = indexed_floor_plan.copy()
  if (
      start_pos[0] < 0
      or start_pos[0] >= floor_plan.shape[0]
      or start_pos[1] < 0
      or start_pos[1] >= floor_plan.shape[1]
  ):
    raise ValueError('Starting position is out of bounds')

  start_row, start_col = start_pos
  start_cell_value = floor_plan[start_row, start_col]

  # Return None if start_pos is neither interior_wall_value nor air_value
  if start_cell_value != interior_wall_value and start_cell_value != air_value:
    return None, None

  # 4-connectivity for all steps
  directions = FOUR_CONNECTED_DIRECTIONS

  # Find all air cells that are connected to the starting position
  connected_air_cells = set()
  air_queue = deque()

  if start_cell_value == air_value:
    # If starting from an air cell, start BFS from that cell
    air_queue.append((start_row, start_col))
    connected_air_cells.add((start_row, start_col))
  else:
    # If starting from an interior wall, find air cells adjacent to it
    for dr, dc in directions:
      new_row, new_col = start_row + dr, start_col + dc
      if (
          0 <= new_row < floor_plan.shape[0]
          and 0 <= new_col < floor_plan.shape[1]
          and floor_plan[new_row, new_col] == air_value
      ):
        air_queue.append((new_row, new_col))
        connected_air_cells.add((new_row, new_col))

  # BFS to find all connected air cells (4-connectivity)
  while air_queue:
    current_row, current_col = air_queue.popleft()
    for dr, dc in directions:
      new_row, new_col = current_row + dr, current_col + dc
      if (
          0 <= new_row < floor_plan.shape[0]
          and 0 <= new_col < floor_plan.shape[1]
          and floor_plan[new_row, new_col] == air_value
          and (new_row, new_col) not in connected_air_cells
      ):
        air_queue.append((new_row, new_col))
        connected_air_cells.add((new_row, new_col))

  # Now find all interior walls that are adjacent to
  #  any of the connected air cells (4-connectivity)
  walls_to_mark = set()
  for air_row, air_col in connected_air_cells:
    for dr, dc in directions:
      wall_row, wall_col = air_row + dr, air_col + dc
      if (
          0 <= wall_row < floor_plan.shape[0]
          and 0 <= wall_col < floor_plan.shape[1]
          and floor_plan[wall_row, wall_col] == interior_wall_value
      ):
        walls_to_mark.add((wall_row, wall_col))

  # Mark all the connected interior walls
  # If starting from an interior wall, exclude it from marking
  # (it will be marked separately)
  # If starting from an air cell, mark all walls found
  for wall_row, wall_col in walls_to_mark:
    if start_cell_value == interior_wall_value and (wall_row, wall_col) == (
        start_row,
        start_col,
    ):
      # Skip marking the starting wall here; will mark it below if any walls
      #  were found
      continue
    floor_plan[wall_row, wall_col] = marked_value

  # If starting from an interior wall and any walls were found, mark the
  # starting position
  if start_cell_value == interior_wall_value and walls_to_mark:
    floor_plan[start_row, start_col] = marked_value

  # Optionally mark the air cells themselves, so a caller can read the whole
  # enclosure, walls and air, straight off the returned floor plan.
  if also_mark_air:
    for air_row, air_col in connected_air_cells:
      floor_plan[air_row, air_col] = air_marked_value

  # Create interior space array containing only air and marked walls
  all_interior_positions = connected_air_cells.union(walls_to_mark)
  if not all_interior_positions:
    return floor_plan, None

  min_row = min(pos[0] for pos in all_interior_positions)
  max_row = max(pos[0] for pos in all_interior_positions)
  min_col = min(pos[1] for pos in all_interior_positions)
  max_col = max(pos[1] for pos in all_interior_positions)

  interior_height = max_row - min_row + 1
  interior_width = max_col - min_col + 1
  interior_space = np.full(
      (interior_height, interior_width),
      interior_wall_value,
      dtype=floor_plan.dtype,
  )

  for air_row, air_col in connected_air_cells:
    interior_space[air_row - min_row, air_col - min_col] = air_value

  # Mark all walls in interior space
  # If starting from interior wall, it will be included in walls_to_mark
  # and marked
  for wall_row, wall_col in walls_to_mark:
    interior_space[wall_row - min_row, wall_col - min_col] = marked_value

  return floor_plan, interior_space


def get_radiative_enclosure_indices(
    indexed_floor_plan: np.ndarray,
    interior_wall_mask: np.ndarray,
    interior_mass_mask: np.ndarray | None = None,
    marked_value: int = TEMPORARY_MARKED_VALUE,
    interior_mass_value: int = AIR_IN_LINE_OF_SIGHT,
) -> tuple[np.ndarray, ...]:
  """Partitions radiating nodes into disconnected enclosure index blocks.

  Indices use the same row-major order as ``lwx_index``. Two nodes belong to
  the same block when the connected-air search used by `get_vf` can place them
  in the same enclosure. The transitive closure is used so the returned blocks
  are disjoint even when a wall touches more than one connected air region.

  Args:
    indexed_floor_plan: Floor plan using the simulator's internal values.
    interior_wall_mask: Radiating interior-wall nodes.
    interior_mass_mask: Optional radiating interior-mass nodes.
    marked_value: Temporary value used to mark enclosure walls.
    interior_mass_value: Temporary value used to mark enclosure mass nodes.

  Returns:
    Disjoint arrays of global radiative-node indices, ordered by their first
    node. Concatenating and sorting the arrays yields ``range(n_nodes)``.
  """
  if interior_mass_mask is None:
    interior_mass_mask = np.zeros_like(interior_wall_mask, dtype=bool)
  if (
      indexed_floor_plan.shape != interior_wall_mask.shape
      or indexed_floor_plan.shape != interior_mass_mask.shape
  ):
    raise ValueError('Floor plan and radiative masks must have the same shape.')

  radiative_mask = interior_wall_mask | interior_mass_mask
  positions = np.argwhere(radiative_mask)
  node_count = len(positions)
  parents = np.arange(node_count)

  def find(node: int) -> int:
    while parents[node] != node:
      parents[node] = parents[parents[node]]
      node = int(parents[node])
    return node

  def union(first: int, second: int) -> None:
    first_root = find(first)
    second_root = find(second)
    if first_root != second_root:
      parents[second_root] = first_root

  for node, position in enumerate(positions):
    marked_floor_plan, _ = mark_air_connected_interior_walls(
        indexed_floor_plan,
        tuple(position),
        marked_value=marked_value,
        also_mark_air=bool(np.any(interior_mass_mask)),
        air_marked_value=interior_mass_value,
    )
    in_enclosure = marked_floor_plan == marked_value
    if np.any(interior_mass_mask):
      in_enclosure |= marked_floor_plan == interior_mass_value
    connected_nodes = np.flatnonzero(in_enclosure[radiative_mask])
    if connected_nodes.size == 0:
      connected_nodes = np.array([node])
    for connected_node in connected_nodes:
      union(node, int(connected_node))

  groups: dict[int, list[int]] = {}
  for node in range(node_count):
    groups.setdefault(find(node), []).append(node)
  return tuple(
      np.asarray(nodes, dtype=int)
      for nodes in sorted(groups.values(), key=lambda nodes: nodes[0])
  )


def fix_view_factors(F: np.ndarray, A: np.ndarray = None) -> np.ndarray:
  """
  Fix approximate view factors and enforce reciprocity and completeness.

  Args:
      F (np.ndarray): Approximate direct view factor matrix (N x N)
      A (np.ndarray, optional): Area vector (N elements). Defaults to None.

  Returns:
      Fixed view factor matrix

  References:
      See `FixViewFactors` function in [EnergyPlus](https://github.com/NatLabRockies/EnergyPlus/blob/develop/src/EnergyPlus/HeatBalanceIntRadExchange.cc) # pylint: disable=line-too-long
  """

  # Parameter definitions
  PRIMARY_CONVERGENCE = 0.001
  DIFFERENCE_CONVERGENCE = 0.00001
  MAX_ITERATIONS = 400

  # Convert inputs to numpy arrays
  A = np.ones(F.shape[0]) if A is None else np.asarray(A)

  F = np.array(F, dtype=np.float64)  # copy so the input is not mutated
  F = F.T  # since EP calculation is based on F[j,i]
  N = F.shape[0]

  # Initialize return values
  results = {
      'original_check_value': 0.0,
      'fixed_check_value': 0.0,
      'final_check_value': 0.0,
      'num_iterations': 0,
      'row_sum': 0.0,
      'enforced_reciprocity': False,
  }

  # OriginalCheckValue is the first pass at a completeness check
  results['original_check_value'] = abs(np.sum(F) - N)

  # EnergyPlus converges the fix on |sum(F) - N|, which is one number for the
  # whole enclosure. get_vf hands over rows that each sum to exactly one, so
  # that test is already satisfied before the loop starts and the routine
  # returns after the single pass that trades the row closure away for
  # reciprocity - the rows come back out anywhere between 0.71 and 2.17.
  # Converge on the worst row instead, which is the quantity the enclosure
  # balance depends on. The global sum is kept for reporting.
  # Allocate and initialize arrays
  FixedAF = F.copy()  # store for largest area check

  ConvrgOld = 10.0
  LargestArea = np.max(A)
  severe_error_present = False
  largest_surf = -1

  # Check for Strange Geometry
  # When one surface has an area that exceeds the sum of all other surface areas
  if LargestArea > 0.99 * (np.sum(A) - LargestArea) and N > 3:
    for i in range(N):
      if LargestArea == A[i]:
        largest_surf = i
        break

    if largest_surf >= 0:
      # Give self view to big surface
      FixedAF[largest_surf, largest_surf] = min(
          0.9, 1.2 * LargestArea / np.sum(A)
      )

  # Set up AF matrix (AREA * DIRECT VIEW FACTOR) MATRIX
  AF = FixedAF * A

  # Enforce reciprocity by averaging AiFij and AjFji
  FixedAF = 0.5 * (AF + AF.T)

  FixedF = np.zeros((N, N))
  has_area = A != 0
  results['num_iterations'] = 0
  results['row_sum'] = 0.0

  # Check for physically unreasonable enclosures (N <= 3)
  if N <= 3:
    np.divide(FixedAF, A, out=FixedF, where=has_area)

    results['row_sum'] = np.sum(FixedF)

    if results['row_sum'] > (N + 0.01):
      # Find the largest row summation and normalize
      sum_FixedF = np.sum(FixedF, axis=1)  # Sum along rows
      MaxFixedFRowSum = np.max(sum_FixedF)

      if MaxFixedFRowSum < 1.0:
        raise RuntimeError(
            'FixViewFactors: Three surface or less zone failing ViewFactorFix'
            ' correction which should never happen.'
        )
      else:
        FixedF *= 1.0 / MaxFixedFRowSum

      results['row_sum'] = np.sum(FixedF)  # Recalculate

    results['final_check_value'] = results['fixed_check_value'] = abs(
        results['row_sum'] - N
    )
    F[:] = FixedF  # Update F in place
    results['enforced_reciprocity'] = True
    return F.T

  # Regular fix cases (N > 3)
  RowCoefficient = np.zeros(N)
  Converged = False

  while not Converged:
    results['num_iterations'] += 1

    for i in range(N):
      # Determine row coefficients which will enforce closure
      sum_FixedAF_i = np.sum(FixedAF[:, i])
      if abs(sum_FixedAF_i) > 1.0e-10:
        RowCoefficient[i] = A[i] / sum_FixedAF_i
      else:
        RowCoefficient[i] = 1.0

      FixedAF[:, i] *= RowCoefficient[i]

    # Enforce reciprocity by averaging AiFij and AjFji
    FixedAF = 0.5 * (FixedAF + FixedAF.T)

    # Form FixedF matrix
    np.divide(FixedAF, A, out=FixedF, where=has_area)
    small = (np.abs(FixedF) < 1.0e-10) & has_area
    FixedF[small] = 0.0
    FixedAF[small] = 0.0

    ConvrgNew = np.max(np.abs(np.sum(FixedF, axis=0) - 1.0))

    # Check convergence
    if (
        abs(ConvrgOld - ConvrgNew) < DIFFERENCE_CONVERGENCE
        or ConvrgNew <= PRIMARY_CONVERGENCE
    ):
      Converged = True

    ConvrgOld = ConvrgNew

    # Emergency exit after too many iterations
    if results['num_iterations'] > MAX_ITERATIONS:
      # Enforce reciprocity by averaging AiFij and AjFji
      FixedAF = 0.5 * (FixedAF + FixedAF.T)

      # Form FixedF matrix
      np.divide(FixedAF, A, out=FixedF, where=has_area)

      sum_FixedF = np.sum(FixedF)
      results['final_check_value'] = results['fixed_check_value'] = abs(
          sum_FixedF - N
      )
      results['row_sum'] = sum_FixedF

      F[:] = FixedF
      results['final_check_value'] = results['fixed_check_value']

      return F.T

  # Normal completion. The loop always ends on a reciprocity pass, and exact
  # reciprocity is what makes the enclosure conserve energy: with A_i F_ij =
  # A_j F_ji the pairwise exchange between any two surfaces is equal and
  # opposite, so the heat the enclosure gains sums to zero. Closure cannot be
  # had at the same time on a floor plan whose line of sight view factors do
  # not close, so keep the reciprocal matrix and report how far the rows are
  # from one.
  results['fixed_check_value'] = ConvrgNew
  results['row_sum'] = np.sum(FixedF)
  F[:] = FixedF
  results['final_check_value'] = ConvrgNew

  if severe_error_present:
    raise RuntimeError(
        'FixViewFactors: View factor calculations significantly out of'
        ' tolerance. See above messages for more information.'
    )

  F = F.T
  return F


def get_vf(
    indexed_floor_plan: np.ndarray,
    interior_wall_mask: np.ndarray,
    view_factor_method: str = 'AreaRatio',
    marked_value: int = TEMPORARY_MARKED_VALUE,
    interior_mass_mask: np.ndarray | None = None,
    interior_mass_value: int = AIR_IN_LINE_OF_SIGHT,
) -> np.ndarray:
  """
  Calculate view factors between interior walls in the floor plan.

  Two methods are available, both of which take the enclosure of a surface to
  be the set of surfaces reachable through the same connected air space, so a
  doorway merges the rooms it joins:

  - `"AreaRatio"` (default) distributes each row by area, following the
    EnergyPlus `CalcApproximateViewFactors` approach. Since every CV in the
    grid has the same face area this reduces to a uniform `1 / (n - 1)` over
    the enclosure. Rows close to one and, with equal areas, reciprocity holds
    exactly, so `fix_view_factors` leaves the matrix alone.
  - `"ScriptF"` additionally removes surfaces that are occluded, keeping only
    those within an unblocked line of sight. Rows still close to one, but the
    number of visible surfaces depends on where a surface sits in the room,
    which violates reciprocity: two surfaces that see each other disagree
    about how much of their emission the other receives, and
    `fix_view_factors` then has to trade closure away to repair it.

  Args:
      indexed_floor_plan (np.ndarray): 2D array representing the floor plan with
          indexed values.
      view_factor_method (str, optional): Method to use for view factors.
          Defaults to 'AreaRatio'. Either "AreaRatio", "ScriptF" or
          "CarrollMRT".
      marked_value (int, optional): Value used to mark connected interior walls.
          Only used internally. Defaults to -33.
      interior_mass_mask (Optional[np.ndarray], optional): Mask for interior
          mass nodes. Defaults to None.
      interior_mass_value (int, optional): Value used to represent interior
          mass nodes. Defaults to 9 (`AIR_IN_LINE_OF_SIGHT`).
  Returns:
      View factor matrix where `VF[i,j]` represents the view factor from wall
          `i` to wall `j`.

  """
  if interior_mass_mask is not None:
    interior_wall_mask_all = interior_wall_mask | interior_mass_mask
  else:
    interior_wall_mask_all = interior_wall_mask

  n_interior_wall = np.sum(interior_wall_mask_all)
  interior_wall_tuples = [
      (r, c)
      for r in range(indexed_floor_plan.shape[0])
      for c in range(indexed_floor_plan.shape[1])
      if interior_wall_mask_all[r, c]
  ]
  vf = np.zeros((n_interior_wall, n_interior_wall))

  if view_factor_method == 'AreaRatio':
    for i in range(n_interior_wall):
      result_floor_plan, _ = mark_air_connected_interior_walls(
          indexed_floor_plan,
          interior_wall_tuples[i],
          marked_value=marked_value,
          also_mark_air=interior_mass_mask is not None,
          air_marked_value=interior_mass_value,
      )
      in_enclosure = result_floor_plan == marked_value
      if interior_mass_mask is not None:
        in_enclosure = in_enclosure | (result_floor_plan == interior_mass_value)

      # Every CV in the grid has the same face area, so the area ratio
      # A_j / sum_{k != i} A_k reduces to a uniform 1 / (n - 1) over the
      # enclosure. A surface does not see itself.
      row = in_enclosure[interior_wall_mask_all].astype(float)
      row[i] = 0.0
      row_total = row.sum()
      if row_total > 0.0:
        row /= row_total
      vf[i, :] = row

  elif view_factor_method == 'ScriptF':
    for i in range(n_interior_wall):
      result_floor_plan, _ = mark_air_connected_interior_walls(
          indexed_floor_plan, interior_wall_tuples[i]
      )
      result_floor_plan = mark_directly_seeing_nodes(
          floor_plan=result_floor_plan, base_node=interior_wall_tuples[i]
      )
      if interior_mass_mask is not None:
        vf_ = 1 / np.sum(
            (result_floor_plan == marked_value)
            | (result_floor_plan == interior_mass_value)
        )
      else:
        vf_ = 1 / np.sum((result_floor_plan == marked_value))

      result_floor_plan_ = np.zeros_like(result_floor_plan).astype('float')

      if interior_mass_mask is not None:
        result_floor_plan_[
            (result_floor_plan == marked_value)
            | (result_floor_plan == interior_mass_value)
        ] = vf_
      else:
        result_floor_plan_[(result_floor_plan == marked_value)] = vf_
      vf[i, :] = result_floor_plan_[interior_wall_mask_all]

  elif view_factor_method == 'CarrollMRT':
    raise NotImplementedError('CarrollMRT view factor method not implemented')
  else:
    raise ValueError(
        f'Invalid view factor method: {view_factor_method}. Either "AreaRatio",'
        ' "ScriptF" or "CarrollMRT"'
    )

  vf = fix_view_factors(vf)
  return vf


def mark_interior_wall_adjacent_to_air(
    arr: np.ndarray,
    interior_wall_value: int = constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
    air_value: int = constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
) -> np.ndarray:
  """Marks interior walls that are adjacent to air spaces.

  Creates a boolean mask identifying interior walls that share an edge with an
  air space (value of 0) in the floor plan. Checks for adjacency in four
   directions:   up, down, left, and right.

  Args:
    arr: 2D array representing the floor plan with interior walls marked as
      interior_wall_value and air spaces as 0.
    interior_wall_value: Value used to represent interior walls in the floor
      plan. Defaults to -3 (constants.INTERIOR_WALL_VALUE_IN_FUNCTION).

  Returns:
    Boolean mask array where True indicates an interior wall that is adjacent to
    at least one air space.
  """
  mask_minus_interior_wall = arr == interior_wall_value
  mask_zero = arr == air_value
  # Find -3s that have a 0 neighbor (up/down/left/right)
  contact = np.zeros_like(arr, dtype=bool)
  # up
  contact[1:, :] |= mask_zero[:-1, :] & mask_minus_interior_wall[1:, :]
  # down
  contact[:-1, :] |= mask_zero[1:, :] & mask_minus_interior_wall[:-1, :]
  # left
  contact[:, 1:] |= mask_zero[:, :-1] & mask_minus_interior_wall[:, 1:]
  # right
  contact[:, :-1] |= mask_zero[:, 1:] & mask_minus_interior_wall[:, :-1]
  # Only mark the -3 cells that are adjacent to a 0
  marked = mask_minus_interior_wall & contact
  return marked


def get_line_points(
    start: tuple[float, float], end: tuple[float, float]
) -> list[tuple[float, float]]:
  """Generate points where the line crosses integer grid lines.

  This function calculates all intersection points between a line segment and
  the integer grid lines. It handles vertical, horizontal, and diagonal lines
  by finding intersections with both vertical (x = integer) and horizontal
  (y = integer) grid lines.

  Args:
      start: Starting point of the line segment as (x, y) coordinates.
      end: Ending point of the line segment as (x, y) coordinates.

  Returns:
      List of intersection points sorted by distance from the start point.
          Each point is a tuple of (x, y) coordinates as floats.


  """
  x1, y1 = start
  x2, y2 = end

  points = []

  # Handle vertical line case
  if abs(x2 - x1) < 1e-10:  # Vertical line
    min_y, max_y = min(y1, y2), max(y2, y1)
    for y in range(int(math.ceil(min_y)), int(math.floor(max_y)) + 1):
      if min_y <= y <= max_y:
        points.append((x1, float(y)))
  # Handle horizontal line case
  elif abs(y2 - y1) < 1e-10:  # Horizontal line
    min_x, max_x = min(x1, x2), max(x1, x2)
    for x in range(int(math.ceil(min_x)), int(math.floor(max_x)) + 1):
      if min_x <= x <= max_x:
        points.append((float(x), y1))
  else:
    # General case: line has slope
    # Find intersections with vertical grid lines (x = integer)
    min_x, max_x = min(x1, x2), max(x1, x2)
    for x in range(int(math.ceil(min_x)), int(math.floor(max_x)) + 1):
      if min_x <= x <= max_x:
        # Calculate y for this x using line equation
        t = (x - x1) / (x2 - x1)
        y = y1 + t * (y2 - y1)
        points.append((float(x), y))

    # Find intersections with horizontal grid lines (y = integer)
    min_y, max_y = min(y1, y2), max(y1, y2)
    for y in range(int(math.ceil(min_y)), int(math.floor(max_y)) + 1):
      if min_y <= y <= max_y:
        # Calculate x for this y using line equation
        t = (y - y1) / (y2 - y1)
        x = x1 + t * (x2 - x1)
        points.append((x, float(y)))

  # Remove duplicates and sort by distance from start
  unique_points = []
  for point in points:
    # Check if this point is already in the list (within tolerance)
    is_duplicate = False
    for existing_point in unique_points:
      if (
          abs(point[0] - existing_point[0]) < 1e-10
          and abs(point[1] - existing_point[1]) < 1e-10
      ):
        is_duplicate = True
        break
    if not is_duplicate:
      unique_points.append(point)

  # Sort by distance from start point
  def distance_from_start(point):
    return (point[0] - x1) ** 2 + (point[1] - y1) ** 2

  unique_points.sort(key=distance_from_start)

  return unique_points


def is_line_blocked(
    floor_plan: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    interior_wall_value: int = constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
    marked_value: int = TEMPORARY_MARKED_VALUE,
    blocked_value: int = TEMPORARY_BLOCKED_VALUE,
) -> bool:
  """Check if the line between start and end is blocked by walls.

  This function determines if a line of sight between two points is blocked
  by walls in the floor plan. It checks all grid intersections along the line
  and determines if the line is blocked by examining the 4 surrounding grid
  cells at each intersection point.

  Args:
      floor_plan: 2D numpy array representing the floor plan where different
          values represent different types of cells (walls, air, etc.).
      start: Starting point of the line as (x, y) coordinates.
      end: Ending point of the line as (x, y) coordinates.
      interior_wall_value: Value used to represent interior walls in the floor
          plan. Defaults to -3 (from "constants.py").
      marked_value: Value used to represent marked wall nodes. Only used
          internally. Defaults: -33. Only used internally.
      blocked_value: Value used to represent blocked wall nodes. Only used
          internally. Default: -34. Only used internally.


  Returns:
      True if the line is blocked by walls, False if the line of sight is clear.

  Note:
      The function considers a line blocked if all 4 grid cells surrounding
      an intersection point are walls (values -3, -33, or -34).
  """
  line_points = get_line_points(start, end)

  # Skip start and end points for blocking check
  for _, point in enumerate(line_points[1:-1], 1):
    x, y = point

    # Get 4 integer coordinates by rounding up/down
    coords = [
        (math.floor(x), math.floor(y)),
        (math.floor(x), math.ceil(y)),
        (math.ceil(x), math.floor(y)),
        (math.ceil(x), math.ceil(y)),
    ]

    # Check if all 4 coordinates are within bounds and get their values
    coord_values = []
    all_walls = True

    for cx, cy in coords:
      if 0 <= cx < floor_plan.shape[0] and 0 <= cy < floor_plan.shape[1]:
        value = floor_plan[cx, cy]
        coord_values.append(value)
        if (
            value != interior_wall_value
            and value != marked_value
            and value != blocked_value
        ):
          all_walls = False
      else:
        coord_values.append('OUT_OF_BOUNDS')
        all_walls = False

    # If all 4 coordinates are walls, the line is blocked
    if all_walls:
      return True

  return False


def are_neighbors(pos1: tuple[int, int], pos2: tuple[int, int]) -> bool:
  """Check if two positions are physically neighboring (adjacent).

  This function determines if two grid positions are adjacent to each other
  using 4-connectivity. Two positions are considered neighbors if they are
  within 1 unit distance in both x and y directions, but not the same position.

  Args:
      pos1: First position as (row, col) coordinates.
      pos2: Second position as (row, col) coordinates.

  Returns:
      True if the positions are neighbors, False otherwise.


  """
  dx = abs(pos1[0] - pos2[0])
  dy = abs(pos1[1] - pos2[1])
  return (dx == 1 and dy == 0) or (dx == 0 and dy == 1)


def mark_directly_seeing_nodes(
    floor_plan: np.ndarray,
    base_node: tuple[int, int],
    interior_wall_value: int = constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
    marked_value: int = TEMPORARY_MARKED_VALUE,
    blocked_value: int = TEMPORARY_BLOCKED_VALUE,
    air_value: int = constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
) -> np.ndarray:
  """Mark nodes that are directly seeing the base node as blocked_value.

  This function identifies and marks wall nodes that have a direct line of sight
  to the base node. It processes all connected wall nodes (marked with
  marked_value) and determines which ones can directly see the base node without
  being blocked by other walls. Additionally, it marks air nodes along unblocked
  lines of sight between wall nodes for interior mass radiative heat transfer.

  When the base node is an air cell, it finds directly seeing nodes among the
  interior walls, but does NOT mark air nodes as `AIR_IN_LINE_OF_SIGHT`.

  Args:
      floor_plan: 2D numpy array representing the floor plan where different
          values represent different types of cells (walls, air, etc.).
      base_node: Position of the base node as (row, col) coordinates. Can be
          either an interior wall node or an air cell.
      interior_wall_value: Value used to represent interior walls in the floor
          plan. Defaults to -3 (from "constants.py").
      marked_value: Value used to represent connected wall nodes that should
          be checked for line of sight. Only used internally. Defaults to -33.
      blocked_value: Value used to mark nodes that cannot directly see the
          base node. Only used internally. Defaults to -34.
      air_value: Value used to represent air spaces in the floor plan.
          Defaults to 0 (from "constants.py").

  Returns:
      Copy of the floor plan with nodes marked according to their visibility
          to the base node. Nodes that cannot see the base node are marked
          with blocked_value, and the base node itself is marked with
          blocked_value + marked_value. When starting from a wall node, air
          nodes along unblocked lines of sight are marked with
          `AIR_IN_LINE_OF_SIGHT` (9).
          When starting from an air node, air nodes are NOT marked.
          Air nodes along blocked lines remain as air_value (0).

  Note:
      - When starting from a wall node: Neighboring wall nodes are automatically
        marked as blocked (no line of sight calculation needed, and no air nodes
        between directly adjacent walls).
      - When starting from an air node: Neighboring wall nodes are NOT marked as
        blocked; they remain as marked_value and can participate in radiative
        transfer.
      - For non-neighboring nodes, the function first checks if the line of
        sight is blocked by walls using is_line_blocked().
      - When starting from a wall node: Air nodes are ONLY marked as 9 along
        lines that are NOT blocked. If a line is blocked, air nodes along that
        line remain as 0 (air_value).
      - When starting from an air node: Air nodes are NOT marked, even along
        unblocked lines.
      - The base node itself is marked with a special value to distinguish it.
      - Value meanings for radiative heat transfer:
        * marked_value (-33): Interior wall nodes connected to the same air
          space (can participate in radiative transfer)
        * blocked_value (-34): Interior wall nodes that cannot see the base node
          (blocked from radiative transfer)
        * blocked_value + marked_value (-67): The starting node itself
        * `AIR_IN_LINE_OF_SIGHT` (9): Air nodes along unblocked line of sight
          between wall nodes (for interior mass radiative transfer)
  """
  floor_plan_copy = floor_plan.copy()
  base_row, base_col = base_node
  base_cell_value = floor_plan_copy[base_row, base_col]
  is_base_air = base_cell_value == air_value

  # Find all marked_value nodes (connected wall nodes)
  connected_nodes = np.where(floor_plan_copy == marked_value)
  connected_positions = list(zip(connected_nodes[0], connected_nodes[1]))

  directly_seeing_count = 0

  for pos in connected_positions:
    row, col = pos

    # Skip if it's the base node itself
    if (row, col) == (base_row, base_col):
      continue
    # Check if not physically neighboring
    is_neighbor = are_neighbors((base_row, base_col), (row, col))

    if is_neighbor:
      # Neighbors are directly adjacent
      # Only mark as blocked if starting from an interior wall node
      # (not when starting from an air node)
      if not is_base_air:
        # When starting from wall, mark neighboring walls as blocked
        # (no air nodes between directly adjacent wall nodes)
        floor_plan_copy[row, col] = blocked_value
      # When starting from air node, leave neighboring walls as marked_value
      # (they can participate in radiative transfer)
    else:
      # Check if line of sight is blocked first
      blocked = is_line_blocked(
          floor_plan_copy,
          (base_row, base_col),
          (row, col),
          interior_wall_value,
          marked_value,
          blocked_value,
      )

      if blocked:
        # Line is blocked, so mark the wall node as blocked
        # and DON'T mark air nodes along this line
        floor_plan_copy[row, col] = blocked_value
        directly_seeing_count += 1
      else:
        # Line is NOT blocked
        # Only mark air nodes along the line if starting from a wall node
        # (not when starting from an air node)
        if not is_base_air:
          line_points = get_line_points(
              (float(base_row), float(base_col)), (float(row), float(col))
          )

          # Mark air nodes along the line (excluding start and end points)
          for point in line_points[1:-1]:
            px, py = point
            # Check all 4 integer coordinates around the floating point
            for cx, cy in [
                (math.floor(px), math.floor(py)),
                (math.floor(px), math.ceil(py)),
                (math.ceil(px), math.floor(py)),
                (math.ceil(px), math.ceil(py)),
            ]:
              if (
                  0 <= cx < floor_plan_copy.shape[0]
                  and 0 <= cy < floor_plan_copy.shape[1]
                  and floor_plan_copy[cx, cy] == air_value
              ):
                floor_plan_copy[cx, cy] = AIR_IN_LINE_OF_SIGHT
        # Wall node is visible (not blocked), so leave it as marked_value (-33)

  # Mark the base node with a special value
  floor_plan_copy[base_row, base_col] = blocked_value + marked_value
  return floor_plan_copy


def _ensure_irradiance_components(
    irradiance_components: (
        solar_radiation.IrradianceComponents | Mapping[str, Any]
    ),
    solar_zenith: float | None = None,
    solar_azimuth: float | None = None,
) -> solar_radiation.IrradianceComponents:
  """Normalizes irradiance input to an IrradianceComponents instance.

  Args:
      irradiance_components: Either an IrradianceComponents dataclass or a
          mapping containing irradiance fields.
      solar_zenith: Optional fallback solar zenith angle in degrees. Used
          when the mapping does not contain 'solar_zenith'.
      solar_azimuth: Optional fallback solar azimuth angle in degrees. Used
          when the mapping does not contain 'solar_azimuth'.

  Returns:
      solar_radiation.IrradianceComponents: Irradiance data with guaranteed
      fields and types.

  Raises:
      KeyError: If 'ghi', 'dni', or 'dhi' are absent, or if 'solar_zenith'/
          'solar_azimuth' are absent and no fallback is provided.
  """
  if isinstance(irradiance_components, solar_radiation.IrradianceComponents):
    return irradiance_components

  required_keys = ('ghi', 'dni', 'dhi')
  missing_keys = [
      key for key in required_keys if key not in irradiance_components
  ]
  if missing_keys:
    missing_key_names = ', '.join(sorted(missing_keys))
    raise KeyError(f'Missing irradiance component keys: {missing_key_names}')

  # Use mapping values if available, else fall back to arguments
  sz = irradiance_components.get('solar_zenith', solar_zenith)
  sa = irradiance_components.get('solar_azimuth', solar_azimuth)

  if sz is None or sa is None:
    missing = []
    if sz is None:
      missing.append('solar_zenith')
    if sa is None:
      missing.append('solar_azimuth')
    missing_key_names = ', '.join(sorted(missing))
    raise KeyError(f'Missing irradiance component keys: {missing_key_names}')

  timestamp = irradiance_components.get('timestamp')
  return solar_radiation.IrradianceComponents(
      ghi=float(irradiance_components['ghi']),
      dni=float(irradiance_components['dni']),
      dhi=float(irradiance_components['dhi']),
      solar_zenith=float(sz),
      solar_azimuth=float(sa),
      timestamp=timestamp,
  )


def validate_fenestration_connectivity(
    floor_plan: np.ndarray,
    fenestration_value: int = constants.FENESTRATION_VALUE_IN_FILE_INPUT,
    air_value: int = constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT,
    exterior_space_value: int = constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT,
    interior_wall_value: int = constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT,
) -> bool:
  """Validate that fenestration nodes bridge exterior space to interior air."""
  if not np.any(floor_plan == fenestration_value):
    return True

  fenestration_groups = _find_connected_groups(floor_plan, fenestration_value)

  directions = FOUR_CONNECTED_DIRECTIONS
  rows, cols = floor_plan.shape

  for group_name, group_info in fenestration_groups.items():
    group_indices: Sequence[tuple[int, int]] = group_info['indices']

    nodes_adjacent_to_air: set[tuple[int, int]] = set()
    nodes_adjacent_to_exterior: set[tuple[int, int]] = set()
    nodes_surrounded_by_air: list[tuple[int, int]] = []

    for row, col in group_indices:
      adjacent_to_air = False
      adjacent_to_exterior = False
      air_neighbor_count = 0

      for dr, dc in directions:
        nr, nc = row + dr, col + dc

        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
          adjacent_to_exterior = True
          continue

        neighbor_val = floor_plan[nr, nc]

        if neighbor_val == air_value:
          adjacent_to_air = True
          air_neighbor_count += 1
        elif neighbor_val == exterior_space_value:
          adjacent_to_exterior = True
        elif neighbor_val == interior_wall_value:
          # Interior wall does not affect adjacent_to_air/exterior flags
          continue

      if adjacent_to_air:
        nodes_adjacent_to_air.add((row, col))
      if adjacent_to_exterior:
        nodes_adjacent_to_exterior.add((row, col))

      if air_neighbor_count == 4 and not adjacent_to_exterior:
        nodes_surrounded_by_air.append((row, col))

    if not nodes_adjacent_to_air:
      raise ValueError(
          f'Fenestration group {group_name} is not connected to indoor air. '
          f'Fenestration must border value {air_value} nodes.'
      )

    if not nodes_adjacent_to_exterior:
      raise ValueError(
          f'Fenestration group {group_name} is not exposed to exterior '
          f'(value {exterior_space_value}).'
      )

    if nodes_surrounded_by_air:
      raise ValueError(
          f'Fenestration group {group_name} has fenestration nodes fully '
          f'surrounded by air at positions {nodes_surrounded_by_air}. '
          'Fenestration must bridge exterior and interior.'
      )

    _validate_fenestration_chain_connectivity(
        floor_plan=floor_plan,
        group_name=group_name,
        group_indices=group_indices,
        nodes_adjacent_to_air=nodes_adjacent_to_air,
        nodes_adjacent_to_exterior=nodes_adjacent_to_exterior,
        fenestration_value=fenestration_value,
        air_value=air_value,
    )

  return True


def _validate_fenestration_chain_connectivity(
    floor_plan: np.ndarray,
    group_name: str,
    group_indices: Sequence[tuple[int, int]],
    nodes_adjacent_to_air: set[tuple[int, int]],
    nodes_adjacent_to_exterior: set[tuple[int, int]],
    fenestration_value: int,
    air_value: int,
) -> None:
  """Ensure each fenestration node can reach both exterior and interior."""
  directions = FOUR_CONNECTED_DIRECTIONS
  direction_names = {
      (-1, 0): 'north',
      (1, 0): 'south',
      (0, -1): 'west',
      (0, 1): 'east',
  }
  rows, cols = floor_plan.shape
  group_set = set(group_indices)

  reachable_from_exterior: set[tuple[int, int]] = set()
  queue = deque(nodes_adjacent_to_exterior)
  reachable_from_exterior.update(nodes_adjacent_to_exterior)

  while queue:
    row, col = queue.popleft()
    for dr, dc in directions:
      nr, nc = row + dr, col + dc
      if (nr, nc) in group_set and (nr, nc) not in reachable_from_exterior:
        reachable_from_exterior.add((nr, nc))
        queue.append((nr, nc))

  reachable_from_air: set[tuple[int, int]] = set()
  queue = deque(nodes_adjacent_to_air)
  reachable_from_air.update(nodes_adjacent_to_air)

  while queue:
    row, col = queue.popleft()
    for dr, dc in directions:
      nr, nc = row + dr, col + dc
      if (nr, nc) in group_set and (nr, nc) not in reachable_from_air:
        reachable_from_air.add((nr, nc))
        queue.append((nr, nc))

  for row, col in group_indices:
    if (row, col) not in reachable_from_exterior:
      raise ValueError(
          f'Fenestration group {group_name} has node {(row, col)} blocked '
          'from exterior exposure.'
      )
    if (row, col) not in reachable_from_air:
      raise ValueError(
          f'Fenestration group {group_name} has node {(row, col)} blocked '
          'from interior air.'
      )

  interior_direction = _determine_interior_direction(
      nodes_adjacent_to_exterior, rows, cols, floor_plan
  )

  if interior_direction is None:
    return

  blocked_by_interior_wall: list[tuple[int, int]] = []
  dr, dc = interior_direction
  for row, col in group_indices:
    nr, nc = row + dr, col + dc
    if (nr, nc) in group_set:
      continue
    if 0 <= nr < rows and 0 <= nc < cols:
      neighbor_val = floor_plan[nr, nc]
      if neighbor_val not in (air_value, fenestration_value):
        blocked_by_interior_wall.append((row, col))
    else:
      blocked_by_interior_wall.append((row, col))

  if blocked_by_interior_wall:
    direction_name = direction_names.get(
        interior_direction, str(interior_direction)
    )
    raise ValueError(
        f'Fenestration group {group_name} has interior-facing nodes blocked '
        f'by non-air cells in the {direction_name} direction: '
        f'{blocked_by_interior_wall}.'
    )


def _determine_interior_direction(
    nodes_adjacent_to_exterior: set[tuple[int, int]],
    rows: int,
    cols: int,
    floor_plan: np.ndarray,
) -> tuple[int, int] | None:
  """Infer the direction pointing from exterior toward interior air."""
  if not nodes_adjacent_to_exterior:
    return None

  for row, col in nodes_adjacent_to_exterior:
    if row == 0:
      return (1, 0)
    if row == rows - 1:
      return (-1, 0)
    if col == 0:
      return (0, 1)
    if col == cols - 1:
      return (0, -1)

  directions = FOUR_CONNECTED_DIRECTIONS
  opposite = {
      (-1, 0): (1, 0),
      (1, 0): (-1, 0),
      (0, -1): (0, 1),
      (0, 1): (0, -1),
  }
  exterior_space_value = constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT

  for row, col in nodes_adjacent_to_exterior:
    for dr, dc in directions:
      nr, nc = row + dr, col + dc
      if (
          0 <= nr < rows
          and 0 <= nc < cols
          and floor_plan[nr, nc] == exterior_space_value
      ):
        return opposite[(dr, dc)]
  return None


def _find_connected_groups(
    floor_plan: np.ndarray,
    target_value: int,
) -> dict[str, dict[str, Any]]:
  """Find 4-connected groups of `target_value`cells and return their indices."""
  visited = np.zeros_like(floor_plan, dtype=bool)
  groups: dict[str, dict[str, Any]] = {}
  group_count = 0
  directions = FOUR_CONNECTED_DIRECTIONS

  for row in range(floor_plan.shape[0]):
    for col in range(floor_plan.shape[1]):
      if floor_plan[row, col] == target_value and not visited[row, col]:
        group_count += 1
        group_indices: list[tuple[int, int]] = []
        queue = deque([(row, col)])
        visited[row, col] = True

        while queue:
          r, c = queue.popleft()
          group_indices.append((r, c))

          for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < floor_plan.shape[0]
                and 0 <= nc < floor_plan.shape[1]
                and floor_plan[nr, nc] == target_value
                and not visited[nr, nc]
            ):
              visited[nr, nc] = True
              queue.append((nr, nc))

        groups[f'group_{group_count}'] = {
            'count': len(group_indices),
            'indices': group_indices,
        }

  return groups


def mark_fenestration_positions(
    floor_plan: np.ndarray,
    fenestration_value: int = constants.FENESTRATION_VALUE_IN_FUNCTION,
    exterior_space_value: int = constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION,
    air_value: int = constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
    exterior_fenestration: int = constants.EXTERIOR_FENESTRATION_VALUE,
    interior_fenestration: int = constants.INTERIOR_FENESTRATION_VALUE,
    inbetween_fenestration: int = constants.INBETWEEN_FENESTRATION_VALUE,
) -> np.ndarray:
  """Classify fenestration nodes into exterior, interior, or in-between."""
  result = floor_plan.copy()
  rows, cols = floor_plan.shape
  directions = FOUR_CONNECTED_DIRECTIONS

  fen_rows, fen_cols = np.where(floor_plan == fenestration_value)
  for row, col in zip(fen_rows, fen_cols):
    adjacent_to_exterior = False
    adjacent_to_air = False
    at_boundary = row == 0 or row == rows - 1 or col == 0 or col == cols - 1

    for dr, dc in directions:
      nr, nc = row + dr, col + dc
      if 0 <= nr < rows and 0 <= nc < cols:
        neighbor_val = floor_plan[nr, nc]
        if neighbor_val == exterior_space_value:
          adjacent_to_exterior = True
        elif neighbor_val == air_value:
          adjacent_to_air = True

    if at_boundary or adjacent_to_exterior and not adjacent_to_air:
      result[row, col] = exterior_fenestration
    elif adjacent_to_air and not adjacent_to_exterior:
      result[row, col] = interior_fenestration
    elif adjacent_to_air and adjacent_to_exterior:
      result[row, col] = interior_fenestration
    else:
      result[row, col] = inbetween_fenestration

  return result


def group_fenestrations(
    floor_plan: np.ndarray,
    exterior_fenestration: int = constants.EXTERIOR_FENESTRATION_VALUE,
    interior_fenestration: int = constants.INTERIOR_FENESTRATION_VALUE,
    inbetween_fenestration: int = constants.INBETWEEN_FENESTRATION_VALUE,
    exterior_space_value: int = constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION,
    floor_plan_orientation: float = 0.0,
) -> dict[str, dict[str, Any]]:
  """Group adjacent fenestration nodes and compute surface properties."""
  fenestration_mask = (
      (floor_plan == exterior_fenestration)
      | (floor_plan == interior_fenestration)
      | (floor_plan == inbetween_fenestration)
  )

  visited = np.zeros_like(floor_plan, dtype=bool)
  groups: dict[str, dict[str, Any]] = {}
  group_count = 0
  directions = FOUR_CONNECTED_DIRECTIONS

  for row in range(floor_plan.shape[0]):
    for col in range(floor_plan.shape[1]):
      if fenestration_mask[row, col] and not visited[row, col]:
        group_count += 1
        group_indices: list[tuple[int, int]] = []
        exterior_count = 0
        queue = deque([(row, col)])
        visited[row, col] = True

        while queue:
          r, c = queue.popleft()
          group_indices.append((r, c))

          if floor_plan[r, c] == exterior_fenestration:
            exterior_count += 1

          for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < floor_plan.shape[0]
                and 0 <= nc < floor_plan.shape[1]
                and fenestration_mask[nr, nc]
                and not visited[nr, nc]
            ):
              visited[nr, nc] = True
              queue.append((nr, nc))

        indices_array = np.zeros_like(floor_plan, dtype=bool)
        for r, c in group_indices:
          indices_array[r, c] = True

        exterior_indices_array = np.zeros_like(floor_plan, dtype=bool)
        for r, c in group_indices:
          if floor_plan[r, c] == exterior_fenestration:
            exterior_indices_array[r, c] = True

        azimuth = _determine_fenestration_azimuth(
            floor_plan,
            group_indices,
            exterior_space_value,
            exterior_fenestration,
            floor_plan_orientation,
        )

        phi = float(constants.FENESTRATION_TILT_ANGLE)
        phi_rad = math.radians(phi)
        cos_phi = math.cos(phi_rad)

        factor = 0.5 * (1 + cos_phi)
        f_gnd = 0.5 * (1 - cos_phi)
        f_sky = factor * math.sqrt(factor)
        f_air = factor * (1 - math.sqrt(factor))
        beta = math.sqrt(factor)

        groups[f'fenestration_{group_count}'] = {
            'count': len(group_indices),
            'exterior_count': exterior_count,
            'indices': group_indices,
            'indices_array': indices_array,
            'exterior_indices_array': exterior_indices_array,
            'phi': phi,
            'azimuth': float(azimuth),
            'F_gnd': f_gnd,
            'F_sky': f_sky,
            'F_air': f_air,
            'beta': beta,
        }

  return groups


def _determine_fenestration_azimuth(
    floor_plan: np.ndarray,
    group_indices: Sequence[tuple[int, int]],
    exterior_space_value: int = constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION,
    exterior_fenestration_value: int = constants.EXTERIOR_FENESTRATION_VALUE,
    floor_plan_orientation: float = 0.0,
) -> float:
  """Determine fenestration azimuth from exterior adjacency using atan2.

  The azimuth is inferred from the floor_plan grid using the Cartesian
  coordinate system. The direction vector is computed from exterior
  fenestration nodes toward adjacent exterior space nodes, then the azimuth
  is calculated using atan2.

  The floor_plan grid uses (row, col) coordinates where:
  - row increases downward (South in default orientation)
  - col increases rightward (East in default orientation)

  In the Cartesian mapping:
  - x = col direction (East)
  - y = -row direction (North, since row increases downward)

  Azimuth is measured clockwise from North:
  - 0° = North (up/decreasing row)
  - 90° = East (right/increasing col)
  - 180° = South (down/increasing row)
  - 270° = West (left/decreasing col)

  The computed azimuth is then offset by floor_plan_orientation.

  Args:
    floor_plan: 2D array of the indexed floor plan.
    group_indices: List of (row, col) indices for the fenestration group.
    exterior_space_value: Value representing exterior space in the floor plan.
    exterior_fenestration_value: Value representing exterior fenestration.
    floor_plan_orientation: Compass angle (degrees) of the floor-plan's
      "up" direction. 0/360 = North, 90 = East, 180 = South, 270 = West.

  Returns:
    Azimuth angle in degrees [0, 360).
  """
  rows, cols = floor_plan.shape
  directions = FOUR_CONNECTED_DIRECTIONS

  # Accumulate direction vectors from exterior fenestration toward exterior
  # space in Cartesian coordinates (dx = col direction, dy = -row direction)
  total_dx = 0.0
  total_dy = 0.0
  count = 0

  for row, col in group_indices:
    if floor_plan[row, col] != exterior_fenestration_value:
      continue
    for dr, dc in directions:
      nr, nc = row + dr, col + dc
      # Check adjacency to exterior space or array boundary
      if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
        # Array boundary counts as exterior
        total_dx += float(dc)
        total_dy += float(-dr)
        count += 1
      elif floor_plan[nr, nc] == exterior_space_value:
        total_dx += float(dc)
        total_dy += float(-dr)
        count += 1

  if count == 0:
    # Fallback: check all nodes (not just exterior fenestration)
    for row, col in group_indices:
      for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
          total_dx += float(dc)
          total_dy += float(-dr)
          count += 1
        elif floor_plan[nr, nc] == exterior_space_value:
          total_dx += float(dc)
          total_dy += float(-dr)
          count += 1

  if count == 0:
    # Default to North if no direction can be determined
    return float(floor_plan_orientation) % 360.0

  # Compute azimuth using atan2
  # atan2(x, y) gives angle clockwise from North (y-axis)
  azimuth_rad = math.atan2(total_dx, total_dy)
  azimuth_deg = math.degrees(azimuth_rad)

  # Normalize to [0, 360)
  azimuth_deg = azimuth_deg % 360.0

  # Apply floor_plan_orientation offset
  return (azimuth_deg + float(floor_plan_orientation)) % 360.0


def group_air_nodes(
    floor_plan: np.ndarray,
    air_value: int = constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
    exterior_fenestration: int = constants.EXTERIOR_FENESTRATION_VALUE,
    interior_fenestration: int = constants.INTERIOR_FENESTRATION_VALUE,
    inbetween_fenestration: int = constants.INBETWEEN_FENESTRATION_VALUE,
    fenestration_groups: Mapping[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
  """Group connected interior air nodes and annotate adjacent fenestrations.

  Finds all connected components of air nodes (using 4-connectivity) and
  records which fenestration groups are adjacent to each air group.

  Args:
    floor_plan: 2D indexed floor plan array.
    air_value: Value representing interior air spaces.
    exterior_fenestration: Value for exterior fenestration nodes.
    interior_fenestration: Value for interior fenestration nodes.
    inbetween_fenestration: Value for in-between fenestration nodes.
    fenestration_groups: Pre-computed fenestration groups (from
      :func:`group_fenestrations`). If provided, each air group will list
      which fenestration groups are adjacent.

  Returns:
    Dictionary mapping group names to group info dicts with keys:
      'count', 'indices', 'indices_array', 'fenestration_groups'.
  """
  visited = np.zeros_like(floor_plan, dtype=bool)
  groups: dict[str, dict[str, Any]] = {}
  group_count = 0
  directions = FOUR_CONNECTED_DIRECTIONS

  fenestration_index_to_group: dict[tuple[int, int], str] = {}
  if fenestration_groups:
    for group_name, group_info in fenestration_groups.items():
      for idx in group_info['indices']:
        fenestration_index_to_group[idx] = group_name

  fenestration_values = {
      exterior_fenestration,
      interior_fenestration,
      inbetween_fenestration,
  }

  for row in range(floor_plan.shape[0]):
    for col in range(floor_plan.shape[1]):
      if floor_plan[row, col] == air_value and not visited[row, col]:
        group_count += 1
        group_indices: list[tuple[int, int]] = []
        adjacent_fenestrations: set[str] = set()
        queue = deque([(row, col)])
        visited[row, col] = True

        while queue:
          r, c = queue.popleft()
          group_indices.append((r, c))

          for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < floor_plan.shape[0] and 0 <= nc < floor_plan.shape[1]:
              neighbor_val = floor_plan[nr, nc]

              if neighbor_val in fenestration_values:
                group_name = fenestration_index_to_group.get((nr, nc))
                if group_name:
                  adjacent_fenestrations.add(group_name)

              if neighbor_val == air_value and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))

        indices_array = np.zeros_like(floor_plan, dtype=bool)
        for r, c in group_indices:
          indices_array[r, c] = True

        groups[f'air_{group_count}'] = {
            'count': len(group_indices),
            'indices': group_indices,
            'indices_array': indices_array,
            'fenestration_groups': sorted(adjacent_fenestrations),
        }

  return groups


def calculate_solar_absorbed_for_fenestration_group(
    fenestration_group: Mapping[str, Any],
    irradiance_components: (
        solar_radiation.IrradianceComponents | Mapping[str, Any]
    ),
    solar_zenith: float,
    solar_azimuth: float,
    alpha: float = constants.FENESTRATION_SOLAR_ABSORPTANCE,
) -> float:
  """Compute absorbed solar flux per node for a fenestration group."""
  exterior_count = int(fenestration_group.get('exterior_count', 0))
  if exterior_count == 0:
    return 0.0

  irr = _ensure_irradiance_components(
      irradiance_components, solar_zenith, solar_azimuth
  )

  surface_tilt = float(fenestration_group['phi'])
  surface_azimuth = float(fenestration_group['azimuth'])

  g_ts = solar_radiation.calculate_poa_irradiance(
      irr,
      surface_tilt,
      surface_azimuth,
      solar_zenith,
      solar_azimuth,
  )

  total_absorbed = g_ts * alpha * exterior_count
  total_count = int(fenestration_group.get('count', 0))
  return total_absorbed / total_count if total_count > 0 else 0.0


def net_solar_absorbed_heatflux_fenestration(
    floor_plan: np.ndarray,
    fenestration_groups: Mapping[str, Mapping[str, Any]] | None,
    irradiance_components: (
        solar_radiation.IrradianceComponents | Mapping[str, Any]
    ),
    solar_zenith: float,
    solar_azimuth: float,
    alpha: float = constants.FENESTRATION_SOLAR_ABSORPTANCE,
) -> np.ndarray:
  """Return array of absorbed solar heat flux at each fenestration node."""
  q_sol_alpha_array = np.zeros_like(floor_plan, dtype=float)

  if not fenestration_groups:
    return q_sol_alpha_array

  for group_info in fenestration_groups.values():
    q_sol_alpha_per_node = calculate_solar_absorbed_for_fenestration_group(
        group_info,
        irradiance_components,
        solar_zenith,
        solar_azimuth,
        alpha,
    )
    indices_array = group_info['indices_array']
    q_sol_alpha_array[indices_array] += q_sol_alpha_per_node

  return q_sol_alpha_array


def calculate_solar_transmitted_for_fenestration_group(
    fenestration_group: Mapping[str, Any],
    irradiance_components: (
        solar_radiation.IrradianceComponents | Mapping[str, Any]
    ),
    solar_zenith: float,
    solar_azimuth: float,
    tau: float = constants.FENESTRATION_SOLAR_TRANSMITTANCE,
) -> float:
  """Compute total transmitted solar radiation for a fenestration group."""
  exterior_count = int(fenestration_group.get('exterior_count', 0))
  if exterior_count == 0:
    return 0.0

  irr = _ensure_irradiance_components(
      irradiance_components, solar_zenith, solar_azimuth
  )

  surface_tilt = float(fenestration_group['phi'])
  surface_azimuth = float(fenestration_group['azimuth'])

  g_ts = solar_radiation.calculate_poa_irradiance(
      irr,
      surface_tilt,
      surface_azimuth,
      solar_zenith,
      solar_azimuth,
  )

  return g_ts * tau * exterior_count


def net_solar_transmitted_heatflux_fenestration(
    floor_plan: np.ndarray,
    fenestration_groups: Mapping[str, Mapping[str, Any]] | None,
    air_groups: Mapping[str, Mapping[str, Any]] | None,
    irradiance_components: (
        solar_radiation.IrradianceComponents | Mapping[str, Any]
    ),
    solar_zenith: float,
    solar_azimuth: float,
    tau: float = constants.FENESTRATION_SOLAR_TRANSMITTANCE,
) -> np.ndarray:
  """Distribute transmitted solar heat flux to air nodes connected to windows"""
  q_sol_tau_array = np.zeros_like(floor_plan, dtype=float)

  if not fenestration_groups or not air_groups:
    return q_sol_tau_array

  fenestration_q_sol_tau: dict[str, float] = {}
  for group_name, group_info in fenestration_groups.items():
    fenestration_q_sol_tau[group_name] = (
        calculate_solar_transmitted_for_fenestration_group(
            group_info,
            irradiance_components,
            solar_zenith,
            solar_azimuth,
            tau,
        )
    )

  for air_group_info in air_groups.values():
    connected_fenestrations = air_group_info.get('fenestration_groups', [])
    total_q_sol_tau = sum(
        fenestration_q_sol_tau.get(group, 0.0)
        for group in connected_fenestrations
    )

    if total_q_sol_tau == 0.0:
      continue

    air_count = int(air_group_info['count'])
    q_sol_tau_per_node = total_q_sol_tau / air_count if air_count > 0 else 0.0

    indices_array = air_group_info['indices_array']
    q_sol_tau_array[indices_array] += q_sol_tau_per_node

  return q_sol_tau_array


# ---------------------------------------------------------------------------
# Interior surface adjacency (walls + fenestration)
# ---------------------------------------------------------------------------


def mark_interior_surface_adjacent_to_air(
    floor_plan: np.ndarray,
    interior_wall_value: int = constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
    interior_fenestration_value: int = constants.INTERIOR_FENESTRATION_VALUE,
    air_value: int = constants.INTERIOR_SPACE_VALUE_IN_FUNCTION,
) -> np.ndarray:
  """Mark interior surfaces (walls and fenestration) adjacent to air.

  Creates a boolean mask identifying interior walls and interior fenestration
  nodes that share an edge with an air space (value 0) in the floor plan.
  Checks adjacency in 4 directions (up, down, left, right).

  Args:
    floor_plan: 2D array representing the indexed floor plan.
    interior_wall_value: Value used to represent interior walls.
    interior_fenestration_value: Value for interior fenestration nodes.
    air_value: Value for interior air spaces.

  Returns:
    Boolean mask where True indicates an interior surface (wall or
    fenestration) that is adjacent to at least one air space.
  """
  mask_surface = (floor_plan == interior_wall_value) | (
      floor_plan == interior_fenestration_value
  )
  mask_air = floor_plan == air_value

  contact = np.zeros_like(floor_plan, dtype=bool)
  # up
  contact[1:, :] |= mask_air[:-1, :] & mask_surface[1:, :]
  # down
  contact[:-1, :] |= mask_air[1:, :] & mask_surface[:-1, :]
  # left
  contact[:, 1:] |= mask_air[:, :-1] & mask_surface[:, 1:]
  # right
  contact[:, :-1] |= mask_air[:, 1:] & mask_surface[:, :-1]

  return mask_surface & contact


# ---------------------------------------------------------------------------
# Exterior longwave radiation (LWR) for fenestration
# ---------------------------------------------------------------------------


def calculate_exterior_lwr_for_fenestration_group(
    fenestration_group: Mapping[str, Any],
    surface_temperatures: np.ndarray,
    emissivity_array: np.ndarray,
    ambient_temperature: float,
    sky_temperature: float,
) -> float:
  """Compute net exterior longwave radiative heat flux for a fenestration group.

  Calculates the net longwave radiation exchange between the fenestration
  surface and the sky, ground, and surrounding air using:

    q_lwr = epsilon * sigma * (F_sky*(T_sky^4 - T_s^4) +
                               F_gnd*(T_air^4 - T_s^4) +
                               F_air*(T_air^4 - T_s^4))

  The result is the net heat flux INTO the surface (positive means surface
  gains heat).

  Args:
    fenestration_group: Dictionary containing fenestration group properties
      including 'F_sky', 'F_gnd', 'F_air', 'exterior_indices_array'.
    surface_temperatures: 2D array of surface temperatures in K.
    emissivity_array: 2D array of surface emissivities.
    ambient_temperature: Outdoor dry-bulb temperature in K.
    sky_temperature: Sky temperature in K.

  Returns:
    Net LWR heat flux in W/m² (positive = surface gains heat).
  """
  sigma = constants.STEFAN_BOLTZMANN_CONSTANT
  f_sky = float(fenestration_group['F_sky'])
  f_gnd = float(fenestration_group['F_gnd'])
  f_air = float(fenestration_group['F_air'])

  exterior_mask = fenestration_group['exterior_indices_array']

  # Average surface temperature and emissivity over exterior fenestration nodes
  if np.any(exterior_mask):
    t_surf = np.mean(surface_temperatures[exterior_mask])
    epsilon = np.mean(emissivity_array[exterior_mask])
  else:
    return 0.0

  t_sky4 = sky_temperature**4
  t_air4 = ambient_temperature**4
  t_surf4 = t_surf**4

  q_lwr = (
      epsilon
      * sigma
      * (
          f_sky * (t_sky4 - t_surf4)
          + f_gnd * (t_air4 - t_surf4)
          + f_air * (t_air4 - t_surf4)
      )
  )

  return float(q_lwr)


def net_exterior_radiative_heatflux(
    floor_plan: np.ndarray,
    fenestration_groups: Mapping[str, Mapping[str, Any]] | None,
    surface_temperatures: np.ndarray,
    emissivity_array: np.ndarray,
    ambient_temperature: float,
    sky_temperature: float,
) -> np.ndarray:
  """Compute net exterior LWR heat flux array for all fenestration nodes.

  For each fenestration group, calculates the net longwave radiation exchange
  and assigns the result uniformly to all nodes in the group.

  Args:
    floor_plan: 2D indexed floor plan array.
    fenestration_groups: Dictionary of fenestration group properties.
    surface_temperatures: 2D array of surface temperatures in K.
    emissivity_array: 2D array of surface emissivities.
    ambient_temperature: Outdoor dry-bulb temperature in K.
    sky_temperature: Sky temperature in K.

  Returns:
    2D array of net LWR heat flux at each grid position.
  """
  q_lwr_array = np.zeros_like(floor_plan, dtype=float)

  if not fenestration_groups:
    return q_lwr_array

  for group_info in fenestration_groups.values():
    q_lwr = calculate_exterior_lwr_for_fenestration_group(
        group_info,
        surface_temperatures,
        emissivity_array,
        ambient_temperature,
        sky_temperature,
    )
    indices_array = group_info['indices_array']
    q_lwr_array[indices_array] = q_lwr

  return q_lwr_array


# ---------------------------------------------------------------------------
# Exterior wall boundary mask
# ---------------------------------------------------------------------------


def get_exterior_wall_boundary_mask(
    floor_plan: np.ndarray,
    wall_value: int = constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT,
    exterior_space_value: int = constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION,
    air_value: int = constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT,
) -> np.ndarray:
  """Identify exterior wall nodes that form the building boundary.

  A wall node is considered an exterior wall boundary if:
  1. It has the specified wall_value.
  2. It is NOT adjacent to an enclosed interior air space.

  Walls adjacent to enclosed interior air (courtyard, rooms) are excluded
  because they face inward. Walls at the array boundary or adjacent to
  exterior_space_value are included.

  Interior walls (value 1) are treated as solid material and do NOT cause
  exclusion of adjacent exterior walls.

  Args:
    floor_plan: 2D array of the floor plan (can use FILE_INPUT or FUNCTION
      values).
    wall_value: The value representing exterior walls.
    exterior_space_value: The value representing exterior space.
    air_value: The value representing interior air spaces.

  Returns:
    Boolean mask where True indicates an exterior wall boundary node.
  """
  rows, cols = floor_plan.shape
  wall_mask = floor_plan == wall_value

  if not np.any(wall_mask):
    return np.zeros_like(floor_plan, dtype=bool)

  # Find all enclosed interior air regions using flood-fill from edges
  # Air connected to edges or exterior space is NOT enclosed interior air
  air_mask = floor_plan == air_value
  if not np.any(air_mask):
    # No interior air, all walls are exterior boundaries
    return wall_mask

  # Find enclosed air: air NOT connected to exterior
  # Use flood fill from exterior space or from array boundary
  visited_air = np.zeros_like(floor_plan, dtype=bool)
  exterior_mask = floor_plan == exterior_space_value

  # BFS from all exterior space nodes and boundary nodes
  queue = deque()

  # Add all exterior space nodes to start BFS
  for r in range(rows):
    for c in range(cols):
      if exterior_mask[r, c]:
        queue.append((r, c))
        visited_air[r, c] = True

  # Also check air at array boundaries (connected to outside)
  for r in range(rows):
    for c in range(cols):
      if (
          air_mask[r, c]
          and not visited_air[r, c]
          and (r == 0 or r == rows - 1 or c == 0 or c == cols - 1)
      ):
        queue.append((r, c))
        visited_air[r, c] = True

  directions = FOUR_CONNECTED_DIRECTIONS
  while queue:
    r, c = queue.popleft()
    for dr, dc in directions:
      nr, nc = r + dr, c + dc
      if 0 <= nr < rows and 0 <= nc < cols and not visited_air[nr, nc]:
        if air_mask[nr, nc] or exterior_mask[nr, nc]:
          visited_air[nr, nc] = True
          queue.append((nr, nc))

  # Enclosed interior air = air nodes NOT visited (not connected to exterior)
  enclosed_air = air_mask & ~visited_air

  # Walls adjacent to enclosed air should be excluded UNLESS they are also
  # at the array boundary or adjacent to exterior space
  adjacent_to_enclosed = np.zeros_like(floor_plan, dtype=bool)
  for dr, dc in directions:
    shifted = np.zeros_like(floor_plan, dtype=bool)
    if dr == -1:
      shifted[:-1, :] = enclosed_air[1:, :]
    elif dr == 1:
      shifted[1:, :] = enclosed_air[:-1, :]
    elif dc == -1:
      shifted[:, :-1] = enclosed_air[:, 1:]
    elif dc == 1:
      shifted[:, 1:] = enclosed_air[:, :-1]
    adjacent_to_enclosed |= shifted

  # Determine which walls are at the array boundary
  at_boundary = np.zeros_like(floor_plan, dtype=bool)
  at_boundary[0, :] = True
  at_boundary[-1, :] = True
  at_boundary[:, 0] = True
  at_boundary[:, -1] = True

  # Determine which walls are adjacent to exterior space
  adjacent_to_exterior = np.zeros_like(floor_plan, dtype=bool)
  for dr, dc in directions:
    shifted = np.zeros_like(floor_plan, dtype=bool)
    if dr == -1:
      shifted[:-1, :] = exterior_mask[1:, :]
    elif dr == 1:
      shifted[1:, :] = exterior_mask[:-1, :]
    elif dc == -1:
      shifted[:, :-1] = exterior_mask[:, 1:]
    elif dc == 1:
      shifted[:, 1:] = exterior_mask[:, :-1]
    adjacent_to_exterior |= shifted

  # A wall faces outward if it's at boundary or adjacent to exterior space
  faces_outward = at_boundary | adjacent_to_exterior

  # Exterior wall boundary = wall AND(faces outward OR NOT adjacent to enclosed)
  # i.e., exclude walls ONLY adjacent to enclosed air that don't face outward
  return wall_mask & (faces_outward | ~adjacent_to_enclosed)


# ---------------------------------------------------------------------------
# Exterior wall azimuth determination
# ---------------------------------------------------------------------------


def determine_exterior_wall_azimuth_array(
    exterior_wall_boundary_mask: np.ndarray,
    floor_plan: np.ndarray,
    exterior_space_value: int = constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION,
    floor_plan_orientation: float = 0.0,
) -> np.ndarray:
  """Determine azimuth for each exterior wall boundary node.

  For each exterior wall boundary node, computes the outward-facing direction
  (toward exterior space) and converts it to an azimuth angle using atan2
  in the Cartesian coordinate system.

  The grid coordinate system:
  - row increases downward (South in default orientation)
  - col increases rightward (East in default orientation)

  Cartesian mapping:
  - x = col direction (East)
  - y = -row direction (North)

  Azimuth (clockwise from North):
  - 0° = North (decreasing row)
  - 90° = East (increasing col)
  - 180° = South (increasing row)
  - 270° = West (decreasing col)

  For corner nodes with multiple adjacent exterior space directions, the
  resulting azimuth is the average direction (e.g. top-right = 45°).

  Args:
    exterior_wall_boundary_mask: Boolean mask of exterior wall nodes.
    floor_plan: 2D indexed floor plan array.
    exterior_space_value: Value representing exterior space.
    floor_plan_orientation: Compass angle (degrees) of the floor-plan's
      "up" direction. 0/360 = North.

  Returns:
    2D array of azimuth angles in degrees. Non-wall positions are 0.0.
  """
  rows, cols = floor_plan.shape
  azimuth_array = np.zeros((rows, cols), dtype=float)
  directions = FOUR_CONNECTED_DIRECTIONS

  for r in range(rows):
    for c in range(cols):
      if not exterior_wall_boundary_mask[r, c]:
        continue

      total_dx = 0.0
      total_dy = 0.0

      for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
          # Array boundary counts as exterior
          total_dx += float(dc)
          total_dy += float(-dr)
        elif floor_plan[nr, nc] == exterior_space_value:
          total_dx += float(dc)
          total_dy += float(-dr)

      if total_dx == 0.0 and total_dy == 0.0:
        continue

      # atan2(x, y) gives clockwise angle from North (y-axis)
      azimuth_rad = math.atan2(total_dx, total_dy)
      azimuth_deg = math.degrees(azimuth_rad) % 360.0

      # Apply floor_plan_orientation offset
      azimuth_array[r, c] = (azimuth_deg + floor_plan_orientation) % 360.0

  return azimuth_array
