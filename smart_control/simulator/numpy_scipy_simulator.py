"""NumPy coefficient assembly with a SciPy sparse direct solve."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from smart_control.simulator import simulator_flexible_floor_plan


class NumpyScipySimulator(
    simulator_flexible_floor_plan.SimulatorFlexibleGeometries
):
  """Solves each linearized heat balance with NumPy and SciPy.

  Radiation and interior mass still require the inherited outer fixed-point
  iteration. Within one outer iteration, however, this class assembles the
  complete air-temperature system in NumPy and solves it with SciPy `spsolve`.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._solves_entire_linear_system = True

  def update_temperature_estimates(
      self,
      temperature_estimates: np.ndarray,
      ambient_temperature: float,
      convection_coefficient: float,
  ) -> tuple[np.ndarray, float]:
    """Assembles and directly solves the current linearized heat balance."""
    building = self.building
    rows_count, cols_count = building.temp.shape
    cv_count = rows_count * cols_count
    flat_index = np.arange(cv_count).reshape(rows_count, cols_count)
    exterior = np.fromiter(
        (
            len(building.neighbors[row][column]) <= 1
            for row in range(rows_count)
            for column in range(cols_count)
        ),
        dtype=bool,
        count=cv_count,
    ).reshape(rows_count, cols_count)

    radiation_coefficient = None
    radiation_driving = None
    radiation_is_wall = None
    if getattr(building, 'include_radiative_heat_transfer', False):
      (
          radiation_coefficient,
          radiation_driving,
          radiation_is_wall,
      ) = building.get_longwave_interior_radiative_exchange_terms(
          temperature_estimates
      )

    matrix_rows = []
    matrix_columns = []
    matrix_values = []
    rhs = np.zeros(cv_count, dtype=np.float64)
    floor_height = building.floor_height_cm / 100.0
    radiation_area = building.cv_size_cm / 100.0 * floor_height
    exterior_terms = (
        self._exterior_forcing.linearized_air_terms(temperature_estimates)
        if self._exterior_forcing is not None
        else None
    )

    for row in range(rows_count):
      for column in range(cols_count):
        index = flat_index[row, column]
        if exterior[row, column]:
          matrix_rows.append(index)
          matrix_columns.append(index)
          matrix_values.append(1.0)
          rhs[index] = ambient_temperature
          continue

        u, v = self._get_cv_dimensions((row, column))
        storage = (
            building.density[row, column]
            * building.heat_capacity[row, column]
            * u
            * v
            * floor_height
            / self._time_step_sec
        )
        diagonal = storage
        rhs_value = (
            storage * building.temp[row, column] + building.input_q[row, column]
        )
        if exterior_terms is not None:
          exterior_coefficient, exterior_driving = exterior_terms
          diagonal += exterior_coefficient[row, column]
          rhs_value += exterior_driving[row, column]

        for neighbor_row, neighbor_column in building.neighbors[row][column]:
          conductance = self._get_face_conductance(
              (row, column), (neighbor_row, neighbor_column)
          )
          diagonal += conductance
          if exterior[neighbor_row, neighbor_column]:
            rhs_value += conductance * ambient_temperature
          else:
            matrix_rows.append(index)
            matrix_columns.append(flat_index[neighbor_row, neighbor_column])
            matrix_values.append(-conductance)

        ambient_conductance = self._get_ambient_conductance(
            (row, column), convection_coefficient
        )
        diagonal += ambient_conductance
        rhs_value += ambient_conductance * ambient_temperature

        if (
            getattr(building, 'include_interior_mass', False)
            and building.interior_mass_mask[row, column]
        ):
          mass_conductance = (
              building.interior_mass_conductivity[row, column]
              * u
              * v
              / floor_height
          )
          diagonal += mass_conductance
          rhs_value += (
              mass_conductance * building.interior_mass_temp[row, column]
          )

        if radiation_coefficient is not None:
          radiation_index = building.lwx_index[row, column]
          if radiation_index >= 0:
            coefficient = radiation_coefficient[radiation_index]
            driving = radiation_driving[radiation_index]
            if radiation_is_wall[radiation_index]:
              diagonal += coefficient * radiation_area
              rhs_value += driving * radiation_area
            else:
              mass_temperature = building.interior_mass_temp[row, column]
              rhs_value += (
                  driving - coefficient * mass_temperature
              ) * radiation_area

        matrix_rows.append(index)
        matrix_columns.append(index)
        matrix_values.append(diagonal)
        rhs[index] = rhs_value

    system = sparse.csr_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(cv_count, cv_count),
    )
    solution = sparse_linalg.spsolve(system, rhs).reshape(
        rows_count, cols_count
    )
    max_delta = float(np.max(np.abs(solution - temperature_estimates)))
    return solution, max_delta
