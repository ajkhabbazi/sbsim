r"""Tensorflow-enabled Finite Difference calculator.

Iterative methods that loop through each control volume sequentially
are very slow with geometries that have many control volumes (CV). The
TFSimulator, instead, processes the finite differences as a set of
tensor operations.

TFSimulator solves the same finite volume energy balance as the iterative
simulator in simulator.py, only written over the whole grid at once. Where
that module evaluates one CV at a time and reaches its neighbors through a
neighbor list, this module holds one tensor per quantity and reaches the
neighbors by shifting the whole field with shift_tensor_left and friends. The
two are expected to agree to within float32 round-off; simulator.py holds the
reference derivation and reference/sbsim_equation.pdf holds the spec both
follow.

Conductance model
-----------------
An interior face is shared by two CVs, so the heat crossing it passes through
both half cells in series,

$$A_f \Gamma_f = A_f \left[\frac{\ell_{P,f}}{2k_P} +
    \frac{\ell_{N_f,f}}{2k_{N_f}}\right]^{-1},$$

and a face that opens onto the ambient puts the convective film in series with
the outer half cell instead,

$$A_f \Gamma_{\infty,f} = A_f \left[\frac{1}{h} +
    \frac{\ell_{P,f}}{2k_P}\right]^{-1}.$$

Taking the owner CV's own conductivity over its own full width, $k_P /
\ell_{P,f}$, would let the two CVs sharing a face disagree about the heat
crossing it wherever the materials differ. The shared area is the narrower of
the two CVs, $A_f = \min(w_P, w_{N_f})\,z$, which matters where a half-width
boundary CV meets a full-width neighbor.

Tensor helpers
--------------
- get_cv_dimension_tensors: the per-CV extents $u$ and $v$, halved on the axis
  along which a boundary CV is truncated.
- get_half_cell_resistance_tensors: $\ell_{P,f} / (2 k_P)$ toward each face.
- get_oriented_face_conductance_tensors: $\Gamma_f$ on interior faces, zero on
  faces that open onto the ambient.
- get_oriented_ambient_conductance_tensors: $\Gamma_{\infty,f}$ on the faces
  that open onto the ambient, zero elsewhere.
- get_oriented_conductivity_tensors: the per-face conductivity, zeroed exactly
  on the faces that open onto the ambient, which is how the two conductance
  builders tell the two kinds of face apart.

Axis convention
---------------
$u$ is the extent of a CV along the column (second index, j) direction, so it
is the half-cell length toward the left and right faces, and $v$ is the extent
along the row (first index, i) direction and the half-cell length toward the
top and bottom faces. reference/sbsim_equation.pdf uses the opposite
assignment. On the square grid this code builds, $u$ and $v$ only ever differ
between a boundary CV and an interior one, never between the two axes of the
same interior CV, so the two conventions give identical numbers.
"""

import enum
import functools
from typing import Mapping, Optional, Sequence

from absl import logging
import gin
import numpy as np
import pandas as pd
import tensorflow as tf

from smart_control.simulator import building as building_py
from smart_control.simulator import building_radiation_utils
from smart_control.simulator import constants
from smart_control.simulator import hvac_floorplan_based as hvac_py
from smart_control.simulator import simulator_flexible_floor_plan as simulator
from smart_control.simulator import weather_controller as weather_controller_py

# List of neighbors for a specific CV.
NeighborCoordinates = Sequence[simulator.CVCoordinates]
# Neighbors, indexable by a specific i,j coord.
Neighbors = Sequence[Sequence[NeighborCoordinates]]


# All CVs must be interior, exterior, or boundary positions.
class CVPositionType(enum.Enum):
  INTERIOR = 1
  EXTERIOR = 2
  BOUNDARY = 3


# All Boundary CVs must be either edge (three non-exterior neighbors), or
# corner (two non-exterior neighbors).
class CVBoundaryType(enum.Enum):
  EDGE = 1
  CORNER = 2


# All edge CVs must be top, bottom, left or right edges.
class CVEdgeOrientationType(enum.Enum):
  """Classifies edge CVs by orientation."""

  TOP = (1, 0.5, 1.0)
  BOTTOM = (2, 0.5, 1.0)
  LEFT = (3, 1.0, 0.5)
  RIGHT = (4, 1.0, 0.5)

  def __init__(
      self,
      value: int,
      vertical_scale_factor: float,
      horizontal_scale_factor: float,
  ):
    self._value_ = value
    self.vertical_scale_factor = vertical_scale_factor
    self.horizontal_scale_factor = horizontal_scale_factor


# All corner CVs must be top left, bottom right, bottom left, or top right.
class CVCornerOrientationType(enum.Enum):
  TOP_LEFT = (1, 0.5)
  BOTTOM_LEFT = (2, 0.5)
  BOTTOM_RIGHT = (3, 0.5)
  TOP_RIGHT = (4, 0.5)

  def __init__(self, value: int, scale_factor: float):
    self._value_ = value
    self.scale_factor = scale_factor


class CVType:
  """Classifies a CV based on its neighborhood and orientation.

  A CV can be interior with four neighbors, exterior with zero or one
  neighbor, or boundary with two or three neighbors. A boundary CV
  can be an edge with three neighbors, or a corner with two neighbors.

  Using a regular class instead of a data class to ensure consistent
  assignment of CV properties.
  """

  def __init__(
      self,
      position: CVPositionType,
      boundary: Optional[CVBoundaryType] = None,
      edge: Optional[CVEdgeOrientationType] = None,
      corner: Optional[CVCornerOrientationType] = None,
  ):
    self.position = position
    self.boundary = boundary
    self.edge = edge
    self.corner = corner

    if self.position in [CVPositionType.EXTERIOR, CVPositionType.INTERIOR]:
      if (
          self.boundary is not None
          or self.edge is not None
          or self.corner is not None
      ):
        raise ValueError(
            'Cannot assign boundary properties to a non-boundary CV.'
        )
    elif self.position == CVPositionType.BOUNDARY:
      if self.edge is None and self.corner is None:
        raise ValueError('Boundary CVs must have edge or corner properties')

      if self.edge is not None and self.corner is not None:
        raise ValueError('Boundary CVs cannot be edge AND corner.')

      if self.boundary == CVBoundaryType.EDGE and self.edge is None:
        raise ValueError('Edge CVs must be assigned an edge orientation.')

      if self.boundary == CVBoundaryType.CORNER and self.corner is None:
        raise ValueError('Corner CVs must be assigned a corner orientation.')

  def __eq__(self, other) -> bool:

    if isinstance(other, CVType):
      return (
          self.position == other.position
          and self.boundary == other.boundary
          and self.edge == other.edge
          and self.corner == other.corner
      )
    return False

  @property
  def horizontal_scale_factor(self) -> float:
    """Returns the fraction of a full CV width this CV spans along j.

    A boundary CV is truncated by the ambient on the axis it is missing a
    neighbor along, so it spans half a full CV there. An edge CV facing left
    or right is truncated along j and a corner CV is truncated along both
    axes; every other CV spans a full width.

    Returns:
      1.0 or 0.5.
    """
    if self.position != CVPositionType.BOUNDARY:
      return 1.0
    match self.boundary:
      case CVBoundaryType.CORNER:
        if self.corner is None:
          raise ValueError('Corner CVs must be assigned a corner orientation.')
        return self.corner.scale_factor
      case CVBoundaryType.EDGE:
        if self.edge is None:
          raise ValueError('Edge CVs must be assigned an edge orientation.')
        return self.edge.horizontal_scale_factor
    raise ValueError('horizontal_scale_factor is not defined for this CV type.')

  @property
  def vertical_scale_factor(self) -> float:
    """Returns the fraction of a full CV width this CV spans along i.

    The counterpart of horizontal_scale_factor for the row direction: an edge
    CV facing top or bottom spans half a full width, a corner CV spans half
    along both axes, everything else spans a full width.

    Returns:
      1.0 or 0.5.
    """
    if self.position != CVPositionType.BOUNDARY:
      return 1.0
    match self.boundary:
      case CVBoundaryType.CORNER:
        if self.corner is None:
          raise ValueError('Corner CVs must be assigned a corner orientation.')
        return self.corner.scale_factor
      case CVBoundaryType.EDGE:
        if self.edge is None:
          raise ValueError('Edge CVs must be assigned an edge orientation.')
        return self.edge.vertical_scale_factor
    raise ValueError('vertical_scale_factor is not defined for this CV type.')


BoundaryCVMapping = Mapping[simulator.CVCoordinates, CVType]


def classify_cv(
    coords: simulator.CVCoordinates, neighbors: Neighbors
) -> CVType:
  """Classifies a CV based on its neighborhood.

  0-1 neighbors: EXTERIOR
  2 neighbors: CORNER
  3 neighbors: EDGE
  4 neighbors: INTERIOR

  A CV with two or three neighbors is also given the orientation that says
  which of its faces open onto the ambient, since that is what decides which
  axis it is truncated along and which faces carry a convective film.

  Args:
    coords: i, j coords of the CV
    neighbors:  a lookup table of all the neighbors.

  Returns:
    The CVType of the CV at coords.

  Raises:
    ValueError: if the CV has more than four neighbors, or if its two or three
      neighbors do not sit in one of the corner or edge arrangements. Two
      neighbors facing each other describe a one-CV-wide bridge or strip, which
      the current half-cell boundary geometry does not represent and rejects.
  """

  def _cv_type_corner_factory(
      corner: CVCornerOrientationType,
  ) -> CVType:

    return CVType(
        position=CVPositionType.BOUNDARY,
        boundary=CVBoundaryType.CORNER,
        corner=corner,
    )

  def _cv_corner_type(cv_neighbors: Sequence[tuple[int, int]]) -> CVType:
    neighbor_set = set(cv_neighbors)
    opposite_pairs = (
        {(i, j - 1), (i, j + 1)},
        {(i - 1, j), (i + 1, j)},
    )
    if neighbor_set in opposite_pairs:
      raise ValueError(
          'Unsupported two-neighbor CV geometry at '
          f'{(i, j)}: neighbors {sorted(neighbor_set)} are opposite. '
          'TFSimulator requires a two-neighbor boundary CV to form an '
          'adjacent corner; one-CV-wide bridges or strips are unsupported.'
      )
    if {(i + 1, j), (i, j + 1)} == neighbor_set:
      return _cv_type_corner_factory(corner=CVCornerOrientationType.TOP_LEFT)
    if {(i - 1, j), (i, j + 1)} == neighbor_set:
      return _cv_type_corner_factory(corner=CVCornerOrientationType.BOTTOM_LEFT)
    if {(i + 1, j), (i, j - 1)} == neighbor_set:
      return _cv_type_corner_factory(corner=CVCornerOrientationType.TOP_RIGHT)
    if {(i - 1, j), (i, j - 1)} == neighbor_set:
      return _cv_type_corner_factory(
          corner=CVCornerOrientationType.BOTTOM_RIGHT
      )
    raise ValueError(
        f'Invalid two-neighbor CV geometry at {(i, j)}: '
        f'neighbors={sorted(neighbor_set)}.'
    )

  def _cv_type_edge_factory(
      edge: CVEdgeOrientationType,
  ) -> CVType:

    return CVType(
        position=CVPositionType.BOUNDARY,
        boundary=CVBoundaryType.EDGE,
        edge=edge,
    )

  def _cv_edge_type(cv_neighbors: Sequence[tuple[int, int]]) -> CVType:
    edge = functools.partial(_cv_type_edge_factory)
    if set([(i, j + 1), (i, j - 1), (i + 1, j)]) == set(cv_neighbors):
      return edge(CVEdgeOrientationType.TOP)
    if set([(i, j - 1), (i, j + 1), (i - 1, j)]) == set(cv_neighbors):
      return edge(CVEdgeOrientationType.BOTTOM)
    if set([(i - 1, j), (i, j + 1), (i + 1, j)]) == set(cv_neighbors):
      return edge(CVEdgeOrientationType.LEFT)
    if set([(i - 1, j), (i, j - 1), (i + 1, j)]) == set(cv_neighbors):
      return edge(CVEdgeOrientationType.RIGHT)
    raise ValueError(f"Wasn't able to determine which edge the CV {(i, j)} is.")

  i, j = coords

  cv_neighbors = neighbors[i][j]
  match len(cv_neighbors):
    case 0:
      return CVType(position=CVPositionType.EXTERIOR)
    case 1:
      return CVType(position=CVPositionType.EXTERIOR)
    case 2:
      return _cv_corner_type(cv_neighbors)
    case 3:
      return _cv_edge_type(cv_neighbors)
    case 4:
      return CVType(position=CVPositionType.INTERIOR)
    case _:
      raise ValueError(
          f"Wasn't able to determine which CV type the CV {(i, j)} is."
      )


def get_cv_mapping(
    neighbors: Neighbors,
    position_criterion: CVPositionType = CVPositionType.BOUNDARY,
) -> BoundaryCVMapping:
  """Gets a map of all the non-interior cv_types."""

  boundary_cv_mapping = {}
  ni = len(neighbors)
  nj = len(neighbors[0])
  for i in range(ni):
    for j in range(nj):
      cv_type = classify_cv((i, j), neighbors)
      if cv_type.position == position_criterion:
        boundary_cv_mapping[(i, j)] = cv_type

  return boundary_cv_mapping


def get_cv_dimension_tensors(
    control_volume_size: float,
    boundary_cv_mapping: BoundaryCVMapping,
    shape: tuple[int, int],
) -> tuple[tf.Tensor, tf.Tensor]:
  r"""Returns horizontal and vertical CV dimension tensors.

  Boundary control volumes have varying heights and widths, since they share
  a part with the outside. For example, edge boundary CVs have one side
  that is half width, depending on orientation, and corner CVs have both edges
  half width.

  $u$ is the extent along the column (j) direction and so the half-cell length
  toward the left and right faces, and $v$ is the extent along the row (i)
  direction and the half-cell length toward the top and bottom faces. Both are
  returned in whatever length unit control_volume_size is given in; TFSimulator
  passes meters, which is what makes the conductances come out in W/K.

  Args:
    control_volume_size: width of each interior control volume, in the length
      unit the caller wants the dimensions back in. Despite the historical name
      of the gin parameter it is read from, TFSimulator passes meters here, not
      centimeters.
    boundary_cv_mapping: dict of all boundary elements.
    shape: shape of the target matrix.

  Returns:
    A pair of tensors of the given shape, the horizontal ($u$) and vertical
    ($v$) extent of every CV.
  """

  def _compute_cv_dimension(cv_type: CVType) -> tuple[float, float]:
    """Sets the horizontal and vertical CV dimension for one CV."""

    return (
        control_volume_size * cv_type.horizontal_scale_factor,
        control_volume_size * cv_type.vertical_scale_factor,
    )

  horizontal_cv_dimension = np.full(
      shape, control_volume_size, dtype=np.float32
  )
  vertical_cv_dimension = np.full(shape, control_volume_size, dtype=np.float32)

  for (i, j), cv_type in boundary_cv_mapping.items():

    cv_horizontal_cv_dimension, cv_vertical_cv_dimension = (
        _compute_cv_dimension(cv_type)
    )
    vertical_cv_dimension[i][j] = cv_vertical_cv_dimension
    horizontal_cv_dimension[i][j] = cv_horizontal_cv_dimension

  t_horizontal_cv_dimension = tf.convert_to_tensor(
      horizontal_cv_dimension, dtype=tf.float32
  )
  t_vertical_cv_dimension = tf.convert_to_tensor(
      vertical_cv_dimension, dtype=tf.float32
  )
  return t_horizontal_cv_dimension, t_vertical_cv_dimension


def get_oriented_convection_coefficient_tensors(
    convection_coefficient_air: float,
    shape: tuple[int, int],
    boundary_cv_mapping: BoundaryCVMapping,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Returns oriented convection coefficient tensors.

  Forced convection (wind) is the primary means of exchanging heat
  across the boundary CVs.

  To accommodate tensor operations, we have to create edge-oriented tensors
  for convection. Each tensor carries the film coefficient only on the CVs
  whose face in that direction opens onto the ambient, and zero everywhere
  else, so that an interior face and an exterior face can be told apart by
  looking at one entry.

  Args:
    convection_coefficient_air: Outside air convection coefficient $h$
      [W/m^2/K].
    shape: Tensor shape - should be identical to the other tensors, i.e., temp
    boundary_cv_mapping: dict of boundary CVs.

  Returns:
    Four tensors of the given shape holding $h$ on the left, right, top and
    bottom faces respectively, in that order, and zero on every face that does
    not open onto the ambient.
  """

  def _set_edge_convection_coefficient(cv_type: CVType) -> None:
    """Sets the convection coefficient for one CV."""
    match cv_type.edge:
      case CVEdgeOrientationType.TOP:
        h_top_edge[i][j] = convection_coefficient_air
      case CVEdgeOrientationType.LEFT:
        h_left_edge[i][j] = convection_coefficient_air
      case CVEdgeOrientationType.RIGHT:
        h_right_edge[i][j] = convection_coefficient_air
      case CVEdgeOrientationType.BOTTOM:
        h_bottom_edge[i][j] = convection_coefficient_air

  def _set_corner_convection_coefficient(cv_type: CVType) -> None:
    match cv_type.corner:
      case CVCornerOrientationType.TOP_LEFT:
        h_top_edge[i][j] = convection_coefficient_air
        h_left_edge[i][j] = convection_coefficient_air
      case CVCornerOrientationType.TOP_RIGHT:
        h_top_edge[i][j] = convection_coefficient_air
        h_right_edge[i][j] = convection_coefficient_air
      case CVCornerOrientationType.BOTTOM_LEFT:
        h_bottom_edge[i][j] = convection_coefficient_air
        h_left_edge[i][j] = convection_coefficient_air
      case CVCornerOrientationType.BOTTOM_RIGHT:
        h_bottom_edge[i][j] = convection_coefficient_air
        h_right_edge[i][j] = convection_coefficient_air

  h_left_edge = np.full(shape, 0, dtype=np.float32)
  h_right_edge = np.full(shape, 0, dtype=np.float32)
  h_top_edge = np.full(shape, 0, dtype=np.float32)
  h_bottom_edge = np.full(shape, 0, dtype=np.float32)

  for (i, j), cv_type in boundary_cv_mapping.items():
    if cv_type.position == CVPositionType.BOUNDARY:
      if cv_type.boundary == CVBoundaryType.EDGE:
        _set_edge_convection_coefficient(cv_type)

      if cv_type.boundary == CVBoundaryType.CORNER:
        _set_corner_convection_coefficient(cv_type)

  t_h_left_edge = tf.convert_to_tensor(h_left_edge, dtype=tf.float32)
  t_h_right_edge = tf.convert_to_tensor(h_right_edge, dtype=tf.float32)
  t_h_top_edge = tf.convert_to_tensor(h_top_edge, dtype=tf.float32)
  t_h_bottom_edge = tf.convert_to_tensor(h_bottom_edge, dtype=tf.float32)
  return t_h_left_edge, t_h_right_edge, t_h_top_edge, t_h_bottom_edge


def get_oriented_conductivity_tensors(
    conductivity: np.ndarray,
    boundary_cv_mapping: BoundaryCVMapping,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  r"""Returns oriented conductivity tensors.

  Conduction is a function of two adjacent CVs, so it only happens across a
  face that has a CV on both sides. The conductivity is split by sides of the
  CV, top, bottom, left and right, and zeroed on the individual faces that open
  onto the ambient. This is per face, not per CV: an edge CV facing left keeps
  its own conductivity on its right, top and bottom faces and only loses it on
  the left one, and a corner CV loses it on exactly two.

  The zeros are what the two conductance builders use to tell the two kinds of
  face apart. get_oriented_face_conductance_tensors puts a series conductance
  only where these are nonzero, and get_oriented_ambient_conductance_tensors
  puts a film in series with the outer half cell only where they are zero, so
  every face is covered exactly once.

  Args:
    conductivity: conductivity of the CV itself, $\mathrm{W/(m \cdot K)}$.
    boundary_cv_mapping: mapping that contains the coordinates of boundary CVs

  Returns:
    Four tensors shaped like conductivity, the left, right, top and bottom face
    conductivities in that order, zero on the faces that open onto the ambient.
  """

  k_left_edge = conductivity.copy()
  k_right_edge = conductivity.copy()
  k_top_edge = conductivity.copy()
  k_bottom_edge = conductivity.copy()

  for (i, j), cv_type in boundary_cv_mapping.items():
    if cv_type.position == CVPositionType.BOUNDARY:
      if cv_type.boundary == CVBoundaryType.EDGE:
        if cv_type.edge == CVEdgeOrientationType.TOP:
          k_top_edge[i][j] = 0.0
        if cv_type.edge == CVEdgeOrientationType.LEFT:
          k_left_edge[i][j] = 0.0
        if cv_type.edge == CVEdgeOrientationType.RIGHT:
          k_right_edge[i][j] = 0.0
        if cv_type.edge == CVEdgeOrientationType.BOTTOM:
          k_bottom_edge[i][j] = 0.0

      if cv_type.boundary == CVBoundaryType.CORNER:
        if cv_type.corner == CVCornerOrientationType.TOP_LEFT:
          k_top_edge[i][j] = 0.0
          k_left_edge[i][j] = 0.0
        if cv_type.corner == CVCornerOrientationType.TOP_RIGHT:
          k_top_edge[i][j] = 0.0
          k_right_edge[i][j] = 0.0
        if cv_type.corner == CVCornerOrientationType.BOTTOM_LEFT:
          k_bottom_edge[i][j] = 0.0
          k_left_edge[i][j] = 0.0
        if cv_type.corner == CVCornerOrientationType.BOTTOM_RIGHT:
          k_bottom_edge[i][j] = 0.0
          k_right_edge[i][j] = 0.0

  t_k_left_edge = tf.convert_to_tensor(k_left_edge, dtype=tf.float32)
  t_k_right_edge = tf.convert_to_tensor(k_right_edge, dtype=tf.float32)
  t_k_top_edge = tf.convert_to_tensor(k_top_edge, dtype=tf.float32)
  t_k_bottom_edge = tf.convert_to_tensor(k_bottom_edge, dtype=tf.float32)
  return t_k_left_edge, t_k_right_edge, t_k_top_edge, t_k_bottom_edge


def get_half_cell_resistance_tensors(
    conductivity: np.ndarray,
    t_u: tf.Tensor,
    t_v: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
  r"""Returns the half-cell conduction resistances of every CV.

  Half of a CV lies between its node and each of its faces, so the conduction
  resistance per unit area from the node to a face normal to u is

  $$\frac{u_P}{2k_P},$$

  and to a face normal to v it is $v_P / (2k_P)$.

  Args:
    conductivity: conductivity of each CV, $\mathrm{W/(m \cdot K)}$.
    t_u: horizontal CV dimension tensor, $\mathrm{m}$.
    t_v: vertical CV dimension tensor, $\mathrm{m}$.

  Returns:
    Half-cell resistance tensors toward the u-normal and the v-normal faces,
    $\mathrm{m^2 \cdot K/W}$.
  """
  t_conductivity = tf.convert_to_tensor(conductivity, dtype=tf.float32)
  t_two_k = tf.scalar_mul(tf.constant(2.0, dtype=tf.float32), t_conductivity)
  return tf.math.divide(t_u, t_two_k), tf.math.divide(t_v, t_two_k)


def get_oriented_face_conductance_tensors(
    conductivity: np.ndarray,
    t_u: tf.Tensor,
    t_v: tf.Tensor,
    boundary_cv_mapping: BoundaryCVMapping,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  r"""Returns the interior face conductances per unit area.

  An interior face is shared by two CVs, so the heat crossing it has to pass
  through both half cells in series:

  $$\Gamma_f = \left[\frac{\ell_{P,f}}{2k_P} +
      \frac{\ell_{N_f,f}}{2k_{N_f}}\right]^{-1}$$

  with $\ell_{P,1} = \ell_{P,3} = u_P$ and $\ell_{P,2} = \ell_{P,4} = v_P$.

  Taking the owner CV's own conductivity across its own full width instead,
  i.e. $k_P / \ell_{P,f}$, makes the two CVs sharing a face disagree about the
  heat crossing it, so the balance is not conservative wherever materials
  differ - inside air next to an exterior wall is off by the ratio of their
  conductivities. For a uniform material on a uniform grid both forms agree.

  The balance multiplies these by the owner CV's own face area, so where a
  half-width boundary CV meets a full-width neighbor the two would claim
  different areas of contact. The conductance returned here is scaled by the
  ratio of the shared width to the owner's width so that the product is the
  same seen from either side.

  Faces that open onto the ambient are excluded here: they carry a convective
  film and are handled by get_oriented_ambient_conductance_tensors.

  Args:
    conductivity: conductivity of each CV, $\mathrm{W/(m \cdot K)}$.
    t_u: horizontal CV dimension tensor, $\mathrm{m}$.
    t_v: vertical CV dimension tensor, $\mathrm{m}$.
    boundary_cv_mapping: dict of boundary CVs.

  Returns:
    Face conductance tensors for the left, right, top and bottom faces,
    $\mathrm{W/(m^2 \cdot K)}$, zero on faces that open onto the ambient.
  """

  (
      t_k_left_edge,
      t_k_right_edge,
      t_k_top_edge,
      t_k_bottom_edge,
  ) = get_oriented_conductivity_tensors(conductivity, boundary_cv_mapping)

  t_r_u, t_r_v = get_half_cell_resistance_tensors(conductivity, t_u, t_v)

  def _series_conductance(
      t_r_own: tf.Tensor,
      t_r_neighbor: tf.Tensor,
      t_w_own: tf.Tensor,
      t_w_neighbor: tf.Tensor,
      t_k_edge: tf.Tensor,
  ) -> tf.Tensor:
    """Inverts the two half-cell resistances in series on the active faces."""
    t_conductance = tf.math.divide(
        tf.constant(1.0, dtype=tf.float32),
        tf.math.add(t_r_own, t_r_neighbor),
    )
    # The balance multiplies this by the owner's own face area, but a boundary
    # CV is only half as wide as a full CV across the face, so the two CVs
    # sharing the face would otherwise disagree about the area of contact. The
    # shared area is the narrower of the two, so scale the owner's conductance
    # by that ratio to make the exchange conservative.
    t_conductance = tf.math.multiply(
        t_conductance,
        tf.math.divide(tf.math.minimum(t_w_own, t_w_neighbor), t_w_own),
    )
    # t_k_edge is zero exactly on the faces that open onto the ambient, which
    # is also where the shifted neighbor resistance is padding rather than a
    # real neighbor.
    return tf.where(
        tf.math.not_equal(t_k_edge, 0.0),
        t_conductance,
        tf.zeros_like(t_conductance),
    )

  return (
      _series_conductance(
          t_r_u,
          shift_tensor_right(t_r_u),
          t_v,
          shift_tensor_right(t_v),
          t_k_left_edge,
      ),
      _series_conductance(
          t_r_u,
          shift_tensor_left(t_r_u),
          t_v,
          shift_tensor_left(t_v),
          t_k_right_edge,
      ),
      _series_conductance(
          t_r_v,
          shift_tensor_down(t_r_v),
          t_u,
          shift_tensor_down(t_u),
          t_k_top_edge,
      ),
      _series_conductance(
          t_r_v,
          shift_tensor_up(t_r_v),
          t_u,
          shift_tensor_up(t_u),
          t_k_bottom_edge,
      ),
  )


def get_oriented_ambient_conductance_tensors(
    convection_coefficient_air: float,
    conductivity: np.ndarray,
    t_u: tf.Tensor,
    t_v: tf.Tensor,
    boundary_cv_mapping: BoundaryCVMapping,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  r"""Returns the exterior boundary face conductances per unit area.

  On a face that opens onto the ambient the heat passes through the outer half
  of the boundary CV and then through the convective film, in series:

  $$\Gamma_{\infty,f} = \left[\frac{1}{h_{ext}} +
      \frac{\ell_{P,f}}{2k_P}\right]^{-1}$$

  Using $h_{ext}$ on its own drops the conduction resistance of that outer
  half cell, which overstates the exchange with the ambient - badly so for an
  insulating exterior wall, where the half-cell resistance dominates the film.

  $h_{ext}$ is used only on exterior boundary faces; interior faces are pure
  conduction through neighboring half cells.

  Args:
    convection_coefficient_air: outside air convection coefficient,
      $\mathrm{W/(m^2 \cdot K)}$.
    conductivity: conductivity of each CV, $\mathrm{W/(m \cdot K)}$.
    t_u: horizontal CV dimension tensor, $\mathrm{m}$.
    t_v: vertical CV dimension tensor, $\mathrm{m}$.
    boundary_cv_mapping: dict of boundary CVs.

  Returns:
    Ambient conductance tensors for the left, right, top and bottom faces,
    $\mathrm{W/(m^2 \cdot K)}$, zero on faces that do not touch the ambient.
  """

  (
      t_h_left_edge,
      t_h_right_edge,
      t_h_top_edge,
      t_h_bottom_edge,
  ) = get_oriented_convection_coefficient_tensors(
      convection_coefficient_air, conductivity.shape, boundary_cv_mapping
  )

  t_r_u, t_r_v = get_half_cell_resistance_tensors(conductivity, t_u, t_v)

  def _film_and_half_cell(t_h_edge: tf.Tensor, t_r_own: tf.Tensor) -> tf.Tensor:
    """Inverts the film and half-cell resistances in series on active faces."""
    t_one = tf.constant(1.0, dtype=tf.float32)
    t_active = tf.math.greater(t_h_edge, 0.0)
    # Keep the reciprocal finite on the inactive faces, which are masked out
    # right after.
    t_h_safe = tf.where(t_active, t_h_edge, tf.ones_like(t_h_edge))
    t_conductance = tf.math.divide(
        t_one, tf.math.add(tf.math.divide(t_one, t_h_safe), t_r_own)
    )
    return tf.where(t_active, t_conductance, tf.zeros_like(t_conductance))

  return (
      _film_and_half_cell(t_h_left_edge, t_r_u),
      _film_and_half_cell(t_h_right_edge, t_r_u),
      _film_and_half_cell(t_h_top_edge, t_r_v),
      _film_and_half_cell(t_h_bottom_edge, t_r_v),
  )


def shift_tensor_right(
    x: tf.Tensor,
    padding_value: float = 0,
) -> tf.Tensor:
  """Shifts a field one column to the right, so entry (i, j) holds (i, j - 1).

  Reading a neighbor's value off a shifted field is how this module replaces
  the neighbor lookups the iterative simulator does one CV at a time. The
  column that shifts in past the left edge of the grid takes padding_value.

  Args:
    x: field to shift, shaped (rows, columns).
    padding_value: value for the column shifted in at the left edge.

  Returns:
    A tensor shaped like x holding each CV's left neighbor's value.
  """
  t_right = tf.pad(x, [[0, 0], [1, 0]], constant_values=padding_value)
  return tf.boolean_mask(t_right, [True] * x.shape[1] + [False], axis=1)


def shift_tensor_left(
    x: tf.Tensor,
    padding_value: float = 0,
) -> tf.Tensor:
  """Shifts a field one column to the left, so entry (i, j) holds (i, j + 1).

  Args:
    x: field to shift, shaped (rows, columns).
    padding_value: value for the column shifted in at the right edge.

  Returns:
    A tensor shaped like x holding each CV's right neighbor's value.
  """
  t_left = tf.pad(x, [[0, 0], [0, 1]], constant_values=padding_value)
  return tf.boolean_mask(t_left, [False] + [True] * x.shape[1], axis=1)


def shift_tensor_up(
    x: tf.Tensor,
    padding_value: float = 0,
) -> tf.Tensor:
  """Shifts a field one row up, so entry (i, j) holds (i + 1, j).

  Args:
    x: field to shift, shaped (rows, columns).
    padding_value: value for the row shifted in at the bottom edge.

  Returns:
    A tensor shaped like x holding each CV's bottom neighbor's value.
  """
  t_up = tf.pad(x, [[0, 1], [0, 0]], constant_values=padding_value)
  return tf.boolean_mask(t_up, [False] + [True] * x.shape[0], axis=0)


def shift_tensor_down(
    x: tf.Tensor,
    padding_value: float = 0,
) -> tf.Tensor:
  """Shifts a field one row down, so entry (i, j) holds (i - 1, j).

  Args:
    x: field to shift, shaped (rows, columns).
    padding_value: value for the row shifted in at the top edge.

  Returns:
    A tensor shaped like x holding each CV's top neighbor's value.
  """
  t_down = tf.pad(x, [[1, 0], [0, 0]], constant_values=padding_value)
  return tf.boolean_mask(t_down, [True] * x.shape[0] + [False], axis=0)


def apply_exterior_temps(
    t_temps: tf.Tensor, temp_inf: float, t_exterior_temps_mask: tf.Tensor
) -> tf.Tensor:
  """Applies the temp_inf to all CVs that are masked as exterior."""
  t_exterior_temp = tf.fill(
      t_temps.shape, tf.constant(temp_inf, dtype=tf.float32)
  )
  return tf.where(t_exterior_temps_mask, t_exterior_temp, t_temps)


@tf.function(reduce_retracing=True)
def _solve_with_preconditioned_conjugate_gradient(
    diagonal: tf.Tensor,
    rhs: tf.Tensor,
    coupling_left: tf.Tensor,
    coupling_right: tf.Tensor,
    coupling_top: tf.Tensor,
    coupling_bottom: tf.Tensor,
    exterior_mask: tf.Tensor,
    initial_temperature: tf.Tensor,
    tolerance: tf.Tensor,
    iteration_limit: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Solves the symmetric active-CV system with Jacobi-preconditioned CG."""
  dtype = diagonal.dtype
  active_mask = tf.math.logical_not(exterior_mask)

  def _matvec(vector: tf.Tensor) -> tf.Tensor:
    grid = tf.reshape(vector, tf.shape(diagonal))
    active_grid = tf.where(active_mask, grid, tf.zeros_like(grid))
    zero_column = tf.zeros_like(active_grid[:, :1])
    zero_row = tf.zeros_like(active_grid[:1, :])
    left_neighbor = tf.concat((zero_column, active_grid[:, :-1]), axis=1)
    right_neighbor = tf.concat((active_grid[:, 1:], zero_column), axis=1)
    top_neighbor = tf.concat((zero_row, active_grid[:-1, :]), axis=0)
    bottom_neighbor = tf.concat((active_grid[1:, :], zero_row), axis=0)
    result = diagonal * grid
    result -= coupling_left * left_neighbor
    result -= coupling_right * right_neighbor
    result -= coupling_top * top_neighbor
    result -= coupling_bottom * bottom_neighbor
    result = tf.where(exterior_mask, grid, result)
    return tf.reshape(result, [-1])

  flat_diagonal = tf.reshape(diagonal, [-1])
  flat_rhs = tf.reshape(rhs, [-1])
  solution = tf.reshape(initial_temperature, [-1])
  residual = flat_rhs - _matvec(solution)
  preconditioned_residual = residual / flat_diagonal
  direction = preconditioned_residual
  residual_dot = tf.tensordot(residual, preconditioned_residual, axes=1)
  rhs_norm = tf.linalg.norm(flat_rhs)
  target_norm = tolerance * tf.maximum(rhs_norm, tf.constant(1.0, dtype))
  iteration = tf.constant(0, dtype=tf.int32)

  def _condition(
      iteration,
      solution,
      residual,
      preconditioned_residual,
      direction,
      residual_dot,
  ):
    del solution, preconditioned_residual, direction, residual_dot
    return tf.math.logical_and(
        iteration < iteration_limit,
        tf.linalg.norm(residual) > target_norm,
    )

  def _body(
      iteration,
      solution,
      residual,
      preconditioned_residual,
      direction,
      residual_dot,
  ):
    del preconditioned_residual
    matrix_direction = _matvec(direction)
    step = residual_dot / tf.tensordot(direction, matrix_direction, axes=1)
    next_solution = solution + step * direction
    next_residual = residual - step * matrix_direction
    next_preconditioned_residual = next_residual / flat_diagonal
    next_residual_dot = tf.tensordot(
        next_residual, next_preconditioned_residual, axes=1
    )
    beta = next_residual_dot / residual_dot
    next_direction = next_preconditioned_residual + beta * direction
    return (
        iteration + 1,
        next_solution,
        next_residual,
        next_preconditioned_residual,
        next_direction,
        next_residual_dot,
    )

  (
      iteration,
      solution,
      residual,
      _,
      _,
      _,
  ) = tf.while_loop(
      _condition,
      _body,
      (
          iteration,
          solution,
          residual,
          preconditioned_residual,
          direction,
          residual_dot,
      ),
  )
  return (
      tf.reshape(solution, tf.shape(diagonal)),
      tf.linalg.norm(residual),
      iteration,
  )


@gin.configurable
class TFSimulator(simulator.SimulatorFlexibleGeometries):
  """Tensor simulator with Jacobi FDM and matrix-free PCG backends.

  Solves the same energy balance as its base class, but over the whole grid at
  once. Everything that depends only on the geometry and the materials, the CV
  dimensions, the half-cell resistances and the interior face conductances, is
  built once here in __init__ and reused every step; only the ambient
  conductances are rebuilt per step, since the convection coefficient moves
  with the weather.
  """

  def __init__(
      self,
      building: building_py.Building,
      hvac: hvac_py.FloorPlanBasedHvac,
      weather_controller: weather_controller_py.WeatherController,
      time_step_sec: float,
      convergence_threshold: float,
      iteration_limit: int,
      iteration_warning: int,
      start_timestamp: pd.Timestamp,
      relative_convergence_threshold: float | None = 1e-6,
      relative_convergence_streak: int = 20,
      solver_method: str = 'fdm',
      cg_tolerance: float = 1e-10,
      cg_iteration_limit: int = 2000,
  ):
    """Initializes the tensor heat-balance simulator.

    Args:
      solver_method: `fdm` for Jacobi relaxation or `cg` for a complete
        matrix-free preconditioned conjugate-gradient solve of each linearized
        air-temperature system.
      cg_tolerance: Relative residual tolerance for each CG inner solve.
      cg_iteration_limit: Maximum CG iterations per outer fixed-point
        iteration. Used only when solver_method is `cg`.
      Remaining arguments: Forwarded to SimulatorFlexibleGeometries.
    """

    if solver_method not in ('fdm', 'cg'):
      raise ValueError(
          f"solver_method must be 'fdm' or 'cg', got {solver_method!r}"
      )
    if cg_tolerance <= 0.0:
      raise ValueError('cg_tolerance must be positive')
    if cg_iteration_limit < 1:
      raise ValueError('cg_iteration_limit must be positive')

    super().__init__(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
        relative_convergence_threshold=relative_convergence_threshold,
        relative_convergence_streak=relative_convergence_streak,
    )

    # Get a mapping of all the boundary CVs that interface between interior
    # and extrior CVs. This mapping will be used to process boundary elements
    # iteratively.
    self._boundary_cv_mapping = get_cv_mapping(
        building.neighbors, position_criterion=CVPositionType.BOUNDARY
    )
    n_boundary_elements = len(self._boundary_cv_mapping)
    logging.info('Number of boundary CVs: %d', n_boundary_elements)
    # Get a binary mask that mark exterior CVs so that they will always be
    # assigned ambinent air temps.
    self._t_exerior_temps_mask = self._get_tensor_exterior_mask(building)
    n_exterior_elements = tf.math.count_nonzero(self._t_exerior_temps_mask)
    logging.info('Number of exterior CVs: %d', n_exterior_elements)

    n_elements = self.building.temp.shape[0] * self.building.temp.shape[1]
    n_interior_elements = n_elements - n_boundary_elements - n_exterior_elements
    logging.info('Number of interior CVs: %d', n_interior_elements)

    self._t_u, self._t_v = get_cv_dimension_tensors(
        self.building.cv_size_cm / 100.0,
        self._boundary_cv_mapping,
        self.building.temp.shape,
    )

    (
        self._t_conductivity_left_edge,
        self._t_conductivity_right_edge,
        self._t_conductivity_top_edge,
        self._t_conductivity_bottom_edge,
    ) = get_oriented_conductivity_tensors(
        self.building.conductivity, self._boundary_cv_mapping
    )

    # An interior face is shared by two CVs, so its conductance comes from the
    # two half-cell resistances in series rather than from the owner CV's
    # conductivity alone. These are geometry and material only, so they are
    # built once here; the ambient faces additionally depend on the wind film
    # coefficient and are rebuilt every step.
    (
        self._t_face_conductance_left,
        self._t_face_conductance_right,
        self._t_face_conductance_top,
        self._t_face_conductance_bottom,
    ) = get_oriented_face_conductance_tensors(
        self.building.conductivity,
        self._t_u,
        self._t_v,
        self._boundary_cv_mapping,
    )

    self.solver_method = solver_method
    self.cg_tolerance = cg_tolerance
    self.cg_iteration_limit = cg_iteration_limit
    self._solves_entire_linear_system = solver_method == 'cg'
    self.last_cg_iteration_count = 0
    self.last_cg_residual_norm = np.nan

    # radiative heat transfer addition
    self.include_radiative_heat_transfer = (
        building.include_radiative_heat_transfer
    )
    # self.interior_wall_mask = building.interior_wall_mask

    # interior mass addition
    self.include_interior_mass = building.include_interior_mass

    # Which nodes of the radiating enclosure are interior walls, in the order
    # _get_input_tensors stacks them. Wall nodes are the unknown their own CV
    # solves for, mass nodes are not, so _get_longwave_grid_terms treats the
    # two differently. The enclosure never changes shape, so this is built once.
    if self.include_radiative_heat_transfer:
      if self.include_interior_mass:
        enclosure_mask = (
            building.interior_wall_mask | building.interior_mass_mask
        )
        self._lwx_is_wall_node = building.interior_wall_mask[enclosure_mask]
      else:
        self._lwx_is_wall_node = np.ones(
            int(np.sum(building.interior_wall_mask)), dtype=bool
        )
      self._lwx_ifa_blocks = tuple(
          (
              tf.constant(indices, dtype=tf.int32),
              tf.constant(ifa_inv, dtype=tf.float32),
          )
          for indices, ifa_inv in zip(
              building.radiative_enclosure_indices,
              building.ifa_inv_by_enclosure,
          )
      )
    if self.include_interior_mass:
      self._initialize_interior_mass_tensors(building)

  def _get_tensor_exterior_mask(
      self, building: building_py.Building
  ) -> tf.Tensor:
    """Returns a binary tensor mask of all CVs that are exterior = True."""
    exterior_cv_mapping = get_cv_mapping(
        building.neighbors, position_criterion=CVPositionType.EXTERIOR
    )
    exterior_mask = np.full(building.temp.shape, False)
    for i, j in exterior_cv_mapping:
      exterior_mask[i][j] = True
    return tf.convert_to_tensor(exterior_mask)

  def _initialize_interior_mass_tensors(
      self, building: building_py.FloorPlanBasedBuilding
  ) -> None:
    """Initializes tensors for interior mass calculations."""
    # Convert interior mass properties to tensors
    self._t_interior_mass_mask = tf.convert_to_tensor(
        building.interior_mass_mask, dtype=tf.bool
    )
    self._t_interior_mass_conductivity = tf.convert_to_tensor(
        building.interior_mass_conductivity, dtype=tf.float32
    )
    self._t_interior_mass_heat_capacity = tf.convert_to_tensor(
        building.interior_mass_heat_capacity, dtype=tf.float32
    )
    self._t_interior_mass_density = tf.convert_to_tensor(
        building.interior_mass_density, dtype=tf.float32
    )

    # Calculate t_0_mass = (rho_mass * C_mass * z^2) / (k_mass * delta_t)
    # z is the floor height in meters (characteristic length for heat exchange)
    z = building.floor_height_cm / 100.0

    # Compute t_0_mass for each CV (will be 0 where there's no interior mass)
    self._t_0_mass = tf.where(
        self._t_interior_mass_mask,
        (
            self._t_interior_mass_density
            * self._t_interior_mass_heat_capacity
            * z
            * z
        )
        / (self._t_interior_mass_conductivity * self._time_step_sec),
        tf.constant(0.0, dtype=tf.float32),
    )

  def update_interior_mass_temperatures(
      self, air_temperature_estimates: np.ndarray
  ) -> tuple[np.ndarray, float]:
    r"""Tensorized version of interior mass temperature update.

    Overrides the parent class's iterative implementation with a tensor-based
    approach for efficiency. The heat exchange occurs through the vertical
    direction (height z) of the control volume.

    Equations:
    --------------------
    After the air CV temperatures converge, update the interior mass
    temperatures using element-wise tensor operations:

    $$T_{\text{mass}} =
      \frac{T+t_{0,\text{mass}}\odot T_{\text{mass}}^{(-)}}{1+t_{0,\text{mass}}}
    $$

    where $\odot$ represents element-wise multiplication and the division is
    element-wise (broadcasting the scalar denominator).

    The temporal parameter tensor is defined as:

    $$t_{0,\text{mass},i,j} = \begin{cases}
    \frac{\rho_{\text{mass}}c_{\text{mass}}z^2}{k_{\text{mass}}\Delta t}
    & \text{if CV has interior mass} \\
    0 & \text{otherwise}
    \end{cases}$$

    This formulation is consistent with the air CV energy balance where the
    interior mass coupling term is $\frac{k_{\text{mass}} u v}{z}
    (T_{\text{mass},i,j} - T_{i,j})$.

    Because that coupling term sits inside the air balance, air and mass form
    one system and the mass has to be re-solved on every Picard iteration
    rather than once per step. finite_differences_timestep does that by
    restoring $T_{\text{mass}}^{(-)}$ before each call here, so this always
    advances the mass one time step from the start of the step and never
    compounds partial iterates.

    Nomenclature and Units:
    -----------------------
    - $T$: Converged air temperature tensor at new time step [K]
    - $T_{\text{mass}}$: Interior mass temperature tensor at new time step [K]
    - $T_{\text{mass}}^{(-)}$: Interior mass temperature tensor at previous
      time step [K]
    - $t_{0,\text{mass}}$: Temporal parameter tensor for interior mass
      [dimensionless]
    - $k_{\text{mass}}$: Thermal conductivity of interior mass
      [$\mathrm{W/(m \cdot K)}$]
    - $\rho_{\text{mass}}$: Density of interior mass [$\mathrm{kg/m^3}$]
    - $c_{\text{mass}}$: Specific heat capacity of interior mass
      [$\mathrm{J/(kg \cdot K)}$]
    - $z$: CV height (floor height), characteristic length for heat exchange
      [$\mathrm{m}$]
    - $\Delta t$: Time step [$\mathrm{s}$]

    Args:
      air_temperature_estimates: Current air temperature estimates for each CV.

    Returns:
      Tuple of (updated interior mass temperatures, maximum temperature change)
    """
    if not self.include_interior_mass:
      return self.building.interior_mass_temp.copy(), 0.0

    # Convert air temperatures to tensor
    t_air_temp = tf.convert_to_tensor(
        air_temperature_estimates, dtype=tf.float32
    )

    # Convert previous interior mass temperature to tensor
    t_temp_mass_prev = tf.convert_to_tensor(
        self.building.interior_mass_temp, dtype=tf.float32
    )

    # Calculate numerator: T + t_0_mass * T_mass^(-)
    numerator = tf.math.add(
        t_air_temp, tf.math.multiply(self._t_0_mass, t_temp_mass_prev)
    )
    if self._exterior_forcing is not None:
      conductance = self._t_interior_mass_conductivity * self._t_u * self._t_v
      conductance /= self.building.floor_height_cm / 100.0
      numerator += tf.math.divide_no_nan(
          tf.convert_to_tensor(
              self._exterior_forcing.mass_gain, dtype=tf.float32
          ),
          conductance,
      )

    # Calculate denominator: 1 + t_0_mass
    denominator = tf.math.add(
        tf.constant(1.0, dtype=tf.float32), self._t_0_mass
    )

    # Update interior mass temperature
    t_temp_mass_new = tf.math.divide(numerator, denominator)

    # Apply mask to only update where interior mass exists
    t_temp_mass_new = tf.where(
        self._t_interior_mass_mask, t_temp_mass_new, t_temp_mass_prev
    )

    # Calculate maximum change for convergence checking
    t_delta = tf.math.subtract(t_temp_mass_new, t_temp_mass_prev)
    max_delta = np.max(tf.math.abs(t_delta))

    # Return as numpy arrays
    return t_temp_mass_new.numpy(), max_delta

  def _get_input_tensors(
      self,
      building,
      temperature_estimates: np.ndarray,
  ) -> tuple[
      tf.Tensor,
      tf.Tensor,
      tf.Tensor,
      tf.Tensor,
      tf.Tensor,
      tf.Tensor,
      tf.Tensor,
      tf.Tensor,
      tf.Tensor,
  ]:
    """Returns the per-iteration input matrices as tensors.

    Everything here changes from one Picard iteration or time step to the next,
    which is why it is rebuilt on every call rather than cached in __init__
    alongside the geometry.

    Args:
      building: the building whose state is being read.
      temperature_estimates: the current iterate $T^{(k)}$ [K].

    Returns:
      Nine tensors, in order: the current estimate, a copy of it kept to measure
      the iteration's movement, the previous time step's temperatures
      $T^{(-)}$, the injected power $Q_x$ [W], the density, the heat capacity,
      the floor height $z$ [m], the interior surface temperatures as a column
      vector, and the interior mass temperatures. The last two are placeholder
      zeros when radiation or interior mass is switched off.
    """
    # Convert a bunch of numpy arrays into TF tensors.
    t_temp = tf.convert_to_tensor(temperature_estimates, dtype=tf.float32)
    t_temp_old = tf.convert_to_tensor(temperature_estimates, dtype=tf.float32)
    t_temp_minus = tf.convert_to_tensor(building.temp, dtype=tf.float32)
    t_input_q = tf.convert_to_tensor(building.input_q, dtype=tf.float32)
    t_density = tf.convert_to_tensor(building.density, dtype=tf.float32)
    t_heat_capacity = tf.convert_to_tensor(
        building.heat_capacity, dtype=tf.float32
    )
    t_z = tf.constant(building.floor_height_cm / 100.0, dtype=tf.float32)
    if self.include_radiative_heat_transfer:
      # For radiative heat transfer, we need to combine interior wall and
      # interior mass temperatures if interior mass is enabled
      if self.include_interior_mass:
        interior_mask_all = (
            building.interior_wall_mask | building.interior_mass_mask
        )
        temperature_estimates_temp = np.zeros_like(temperature_estimates)
        temperature_estimates_temp[building.interior_mass_mask] = (
            building.interior_mass_temp[building.interior_mass_mask]
        )
        temperature_estimates_temp[building.interior_wall_mask] = (
            temperature_estimates[building.interior_wall_mask]
        )
        t_temp_interior_wall = tf.convert_to_tensor(
            temperature_estimates_temp[interior_mask_all], dtype=tf.float32
        )
      else:
        t_temp_interior_wall = tf.convert_to_tensor(
            temperature_estimates[building.interior_wall_mask],
            dtype=tf.float32,
        )
      # Ensure t_temp_interior_wall is a column vector for matrix
      # multiplication
      t_temp_interior_wall = tf.reshape(t_temp_interior_wall, [-1, 1])
    else:
      # Create minimal zero tensors with appropriate shapes
      # These won't be used when radiative heat transfer is disabled
      t_temp_interior_wall = tf.zeros((1,), dtype=tf.float32)  # Minimal shape

    # Interior mass temperature tensor
    if self.include_interior_mass:
      t_temp_mass = tf.convert_to_tensor(
          building.interior_mass_temp, dtype=tf.float32
      )
    else:
      t_temp_mass = tf.zeros((1, 1), dtype=tf.float32)  # Minimal shape

    return (
        t_temp,
        t_temp_old,
        t_temp_minus,
        t_input_q,
        t_density,
        t_heat_capacity,
        t_z,
        t_temp_interior_wall,
        t_temp_mass,
    )

  def _get_neighbor_temps(
      self, t_temp: tf.Tensor, ambient_temperature: float
  ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Creates left, right, up, down neighbor temp matrices."""

    # Create shifted tensor to be able to evaluate neighbors in the equation.
    t_temp_right = shift_tensor_left(t_temp, padding_value=ambient_temperature)

    t_temp_left = shift_tensor_right(t_temp, padding_value=ambient_temperature)

    t_temp_above = shift_tensor_down(t_temp, padding_value=ambient_temperature)

    t_temp_below = shift_tensor_up(t_temp, padding_value=ambient_temperature)

    return (t_temp_left, t_temp_right, t_temp_above, t_temp_below)

  def _get_longwave_grid_terms(
      self,
      t_temp_interior_wall: tf.Tensor,
      t_z: tf.Tensor,
  ) -> tuple[tf.Tensor, tf.Tensor]:
    r"""Splits the interior longwave exchange into a diagonal and a source.

    The net radiative gain of enclosure node $i$ is
    $q_i = \sum_j M_{ij} \sigma (T_j^4 - T_i^4)$ with $M = -\text{ifa\_inv}$ off
    the diagonal. Because $a^4 - b^4 = (a^2 + b^2)(a + b)(a - b)$ exactly, this
    can be rewritten without approximating anything as

    $$q_i = \sum_j M_{ij} h_{ij} (T_j - T_i), \quad
      h_{ij} = \sigma (T_i^2 + T_j^2)(T_i + T_j),$$

    which separates into a term proportional to the CV's own unknown and a term
    that is not. Moving the first onto the denominator instead of leaving the
    whole of $q_i$ in the numerator is what makes the Picard sweep stable: with
    $M_{ij} \ge 0$ and $h_{ij} > 0$ the added diagonal is exactly the sum of the
    added numerator weights, so the update stays a convex combination of
    temperatures and cannot run away. Left fully explicit, the same balance
    diverges to NaN at large time steps. The fixed point is untouched, since
    $T_i (d + \sum_j M_{ij} h_{ij}) = n + \sum_j M_{ij} h_{ij} T_j$ is just
    $T_i d = n + q_i$ rearranged. $h_{ij}$ is the exact secant of $\sigma T^4$
    between the two temperatures; the textbook $4 \sigma \bar{T}^3$ is its
    $T_i \to T_j$ limit.

    The off-diagonal coupling stays lagged at the last iterate. Making it
    implicit would fill in the sparse five-point stencil used by the
    matrix-free PCG operator, since $M$ is dense.

    Args:
      t_temp_interior_wall: enclosure temperatures [K] as a column, in the order
        _get_input_tensors stacks them.
      t_z: floor height $z$ [m].

    Returns:
      A pair of tensors shaped like the grid, both already multiplied by the
      area the flux crosses. The first is the exchange coefficient
      $A \sum_j M_{ij} h_{ij}$ [W/K] belonging on the denominator, the second
      the driving power $A \sum_j M_{ij} h_{ij} T_j$ [W] belonging in the
      numerator. Both are zero when radiative heat transfer is switched off.
    """
    zeros = tf.zeros_like(self._t_u)
    if not self.include_radiative_heat_transfer:
      return zeros, zeros

    sigma = tf.constant(constants.STEFAN_BOLTZMANN_CONSTANT, dtype=tf.float32)
    t_temps = tf.reshape(t_temp_interior_wall, [-1])
    t_coefficient = tf.zeros_like(t_temps)
    t_driving = tf.zeros_like(t_temps)
    for t_enclosure_indices, t_ifa_inv in self._lwx_ifa_blocks:
      t_enclosure_temps = tf.gather(t_temps, t_enclosure_indices)
      t_exchange = tf.math.negative(
          tf.linalg.set_diag(
              t_ifa_inv,
              tf.zeros(tf.shape(t_ifa_inv)[0], dtype=tf.float32),
          )
      )
      t_squared = tf.math.square(t_enclosure_temps)
      t_cubed = tf.math.multiply(t_squared, t_enclosure_temps)
      t_fourth_power = tf.math.square(t_squared)
      t_powers = tf.stack(
          (
              tf.ones_like(t_enclosure_temps),
              t_enclosure_temps,
              t_squared,
              t_cubed,
              t_fourth_power,
          ),
          axis=1,
      )
      (
          t_moment_0,
          t_moment_1,
          t_moment_2,
          t_moment_3,
          t_moment_4,
      ) = tf.unstack(tf.linalg.matmul(t_exchange, t_powers), axis=1)
      t_enclosure_coefficient = tf.math.scalar_mul(
          sigma,
          t_cubed * t_moment_0
          + t_squared * t_moment_1
          + t_enclosure_temps * t_moment_2
          + t_moment_3,
      )
      t_enclosure_driving = tf.math.scalar_mul(
          sigma,
          t_cubed * t_moment_1
          + t_squared * t_moment_2
          + t_enclosure_temps * t_moment_3
          + t_moment_4,
      )
      scatter_indices = t_enclosure_indices[:, tf.newaxis]
      t_coefficient = tf.tensor_scatter_nd_update(
          t_coefficient, scatter_indices, t_enclosure_coefficient
      )
      t_driving = tf.tensor_scatter_nd_update(
          t_driving, scatter_indices, t_enclosure_driving
      )

    # A mass node radiates at its own temperature but deposits its flux into
    # the air CV sharing its coordinates, so T_i is not the unknown that CV
    # solves for and its self term cannot move onto the denominator. Put it
    # back into the source instead, which leaves that node exactly as explicit
    # as it was before. Wall nodes are their own CV's unknown, so theirs moves.
    t_is_wall = tf.constant(self._lwx_is_wall_node, dtype=tf.float32)
    t_driving = tf.math.subtract(
        t_driving,
        tf.math.multiply(
            tf.math.multiply(1.0 - t_is_wall, t_coefficient), t_temps
        ),
    )
    t_coefficient = tf.math.multiply(t_is_wall, t_coefficient)

    indices = tf.where(self.building.lwx_index >= 0)
    nodes = self.building.lwx_index[self.building.lwx_index >= 0]
    # Both terms are fluxes per unit area, so they only enter the balance once
    # multiplied by the area they cross. An edge CV exposes one full face and a
    # corner CV two half faces, so either way the exposed area is delta_x * z.
    # The iterative simulator already does this; without it the two solvers
    # disagree by that factor.
    area = tf.constant(self.building.cv_size_cm / 100.0, dtype=tf.float32) * t_z
    return (
        tf.scalar_mul(
            area,
            tf.tensor_scatter_nd_update(
                zeros, indices, tf.gather(t_coefficient, nodes)
            ),
        ),
        tf.scalar_mul(
            area,
            tf.tensor_scatter_nd_update(
                zeros, indices, tf.gather(t_driving, nodes)
            ),
        ),
    )

  def _get_denominator(
      self,
      t_gamma_left: tf.Tensor,
      t_gamma_right: tf.Tensor,
      t_gamma_ambient_left: tf.Tensor,
      t_gamma_ambient_right: tf.Tensor,
      t_gamma_ambient_top: tf.Tensor,
      t_gamma_ambient_bottom: tf.Tensor,
      t_vz: tf.Tensor,
      t_gamma_bottom: tf.Tensor,
      t_gamma_top: tf.Tensor,
      t_uz: tf.Tensor,
      t_density: tf.Tensor,
      t_heat_capacity: tf.Tensor,
      t_z: tf.Tensor,
      t_delta_t: tf.Tensor,
      t_lwx_coefficient: tf.Tensor,
  ) -> tf.Tensor:
    r"""Returns the denominator of the CV energy balance as a tensor.

    Every CV's updated temperature is a conductance-weighted average, and this
    is the total conductance out of the CV that it is divided by,

    $$\sum_{f \in I_P} A_f \Gamma_f + \sum_{f \in B_P} A_f \Gamma_{\infty,f}
      + A \sum_j M_{ij} h_{ij}
      + \frac{k_{\text{mass}} u v}{z} + \frac{\rho c u v z}{\Delta t},$$

    in W/K. The interior and ambient sums are disjoint, since a face either has
    a neighbor or opens onto the ambient, so passing both sets of gammas in and
    adding them all up covers each face exactly once. The radiation and interior
    mass terms are zero when those models are switched off.

    Args:
      t_gamma_left: interior face conductance per unit area on the left face,
        $\Gamma_f$ [W/m^2/K], zero where that face opens onto the ambient.
      t_gamma_right: same, right face.
      t_gamma_ambient_left: ambient face conductance per unit area on the left
        face, $\Gamma_{\infty,f}$, zero where that face has a neighbor.
      t_gamma_ambient_right: same, right face.
      t_gamma_ambient_top: same, top face.
      t_gamma_ambient_bottom: same, bottom face.
      t_vz: area $v z$ [m^2] of the left and right faces.
      t_gamma_bottom: interior face conductance per unit area, bottom face.
      t_gamma_top: interior face conductance per unit area, top face.
      t_uz: area $u z$ [m^2] of the top and bottom faces.
      t_density: $\rho$ [kg/m^3].
      t_heat_capacity: $c$ [J/kg/K].
      t_z: floor height $z$ [m].
      t_delta_t: time step $\Delta t$ [s].
      t_lwx_coefficient: the radiative exchange coefficient
        $A \sum_j M_{ij} h_{ij}$ [W/K] from _get_longwave_grid_terms, already
        area-weighted and zero away from interior walls.

    Returns:
      A tensor shaped like the grid holding the total conductance [W/K] out of
      every CV.
    """

    # Compute conductivity/conduction transfer terms on the v-z surface.
    dt1 = tf.math.add(t_gamma_left, t_gamma_right)
    dt1 = tf.math.add(dt1, t_gamma_ambient_left)
    dt1 = tf.math.add(dt1, t_gamma_ambient_right)
    dt1 = tf.math.multiply(t_vz, dt1)

    # Compute conductivity/conduction transfer terms on the u-z surface.
    dt2 = tf.math.add(t_gamma_bottom, t_gamma_top)
    dt2 = tf.math.add(dt2, t_gamma_ambient_bottom)
    dt2 = tf.math.add(dt2, t_gamma_ambient_top)
    dt2 = tf.math.multiply(t_uz, dt2)

    # Create the thermal absorption (storage) term: C * rho * U * V * z / dt.
    dt3 = tf.math.multiply(t_density, self._t_u)
    dt3 = tf.math.multiply(dt3, self._t_v)
    dt3 = tf.math.multiply(dt3, t_heat_capacity)
    dt3 = tf.scalar_mul(t_z, dt3)
    dt3 = tf.math.divide(dt3, t_delta_t)

    # Add interior mass coupling term: K_mass * U * V / Z
    dt4 = tf.zeros_like(dt3)
    if self.include_interior_mass:
      dt4 = tf.math.multiply(self._t_interior_mass_conductivity, self._t_u)
      dt4 = tf.math.multiply(dt4, self._t_v)
      dt4 = tf.math.divide(dt4, t_z)

    # Sum up v-z and u-z face transfer, storage, interior mass, and the
    # radiative self term that _get_longwave_grid_terms moved off the numerator.
    t_denom = tf.math.add(dt1, dt2)
    t_denom = tf.math.add(t_denom, dt3)
    t_denom = tf.math.add(t_denom, dt4)
    t_denom = tf.math.add(t_denom, t_lwx_coefficient)
    return t_denom

  def _get_numerator(
      self,
      t_gamma_left: tf.Tensor,
      t_gamma_right: tf.Tensor,
      t_gamma_ambient_left: tf.Tensor,
      t_gamma_ambient_right: tf.Tensor,
      t_gamma_ambient_top: tf.Tensor,
      t_gamma_ambient_bottom: tf.Tensor,
      t_vz: tf.Tensor,
      t_gamma_bottom: tf.Tensor,
      t_gamma_top: tf.Tensor,
      t_uz: tf.Tensor,
      t_density: tf.Tensor,
      t_heat_capacity: tf.Tensor,
      t_z: tf.Tensor,
      t_delta_t: tf.Tensor,
      t_temp_left: tf.Tensor,
      t_temp_right: tf.Tensor,
      t_temp_above: tf.Tensor,
      t_temp_below: tf.Tensor,
      t_temp_inf: tf.Tensor,
      t_input_q: tf.Tensor,
      t_temp_minus: tf.Tensor,
      t_lwx_driving: tf.Tensor,
      t_temp_mass: tf.Tensor,
  ) -> tf.Tensor:
    r"""Returns the numerator of the CV energy balance as a tensor.

    The counterpart of _get_denominator: the same conductances, each now
    weighting the temperature on the other side of its face, plus every source
    that does not scale with the CV's own temperature,

    $$\sum_{f \in I_P} A_f \Gamma_f T_{N_f}
      + \sum_{f \in B_P} A_f \Gamma_{\infty,f} T_\infty
      + Q_x + Q_{\text{lwx}}
      + \frac{k_{\text{mass}} u v}{z} T_{\text{mass}}
      + \frac{\rho c u v z}{\Delta t} T^{(-)},$$

    in W. Dividing this by the denominator gives the next iterate directly. The
    radiation and interior mass terms are zero when those models are switched
    off.

    Args:
      t_gamma_left: interior face conductance per unit area, left face
        [W/m^2/K].
      t_gamma_right: same, right face.
      t_gamma_ambient_left: ambient face conductance per unit area, left face.
      t_gamma_ambient_right: same, right face.
      t_gamma_ambient_top: same, top face.
      t_gamma_ambient_bottom: same, bottom face.
      t_vz: area $v z$ [m^2] of the left and right faces.
      t_gamma_bottom: interior face conductance per unit area, bottom face.
      t_gamma_top: interior face conductance per unit area, top face.
      t_uz: area $u z$ [m^2] of the top and bottom faces.
      t_density: $\rho$ [kg/m^3].
      t_heat_capacity: $c$ [J/kg/K].
      t_z: floor height $z$ [m].
      t_delta_t: time step $\Delta t$ [s].
      t_temp_left: left neighbor's temperature [K].
      t_temp_right: right neighbor's temperature [K].
      t_temp_above: top neighbor's temperature [K].
      t_temp_below: bottom neighbor's temperature [K].
      t_temp_inf: ambient temperature $T_\infty$ [K].
      t_input_q: injected power $Q_x$ [W].
      t_temp_minus: previous time step's temperature $T^{(-)}$ [K].
      t_lwx_driving: the radiative driving power
        $A \sum_j M_{ij} h_{ij} T_j$ [W] from _get_longwave_grid_terms, already
        area-weighted. Its counterpart sits on the denominator.
      t_temp_mass: interior mass temperatures [K].

    Returns:
      A tensor shaped like the grid holding the total power [W] driving every
      CV toward its next iterate.
    """

    # Compute numerator's conductivity transfer terms.
    t_gamma_left_temp_left = tf.math.multiply(t_gamma_left, t_temp_left)
    t_gamma_right_temp_right = tf.math.multiply(t_gamma_right, t_temp_right)
    t_gamma_bottom_temp_below = tf.math.multiply(t_gamma_bottom, t_temp_below)
    t_gamma_top_temp_above = tf.math.multiply(t_gamma_top, t_temp_above)

    # Compute numerator's convection transfer terms.
    t_h_left_tinf = tf.math.scalar_mul(t_temp_inf, t_gamma_ambient_left)
    t_h_right_tinf = tf.math.scalar_mul(t_temp_inf, t_gamma_ambient_right)
    t_h_above_tinf = tf.math.scalar_mul(t_temp_inf, t_gamma_ambient_top)
    t_h_below_tinf = tf.math.scalar_mul(t_temp_inf, t_gamma_ambient_bottom)

    # Merge the conduction/convection transfer terms across the v-z surfaces.
    nt1 = tf.math.add(t_gamma_left_temp_left, t_gamma_right_temp_right)
    nt1 = tf.math.add(nt1, t_h_left_tinf)
    nt1 = tf.math.add(nt1, t_h_right_tinf)
    nt1 = tf.math.multiply(t_vz, nt1)

    # Merge the conduction/convection transfer terms across the u-z surfaces.
    nt2 = tf.math.add(t_gamma_bottom_temp_below, t_gamma_top_temp_above)
    nt2 = tf.math.add(nt2, t_h_below_tinf)
    nt2 = tf.math.add(nt2, t_h_above_tinf)
    nt2 = tf.math.multiply(t_uz, nt2)

    # Create the thermal absorption (storage) term:
    # C * rho * U * V * z / dt * T^(-).
    nt3 = tf.math.multiply(t_density, self._t_u)
    nt3 = tf.math.multiply(nt3, self._t_v)
    nt3 = tf.math.multiply(nt3, t_heat_capacity)
    nt3 = tf.scalar_mul(t_z, nt3)
    nt3 = tf.math.multiply(nt3, t_temp_minus)
    nt3 = tf.math.divide(nt3, t_delta_t)

    # The radiative exchange between the interior surfaces, sum_j M_ij h_ij T_j
    # with M = -ifa_inv off the diagonal. Only the part that does not scale
    # with the CV's own temperature is here; the rest is on the denominator.
    # See _get_longwave_grid_terms for why the split is exact and why it is
    # needed.
    nt4 = t_lwx_driving

    # Add interior mass coupling term: K_mass * U * V / Z * T_mass
    nt5 = tf.zeros_like(t_temp_minus)
    if self.include_interior_mass:
      nt5 = tf.math.multiply(self._t_interior_mass_conductivity, self._t_u)
      nt5 = tf.math.multiply(nt5, self._t_v)
      nt5 = tf.math.multiply(nt5, t_temp_mass)
      nt5 = tf.math.divide(nt5, t_z)

    # Add the v-z and u-z face transfer, absorption, external source,
    #  and interior mass terms.
    t_numer = tf.math.add(nt1, nt2)
    t_numer = tf.math.add(t_numer, nt3)
    t_numer = tf.math.add(t_numer, t_input_q)
    t_numer = tf.math.add(t_numer, nt4)
    t_numer = tf.math.add(t_numer, nt5)
    return t_numer

  def _tf_exterior_forcing(
      self, forcing: building_radiation_utils.ExteriorRadiationForcing | None
  ) -> building_radiation_utils.ExteriorRadiationForcing | None:
    if forcing is None:
      return None
    return building_radiation_utils.ExteriorRadiationForcing(
        *(tf.convert_to_tensor(value, dtype=tf.float32) for value in forcing)
    )

  def update_temperature_estimates(
      self,
      temperature_estimates: np.ndarray,
      ambient_temperature: float,
      convection_coefficient: float,
  ) -> tuple[np.ndarray, float]:
    r"""Iterates across all CVs and updates the temperature estimate.

    Corner and edge CVs are exposed to thermal exchange with the ambient air
    through convection. This tensorized implementation overrides the parent
    class's iterative approach for computational efficiency.

    Equations:
    --------------------
    The tensorized heat balance equation for air CV with interior mass
    coupling is:

    $$\begin{multline}
      T = \left[Q_x + Vz\left[\Gamma_1T_1 + \Gamma_{\infty,1}T_\infty +
        \Gamma_3T_3 + \Gamma_{\infty,3}T_\infty\right] \right. \\
      \left. + Uz\left[\Gamma_2T_2 + \Gamma_{\infty,2}T_\infty + \Gamma_4T_4 +
        \Gamma_{\infty,4}T_\infty\right] \right. \\
      \left. + K_{\text{mass}}UVz^{-1}T_{\text{mass}} + Q_{\text{lwx}} +
        \frac{C\rho UVz}{\Delta t}T^{(-)}\right] \\
      \cdot \left[Vz\left[\Gamma_1 + \Gamma_{\infty,1} + \Gamma_3 +
        \Gamma_{\infty,3}\right] +
        Uz\left[\Gamma_2 + \Gamma_{\infty,2} + \Gamma_4 +
        \Gamma_{\infty,4}\right] \right. \\
      \left. + K_{\text{mass}}UVz^{-1} +
        \frac{C\rho UVz}{\Delta t}\right]^{-1}
      \end{multline}$$

    where the shifted temperature tensors represent neighboring CVs:
    - $T_1 = \text{shift}(T, \text{LEFT})$
    - $T_2 = \text{shift}(T, \text{DOWN})$
    - $T_3 = \text{shift}(T, \text{RIGHT})$
    - $T_4 = \text{shift}(T, \text{UP})$

    An interior face is shared by two CVs, so its conductance per unit area is
    the inverse of the two half-cell conduction resistances in series,

    $$\Gamma_1 = \left[\frac{U}{2K} + \frac{U_1}{2K_1}\right]^{-1}, \quad
      \Gamma_2 = \left[\frac{V}{2K} + \frac{V_2}{2K_2}\right]^{-1},$$

    and likewise for $\Gamma_3$ and $\Gamma_4$. A face that opens onto the
    ambient instead puts the convective film in series with the outer half
    cell,

    $$\Gamma_{\infty,1} = \Gamma_{\infty,3} =
        \left[\frac{1}{H_{ext}} + \frac{U}{2K}\right]^{-1}, \quad
      \Gamma_{\infty,2} = \Gamma_{\infty,4} =
        \left[\frac{1}{H_{ext}} + \frac{V}{2K}\right]^{-1},$$

    so $H_{ext}$ appears only on exterior boundary faces. Using $KU^{-1}$ on
    interior faces, i.e. the owner CV's own conductivity over its own full
    width, would make the two CVs sharing a face disagree about the heat
    crossing it and break conservation at material interfaces.

    The area a face is multiplied by is the shared one, the narrower of the two
    CVs, $A_1 = A_3 = \min(V, V_f) z$ and $A_2 = A_4 = \min(U, U_f) z$. Where a
    half-width boundary CV meets a full-width neighbor, taking each CV's own
    width would have the two disagree about the area as well as the
    conductivity. The narrowing is folded into $\Gamma_f$ here, so the balance
    below can multiply by the owner CV's own $Vz$ and $Uz$ and still get the
    shared area.

    Nomenclature and Units:
    -----------------------
    - $T$: Air temperature tensor at new time step [K]
    - $T^{(-)}$: Air temperature tensor at previous time step [K]
    - $T_1, T_2, T_3, T_4$: Temperature tensors of neighboring CVs
      (left, down, right, up) [K]
    - $T_{\text{mass}}$: Interior mass temperature tensor [K]
    - $T_\infty$: Ambient temperature (scalar) [K]
    - $Q_x$: External heat source tensor [$\mathrm{W}$]
    - $Q_{\text{lwx}}$: Longwave radiative exchange tensor [$\mathrm{W}$]
    - $K, K_1, K_2, K_3, K_4$: Thermal conductivity of the CV itself and of
      its left, down, right and up neighbors [$\mathrm{W/(m \cdot K)}$]
    - $\Gamma_1, \Gamma_2, \Gamma_3, \Gamma_4$: Interior face conductance
      tensors for the left, down, right, up faces
      [$\mathrm{W/(m^2 \cdot K)}$]
    - $\Gamma_{\infty,f}$: Ambient face conductance tensors, nonzero only on
      exterior boundary faces [$\mathrm{W/(m^2 \cdot K)}$]
    - $K_{\text{mass}}$: Interior mass conductivity tensor
      [$\mathrm{W/(m \cdot K)}$]
    - $H_{ext}$: Outside air convection coefficient
      [$\mathrm{W/(m^2 \cdot K)}$]
    - $U, V$: CV extent along the column (j) and row (i) directions
      respectively, so $U$ is the half-cell length toward the left and right
      faces and $V$ toward the top and bottom ones [$\mathrm{m}$]
    - $z$: CV height (floor height) [$\mathrm{m}$]
    - $C$: Specific heat capacity tensor [$\mathrm{J/(kg \cdot K)}$]
    - $\rho$: Density tensor [$\mathrm{kg/m^3}$]
    - $\Delta t$: Time step [$\mathrm{s}$]

    Note: reference/sbsim_equation.pdf assigns $u$ to the first index and $v$
    to the second, the opposite of the convention used here. On the square grid
    this code builds the two agree numerically, since $U$ and $V$ only ever
    differ between a boundary CV and an interior one, never between the two
    axes of the same interior CV.

    References:
    -----------
    - reference/sbsim_equation.pdf, sections 7 and 8
    - Equation 22 derived in go/smart-buildings-simulator-design

    Args:
      temperature_estimates: Current temperature estimate for each CV, will be
        updated with new values.
      ambient_temperature: Current temperature in K of external air.
      convection_coefficient: Current wind convection coefficient (W/m2/K).

    Returns:
      A pair: the updated temperature estimates, and the largest change any CV
      saw over this iteration, which the caller's Picard loop uses as its
      convergence measure.
    """

    # Get the inputs to the equation as Tensors from the building.
    (
        t_temp,
        t_temp_old,
        t_temp_minus,
        t_input_q,
        t_density,
        t_heat_capacity,
        t_z,
        t_temp_interior_wall,
        t_temp_mass,
    ) = self._get_input_tensors(self.building, temperature_estimates)

    (
        t_gamma_ambient_left,
        t_gamma_ambient_right,
        t_gamma_ambient_top,
        t_gamma_ambient_bottom,
    ) = get_oriented_ambient_conductance_tensors(
        convection_coefficient,
        self.building.conductivity,
        self._t_u,
        self._t_v,
        self._boundary_cv_mapping,
    )

    # Create shifted tensor to be able to evaluate neighbors in the equation.
    (t_temp_left, t_temp_right, t_temp_above, t_temp_below) = (
        self._get_neighbor_temps(t_temp, ambient_temperature)
    )

    # Get the ambient temperature as a tensor.
    t_temp_inf = tf.constant(ambient_temperature, dtype=tf.float32)

    # Convert the timestep input to tensor.
    t_delta_t = tf.constant(self._time_step_sec, dtype=tf.float32)

    # Create surface area tensors: horizontal u or v x vertical z dim.
    t_uz = tf.scalar_mul(t_z, self._t_u)
    t_vz = tf.scalar_mul(t_z, self._t_v)

    # Interior face conductances already account for the half cell on either
    # side of the face, so they are used as-is rather than divided by the
    # owner CV's own width.
    t_gamma_left = self._t_face_conductance_left
    t_gamma_right = self._t_face_conductance_right
    t_gamma_bottom = self._t_face_conductance_bottom
    t_gamma_top = self._t_face_conductance_top

    # Radiation is split so that the part scaling with the CV's own unknown
    # lands on the denominator and the rest stays a source. Both are evaluated
    # at the current field, so the exchange itself is still lagged.
    t_lwx_coefficient, t_lwx_driving = self._get_longwave_grid_terms(
        t_temp_interior_wall, t_z
    )
    exterior_forcing = self._tf_exterior_forcing(self._exterior_forcing)
    if exterior_forcing is not None:
      coefficient, driving = exterior_forcing.linearized_air_terms(t_temp)
      t_lwx_coefficient += coefficient
      t_input_q += driving

    t_denom = self._get_denominator(
        t_gamma_left,
        t_gamma_right,
        t_gamma_ambient_left,
        t_gamma_ambient_right,
        t_gamma_ambient_top,
        t_gamma_ambient_bottom,
        t_vz,
        t_gamma_bottom,
        t_gamma_top,
        t_uz,
        t_density,
        t_heat_capacity,
        t_z,
        t_delta_t,
        t_lwx_coefficient,
    )

    if self.solver_method == 'fdm':
      t_numer = self._get_numerator(
          t_gamma_left,
          t_gamma_right,
          t_gamma_ambient_left,
          t_gamma_ambient_right,
          t_gamma_ambient_top,
          t_gamma_ambient_bottom,
          t_vz,
          t_gamma_bottom,
          t_gamma_top,
          t_uz,
          t_density,
          t_heat_capacity,
          t_z,
          t_delta_t,
          t_temp_left,
          t_temp_right,
          t_temp_above,
          t_temp_below,
          t_temp_inf,
          t_input_q,
          t_temp_minus,
          t_lwx_driving,
          t_temp_mass,
      )
      t_temperature_estimates = tf.math.divide(t_numer, t_denom)
    else:
      t_zeros = tf.zeros_like(t_temp)
      t_rhs = self._get_numerator(
          t_gamma_left,
          t_gamma_right,
          t_gamma_ambient_left,
          t_gamma_ambient_right,
          t_gamma_ambient_top,
          t_gamma_ambient_bottom,
          t_vz,
          t_gamma_bottom,
          t_gamma_top,
          t_uz,
          t_density,
          t_heat_capacity,
          t_z,
          t_delta_t,
          t_zeros,
          t_zeros,
          t_zeros,
          t_zeros,
          t_temp_inf,
          t_input_q,
          t_temp_minus,
          t_lwx_driving,
          t_temp_mass,
      )
      coupling_left = tf.math.multiply(t_vz, t_gamma_left)
      coupling_right = tf.math.multiply(t_vz, t_gamma_right)
      coupling_top = tf.math.multiply(t_uz, t_gamma_top)
      coupling_bottom = tf.math.multiply(t_uz, t_gamma_bottom)
      exterior_float = tf.cast(self._t_exerior_temps_mask, tf.float32)
      t_rhs += t_temp_inf * (
          coupling_left * shift_tensor_right(exterior_float)
          + coupling_right * shift_tensor_left(exterior_float)
          + coupling_top * shift_tensor_down(exterior_float)
          + coupling_bottom * shift_tensor_up(exterior_float)
      )
      t_rhs = tf.where(
          self._t_exerior_temps_mask,
          tf.fill(tf.shape(t_rhs), t_temp_inf),
          t_rhs,
      )
      t_cg_diagonal = tf.where(
          self._t_exerior_temps_mask, tf.ones_like(t_denom), t_denom
      )
      t_initial = apply_exterior_temps(
          t_temp, t_temp_inf, self._t_exerior_temps_mask
      )
      (
          t_temperature_estimates,
          t_residual_norm,
          t_iteration_count,
      ) = _solve_with_preconditioned_conjugate_gradient(
          tf.cast(t_cg_diagonal, tf.float64),
          tf.cast(t_rhs, tf.float64),
          tf.cast(coupling_left, tf.float64),
          tf.cast(coupling_right, tf.float64),
          tf.cast(coupling_top, tf.float64),
          tf.cast(coupling_bottom, tf.float64),
          self._t_exerior_temps_mask,
          tf.cast(t_initial, tf.float64),
          tf.constant(self.cg_tolerance, dtype=tf.float64),
          tf.constant(self.cg_iteration_limit, dtype=tf.int32),
      )
      self.last_cg_iteration_count = int(t_iteration_count.numpy())
      self.last_cg_residual_norm = float(t_residual_norm.numpy())
      t_temperature_estimates = tf.cast(
          t_temperature_estimates, dtype=tf.float32
      )
      rhs_norm = float(tf.linalg.norm(tf.cast(t_rhs, tf.float64)).numpy())
      target_norm = self.cg_tolerance * max(rhs_norm, 1.0)
      if self.last_cg_residual_norm > target_norm:
        raise RuntimeError(
            'TensorFlow CG did not converge: '
            f'iterations={self.last_cg_iteration_count}, '
            f'residual_norm={self.last_cg_residual_norm}, '
            f'target_norm={target_norm}'
        )

    # The tensor operation potentially altered the exterior air conditions,
    # so we need to reset exterior CVs to the exterior air conditioners.
    t_temperature_estimates = apply_exterior_temps(
        t_temperature_estimates, t_temp_inf, self._t_exerior_temps_mask
    )

    t_delta = tf.math.subtract(t_temperature_estimates, t_temp_old)

    # Note: Interior mass temperatures are updated by the parent class's
    # finite_differences_timestep() method after this function returns.
    # Do NOT update them here to avoid double-updating.

    return t_temperature_estimates.numpy(), np.max(tf.math.abs(t_delta))
