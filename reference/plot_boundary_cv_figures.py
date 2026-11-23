"""Draws the figures for boundary_cv_heat_transfer_ko.md.

Run from the repository root:

  .venv/bin/python reference/plot_boundary_cv_figures.py

Writes PNGs to reference/figures/boundary_cv/, which the markdown embeds.
Lengths are drawn in units
of the grid spacing dx; the CV extents follow Simulator._get_cv_dimensions.
"""

import pathlib

import matplotlib

matplotlib.use('Agg')
from matplotlib import patches  # pylint: disable=g-import-not-at-top
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=g-import-not-at-top

HERE = pathlib.Path(__file__).parent
OUT_DIR = HERE / 'figures' / 'boundary_cv'

plt.rcParams.update({
    'mathtext.fontset': 'cm',
    'font.family': 'DejaVu Sans',
    'font.size': 11,
})

SOLID = '#17607f'  # the dark teal of the slide deck
SOLID_LIGHT = '#4f8fab'
AMBIENT = '#dcebf7'
CORNER = '#f4a261'
EDGE = '#e9c46a'
INTERIOR = '#8ecae6'
ARROW = '#e8702a'
RED = '#c0392b'
GREEN = '#2e7d32'
GRAY = '#7f7f7f'

# A building with convex corners, straight edges and one reentrant corner.
MASK = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0],
])


def _missing_faces(i, j):
  """Returns which of left, right, up, down has no building neighbor."""
  return {
      'left': not MASK[i, j - 1],
      'right': not MASK[i, j + 1],
      'up': not MASK[i - 1, j],
      'down': not MASK[i + 1, j],
  }


def _cv_rectangle(i, j):
  """Returns (x0, x1, y0, y1) of the CV in plot coordinates (y = -row).

  Same rule as Simulator._get_cv_dimensions: an axis is halved when a face
  along it opens onto the ambient, and the half that is kept is the one that
  faces into the building.
  """
  m = _missing_faces(i, j)
  x0, x1 = j - 0.5, j + 0.5
  y0, y1 = -i - 0.5, -i + 0.5
  if m['left']:
    x0 = j
  if m['right']:
    x1 = j
  if m['up']:
    y1 = -i
  if m['down']:
    y0 = -i
  return x0, x1, y0, y1


def _arrow(ax, start, end, color=ARROW, lw=2.0, style='-|>', ms=14):
  ax.annotate(
      '',
      xy=end,
      xytext=start,
      arrowprops=dict(arrowstyle=style, color=color, lw=lw, mutation_scale=ms),
  )


def _dim(ax, p0, p1, text, offset=(0, 0), color='k', fontsize=11):
  """Draws a two-headed dimension line with a label."""
  ax.annotate(
      '',
      xy=p1,
      xytext=p0,
      arrowprops=dict(arrowstyle='<->', color=color, lw=1.0),
  )
  mid = ((p0[0] + p1[0]) / 2 + offset[0], (p0[1] + p1[1]) / 2 + offset[1])
  ax.text(mid[0], mid[1], text, ha='center', va='center', color=color,
          fontsize=fontsize,
          bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.85))


def _resistor(ax, x0, x1, y, label, color='k', n=6, amp=0.08):
  """Draws a zigzag resistor between x0 and x1 at height y."""
  lead = (x1 - x0) * 0.2
  xs = [x0, x0 + lead]
  ys = [y, y]
  step = (x1 - x0 - 2 * lead) / n
  for k in range(n):
    xs.append(x0 + lead + step * (k + 0.5))
    ys.append(y + (amp if k % 2 == 0 else -amp))
  xs += [x1 - lead, x1]
  ys += [y, y]
  ax.plot(xs, ys, color=color, lw=1.6)
  ax.text((x0 + x1) / 2, y - 0.28, label, ha='center', va='top',
          fontsize=13, color=color)


def _node(ax, x, y, label=None, color='k', hollow=False, size=9,
          label_offset=(0.0, 0.12), fontsize=13, label_color=None):
  ax.plot(x, y, 'o', ms=size, mfc='white' if hollow else color, mec=color,
          mew=1.6, zorder=6)
  if label:
    ax.text(x + label_offset[0], y + label_offset[1], label, ha='center',
            va='bottom', fontsize=fontsize, color=label_color or color,
            zorder=7)


def _colored_line(ax, x, y, parts, fontsize=14):
  """Writes math segments one after another, each in its own color."""
  renderer = ax.figure.canvas.get_renderer()
  ax_width = ax.get_window_extent(renderer).width
  for text, color in parts:
    t = ax.text(x, y, text, color=color, fontsize=fontsize, va='baseline',
                ha='left', transform=ax.transAxes)
    x += t.get_window_extent(renderer).width / ax_width


# ---------------------------------------------------------------------------
# Figure 1: the grid, the CV classes and the truncated CV extents.
# ---------------------------------------------------------------------------
def figure_grid():
  fig, ax = plt.subplots(figsize=(13, 10.5))
  rows, cols = MASK.shape
  for i in range(rows):
    for j in range(cols):
      if not MASK[i, j]:
        ax.add_patch(patches.Rectangle((j - 0.5, -i - 0.5), 1, 1, fc=AMBIENT,
                                       ec='none'))
        ax.text(j, -i, 'EXT\n$T_\\infty$', ha='center', va='center',
                fontsize=8, color=GRAY)

  # The finite-difference grid: one cell per pixel of the floor plan.
  for k in np.arange(-0.5, cols, 1.0):
    ax.plot([k, k], [0.5, -rows + 0.5], color=GRAY, lw=0.6, ls=':')
  for k in np.arange(-0.5, rows, 1.0):
    ax.plot([-0.5, cols - 0.5], [-k, -k], color=GRAY, lw=0.6, ls=':')

  colors = {2: CORNER, 3: EDGE, 4: INTERIOR}
  names = {2: 'C', 3: 'E', 4: 'I'}
  for i in range(rows):
    for j in range(cols):
      if not MASK[i, j]:
        continue
      m = _missing_faces(i, j)
      n_neighbors = 4 - sum(m.values())
      x0, x1, y0, y1 = _cv_rectangle(i, j)
      ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                     fc=colors[n_neighbors], ec='k', lw=0.8,
                                     alpha=0.9))
      # Faces that open onto the ambient: the building envelope.
      if m['left']:
        ax.plot([x0, x0], [y0, y1], color=RED, lw=3.5)
      if m['right']:
        ax.plot([x1, x1], [y0, y1], color=RED, lw=3.5)
      if m['up']:
        ax.plot([x0, x1], [y1, y1], color=RED, lw=3.5)
      if m['down']:
        ax.plot([x0, x1], [y0, y0], color=RED, lw=3.5)
      cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
      _node(ax, cx, cy, size=6)
      if n_neighbors < 4:
        # Where the paper (and the original code) puts the node: on the
        # exterior surface, at the center of the grid cell.
        _node(ax, j, -i, hollow=True, size=7, color='k')
      ax.text(cx, cy - 0.06, names[n_neighbors], ha='center', va='top',
              fontsize=9, fontweight='bold')
      u, v = x1 - x0, y1 - y0
      ax.text(x0 + 0.03, y1 - 0.03, f'u={u:g}\nv={v:g}', ha='left', va='top',
              fontsize=6.5, linespacing=1.0)

  # The reentrant corner: the part of CV (3,3)'s perimeter that faces neither
  # a neighbor nor the ambient.
  ax.plot([3.0, 3.5], [-2.5, -2.5], color=GREEN, lw=3.5, ls='--')
  ax.plot([3.5, 3.5], [-2.5, -3.0], color=GREEN, lw=3.5, ls='--')
  ax.add_patch(patches.Rectangle((3.0, -3.0), 0.5, 0.5, fc='none', ec=GREEN,
                                 hatch='///', lw=0))
  ax.annotate(
      'reentrant corner: CV (3,3) has 4\n'
      'neighbors -> INTERIOR, full size.\n'
      'Faces with (2,3) and (3,4):\n'
      'A = min(width) z = dx z/2.\n'
      'Dashed green half-faces touch\n'
      'neither a neighbor nor the\n'
      'ambient -> ADIABATIC.',
      xy=(3.4, -2.6), xytext=(6.7, -1.6), fontsize=9.5, color=GREEN,
      bbox=dict(fc='white', ec=GREEN, pad=3),
      arrowprops=dict(arrowstyle='->', color=GREEN))

  ax.annotate('corner CV: u = v = dx/2\nV = dx$^2$z/4', xy=(1.25, -1.25),
              xytext=(-0.45, 1.05), fontsize=10,
              arrowprops=dict(arrowstyle='->'))
  ax.annotate('edge CV (left face exposed):\nu = dx/2, v = dx,  V = dx$^2$z/2',
              xy=(1.25, -2.0), xytext=(-2.6, -2.0), fontsize=10,
              arrowprops=dict(arrowstyle='->'))
  ax.annotate('edge CV (top face exposed):\nu = dx, v = dx/2', xy=(2.0, -1.25),
              xytext=(1.6, 1.05), fontsize=10,
              arrowprops=dict(arrowstyle='->'))

  legend = [
      patches.Patch(fc=AMBIENT, label='Exterior CV  (fixed at $T_\\infty$)'),
      patches.Patch(fc=CORNER, ec='k', label='Corner CV (2 neighbors) - 1/4'),
      patches.Patch(fc=EDGE, ec='k', label='Edge CV (3 neighbors) - 1/2'),
      patches.Patch(fc=INTERIOR, ec='k', label='Interior CV (4 neighbors) - 1'),
      plt.Line2D([], [], color=RED, lw=3.5,
                 label='exterior surface: convective face $B_P$'),
      plt.Line2D([], [], color=GRAY, lw=0.8, ls=':',
                 label='FD grid cell ($\\delta_x \\times \\delta_x$)'),
      plt.Line2D([], [], marker='o', color='k', ls='', ms=6,
                 label='node, current code (center of truncated CV)'),
      plt.Line2D([], [], marker='o', mfc='white', mec='k', ls='', ms=7,
                 label='node, paper / original code (on the surface)'),
  ]
  ax.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, 0.0),
            ncol=2, fontsize=9.5, frameon=False)
  ax.set_xlim(-2.7, cols + 2.2)
  ax.set_ylim(-rows + 0.5, 1.6)
  ax.set_aspect('equal')
  ax.axis('off')
  ax.set_title(
      'Fig. 1  CV classes and truncated control volumes on the FD grid '
      '(units of $\\delta_x$, row index $i$ downward, column $j$ rightward)',
      fontsize=12, loc='left')
  return fig


# ---------------------------------------------------------------------------
# Figure 2: interior, edge and corner CVs with faces, areas and fluxes.
# ---------------------------------------------------------------------------
def _cv_panel(ax, kind):
  ax.set_xlim(-1.3, 1.75)
  ax.set_ylim(-1.75, 1.3)
  ax.set_aspect('equal')
  ax.axis('off')

  exposed = {'interior': (), 'edge': ('left',),
             'corner': ('left', 'top')}[kind]
  x0 = 0.0 if 'left' in exposed else -0.5
  y1 = 0.0 if 'top' in exposed else 0.5
  x1, y0 = 0.5, -0.5

  if 'left' in exposed:
    ax.add_patch(patches.Rectangle((-1.3, -1.2), 1.3, 2.5, fc=AMBIENT,
                                   ec='none'))
  if 'top' in exposed:
    ax.add_patch(patches.Rectangle((-1.3, 0.0), 3.05, 1.3, fc=AMBIENT,
                                   ec='none'))
  if exposed:
    ax.text(-1.25, 1.22, '$T_\\infty,\\ h$  (ambient)', fontsize=12,
            va='top', color=SOLID)

  # Full FD grid cell and the part of it that is kept.
  ax.add_patch(patches.Rectangle((-0.5, -0.5), 1, 1, fc='none', ec=GRAY,
                                 ls='--', lw=1.2))
  ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fc=SOLID,
                                 ec='k', lw=1.5, alpha=0.85))
  if kind == 'edge':
    ax.text(-0.25, -0.4, 'removed', ha='center', fontsize=8, color=GRAY)
  if kind == 'corner':
    ax.text(-0.25, 0.4, 'removed', ha='center', fontsize=8, color=GRAY)
  for side in exposed:
    if side == 'left':
      ax.plot([0, 0], [y0, y1], color=RED, lw=4)
    else:
      ax.plot([x0, x1], [0, 0], color=RED, lw=4)

  cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
  _node(ax, cx, cy, color='white')
  ax.text(cx + 0.04, cy + 0.04, '$T_P$', color='white', fontsize=13)
  if kind != 'interior':
    _node(ax, 0, 0, hollow=True, size=8)

  # Neighbor nodes (at the centers of the neighboring CVs) and the
  # node-to-node distances to the right and bottom neighbors.
  if 'left' not in exposed:
    _node(ax, -1.0, cy, size=7, color=SOLID_LIGHT)
  if 'top' not in exposed:
    _node(ax, cx, 1.0, size=7, color=SOLID_LIGHT)
  _node(ax, 1.0, cy, size=7, color=SOLID_LIGHT)
  _node(ax, cx, -1.0, size=7, color=SOLID_LIGHT)
  dist = lambda d: '$\\delta_x$' if abs(d - 1) < 1e-9 else '$\\frac{3}{4}\\delta_x$'
  ax.annotate('', xy=(1.0, cy), xytext=(cx, cy),
              arrowprops=dict(arrowstyle='<->', lw=0.9, color=GRAY))
  ax.text(x1 + (1.0 - x1) / 2, cy - 0.06, dist(1.0 - cx), ha='center',
          va='top', fontsize=11)
  ax.annotate('', xy=(cx, -1.0), xytext=(cx, cy),
              arrowprops=dict(arrowstyle='<->', lw=0.9, color=GRAY))
  ax.text(cx + 0.06, y0 - 0.25, dist(cy + 1.0), ha='left', va='center',
          fontsize=11)

  # Flux arrows with the paper's sign convention: in through face 1 (left)
  # and 2 (bottom), out through 3 (right) and 4 (top).
  dy = 0.17
  _arrow(ax, (x0 - 0.42, cy + dy), (x0 - 0.02, cy + dy))
  ax.text(x0 - 0.22, cy + dy + 0.04,
          '$Q_1^{conv}$' if 'left' in exposed else '$Q_1^{cond}$',
          fontsize=12, color=ARROW, ha='center', va='bottom')
  _arrow(ax, (x1 + 0.02, cy + dy), (x1 + 0.42, cy + dy))
  ax.text(x1 + 0.22, cy + dy + 0.04, '$Q_3^{cond}$', fontsize=12,
          color=ARROW, ha='center', va='bottom')
  _arrow(ax, (cx - 0.15, y0 - 0.42), (cx - 0.15, y0 - 0.02))
  ax.text(cx - 0.2, y0 - 0.22, '$Q_2^{cond}$', fontsize=12, color=ARROW,
          ha='right', va='center')
  _arrow(ax, (cx - 0.15, y1 + 0.02), (cx - 0.15, y1 + 0.42))
  ax.text(cx - 0.2, y1 + 0.22,
          '$Q_4^{conv}$' if 'top' in exposed else '$Q_4^{cond}$',
          fontsize=12, color=ARROW, ha='right', va='center')

  u = x1 - x0
  v = y1 - y0
  frac = {1.0: '\\delta_x', 0.5: '\\delta_x/2'}
  ytop = y1 + 0.55
  _dim(ax, (x0, ytop), (x1, ytop), f'$u={frac[u]}$', offset=(0, 0.1),
       fontsize=11)
  ax.plot([x0, x0], [y1, ytop], color=GRAY, lw=0.5, ls=':')
  ax.plot([x1, x1], [y1, ytop], color=GRAY, lw=0.5, ls=':')
  _dim(ax, (1.45, y0), (1.45, y1), f'$v={frac[v]}$', offset=(0.0, 0.0),
       fontsize=11)
  ax.plot([x1, 1.45], [y0, y0], color=GRAY, lw=0.5, ls=':')
  ax.plot([x1, 1.45], [y1, y1], color=GRAY, lw=0.5, ls=':')

  title = {'interior': 'Interior CV  (4 neighbors)',
           'edge': 'Edge CV  (left face on ambient)',
           'corner': 'Corner CV  (left + top on ambient)'}[kind]
  ax.set_title(title, fontsize=12)
  vol = {'interior': '\\delta_x^2 z', 'edge': '\\frac{1}{2}\\delta_x^2 z',
         'corner': '\\frac{1}{4}\\delta_x^2 z'}[kind]
  ax.text(
      -1.25, -1.72,
      f'$V = uvz = {vol}$\n'
      f'$A_1 = A_3 = vz = ({frac[v]})\\,z$\n'
      f'$A_2 = A_4 = uz = ({frac[u]})\\,z$',
      fontsize=12, va='bottom')


def figure_cv_types():
  fig, axes = plt.subplots(1, 3, figsize=(17, 7.2))
  for ax, kind in zip(axes, ('interior', 'edge', 'corner')):
    _cv_panel(ax, kind)
  fig.suptitle(
      'Fig. 2  The three CV classes: kept volume (solid), removed part '
      '(dashed), node-to-node distances and face areas', fontsize=13, x=0.01,
      ha='left')
  fig.text(0.01, 0.01,
           'Red line = exterior surface (convective face).  Hollow circle = '
           'node of the paper / original code (on the surface).  Filled = '
           'node of the current code (center of the kept volume).\nArrows '
           'follow the paper sign convention (in through faces 1, 2; out '
           'through 3, 4).  Faces: 1 left, 2 bottom, 3 right, 4 top.',
           fontsize=10, color=GRAY)
  fig.tight_layout(rect=(0, 0.06, 1, 0.94))
  return fig


# ---------------------------------------------------------------------------
# Figure 3: the series resistances of a boundary CV (slide 1 style).
# ---------------------------------------------------------------------------
def figure_resistance():
  fig = plt.figure(figsize=(16, 8.4))
  ax = fig.add_axes((0.0, 0.0, 0.6, 0.92))
  ax.set_xlim(-2.4, 4.6)
  ax.set_ylim(-4.0, 2.3)
  ax.set_aspect('equal')
  ax.axis('off')

  lp, ln = 1.5, 3.0  # l_P = dx/2 and l_N = dx, drawn at 3 units per dx
  ax.add_patch(patches.Rectangle((-2.2, 0.0), 2.2, 1.5, fc=AMBIENT,
                                 ec='none'))
  ax.add_patch(patches.Rectangle((0, 0.0), lp, 1.5, fc=SOLID, ec='k'))
  ax.add_patch(patches.Rectangle((lp, 0.0), ln, 1.5, fc=SOLID_LIGHT, ec='k'))
  ax.plot([0, 0], [0, 1.5], color=RED, lw=4)
  ax.text(0.0, 1.6, '$g$ plane\n(exterior surface)', ha='center',
          fontsize=11)
  ax.text(lp, 1.6, '$f$ plane\n(shared face)', ha='center', fontsize=11)
  ax.text(-1.1, 1.15, '$T_{amb}$', fontsize=15, ha='center')
  ax.text(-1.1, 0.35, 'film $h$', fontsize=11, ha='center', color=SOLID)
  _node(ax, lp / 2, 0.75, color='white', size=8)
  ax.text(lp / 2, 0.95, '$T_P$', color='white', fontsize=15, ha='center')
  ax.text(lp / 2, 0.15, '$k_P$', color='white', fontsize=14, ha='center')
  _node(ax, lp + ln / 2, 0.75, color='white', size=8)
  ax.text(lp + ln / 2, 0.95, '$T_{N_f}$', color='white', fontsize=15,
          ha='center')
  ax.text(lp + ln / 2, 0.15, '$k_{N_f}$', color='white', fontsize=14,
          ha='center')
  _arrow(ax, (-1.7, 0.75), (-0.05, 0.75))
  ax.text(-0.9, 0.85, '$Q_{amb,g}$', color=ARROW, fontsize=13, ha='center',
          va='bottom')
  _arrow(ax, (lp + ln / 2 - 0.2, 0.55), (lp / 2 + 0.2, 0.55))
  ax.text(lp + 0.75, 0.5, '$Q_{P\\leftarrow N_f}$', color='white',
          fontsize=13, ha='center', va='top')

  # Lengths: each node sits at the middle of its own CV.
  _dim(ax, (0, -0.25), (lp / 2, -0.25), '$\\frac{\\ell_P}{2}$',
       offset=(0, -0.22), fontsize=13)
  _dim(ax, (lp / 2, -0.25), (lp, -0.25), '$\\frac{\\ell_P}{2}$',
       offset=(0, -0.22), fontsize=13)
  _dim(ax, (lp, -0.25), (lp + ln / 2, -0.25), '$\\frac{\\ell_{N_f}}{2}$',
       offset=(0, -0.22), fontsize=13)
  _dim(ax, (0, -0.9), (lp, -0.9), '$\\ell_P$ (edge: $\\delta_x/2$)',
       offset=(0, -0.2), fontsize=11)
  _dim(ax, (lp, -0.9), (lp + ln, -0.9), '$\\ell_{N_f}$ (interior: $\\delta_x$)',
       offset=(0, -0.2), fontsize=11)

  # The equivalent circuit.
  y = -2.3
  xs = [-1.9, 0.0, lp / 2, lp, lp + ln / 2]
  labels = ['$T_{amb}$', '$T_g$', '$T_P$', '$T_f$', '$T_{N_f}$']
  for x, lab in zip(xs, labels):
    _node(ax, x, y, lab, size=7, label_offset=(0, 0.14), fontsize=13)
  _resistor(ax, xs[0], xs[1], y, '$\\frac{1}{h}$', color=RED)
  _resistor(ax, xs[1], xs[2], y, '$\\frac{\\ell_P}{2k_P}$')
  _resistor(ax, xs[2], xs[3], y, '$\\frac{\\ell_P}{2k_P}$')
  _resistor(ax, xs[3], xs[4], y, '$\\frac{\\ell_{N_f}}{2k_{N_f}}$')
  ax.annotate('', xy=(xs[0], -3.35), xytext=(xs[2], -3.35),
              arrowprops=dict(arrowstyle='-', color=RED, lw=1.5))
  ax.text((xs[0] + xs[2]) / 2, -3.45, '$1/\\Gamma_{\\infty,g}$', color=RED,
          ha='center', va='top', fontsize=14)
  ax.annotate('', xy=(xs[2], -3.35), xytext=(xs[4], -3.35),
              arrowprops=dict(arrowstyle='-', color=SOLID, lw=1.5))
  ax.text((xs[2] + xs[4]) / 2, -3.45, '$1/\\Gamma_f$', color=SOLID,
          ha='center', va='top', fontsize=14)
  fig.text(0.01, 0.975,
           'Fig. 3  Series resistances of a boundary CV (current code)',
           fontsize=13, va='top')

  eq = fig.add_axes((0.6, 0.0, 0.4, 0.92))
  eq.axis('off')
  lines = [
      ('Interior face $f$ (two half cells in series)', 12, 'k'),
      ('$\\Gamma_f = \\left(\\frac{\\ell_P}{2k_P} + '
       '\\frac{\\ell_{N_f}}{2k_{N_f}}\\right)^{-1}$', 19, SOLID),
      ('$A_f = \\min(w_P,\\ w_{N_f})\\ z$', 16, SOLID),
      ('$Q_{P\\leftarrow N_f} = A_f\\,\\Gamma_f\\,(T_{N_f} - T_P)'
       ' = -Q_{N_f\\leftarrow P}$', 16, SOLID),
      ('Exterior face $g$ (film + outer half cell)', 12, 'k'),
      ('$\\Gamma_{\\infty,g} = \\left(\\frac{1}{h} + '
       '\\frac{\\ell_P}{2k_P}\\right)^{-1}$', 19, RED),
      ('$Q_{amb,g} = A_g\\,\\Gamma_{\\infty,g}\\,(T_{amb} - T_P)$', 16, RED),
      ('Surface temperature (post-processing)', 12, 'k'),
      ('$T_g = T_{amb} - \\frac{Q_{amb,g}}{A_g\\,h}$', 16, 'k'),
      ('$w$: CV width along the face,  $\\ell$: CV extent normal to the face',
       11, GRAY),
      ('$\\ell = u$ for left/right faces,  $\\ell = v$ for top/bottom faces',
       11, GRAY),
  ]
  yy = 0.96
  for text, size, color in lines:
    eq.text(0.02, yy, text, fontsize=size, color=color, va='top')
    yy -= 0.03 + size * 0.0036
  return fig


# ---------------------------------------------------------------------------
# Figure 4: a corner CV, its neighbors and every conductance it uses.
# ---------------------------------------------------------------------------
def figure_corner_detail():
  fig = plt.figure(figsize=(17, 9.6))
  ax = fig.add_axes((0.0, 0.3, 0.52, 0.65))
  ax.set_xlim(-1.0, 2.6)
  ax.set_ylim(-1.75, 0.95)
  ax.set_aspect('equal')
  ax.axis('off')
  ax.add_patch(patches.Rectangle((-1.0, -1.75), 1.0, 2.7, fc=AMBIENT,
                                 ec='none'))
  ax.add_patch(patches.Rectangle((-1.0, 0.0), 2.5, 0.95, fc=AMBIENT,
                                 ec='none'))
  ax.text(-0.95, 0.88, '$T_\\infty,\\ h$', fontsize=14, va='top',
          color=SOLID)

  for gx in (-0.5, 0.5, 1.5):
    ax.plot([gx, gx], [0.5, -1.5], color=GRAY, lw=0.6, ls=':')
  for gy in (0.5, -0.5, -1.5):
    ax.plot([-0.5, 1.5], [gy, gy], color=GRAY, lw=0.6, ls=':')

  cvs = {
      'P': ((0.0, 0.5, -0.5, 0.0), CORNER, 'corner $P$'),
      'E1': ((0.5, 1.5, -0.5, 0.0), EDGE, 'edge $E_1$'),
      'E2': ((0.0, 0.5, -1.5, -0.5), EDGE, 'edge $E_2$'),
      'I': ((0.5, 1.5, -1.5, -0.5), INTERIOR, 'interior'),
  }
  centers = {}
  for key, ((x0, x1, y0, y1), color, label) in cvs.items():
    ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fc=color,
                                   ec='k', lw=1.2))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    centers[key] = (cx, cy)
    _node(ax, cx, cy, size=7)
    ax.text(cx + 0.04, cy + 0.03, label, ha='left', va='bottom', fontsize=10)
  ax.plot([0, 0], [-1.5, 0], color=RED, lw=4)
  ax.plot([0, 1.5], [0, 0], color=RED, lw=4)

  # Distances between nodes.
  ax.annotate('', xy=(1.0, -0.25), xytext=(0.25, -0.25),
              arrowprops=dict(arrowstyle='<->', lw=0.9, color='k'))
  ax.text(0.62, -0.23, '$\\frac{3}{4}\\delta_x$', fontsize=12,
          va='bottom', ha='center')
  ax.annotate('', xy=(0.25, -1.0), xytext=(0.25, -0.25),
              arrowprops=dict(arrowstyle='<->', lw=0.9, color='k'))
  ax.text(0.2, -1.12, '$\\frac{3}{4}\\delta_x$', fontsize=12, ha='center',
          va='top')
  # Shared face areas, written outside the CVs with a leader line.
  ax.annotate('shared face $P|E_1$:\n$A=\\min(\\frac{\\delta_x}{2},'
              '\\frac{\\delta_x}{2})z=\\frac{\\delta_x}{2}z$',
              xy=(0.5, -0.06), xytext=(1.6, -0.3), fontsize=11,
              arrowprops=dict(arrowstyle='->', color=GRAY))
  ax.annotate('shared face $P|E_2$:\n$A=\\frac{\\delta_x}{2}z$',
              xy=(0.45, -0.5), xytext=(1.6, -1.0), fontsize=11,
              arrowprops=dict(arrowstyle='->', color=GRAY))
  # Exterior faces.
  ax.text(-0.05, -0.25, '$A_1=\\frac{\\delta_x}{2}z$', fontsize=11,
          ha='right', va='center', color=RED)
  ax.text(0.25, 0.05, '$A_4=\\frac{\\delta_x}{2}z$', fontsize=11,
          ha='center', va='bottom', color=RED)
  _arrow(ax, (-0.7, -0.15), (-0.03, -0.15))
  _arrow(ax, (0.2, 0.7), (0.2, 0.03))
  ax.text(-0.72, -0.1, '$Q_{amb,1}$', color=ARROW, fontsize=12)
  ax.text(0.24, 0.55, '$Q_{amb,4}$', color=ARROW, fontsize=12)
  _arrow(ax, (0.9, -0.42), (0.4, -0.42), lw=1.5)
  ax.text(0.93, -0.42, '$Q_{P\\leftarrow E_1}$', color=ARROW, fontsize=11,
          ha='left', va='center')
  _arrow(ax, (0.1, -0.85), (0.1, -0.35), lw=1.5)
  ax.text(0.08, -0.6, '$Q_{P\\leftarrow E_2}$', color=ARROW, fontsize=11,
          rotation=90, ha='right', va='center')
  fig.text(0.01, 0.985,
           'Fig. 4  Corner CV $P$ and its neighbors (uniform $k$, '
           'current code)', fontsize=13, va='top')

  tab = fig.add_axes((0.54, 0.3, 0.45, 0.65))
  tab.axis('off')
  rows = [
      ('face', 'area $A_f$', 'resistance  [m$^2$K/W]', 'conductance [W/K]'),
      ('$P \\leftrightarrow E_1$', '$\\frac{\\delta_x}{2}z$',
       '$\\frac{\\delta_x}{4k}+\\frac{\\delta_x}{2k}=\\frac{3\\delta_x}{4k}$',
       '$\\frac{2}{3}kz$'),
      ('$P \\leftrightarrow E_2$', '$\\frac{\\delta_x}{2}z$',
       '$\\frac{3\\delta_x}{4k}$', '$\\frac{2}{3}kz$'),
      ('$P \\to$ ambient (1)', '$\\frac{\\delta_x}{2}z$',
       '$\\frac{1}{h}+\\frac{\\delta_x}{4k}$',
       '$\\frac{\\delta_x z/2}{1/h+\\delta_x/4k}$'),
      ('$P \\to$ ambient (4)', '$\\frac{\\delta_x}{2}z$',
       '$\\frac{1}{h}+\\frac{\\delta_x}{4k}$',
       '$\\frac{\\delta_x z/2}{1/h+\\delta_x/4k}$'),
      ('$E_1 \\leftrightarrow$ interior', '$\\delta_x z$',
       '$\\frac{3\\delta_x}{4k}$', '$\\frac{4}{3}kz$'),
      ('$E_2 \\leftrightarrow E_3$ (along wall)', '$\\frac{\\delta_x}{2}z$',
       '$\\frac{\\delta_x}{2k}+\\frac{\\delta_x}{2k}$', '$\\frac{1}{2}kz$'),
      ('interior $\\leftrightarrow$ interior', '$\\delta_x z$',
       '$\\frac{\\delta_x}{k}$', '$kz$'),
  ]
  xcols = (0.0, 0.3, 0.47, 0.8)
  yy = 0.95
  for r, row in enumerate(rows):
    for x, cell in zip(xcols, row):
      tab.text(x, yy, cell, fontsize=11 if r == 0 else 13, va='center',
               fontweight='bold' if r == 0 else 'normal')
    tab.plot([0, 1], [yy - 0.05, yy - 0.05], color=GRAY, lw=0.5)
    yy -= 0.105
  tab.text(0.0, yy + 0.01,
           'Checked against Simulator._get_face_conductance / '
           '_get_ambient_conductance\n'
           '(dx = z = 1 m, k = 1, h = 10: 0.667, 0.667, 1.429, 1.429, 1.333, '
           '0.5, 1.0 W/K)', fontsize=9.5, color=GRAY, va='top')

  eq = fig.add_axes((0.0, 0.0, 1.0, 0.3))
  eq.axis('off')
  eq.text(0.01, 0.92, 'Energy balance of the corner CV:', fontsize=12,
          va='top')
  eq.text(0.01, 0.62,
          '$\\frac{\\rho c\\,(\\delta_x/2)(\\delta_x/2)\\,z}{\\Delta t}'
          '\\left(T_P - T_P^{(-)}\\right) = '
          '\\frac{2}{3}kz\\,(T_{E_1}-T_P) + \\frac{2}{3}kz\\,(T_{E_2}-T_P) + '
          '2\\,\\frac{\\delta_x z/2}{1/h + \\delta_x/(4k)}\\,'
          '(T_\\infty - T_P)$', fontsize=17, va='center')
  eq.text(0.01, 0.2,
          '$T_P = \\frac{\\frac{2}{3}kz\\,(T_{E_1}+T_{E_2}) + '
          '\\frac{\\delta_x z}{1/h+\\delta_x/(4k)}\\,T_\\infty + '
          '\\frac{\\rho c\\,\\delta_x^2 z}{4\\Delta t}\\,T_P^{(-)}}'
          '{\\frac{4}{3}kz + \\frac{\\delta_x z}{1/h+\\delta_x/(4k)} + '
          '\\frac{\\rho c\\,\\delta_x^2 z}{4\\Delta t}}$', fontsize=19,
          va='center')
  return fig


# ---------------------------------------------------------------------------
# Figure 5: the three formulations side by side (slide 2 style).
# ---------------------------------------------------------------------------
def figure_comparison():
  fig = plt.figure(figsize=(16, 11.5))
  ax = fig.add_axes((0, 0, 1, 1))
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.set_autoscale_on(False)
  ax.axis('off')
  fig.canvas.draw()
  ax.text(0.01, 0.985,
          'Fig. 5  Corner CV energy balance in three formulations '
          '(uniform $k$, red = term that differs from the current code)',
          fontsize=13, va='top')

  k = 'k'
  y = 0.91
  blocks = [
      ('(a) Paper Eq. 12-13, literally\n(node on surface,\n$L$ = own width)', [
          ('$\\frac{\\rho c\\,uvz}{\\Delta t}(T_P - T_P^{(-)}) = $', k),
          ('$k\\frac{vz}{u}$', RED), ('$(T_{E_1}-T_P) + $', k),
          ('$k\\frac{uz}{v}$', RED), ('$(T_{E_2}-T_P) + $', k),
          ('$h\\,vz$', RED), ('$(T_\\infty-T_P) + $', k),
          ('$h\\,uz$', RED), ('$(T_\\infty-T_P)$', k)], [
              ('$u=v=\\frac{\\delta_x}{2}:\\quad$', k),
              ('$k z$', RED), ('$(T_{E_1}-T_P) + $', k),
              ('$k z$', RED), ('$(T_{E_2}-T_P) + $', k),
              ('$h\\,\\delta_x z$', RED), ('$(T_\\infty-T_P)$', k)]),
      ('(b) Original code\n(copybara_push,\nvertex-centered)', [
          ('$\\frac{\\rho c\\,\\delta_x^2 z}{4\\Delta t}(T_P - T_P^{(-)}) = $',
           k),
          ('$\\frac{k z}{2}$', RED), ('$(T_{E_1}-T_P) + $', k),
          ('$\\frac{k z}{2}$', RED), ('$(T_{E_2}-T_P) + $', k),
          ('$h\\,\\delta_x z$', RED), ('$(T_\\infty-T_P)$', k)], [
              ('code: $T=\\frac{k\\sum T_n + 2h\\delta_x T_\\infty + '
               't_0 T^{(-)}}{2k + 2h\\delta_x + t_0},\\quad '
               't_0=\\frac{\\rho c \\delta_x^2}{2\\Delta t}$'
               '   (whole equation $\\times 2/z$)', GRAY)]),
      ('(c) Current code\n(heatbalance_fix,\ncell-centered FVM)', [
          ('$\\frac{\\rho c\\,\\delta_x^2 z}{4\\Delta t}(T_P - T_P^{(-)}) = $',
           k),
          ('$\\frac{2kz}{3}$', GREEN), ('$(T_{E_1}-T_P) + $', k),
          ('$\\frac{2kz}{3}$', GREEN), ('$(T_{E_2}-T_P) + $', k),
          ('$\\frac{\\delta_x z}{1/h + \\delta_x/4k}$', GREEN),
          ('$(T_\\infty-T_P)$', k)], [
              ('general: $\\sum_{f\\in I_P} A_f\\Gamma_f(T_{N_f}-T_P) + '
               '\\sum_{f\\in B_P} A_f\\Gamma_{\\infty,f}(T_\\infty-T_P) = '
               '\\frac{\\rho c\\,uvz}{\\Delta t}(T_P-T_P^{(-)})$', GRAY)]),
  ]
  for label, line1, line2 in blocks:
    ax.text(0.01, y, label, fontsize=11.5, va='top')
    _colored_line(ax, 0.17, y - 0.035, line1, fontsize=17)
    _colored_line(ax, 0.17, y - 0.1, line2, fontsize=15)
    ax.plot([0.01, 0.99], [y - 0.14, y - 0.14], color=SOLID, lw=1)
    y -= 0.175

  # Face consistency between an edge CV and its inward interior neighbor.
  ax.text(0.01, y, 'Face consistency, edge CV $E$ $\\leftrightarrow$ '
          'interior CV $I$ (same face, $A = \\delta_x z$):', fontsize=12.5,
          va='top')
  y -= 0.06
  rows = [
      ('(a) paper', '$q_{E\\leftarrow I} = k\\frac{\\delta_x z}{\\delta_x/2}'
       '(T_I-T_E) = 2kz\\,\\Delta T$',
       '$q_{I\\leftarrow E} = k\\frac{\\delta_x z}{\\delta_x}(T_E-T_I) = '
       'kz\\,\\Delta T$', 'not equal and opposite: energy is created', RED),
      ('(b) original', '$\\frac{k z}{1}\\Delta T$ (vertex, distance '
       '$\\delta_x$)', '$kz\\,\\Delta T$',
       'consistent only for uniform $k$', ARROW),
      ('(c) current', '$\\frac{\\delta_x z}{\\delta_x/4k_E + \\delta_x/2k_I}'
       '\\Delta T$', 'same expression seen from $I$',
       'conservative for any $k_E, k_I$', GREEN),
  ]
  for name, a, b, verdict, color in rows:
    ax.text(0.02, y, name, fontsize=12, va='center')
    ax.text(0.13, y, a, fontsize=15, va='center')
    ax.text(0.45, y, b, fontsize=15, va='center')
    ax.text(0.72, y, verdict, fontsize=12, va='center', color=color)
    y -= 0.06

  y -= 0.01
  ax.text(0.01, y,
          'With different materials (e.g. wall $k_w$ next to air $k_a$) the '
          'paper / original form gives', fontsize=12.5, va='top')
  ax.text(0.03, y - 0.075,
          '$q_{w\\to a} = A_f\\frac{k_w}{u_w}(T_a-T_w)\\quad\\neq\\quad '
          'q_{a\\to w} = A_f\\frac{k_a}{u_a}(T_a-T_w)$', fontsize=17,
          va='center', color=RED)
  ax.text(0.55, y - 0.075,
          '$\\Rightarrow\\quad A_f\\left(\\frac{u_w}{2k_w}+\\frac{u_a}{2k_a}'
          '\\right)^{-1}(T_a-T_w)$  for both', fontsize=17, va='center',
          color=GREEN)
  return fig


# ---------------------------------------------------------------------------
# Figure 6: one unified equation, tensorized (paper Eq. 13-24 / TF code).
# ---------------------------------------------------------------------------
def figure_tensor():
  fig = plt.figure(figsize=(16, 9.0))
  ax = fig.add_axes((0, 0, 1, 1))
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.set_autoscale_on(False)
  ax.axis('off')
  ax.text(0.01, 0.98,
          'Fig. 6  One equation for every CV: face masks and shifted tensors '
          '(paper Eq. 13-24, tf_simulator.py)', fontsize=13, va='top')

  # The masks of the three CV classes, face by face.
  gx, gy = 0.04, 0.6
  for c, (name, cond, conv) in enumerate([
      ('Interior', (1, 1, 1, 1), (0, 0, 0, 0)),
      ('Edge (left)', (0, 1, 1, 1), (1, 0, 0, 0)),
      ('Corner (top-left)', (0, 1, 1, 0), (1, 0, 0, 1)),
  ]):
    x = gx + c * 0.32
    sq = 0.11
    ax.add_patch(patches.Rectangle((x + 0.06, gy), sq, sq * 1.6,
                                   transform=ax.transAxes, fc=SOLID,
                                   ec='k'))
    ax.text(x + 0.06 + sq / 2, gy + sq * 1.6 + 0.07, name, ha='center',
            fontsize=12, transform=ax.transAxes)
    # Face order of the paper: left 1, bottom 2, right 3, top 4.
    pos = {
        0: (x + 0.06 - 0.005, gy + sq * 0.8, 'right', 'center'),
        1: (x + 0.06 + sq / 2, gy - 0.01, 'center', 'top'),
        2: (x + 0.06 + sq + 0.005, gy + sq * 0.8, 'left', 'center'),
        3: (x + 0.06 + sq / 2, gy + sq * 1.6 + 0.01, 'center', 'bottom'),
    }
    for f in range(4):
      px, py, ha, va = pos[f]
      if conv[f]:
        txt, color = f'$k_{f+1}=0$\n$h_{f+1}=h$', RED
      else:
        txt, color = f'$k_{f+1}=k$\n$h_{f+1}=0$', SOLID
      ax.text(px, py, txt, ha=ha, va=va, fontsize=10.5, color=color,
              transform=ax.transAxes)
    sides = []
    if conv[0]:
      sides.append(((x + 0.06, gy), (x + 0.06, gy + sq * 1.6)))
    if conv[3]:
      sides.append(((x + 0.06, gy + sq * 1.6),
                    (x + 0.06 + sq, gy + sq * 1.6)))
    for p0, p1 in sides:
      ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=RED, lw=4,
              transform=ax.transAxes)

  ax.text(0.01, 0.5, 'Paper Eq. 13 (every face carries both a conduction '
          'and a convection term; masks switch one off):', fontsize=12,
          va='top')
  ax.text(0.02, 0.4,
          '$T_{i,j} = \\frac{Q_x + vz\\left(\\frac{k_1}{u}T_{i-1,j} + '
          'h_1T_\\infty + \\frac{k_3}{u}T_{i+1,j} + h_3T_\\infty\\right) + '
          'uz\\left(\\frac{k_2}{v}T_{i,j-1} + h_2T_\\infty + '
          '\\frac{k_4}{v}T_{i,j+1} + h_4T_\\infty\\right) + '
          '\\frac{c\\rho uvz}{\\Delta t}T^{(-)}_{i,j}}'
          '{vz\\left(\\frac{k_1}{u}+h_1+\\frac{k_3}{u}+h_3\\right) + '
          'uz\\left(\\frac{k_2}{v}+h_2+\\frac{k_4}{v}+h_4\\right) + '
          '\\frac{c\\rho uvz}{\\Delta t}}$', fontsize=16, va='center')
  ax.text(0.01, 0.29, 'Current tf_simulator.py (same structure, '
          'conductances replaced by series $\\Gamma$):', fontsize=12,
          va='top')
  ax.text(0.02, 0.19,
          '$\\mathbf{T} = \\frac{\\mathbf{Q}_x + \\mathbf{V}z\\left('
          '\\Gamma_1 S_R(\\mathbf{T}) + \\Gamma_3 S_L(\\mathbf{T}) + '
          '(\\Gamma_{\\infty,1}+\\Gamma_{\\infty,3})T_\\infty\\right) + '
          '\\mathbf{U}z\\left(\\Gamma_4 S_D(\\mathbf{T}) + '
          '\\Gamma_2 S_U(\\mathbf{T}) + '
          '(\\Gamma_{\\infty,2}+\\Gamma_{\\infty,4})T_\\infty\\right) + '
          '\\frac{\\rho c\\,\\mathbf{UV}z}{\\Delta t}\\mathbf{T}^{(-)}}'
          '{\\mathbf{V}z\\left(\\Gamma_1+\\Gamma_3+\\Gamma_{\\infty,1}+'
          '\\Gamma_{\\infty,3}\\right) + \\mathbf{U}z\\left(\\Gamma_2+\\Gamma_4+'
          '\\Gamma_{\\infty,2}+\\Gamma_{\\infty,4}\\right) + '
          '\\frac{\\rho c\\,\\mathbf{UV}z}{\\Delta t}}$', fontsize=16,
          va='center')
  ax.text(0.02, 0.06,
          '$S_{L,R,U,D}$: shift_tensor_left/right/up/down (the neighbor field '
          'without indexing).   $\\Gamma_f = 0$ where face $f$ opens onto the '
          'ambient, $\\Gamma_{\\infty,f} = 0$ where it has a neighbor, so '
          'each face is counted exactly once.\n'
          'All products are element-wise.  $\\Gamma_f$ already includes the '
          'width ratio $\\min(w_P,w_N)/w_P$ (get_oriented_face_conductance_'
          'tensors).', fontsize=10.5, color=GRAY, va='top')
  return fig


def main():
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  figures = [
      ('fig1_grid_cv_classes', figure_grid),
      ('fig2_cv_types', figure_cv_types),
      ('fig3_series_resistance', figure_resistance),
      ('fig4_corner_detail', figure_corner_detail),
      ('fig5_formulation_comparison', figure_comparison),
      ('fig6_tensorized_equation', figure_tensor),
  ]
  for name, build in figures:
    fig = build()
    fig.savefig(OUT_DIR / f'{name}.png', dpi=130)
    plt.close(fig)
    print('wrote', OUT_DIR / f'{name}.png')


if __name__ == '__main__':
  main()
