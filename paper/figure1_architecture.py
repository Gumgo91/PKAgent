# -*- coding: utf-8 -*-
"""Figure 1: architecture of PKAgent as a three-layer stack.

The language model (top), the PKAgent session with its tools in functional groups (middle) and the
open-source PKPy2 engine (bottom), with the task and the data file as inputs on the left and the
outputs of a run on the right. Vector drawing with matplotlib primitives only, created at the
printed size (7.0 x 3.86 in, double column, Arial, text >= 8 pt, strokes 0.5 to 1.0 pt) and never
rescaled.

Numbers come from code: the tool groups, the badge counts and the tool total from
paper/tool_groups.py (asserted against src/pkagent/tools.py); the per-run limits from
pkagent.config.Budget(). The language model is told the limits on fits, responses and hours (Task
box); the fee limit is enforced by the session, which stops a run when the LLM fees exceed it (the
model sees only the fees spent), so it is shown with the session's Budgets item.

Outputs paper/figures/Figure_1.pdf (vector), Figure_1.png (600 dpi) and Figure_1.tiff (CMYK, via
benchmarks/figures.py:tiff_cmyk). Running it also prints a text audit (every text artist with its
size, the minimum size, the word count, text overlaps and clipping).
"""
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import (Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon,
                                Rectangle)
from matplotlib.text import Text

HERE = Path(__file__).resolve().parent
OUT = HERE / 'figures'
sys.path.insert(0, str(HERE.parent / 'benchmarks'))
sys.path.insert(0, str(HERE.parent / 'src'))
sys.path.insert(0, str(HERE))
from tool_groups import N_TOOLS, TOOL_GROUPS          # noqa: E402
from pkagent.config import Budget                     # noqa: E402

BUDGET = Budget()                                     # the per-run limits of every benchmark run

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.unicode_minus': False,
})

# ---------------------------------------------------------------- palette
TEXT = '#1F2933'
BLUE, BLUE_T = '#4A7FB5', '#E8F1FB'
TEAL, TEAL_T = '#2B7A8C', '#E3F1F4'      # language model (orange #C8801E is reserved for the expert statement)
GREEN, GREEN_T, GREEN_M = '#4C8C4A', '#EEF6EE', '#D3E8D2'
PURPLE, PURPLE_T, PURPLE_M = '#7A5BA6', '#F3EEF8', '#DCD0EC'
GRAY, GRAY_T = '#7B8794', '#F4F5F7'
WHITE = '#FFFFFF'

LW = 1.0        # block outlines, arrows and the heaviest pictogram strokes (pt; CPT: 0.5 to 1 pt)
LW_ICON = 0.8   # pictogram strokes
LW_THIN = 0.6   # thin rules and pictogram details
LW_RANGE = (0.5, 1.0)

# ---------------------------------------------------------------- canvas (1 data unit = 1 inch)
W, H = 7.0, 3.86
fig = plt.figure(figsize=(W, H), facecolor=WHITE)
ax = fig.add_axes((0, 0, 1, 1))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_axis_off()
ax.set_facecolor(WHITE)
BOXES = []      # every drawn box, for the text-clearance check
ARROWS = []     # every block-to-block arrow (tail, tip, head length, head half-width in inches)


# ---------------------------------------------------------------- helpers
def measure(s, size=8, weight='normal'):
    """Width of a one-line string in inches at print size."""
    t = ax.text(0, 0, s, fontsize=size, fontweight=weight)
    w = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def txt(x, y, s, size=8, weight='normal', ha='left', va='center', gid=None, **kw):
    t = ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va, color=TEXT, zorder=20, **kw)
    if gid:
        t.set_gid(gid)
    return t


def rbox(x, y, w, h, fc, ec, lw=LW, r=0.07, z=1):
    BOXES.append((x, y, x + w, y + h))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec,
                                lw=lw, zorder=z))


def ln(xs, ys, c, lw=LW_ICON, z=8, ls='-'):
    ax.add_line(Line2D(xs, ys, color=c, lw=lw, zorder=z, linestyle=ls, solid_capstyle='round',
                       solid_joinstyle='round', dash_capstyle='butt'))


def arrow(p1, p2, c, both=False, lw=LW, hl=5.0, hw=2.6, z=9, track=False):
    style = ('<|-|>' if both else '-|>') + f',head_length={hl},head_width={hw}'
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=1, color=c, lw=lw, shrinkA=0,
                                 shrinkB=0, zorder=z, joinstyle='miter', capstyle='butt'))
    if track:
        ARROWS.append((p1, p2, hl / 72, hw / 72, lw / 72))


def link(p1, p2, c):
    """Block-to-block arrow, recorded for the label-clearance check."""
    arrow(p1, p2, c, track=True)


def vdivider(x, y0, y1, c):
    ln([x, x], [y0, y1], c, LW_THIN, 3)


# ---------------------------------------------------------------- pictograms
def icon_doc(cx, cy, w, h, c, fill=WHITE, nlines=3, check=False, z=8):
    """Page with a folded corner, text lines and an optional check mark."""
    x0, y0, f = cx - w / 2, cy - h / 2, min(w, h) * 0.32
    ax.add_patch(Polygon([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h - f), (x0 + w - f, y0 + h), (x0, y0 + h)],
                         closed=True, fc=fill, ec=c, lw=LW_ICON, joinstyle='round', zorder=z))
    ln([x0 + w - f, x0 + w - f, x0 + w], [y0 + h, y0 + h - f, y0 + h - f], c, LW_THIN, z + 0.1)
    top = y0 + h - f * 1.15
    step = (h - f * 1.15) / (nlines + 1.3)
    for i in range(nlines):
        y = top - (i + 0.8) * step
        x1 = x0 + w * (0.78 if i < nlines - 1 else 0.55)
        if check and i == nlines - 1:
            x1 = x0 + w * 0.42
        ln([x0 + w * 0.2, x1], [y, y], c, LW_THIN, z + 0.1)
    if check:
        s = w * 0.30
        bx, by = x0 + w * 0.62, y0 + h * 0.22
        ln([bx - s * 0.5, bx - s * 0.1, bx + s * 0.6], [by + s * 0.05, by - s * 0.35, by + s * 0.55], c, LW,
           z + 0.2)


def icon_table(cx, cy, w, h, c, fill=WHITE, rows=3, cols=3, z=8):
    x0, y0 = cx - w / 2, cy - h / 2
    hh = h / (rows + 1)
    ax.add_patch(Rectangle((x0, y0), w, h, fc=fill, ec=c, lw=LW_ICON, zorder=z))
    ax.add_patch(Rectangle((x0, y0 + h - hh), w, hh, fc=c, ec=c, lw=LW_ICON, zorder=z + 0.1))
    for i in range(1, rows):
        ln([x0, x0 + w], [y0 + i * hh] * 2, c, LW_THIN, z + 0.1)
    for j in range(1, cols):
        ln([x0 + j * w / cols] * 2, [y0, y0 + h - hh], c, LW_THIN, z + 0.1)


def icon_datafile(cx, cy, w, h, c, fill=WHITE, z=8):
    """File outline with a small table inside."""
    x0, y0, f = cx - w / 2, cy - h / 2, w * 0.30
    ax.add_patch(Polygon([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h - f), (x0 + w - f, y0 + h), (x0, y0 + h)],
                         closed=True, fc=fill, ec=c, lw=LW, joinstyle='round', zorder=z))
    ln([x0 + w - f, x0 + w - f, x0 + w], [y0 + h, y0 + h - f, y0 + h - f], c, LW_THIN, z + 0.1)
    icon_table(cx, y0 + h * 0.40, w * 0.62, h * 0.46, c, fill=WHITE, z=z + 0.2)


def icon_compartments(cx, cy, c, s=1.0, z=8):
    """Two-compartment PK model: dose in, exchange, elimination out."""
    b1, b2 = 0.12 * s, 0.092 * s
    c1x, c2x = cx - 0.105 * s, cx + 0.12 * s
    ax.add_patch(Rectangle((c1x - b1 / 2, cy - b1 / 2), b1, b1, fc=GREEN_M, ec=c, lw=LW_ICON, zorder=z))
    ax.add_patch(Rectangle((c2x - b2 / 2, cy - b2 / 2), b2, b2, fc=WHITE, ec=c, lw=LW_ICON, zorder=z))
    gx0, gx1 = c1x + b1 / 2 + 0.012, c2x - b2 / 2 - 0.012
    kw = dict(lw=LW_THIN, hl=2.6, hw=1.4, z=z)
    arrow((gx0, cy + 0.02 * s), (gx1, cy + 0.02 * s), c, **kw)
    arrow((gx1, cy - 0.02 * s), (gx0, cy - 0.02 * s), c, **kw)
    arrow((c1x, cy + b1 / 2 + 0.085 * s), (c1x, cy + b1 / 2 + 0.008), c, **kw)
    arrow((c1x, cy - b1 / 2 - 0.008), (c1x, cy - b1 / 2 - 0.085 * s), c, **kw)


def icon_gof(cx, cy, w, h, c, z=8):
    """Observed versus predicted scatter with identity line."""
    x0, y0 = cx - w / 2, cy - h / 2
    ln([x0, x0, x0 + w], [y0 + h, y0, y0], c, LW_ICON, z)
    ln([x0 + 0.02, x0 + w - 0.01], [y0 + 0.02, y0 + h - 0.01], c, LW_THIN, z, ls=(0, (2.2, 1.4)))
    for px, py in [(0.18, 0.26), (0.30, 0.22), (0.38, 0.47), (0.52, 0.44), (0.60, 0.70), (0.74, 0.62),
                   (0.84, 0.88)]:
        ax.add_patch(Circle((x0 + px * w, y0 + py * h), 0.016, fc=c, ec='none', zorder=z + 0.1))


def icon_forest(cx, cy, w, h, c, z=8):
    """Forest plot: reference line and three effects with intervals."""
    x0, y0 = cx - w / 2, cy - h / 2
    ln([cx, cx], [y0, y0 + h], c, LW_THIN, z, ls=(0, (2.0, 1.4)))
    for fy, a, b, m in [(0.84, 0.30, 0.90, 0.62), (0.50, 0.10, 0.62, 0.36), (0.16, 0.42, 0.80, 0.58)]:
        y = y0 + fy * h
        ln([x0 + a * w, x0 + b * w], [y, y], c, LW_ICON, z + 0.1)
        s = 0.038
        ax.add_patch(Rectangle((x0 + m * w - s / 2, y - s / 2), s, s, fc=c, ec=c, lw=LW_THIN, zorder=z + 0.2))


def icon_network(cx, cy, w, h, c, z=8):
    """Small neural network: 2-3-2 nodes."""
    xs = [cx - w / 2, cx, cx + w / 2]
    nodes = []
    for x, n in zip(xs, [2, 3, 2]):
        ys = np.linspace(cy - h / 2 * (n - 1) / 2, cy + h / 2 * (n - 1) / 2, n)
        nodes.append([(x, y) for y in ys])
    for a, b in zip(nodes[:-1], nodes[1:]):
        for p in a:
            for q in b:
                ln([p[0], q[0]], [p[1], q[1]], c, LW_THIN, z)
    for layer in nodes:
        for p in layer:
            ax.add_patch(Circle(p, 0.027, fc=WHITE, ec=c, lw=LW_ICON, zorder=z + 1))


def icon_gear(cx, cy, R, c, fill, z=8, n=8):
    r_in = R * 0.74
    pts = []
    for i in range(n):
        a = 2 * np.pi * i / n
        tw = np.pi / n * 0.55
        for ang, rad in [(a - tw - 0.07, r_in), (a - tw + 0.03, R), (a + tw - 0.03, R), (a + tw + 0.07, r_in)]:
            pts.append((cx + rad * np.cos(ang), cy + rad * np.sin(ang)))
    ax.add_patch(Polygon(pts, closed=True, fc=fill, ec=c, lw=LW_ICON, joinstyle='round', zorder=z))
    ax.add_patch(Circle((cx, cy), R * 0.32, fc=WHITE, ec=c, lw=LW_ICON, zorder=z + 1))


def icon_nocode(cx, cy, r, c_ring, c_glyph, z=8):
    """'</>' glyph under a prohibition ring."""
    ax.add_patch(Circle((cx, cy), r, fc=WHITE, ec=c_ring, lw=LW, zorder=z))
    g = r * 0.50
    ln([cx - g * 0.45, cx - g * 1.15, cx - g * 0.45], [cy + g * 0.6, cy, cy - g * 0.6], c_glyph, 0.9, z + 1)
    ln([cx + g * 0.45, cx + g * 1.15, cx + g * 0.45], [cy + g * 0.6, cy, cy - g * 0.6], c_glyph, 0.9, z + 1)
    ln([cx - g * 0.22, cx + g * 0.22], [cy - g * 0.75, cy + g * 0.75], c_glyph, 0.9, z + 1)
    d = r * np.cos(np.pi / 4)
    ln([cx - d, cx + d], [cy + d, cy - d], c_ring, LW, z + 2)


def icon_bubble(cx, cy, w, h, c, fill=WHITE, z=8):
    x0, y0 = cx - w / 2, cy - h / 2 + h * 0.18
    hb = h * 0.82
    ax.add_patch(FancyBboxPatch((x0, y0), w, hb, boxstyle='round,pad=0,rounding_size=0.03', fc=fill, ec=c,
                                lw=LW_ICON, zorder=z))
    ax.add_patch(Polygon([(x0 + w * 0.22, y0 + 0.004), (x0 + w * 0.18, cy - h / 2), (x0 + w * 0.46, y0 + 0.004)],
                         closed=True, fc=fill, ec='none', zorder=z + 0.1))
    ln([x0 + w * 0.22, x0 + w * 0.18, x0 + w * 0.46], [y0, cy - h / 2, y0], c, LW_ICON, z + 0.2)
    for fy, fx in [(0.66, 0.78), (0.36, 0.55)]:
        ln([x0 + w * 0.2, x0 + w * fx], [y0 + hb * fy] * 2, c, LW_THIN, z + 0.2)


def icon_gauge(cx, cy, r, c, z=8):
    """Half dial with needle; cy is the dial's base line."""
    ax.add_patch(Arc((cx, cy), 2 * r, 2 * r, theta1=0, theta2=180, ec=c, lw=LW_ICON, zorder=z))
    ln([cx - r, cx + r], [cy, cy], c, LW_ICON, z)
    a = np.deg2rad(52)
    ln([cx, cx + 0.78 * r * np.cos(a)], [cy, cy + 0.78 * r * np.sin(a)], c, LW_ICON, z + 0.1)
    ax.add_patch(Circle((cx, cy), r * 0.16, fc=c, ec='none', zorder=z + 0.2))


def icon_versions(cx, cy, w, h, c, z=8):
    """Versioned data: three stacked sheets."""
    d = 0.024
    for k in (2, 1, 0):
        ax.add_patch(Rectangle((cx - w / 2 + k * d, cy - h / 2 + k * d), w - 2 * d, h - 2 * d, fc=WHITE, ec=c,
                               lw=LW_THIN, zorder=z + (2 - k) * 0.1))


def icon_registry(cx, cy, w, h, c, z=8):
    """Model registry: a database cylinder."""
    e = h * 0.30
    x0, x1, yb, yt = cx - w / 2, cx + w / 2, cy - h / 2 + e / 2, cy + h / 2 - e / 2
    ax.add_patch(Rectangle((x0, yb), w, yt - yb, fc=WHITE, ec='none', zorder=z))
    ax.add_patch(Arc((cx, yb), w, e, theta1=180, theta2=360, ec=c, lw=LW_THIN, zorder=z + 0.1))
    ax.add_patch(Arc((cx, (yb + yt) / 2), w, e, theta1=180, theta2=360, ec=c, lw=LW_THIN, zorder=z + 0.1))
    ax.add_patch(Ellipse((cx, yb), w, e, fc=WHITE, ec='none', zorder=z - 0.1))
    ln([x0, x0], [yb, yt], c, LW_THIN, z + 0.1)
    ln([x1, x1], [yb, yt], c, LW_THIN, z + 0.1)
    ax.add_patch(Ellipse((cx, yt), w, e, fc=GREEN_M, ec=c, lw=LW_THIN, zorder=z + 0.2))


def icon_log(cx, cy, w, h, c, z=8):
    """Log: vertical timeline with three timed entries."""
    x = cx - w / 2 + 0.016
    ln([x, x], [cy - h / 2, cy + h / 2], c, LW_THIN, z)
    for i in range(3):
        y = cy + h / 2 - (i + 0.5) * h / 3
        ax.add_patch(Circle((x, y), 0.015, fc=c, ec='none', zorder=z + 0.1))
        ln([x + 0.03, cx + w / 2 - (0.03 if i == 1 else 0)], [y, y], c, LW_THIN, z)


def icon_audit(cx, cy, w, h, c, z=8):
    """Document with a magnifier."""
    icon_doc(cx - w * 0.12, cy + h * 0.04, w * 0.72, h * 0.92, c, nlines=3, z=z)
    r = w * 0.20
    mx, my = cx + w * 0.16, cy - h * 0.12
    ax.add_patch(Circle((mx, my), r, fc=WHITE, ec=c, lw=LW_ICON, zorder=z + 1))
    ln([mx + r * 0.72, mx + r * 1.75], [my - r * 0.72, my - r * 1.75], c, LW, z + 1)


def icon_clockcoin(cx, cy, s, c, fill, z=8):
    """Coin behind a clock: tokens, fees and time."""
    rc = s * 0.30
    ax.add_patch(Circle((cx - s * 0.17, cy - s * 0.10), rc, fc=fill, ec=c, lw=LW_ICON, zorder=z))
    ax.add_patch(Circle((cx - s * 0.17, cy - s * 0.10), rc * 0.62, fc='none', ec=c, lw=LW_THIN, zorder=z + 0.1))
    rk = s * 0.32
    kx, ky = cx + s * 0.13, cy + s * 0.08
    ax.add_patch(Circle((kx, ky), rk, fc=WHITE, ec=c, lw=LW_ICON, zorder=z + 1))
    ln([kx, kx], [ky, ky + rk * 0.62], c, LW_ICON, z + 1.1)
    ln([kx, kx + rk * 0.48], [ky, ky - rk * 0.18], c, LW_ICON, z + 1.1)


def icon_objective(cx, cy, w, h, c, z=8):
    """Objective function curve with its minimum marked."""
    x0, y0 = cx - w / 2, cy - h / 2
    ln([x0, x0, x0 + w], [y0 + h, y0, y0], c, LW_ICON, z)
    u = np.linspace(0.08, 0.96, 60)
    v = 0.18 + 0.78 * ((u - 0.56) / 0.48) ** 2
    ln(x0 + u * w, y0 + v * h, c, LW_ICON, z + 0.1)
    mx, my = x0 + 0.56 * w, y0 + 0.18 * h
    ln([mx, mx], [y0, my], c, LW_THIN, z, ls=(0, (1.6, 1.2)))
    ax.add_patch(Circle((mx, my), 0.022, fc=c, ec='none', zorder=z + 0.2))


def icon_residuals(cx, cy, w, h, c, z=8):
    """Residuals scattered around a zero line."""
    x0, y0 = cx - w / 2, cy - h / 2
    ln([x0, x0], [y0, y0 + h], c, LW_ICON, z)
    ln([x0, x0 + w], [cy, cy], c, LW_THIN, z, ls=(0, (2.2, 1.4)))
    for px, py in [(0.14, 0.30), (0.26, -0.22), (0.38, 0.12), (0.50, -0.36), (0.62, 0.34), (0.74, -0.10),
                   (0.86, 0.22), (0.94, -0.28)]:
        ax.add_patch(Circle((x0 + px * w, cy + py * h), 0.016, fc=c, ec='none', zorder=z + 0.1))


def icon_vpc(cx, cy, w, h, c, z=8):
    """Visual predictive check: prediction band around a median profile."""
    x0, y0 = cx - w / 2, cy - h / 2
    ln([x0, x0, x0 + w], [y0 + h, y0, y0], c, LW_ICON, z)
    u = np.linspace(0.06, 0.98, 50)
    med = 0.12 + 0.62 * (np.exp(-2.2 * u) - np.exp(-9.0 * u)) / 0.62
    half = 0.07 + 0.16 * med
    lo, hi = med - half, med + half
    pts = list(zip(x0 + u * w, y0 + hi * h)) + list(zip(x0 + u[::-1] * w, y0 + lo[::-1] * h))
    ax.add_patch(Polygon(pts, closed=True, fc=PURPLE_M, ec='none', zorder=z - 0.1))
    ln(x0 + u * w, y0 + hi * h, c, LW_THIN, z)
    ln(x0 + u * w, y0 + lo * h, c, LW_THIN, z)
    ln(x0 + u * w, y0 + med * h, c, LW_ICON, z + 0.1)


def badge(cx, cy, n, r=0.072):
    """Small circled tool count on a tile corner."""
    ax.add_patch(Circle((cx, cy), r, fc=GREEN_M, ec=GREEN, lw=LW_ICON, zorder=12))
    txt(cx, cy - 0.003, str(n), 8, 'bold', ha='center', gid='badge')


# ================================================================ layout (inches)
LX0, LX1 = 0.04, 1.52           # inputs column
SX0, SX1 = 1.76, 5.44           # session column (the stack)
OX0, OX1 = 5.84, 6.96           # outputs column
SCX = (SX0 + SX1) / 2
EN_Y0, EN_Y1 = 0.04, 0.58       # PKPy2 engine row
SE_Y0, SE_Y1 = 0.80, 2.16       # PKAgent session row
LM_Y0, LM_Y1 = 2.48, 3.82       # language model row
SE_MID = (SE_Y0 + SE_Y1) / 2
LM_MID = (LM_Y0 + LM_Y1) / 2
HEAD = 0.15                     # header centre below the top edge of a block
PADX = 0.10                     # header inset from the left edge of a block


def draw():
    # ------------------------------------------------------------ Task (input, aligned with the language model)
    top = LM_Y1
    rbox(LX0, LM_Y0, LX1 - LX0, LM_Y1 - LM_Y0, BLUE_T, BLUE)
    txt(LX0 + PADX, top - HEAD, 'Task', 10, 'bold')
    ix, tx = LX0 + 0.18, LX0 + 0.33
    icon_doc(ix, top - 0.37, 0.15, 0.19, BLUE)
    txt(tx, top - 0.37, 'Study description')
    icon_bubble(ix, top - 0.62, 0.19, 0.17, BLUE)
    txt(tx, top - 0.55, 'Expert statement')
    txt(tx, top - 0.685, '(optional)')
    icon_gauge(ix, top - 0.895, 0.085, BLUE)
    txt(tx, top - 0.865, 'Budget per run')
    # budget strip: the limits the language model is told (prompts.task); the fee limit is the session's (below)
    sx0, sx1, sy1 = LX0 + 0.05, LX1 - 0.05, top - 0.945
    sy0 = sy1 - 0.32
    cells = [(f'{BUDGET.max_fits}', 'fits'), (f'{BUDGET.max_turns}', 'responses'), (f'{BUDGET.max_hours:g}', 'h')]
    need = [max(measure(n, 10, 'bold'), measure(u, 8)) for n, u in cells]
    extra = (sx1 - sx0 - sum(need)) / len(cells)
    rbox(sx0, sy0, sx1 - sx0, sy1 - sy0, WHITE, BLUE, lw=LW_THIN, r=0.04, z=4)
    x = sx0
    for k, ((num, unit), wd) in enumerate(zip(cells, need)):
        cw = wd + extra
        if k:
            ln([x, x], [sy0 + 0.06, sy1 - 0.06], BLUE, LW_THIN, 5)
        if unit:
            txt(x + cw / 2, sy0 + 0.218, num, 10, 'bold', ha='center', gid='strip')
            txt(x + cw / 2, sy0 + 0.083, unit, 8, ha='center', gid='strip')
        else:
            txt(x + cw / 2, (sy0 + sy1) / 2, num, 10, 'bold', ha='center', gid='strip')
        x += cw
    link((LX1, LM_MID), (SX0, LM_MID), BLUE)

    # ------------------------------------------------------------ Data file (input, centred on the session)
    rbox(LX0, SE_Y0, LX1 - LX0, SE_Y1 - SE_Y0, BLUE_T, BLUE)
    txt(LX0 + PADX, SE_Y1 - HEAD, 'Data file', 10, 'bold')
    icon_datafile((LX0 + LX1) / 2, SE_MID, 0.50, 0.62, BLUE, fill=WHITE)
    link((LX1, SE_MID), (SX0, SE_MID), BLUE)

    # ------------------------------------------------------------ language model layer
    rbox(SX0, LM_Y0, OX1 - SX0, LM_Y1 - LM_Y0, TEAL_T, TEAL)
    txt(SX0 + PADX, LM_Y1 - HEAD, 'Language model', 10, 'bold')
    c_top, c_bot = LM_Y1 - 0.30, LM_Y0 + 0.06    # content band under the header
    cy = (c_top + c_bot) / 2
    d1, d2 = SX0 + 1.52, OX0                      # cell dividers, clear of the arrows below
    vdivider(d1, c_bot + 0.08, c_top - 0.02, TEAL)
    vdivider(d2, c_bot + 0.08, c_top - 0.02, TEAL)
    # models
    icon_network(SX0 + PADX + 0.16, cy, 0.30, 0.34, TEAL)
    txt(SX0 + PADX + 0.40, cy + 0.12, 'GPT-6.1 Sol', 9)
    txt(SX0 + PADX + 0.40, cy - 0.12, 'Claude Opus 5.5', 9)
    # system prompt with the decision criteria
    tw = 1.10                                     # width of the criteria table
    blk = 0.36 + tw
    bx = (d1 + d2) / 2 - blk / 2
    icon_doc(bx + 0.12, cy + 0.13, 0.23, 0.30, TEAL)
    px = bx + 0.36
    txt(px, cy + 0.32, 'System prompt', 9, 'bold')
    txt(px + tw, cy + 0.12, 'OFV decrease', ha='right')
    ln([px, px + tw], [cy + 0.035, cy + 0.035], TEAL, LW_THIN, 3)
    txt(px, cy - 0.08, 'Add')
    txt(px + tw, cy - 0.08, '≥ 3.84', weight='bold', ha='right')
    txt(px, cy - 0.27, 'Keep')
    txt(px + tw, cy - 0.27, '≥ 6.63', weight='bold', ha='right')
    # never writes or runs code
    ncx = (OX0 + OX1) / 2
    icon_nocode(ncx, cy + 0.10, 0.18, TEAL, TEXT)
    txt(ncx, cy - 0.25, 'No code', ha='center')

    # ------------------------------------------------------------ PKAgent session layer
    rbox(SX0, SE_Y0, SX1 - SX0, SE_Y1 - SE_Y0, GREEN_T, GREEN)
    txt(SX0 + PADX, SE_Y1 - HEAD, 'PKAgent session', 10, 'bold')
    txt(SX1 - PADX, SE_Y1 - HEAD, f'{N_TOOLS} tools', 9, 'bold', ha='right')
    icons = {'Data': 'table', 'Models': 'comp', 'Diagnostics': 'gof', 'Covariates and uncertainty': 'forest',
             'Report': 'report'}
    assert list(icons) == list(TOOL_GROUPS), 'a tool group of paper/tool_groups.py has no pictogram'
    assert sum(len(v) for v in TOOL_GROUPS.values()) == N_TOOLS
    groups = [(g.replace(' and ', ' and\n'), len(names), icons[g]) for g, names in TOOL_GROUPS.items()]
    t_y1 = SE_Y1 - 0.33
    t_y0 = t_y1 - 0.63
    pad, gap = 0.08, 0.06
    need = [max(max(measure(s, 8, 'bold') for s in n.split('\n')) + 0.10, 0.52) for n, _, _ in groups]
    spare = (SX1 - SX0 - 2 * pad - gap * (len(groups) - 1) - sum(need)) / len(groups)
    x = SX0 + pad
    for (name, n, kind), wneed in zip(groups, need):
        tw_ = wneed + spare
        rbox(x, t_y0, tw_, t_y1 - t_y0, WHITE, GREEN, lw=LW_ICON, r=0.05, z=2)
        cx, icy = x + tw_ / 2, t_y1 - 0.215
        if kind == 'table':
            icon_table(cx, icy, 0.30, 0.22, GREEN)
        elif kind == 'comp':
            icon_compartments(cx, icy - 0.005, GREEN)
        elif kind == 'gof':
            icon_gof(cx, icy, 0.28, 0.24, GREEN)
        elif kind == 'forest':
            icon_forest(cx, icy, 0.30, 0.23, GREEN)
        else:
            icon_doc(cx, icy, 0.20, 0.26, GREEN, nlines=3, check=True)
        txt(cx, t_y0 + (0.14 if '\n' in name else 0.125), name, 8, 'bold', ha='center', linespacing=1.0)
        badge(x + tw_ - 0.035, t_y1, n)
        x += tw_ + gap
    # what the session keeps
    ln([SX0 + 0.08, SX1 - 0.08], [t_y0 - 0.075] * 2, GREEN, LW_THIN, 3, ls=(0, (2.5, 2)))
    st_y = (t_y0 - 0.075 + SE_Y0) / 2
    # the session counts every budget and stops a run at the fee limit, which the model is not told
    fee = f'(${BUDGET.max_cost_usd:g} fee limit)'
    state = [('Versioned data', 'versions'), ('Model registry', 'registry'), (f'Budgets\n{fee}', 'gauge'),
             ('Log', 'log')]
    iw = 0.15
    widths = [iw + 0.06 + max(measure(t) for t in s.split('\n')) for s, _ in state]
    sgap = (SX1 - SX0 - 0.20 - sum(widths)) / (len(state) - 1)
    x = SX0 + 0.10
    for (s, kind), wd in zip(state, widths):
        icx = x + iw / 2
        if kind == 'versions':
            icon_versions(icx, st_y, iw, 0.13, GREEN)
        elif kind == 'registry':
            icon_registry(icx, st_y, 0.13, 0.15, GREEN)
        elif kind == 'gauge':
            icon_gauge(icx, st_y - 0.035, 0.07, GREEN)
        else:
            icon_log(icx, st_y, iw, 0.13, GREEN)
        txt(x + iw + 0.06, st_y, s, linespacing=1.0)
        x += wd + sgap

    # language model <-> session: two one-way arrows, each in the colour of its source block
    a_dn, a_up = SCX - 0.17, SCX + 0.17
    link((a_dn, LM_Y0), (a_dn, SE_Y1), TEAL)
    link((a_up, SE_Y1), (a_up, LM_Y0), GREEN)
    mid = (LM_Y0 + SE_Y1) / 2
    txt(a_dn - 0.07, mid, 'tool calls', ha='right')
    txt(a_up + 0.07, mid, 'results, plots')

    # ------------------------------------------------------------ PKPy2 engine layer
    rbox(LX0, EN_Y0, SX1 - LX0, EN_Y1 - EN_Y0, PURPLE_T, PURPLE)
    ecy = (EN_Y0 + EN_Y1) / 2
    icon_gear(LX0 + 0.18, ecy, 0.115, PURPLE, PURPLE_M)
    hx = LX0 + 0.36
    txt(hx, ecy + 0.10, 'PKPy2 engine', 10, 'bold')
    ow = measure('open source') + 0.10
    rbox(hx, ecy - 0.10 - 0.08, ow, 0.16, WHITE, PURPLE, lw=LW_THIN, r=0.06, z=6)
    txt(hx + ow / 2, ecy - 0.10, 'open source', ha='center')
    dv = hx + measure('PKPy2 engine', 10, 'bold') + 0.10
    vdivider(dv, EN_Y0 + 0.09, EN_Y1 - 0.09, PURPLE)
    items = [('objective', 'Laplace (FOCE-I type),\nL-BFGS-B'), ('residuals', 'Standard errors,\nCWRES, NPDE'),
             ('vpc', 'VPC\nsimulation')]
    iw, ig = 0.30, 0.07
    widths = [iw + ig + max(measure(s) for s in lab.split('\n')) for _, lab in items]
    x0, x1 = dv + 0.12, SX1 - 0.10
    egap = (x1 - x0 - sum(widths)) / (len(items) - 1)
    x = x0
    for (kind, lab), wd in zip(items, widths):
        icx = x + iw / 2
        if kind == 'objective':
            icon_objective(icx, ecy, iw, 0.26, PURPLE)
        elif kind == 'residuals':
            icon_residuals(icx, ecy, iw, 0.26, PURPLE)
        else:
            icon_vpc(icx, ecy, iw, 0.26, PURPLE)
        txt(x + iw + ig, ecy, lab, linespacing=1.15)
        x += wd + egap

    # session <-> engine: same pair of one-way arrows as above (green requests down, purple results up),
    # on the same x positions, with the single label between them
    link((a_dn, SE_Y0), (a_dn, EN_Y1), GREEN)
    link((a_up, EN_Y1), (a_up, SE_Y0), PURPLE)
    txt(SCX, (SE_Y0 + EN_Y1) / 2, 'fits', ha='center')

    # ------------------------------------------------------------ outputs (span the session and engine rows)
    rbox(OX0, EN_Y0, OX1 - OX0, SE_Y1 - EN_Y0, GRAY_T, GRAY)
    txt(OX0 + PADX, SE_Y1 - HEAD, 'Outputs', 10, 'bold')
    ocx = (OX0 + OX1) / 2
    lh = 0.135                                    # 8-pt line pitch
    og = 0.05                                     # icon to label
    outs = [('doc', ['Final model', 'and report'], 0.27), ('audit', ['Audit trail'], 0.27),
            ('clock', ['Tokens, fees, time'], 0.27)]
    heights = [ih + og + lh * len(lab) for _, lab, ih in outs]
    o_top, o_bot = SE_Y1 - 0.27, EN_Y0 + 0.02
    ogap = (o_top - o_bot - sum(heights)) / len(outs)
    y = o_top - ogap / 2
    for (kind, lab, ih), hh in zip(outs, heights):
        icy = y - ih / 2
        if kind == 'doc':
            icon_doc(ocx, icy, 0.22, ih, GRAY, nlines=3, check=True)
        elif kind == 'audit':
            icon_audit(ocx, icy, 0.30, ih, GRAY)
        else:
            icon_clockcoin(ocx, icy, ih / 0.8, GRAY, GRAY_T)     # drawn extent is 0.8 of its size
        for k, s in enumerate(lab):
            txt(ocx, y - ih - og - lh * (k + 0.5), s, ha='center')
        y -= hh + ogap
    link((SX1, SE_MID), (OX0, SE_MID), GREEN)
    hl = ARROWS[-1][2]                            # centre the label over the shaft, clear of the head
    txt((SX1 + OX0 - hl) / 2, SE_MID + 0.10, 'final', ha='center')


# ================================================================ audit
def audit():
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_text().strip()]
    print(f'text artists: {len(texts)}')
    for t in sorted(texts, key=lambda t: (t.get_fontsize(), t.get_text())):
        print(f'  {t.get_fontsize():4.1f} pt  {t.get_text()!r}')
    sizes = [t.get_fontsize() for t in texts]
    words = [w for t in texts if t.get_gid() != 'badge' for w in t.get_text().split()
             if re.search('[A-Za-z]', w)]
    print(f'min font size {min(sizes):.1f} pt, max {max(sizes):.1f} pt; words {len(words)} '
          f'(tokens with letters, badge digits excluded)')
    bbs = [(t, t.get_window_extent(r)) for t in texts]
    for t, bb in bbs:
        if bb.x0 < 2 or bb.y0 < 2 or bb.x1 > fig.bbox.x1 - 2 or bb.y1 > fig.bbox.y1 - 2:
            print('  CLIP', repr(t.get_text()))
    for i in range(len(bbs)):
        for j in range(i + 1, len(bbs)):
            if bbs[i][1].overlaps(bbs[j][1]):
                print('  OVERLAP', repr(bbs[i][0].get_text()), repr(bbs[j][0].get_text()))
    for t, bb in bbs:                      # clearance from the smallest enclosing box (badges straddle corners)
        if t.get_gid() == 'badge':
            continue
        x0, x1, y0, y1 = bb.x0 / fig.dpi, bb.x1 / fig.dpi, bb.y0 / fig.dpi, bb.y1 / fig.dpi
        hits = [b for b in BOXES if b[0] <= (x0 + x1) / 2 <= b[2] and b[1] <= (y0 + y1) / 2 <= b[3]]
        if hits:
            b = min(hits, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            m = min(x0 - b[0], b[2] - x1, y0 - b[1], b[3] - y1)
            if m < 0.03:
                print(f'  TIGHT {m:.3f} in', repr(t.get_text()))
    # free-standing labels (inside no box): clearance to every block outline and every arrow
    print('free-standing labels (clearance to nearest block outline / nearest arrow, in):')
    for t, bb in bbs:
        x0, x1, y0, y1 = bb.x0 / fig.dpi, bb.x1 / fig.dpi, bb.y0 / fig.dpi, bb.y1 / fig.dpi
        if any(b[0] <= (x0 + x1) / 2 <= b[2] and b[1] <= (y0 + y1) / 2 <= b[3] for b in BOXES):
            continue

        def rect_dist(px, py):
            return np.hypot(max(x0 - px, 0, px - x1), max(y0 - py, 0, py - y1))

        def box_dist(b):
            if b[0] < x1 and x0 < b[2] and b[1] < y1 and y0 < b[3]:
                return -1.0
            return np.hypot(max(b[0] - x1, 0, x0 - b[2]), max(b[1] - y1, 0, y0 - b[3]))

        m_box = min(box_dist(b) for b in BOXES)
        m_arr = np.inf
        for (ax0, ay0), (ax1, ay1), hl, hw, lw in ARROWS:
            length = np.hypot(ax1 - ax0, ay1 - ay0)
            for s in np.linspace(0, 1, 400):
                px, py = ax0 + s * (ax1 - ax0), ay0 + s * (ay1 - ay0)
                to_tip = (1 - s) * length
                half = max(hw * to_tip / hl, lw / 2) if to_tip <= hl else lw / 2
                m_arr = min(m_arr, rect_dist(px, py) - half)
        flag = '  <-- TIGHT' if m_box < 0.05 or m_arr < 0.04 else ''
        print(f'  {t.get_text()!r:18s} box {m_box:.3f}  arrow {m_arr:.3f}{flag}')
    lws = [l.get_linewidth() for l in ax.lines] + [p.get_linewidth() for p in ax.patches
                                                   if p.get_edgecolor()[3] > 0 and p.get_linewidth() > 0]
    print(f'line widths {min(lws):.2f} to {max(lws):.2f} pt')
    assert LW_RANGE[0] <= min(lws) and max(lws) <= LW_RANGE[1], f'stroke widths outside {LW_RANGE} pt'
    assert min(sizes) >= 8, 'text below 8 pt'
    strip = [t.get_text() for t in texts if t.get_gid() == 'strip']
    assert strip == [f'{BUDGET.max_fits}', 'fits', f'{BUDGET.max_turns}', 'responses', f'{BUDGET.max_hours:g}',
                     'h'], strip
    assert any(t.get_text() == f'{N_TOOLS} tools' for t in texts)
    assert sorted(t.get_text() for t in texts if t.get_gid() == 'badge') == sorted(
        str(len(v)) for v in TOOL_GROUPS.values())


def main():
    draw()
    audit()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'Figure_1.pdf', facecolor=WHITE)
    fig.savefig(OUT / 'Figure_1.png', dpi=600, facecolor=WHITE)
    plt.close(fig)
    from figures import tiff_cmyk
    tiff_cmyk(OUT / 'Figure_1.png', OUT / 'Figure_1.tiff')
    print('saved', *(OUT / f'Figure_1.{e}' for e in ('pdf', 'png', 'tiff')))


if __name__ == '__main__':
    main()
