"""Matplotlib version of the graphical abstract (vector, matplotlib primitives only), kept for comparison.

The submitted graphical abstract is the FigureLabs drawing built by paper/graphical_abstract_figurelabs.py, which
uses ga_text() and load_numbers() of this module for the graphical abstract text and its checks. This script
therefore writes its image under another name and copies nothing to paper/submission_cpt/.

Three columns joined by lanes: datasets and the three knowledge conditions (left) enter the PKAgent loop (middle);
one arrow per result card leaves it (right), colored by the condition the result belongs to (neutral for the card
on the 36 runs without or with the expert statement, which says so, and for the card on the two weakly supported
effects, which shows both conditions). The take-home line sits in a banner along
the bottom. Drawn at the exact CPT print size (7.0 x 4.375 in, width = 1.6 x height) and never rescaled; all
coordinates are in inches. Colors follow Figures 2 to 4: gray no knowledge, orange expert statement (and nothing
else), red misleading statement; the language model is teal.

Every number in the image and in the text comes from paper/build/numbers.json (written by
paper/manuscript_numbers.py; run it first), the tool count from paper/tool_groups.py and the model names from
benchmarks/figures.py; the text is assembled from the same values, and both are checked against each other and
against the claims on the cards.

Writes paper/figures/Graphical_abstract_matplotlib.pdf (vector), .png (600 dpi), .tiff (CMYK, via
benchmarks/figures.py tiff_cmyk) and .txt (the graphical abstract text), and prints a text audit (every text artist
with its size, the minimum size, word counts, stroke widths, overlaps).
Usage: python paper/graphical_abstract.py
"""
import json
import math
import re
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon,  # noqa: E402
                                Rectangle, Wedge)
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.text import Text  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'benchmarks'))
sys.path.insert(0, str(HERE))
from figures import COLOR, LLM, tiff_cmyk  # noqa: E402  (importing it also sets its own rcParams; ours are set below)
from tool_groups import N_TOOLS  # noqa: E402

OUT = HERE / 'figures'
NUMBERS = HERE / 'build' / 'numbers.json'
STEM = 'Graphical_abstract_matplotlib'  # the submitted Graphical_abstract.* come from graphical_abstract_figurelabs.py
W, H = 7.0, 4.375                       # print size in inches (CPT: width = 1.6 x height)
DPI = 600
MIN_PT = 10                             # smallest text in the image (pt)
LW_RANGE = (0.5, 1.0)                   # CPT line widths (pt)

TAKE_HOME = ('The knowledge an analyst gives the agent shapes weakly supported effects',
             'and should be reported with the model.')

# palette
TXT = '#1F2933'
BLUE, BLUE_T = '#4A7FB5', '#E8F1FB'
ORANGE, ORANGE_T = '#C8801E', '#FDF1E3'     # expert statement only
TEAL, TEAL_T = '#2B7A8C', '#E3F1F4'         # language model
GREEN, GREEN_T = '#4C8C4A', '#EEF6EE'
PURPLE, PURPLE_T = '#7A5BA6', '#F3EEF8'
OUTG, OUTG_T = '#7B8794', '#F4F5F7'
NOK, NOK_T = '#8A94A6', '#EEF0F4'           # no knowledge
RED, RED_T = '#B5473A', '#F9E9E6'           # misleading statement
DARK = '#3E4C59'
WHITE = '#FFFFFF'
assert (NOK, ORANGE) == (COLOR['none'], COLOR['knowledge']), 'condition colors differ from Figures 2 to 4'

LOOP = dict(gap=0.06, box_dx=0.18, pin=14, rad=0.36)    # call/return arcs: ends at the boxes and on the tool ring

LW = 0.8                                # default outline width (pt)
LW_THIN = 0.6
LW_BOLD = 1.0                           # heaviest stroke (CPT: 0.5 to 1 pt)

# ----------------------------------------------------------------------------------------------- numbers
NUMBER_WORDS = {w: k for k, w in enumerate('zero one two three four five six seven eight nine ten eleven twelve'
                                           .split())}


def runs_of(phrase):
    """(k, n) from a run count written by paper/manuscript_numbers.py ('36 of 36', '11 of 12 runs', 'all six runs',
    'none of the four runs', 'five of six runs', 'both runs', 'neither run')."""
    def num(w):
        return int(w) if w.isdigit() else NUMBER_WORDS[w]
    p = re.sub(r' runs?$', '', phrase.strip())
    if p in ('both', 'neither'):
        return (2 if p == 'both' else 0), 2
    for pattern, kn in ((r'all (\w+)', lambda m: (num(m[1]), num(m[1]))),
                        (r'none of the (\w+)', lambda m: (0, num(m[1]))),
                        (r'(\w+) of (?:the )?(\w+)', lambda m: (num(m[1]), num(m[2])))):
        m = re.fullmatch(pattern, p)
        if m:
            return kn(m)
    raise ValueError(f'cannot read a run count from {phrase!r}')


def word(k):
    return next(w for w, v in NUMBER_WORDS.items() if v == k)


def load_numbers():
    """The values shown in the image and the text, read from paper/build/numbers.json, with checks that the
    claims printed on the cards hold."""
    if not NUMBERS.exists():
        raise SystemExit(f'{NUMBERS} is missing: run python paper/manuscript_numbers.py first')
    n = json.loads(NUMBERS.read_text(encoding='utf-8'))
    datasets = [k[:-len('_hours_span')] for k in n if k.endswith('_hours_span')]
    v = dict(n_runs=int(n['n_runs']), n_datasets=len(datasets), n_llm=len(LLM), hours=n['hours_median'],
             cost=n['cost_median'], strong_phrase=n['strong_kept'])
    # main grid: runs and reference structures per condition (no knowledge, expert statement)
    v['cond_runs'] = {c: sum(int(n[f'{ds}_{c}_runs']) for ds in datasets) for c in ('none', 'knowledge')}
    v['cond_struct'] = {c: sum(runs_of(n[f'{ds}_{c}_structure'])[0] for ds in datasets) for c in ('none', 'knowledge')}
    v['struct_k'], v['struct_n'] = runs_of(n['structure_all'])
    assert v['struct_n'] == v['n_runs'] == sum(v['cond_runs'].values()), 'main-grid run counts disagree'
    assert v['struct_k'] == sum(v['cond_struct'].values()), 'reference-structure counts disagree'
    # no knowledge: strongly supported covariate effects kept
    v['strong_k'], v['strong_n'] = runs_of(n['strong_kept'])
    # expert statement: 'every reference relationship implemented' must hold in every run of every dataset
    for ds in datasets:
        k, t = runs_of(n[f'{ds}_knowledge_all_ref'])
        assert k == t, f'{ds}: not every expert-statement run implemented every reference relationship'
    # no knowledge: the two weakly supported effects left out in every run
    for key in ('pheno_none_V_APGR', 'remifentanil_none_V1_AGE'):
        assert runs_of(n[key])[0] == 0, f'{key}: a run without knowledge kept a weakly supported effect'
    # misleading statement: 'data-contradicted claims rejected', 'unsupported second compartment adopted'
    assert runs_of(n['pm_weight_kept'])[0] == runs_of(n['pm_weight_kept'])[1], 'a run dropped the weight effect'
    assert runs_of(n['pm_apgar_cl_adopted'])[0] == 0, 'a run adopted the Apgar effect on CL'
    assert runs_of(n['om_mm_kept'])[0] == runs_of(n['om_mm_kept'])[1], 'a run adopted linear elimination'
    assert runs_of(n['om_2cmt_adopted'])[0] > 0, 'no run adopted the second compartment'
    return v


def ga_text(v):
    """Graphical abstract text (method, main result, conclusion), assembled from the same values as the image."""
    structure = (f"all {v['struct_n']} runs" if v['struct_k'] == v['struct_n']
                 else f"{v['struct_k']} of {v['struct_n']} runs")
    return ('PKAgent lets a large language model develop population pharmacokinetic models through the tools of an '
            'open-source engine. '
            f"On {word(v['n_datasets'])} public datasets, {word(v['n_llm'])} language models found the reference "
            f"structures in {structure}. Without knowledge, they followed the data, keeping the strongly supported "
            f"covariate effects in {v['strong_phrase']} and leaving out the two weakly supported ones; with an expert "
            'statement, they included both. '
            'Where data are weak, the model follows what the analyst states.')

plt.rcParams.update({
    'font.family': 'Arial',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'text.color': TXT,
    'axes.edgecolor': TXT,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
})


# ----------------------------------------------------------------------------------------------- helpers
def rbox(ax, x, y, w, h, fc, ec, lw=LW, r=0.07, z=1, ls='-'):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw,
                       ls=ls, zorder=z)
    ax.add_patch(p)
    return p


def card(ax, x0, y0, x1, y1, fc, ec, accent=None, bar=0.05):
    """Result card; an accent color adds a thin bar along the left edge, clipped to the rounded outline."""
    p = rbox(ax, x0, y0, x1 - x0, y1 - y0, fc, ec, lw=LW, r=0.07, z=1)
    if accent is not None:
        b = Rectangle((x0, y0), bar, y1 - y0, fc=accent, ec='none', zorder=1.5)
        ax.add_patch(b)
        b.set_clip_path(p)
    return p


def text(ax, x, y, s, size=10, weight='normal', ha='left', va='center', z=10, **kw):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va, color=TXT, zorder=z, **kw)


def arrow(ax, p0, p1, color=DARK, lw=LW_BOLD, ms=9, z=6, rad=0.0, style='-|>'):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, connectionstyle=f'arc3,rad={rad}', mutation_scale=ms, lw=lw,
                        color=color, shrinkA=0, shrinkB=0, zorder=z, joinstyle='miter', capstyle='butt')
    ax.add_patch(a)
    return a


def text_width(fig, s, size, weight='normal'):
    """Width of a string in inches at print size."""
    t = fig.text(0, 0, s, fontsize=size, fontweight=weight)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


# ----------------------------------------------------------------------------------------------- pictograms
def icon_disc(ax, cx, cy, r, ec, fc=WHITE, lw=LW):
    c = Circle((cx, cy), r, fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(c)
    return c


def icon_neonate(ax, cx, cy, c, s=1.0):
    """Swaddled newborn: small head with a curl on an upright wrap with a fold."""
    ax.add_patch(Ellipse((cx, cy - 0.05 * s), 0.15 * s, 0.18 * s, fc=c, ec='none', zorder=4))       # wrap
    fold = MPath([(cx - 0.065 * s, cy - 0.005 * s), (cx, cy - 0.072 * s), (cx + 0.065 * s, cy - 0.005 * s)],
                 [MPath.MOVETO, MPath.LINETO, MPath.LINETO])
    ax.add_patch(PathPatch(fold, fc='none', ec=WHITE, lw=LW, zorder=5, capstyle='round', joinstyle='round'))
    ax.add_patch(Circle((cx, cy + 0.064 * s), 0.056 * s, fc=c, ec=WHITE, lw=LW, zorder=6))            # head
    ax.add_patch(Wedge((cx + 0.014 * s, cy + 0.127 * s), 0.023 * s, 190, 30, width=0.010 * s, fc=c, ec='none',
                       zorder=6))


def icon_adult(ax, cx, cy, c, clip, s=1.0):
    """Adult: head and shoulders (shoulders clipped to the surrounding disc)."""
    body = ax.add_patch(Ellipse((cx, cy - 0.14 * s), 0.225 * s, 0.225 * s, fc=c, ec='none', zorder=4))
    body.set_clip_path(clip)
    ax.add_patch(Circle((cx, cy + 0.052 * s), 0.058 * s, fc=c, ec='none', zorder=4))


def icon_capsule(ax, cx, cy, c, length=0.245, width=0.095, ang=40):
    """Two-tone oral capsule."""
    r = width / 2
    half = length / 2 - r
    th = np.linspace(np.pi / 2, 3 * np.pi / 2, 24)
    left = np.r_[[[0.0, r]], np.c_[-half + r * np.cos(th), r * np.sin(th)], [[0.0, -r]]]
    right = left * np.array([-1, 1])
    rot = np.deg2rad(ang)
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    for pts, fc in ((left, c), (right, WHITE)):
        xy = pts @ R.T + np.array([cx, cy])
        ax.add_patch(Polygon(xy, closed=True, fc=fc, ec=c, lw=LW, zorder=4, joinstyle='round'))


def bubble_path(cx, cy, w, h, r=0.045, tail=0.06):
    """Speech bubble (rounded rectangle with a tail at the lower left) as one closed path."""
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2 + tail * 0.5, cy + h / 2 + tail * 0.5
    v, c = [], []

    def mv(p):
        v.append(p); c.append(MPath.MOVETO)

    def ln(p):
        v.append(p); c.append(MPath.LINETO)

    def qd(ctrl, p):
        v.extend([ctrl, p]); c.extend([MPath.CURVE3, MPath.CURVE3])

    mv((x0 + r, y1)); ln((x1 - r, y1)); qd((x1, y1), (x1, y1 - r)); ln((x1, y0 + r)); qd((x1, y0), (x1 - r, y0))
    if tail > 0:
        ln((x0 + w * 0.42, y0)); ln((x0 + w * 0.18, y0 - tail)); ln((x0 + w * 0.24, y0))
    ln((x0 + r, y0)); qd((x0, y0), (x0, y0 + r)); ln((x0, y1 - r)); qd((x0, y1), (x0 + r, y1))
    v.append((x0 + r, y1)); c.append(MPath.CLOSEPOLY)
    return MPath(v, c)


def star(cx, cy, ro, ri=None, n=5):
    ri = ri if ri is not None else ro * 0.45
    ang = np.pi / 2 + np.arange(2 * n) * np.pi / n
    rad = np.where(np.arange(2 * n) % 2 == 0, ro, ri)
    return np.c_[cx + rad * np.cos(ang), cy + rad * np.sin(ang)]


def icon_condition(ax, cx, cy, kind, s=1.0):
    """Condition pictogram: empty dashed bubble (no knowledge), bubble with a star (expert statement),
    bubble with a cross (misleading statement)."""
    w, h, tail = 0.26 * s, 0.17 * s, 0.06 * s
    path = bubble_path(cx, cy, w, h, r=0.045 * s, tail=tail)
    if kind == 'none':
        # dashed body without a tail (same offset as the tailed bubbles), then a solid open tail on top whose
        # white fill, plus a white strip over the joint, hides the body dash under the tail base
        body = bubble_path(cx, cy + tail * 0.5, w, h, r=0.045 * s, tail=0)
        ax.add_patch(PathPatch(body, fc=WHITE, ec=NOK, lw=LW_BOLD, ls=(0, (2.2, 1.4)), zorder=4, joinstyle='round'))
        x0, y0 = cx - w / 2, cy - h / 2 + tail * 0.5
        half = LW_BOLD / 72 / 2 + 0.002
        ax.add_patch(Rectangle((x0 + 0.24 * w, y0 - half), 0.18 * w, 2 * half, fc=WHITE, ec='none', zorder=4.4))
        tp = MPath([(x0 + 0.42 * w, y0), (x0 + 0.18 * w, y0 - tail), (x0 + 0.24 * w, y0)],
                   [MPath.MOVETO, MPath.LINETO, MPath.LINETO])
        ax.add_patch(PathPatch(tp, fc=WHITE, ec=NOK, lw=LW_BOLD, zorder=4.5, joinstyle='round', capstyle='round'))
        return
    col = ORANGE if kind == 'expert' else RED
    ax.add_patch(PathPatch(path, fc=col, ec=col, lw=LW, zorder=4, joinstyle='round'))
    my = cy + 0.03 * s
    if kind == 'expert':
        ax.add_patch(Polygon(star(cx, my, 0.062 * s), closed=True, fc=WHITE, ec='none', zorder=5))
    else:
        d = 0.042 * s
        for sx in (1, -1):
            ax.plot([cx - d, cx + d], [my - sx * d, my + sx * d], color=WHITE, lw=LW_BOLD, zorder=5,
                    solid_capstyle='round')


def icon_network(ax, cx, cy, c, s=1.0):
    """Language model: small three-layer network."""
    layers = [(-0.1, [-0.075, 0.0, 0.075]), (0.0, [-0.04, 0.04]), (0.1, [-0.075, 0.0, 0.075])]
    pts = [[(cx + dx * s, cy + dy * s) for dy in ys] for dx, ys in layers]
    for a_, b_ in ((0, 1), (1, 2)):
        for p in pts[a_]:
            for q in pts[b_]:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=c, lw=LW_THIN, zorder=4)
    for layer in pts:
        for p in layer:
            ax.add_patch(Circle(p, 0.025 * s, fc=c, ec=WHITE, lw=LW_THIN, zorder=5))


def gear(cx, cy, ro, ri, n=8, frac=0.45):
    pts = []
    step = 2 * np.pi / n
    for k in range(n):
        a = k * step
        t0, t1 = a - step * frac / 2, a + step * frac / 2
        pts += [(ri, a - step / 2 + 0.02), (ri, t0 - 0.06), (ro, t0), (ro, t1), (ri, t1 + 0.06)]
    return np.array([(cx + r * np.cos(t), cy + r * np.sin(t)) for r, t in pts])


def icon_gear(ax, cx, cy, c, s=1.0):
    ax.add_patch(Polygon(gear(cx, cy, 0.105 * s, 0.078 * s), closed=True, fc=c, ec='none', zorder=4))
    ax.add_patch(Circle((cx, cy), 0.034 * s, fc=WHITE, ec='none', zorder=5))


def icon_log(ax, cx, cy, c, s=1.0):
    """Log file: page with folded corner and text lines."""
    w, h, f = 0.17 * s, 0.22 * s, 0.05 * s
    x0, y0 = cx - w / 2, cy - h / 2
    page = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h - f), (x0 + w - f, y0 + h), (x0, y0 + h)]
    ax.add_patch(Polygon(page, closed=True, fc=WHITE, ec=c, lw=LW, zorder=4, joinstyle='round'))
    ax.add_patch(Polygon([(x0 + w - f, y0 + h), (x0 + w - f, y0 + h - f), (x0 + w, y0 + h - f)], closed=True,
                         fc=c, ec=c, lw=LW_THIN, zorder=5))
    for k, frac in enumerate((0.55, 0.75, 0.6, 0.7)):
        yy = y0 + h - 0.075 * s - k * 0.04 * s
        ax.plot([x0 + 0.03 * s, x0 + 0.03 * s + (w - 0.06 * s) * frac], [yy, yy], color=c, lw=LW, zorder=5,
                solid_capstyle='round')


def icon_check(ax, cx, cy, c, r=0.1):
    ax.add_patch(Circle((cx, cy), r, fc=c, ec='none', zorder=4))
    k = r / 0.075
    ax.plot([cx - 0.035 * k, cx - 0.008 * k, cx + 0.04 * k], [cy + 0.0, cy - 0.028 * k, cy + 0.032 * k],
            color=WHITE, lw=LW_BOLD, zorder=5, solid_capstyle='round', solid_joinstyle='round')


def icon_blocked(ax, cx, cy, c, r=0.075):
    """Rejected: neutral prohibition sign (circle with a diagonal bar)."""
    ax.add_patch(Circle((cx, cy), r, fc=WHITE, ec=c, lw=LW_BOLD, zorder=4))
    d = r * np.cos(np.pi / 4)
    ax.plot([cx - d, cx + d], [cy + d, cy - d], color=c, lw=LW_BOLD, zorder=5, solid_capstyle='butt')


def icon_warning(ax, cx, cy, c, r=0.085):
    """Warning triangle with an exclamation mark (drawn, not typed)."""
    tri = [(cx, cy + r), (cx - r * 0.95, cy - r * 0.7), (cx + r * 0.95, cy - r * 0.7)]
    ax.add_patch(Polygon(tri, closed=True, fc=c, ec=c, lw=LW, zorder=4, joinstyle='round'))
    ax.plot([cx, cx], [cy + r * 0.45, cy - r * 0.15], color=WHITE, lw=LW_BOLD, zorder=5, solid_capstyle='round')
    ax.add_patch(Circle((cx, cy - r * 0.42), 0.011, fc=WHITE, ec='none', zorder=5))


def icon_two_compartment(ax, x0, cy, s=0.12, gap=0.11):
    """Solid central box, double arrow, dashed red peripheral box; returns the right edge."""
    ax.add_patch(FancyBboxPatch((x0, cy - s / 2), s, s, boxstyle='round,pad=0,rounding_size=0.018', fc=WHITE,
                                ec=DARK, lw=LW_BOLD, zorder=4))
    xp = x0 + s + gap
    ax.add_patch(FancyBboxPatch((xp, cy - s / 2), s, s, boxstyle='round,pad=0,rounding_size=0.018', fc=WHITE,
                                ec=RED, lw=LW_BOLD, ls=(0, (2.0, 1.2)), zorder=4))
    xa, xb, hl, hw = x0 + s + 0.022, xp - 0.022, 0.05, 0.03          # double arrow: shaft and two filled heads
    ax.plot([xa + hl * 0.5, xb - hl * 0.5], [cy, cy], color=DARK, lw=LW, zorder=5, solid_capstyle='butt')
    for tip, d in ((xa, 1), (xb, -1)):
        ax.add_patch(Polygon([(tip, cy), (tip + d * hl, cy + hw), (tip + d * hl, cy - hw)], closed=True, fc=DARK,
                             ec='none', zorder=5))
    return xp + s


def icon_clock(ax, cx, cy, c, r=0.085):
    ax.add_patch(Circle((cx, cy), r, fc=WHITE, ec=c, lw=LW_BOLD, zorder=4))
    ax.plot([cx, cx], [cy, cy + r * 0.62], color=c, lw=LW_BOLD, zorder=5, solid_capstyle='round')
    ax.plot([cx, cx + r * 0.5], [cy, cy - r * 0.2], color=c, lw=LW_BOLD, zorder=5, solid_capstyle='round')


def icon_banknote(ax, cx, cy, c, s=1.0):
    """Banknote (fees): note with a central coin mark and side dots."""
    w, h = 0.2 * s, 0.125 * s
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f'round,pad=0,rounding_size={0.018 * s}',
                                fc=c, ec='none', zorder=4))
    ax.add_patch(Circle((cx, cy), 0.032 * s, fc='none', ec=WHITE, lw=LW, zorder=5))
    for sx in (-1, 1):
        ax.add_patch(Circle((cx + sx * 0.066 * s, cy), 0.011 * s, fc=WHITE, ec='none', zorder=5))


def icon_bulb(ax, cx, cy, c, s=1.0):
    ax.add_patch(Circle((cx, cy + 0.035 * s), 0.085 * s, fc=c, ec='none', zorder=4))
    ax.add_patch(Polygon([(cx - 0.05 * s, cy - 0.03 * s), (cx + 0.05 * s, cy - 0.03 * s),
                          (cx + 0.038 * s, cy - 0.085 * s), (cx - 0.038 * s, cy - 0.085 * s)],
                         closed=True, fc=c, ec='none', zorder=4))
    for k in range(2):
        y = cy - 0.105 * s - k * 0.03 * s
        ax.add_patch(FancyBboxPatch((cx - 0.036 * s, y - 0.009 * s), 0.072 * s, 0.018 * s,
                                    boxstyle=f'round,pad=0,rounding_size={0.008 * s}', fc=DARK, ec='none',
                                    zorder=4))
    for ang in (90, 30, 150, 0, 180):
        a = np.deg2rad(ang)
        r0, r1 = 0.12 * s, 0.155 * s
        ax.plot([cx + r0 * np.cos(a), cx + r1 * np.cos(a)],
                [cy + 0.035 * s + r0 * np.sin(a), cy + 0.035 * s + r1 * np.sin(a)],
                color=c, lw=LW_BOLD, zorder=4, solid_capstyle='round')


def waffle(ax, x0, ytop, ncol, colors, filled, pitch=0.09, r=0.032, row_gap=None):
    """Grid of run markers, row by row from the top; colors[k] and filled[k] for marker k. row_gap adds space
    after the given row (used to split the two conditions)."""
    for k, (col, fl) in enumerate(zip(colors, filled)):
        i, j = divmod(k, ncol)
        extra = row_gap[1] if row_gap is not None and i >= row_gap[0] else 0.0
        cx, cy = x0 + r + j * pitch, ytop - r - i * pitch - extra
        ax.add_patch(Circle((cx, cy), r, fc=col if fl else WHITE, ec=col, lw=LW, zorder=4))


# ----------------------------------------------------------------------------------------------- figure
def build(v):
    fig = plt.figure(figsize=(W, H), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')
    fig.canvas.draw()
    shown = {}                          # numbers drawn on the cards, checked against the text in main()

    m = 0.07
    yb0, yb1 = m, 0.56                  # take-home banner
    y0, y1 = 0.66, H - m                # main zone
    L0, L1 = m, 1.93                    # left: datasets and conditions
    M0, M1 = 2.23, 4.08                 # middle: PKAgent
    R0, R1 = 4.38, W - m                # right: results
    G = 0.035                           # gap between arrow ends and boxes

    # ---------------- right column first: its cards define the lanes
    gap = 0.07
    bar = 0.04
    icol0 = R0 + bar + 0.08                    # leading icon column [icol0, icol1]
    pitch, rdot, ncol = 0.09, 0.032, 6
    icol1 = icol0 + (ncol - 1) * pitch + 2 * rdot
    icx = (icol0 + icol1) / 2
    tx = icol1 + 0.13                          # text column of every card

    # main-grid waffle: one marker per run, the no-knowledge rows (gray) above the expert-statement rows (orange)
    nk, ex = v['cond_runs']['none'], v['cond_runs']['knowledge']
    split = (nk // ncol, 0.03) if nk % ncol == 0 else None
    rows36 = math.ceil((nk + ex) / ncol)
    gh = (rows36 - 1) * pitch + 2 * rdot + (split[1] if split else 0.0)
    scope_h = 0.245                            # scope line under the waffle (centre 0.115 below it)
    c36 = (y1 - (0.06 + gh + scope_h), y1)
    cmed = (c36[0] - gap - 0.47, c36[0] - gap)
    cnok = (cmed[0] - gap - 0.70, cmed[0] - gap)
    cexp = (cnok[0] - gap - 0.50, cnok[0] - gap)
    cmis = (y0, cexp[0] - gap)
    mid = lambda c: (c[0] + c[1]) / 2          # noqa: E731

    # reference structure in the runs without or with the expert statement
    card(ax, R0, c36[0], R1, c36[1], WHITE, OUTG)
    wtop = c36[1] - 0.06
    colors = [NOK] * nk + [ORANGE] * ex
    filled = ([True] * v['cond_struct']['none'] + [False] * (nk - v['cond_struct']['none'])
              + [True] * v['cond_struct']['knowledge'] + [False] * (ex - v['cond_struct']['knowledge']))
    waffle(ax, icol0, wtop, ncol, colors, filled, pitch, rdot, row_gap=split)
    cy = wtop - gh / 2
    big = f"{v['struct_k']}/{v['struct_n']}"
    shown['structure'] = big
    text(ax, tx, cy + 0.03, big, 20, 'bold', va='baseline')
    text(ax, tx + text_width(fig, big, 20, 'bold') + 0.05, cy + 0.03, 'runs', 10, va='baseline')
    text(ax, tx, cy - 0.15, 'reference structure', 10)
    text(ax, icol0, wtop - gh - 0.115, 'without or with the expert statement', 10)

    # the two weakly supported effects: left out without knowledge, included with the expert statement (each outcome
    # after the pictogram of its condition, as on the condition chips)
    card(ax, R0, cmed[0], R1, cmed[1], OUTG_T, OUTG)
    text(ax, icol0, cmed[1] - 0.12, 'two weakly supported effects', 10)
    ry = cmed[0] + 0.135
    x = icol0 + 0.08
    for kind, label in (('none', 'left out'), ('expert', 'included')):
        icon_condition(ax, x, ry - 0.012, kind, s=0.62)
        text(ax, x + 0.14, ry, label, 10, 'bold')
        x += 0.14 + text_width(fig, label, 10, 'bold') + 0.36

    # no knowledge: strongly supported covariate effects kept
    card(ax, R0, cnok[0], R1, cnok[1], NOK_T, NOK, accent=NOK, bar=bar)
    cy = mid(cnok)
    k, t = v['strong_k'], v['strong_n']
    rows = math.ceil(t / ncol)
    waffle(ax, icol0, cy + ((rows - 1) * pitch + 2 * rdot) / 2, ncol, [NOK] * t, [True] * k + [False] * (t - k),
           pitch, rdot)
    big = f'{k}/{t}'
    shown['strong'] = big
    text(ax, tx, cy + 0.1, big, 20, 'bold', va='baseline')
    text(ax, tx + text_width(fig, big, 20, 'bold') + 0.05, cy + 0.1, 'runs', 10, va='baseline')
    text(ax, tx, cy - 0.06, 'strongly supported', 10)
    text(ax, tx, cy - 0.225, 'covariate effects kept', 10)

    # expert statement: every reference relationship implemented
    card(ax, R0, cexp[0], R1, cexp[1], ORANGE_T, ORANGE, accent=ORANGE, bar=bar)
    cy = mid(cexp)
    icon_check(ax, icx, cy, ORANGE, r=0.105)
    text(ax, tx, cy + 0.085, 'every reference', 10)
    text(ax, tx, cy - 0.085, 'relationship implemented', 10)

    # misleading statement: data-contradicted claims rejected; unsupported second compartment adopted
    card(ax, R0, cmis[0], R1, cmis[1], RED_T, RED, accent=RED, bar=bar)
    cy = mid(cmis) + 0.008                    # pictogram below the last line: balance the margins
    ra, rb = cy + 0.21, cy - 0.21
    icon_blocked(ax, icx, ra, DARK, r=0.085)
    text(ax, tx, ra + 0.085, 'data-contradicted claims', 10)
    text(ax, tx, ra - 0.085, 'rejected', 10, 'bold')
    s2, g2 = 0.15, 0.17                        # two-compartment pictogram, warning triangle over the peripheral box
    gx0 = icx - (2 * s2 + g2) / 2
    icon_two_compartment(ax, gx0, rb - 0.09, s=s2, gap=g2)
    icon_warning(ax, gx0 + 1.5 * s2 + g2, rb + 0.08, RED, r=0.075)
    text(ax, tx, rb + 0.085, 'unsupported second', 10)
    text(ax, tx, rb - 0.085, 'compartment', 10)
    text(ax, tx + text_width(fig, 'compartment ', 10), rb - 0.085, 'adopted', 10, 'bold')

    # ---------------- left: datasets (aligned with the two overall cards)
    ds = (cmed[0], y1)
    rbox(ax, L0, ds[0], L1 - L0, ds[1] - ds[0], BLUE_T, BLUE, lw=LW)
    rows = np.linspace(ds[1] - 0.205, ds[0] + 0.205, 3)
    ix, lx = L0 + 0.21, L0 + 0.42
    datasets = [('Phenobarbital', '(neonates)'), ('Remifentanil', '(adults)'),
                ('Simulated oral drug', '(saturable elimination)')]
    for k, (cy, (name, sub)) in enumerate(zip(rows, datasets)):
        disc = icon_disc(ax, ix, cy, 0.14, BLUE)
        if k == 0:
            icon_neonate(ax, ix, cy, BLUE, s=0.9)
        elif k == 1:
            icon_adult(ax, ix, cy, BLUE, clip=disc, s=0.9)
        else:
            icon_capsule(ax, ix, cy, BLUE, length=0.22, width=0.085)
        text(ax, lx, cy + 0.085, name, 10, 'bold')
        text(ax, lx, cy - 0.085, sub, 10)

    # ---------------- left: condition chips, one per lane
    chip_h = 0.46
    lanes = [('none', 'no knowledge', NOK, NOK_T, mid(cnok)),
             ('expert', 'expert statement', ORANGE, ORANGE_T, mid(cexp)),
             ('misleading', 'misleading statement', RED, RED_T, mid(cmis))]
    for kind, lab, col, tint, cy in lanes:
        rbox(ax, L0, cy - chip_h / 2, L1 - L0, chip_h, tint, col, lw=LW, r=0.07)
        icon_condition(ax, ix, cy - 0.012, kind)
        text(ax, lx, cy, lab, 10)

    # ---------------- middle: PKAgent loop
    rbox(ax, M0, y0, M1 - M0, y1 - y0, GREEN_T, GREEN, lw=LW_BOLD, r=0.09)
    mc = (M0 + M1) / 2
    text(ax, mc, y1 - 0.165, 'PKAgent', 11, 'bold', ha='center')

    bx0, bx1 = M0 + 0.08, M1 - 0.08
    bicx, btx = bx0 + 0.2, bx0 + 0.4
    llm = (3.29, 3.98)
    rbox(ax, bx0, llm[0], bx1 - bx0, llm[1] - llm[0], TEAL_T, TEAL, lw=LW)
    lcy = mid(llm)
    icon_network(ax, bicx, lcy, TEAL, s=1.1)
    text(ax, btx, lcy + 0.165, 'Language model', 10, 'bold')
    gpt, claude = LLM['gpt'], LLM['claude']
    text(ax, btx, lcy - 0.005, gpt, 10)
    text(ax, btx, lcy - 0.17, claude, 10)

    eng = (1.36, 1.96)
    tcy = (llm[0] + eng[1]) / 2                # tools hub halfway between language model and engine
    ring, disc_r = 0.3, 0.255
    ax.add_patch(Circle((mc, tcy), disc_r, fc=PURPLE_T, ec=PURPLE, lw=LW, zorder=3))
    for k in range(N_TOOLS):                    # one dot per tool (paper/tool_groups.py)
        a = np.pi / 2 + k * 2 * np.pi / N_TOOLS
        ax.add_patch(Circle((mc + ring * np.cos(a), tcy + ring * np.sin(a)), 0.021, fc=PURPLE, ec='none',
                            zorder=4))
    shown['tools'] = str(N_TOOLS)
    text(ax, mc, tcy + 0.06, str(N_TOOLS), 16, 'bold', ha='center', va='center')
    text(ax, mc, tcy - 0.115, 'tools', 10, ha='center', va='center')

    rbox(ax, bx0, eng[0], bx1 - bx0, eng[1] - eng[0], PURPLE_T, PURPLE, lw=LW)
    ecy = mid(eng)
    icon_gear(ax, bicx, ecy, PURPLE, s=1.15)
    text(ax, btx, ecy + 0.09, 'PKPy2', 10, 'bold')
    text(ax, btx, ecy - 0.09, 'open-source engine', 10)

    # call (down, left) and return (up, right) arcs: language model -> tools -> engine -> tools -> language model
    ra_ = ring + LOOP['gap']
    dx_box, deg = LOOP['box_dx'], LOOP['pin']
    pin = lambda d: (mc + ra_ * np.cos(np.deg2rad(d)), tcy + ra_ * np.sin(np.deg2rad(d)))  # noqa: E731
    kw = dict(color=DARK, lw=LW_BOLD, ms=10, rad=LOOP['rad'])
    arrow(ax, (mc - dx_box, llm[0] - 0.03), pin(180 - deg), **kw)
    arrow(ax, pin(180 + deg), (mc - dx_box, eng[1] + 0.03), **kw)
    arrow(ax, (mc + dx_box, eng[1] + 0.03), pin(360 - deg), **kw)
    arrow(ax, pin(deg), (mc + dx_box, llm[0] - 0.03), **kw)

    # footer: every step logged
    sep = eng[0] - 0.16
    ax.plot([bx0, bx1], [sep, sep], color=GREEN, lw=LW_THIN, ls=(0, (3, 2)), zorder=2)
    lgy = (y0 + sep) / 2
    icon_log(ax, bicx, lgy, GREEN, s=1.1)
    text(ax, btx, lgy, 'every step logged', 10)

    # ---------------- lanes
    arrow(ax, (L1 + G, mid(ds)), (M0 - G, mid(ds)), color=BLUE)
    for kind, lab, col, tint, cy in lanes:
        arrow(ax, (L1 + G, cy), (M0 - G, cy), color=col)
        arrow(ax, (M1 + G, cy), (R0 - G, cy), color=col)
    arrow(ax, (M1 + G, mid(c36)), (R0 - G, mid(c36)), color=DARK)

    # ---------------- banner: take-home line, centered in the space right of the bulb
    rbox(ax, m, yb0, W - 2 * m, yb1 - yb0, OUTG_T, OUTG, lw=LW, r=0.08)
    bulb_x = m + 0.3
    icon_bulb(ax, bulb_x, (yb0 + yb1) / 2 + 0.005, DARK, s=1.1)      # neutral: orange is the expert statement
    free0 = bulb_x + 0.155 * 1.1 + 0.05
    text(ax, (free0 + W - m) / 2, (yb0 + yb1) / 2, '\n'.join(TAKE_HOME), 11, 'bold', ha='center', va='center',
         linespacing=1.3)

    return fig, ax, shown


# ----------------------------------------------------------------------------------------------- audit
def audit(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    rows = []
    for t in fig.findobj(Text):
        s = t.get_text()
        if not s.strip() or not t.get_visible():
            continue
        bb = t.get_window_extent(r)
        rows.append(dict(size=t.get_fontsize(), weight=t.get_fontweight(), font=t.get_fontname(),
                         text=s.replace('\n', ' / '), bb=(bb.x0 / fig.dpi, bb.y0 / fig.dpi, bb.x1 / fig.dpi,
                                                          bb.y1 / fig.dpi), color=t.get_color()))
    take = ' '.join(TAKE_HOME)
    body = [q for q in rows if q['text'].replace(' / ', ' ') != take]
    words = sum(len(q['text'].replace(' / ', ' ').split()) for q in body)
    numeric = sum(1 for q in body for w_ in q['text'].split() if any(ch.isdigit() for ch in w_))
    print(f'{"size":>5} {"weight":>7} {"font":>6}  text')
    for q in rows:
        print(f'{q["size"]:5.1f} {str(q["weight"]):>7} {q["font"]:>6}  {q["text"]}')
    print(f'text artists: {len(rows)}; minimum font size: {min(q["size"] for q in rows):.1f} pt; '
          f'fonts: {sorted({q["font"] for q in rows})}; colors: {sorted({str(q["color"]) for q in rows})}')
    print(f'words besides the take-home line: {words} ({numeric} of them numbers); '
          f'take-home line: {len(take.split())}; total {words + len(take.split())}')
    out = [q['text'] for q in rows if q['bb'][0] < 0 or q['bb'][1] < 0 or q['bb'][2] > W or q['bb'][3] > H]
    print('text outside the page:', out or 'none')
    hits = [(a['text'], b['text']) for a, b in combinations(rows, 2)
            if min(a['bb'][2], b['bb'][2]) > max(a['bb'][0], b['bb'][0])
            and min(a['bb'][3], b['bb'][3]) > max(a['bb'][1], b['bb'][1])]
    print('overlapping text boxes:', hits or 'none')
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    lws = []
    for art in fig.axes[0].get_children():
        if isinstance(art, Line2D):
            lws.append(art.get_linewidth())
        elif isinstance(art, Patch) and art.get_visible() and to_rgba(art.get_edgecolor())[3] > 0 \
                and art.get_linewidth() > 0:
            lws.append(art.get_linewidth())
    print(f'stroke widths (pt): min {min(lws):.2f}, max {max(lws):.2f} over {len(lws)} stroked artists')
    assert not out and not hits, 'text outside the page or overlapping'
    assert LW_RANGE[0] <= min(lws) and max(lws) <= LW_RANGE[1], f'stroke widths outside {LW_RANGE} pt'
    assert min(q['size'] for q in rows) >= MIN_PT, f'text below {MIN_PT} pt'
    return words, min(q['size'] for q in rows)


def main():
    v = load_numbers()
    text_ = ga_text(v)
    fig, _, shown = build(v)
    audit(fig)
    # the image and the text show the same values
    assert shown['structure'] == f"{v['struct_k']}/{v['struct_n']}" and (
        f"all {v['struct_n']} runs" if v['struct_k'] == v['struct_n'] else f"{v['struct_k']} of {v['struct_n']} runs"
    ) in text_
    assert shown['strong'] == f"{v['strong_k']}/{v['strong_n']}" and v['strong_phrase'] in text_
    assert runs_of(v['strong_phrase']) == (v['strong_k'], v['strong_n'])
    assert shown['tools'] == str(N_TOOLS)
    assert 50 <= len(text_.split()) <= 80, 'graphical abstract text outside 50 to 80 words'
    assert '\u2014' not in text_
    print('values:', {k: shown[k] for k in sorted(shown)}, '| runs per condition', v['cond_runs'])
    print('text:', text_)
    assert tuple(fig.get_size_inches()) == (W, H)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{STEM}.pdf')
    fig.savefig(OUT / f'{STEM}.png', dpi=DPI)
    plt.close(fig)
    tiff_cmyk(OUT / f'{STEM}.png', OUT / f'{STEM}.tiff')
    (OUT / f'{STEM}.txt').write_text(text_ + '\n', encoding='utf-8')
    print(f'Graphical abstract: {W} x {H} in (ratio {W / H:.3f}), PNG and TIFF at {DPI} dpi; '
          f'text {len(text_.split())} words')


if __name__ == '__main__':
    main()
