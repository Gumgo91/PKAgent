"""Figures of the simulation experiment (PROTOCOL.md, section 8.4): adoption against the evidence in the data.

Reads design.json, evaluation/runs.csv and evaluation/summary.json (analyze.py, which holds the fitted curves).

adoption_evidence         headline figure, three panels (benchmark prompt; other prompts; two-compartment statement
                          of A2). x: the reference evidence z_ref, labeled as dOFV, with lines at 3.84 and 6.63 and a
                          second axis with the typical true effect. One point per dataset and condition at the number
                          of adopting LLM runs (0, 1 or 2; the left half of a point is GPT, the right half Claude),
                          the fitted curve of each condition over the range in which it was run with its bootstrap
                          band, the ranges that were not run, the rule-following baseline as a gray step and twin
                          adoption as a dashed line.
adoption_evidence_by_llm  the same by LLM (supplement).
adoption_agent_evidence   adoption against the evidence each run obtained itself (section 8.8, supplement).
outcome_grid              every outcome by dataset, condition and LLM (supplement).
Outputs: evaluation/figures/<name>.pdf (vector) and <name>.png (600 dpi).

Usage:
  python benchmarks/simulation/sim_figures.py [--design PATH] [--runs PATH] [--summary PATH] [--out FOLDER]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402
from matplotlib.patches import Rectangle             # noqa: E402

import analyze                                       # noqa: E402
import common                                        # noqa: E402

# Fonts, line widths and base colors of benchmarks/figures.py
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.linewidth': .6, 'lines.linewidth': 1.,
                     'xtick.major.width': .6, 'ytick.major.width': .6, 'pdf.fonttype': 42,
                     'font.family': 'Arial', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
                     'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold', 'text.color': '#1F2933',
                     'axes.labelcolor': '#1F2933', 'axes.edgecolor': '#1F2933', 'xtick.color': '#1F2933',
                     'ytick.color': '#1F2933'})
WIDTH = 7.0                                          # inches: a double-column journal figure (178 mm)
INK, MUTED, FAINT, PALE = '#1F2933', '#52606D', '#7B8794', '#CBD2D9'
LLM = dict(gpt='GPT-6.1 Sol', claude='Claude Opus 5.5')
MARK = dict(gpt='o', claude='s')
NONE = analyze.NONE
# label and color of every condition: gray for no statement and orange for a statement as in the benchmark
# figures; the hue follows the statement (assert orange to brown, deny blue, two compartments purple)
ARM = {NONE: ('no statement', '#8A94A6'),
       'assert-respect': ('effect asserted', '#C8801E'),
       'assert-neutral': ('effect asserted,\nneutral paragraph', '#C4960A'),
       'assert-criteria': ('effect asserted,\nexplicit rule', '#8F3519'),
       'deny-respect': ('effect denied', '#2F6FAB'),
       'deny-neutral': ('effect denied,\nneutral paragraph', '#245A9C'),
       'twocmt-respect': ('two compartments stated', '#8E5BC2'),
       'twocmt-criteria': ('two compartments stated,\nexplicit rule', '#4A2E73')}
PANELS = (dict(title='Benchmark prompt', arms=(NONE, 'assert-respect', 'deny-respect'), endpoint='Y',
               baseline=True, twin='twin/benchmark', ylabel='P(target effect in the final model)'),
          dict(title='Other prompts', arms=(NONE, 'assert-neutral', 'assert-criteria', 'deny-neutral'), endpoint='Y',
               ylabel='P(target effect in the final model)'),
          dict(title='Two-compartment statement (module A2)', arms=(NONE, 'twocmt-respect', 'twocmt-criteria'),
               endpoint='two_cmt', datasets='A2', curves={NONE: 'none-respect/A2'},
               ylabel='P(two compartments in the final model)'))
TICKS = (0, 1.5, 3.84, 6.63, 10, 20, 40)             # dOFV labels of the evidence axis (square-root scale)
XLIM = (-1.0, 6.65)
STEP, ROW = .068, .105                               # height of a lane of points and of a range row (y units)
GAP = .095                                           # points of a lane closer than this are moved apart (x units)
XLABEL = ('evidence for the target effect in the data: reference ΔOFV (square-root scale;\n'
          'left of 0, the reference estimate points to higher clearance)')
GRID_GROUPS = (('no statement', 'none'), ('effect asserted', 'assert'), ('effect denied', 'deny'),
               ('two compartments\nstated', 'twocmt'))
GRID_PROMPT = dict(respect='benchmark', neutral='neutral', criteria='rule')


def _save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f'{name}.pdf', out / f'{name}.png']
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=600)
    plt.close(fig)
    return paths


def _array(values):
    return np.array([np.nan if v is None else v for v in values], float)


def spread(x, gap=GAP):
    """Positions for drawing: points closer than gap are moved apart around their mean (display only)."""
    x = np.asarray(x, float)
    out, groups = x.copy(), []
    for i in np.argsort(x, kind='stable'):
        if groups and x[i] - x[groups[-1][-1]] < gap:
            groups[-1].append(i)
        else:
            groups.append([i])
    for g in groups:
        out[g] = x[g].mean() + gap * (np.arange(len(g)) - (len(g) - 1) / 2)
    return out


def lane_y(level, lane, lanes):
    """y of the points of a condition: below 0 for no adopting run, above 1 for all runs, around 0.5 between."""
    if level == 0:
        return -.075 - STEP * lane
    if level == 1:
        return 1.075 + STEP * lane
    return .5 + STEP * (lane - (lanes - 1) / 2)


def frame(ax, lanes, rows):
    """Axes of a panel: the probability scale from 0 to 1, lanes of points below and above it, and room for the
    range rows at the top. Returns the y of the first range row."""
    top = 1.075 + STEP * (lanes - 1) + .1
    ax.set_ylim(-.075 - STEP * (lanes - 1) - .06, top + ROW * max(rows - 1, 0) + .07)
    ax.set_yticks([0, .5, 1])
    ax.set_yticklabels(['0', '0.5', '1'])
    for y in (0, 1):
        ax.axhline(y, color=PALE, lw=.5, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    return top


def evidence_axis(ax, labels=True, ticks=TICKS, rotation=0):
    for t in (common.INCLUDE, common.RETAIN):
        ax.axvline(np.sqrt(t), color=FAINT, lw=.6, linestyle=':', zorder=1)
    ax.set_xticks(np.sqrt(ticks))
    ax.set_xticklabels([f'{t:g}' for t in ticks] if labels else [], rotation=rotation,
                       ha='right' if rotation else 'center', rotation_mode='anchor' if rotation else None)


def effect_axis(ax, design):
    """Second x axis: the typical true effect at an evidence level as percent change in clearance, exp(k z) - 1
    with k the least-squares slope through the origin of beta on z_ref over the datasets with an effect."""
    pairs = [(d['z_ref'], d['beta']) for d in common.datasets(design) if d.get('beta') and d.get('z_ref')]
    if not pairs:
        return
    z, beta = np.array(pairs).T
    k = float(z @ beta / (z @ z))
    low, high = ax.get_xlim()
    percent = [p for p in (0, -10, -20, -30, -40, -50) if low <= np.log(1 + p / 100) / k <= high]
    top = ax.twiny()
    top.set_xlim(low, high)
    top.set_xticks([np.log(1 + p / 100) / k for p in percent])
    top.set_xticklabels([str(p).replace('-', '−') for p in percent])
    top.set_xlabel('typical true effect on clearance (%)')
    top.spines[['right']].set_visible(False)


def draw_curve(ax, c, color, key='p', band=True, **style):
    """A fitted curve of summary.json with its bootstrap band."""
    if not c:
        return
    g = c['grid']
    if band:
        ax.fill_between(_array(g['x']), _array(g[key + '_low']), _array(g[key + '_high']), color=color, alpha=.13,
                        lw=0, zorder=2)
    ax.plot(_array(g['x']), _array(g[key]), **dict(dict(color=color, lw=1.2, zorder=3), **style))


def baseline_step(ax, design):
    """The rule-following baseline of calibration step 6 (CL~target selected or not) as a step over z_ref."""
    points = sorted((d['z_ref'], float(d['baseline']['target_selected'])) for d in common.datasets(design)
                    if (d.get('baseline') or {}).get('target_selected') is not None)
    if points:
        ax.step(*np.array(points).T, where='mid', color=PALE, lw=2.4, zorder=1.5, solid_joinstyle='miter')


def run_segments(design, arm):
    """The evidence axis cut into ranges in which a condition was run or not: [(z_low, z_high, run)], with the
    cuts midway between neighbouring datasets. A2 conditions count as run on the slot types that hold an A2
    dataset (E0+ and E0- together)."""
    records = sorted(common.datasets(design), key=lambda d: d['z_ref'])
    statement, instruction = arm.split('-')
    if statement == 'twocmt':
        def group(d):
            return 'E0' if d['stratum'] == 'E0' else d['slot_type']
        held = {group(d) for d in records if d.get('a2')}
        state = [group(d) in held for d in records]
    else:
        state = [(statement, instruction) in common.A1_ARMS[d['slot_type']] for d in records]
    z = [d['z_ref'] for d in records]
    out, start = [], z[0] - .15
    for i in range(1, len(z) + 1):
        if i == len(z) or state[i] != state[i - 1]:
            end = z[-1] + .15 if i == len(z) else (z[i - 1] + z[i]) / 2
            out.append((start, end, state[i - 1]))
            start = end
    return out


def range_row(ax, design, arm, y):
    """One row above a panel: a line in the color of the condition over the ranges in which it was run, and
    'not run' over the others."""
    for low, high, run in run_segments(design, arm):
        if run:
            ax.plot([low, high], [y, y], color=ARM[arm][1], lw=1.8, solid_capstyle='butt', zorder=3)
        else:
            ax.plot([low, high], [y, y], color=FAINT, lw=.6, linestyle=(0, (1, 2.5)), zorder=3)
            if high - low > .8:
                ax.text((low + high) / 2, y, 'not run', ha='center', va='center', fontsize=7, color=MUTED,
                        style='italic', zorder=3.5, bbox=dict(facecolor='white', edgecolor='none', pad=1.2))


def dataset_points(ax, o, arm, endpoint, rows, lane, lanes):
    """One point per dataset at the number of adopting runs. The left half is GPT and the right half Claude:
    filled if that run adopted, white if not, pale if there is no accepted run."""
    color = ARM[arm][1]
    y = o.values(arm, endpoint)[rows]
    runs = np.sum(~np.isnan(y), axis=1)
    level = np.where(runs > 0, np.nansum(y, axis=1) / np.maximum(runs, 1), np.nan)
    for value in (0, .5, 1):
        at = np.flatnonzero(level == value)
        for x, pair in zip(spread(o.z[rows][at]), y[at]):
            faces = [color if v == 1 else 'white' if v == 0 else PALE for v in pair]
            ax.plot([x], [lane_y(value, lane, lanes)], marker='o', linestyle='', fillstyle='left', markersize=4.6,
                    markerfacecolor=faces[0], markerfacecoloralt=faces[1], markeredgecolor=color,
                    markeredgewidth=.5, zorder=4, clip_on=False)


def run_points(ax, x, y, color, lane, lanes, llms=analyze.LLMS, gap=GAP):
    """One point per run (x and y: datasets x LLMs): filled above the panel if the run adopted, open below it if
    not; the marker tells the LLM. Points of a lane closer than gap are moved apart."""
    for value in (0, 1):
        at = (y == value) & ~np.isnan(x)
        position = np.full(x.shape, np.nan)
        position[at] = spread(x[at], gap)
        for j, llm in enumerate(llms):
            ax.plot(position[at[:, j], j], np.full(at[:, j].sum(), lane_y(value, lane, lanes)), marker=MARK[llm],
                    linestyle='', markersize=3.6, markerfacecolor=color if value else 'white',
                    markeredgecolor=color, markeredgewidth=.5, zorder=4, clip_on=False)


def run_handles():
    """Legend entries of the figures with one point per run."""
    def point(marker, face, label):
        return Line2D([], [], marker=marker, linestyle='', markersize=4, markerfacecolor=face, markeredgecolor=MUTED,
                      label=label)
    return [point(MARK[m], MUTED, LLM[m]) for m in LLM] + [
        point('o', MUTED, 'filled, above 1: in the final model'),
        point('o', 'white', 'open, below 0: not in the final model')]


def arm_handle(arm):
    return Line2D([], [], color=ARM[arm][1], lw=1.2, label=ARM[arm][0])


def extra_handles(spec):
    out = []
    if spec.get('twin'):
        out.append(Line2D([], [], color=MUTED, lw=.9, linestyle='--', label='twin comedication,\nthese conditions'))
    if spec.get('baseline'):
        out.append(Line2D([], [], color=PALE, lw=2.4, label='rule-following baseline\n(no LLM)'))
    return out + [Line2D([], [], color=FAINT, lw=.6, linestyle=':', label='ΔOFV 3.84 and 6.63')]


def panel(ax, o, design, summary, spec, points, key='p'):
    """Curves, points, baseline, twin line and range rows of one panel. points(arm, rows, lane, lanes) draws the
    points of a condition; key is the curve of summary.json ('p': mean of the LLMs, 'gpt', 'claude')."""
    arms = [a for a in spec['arms'] if summary['curves'].get(spec.get('curves', {}).get(a, a))]
    top = frame(ax, len(arms), len(arms) - 1)
    rows = np.array([o.row[k] for k in analyze.members(design, spec.get('datasets', 'all'))], dtype=int)
    if spec.get('baseline'):
        baseline_step(ax, design)
    for lane, arm in enumerate(arms):
        draw_curve(ax, summary['curves'][spec.get('curves', {}).get(arm, arm)], ARM[arm][1], key)
        points(arm, rows, lane, len(arms))
    if spec.get('twin'):
        draw_curve(ax, summary['curves'].get(spec['twin']), MUTED, key, band=False, lw=.9, linestyle='--')
    for r, arm in enumerate(arms[1:]):
        range_row(ax, design, arm, top + ROW * r)
    ax.set_xlim(*XLIM)
    return arms


def figure_evidence(o, design, summary, out):
    """Headline figure of section 8.4."""
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(WIDTH, 8.6), sharex=True)
    for n, (ax, spec) in enumerate(zip(axes, PANELS)):
        arms = panel(ax, o, design, summary, spec,
                     lambda arm, rows, lane, lanes: dataset_points(ax, o, arm, spec['endpoint'], rows, lane, lanes))
        evidence_axis(ax, labels=n == len(PANELS) - 1)
        for value, text in ((0, '0 of 2'), (.5, '1 of 2'), (1, '2 of 2')):
            ax.text(1.012, lane_y(value, (len(arms) - 1) / 2, len(arms)), text, transform=ax.get_yaxis_transform(),
                    va='center', fontsize=7, color=MUTED)
        ax.set_ylabel(spec['ylabel'])
        ax.set_title(spec['title'], loc='left', pad=4 if n else 30)
        ax.text(-.105, 1.02 if n else 1.16, 'abc'[n], transform=ax.transAxes, fontsize=10, fontweight='bold')
        ax.legend(handles=[arm_handle(a) for a in arms] + extra_handles(spec), loc='upper left',
                  bbox_to_anchor=(1.115, 1.0), frameon=False, handlelength=1.8, borderaxespad=0, labelspacing=.6)
    effect_axis(axes[0], design)
    axes[-1].set_xlabel(XLABEL)
    halves = (((INK, INK), 'both LLMs'), ((INK, 'white'), f"{LLM['gpt']} only"),
              (('white', INK), f"{LLM['claude']} only"), (('white', 'white'), 'neither'))
    handles = [Line2D([], [], marker='o', linestyle='', fillstyle='left', markersize=5.5, markerfacecolor=f[0],
                      markerfacecoloralt=f[1], markeredgecolor=INK, markeredgewidth=.5, label=label)
               for f, label in halves]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.075, .004), ncol=4, frameon=False,
               columnspacing=1.4, handletextpad=.3, alignment='left',
               title='One point per dataset and condition, at the number of adopting LLM runs; adopted by:')
    fig.subplots_adjust(left=.085, right=.675, top=.925, bottom=.135, hspace=.2)
    return _save(fig, out, 'adoption_evidence')


def figure_by_llm(o, design, summary, out):
    """The headline figure by LLM (supplement): one column per LLM, one point per run."""
    fig, axes = plt.subplots(len(PANELS), len(LLM), figsize=(WIDTH, 9.4), sharex=True, sharey='row')
    used = []
    for n, spec in enumerate(PANELS):
        for j, llm in enumerate(analyze.LLMS):
            ax = axes[n, j]

            def points(arm, rows, lane, lanes):
                run_points(ax, o.z[rows][:, None], o.values(arm, spec['endpoint'])[rows][:, [j]], ARM[arm][1], lane,
                           lanes, llms=(llm,))
            arms = panel(ax, o, design, summary, spec, points, key=llm)
            used += [a for a in arms if a not in used]
            evidence_axis(ax, labels=n == len(PANELS) - 1, rotation=45)
            ax.set_title(f"{spec['title']}: {LLM[llm]}", loc='left', fontsize=8, pad=4 if n else 30)
            if j == 0:
                ax.set_ylabel(spec['ylabel'])
    for ax in axes[0]:
        effect_axis(ax, design)
    fig.supxlabel(XLABEL, fontsize=8, y=.168)
    handles = [arm_handle(a) for a in used] + extra_handles(dict(twin=True, baseline=True)) + run_handles()[2:]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, columnspacing=1.6, handlelength=1.8,
               labelspacing=.6)
    fig.subplots_adjust(left=.085, right=.985, top=.935, bottom=.24, hspace=.22, wspace=.06)
    return _save(fig, out, 'adoption_evidence_by_llm')


def figure_agent_evidence(o, design, summary, out):
    """Adoption against the evidence each run obtained itself (section 8.8): d_final of the run, d_max where it
    is absent; runs that did not test the effect are drawn at the left. One point per run."""
    curves = summary['displays']['agent_evidence']['curves']
    high = max([XLIM[1]] + [c['x_range'][1] + .3 for c in curves.values() if c])
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 5.8), sharex=True)
    for n, (ax, spec) in enumerate(zip(axes, PANELS[:2])):
        arms = [a for a in spec['arms'] if curves.get(a)]
        frame(ax, len(arms), 0)
        for lane, arm in enumerate(arms):
            draw_curve(ax, curves[arm], ARM[arm][1])
            d = analyze.agent_evidence(o, arm)
            y = o.values(arm, 'Y')
            run_points(ax, np.sqrt(np.clip(d, 0, None)), y, ARM[arm][1], lane, len(arms))
            run_points(ax, np.where(np.isnan(d) & ~np.isnan(y), -1.2, np.nan), y, ARM[arm][1], lane, len(arms),
                       gap=.045)
        ax.axvline(-.6, color=PALE, lw=.6)
        ax.set_xlim(-1.9, high)
        evidence_axis(ax, labels=False)
        ax.set_ylabel(spec['ylabel'])
        ax.set_title(spec['title'], loc='left')
        ax.text(-.105, 1.02, 'ab'[n], transform=ax.transAxes, fontsize=10, fontweight='bold')
        ax.legend(handles=[arm_handle(a) for a in arms] + extra_handles({}), loc='upper left',
                  bbox_to_anchor=(1.03, 1.0), frameon=False, handlelength=1.8, borderaxespad=0, labelspacing=.6)
    ticks = [t for t in TICKS if np.sqrt(t) <= high]
    axes[-1].set_xticks([-1.2] + list(np.sqrt(ticks)))
    axes[-1].set_xticklabels(['not\ntested'] + [f'{t:g}' for t in ticks])
    axes[-1].set_xlabel('ΔOFV for the target effect in the models of the run itself (square-root scale)')
    fig.legend(handles=run_handles(), loc='lower left', bbox_to_anchor=(.075, .005), ncol=2, frameon=False,
               columnspacing=1.4, handletextpad=.3, title='One point per run', alignment='left')
    fig.subplots_adjust(left=.085, right=.72, top=.95, bottom=.215, hspace=.22)
    return _save(fig, out, 'adoption_agent_evidence')


def figure_grid(o, design, out):
    """Every outcome by dataset (rows, in the order of the evidence) and condition x LLM (columns): dark = the
    effect of the condition is in the final model (the target effect; two compartments in the A2 conditions),
    white = it is not, a cross = no accepted attempt, no cell = not run. A dot marks a final model with an effect
    of the twin comedication. The first column is the rule-following baseline without an LLM."""
    arms = [a for a in analyze.ARMS if np.any(o.status(a) > 0)]
    order = list(np.argsort(o.z, kind='stable'))
    left, spans, x = {}, {}, 1.7
    for label, statement in GRID_GROUPS:
        own = [a for a in arms if a.split('-')[0] == statement]
        for arm in own:
            left[arm] = x
            x += 2.3
        if own:
            spans[label] = (left[own[0]], left[own[-1]] + 1.9)
            x += .6
    right = x - .4
    y, ys, previous = 0., [], None
    for i in order:
        y += .45 if previous not in (None, o.records[i]['slot_type']) else 0.
        ys.append(y)
        previous = o.records[i]['slot_type']
        y += 1.
    fig, ax = plt.subplots(figsize=(WIDTH, 1.5 + .2 * len(order)))

    def cell(x, y, value):
        ax.add_patch(Rectangle((x, y), .9, .9, facecolor=INK if value == 1 else 'white', edgecolor=MUTED, lw=.6))
    for y, i in zip(ys, order):
        d = o.records[i]
        selected = (d.get('baseline') or {}).get('target_selected')
        if selected is not None:
            cell(0, y, float(selected))
        ax.text(-.5, y + .45, f"{d['id']}   {d['slot_type']}", ha='right', va='center')
        ax.text(-3.9, y + .45, '' if d.get('dofv_ref') is None else f"{d['dofv_ref']:.1f}", ha='right', va='center')
        for arm in arms:
            column = 'two_cmt' if arm.startswith('twocmt') else 'Y'
            for j in range(len(analyze.LLMS)):
                x, status = left[arm] + j, o.status(arm)[i, j]
                if status == 0:
                    continue
                value = o.raw(arm, column)[i, j] if status == 1 else 0
                cell(x, y, value)
                if status == 2:
                    ax.plot([x + .2, x + .7, np.nan, x + .2, x + .7], [y + .2, y + .7, np.nan, y + .7, y + .2],
                            color=MUTED, lw=.7)
                elif o.raw(arm, 'twin_Y')[i, j] == 1:
                    ax.plot([x + .45], [y + .45], marker='o', markersize=2.2, color='white' if value == 1 else INK)
    for label, (low, high) in spans.items():
        ax.text((low + high) / 2, -2.75, label, ha='center', va='bottom', fontsize=7.5, linespacing=1.05)
        ax.plot([low, high], [-2.55, -2.55], color=MUTED, lw=.6)
    for arm in arms:
        ax.text(left[arm] + .95, -1.35, GRID_PROMPT[arm.split('-')[1]], ha='center', va='bottom', fontsize=7)
        for j, llm in enumerate(analyze.LLMS):
            ax.text(left[arm] + j + .45, -.3, 'G' if llm == 'gpt' else 'C', ha='center', va='bottom', fontsize=7)
    ax.text(.45, -.3, 'rule', ha='center', va='bottom', fontsize=7)
    ax.text(-.5, -.3, 'dataset   slot', ha='right', va='bottom', fontsize=7)
    ax.text(-3.9, -.3, 'ΔOFV', ha='right', va='bottom', fontsize=7)
    ax.set_xlim(-5.3, right)
    ax.set_ylim(ys[-1] + 1.2, -4.6)
    ax.axis('off')
    handles = [Rectangle((0, 0), 1, 1, facecolor=INK, edgecolor=MUTED, lw=.6,
                         label='in the final model: the target effect\n(two compartments stated: two compartments)'),
               Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=MUTED, lw=.6, label='not in the final model'),
               Line2D([], [], marker='x', linestyle='', markersize=5, color=MUTED, label='no accepted attempt'),
               Line2D([], [], marker='o', linestyle='', markersize=2.2, color=INK,
                      label='effect of the twin comedication\nin the final model')]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.03, .042), ncol=4, frameon=False, fontsize=7,
               columnspacing=1.5, handlelength=1.2, handleheight=1.2)
    fig.text(.04, .008, f"rule: rule-following baseline without an LLM. G {LLM['gpt']}, C {LLM['claude']}. "
                        'No cell: not run.\nPrompt: benchmark paragraph, neutral paragraph, or neutral paragraph '
                        'with the explicit rule.', fontsize=7, color=MUTED, linespacing=1.3)
    fig.subplots_adjust(left=.01, right=.99, top=.995, bottom=.095)
    return _save(fig, out, 'outcome_grid')


def draw(design, runs_path, summary_path, out):
    """Draw all figures; returns the files written."""
    o = analyze.Outcomes(analyze.read_runs(runs_path), design)
    summary = json.loads(Path(summary_path).read_text(encoding='utf-8'))
    out = Path(out)
    return (figure_evidence(o, design, summary, out) + figure_by_llm(o, design, summary, out)
            + figure_agent_evidence(o, design, summary, out) + figure_grid(o, design, out))


def main(argv=None):
    ap = argparse.ArgumentParser(description='Figures of the simulation experiment (section 8.4).')
    ap.add_argument('--design', help='design file (default: design.json)')
    ap.add_argument('--runs', help='runs.csv (default: evaluation/runs.csv below the root of the design)')
    ap.add_argument('--summary', help='summary.json of analyze.py (default: next to runs.csv)')
    ap.add_argument('--out', help='output folder (default: figures/ next to summary.json)')
    a = ap.parse_args(argv)
    path = common.design_path(a.design)
    if not path.exists():
        ap.error(f'{path} does not exist (--design)')
    design = common.load_design(path)
    runs = Path(a.runs) if a.runs else common.folders(design)['evaluation'] / 'runs.csv'
    summary = Path(a.summary) if a.summary else runs.parent / 'summary.json'
    for needed in (runs, summary):
        if not needed.exists():
            ap.error(f'{needed} does not exist: score.py writes runs.csv, analyze.py writes summary.json')
    for p in draw(design, runs, summary, Path(a.out) if a.out else summary.parent / 'figures'):
        print(p)
    return 0


if __name__ == '__main__':
    sys.exit(main())
