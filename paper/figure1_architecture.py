"""Figure 1: PKAgent architecture, drawn at the printed size of a double-column CPT figure (178 mm wide, text >= 8 pt).

Outputs paper/figures/Figure_1.pdf, .png and .tiff (CMYK).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / 'figures'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
INK, MUTED = '#1F2933', '#3E4C59'
FILL = dict(input='#E8F1FB', llm='#FDF1E3', session='#EEF6EE', engine='#F3EEF8', output='#F4F5F7')
EDGE = dict(input='#4A7FB5', llm='#C8801E', session='#4C8C4A', engine='#7A5BA6', output='#7B8794')
LINE = 0.165                                   # line height in inches for 8-point text


def box(ax, x, y, w, h, kind, title, lines=()):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06', linewidth=1.0,
                                edgecolor=EDGE[kind], facecolor=FILL[kind]))
    ax.text(x + w / 2, y + h - 0.08, title, ha='center', va='top', fontsize=9, fontweight='bold', color=INK)
    for k, line in enumerate(lines):
        bold = line.startswith('**')
        ax.text(x + 0.09, y + h - 0.36 - k * LINE, line.strip('*'), ha='left', va='top', fontsize=8,
                color=INK if bold else MUTED, fontweight='bold' if bold else 'normal')


def arrow(ax, a, b, text=None, both=False, dx=0., dy=0.05, ha='center'):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='<|-|>' if both else '-|>', mutation_scale=9, linewidth=0.9,
                                 color=INK))
    if text:
        ax.text((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2 + dy, text, ha=ha, va='bottom', fontsize=8, color=INK,
                style='italic')


def main():
    OUT.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(7.0, 4.6))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 7.0)
    ax.set_ylim(0, 4.6)
    ax.axis('off')
    box(ax, 0.05, 3.12, 2.15, 1.43, 'input', 'Analysis inputs',
        ['data file (NONMEM format)', 'study description: design,', '  units, columns', 'optional expert knowledge',
         '  (one sentence)', 'budgets: fits, turns, hours, cost'][:6])
    box(ax, 0.05, 1.55, 2.15, 1.30, 'llm', 'Language model',
        ['GPT-6.1 Sol or Claude Opus 5.5', 'system prompt: workflow,', '  decision criteria, conventions',
         'reads tool results and plots', 'never writes or runs code'])
    box(ax, 0.05, 0.05, 2.15, 1.25, 'output', 'Outputs of a run',
        ['final model and structured report', 'every model: specification, fit,', '  diagnostics, plots',
         'transcript and tool log', 'tokens, cost, time'])
    box(ax, 2.80, 0.05, 2.25, 4.50, 'session', 'PKAgent session (14 tools)',
        ['**Data', 'describe_data, plot_data,', 'run_nca, add_data_column', '**Models',
         'fit_models (≤ 6 validated JSON', '  specifications in parallel),', 'list_models, get_model,',
         'compare_models', '**Diagnostics', 'view_plots, screen_covariates,', 'run_vpc',
         '**Covariates and uncertainty', 'covariate_search,', 'resample_uncertainty', '**Report', 'finalize_model',
         '', 'versioned data, model registry,', 'budgets, audit trail'])
    box(ax, 5.38, 2.40, 1.57, 2.15, 'engine', 'PKPy2 engine',
        ['event-record models', 'Laplace (FOCE-I type)', '  objective, L-BFGS-B', 'conditional-mode', '  search',
         'SEs, CWRES, NPDE,', '  VPC, bootstrap', 'SCM'])
    arrow(ax, (1.12, 3.10), (1.12, 2.88), 'task', dx=0.08, dy=-0.07, ha='left')
    arrow(ax, (2.24, 2.42), (2.76, 2.42), 'calls', dy=0.04)
    arrow(ax, (2.76, 2.02), (2.24, 2.02), 'results,', dy=0.04)
    ax.text(2.50, 1.84, 'images', ha='center', va='bottom', fontsize=8, color=INK, style='italic')
    arrow(ax, (5.07, 3.45), (5.36, 3.45), 'fits', both=True, dy=0.05)
    arrow(ax, (2.76, 0.68), (2.24, 0.68), 'final', dy=0.04)
    fig.savefig(OUT / 'Figure_1.pdf')
    fig.savefig(OUT / 'Figure_1.png', dpi=600)
    plt.close(fig)
    from figures import tiff_cmyk
    tiff_cmyk(OUT / 'Figure_1.png', OUT / 'Figure_1.tiff')


if __name__ == '__main__':
    main()
