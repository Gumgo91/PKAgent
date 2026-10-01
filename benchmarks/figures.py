"""Result figures of the benchmark (reads benchmarks/evaluation/runs.csv and the run folders).

Figure 2  recovery of the reference model: structure, covariate relationships and typical values (median ratio
          final/reference per parameter), by dataset, condition and LLM.
Figure 3  OFV of the final model minus the OFV of the reference model (PKPy2 Laplace fits, same data).
Figure 4  model-development trajectories: OFV of every fitted model in the order of fitting.
Figure 5  cost, LLM responses, fits and wall-clock hours per run.
Outputs: paper/figures/Figure_<n>.png and .pdf.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / 'paper' / 'figures'
DATASETS = json.loads((HERE / 'datasets.json').read_text(encoding='utf-8'))
LABEL = dict(pheno='Phenobarbital', remifentanil='Remifentanil', oral_mm='Oral MM (simulated)')
LLM = dict(gpt='GPT-6.1 Sol', claude='Claude Opus 5.5')
COND = dict(none='No knowledge', knowledge='Expert sentence')
COLOR = dict(none='#8A94A6', knowledge='#C8801E')
MARK = dict(gpt='o', claude='s')


def _save(fig, n):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f'Figure_{n}.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG / f'Figure_{n}.pdf', bbox_inches='tight')
    plt.close(fig)


def _range(text):
    lo, hi = str(text).split('-', 1) if isinstance(text, str) and '-' in str(text)[1:] else (text, text)
    return float(lo), float(hi)


def figure_recovery(runs):
    """Subject-level typical-value ratio (final model / reference model) of each reference parameter: a line from the
    lowest to the highest ratio over subjects and a marker at the median, one row per run, one panel per dataset."""
    from matplotlib.ticker import FixedLocator, NullLocator
    names = [d for d in DATASETS if d in set(runs['dataset'])]
    path = HERE / 'evaluation' / 'reference_fit_ratios.json'
    reference = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    order = dict(pheno=['CL', 'V'], remifentanil=['CL', 'V1', 'Q2', 'V2', 'Q3', 'V3'],
                 oral_mm=['Ka', 'V', 'VMAX', 'KM'])
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 5.2), squeeze=False)
    for ax, ds in zip(axes[0], names):
        sub = runs[runs['dataset'] == ds]
        params = [p for p in order.get(ds, []) if f'ratio_{p}_median' in sub and sub[f'ratio_{p}_median'].notna().any()]
        for i, p in enumerate(params):
            if p in reference.get(ds, {}):
                v = reference[ds][p]['median']
                ax.plot([v, v], [i - .42, i + .42], color='#1F2933', lw=1.6, zorder=4, solid_capstyle='butt')
        slots = [(c, m) for c in COND for m in LLM]
        for i, p in enumerate(params):
            k = 0
            for cond, llm in slots:
                cell = sub[(sub['condition'] == cond) & (sub['llm'] == llm)].sort_values('rep')
                for _, r in cell.iterrows():
                    if pd.isna(r.get(f'ratio_{p}_median')):
                        continue
                    lo, hi = _range(r[f'ratio_{p}_range'])
                    y = i + (k - 5.5) * .065
                    ax.plot([lo, hi], [y, y], color=COLOR[cond], lw=1.2, alpha=.9, solid_capstyle='round')
                    ax.scatter([r[f'ratio_{p}_median']], [y], marker=MARK[llm], s=22, color=COLOR[cond],
                               edgecolor='white', linewidth=.4, zorder=3)
                    k += 1
        ax.axvline(1, color='#1F2933', lw=.8)
        ax.axvspan(.8, 1.25, color='#E8F1FB', zorder=0)
        ax.set_xscale('log')
        ticks = [.5, .67, .8, 1, 1.25, 1.5, 2]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f'{t:g}' for t in ticks])
        ax.set_xlim(.45, 2.2)
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels(params)
        ax.set_ylim(len(params) - .5, -.5)
        ax.set_title(LABEL.get(ds, ds), fontsize=11)
        ax.set_xlabel('typical value, final / reference model')
    handles = [plt.Line2D([], [], marker=MARK[m], color=COLOR[c], linestyle='-', label=f'{LLM[m]}, {COND[c].lower()}')
               for c in COND for m in LLM]
    handles.append(plt.Line2D([], [], marker='|', markersize=12, markeredgewidth=1.6, color='#1F2933', linestyle='',
                              label='PKPy2 fit of the reference model'))
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(.5, -.05), fontsize=8.5)
    _save(fig, 2)


def figure_delta_ofv(runs):
    names = [d for d in DATASETS if d in set(runs['dataset'])]
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.6), squeeze=False)
    for ax, ds in zip(axes[0], names):
        sub = runs[runs['dataset'] == ds]
        for k, (cond, llm) in enumerate([(c, m) for c in COND for m in LLM]):
            vals = sub[(sub['condition'] == cond) & (sub['llm'] == llm)]['delta_ofv_vs_reference'].dropna()
            ax.scatter([k] * len(vals), vals, marker=MARK[llm], color=COLOR[cond], s=30, zorder=3)
        ax.axhline(0, color='#1F2933', lw=.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels([f'{LLM[m].split()[0]}\n{COND[c].split()[0].lower()}' for c in COND for m in LLM],
                           fontsize=8)
        ax.set_title(LABEL.get(ds, ds), fontsize=11)
        ax.set_ylabel('OFV final - OFV reference')
    _save(fig, 3)


def trajectory(run_dir):
    """Converged models of a run in the order of fitting (the model registry of results.json)."""
    res = json.loads((run_dir / 'results.json').read_text(encoding='utf-8'))
    rows = [dict(model=m['model_id'], ofv=m['ofv']) for m in res.get('models', [])
            if m.get('status') == 'converged' and m.get('ofv') is not None]
    return pd.DataFrame(rows)


def figure_trajectories(runs):
    names = [d for d in DATASETS if d in set(runs['dataset'])]
    ref = {d: json.loads((HERE / 'reference_fits' / d / 'reference_fit.json').read_text(encoding='utf-8'))['ofv']
           for d in names if (HERE / 'reference_fits' / d / 'reference_fit.json').exists()}
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.8), squeeze=False)
    for ax, ds in zip(axes[0], names):
        for _, r in runs[runs['dataset'] == ds].iterrows():
            t = trajectory(HERE / 'runs' / ds / r['condition'] / r['llm'] / r['rep'])
            if t.empty:
                continue
            best = t['ofv'].cummin()
            ax.step(range(1, len(best) + 1), best, where='post', color=COLOR[r['condition']], alpha=.8,
                    linestyle='-' if r['llm'] == 'claude' else '--', lw=1.2)
        if ds in ref:
            ax.axhline(ref[ds], color='#1F2933', lw=.8, linestyle=':')
        ax.set_title(LABEL.get(ds, ds), fontsize=11)
        ax.set_xlabel('fitted models')
        ax.set_ylabel('lowest OFV so far')
    _save(fig, 4)


def figure_process(runs):
    cols = [('cost_usd', 'cost (USD)'), ('llm_calls', 'LLM responses'), ('fits', 'fits'), ('hours', 'hours')]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.3 * len(cols), 3.3))
    names = [d for d in DATASETS if d in set(runs['dataset'])]
    for ax, (col, lab) in zip(axes, cols):
        for i, ds in enumerate(names):
            for j, (cond, llm) in enumerate([(c, m) for c in COND for m in LLM]):
                v = runs[(runs['dataset'] == ds) & (runs['condition'] == cond) & (runs['llm'] == llm)][col].dropna()
                ax.scatter([i + (j - 1.5) * .15] * len(v), v, marker=MARK[llm], color=COLOR[cond], s=24)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([LABEL[d].split()[0] for d in names], fontsize=8)
        ax.set_ylabel(lab)
    _save(fig, 5)


def main():
    runs = pd.read_csv(HERE / 'evaluation' / 'runs.csv')
    figure_recovery(runs)
    figure_delta_ofv(runs)
    figure_trajectories(runs)
    figure_process(runs)


if __name__ == '__main__':
    main()
