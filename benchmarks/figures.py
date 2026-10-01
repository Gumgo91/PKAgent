"""Result figures of the benchmark (reads benchmarks/evaluation/runs.csv, evaluation.json, effect_evidence.json and
the run folders).

Figure 2  typical values of the final models relative to the reference model (per-subject ratios), by run.
Figure 3  covariate relationships of the reference models recovered by each run, with their likelihood evidence.
Figure 4  model development (lowest OFV so far relative to the reference fit) and resources per run.
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
COND = dict(none='No knowledge', knowledge='Expert statement')
COLOR = dict(none='#8A94A6', knowledge='#C8801E')
MARK = dict(gpt='o', claude='s')


plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.linewidth': .6, 'lines.linewidth': 1.,
                     'xtick.major.width': .6, 'ytick.major.width': .6, 'pdf.fonttype': 42})
WIDTH = 7.0                       # inches: a double-column CPT figure (178 mm)


def tiff_cmyk(png, tiff):
    """CMYK TIFF (LZW) from a 600-dpi PNG, as CPT asks for color figures."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(png) as im:
        im.convert('CMYK').save(tiff, compression='tiff_lzw', dpi=(600, 600))


def _save(fig, n):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f'Figure_{n}.pdf')
    fig.savefig(FIG / f'Figure_{n}.png', dpi=600)
    plt.close(fig)
    tiff_cmyk(FIG / f'Figure_{n}.png', FIG / f'Figure_{n}.tiff')


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
    fig, axes = plt.subplots(1, len(names), figsize=(WIDTH, 3.9), squeeze=False)
    for ax, ds in zip(axes[0], names):
        sub = runs[runs['dataset'] == ds]
        params = [p for p in order.get(ds, []) if f'ratio_{p}_median' in sub and sub[f'ratio_{p}_median'].notna().any()]
        for i, p in enumerate(params):
            if p in reference.get(ds, {}):
                v = reference[ds][p]['median']
                ax.plot([v, v], [i - .42, i + .42], color='#1F2933', lw=1.0, zorder=4, solid_capstyle='butt')
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
                    ax.plot([lo, hi], [y, y], color=COLOR[cond], lw=.7, alpha=.9, solid_capstyle='round')
                    ax.scatter([r[f'ratio_{p}_median']], [y], marker=MARK[llm], s=9, color=COLOR[cond],
                               edgecolor='white', linewidth=.4, zorder=3)
                    k += 1
        ax.axvline(1, color='#1F2933', lw=.8)
        ax.axvspan(.8, 1.25, color='#E8F1FB', zorder=0)
        ax.set_xscale('log')
        ticks = [.125, .25, .5, 1, 2]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f'{t:g}' for t in ticks])
        ax.set_xlim(.1, 2.2)
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels(params)
        ax.set_ylim(len(params) - .5, -.5)
        ax.set_title(LABEL.get(ds, ds))
        ax.set_xlabel('typical value ratio\n(final / reference model)')
    handles = [plt.Line2D([], [], marker=MARK[m], color=COLOR[c], linestyle='-', label=f'{LLM[m]}, {COND[c].lower()}')
               for c in COND for m in LLM]
    handles.append(plt.Line2D([], [], marker='|', markersize=12, markeredgewidth=1.6, color='#1F2933', linestyle='',
                              label='PKPy2 fit of the reference model'))
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, 0), columnspacing=1.2,
               handlelength=1.6)
    fig.tight_layout(rect=(0, .13, 1, 1), w_pad=.6)
    _save(fig, 2)


REF_FORMS = {
    'pheno': {('CL', 'WT'): 'power', ('V', 'WT'): 'power', ('V', 'APGR'): 'categorical'},
    'remifentanil': {(p, c): 'linear' for p, c in (('V1', 'AGE'), ('V1', 'LBM'), ('V2', 'AGE'), ('V2', 'LBM'),
                                                   ('CL', 'AGE'), ('CL', 'LBM'), ('Q2', 'AGE'), ('Q3', 'AGE'))},
}


def figure_covariates(runs):
    """Covariate recovery: reference relationships (rows, with the OFV increase on removing each one from the
    reference fit) against runs (columns); dark = present with the reference form, light = present with another
    form, white = absent; the last row counts relationships that are not in the reference model."""
    from matplotlib.patches import Rectangle
    ev = HERE / 'evaluation' / 'evaluation.json'
    details = json.loads(ev.read_text(encoding='utf-8'))['details'] if ev.exists() else {}
    evidence_path = HERE / 'evaluation' / 'effect_evidence.json'
    evidence = json.loads(evidence_path.read_text(encoding='utf-8')) if evidence_path.exists() else {}
    names = [d for d in REF_FORMS if d in set(runs['dataset'])]
    fig, axes = plt.subplots(len(names), 1, figsize=(WIDTH, .5 + .21 * sum(len(REF_FORMS[d]) + 3 for d in names)),
                             squeeze=False, gridspec_kw=dict(height_ratios=[len(REF_FORMS[d]) + 1 for d in names]))
    for ax, ds in zip(axes[:, 0], names):
        sub = runs[runs['dataset'] == ds].copy()
        sub['order'] = sub['condition'].map({'none': 0, 'knowledge': 1}) * 10 + sub['llm'].map({'gpt': 0, 'claude': 1}) * 5
        sub = sub[sub['condition'].isin(['none', 'knowledge'])].sort_values(['order', 'rep'])
        dev = {r['effect'].split('_')[0]: r['delta_ofv']           # V~APGR_LT5 -> V~APGR (derived column)
               for r in evidence.get(ds, {}).get('removals', [])}
        rels = sorted(REF_FORMS[ds], key=lambda k: -(dev.get(f'{k[0]}~{k[1]}') or 1e9))
        for j, (_, r) in enumerate(sub.iterrows()):
            key = next((k for k in details if k.replace('\\', '/') == f"{ds}/{r['condition']}/{r['llm']}/{r['rep']}"), None)
            found = {(p, c): set(f) for p, c, f in (details.get(key) or {}).get('relationships', [])}
            for i, rel in enumerate(rels):
                forms = found.get(rel)
                color = 'white' if not forms else ('#1F2933' if REF_FORMS[ds][rel] in forms else '#9AA5B1')
                ax.add_patch(Rectangle((j, i), .9, .9, facecolor=color, edgecolor='#52606D', lw=.6))
            extra = len([k for k in found if k not in REF_FORMS[ds]])
            ax.text(j + .45, len(rels) + .45, str(extra), ha='center', va='center', fontsize=8,
                    color='#B44D12')
        labels = []
        for p, c in rels:
            d = dev.get(f'{p}~{c}')
            labels.append(f'{p}~{c}' + (f'  (ΔOFV {d:.1f})' if d is not None else ''))
        ax.set_yticks([i + .45 for i in range(len(rels) + 1)])
        ax.set_yticklabels(labels + ['other relationships'])
        ax.set_xticks([j + .45 for j in range(len(sub))])
        ax.set_xticklabels([f"{'G' if m == 'gpt' else 'C'}{rep[-1]}" for m, rep in zip(sub['llm'], sub['rep'])],
                           fontsize=8)
        ncond = (sub['condition'] == 'none').sum()
        ax.axvline(ncond - .05, color='#C8801E', lw=1.0)
        ax.text(ncond / 2, -.45, 'No knowledge', ha='center')
        ax.text(ncond + (len(sub) - ncond) / 2, -.45, 'Expert statement', ha='center', color='#C8801E')
        ax.set_xlim(-.2, len(sub) + .1)
        ax.set_ylim(len(rels) + 1, -1)
        ax.set_title(LABEL.get(ds, ds), loc='left', pad=12)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout(h_pad=1.4)
    _save(fig, 3)


def trajectory(run_dir):
    """Converged models of a run in the order of fitting (the model registry of results.json) and the final model."""
    res = json.loads((run_dir / 'results.json').read_text(encoding='utf-8'))
    rows = [dict(model=m['model_id'], ofv=m['ofv']) for m in res.get('models', [])
            if m.get('status') == 'converged' and m.get('ofv') is not None]
    return pd.DataFrame(rows), (res.get('final_model') or {}).get('model_id')


def figure_process(runs):
    """Model development (upper row: lowest OFV so far minus the OFV of the reference model, in the order of fitting;
    the final model is marked) and resources per run (lower row: fitted models, wall-clock hours, LLM cost)."""
    names = [d for d in DATASETS if d in set(runs['dataset'])]
    ref = {d: json.loads((HERE / 'reference_fits' / d / 'reference_fit.json').read_text(encoding='utf-8'))['ofv']
           for d in names}
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 5.0))
    for ax, ds in zip(axes[0], names):
        for _, r in runs[(runs['dataset'] == ds) & runs['condition'].isin(list(COND))].iterrows():
            t, final = trajectory(HERE / 'runs' / ds / r['condition'] / r['llm'] / r['rep'])
            if t.empty:
                continue
            d = t['ofv'].cummin() - ref[ds]
            x = np.arange(1, len(d) + 1)
            ax.step(x, d, where='post', color=COLOR[r['condition']], alpha=.75, lw=.8,
                    linestyle='-' if r['llm'] == 'claude' else '--')
            if final in set(t['model']):
                k = int(np.flatnonzero(t['model'].to_numpy() == final)[0])
                ax.scatter([k + 1], [t['ofv'].iloc[k] - ref[ds]], marker=MARK[r['llm']], s=14, zorder=3,
                           facecolor=COLOR[r['condition']], edgecolor='#1F2933', lw=.6)
        ax.axhline(0, color='#1F2933', lw=.8, linestyle=':')
        ax.set_yscale('symlog', linthresh=1)
        ax.set_title(LABEL.get(ds, ds), loc='left')
        ax.set_xlabel('models fitted')
    axes[0, 0].set_ylabel('lowest OFV so far − reference OFV')
    cols = [('fits', 'models fitted'), ('hours', 'wall-clock hours'), ('cost_usd', 'LLM cost (USD)')]
    for ax, (col, lab) in zip(axes[1], cols):
        for i, ds in enumerate(names):
            for j, (cond, llm) in enumerate([(c, m) for c in COND for m in LLM]):
                v = runs[(runs['dataset'] == ds) & (runs['condition'] == cond) & (runs['llm'] == llm)][col].dropna()
                ax.scatter(i + (j - 1.5) * .16 + np.linspace(-.03, .03, len(v)), v, marker=MARK[llm], s=12,
                           facecolor=COLOR[cond], edgecolor='#1F2933', lw=.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([LABEL[d].replace(' (simulated)', '') for d in names], rotation=20)
        ax.set_ylabel(lab)
        ax.set_ylim(bottom=0)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLOR[c], lw=2, label=COND[c]) for c in COND] + \
              [Line2D([], [], color='#52606D', marker=MARK[m], linestyle='--' if m == 'gpt' else '-', label=LLM[m])
               for m in LLM]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False)
    for k, ax in enumerate(axes.flat):
        ax.text(-.2, 1.05, 'abcdef'[k], transform=ax.transAxes, fontsize=10, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(rect=(0, .05, 1, 1))
    _save(fig, 4)


def main():
    runs = pd.read_csv(HERE / 'evaluation' / 'runs.csv')
    figure_recovery(runs)
    figure_covariates(runs)
    figure_process(runs)


if __name__ == '__main__':
    main()
