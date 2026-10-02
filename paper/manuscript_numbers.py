"""Numbers for the manuscript, computed from the benchmark evaluation (benchmarks/evaluation/) and the run folders.

Writes paper/build/numbers.json (values used as {{key}} in the manuscript), paper/build/table2.json (rows of the
run-outcome table), paper/build/recall.json and paper/build/run_profiles.json (Supplementary Material S3).
Run benchmarks/evaluate.py, benchmarks/agent_tests.py (and effect_evidence.py, scm_baseline.py) first.
"""
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / 'benchmarks'
EVAL = BENCH / 'evaluation'
BUILD = HERE / 'build'
sys.path.insert(0, str(BENCH))
from recall import recalled                                         # noqa: E402
from evaluate import reference_typical, typical_values, _match     # noqa: E402
from agent_tests import rest as _rest                                # noqa: E402


def same_rest(a, b):
    return _rest(a) == _rest(b)

DATASETS = json.loads((BENCH / 'datasets.json').read_text(encoding='utf-8'))
REFERENCE = {
    'pheno': {('CL', 'WT'): 'power', ('V', 'WT'): 'power', ('V', 'APGR'): 'categorical'},
    'remifentanil': {(p, c): 'linear' for p, c in (('V1', 'AGE'), ('V1', 'LBM'), ('V2', 'AGE'), ('V2', 'LBM'),
                                                   ('CL', 'AGE'), ('CL', 'LBM'), ('Q2', 'AGE'), ('Q3', 'AGE'))},
    'oral_mm': {},
}
LABEL = dict(pheno='Phenobarbital', remifentanil='Remifentanil', oral_mm='Oral MM (simulated)')
COND = dict(none='No knowledge', knowledge='Expert statement', misleading='Misleading statement')
LLM = dict(gpt='GPT-6.1 Sol', claude='Claude Opus 5.5')
WORDS = {0: 'no', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight',
         9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve'}
TIMES = {1: 'once', 2: 'twice', 3: 'three times'}


def fmt(x, digits=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'NA'
    return f'{x:,.{digits}f}'.replace('-', '−')                # typographic minus sign


def med_range(v, digits=1):
    v = pd.Series(v).dropna()
    if v.empty:
        return 'NA'
    if len(v) == 1:
        return fmt(v.iloc[0], digits)
    if len(v) == 2:                                             # two values: list them rather than a median
        a, b = sorted(v)
        return f'{fmt(a, digits)} and {fmt(b, digits)}'
    return f'{fmt(v.median(), digits)} ({fmt(v.min(), digits)} to {fmt(v.max(), digits)})'


def span(v, digits=1):
    """'a to b' (or 'a' when all values are equal)."""
    v = [x for x in v if x is not None]
    if not v:
        return 'NA'
    a, b = fmt(min(v), digits), fmt(max(v), digits)
    return a if a == b else f'{a} to {b}'


def word(k):
    return WORDS.get(k, str(k))


def none_of(k):
    return 'none' if k == 0 else word(k)


def of_runs(k, total):
    """'both runs', 'all four runs', 'three of four runs', 'neither run', 'none of the four runs', 'one run'."""
    if total == 1:
        return 'the run' if k == 1 else 'not the run'
    if total == 2:
        return {2: 'both runs', 1: 'one of the two runs', 0: 'neither run'}[k]
    if k == total:
        return f'all {word(total)} runs'
    if k == 0:
        return f'none of the {word(total)} runs'
    return f'{word(k)} of {word(total)} runs'


def run_dir(r):
    return BENCH / 'runs' / r['dataset'] / r['condition'] / r['llm'] / r['rep']


def relationships(details, row):
    key = next((k for k in details if k.replace('\\', '/') == f"{row['dataset']}/{row['condition']}/{row['llm']}/{row['rep']}"),
               None)
    return {(p, c): set(f) for p, c, f in (details.get(key) or {}).get('relationships', [])}


def subjects_of(path):
    data = sorted((path / 'data').glob('data_v*.csv'))
    return pd.read_csv(data[-1], na_values=['.']).groupby('ID').first().reset_index()


def subgroup_ratio(r, param, mask_fn):
    """Median ratio (final / reference typical value) of a parameter in a subgroup of subjects."""
    path = run_dir(r)
    res = json.loads((path / 'results.json').read_text(encoding='utf-8'))
    fm = res['final_model']
    subjects = subjects_of(path)
    tv = typical_values(fm['specification'], fm['summary'], subjects)
    ref = reference_typical(r['dataset'], subjects)
    q = _match(param, tv)
    if q is None:
        return None
    mask = mask_fn(subjects).to_numpy()
    return float(np.median(tv[q][mask] / ref[param][mask]))


def main():
    BUILD.mkdir(exist_ok=True)
    ev = json.loads((EVAL / 'evaluation.json').read_text(encoding='utf-8'))
    runs = pd.DataFrame(ev['runs'])
    details = ev['details']
    evidence = json.loads((EVAL / 'effect_evidence.json').read_text(encoding='utf-8')) \
        if (EVAL / 'effect_evidence.json').exists() else {}
    ratios = json.loads((EVAL / 'reference_fit_ratios.json').read_text(encoding='utf-8'))
    tests = json.loads((EVAL / 'agent_tests.json').read_text(encoding='utf-8'))
    tests_all = {(t['dataset'], t['condition'], t['llm'], t['rep']): t for t in tests}
    tests = {k: t for k, t in tests_all.items() if k[1] in ('none', 'knowledge')}

    rel = [relationships(details, r) for _, r in runs.iterrows()]
    runs['n_ref'] = [len(REFERENCE[d]) for d in runs['dataset']]
    runs['n_found'] = [sum(k in f for k in REFERENCE[d]) for d, f in zip(runs['dataset'], rel)]
    runs['n_form'] = [sum(REFERENCE[d][k] in f.get(k, ()) for k in REFERENCE[d]) for d, f in zip(runs['dataset'], rel)]
    runs['n_extra'] = [sum(k not in REFERENCE[d] for k in f) for d, f in zip(runs['dataset'], rel)]
    n = {}

    # ------------------------------------------------------------------ Table 2: one row per dataset and condition
    rows = []
    for ds in LABEL:
        for cond in COND:
            g = runs[(runs['dataset'] == ds) & (runs['condition'] == cond)]
            if g.empty:
                continue
            within = g['all_within_20pct'].fillna(False).astype(bool)
            row = dict(dataset=LABEL[ds], condition=COND[cond], runs=f"{len(g)} ({(g['llm'] == 'gpt').sum()}/"
                                                                       f"{(g['llm'] == 'claude').sum()})",
                       structure=f"{int(g['structure_match'].sum())}/{len(g)}")
            if REFERENCE[ds]:
                row['relationships'] = f"{int(g['n_found'].sum())}/{int(g['n_ref'].sum())}"
                row['forms'] = f"{int(g['n_form'].sum())}/{int(g['n_ref'].sum())}"
            else:
                row['relationships'] = row['forms'] = 'NA'
            row['extra'] = med_range(g['n_extra'], 0)
            row['typical'] = f'{int(within.sum())}/{len(g)}'
            row['reproduced'] = f"{int(g['reproduced'].fillna(False).astype(bool).sum())}/{len(g)}"
            row['delta_ofv'] = med_range(g['delta_ofv_vs_reference'], 1)
            row['delta_aic'] = med_range(g['delta_aic_vs_reference'], 1)
            row['fits'] = med_range(g['fits'], 0)
            row['hours'] = med_range(g['hours'], 1)
            row['cost'] = med_range(g['cost_usd'], 2)
            if cond == 'misleading':
                row['misleading'] = '; '.join(f'{k}: {v}' for k, v in g['misleading_followed'].value_counts().items())
            rows.append(row)
    (BUILD / 'table2.json').write_text(json.dumps(rows, indent=1), encoding='utf-8')

    # ------------------------------------------------------------------ design and process
    main_grid = runs[runs['condition'].isin(['none', 'knowledge'])]
    cells = main_grid.groupby(['dataset', 'condition', 'llm']).size()
    lo, hi = (int(cells.min()), int(cells.max())) if len(cells) else (0, 0)
    n['n_runs'] = len(main_grid)
    n['n_runs_all'] = len(runs)
    n['reps_times'] = TIMES.get(lo, f'{lo} times') if lo == hi else f'{word(lo)} to {word(hi)} times' if hi - lo > 1 \
        else f'{TIMES.get(lo)} or {TIMES.get(hi)}'
    n['reps_phrase'] = (f'{word(lo)} replicate{"s" if lo > 1 else ""} per combination' if lo == hi else
                        f'{word(lo)} or {word(hi)} replicates per combination' if hi - lo == 1 else
                        f'{word(lo)} to {word(hi)} replicates per combination')
    n['reps_abstract'] = (f'in {word(lo)} replicate{"s" if lo > 1 else ""}' if lo == hi else
                          f'in {word(lo)} or {word(hi)} replicates' if hi - lo == 1 else
                          f'in {word(lo)} to {word(hi)} replicates')
    n['grid_complete'] = 'yes' if lo == hi == 3 else 'no'
    n['n_finalized'] = int(main_grid['finalized'].sum())
    n['cost_total'] = fmt(main_grid['cost_usd'].sum(), 2)
    n['cost_median'] = fmt(main_grid['cost_usd'].median(), 2)
    n['hours_median'] = fmt(main_grid['hours'].median(), 1)
    n['fits_median'] = fmt(main_grid['fits'].median(), 0)
    n['fits_range'] = f"{int(main_grid['fits'].min())} to {int(main_grid['fits'].max())}"
    n['llm_calls_median'] = fmt(main_grid['llm_calls'].median(), 0)
    n['llm_calls_total'] = f"{int(main_grid['llm_calls'].sum()):,}"
    n['structure_all'] = f"{int(main_grid['structure_match'].sum())} of {len(main_grid)}"
    n['reproduced_all'] = f"{int(main_grid['reproduced'].fillna(False).astype(bool).sum())} of {len(main_grid)}"
    for ds in LABEL:
        g = main_grid[main_grid['dataset'] == ds]
        n[f'{ds}_hours_span'] = span(g['hours'], 1)
        n[f'{ds}_cost_span'] = span(g['cost_usd'], 2)
        n[f'{ds}_fits_span'] = span(g['fits'], 0)
    for ds in LABEL:
        for cond in ('none', 'knowledge', 'misleading'):
            g = runs[(runs['dataset'] == ds) & (runs['condition'] == cond)]
            if g.empty:
                continue
            k = f'{ds}_{cond}'
            n[f'{k}_runs'] = len(g)
            n[f'{k}_runs_word'] = word(len(g))
            n[f'{k}_all'] = of_runs(len(g), len(g))
            n[f'{k}_structure'] = of_runs(int(g['structure_match'].sum()), len(g))
            n[f'{k}_hours'] = med_range(g['hours'], 1)
            n[f'{k}_fits'] = med_range(g['fits'], 0)
            n[f'{k}_cost'] = med_range(g['cost_usd'], 2)
            n[f'{k}_dofv'] = med_range(g['delta_ofv_vs_reference'], 1)
            n[f'{k}_dofv_span'] = span(g['delta_ofv_vs_reference'], 0)
            n[f'{k}_reproduced'] = of_runs(int(g['reproduced'].fillna(False).astype(bool).sum()), len(g))
            sub = [f for f, (_, r) in zip(rel, runs.iterrows()) if r['dataset'] == ds and r['condition'] == cond]
            for (p, c), form in REFERENCE[ds].items():
                n[f'{k}_{p}_{c}'] = of_runs(sum((p, c) in f for f in sub), len(sub))
                n[f'{k}_{p}_{c}_form'] = of_runs(sum(form in f.get((p, c), ()) for f in sub), len(sub))
            n[f'{k}_all_ref'] = of_runs(sum(all(r in f for r in REFERENCE[ds]) for f in sub), len(sub))
            n[f'{k}_all_forms'] = of_runs(sum(all(REFERENCE[ds][r] in f.get(r, ()) for r in REFERENCE[ds])
                                              for f in sub), len(sub))
            n[f'{k}_mm'] = of_runs(int((g['elimination'] == 'Michaelis-Menten').sum()), len(g))
            for col in ('KM', 'Ka', 'V', 'VMAX', 'V1', 'V3', 'Q3'):  # noqa: E501
                if f'ratio_{col}_median' in g and g[f'ratio_{col}_median'].notna().any():
                    n[f'{k}_{col}_ratio_span'] = span(g[f'ratio_{col}_median'], 2)
            if cond == 'misleading':
                for claim in ('no weight effect', 'Apgar on CL', 'linear elimination', 'two compartments'):
                    n[f'{k}_{claim.replace(" ", "_")}'] = of_runs(
                        int(g['misleading_followed'].fillna('').str.contains(claim).sum()), len(g))
            if ds == 'remifentanil':
                n[f'{k}_V3_AGE'] = of_runs(sum(('V3', 'AGE') in f for f in sub), len(sub))
                for llm in LLM:
                    sl = [f for f, (_, r) in zip(rel, runs.iterrows())
                          if r['dataset'] == ds and r['condition'] == cond and r['llm'] == llm]
                    n[f'{k}_V3_AGE_{llm}'] = of_runs(sum(('V3', 'AGE') in f for f in sl), len(sl))
                n[f'{k}_V3_ratio'] = med_range(g['ratio_V3_median'], 2)
                n[f'{k}_subject_span'] = span([v for c in ('V1', 'V2', 'CL', 'Q2', 'Q3')
                                               for rr in g[f'ratio_{c}_range'].dropna()
                                               for v in map(float, str(rr).split('-', 1))], 2)

    # ------------------------------------------------------------------ the agents' own tests (agent_tests.py)
    def agent_test_values(ds, cond, relname, forms=None, llm=None, context=False, last=False):
        """Per run: the largest evidence (ΔOFV) the agent obtained for a relationship (optionally only in the final
        model's covariate context, or only for some functional forms)."""
        out = []
        for key, t in tests.items():
            if key[0] != ds or key[1] != cond or (llm and key[2] != llm):
                continue
            sel = [x for x in t['covariate_tests'] if x['relationship'] == relname
                   and (forms is None or x['form'] in forms) and (not context or x.get('final_context'))
                   and (context != 'final' or x.get('final_stochastic'))]
            if sel and last:
                out.append(max(sel, key=lambda x: max(x['with_model'], x['without_model']))['delta_ofv'])
            elif sel:
                out.append(max(x['delta_ofv'] for x in sel))
        return out
    for ds, cond, relname, forms, tag in [
            ('pheno', 'none', 'V~APGR', ('categorical',), 'pheno_none_apgar_indicator'),
            ('pheno', 'none', 'V~APGR', None, 'pheno_none_apgar_any'),
            ('pheno', 'knowledge', 'V~APGR', ('categorical',), 'pheno_knowledge_apgar'),
            ('remifentanil', 'none', 'V1~AGE', 'last', 'remi_none_v1age'),
            ('remifentanil', 'knowledge', 'V1~AGE', 'last', 'remi_knowledge_v1age'),
            ('remifentanil', 'knowledge', 'V3~AGE', None, 'remi_knowledge_v3age'),
            ('remifentanil', 'none', 'V3~AGE', None, 'remi_none_v3age')]:
        for llm in (None, 'gpt', 'claude'):
            v = agent_test_values(ds, cond, relname, None if forms == 'last' else forms, llm, last=forms == 'last')
            suffix = f'_{llm}' if llm else ''
            n[f't_{tag}{suffix}'] = span(v, 1)
            n[f't_{tag}{suffix}_runs'] = len(v)
            n[f't_{tag}{suffix}_below_inclusion'] = sum(x < 3.84 for x in v)
            n[f't_{tag}{suffix}_below_retention'] = sum(x < 6.63 for x in v)
    k_ind = n['t_pheno_none_apgar_indicator_runs']
    n['t_pheno_none_apgar_indicator_runs_word'] = word(k_ind)
    k_other = len(main_grid[(main_grid['dataset'] == 'pheno') & (main_grid['condition'] == 'none')]) - k_ind
    n['pheno_apgar_cont_only'] = 'other run' if k_other == 1 else f'other {word(k_other)} runs'
    # the Apgar indicator in the covariate context of the final model (Claude Opus 5.5: with a CL-V correlation)
    for llm in (None, 'gpt', 'claude'):
        v = agent_test_values('pheno', 'none', 'V~APGR', ('categorical',), llm, context='final')
        n['t_pheno_none_apgar_final' + (f'_{llm}' if llm else '')] = span(v, 1)
    # linear vs Michaelis-Menten (like-for-like pairs) and two vs three compartments (best base models)
    mm = [abs(x['ofv'][0] - x['ofv'][1]) for key, t in tests.items() if key[0] == 'oral_mm'
          for x in t['structural_tests']
          if {x['structures'][0][0], x['structures'][1][0]} == {'pk', 'michaelis_menten'}
          and x['structures'][0][1] == x['structures'][1][1] == 1 and not any(s[2] for s in x['structures'])]
    n['mm_vs_linear'] = span(mm, 0)
    three = [c['delta_ofv'] for key, t in tests.items() if key[0] == 'remifentanil' for c in t['compartment_tests']
             if c['compartments'] == [2, 3] and c['covariates'] == 0 and c['extra_parameters'] == 4]
    n['remi_3v2'] = span(three, 0)
    two = {key: [c['delta_ofv'] for c in t['compartment_tests'] if c['compartments'] == [1, 2] and c['covariates'] >= 2
                 and c['extra_parameters'] == 2] for key, t in tests.items() if key[0] == 'pheno'}
    two_v = [v for vs in two.values() for v in vs]
    n['pheno_2cmt'] = span(two_v, 1)
    n['pheno_2cmt_aic'] = span([v - 4 for v in two_v], 1)
    for llm in LLM:
        n[f'pheno_2cmt_{llm}'] = of_runs(sum(bool(v) for k, v in two.items() if k[2] == llm),
                                         sum(1 for k in two if k[2] == llm))
    dose = [x['delta_ofv'] for key, t in tests.items() if key[0] == 'oral_mm' for x in t['covariate_tests']
            if x['relationship'].endswith('~DOSE') and not x['relationship'].startswith('CL')]
    n['oral_dose'] = span(dose, 1)
    n['oral_dose_runs'] = of_runs(sum(any(x['relationship'].endswith('~DOSE') and not x['relationship'].startswith('CL')
                                          for x in t['covariate_tests']) for key, t in tests.items() if key[0] == 'oral_mm'),
                                  sum(1 for key in tests if key[0] == 'oral_mm'))
    v1 = {llm: agent_test_values('remifentanil', 'none', 'V1~AGE', llm=llm, last=True)
               + agent_test_values('remifentanil', 'knowledge', 'V1~AGE', llm=llm, last=True) for llm in LLM}
    for llm in LLM:
        n[f't_remi_v1age_all_{llm}'] = span(v1[llm], 1)
    # tool use, time budget, bootstrap, fits
    prof = []
    for key, t in tests.items():
        boots = t['bootstraps']
        prof.append(dict(dataset=key[0], condition=key[1], llm=key[2], rep=key[3],
                         plot_data=t['plotted_data'], run_nca=t['ran_nca'], screen_covariates=t['screened_covariates'],
                         fitted_after_plots=t['fitted_after_viewing_plots'], hours_left=t['hours_left'],
                         bootstrap=', '.join(f"{b['converged']}/{b['requested']}" for b in boots) or '–',
                         fits=t['fits_total'], converged=t['fits_converged'],
                         final_se=t['final_uncertainty']))
    prof = pd.DataFrame(prof)
    (BUILD / 'run_profiles.json').write_text(prof.to_json(orient='records', indent=1), encoding='utf-8')
    assert len(prof) == len(main_grid), (len(prof), len(main_grid))
    for llm in LLM:
        p = prof[prof['llm'] == llm]
        n[f'{llm}_bootstrap_runs'] = of_runs(int((p['bootstrap'] != '–').sum()), len(p))
        n[f'{llm}_runs'] = len(p)
        n[f'{llm}_fitted_after_plots'] = of_runs(int(p['fitted_after_plots'].sum()), len(p))
        n[f'{llm}_plot_data'] = of_runs(int(p['plot_data'].sum()), len(p))
        n[f'{llm}_screen'] = of_runs(int(p['screen_covariates'].sum()), len(p))
    n['fits_total'] = f"{int(prof['fits'].sum()):,}"
    n['fits_converged'] = f"{int(prof['converged'].sum()):,}"
    n['final_se_missing'] = word(int((prof['final_se'] != 'computed').sum()))
    more = True
    for (ds, cond), g in main_grid.groupby(['dataset', 'condition']):
        a, b = g[g['llm'] == 'gpt']['fits'], g[g['llm'] == 'claude']['fits']
        more &= bool(len(a) and len(b) and a.median() > b.median())
    if more:
        n['gpt_more_fits_every_cell'] = 'yes'
    late = prof[prof['hours_left'].astype(float) < 1]
    n['runs_near_time_limit'] = word(len(late))
    n['runs_near_time_limit_desc'] = ', '.join(f"{LLM[r['llm']]} {LABEL[r['dataset']].lower()} "
                                                f"({'no knowledge' if r['condition'] == 'none' else r['condition']})"
                                                for _, r in late.iterrows()) or 'none'
    boot_runs = [b for b in prof['bootstrap'] if b != '–']
    n['bootstrap_runs'] = word(len(boot_runs))
    remi_boot = [b for b, ds in zip(prof['bootstrap'], prof['dataset']) if ds == 'remifentanil' and b != '–']
    n['bootstrap_remi'] = ' and '.join(remi_boot) or 'none'
    dirs = [run_dir(r) for _, r in main_grid.iterrows()]
    secs = [json.loads(p.read_text(encoding='utf-8')).get('seconds') or 0
            for d_ in dirs for p in d_.glob('models/*/summary.json')]
    n['max_fit_minutes'] = fmt(max(secs) / 60, 0) if secs else 'NA'
    finish = [json.loads(line).get('finish_reason') for d_ in dirs
              for line in (d_ / 'transcript.jsonl').read_text(encoding='utf-8').splitlines()]
    n['llm_responses_total'] = f'{len(finish):,}'
    n['llm_responses_truncated'] = none_of(sum(f == 'length' for f in finish))
    # diagnostics of the final models
    for ds in LABEL:
        g = main_grid[main_grid['dataset'] == ds]
        n[f'{ds}_vpc_span'] = span(g['vpc_inside_fraction'] * 100, 0)
        pc = []
        for _, r in g.iterrows():
            final = json.loads((run_dir(r) / 'results.json').read_text(encoding='utf-8'))['final_model']['model_id']
            last = None
            for line in (run_dir(r) / 'tool_log.jsonl').read_text(encoding='utf-8').splitlines():
                t = json.loads(line)
                if t['tool'] == 'run_vpc' and t['args'].get('model_id') == final \
                        and t['args'].get('prediction_corrected') and isinstance(t.get('result'), dict):
                    for out in t['result'].values():
                        if isinstance(out, dict) and out.get('observed_percentiles_inside'):
                            a_, b_ = map(int, out['observed_percentiles_inside'].split('/'))
                            last = 100 * a_ / b_
            if last is not None:
                pc.append(last)
        n[f'{ds}_pcvpc_span'] = span(pc, 0)
        misses = []
        for _, r in g.iterrows():
            res = json.loads((run_dir(r) / 'results.json').read_text(encoding='utf-8'))
            vpc = ((res['final_model'].get('vpc') or {}).get('result') or {})
            for out in vpc.values():
                for b in (out or {}).get('bins', []):
                    misses += [(b['time'], q) for q in ('p5', 'p50', 'p95') if not b[q]['inside']]
        n[f'{ds}_vpc_miss_times'] = span([t for t, _ in misses], 0)
        n[f'{ds}_vpc_miss_upper'] = f"{sum(q == 'p95' for _, q in misses)} of {len(misses)}"

    # phenobarbital weight exponents without knowledge, and identical reference relationships within cells
    for llm in LLM:
        fixed, est = [], []
        for _, r in runs[(runs['dataset'] == 'pheno') & (runs['condition'] == 'none') & (runs['llm'] == llm)].iterrows():
            spec = json.loads((run_dir(r) / 'results.json').read_text(encoding='utf-8'))['final_model']['specification']
            for c in spec['covariates']:
                if c['covariate'] == 'WT':
                    (fixed if c['coefficient']['fixed'] else est).append(c['coefficient']['value'])
        # per run: estimated (two exponents) or fixed at 1
        n_fixed, n_est = len(fixed) // 2, len(est) // 2
        n[f'pheno_none_wt_{llm}'] = ('fixed at 1' if fixed and not est and set(fixed) == {1.0} else
                                     f'estimated at {span(est, 2)}' if est and not fixed else
                                     f'estimated at {span(est, 2)} in {word(n_est)} run{"s" if n_est > 1 else ""} '
                                     f'and fixed at 1 in {word(n_fixed)}')
    same = True
    for (ds, cond), g in runs[runs['condition'].isin(['none', 'knowledge'])].groupby(['dataset', 'condition']):
        sets = {frozenset(k for k in f if k in REFERENCE[ds])
                for f, (_, r) in zip(rel, runs.iterrows()) if r['dataset'] == ds and r['condition'] == cond}
        same &= len(sets) == 1
    if same:
        n['same_reference_sets'] = 'yes'

    # ------------------------------------------------------------------ subgroups and clinical meaning
    def sub_span(ds, cond, param, mask_fn):
        v = [subgroup_ratio(r, param, mask_fn) for _, r in runs.iterrows() if r['dataset'] == ds and r['condition'] == cond]
        return span([x for x in v if x is not None], 2)
    n['pheno_none_V_lowapgar'] = sub_span('pheno', 'none', 'V', lambda s: s['APGR'] < 5)
    n['pheno_n_lowapgar'] = int((subjects_of(BENCH / 'reference_fits' / 'pheno')['APGR'] < 5).sum())
    lowv = [subgroup_ratio(r, 'V', lambda s: s['APGR'] < 5) for _, r in runs.iterrows()
            if r['dataset'] == 'pheno' and r['condition'] == 'none']
    n['remi_none_V1_elderly'] = sub_span('remifentanil', 'none', 'V1', lambda s: s['AGE'] >= 65)
    n['remi_knowledge_V3_elderly'] = sub_span('remifentanil', 'knowledge', 'V3', lambda s: s['AGE'] >= 65)
    n['remi_n_elderly'] = int((subjects_of(BENCH / 'reference_fits' / 'remifentanil')['AGE'] >= 65).sum())
    est = DATASETS['pheno']['reference']['estimates']
    v_kg = est['V_per_kg']
    n['apgar_conc_high'] = fmt(20 / v_kg, 1)
    c_low = 20 / (v_kg * (1 + est['APGR_LT5_fractional_increase_in_V']))
    n['apgar_conc_low'] = fmt(c_low, 1)
    n['apgar_conc_agents'] = span([c_low / x for x in lowv if x], 1)
    n['apgar_effect_pct'] = fmt(est['APGR_LT5_fractional_increase_in_V'] * 100, 0)
    # oral MM: the Michaelis constant against the nominal and the realized (subset) value
    km_real = 232.
    n['oral_mm_KM_ratio_span'] = span(main_grid.loc[main_grid['dataset'] == 'oral_mm', 'ratio_KM_median'], 2)
    rr = ratios.get('remifentanil', {})
    if rr:
        n['reffit_remi_core_pct'] = fmt(max(abs(1 - rr[q]['median']) for q in ('CL', 'V1', 'V2')) * 100, 1)
    g = runs[runs['dataset'] == 'oral_mm']
    n['km_realized_span'] = span(g['ratio_KM_median'] * 250 / km_real, 2)
    n['reffit_km_realized'] = fmt(ratios['oral_mm']['KM']['median'] * 250 / km_real, 2)

    # ------------------------------------------------------------------ reference fits and drop-one evidence
    for ds, d in evidence.items():
        rem = [r for r in d.get('removals', []) if r.get('delta_ofv') is not None]
        for r in rem:
            n[f"evidence_{ds}_{r['effect'].split('_')[0]}"] = fmt(r['delta_ofv'], 1)
        if rem:
            n[f'evidence_{ds}_min'] = fmt(min(r['delta_ofv'] for r in rem), 1)
            n[f'evidence_{ds}_max'] = fmt(max(r['delta_ofv'] for r in rem), 1)
            others = [r['delta_ofv'] for r in rem if r['effect'].split('_')[0] not in ('V1~AGE', 'V~APGR')]
            n[f'evidence_{ds}_others_min'] = fmt(min(others), 1) if others else 'NA'
    for ds, d in ratios.items():
        for p, r in d.items():
            n[f'reffit_{ds}_{p}'] = fmt(r['median'], 2)
            n[f'reffit_{ds}_{p}_pct'] = f"{abs(1 - r['median']) * 100:.0f}"

    # remifentanil with the expert statement: effects on V3 implied by the statement, LBM on Q2/Q3
    rk = [(r, f) for f, (_, r) in zip(rel, runs.iterrows())
          if r['dataset'] == 'remifentanil' and r['condition'] == 'knowledge']
    if rk:
        def kept_v3(f):
            return [e for e in (('V3', 'AGE'), ('V3', 'LBM')) if e in f]
        n['rk_v3_any'] = of_runs(sum(bool(kept_v3(f)) for _, f in rk), len(rk))
        n['rk_v3age'] = of_runs(sum(('V3', 'AGE') in f for _, f in rk), len(rk))
        n['rk_v3lbm'] = of_runs(sum(('V3', 'LBM') in f for _, f in rk), len(rk))
        support, weak = [], []
        for r, f in rk:
            t = tests.get((r['dataset'], r['condition'], r['llm'], r['rep']), {})
            for e in kept_v3(f):
                v = [x['delta_ofv'] for x in t.get('covariate_tests', []) if x['relationship'] == '~'.join(e)]
                if v:
                    support.append(max(v))
                    if max(v) < 6.63:
                        weak.append((r['llm'], e, max(v)))
        n['rk_v3_support'] = span(support, 1)
        if weak:
            n['rk_v3_weak'] = 'yes'
            n['rk_v3_weak_sentence'] = '; '.join(
                f"one {LLM[llm]} run kept an effect of {e[1].replace('AGE', 'age')} on V3 that failed its own retention "
                f"criterion (OFV change {fmt(v, 1)})" for llm, e, v in weak)
        age_runs = [r for r, f in rk if ('V3', 'AGE') in f]
        n['rk_v3age_ratio'] = span([r['ratio_V3_median'] for r in age_runs], 2)
        v3e = [subgroup_ratio(r, 'V3', lambda s: s['AGE'] >= 65) for r in age_runs]
        n['rk_v3age_elderly'] = span([x for x in v3e if x is not None], 2)
        tested_q = kept_q = 0
        for r, f in rk:
            kept_q += any(e in f for e in (('Q2', 'LBM'), ('Q3', 'LBM')))
            any_spec = False
            for sp in run_dir(r).glob('models/*/spec.json'):
                cs = json.loads(sp.read_text(encoding='utf-8')).get('covariates', [])
                any_spec |= any(c['covariate'] == 'LBM' and c['parameter'] in ('Q2', 'Q3') for c in cs)
            tested_q += any_spec
        n['rk_lbm_q_tested'] = of_runs(tested_q, len(rk))
        n['rk_lbm_q_kept'] = of_runs(kept_q, len(rk))

    # direct comparisons of linear and exponential covariate forms (same relationships and parameter count)
    cmp_ = []
    for _, r in main_grid[main_grid['dataset'] == 'remifentanil'].iterrows():
        res = json.loads((run_dir(r) / 'results.json').read_text(encoding='utf-8'))
        conv = {m['model_id']: m for m in res['models'] if m.get('status') == 'converged'}
        specs = {mid: json.loads((run_dir(r) / 'models' / mid / 'spec.json').read_text(encoding='utf-8')) for mid in conv}
        def sig(sp):
            return {(c['parameter'], c['covariate']) for c in sp.get('covariates', [])}, {c['form'] for c in sp.get('covariates', [])}
        for a in specs:
            for b in specs:
                (pa, fa), (pb, fb) = sig(specs[a]), sig(specs[b])
                if pa and pa == pb and fa == {'linear'} and fb == {'exponential'}                         and conv[a]['n_estimated'] == conv[b]['n_estimated'] and same_rest(specs[a], specs[b]):
                    cmp_.append(conv[a]['ofv'] - conv[b]['ofv'])
    n['form_cmp_n'] = len(cmp_)
    n['form_cmp_word'] = word(len(cmp_))
    n['form_cmp_span'] = span(cmp_, 1)
    if cmp_ and all(x > 0 for x in cmp_):
        n['form_cmp_exp_better'] = 'yes'

    # functional forms of age and LBM in the remifentanil final models without knowledge
    forms = {'AGE': set(), 'LBM': set()}
    for _, r in main_grid[(main_grid['dataset'] == 'remifentanil') & (main_grid['condition'] == 'none')].iterrows():
        spec = json.loads((run_dir(r) / 'results.json').read_text(encoding='utf-8'))['final_model']['specification']
        for c in spec['covariates']:
            if c['covariate'] in forms:
                forms[c['covariate']].add(c['form'])
    for cov, f in forms.items():
        n[f'remi_none_{cov.lower()}_forms'] = ' or '.join(sorted(f)) or 'NA'

    # GPT phenobarbital runs without knowledge that chose or mentioned the published Apgar cutoff of 5
    import re as _re
    gp = main_grid[(main_grid['dataset'] == 'pheno') & (main_grid['condition'] == 'none') & (main_grid['llm'] == 'gpt')]
    k = 0
    for _, r in gp.iterrows():
        res = json.loads((run_dir(r) / 'results.json').read_text(encoding='utf-8'))
        transcript = (run_dir(r) / 'transcript.jsonl').read_text(encoding='utf-8')
        chose = any(_re.search(r'APGR\s*<\s*5', t.get('expression', '')) for t in res.get('data_transformations', []))
        k += bool(chose or _re.search(r'(less than|below) 5', transcript))
    n['gpt_apgar_cutoff'] = of_runs(k, len(gp))

    # strongly supported reference relationships (removal from the reference fit costs >= 6.63) kept without knowledge
    strong = {(ds, r['effect'].split('_')[0]) for ds, d in evidence.items() for r in d.get('removals', [])
              if r.get('delta_ofv') is not None and r['delta_ofv'] >= 6.63}
    nk = [(r, f) for f, (_, r) in zip(rel, runs.iterrows())
          if r['condition'] == 'none' and r['dataset'] in ('pheno', 'remifentanil')]
    missed = []
    for r, f in nk:
        miss = [e for ds, e in strong if ds == r['dataset'] and tuple(e.split('~')) not in f]
        if miss:
            secs_r = [json.loads(p.read_text(encoding='utf-8')).get('seconds') or 0
                      for p in run_dir(r).glob('models/*/summary.json')]
            missed.append(dict(llm=r['llm'], dataset=r['dataset'], rep=r['rep'], missing=miss, fits=int(r['fits']),
                               fit_minutes=float(np.median(secs_r)) / 60 if secs_r else None,
                               hours_left=json.loads((run_dir(r) / 'results.json').read_text(encoding='utf-8'))
                               ['budget'].get('hours_left')))
    if strong and nk:
        n['strong_kept'] = of_runs(len(nk) - len(missed), len(nk))
        if not missed:
            n['all_strong_kept'] = 'yes'
        else:
            n['strong_missed_desc'] = '; '.join(
                f"{word(1)} {LLM[m['llm']]} {LABEL[m['dataset']].lower()} run omitted "
                f"{', '.join(e.replace('~', '–') for e in m['missing'])} after {word(m['fits'])} fits"
                f" (median {fmt(m['fit_minutes'], 0)} minutes per fit, {fmt(m['hours_left'], 1)} hours left)"
                for m in missed)

    # ------------------------------------------------------------------ deterministic stepwise baseline
    scm_path = EVAL / 'scm_baseline.json'
    scm = json.loads(scm_path.read_text(encoding='utf-8')) if scm_path.exists() else {}
    for ds, d in scm.items():
        def short(label):                     # 'V~APGR_LT5=1(categorical)' -> 'V–APGR'
            p, c = label.split('(')[0].split('~')
            return f"{p}–{c.split('_')[0].split('=')[0]}"
        n[f'scm_{ds}_included'] = ', '.join(short(x) for x in d['included'])
        n[f'scm_{ds}_n_included'] = len(d['included'])
        n[f'scm_{ds}_fits'] = d['fits']
        for h in d['history']:
            if h['step'] == 'forward add':
                n[f"scm_{ds}_forward_{short(h['added']).replace('–', '_')}"] = fmt(h['delta_ofv'], 1)
            if h['step'] == 'backward remove':
                n[f"scm_{ds}_backward_{short(h['removed']).replace('–', '_')}"] = fmt(h['delta_ofv'], 1)
        n[f'scm_{ds}_removed'] = ', '.join(short(h['removed']) for h in d['history'] if h['step'] == 'backward remove') \
            or 'none'

    # misleading statements: what the final models adopted, and the evidence against the wrong claims
    mis = runs[runs['condition'] == 'misleading']
    if len(mis):
        def models_of(r):
            res = json.loads((run_dir(r) / 'results.json').read_text(encoding='utf-8'))
            out = []
            for m in res['models']:
                if m.get('status') != 'converged' or m.get('ofv') is None:
                    continue
                d_ = run_dir(r) / 'models' / m['model_id']
                out.append((m, json.loads((d_ / 'spec.json').read_text(encoding='utf-8')),
                            json.loads((d_ / 'summary.json').read_text(encoding='utf-8'))))
            return out
        pm = mis[mis['dataset'] == 'pheno']
        om = mis[mis['dataset'] == 'oral_mm']
        if len(pm):
            gain, kept, opposite, cl_tests = [], 0, 0, []
            for _, r in pm.iterrows():
                ms = models_of(r)
                with_wt = [m['ofv'] for m, s_, _ in ms if any(c['covariate'] == 'WT' for c in s_['covariates'])]
                no_wt = [m['ofv'] for m, s_, _ in ms if not any(c['covariate'] == 'WT' for c in s_['covariates'])]
                if with_wt and no_wt:
                    gain.append(min(no_wt) - min(with_wt))
                kept += 'no weight effect' not in str(r['misleading_followed'])
                coefs = [ce['coefficient'] for m, s_, su in ms for ce in su.get('covariate_effects', [])
                         if ce['effect'].startswith('CL~') and 'APG' in ce['effect']]
                opposite += bool(coefs) and max(coefs) > 0 and not any(c < -1e-6 for c in coefs)
                t = tests_all.get((r['dataset'], r['condition'], r['llm'], r['rep']), {})
                cl_tests += [x['delta_ofv'] for x in t.get('covariate_tests', []) if x['relationship'] == 'CL~APGR']
            n['pm_runs'] = of_runs(len(pm), len(pm))
            n['pm_weight_kept'] = of_runs(kept, len(pm))
            n['pm_weight_gain'] = span(gain, 0)
            n['pm_apgar_cl_opposite'] = of_runs(opposite, len(pm))
            n['pm_apgar_cl_adopted'] = of_runs(int(pm['misleading_followed'].fillna('').str.contains('Apgar on CL').sum()),
                                               len(pm))
            n['pm_apgar_cl_tests'] = span(cl_tests, 1)
        if len(om):
            gain, two = [], []
            for _, r in om.iterrows():
                ms = models_of(r)
                lin = [m['ofv'] for m, s_, _ in ms if s_['structure'].get('type', 'pk') == 'pk']
                mm_ = [m['ofv'] for m, s_, _ in ms if s_['structure'].get('type') == 'michaelis_menten']
                if lin and mm_:
                    gain.append(min(lin) - min(mm_))
                t = tests_all.get((r['dataset'], r['condition'], r['llm'], r['rep']), {})
                # best AIC of one- vs two-compartment Michaelis-Menten models (positive: two compartments better)
                a1 = [m['aic'] for m, s_, _ in ms if s_['structure'].get('type') == 'michaelis_menten'
                      and s_['structure'].get('compartments', 1) == 1 and m.get('aic') is not None]
                a2 = [m['aic'] for m, s_, _ in ms if s_['structure'].get('type') == 'michaelis_menten'
                      and s_['structure'].get('compartments', 1) == 2 and m.get('aic') is not None]
                if a1 and a2:
                    two.append(min(a1) - min(a2))
            n['om_runs'] = of_runs(len(om), len(om))
            n['om_mm_kept'] = of_runs(int((om['elimination'] == 'Michaelis-Menten').sum()), len(om))
            n['om_mm_gain'] = span(gain, 0)
            n['om_2cmt_adopted'] = of_runs(int((om['compartments'] == 2).sum()), len(om))
            n['om_2cmt_daic'] = span(two, 1)
        n['misleading_runs'] = len(mis)

    # remifentanil: backward elimination from the reference model (benchmarks/backward_baseline.py)
    bb_path = EVAL / 'backward_baseline.json'
    if bb_path.exists():
        bb = json.loads(bb_path.read_text(encoding='utf-8')).get('remifentanil')
        if bb and 'removed' in bb:
            n['backward_remifentanil_removed'] = ', '.join(e.replace('~', '–') for e in bb['removed']) or 'nothing'
            last = bb['history'][-1]['tests']
            kept = [v for k, v in last.items() if v is not None and k not in bb['removed']]
            n['backward_remifentanil_min'] = fmt(min(kept), 1) if kept else 'NA'
            if bb['removed'] == ['V1~AGE']:
                n['baseline_remi_agrees'] = 'yes'

    # ------------------------------------------------------------------ dates and recall
    started = sorted(json.loads((run_dir(r) / 'run.json').read_text(encoding='utf-8'))['started'][:10]
                     for _, r in runs.iterrows())
    if started:
        a, b = (dt.date.fromisoformat(s) for s in (started[0], started[-1]))
        n['run_dates'] = (f'on {a:%B} {a.day}, {a.year}' if a == b else
                          f'between {a:%B} {a.day} and {b:%B} {b.day}, {b.year}')
    recall = {}
    for _, r in main_grid[main_grid['dataset'].isin(['pheno', 'remifentanil'])].iterrows():
        recall[(r['dataset'], r['condition'], r['llm'], r['rep'])] = recalled(run_dir(r))
    for llm in LLM:
        ks = [k for k in recall if k[2] == llm]
        n[f'recall_{llm}'] = of_runs(sum(bool(recall[k]) for k in ks), len(ks))
        n[f'recall_{llm}_total'] = len(ks)
        for cond in ('none', 'knowledge'):
            kc = [k for k in ks if k[1] == cond]
            n[f'recall_{llm}_{cond}'] = of_runs(sum(bool(recall[k]) for k in kc), len(kc))
    (BUILD / 'recall.json').write_text(json.dumps({'/'.join(k): v for k, v in recall.items()}, indent=1,
                                                  ensure_ascii=False), encoding='utf-8')
    (BUILD / 'numbers.json').write_text(json.dumps(n, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(n, indent=1, ensure_ascii=False))
    print(pd.DataFrame(rows).to_string())


if __name__ == '__main__':
    main()
