"""Numbers for the manuscript, computed from the benchmark evaluation (benchmarks/evaluation/) and the run folders.

Writes paper/build/numbers.json (values used as {{key}} in the manuscript), paper/build/table2.json (rows of the
run-outcome table), paper/build/recall.json and paper/build/run_profiles.json (Supplementary Material S3).
Run benchmarks/evaluate.py, benchmarks/agent_tests.py (and effect_evidence.py, scm_baseline.py) first.
"""
import datetime as dt
import json
from decimal import Decimal, ROUND_HALF_UP
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
sys.path.insert(0, str(HERE))
from recall import recalled                                         # noqa: E402
from evaluate import reference_typical, typical_values, _match     # noqa: E402
from agent_tests import rest as _rest                                # noqa: E402
from tool_groups import N_TOOLS                                      # noqa: E402


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
    q = Decimal(float(x)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)   # 14.5 -> 15, not 14
    return f'{q:,.{digits}f}'.replace('-', '−')                # typographic minus sign


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
    w = str if total >= 10 else word                            # numerals for both counts when one is 10 or more
    if k == total:
        return f'all {w(total)} runs'
    if k == 0:
        return f'none of the {w(total)} runs'
    return f'{w(k)} of {w(total)} runs'


def run_dir(r):
    return BENCH / 'runs' / r['dataset'] / r['condition'] / r['llm'] / r['rep']


def effect_words(e):
    """'V1~AGE' -> 'the age effect on V1' (abbreviated parameter names are defined in the text)."""
    p, c = e.split('~')
    return f"the {dict(AGE='age', WT='weight', APGR='Apgar score').get(c, c)} effect on {p}"


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
    sys.stdout.reconfigure(encoding='utf-8')     # the printout has − and other non-ASCII characters (cp949 consoles)
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
    # runs on the datasets whose reference model has covariates (Figure 3)
    n['n_runs_covariates'] = int(main_grid['dataset'].isin([d for d in REFERENCE if REFERENCE[d]]).sum())
    n['n_tools'] = N_TOOLS                                          # paper/tool_groups.py (Methods, Figure 1)
    # per-run limits from the run records (Methods, Figure 1): the task message stated the fits, responses and hours;
    # the fee limit was enforced by the session, and the model saw only the fees spent
    limits = {json.dumps(json.loads((run_dir(r) / 'run.json').read_text(encoding='utf-8'))['budget'], sort_keys=True)
              for _, r in runs.iterrows()}
    if len(limits) != 1:
        raise SystemExit(f'the runs differ in their limits: {sorted(limits)}')
    limits = json.loads(limits.pop())
    n['budget_fits'], n['budget_responses'] = limits['max_fits'], limits['max_turns']
    n['budget_hours'], n['budget_fee_usd'] = f"{limits['max_hours']:g}", f"{limits['max_cost_usd']:g}"
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
    # reproduction on the datasets whose reference models have covariate effects (phenobarbital, remifentanil), by
    # condition (abstract)
    cov = main_grid[main_grid['dataset'].isin([ds for ds in LABEL if REFERENCE[ds]])]
    for cond in ('none', 'knowledge'):
        g = cov[cov['condition'] == cond]
        k = int(g['reproduced'].fillna(False).astype(bool).sum())
        n[f'covariate_{cond}_reproduced'] = k
        n[f'covariate_{cond}_runs'] = len(g)
        n[f'covariate_{cond}_reproduced_phrase'] = 'none of them' if k == 0 else f'{word(k)} of them'
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
            n[f't_{tag}{suffix}2'] = span(v, 2)
            n[f't_{tag}{suffix}_runs'] = len(v)
            n[f't_{tag}{suffix}_runs_word'] = word(len(v))
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
    mm = [abs(x['ofv'][0] - x['ofv'][1]) for key, t in tests.items() if key[0] == 'oral_mm' and key[1] == 'none'
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
    dose_runs = [key for key, t in tests.items() if key[0] == 'oral_mm'
                 and any(x['relationship'].endswith('~DOSE') and not x['relationship'].startswith('CL')
                         for x in t['covariate_tests'])]
    n['oral_dose_runs'] = of_runs(len(dose_runs), sum(1 for key in tests if key[0] == 'oral_mm'))
    n['oral_dose_params'] = sorted({x['relationship'].split('~')[0] for key in dose_runs
                                    for x in tests[key]['covariate_tests'] if x['relationship'].endswith('~DOSE')})
    if dose_runs and {k[2] for k in dose_runs} == {'gpt'}:
        n['oral_dose_all_gpt'] = 'yes'
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
    remi_boot = [b for key, t in tests.items() if key[0] == 'remifentanil' for b in t['bootstraps']]
    n['bootstrap_remi_n'] = word(len(remi_boot))
    n['bootstrap_remi_requested'] = span([b['requested'] for b in remi_boot], 0)
    n['bootstrap_remi_completed'] = span([b['completed'] for b in remi_boot], 0)
    n['bootstrap_remi_converged'] = span([b['converged'] for b in remi_boot], 0)
    all_boot = [b for t in tests.values() for b in t['bootstraps']]
    n['bootstrap_n'] = len(all_boot)
    n['bootstrap_complete'] = word(sum(b['completed'] >= b['requested'] for b in all_boot))
    oral_boot = [b for key, t in tests.items() if key[0] == 'oral_mm' for b in t['bootstraps']]
    n['bootstrap_oral_requested'] = span([b['requested'] for b in oral_boot], 0)
    n['bootstrap_oral_completed'] = span([b['completed'] for b in oral_boot], 0)
    # fit times per dataset (the 36 runs, as the run times, fees and fit counts reported with them)
    for ds in LABEL:
        secs_ds = [json.loads(p.read_text(encoding='utf-8')).get('seconds') or 0
                   for _, r in main_grid[main_grid['dataset'] == ds].iterrows()
                   for p in run_dir(r).glob('models/*/summary.json')]
        n[f'{ds}_fit_median_min'] = fmt(float(np.median(secs_ds)) / 60, 0)
        n[f'{ds}_fit_max_min'] = fmt(max(secs_ds) / 60, 0)
        n[f'{ds}_n_subjects'] = int(subjects_of(run_dir(runs[runs['dataset'] == ds].iloc[0])).shape[0])
    dirs = [run_dir(r) for _, r in runs.iterrows()]               # the Methods describe all runs
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
        n[f'{ds}_vpc_miss_times'] = span([t for t, q in misses if q == 'p95'], 0)
        n[f'{ds}_vpc_miss_upper'] = f"{sum(q == 'p95' for _, q in misses)} of {len(misses)}"
        # the same VPCs of the PKPy2 fit of the reference model (benchmarks/reference_vpc.py)
        rv = BENCH / 'reference_fits' / ds / 'vpc.json'
        if rv.exists():
            rv = json.loads(rv.read_text(encoding='utf-8'))
            n[f'reffit_{ds}_vpc'] = fmt(rv['plain']['inside_percent'], 0)
            n[f'reffit_{ds}_pcvpc'] = fmt(rv['prediction_corrected']['inside_percent'], 0)
            if ds == 'remifentanil':
                n['reffit_remifentanil_vpc_upper'] = f"{rv['plain']['misses_p95']} of {rv['plain']['misses']}"

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
    # oral MM: residual model of the final models (the simulation used exponential error)
    og = main_grid[main_grid['dataset'] == 'oral_mm']
    logn = []
    for _, r in og.iterrows():
        res_ = (json.loads((run_dir(r) / 'results.json').read_text(encoding='utf-8'))['final_model']['specification']
                .get('residual') or {})
        logn.append('lognormal' in json.dumps(res_))
    og = og.assign(lognormal=logn)
    n['oral_mm_lognormal'] = of_runs(int(og['lognormal'].sum()), len(og))
    n['oral_mm_lognormal_llm'] = ', '.join(sorted({LLM[x] for x in og.loc[og['lognormal'], 'llm']}))
    n['oral_mm_lognormal_dofv'] = span(og.loc[og['lognormal'], 'delta_ofv_vs_reference'], 1)
    fitted_logn = [any('lognormal' in json.dumps(json.loads(sp.read_text(encoding='utf-8')).get('residual'))
                       for sp in run_dir(r).glob('models/*/spec.json')) for _, r in og.iterrows()]
    if fitted_logn == list(og['lognormal']):
        n['oral_mm_lognormal_when_fitted'] = 'yes'         # every run that fitted log-normal error selected it
    n['oral_mm_other_dofv'] = span(og.loc[~og['lognormal'], 'delta_ofv_vs_reference'], 1)
    # remifentanil: what a quartered V3 changes in older subjects (published parameters, LBM 55 kg): 50% and 80%
    # decrement times after 1- to 10-hour infusions
    from scipy.linalg import expm

    def minto(age, lbm=55.):
        return dict(V1=5.1 - .0201 * (age - 40) + .072 * (lbm - 55), V2=9.82 - .0811 * (age - 40) + .108 * (lbm - 55),
                    V3=5.42, CL=2.6 - .0162 * (age - 40) + .0191 * (lbm - 55), Q2=2.05 - .0301 * (age - 40),
                    Q3=.076 - .00113 * (age - 40))

    def decrement(p, minutes, frac, dt=.01):
        A = np.array([[-(p['CL'] + p['Q2'] + p['Q3']) / p['V1'], p['Q2'] / p['V2'], p['Q3'] / p['V3']],
                      [p['Q2'] / p['V1'], -p['Q2'] / p['V2'], 0.], [p['Q3'] / p['V1'], 0., -p['Q3'] / p['V3']]])
        x = np.linalg.solve(A, (expm(A * minutes) - np.eye(3)) @ np.array([1., 0., 0.]))  # unit-rate infusion
        c0, step, t = x[0] / p['V1'], expm(A * dt), 0.
        while x[0] / p['V1'] > (1 - frac) * c0:
            x, t = step @ x, t + dt
        return t, c0
    ratio_q3, diff, conc = [], [], []
    for age in (65, 75, 85):
        p = minto(age)
        ratio_q3.append(100 * p['Q3'] / p['CL'])
        for hours in (1, 4, 10):
            for frac in (.5, .8):
                (a, ca), (b, cb) = decrement(p, 60 * hours, frac), decrement(dict(p, V3=p['V3'] / 4), 60 * hours, frac)
                diff.append(abs(a - b))
                conc.append(abs(cb / ca - 1) * 100)
    n['v3_q3_cl_pct'] = span(ratio_q3, 0)
    n['v3_decrement_diff_max'] = fmt(max(diff), 2)
    n['v3_conc_diff_max'] = fmt(max(conc), 0)
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
        support, weak, untested = [], [], 0
        for r, f in rk:
            t = tests.get((r['dataset'], r['condition'], r['llm'], r['rep']), {})
            tested_any = False
            for e in kept_v3(f):
                v = [x['delta_ofv'] for x in t.get('covariate_tests', []) if x['relationship'] == '~'.join(e)]
                if v:
                    tested_any = True
                    support.append(max(v))
                    if max(v) < 6.63:
                        weak.append((r['llm'], e, max(v)))
            untested += bool(kept_v3(f)) and not tested_any
        n['rk_v3_support'] = span(support, 1)
        n['rk_v3_strong_support'] = span([v for v in support if v >= 6.63], 1)
        n['rk_v3_untested'] = word(untested)
        if weak:
            n['rk_v3_weak'] = 'yes'
            n['rk_v3_weak_sentence'] = '; '.join(
                f"one {LLM[llm]} run retained as analyst-supported an effect of {e[1].replace('AGE', 'age')} on V3 that "
                f"failed its own retention criterion (OFV change {fmt(v, 1)})" for llm, e, v in weak)
            n['rk_v3_weak_short'] = '; '.join(f"{e[1].replace('AGE', 'age')}, OFV change {fmt(v, 1)}"
                                              for llm, e, v in weak)
        # the age effect on V1 with the statement: runs that tested it alone (weak support reported: see check_claims)
        v1k = [(key, x) for key, t in tests.items() if key[:2] == ('remifentanil', 'knowledge')
               for x in t['covariate_tests'] if x['relationship'] == 'V1~AGE']
        n['rk_v1age_tested'] = word(len({key for key, _ in v1k}))
        untested_v1 = [key for key in tests if key[:2] == ('remifentanil', 'knowledge') and key not in {k for k, _ in v1k}]
        n['rk_v1age_untested'] = word(len(untested_v1))
        if untested_v1 and {k[2] for k in untested_v1} == {'claude'}:
            n['rk_v1age_untested_llm'] = LLM['claude']
    # remifentanil V3: the only typical value outside 0.80-1.25 (and the reason typical values rarely agree)
    rg = main_grid[main_grid['dataset'] == 'remifentanil']
    v3 = rg['ratio_V3_median'].dropna()
    n['remi_v3_agree'] = f"{int(((v3 >= .8) & (v3 <= 1.25)).sum())} of {len(v3)}"
    n['remi_v3_span'] = span(v3, 2)
    others_ok = all(((rg[f'ratio_{c}_median'] >= .8) & (rg[f'ratio_{c}_median'] <= 1.25)).all()
                    for c in ('CL', 'V1', 'V2', 'Q2', 'Q3'))
    if others_ok:
        n['remi_others_agree'] = 'yes'
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
        n['rk_lbm_q_kept'] = none_of(kept_q)

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
            n['strong_missed_word'] = word(len(missed))           # 'in all but one run' (checked in check_claims.py)
            n['strong_missed_desc'] = '; '.join(
                f"{word(1)} {LLM[m['llm']]} {LABEL[m['dataset']].lower()} run omitted "
                f"{', '.join(effect_words(e) for e in m['missing'])} after {word(m['fits'])} fits"
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
                # Apgar effects on CL estimated without a bound: is a low score linked to higher CL (opposite to the
                # statement)? Indicator columns of a low score: positive coefficient; the continuous score: negative.
                res_ = json.loads((run_dir(r) / 'results.json').read_text(encoding='utf-8'))
                expr = {(t.get('column') or t.get('name')): t.get('expression', '') for t in res_.get('data_transformations', [])}
                signs = []
                for m, s_, su in ms:
                    est_ = {ce['effect'].split('=')[0]: ce['coefficient'] for ce in su.get('covariate_effects', [])}
                    for c in s_['covariates']:
                        co = c.get('coefficient') or {}
                        source = 'APGR' if c['covariate'] == 'APGR' else ('APGR' if 'APGR' in expr.get(c['covariate'], '')
                                                                          else None)
                        if c['parameter'] != 'CL' or source is None or co.get('fixed') \
                                or co.get('lower') is not None or co.get('upper') is not None:
                            continue
                        b_ = est_.get(f"CL~{c['covariate']}")
                        if b_ is None:
                            continue
                        low_indicator = c['covariate'] != 'APGR' and '<' in expr.get(c['covariate'], '')
                        signs.append(b_ > 0 if low_indicator else b_ < 0)
                opposite += bool(signs) and all(signs)
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
            om_second, om_final_daic, om_rse, om_fixed, om_no_se, om_dropped = [], [], [], 0, 0, []

            def _rest_residual(sp):
                r_ = sp.get('residual') or {}
                return sorted(r_) if set(r_) <= {'proportional', 'additive', 'lognormal'} \
                    else sorted((o, sorted(v)) for o, v in r_.items())
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
                    two.append(min(a2) - min(a1))                    # AIC of two minus one compartment
                # the final (two-compartment) model against the one-compartment model differing only in the
                # compartments, against the best one-compartment model of the run, and its peripheral parameters
                res_ = json.loads((run_dir(r) / 'results.json').read_text(encoding='utf-8'))
                fid = res_['final_model']['model_id']
                fin = next(((m, s_, su) for m, s_, su in ms if m['model_id'] == fid), None)
                if fin and fin[1]['structure'].get('compartments', 1) == 2:
                    fm_, fs_, fsu = fin
                    def iiv1(sp):
                        return sorted({'V1': 'V'}.get(k, k) for k in sp.get('iiv', {}) if k not in ('Q', 'V2'))
                    pair = [m['ofv'] - fm_['ofv'] for m, s_, _ in ms if s_['structure'].get('type') == 'michaelis_menten'
                            and s_['structure'].get('compartments', 1) == 1 and iiv1(s_) == iiv1(fs_)
                            and _rest_residual(s_) == _rest_residual(fs_) and not s_['covariates'] and not fs_['covariates']]
                    if pair:
                        om_second.append(min(pair))                 # against the best such model
                    if a1:
                        om_final_daic.append(fm_['aic'] - min(a1))
                    for q in fsu.get('parameters', []):
                        if q['parameter'] in ('Q', 'V2'):
                            if q.get('fixed'):
                                om_fixed += 1
                            elif q.get('rse_percent') is not None:
                                om_rse.append(q['rse_percent'])
                    om_no_se += fsu.get('uncertainty_status') != 'computed'
                    # variability terms supported by the data but left out of the final model
                    for m, s_, _ in ms:
                        if s_['structure'] == fs_['structure'] and _rest_residual(s_) == _rest_residual(fs_) \
                                and set(s_.get('iiv', {})) > set(fs_.get('iiv', {})) \
                                and len(set(s_['iiv']) - set(fs_['iiv'])) == 1 and fm_['ofv'] - m['ofv'] >= 6.63:
                            om_dropped.append((sorted(set(s_['iiv']) - set(fs_['iiv']))[0], fm_['ofv'] - m['ofv']))
            # control: runs without the statement that compared one and two compartments (MM, two parameters)
            ctrl = [(key, c['delta_ofv']) for key, t in tests.items() if key[:2] == ('oral_mm', 'none')
                    for c in t['compartment_tests'] if c['compartments'] == [1, 2] and c.get('type') == 'michaelis_menten'
                    and c['extra_parameters'] == 2]
            n['om_control_runs'] = word(len({k for k, _ in ctrl}))
            n['om_control_dofv'] = span([v for _, v in ctrl], 1)
            n['om_control_all_one'] = 'yes' if all(
                runs[(runs['dataset'] == 'oral_mm') & (runs['condition'] == 'none') & (runs['llm'] == k[2])
                     & (runs['rep'] == k[3])]['compartments'].iloc[0] == 1 for k, _ in ctrl) else 'no'
            n['om_runs'] = of_runs(len(om), len(om))
            n['om_mm_kept'] = of_runs(int((om['elimination'] == 'Michaelis-Menten').sum()), len(om))
            n['om_mm_gain'] = span(gain, 0)
            n['om_2cmt_adopted'] = of_runs(int((om['compartments'] == 2).sum()), len(om))
            n['om_2cmt_daic'] = span(two, 1)
            n['om_2cmt_second'] = span(om_second, 1)
            n['om_2cmt_final_daic'] = span(om_final_daic, 1)
            n['om_2cmt_rse'] = span(om_rse, 0)
            n['om_2cmt_fixed'] = word(om_fixed)
            n['om_2cmt_no_se'] = word(om_no_se)
            if om_dropped:
                n['om_dropped_iiv'] = '; '.join(f'interindividual variability of {k} (OFV decrease {fmt(v, 1)})'
                                                for k, v in om_dropped)
                n['om_dropped_runs'] = word(len(om_dropped))
        n['misleading_runs'] = len(mis)

    # remifentanil: backward elimination from the reference model (benchmarks/backward_baseline.py)
    bb_path = EVAL / 'backward_baseline.json'
    if bb_path.exists():
        bb = json.loads(bb_path.read_text(encoding='utf-8')).get('remifentanil')
        if bb and 'removed' in bb:
            n['backward_remifentanil_removed'] = ', '.join(effect_words(e) for e in bb['removed']) or 'nothing'
            last = bb['history'][-1]['tests']
            kept = [v for k, v in last.items() if v is not None and k not in bb['removed']]
            n['backward_remifentanil_min'] = fmt(min(kept), 1) if kept else 'NA'
            if bb['removed'] == ['V1~AGE']:
                n['baseline_remi_agrees'] = 'yes'
            # runs without knowledge whose final relationships equal those of the deterministic baseline
            base = {'remifentanil': set(REFERENCE['remifentanil']) - {tuple(e.split('~')) for e in bb['removed']}}
            if 'pheno' in scm:
                base['pheno'] = {(x.split('(')[0].split('~')[0], x.split('(')[0].split('~')[1].split('_')[0].split('=')[0])
                                 for x in scm['pheno']['included']}
            for ds, b in base.items():
                sub = [set(f) for f, (_, r) in zip(rel, runs.iterrows()) if r['dataset'] == ds and r['condition'] == 'none']
                n[f'baseline_match_{ds}'] = of_runs(sum(s == b for s in sub), len(sub))

    # ------------------------------------------------------------------ robustness of single-start fits (all runs)
    def cov_sig(sp):
        return tuple(sorted((str((c['parameter'], c['covariate'], c['form'], c.get('center'), c.get('level'),
                                  (c.get('coefficient') or {}).get('value') if (c.get('coefficient') or {}).get('fixed')
                                  else None, (c.get('coefficient') or {}).get('lower'),
                                  (c.get('coefficient') or {}).get('upper'))) for c in sp.get('covariates', []))))

    def fixed_params(sp):
        return tuple(sorted((k, v.get('value')) for k, v in sp.get('parameters', {}).items() if (v or {}).get('fixed')))

    groups, inversions = {}, []
    for _, r in runs.iterrows():
        res = json.loads((run_dir(r) / 'results.json').read_text(encoding='utf-8'))
        conv = {}
        for m in res['models']:
            if m.get('status') != 'converged' or m.get('ofv') is None:
                continue
            sp = json.loads((run_dir(r) / 'models' / m['model_id'] / 'spec.json').read_text(encoding='utf-8'))
            conv[m['model_id']] = (m, sp)
            groups.setdefault((r['dataset'], _rest(sp), cov_sig(sp), fixed_params(sp)), []).append(
                (m['ofv'], json.dumps(sp.get('parameters'), sort_keys=True)))
        # a converged model with one more residual component or IIV term but a higher OFV than its submodel
        for a, (ma, sa) in conv.items():
            for b, (mb, sb) in conv.items():
                if a == b or cov_sig(sa) != cov_sig(sb) or fixed_params(sa) != fixed_params(sb) \
                        or sa.get('structure') != sb.get('structure') \
                        or sorted(map(sorted, sa.get('iiv_blocks', []))) != sorted(map(sorted, sb.get('iiv_blocks', []))):
                    continue
                def comps(sp):
                    r_ = sp.get('residual') or {}
                    return {('', k) for k in r_} if set(r_) <= {'proportional', 'additive', 'lognormal'} \
                        else {(o, k) for o, v in r_.items() for k in v}
                ca, cb, ia, ib = comps(sa), comps(sb), set(sa.get('iiv', {})), set(sb.get('iiv', {}))
                one_more = (ia == ib and ca < cb and len(cb - ca) == 1) or (ca == cb and ia < ib and len(ib - ia) == 1)
                if one_more and mb['ofv'] > ma['ofv'] + 1:
                    inversions.append(mb['ofv'] - ma['ofv'])
    spreads = []
    for v in groups.values():
        if len({x[1] for x in v}) > 1:
            spreads.append(max(x[0] for x in v) - min(x[0] for x in v))
        else:
            assert max(x[0] for x in v) - min(x[0] for x in v) < 1e-6      # identical specifications, identical OFV
    spreads.sort(reverse=True)
    n['start_groups'] = len(spreads)
    n['start_spread_max'] = fmt(spreads[0], 0) if spreads else 'NA'
    n['start_spread_next'] = fmt(spreads[1], 0) if len(spreads) > 1 else 'NA'
    n['start_spread_over1'] = word(sum(x > 1 for x in spreads))
    n['nested_inversions'] = word(len(set(round(x, 3) for x in inversions)))
    n['nested_inversion_max'] = fmt(max(inversions), 0) if inversions else 'NA'

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
