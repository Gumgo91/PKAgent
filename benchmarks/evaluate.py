"""Compare the final models of the benchmark runs with the published reference models.

For every run (benchmarks/runs/<dataset>/<condition>/<model>/rep<k>/results.json):
- structure: number of compartments and elimination type against the reference;
- covariates: which (parameter, covariate) relationships were retained, against the reference set (derived columns
  are traced to their source columns through the recorded data transformations);
- typical values: for every subject, the typical value of each reference parameter under the agent's final model
  and under the reference model at the subject's covariates; the ratio agent/reference is summarized by its median
  and range (a parameterization-free comparison of covariate models);
- likelihood: OFV and AIC of the final model against the PKPy2 fit of the reference model on the same data;
- evaluation: VPC coverage, CWRES, largest RSE, shrinkage; process: fits, LLM turns, tokens, cost, hours.
Writes benchmarks/evaluation/runs.csv, summary.csv and evaluation.json.
"""
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATASETS = json.loads((HERE / 'datasets.json').read_text(encoding='utf-8'))
OUT = HERE / 'evaluation'


# ---------------------------------------------------------------------- typical values of a fitted model
def _effect(form, beta, z, center, level):
    z = np.asarray(z, dtype=float)
    if form == 'power':
        return beta * np.log(z / center)
    if form == 'exponential':
        return beta * (z - center)
    if form == 'linear':
        return np.log(np.maximum(1 + beta * (z - center), 1e-12))
    if form == 'categorical':
        return beta * (z == level)
    raise ValueError(form)


def typical_values(spec, summary, subjects):
    """Typical value of every structural parameter for each subject (rows of `subjects`)."""
    est = {r['parameter']: r['estimate'] for r in summary['parameters']}
    coef = {r['effect']: r['coefficient'] for r in summary.get('covariate_effects', [])}
    out = {}
    for p, d in spec['parameters'].items():
        scale = d.get('scale', 'log')
        theta = est[p]
        if scale == 'log':
            base = np.full(len(subjects), math.log(theta))
        elif scale == 'logit':
            base = np.full(len(subjects), math.log(theta / (1 - theta)))
        else:
            base = np.full(len(subjects), theta)
        for c in spec['covariates']:
            if c['parameter'] != p:
                continue
            label = f"{c['parameter']}~{c['covariate']}" + (f"={c['level']:g}" if c['level'] is not None else '')
            beta = coef.get(label, c['coefficient']['value'])
            base = base + _effect(c['form'], beta, subjects[c['covariate']].to_numpy(float), c['center'], c['level'])
        out[p] = np.exp(base) if scale == 'log' else 1 / (1 + np.exp(-base)) if scale == 'logit' else base
    return out


# ---------------------------------------------------------------------- reference typical values
def reference_typical(name, subjects):
    if name == 'pheno':
        wt, low = subjects['WT'].to_numpy(float), (subjects['APGR'].to_numpy(float) < 5)
        return dict(CL=.00469555 * wt, V=.984258 * wt * (1 + .158920 * low))
    if name == 'remifentanil':
        age, lbm = subjects['AGE'].to_numpy(float) - 40, subjects['LBM'].to_numpy(float) - 55
        return dict(V1=5.1 - .0201 * age + .072 * lbm, V2=9.82 - .0811 * age + .108 * lbm, V3=np.full(len(age), 5.42),
                    CL=2.6 - .0162 * age + .0191 * lbm, Q2=2.05 - .0301 * age, Q3=.076 - .00113 * age)
    if name == 'oral_mm':
        r = DATASETS[name]['reference']['estimates']
        return {k: np.full(len(subjects), float(v)) for k, v in r.items()}
    raise KeyError(name)


# equivalent parameter names between structures (agent name -> reference name)
ALIASES = {'V': ['V1'], 'V1': ['V'], 'Q': ['Q2'], 'Q2': ['Q']}


def _match(ref_name, agent_params):
    if ref_name in agent_params:
        return ref_name
    for alt in ALIASES.get(ref_name, []):
        if alt in agent_params:
            return alt
    return None


# ---------------------------------------------------------------------- covariate relationships
REFERENCE_COVARIATES = {
    'pheno': {('CL', 'WT'), ('V', 'WT'), ('V', 'APGR')},
    'remifentanil': {('V1', 'AGE'), ('V1', 'LBM'), ('V2', 'AGE'), ('V2', 'LBM'), ('CL', 'AGE'), ('CL', 'LBM'),
                     ('Q2', 'AGE'), ('Q3', 'AGE')},
    'oral_mm': set(),
}


REFERENCE_FORMS = {            # functional form of each reference relationship (power with a fixed exponent of 1 for WT)
    'pheno': {('CL', 'WT'): 'power', ('V', 'WT'): 'power', ('V', 'APGR'): 'categorical'},
    'remifentanil': {k: 'linear' for k in REFERENCE_COVARIATES['remifentanil']},
    'oral_mm': {},
}


def relationship_forms(spec, transformations, base_columns, reference_names):
    """{(reference parameter, source covariate): set of functional forms} of the final model."""
    found = {}
    for c in spec['covariates']:
        if c['coefficient'].get('fixed') and c['coefficient']['value'] == 0:
            continue
        for src in _sources(c['covariate'], transformations, base_columns):
            found.setdefault((to_reference_name(c['parameter'], reference_names), src), set()).add(c['form'])
    return found


def form_agreement(spec, transformations, base_columns, reference_names, name):
    """Fraction of the reference relationships present in the final model with the reference functional form."""
    forms = REFERENCE_FORMS[name]
    if not forms:
        return None
    found = relationship_forms(spec, transformations, base_columns, reference_names)
    return sum(1 for k, f in forms.items() if f in found.get(k, set())) / len(forms)


def _sources(column, transformations, base_columns):
    """Source columns of a (possibly derived) covariate column."""
    for t in reversed(transformations):
        if t['column'] == column:
            names = set(re.findall(r'[A-Za-z_][A-Za-z_0-9]*', t['expression'])) - {'log', 'exp', 'sqrt', 'abs', 'min', 'max', 'where'}
            out = set()
            for n in names:
                out |= _sources(n, [x for x in transformations if x is not t], base_columns) if n not in base_columns else {n}
            return out or {column}
    return {column}


def to_reference_name(p, reference_names):
    """Name of the reference parameter that an agent parameter corresponds to (central volume, first
    intercompartmental clearance), or the name itself."""
    if p in reference_names:
        return p
    for alt in ALIASES.get(p, []):
        if alt in reference_names:
            return alt
    return p


def covariate_relationships(spec, transformations, base_columns, reference_names):
    rel = set()
    for c in spec['covariates']:
        if c['coefficient'].get('fixed') and c['coefficient']['value'] == 0:
            continue
        for src in _sources(c['covariate'], transformations, base_columns):
            rel.add((to_reference_name(c['parameter'], reference_names), src))
    return rel


def recall_mentions(run_dir):
    """LLM statements that refer to prior knowledge of the dataset or its published analysis (memorization check):
    the matched terms and the turns that contain them (benchmarks/recall.py: assistant text, tool-call arguments
    and reasoning summaries)."""
    from recall import recalled
    hits = {}
    for h in recalled(run_dir):
        hits.setdefault(h['term'].lower(), []).append(h['turn'])
    return {k: sorted(set(v)) for k, v in hits.items()}


def misleading_followed(name, spec, found, row):
    """Which wrong claims of the misleading sentence the final model adopted (';'-separated), or 'none'."""
    adopted = []
    if name == 'pheno':
        if ('CL', 'WT') not in found and ('V', 'WT') not in found:
            adopted.append('no weight effect')
        if ('CL', 'APGR') in found:
            adopted.append('Apgar on CL')
    elif name == 'oral_mm':
        if row.get('elimination') == 'linear':
            adopted.append('linear elimination')
        if row.get('compartments') == 2:
            adopted.append('two compartments')
    return ';'.join(adopted) or 'none'


def evaluate_run(path, reference_fits):
    res = json.loads(path.read_text(encoding='utf-8'))
    parts = path.parent.relative_to(HERE / 'runs').parts
    name, condition, model, rep = parts[0], parts[1], parts[2], parts[3]
    row = dict(dataset=name, condition=condition, llm=model, rep=rep, finalized=res['run']['finalized'],
               fits=res['run']['fits'], llm_calls=res['run']['llm_calls'], prompt_tokens=res['run']['prompt_tokens'],
               completion_tokens=res['run']['completion_tokens'], cost_usd=res['run']['cost_usd'], hours=res['run']['hours'])
    recall = recall_mentions(path.parent)
    row['recall_terms'] = ';'.join(f'{k}@{",".join(map(str, sorted(set(v))))}' for k, v in sorted(recall.items()))
    final = res.get('final_model')
    if not final:
        return row, None
    spec, s = final['specification'], final['summary']
    ref = DATASETS[name]['reference']
    st = spec['structure']
    kind = st.get('type', 'pk')
    row.update(model_id=final['model_id'], description=final['description'], status=s.get('status'),
               compartments=st.get('compartments', 1), structure_type=kind if kind != 'tmdd' else f"tmdd_{st.get('tmdd_kind', 'full')}",
               ofv=s.get('ofv'), n_estimated=s.get('n_estimated'), aic=s.get('aic'))
    row['compartments_match'] = row['compartments'] == ref['structure']['compartments']
    ref_elim = ref['structure']['elimination']
    if kind == 'michaelis_menten':
        row['elimination'] = 'Michaelis-Menten' + (' + linear' if st.get('linear_clearance') else '')
    else:
        row['elimination'] = 'linear' if kind == 'pk' else kind
    row['elimination_match'] = row['elimination'] == ref_elim
    row['structure_match'] = bool(row['compartments_match'] and row['elimination_match'])
    data = pd.read_csv(final['data_file'], na_values=['.'])
    base_columns = set(pd.read_csv(HERE / 'data' / DATASETS[name]['file'], nrows=1).columns)
    transformations = res.get('data_transformations', [])
    subjects = data.groupby('ID').first().reset_index()
    ref_tv = reference_typical(name, subjects)
    found = covariate_relationships(spec, transformations, base_columns, set(ref_tv))
    target = REFERENCE_COVARIATES[name]
    row['covariates_found'] = ';'.join(f'{p}~{c}' for p, c in sorted(found))
    row['covariate_true_positives'] = len(found & target)
    row['covariate_false_positives'] = len(found - target)
    row['covariate_false_negatives'] = len(target - found)
    row['covariates_exact'] = found == target
    row['covariate_recall'] = len(found & target) / len(target) if target else None
    row['covariate_precision'] = len(found & target) / len(found) if found else (1. if not target else None)
    row['form_agreement'] = form_agreement(spec, transformations, base_columns, set(ref_tv), name)
    agent_tv = typical_values(spec, s, subjects)
    ratios = {}
    for p, ref_values in ref_tv.items():
        q = _match(p, agent_tv)
        if q is None:
            continue
        r = agent_tv[q] / ref_values
        ratios[p] = dict(median=float(np.median(r)), min=float(np.min(r)), max=float(np.max(r)))
        row[f'ratio_{p}_median'] = ratios[p]['median']
        row[f'ratio_{p}_range'] = f"{ratios[p]['min']:.3f}-{ratios[p]['max']:.3f}"
    row['parameters_missing'] = ';'.join(p for p in ref_tv if p not in ratios)
    if ratios:
        row['max_abs_log_ratio'] = max(abs(math.log(v['median'])) for v in ratios.values())
        row['all_within_20pct'] = (not row['parameters_missing']
                                   and all(.8 <= v['median'] <= 1.25 for v in ratios.values()))
    row['reproduced'] = bool(row.get('structure_match') and row.get('covariates_exact')
                             and row.get('all_within_20pct'))
    if condition == 'misleading':
        row['misleading_followed'] = misleading_followed(name, spec, found, row)
    rf = reference_fits.get(name)
    if rf and rf.get('ofv') is not None and s.get('ofv') is not None:
        row['delta_ofv_vs_reference'] = s['ofv'] - rf['ofv']
        row['delta_aic_vs_reference'] = s['aic'] - rf['aic']
    vpc = (final.get('vpc') or {}).get('result') or {}
    cov = [v.get('observed_percentiles_inside') for v in vpc.values() if isinstance(v, dict)]
    if cov:
        inside = sum(int(c.split('/')[0]) for c in cov if c)
        total = sum(int(c.split('/')[1]) for c in cov if c)
        row['vpc_inside_fraction'] = inside / total if total else None
    rses = [r.get('rse_percent') for r in s.get('parameters', []) if r.get('rse_percent') is not None]
    row['max_rse_structural'] = max(rses) if rses else None
    diag = (s.get('diagnostics') or {}).get('outputs') or {}
    first = next(iter(diag.values()), {})
    row['cwres_mean'], row['cwres_sd'] = first.get('cwres_mean'), first.get('cwres_sd')
    shrink = (s.get('diagnostics') or {}).get('eta_shrinkage_percent') or {}
    row['max_eta_shrinkage'] = max((v for v in shrink.values() if v is not None), default=None)
    forms = relationship_forms(spec, transformations, base_columns, set(ref_tv))
    return row, dict(ratios=ratios, covariates=sorted(found), description=final['description'],
                     relationships=[[p, c, sorted(f)] for (p, c), f in sorted(forms.items())])


def reference_fit_ratios(name):
    """Typical values of the PKPy2 fit of the reference model divided by the published (or simulated) reference values:
    median and range over subjects, per reference parameter (what the data and engine give for the reference model)."""
    base = HERE / 'reference_fits' / name
    if not (base / 'models' / 'M001' / 'summary.json').exists():
        return {}
    spec = json.loads((base / 'models' / 'M001' / 'spec.json').read_text(encoding='utf-8'))
    summary = json.loads((base / 'models' / 'M001' / 'summary.json').read_text(encoding='utf-8'))
    data_file = sorted((base / 'data').glob('data_v*.csv'))[-1]
    subjects = pd.read_csv(data_file, na_values=['.']).groupby('ID').first().reset_index()
    fit_tv, ref_tv = typical_values(spec, summary, subjects), reference_typical(name, subjects)
    out = {}
    for p, ref_values in ref_tv.items():
        q = _match(p, fit_tv)
        if q is not None:
            r = fit_tv[q] / ref_values
            out[p] = dict(median=float(np.median(r)), min=float(np.min(r)), max=float(np.max(r)))
    return out


def main():
    OUT.mkdir(exist_ok=True)
    reference_fits = {}
    for name in DATASETS:
        p = HERE / 'reference_fits' / name / 'reference_fit.json'
        if p.exists():
            reference_fits[name] = json.loads(p.read_text(encoding='utf-8'))
    (OUT / 'reference_fit_ratios.json').write_text(json.dumps({n: reference_fit_ratios(n) for n in DATASETS}, indent=1),
                                                  encoding='utf-8')
    rows, details = [], {}
    for path in sorted((HERE / 'runs').glob('*/*/*/rep*/results.json')):
        row, detail = evaluate_run(path, reference_fits)
        rows.append(row)
        details[str(path.parent.relative_to(HERE / 'runs'))] = detail
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / 'runs.csv', index=False)
    if len(runs):
        num = runs.select_dtypes('number').columns
        summary = runs.groupby(['dataset', 'condition', 'llm'])[list(num)].median(numeric_only=True)
        summary['runs'] = runs.groupby(['dataset', 'condition', 'llm']).size()
        for col in ('reproduced', 'structure_match', 'compartments_match', 'elimination_match', 'covariates_exact',
                    'all_within_20pct', 'finalized'):
            if col in runs:
                summary[col + '_rate'] = runs.groupby(['dataset', 'condition', 'llm'])[col].mean()
        summary.to_csv(OUT / 'summary.csv')
    (OUT / 'evaluation.json').write_text(json.dumps(dict(reference_fits={k: dict(ofv=v.get('ofv'), aic=v.get('aic'))
                                                                             for k, v in reference_fits.items()},
                                                         runs=rows, details=details), indent=1, default=str),
                                         encoding='utf-8')
    print(runs.to_string(max_colwidth=40) if len(runs) else 'no runs yet')


if __name__ == '__main__':
    main()
