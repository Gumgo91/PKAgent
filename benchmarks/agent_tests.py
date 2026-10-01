"""What the agents tested and how they worked, extracted from the run folders (no refitting).

For every run:
- covariate tests: pairs of converged models that differ by exactly one parameter-covariate relationship and are
  otherwise identical (structure, interindividual variability incl. blocks, residual model, other covariates); the
  OFV of the model without the relationship minus the OFV of the model with it is the evidence the agent had for that
  relationship in that model (derived covariates are traced to their source columns);
- structural tests: pairs without covariates or covariance blocks, with the same residual model and number of
  interindividual variances, that differ only in structure (compartments or type of elimination);
- tool use: calls per tool, whether the raw data were plotted, NCA and covariate screening run, the turn of the first
  goodness-of-fit plot view relative to the last fit, bootstrap replicates completed/converged, hours left at
  finalization;
- fits: status of every fitted model and the uncertainty status of the final model.
Writes evaluation/agent_tests.json. Usage: python benchmarks/agent_tests.py
"""
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
DATASETS = json.loads((HERE / 'datasets.json').read_text(encoding='utf-8'))

import sys                                                   # noqa: E402
sys.path.insert(0, str(HERE))
from evaluate import _sources, to_reference_name, reference_typical   # noqa: E402

REFERENCE_NAMES = {'pheno': {'CL', 'V'}, 'remifentanil': {'CL', 'V1', 'V2', 'V3', 'Q2', 'Q3'},
                   'oral_mm': {'Ka', 'V', 'VMAX', 'KM'}}


def base_columns(run_dir):
    data = sorted((run_dir / 'data').glob('data_v*.csv'))
    if data:
        return set(pd.read_csv(data[0], nrows=1).columns)
    return set()


def relationships(spec, transformations, cols, ref_names):
    """{(parameter, source covariate): (form, fixed coefficient)} of a specification."""
    out = {}
    for c in spec.get('covariates', []):
        p = to_reference_name(c['parameter'], ref_names)
        for src in _sources(c['covariate'], transformations, cols):
            coef = c.get('coefficient') or {}
            fixed = bool(coef.get('fixed'))
            out[(p, src)] = (c.get('form'), fixed, coef.get('value') if fixed else None, c.get('level'))
    return out


def rest(spec):
    """Everything of a specification except its covariates (for 'otherwise identical')."""
    s = {k: v for k, v in spec.items() if k not in ('covariates', 'name', 'parent', 'parameters', 'defaulted')}
    s['iiv'] = sorted(spec.get('iiv', {}))
    s['iiv_blocks'] = sorted(sorted(b) for b in spec.get('iiv_blocks', []))
    s['residual'] = {k: sorted(v) for k, v in (spec.get('residual') or {}).items()}
    s['fixed'] = sorted(k for k, v in spec.get('parameters', {}).items() if (v or {}).get('fixed'))
    return json.dumps(s, sort_keys=True)


def residual_kind(spec):
    return {k: sorted(v) for k, v in (spec.get('residual') or {}).items()}


def structure_label(spec):
    st = spec.get('structure', {})
    return (st.get('type', 'pk'), st.get('compartments', 1), bool(st.get('linear_clearance')))


def analyze(run_dir):
    parts = run_dir.relative_to(RUNS).parts
    ds = parts[0]
    res = json.loads((run_dir / 'results.json').read_text(encoding='utf-8'))
    trans = res.get('data_transformations', [])
    cols = base_columns(run_dir)
    ref_names = REFERENCE_NAMES[ds]
    models = {}
    for m in res.get('models', []):
        p = run_dir / 'models' / m['model_id'] / 'spec.json'
        if p.exists():
            models[m['model_id']] = dict(m, spec=json.loads(p.read_text(encoding='utf-8')))
    conv = {k: m for k, m in models.items() if m.get('status') == 'converged' and m.get('ofv') is not None}
    tests, structural, compartment_tests = [], [], []
    for a, b in combinations(sorted(conv), 2):
        sa, sb = conv[a]['spec'], conv[b]['spec']
        ra, rb = relationships(sa, trans, cols, ref_names), relationships(sb, trans, cols, ref_names)
        if rest(sa) == rest(sb):
            diff = set(ra) ^ set(rb)
            shared = set(ra) & set(rb)
            if len(diff) == 1 and all(ra[k] == rb[k] for k in shared):
                rel = diff.pop()
                with_id, without_id = (a, b) if rel in ra else (b, a)
                form, fixed = (ra if rel in ra else rb)[rel][:2]
                tests.append(dict(relationship=f'{rel[0]}~{rel[1]}', form=form, fixed=fixed, with_model=with_id,
                                  without_model=without_id,
                                  delta_ofv=round(conv[without_id]['ofv'] - conv[with_id]['ofv'], 3),
                                  n_relationships_with=len(ra if rel in ra else rb),
                                  iiv_blocks=bool(sa.get('iiv_blocks'))))
        # compartment tests: same type of elimination, residual model and covariate relationships, no covariance
        # blocks, different number of compartments (the added compartment may carry its own variances)
        st_a, st_b = structure_label(sa), structure_label(sb)
        small = sa if st_a[1] < st_b[1] else sb
        kept = {to_reference_name(q, ref_names) for q in small.get('parameters', {})}
        same_cov = {k: v for k, v in ra.items() if k[0] in kept} == {k: v for k, v in rb.items() if k[0] in kept}
        if st_a[0] == st_b[0] and st_a[2] == st_b[2] and st_a[1] != st_b[1] and same_cov \
                and not sa.get('iiv_blocks') and not sb.get('iiv_blocks') \
                and residual_kind(sa) == residual_kind(sb):
            lo_, hi_ = (a, b) if st_a[1] < st_b[1] else (b, a)
            compartment_tests.append(dict(fewer=lo_, more=hi_, compartments=[min(st_a[1], st_b[1]), max(st_a[1], st_b[1])],
                                          covariates=len(relationships(small, trans, cols, ref_names)),
                                          delta_ofv=round(conv[lo_]['ofv'] - conv[hi_]['ofv'], 3),
                                          extra_parameters=conv[hi_].get('n_estimated', 0) - conv[lo_].get('n_estimated', 0)))
        if not ra and not rb and not sa.get('iiv_blocks') and not sb.get('iiv_blocks') \
                and structure_label(sa) != structure_label(sb) \
                and residual_kind(sa) == residual_kind(sb) \
                and len(sa.get('iiv', {})) == len(sb.get('iiv', {})):
            structural.append(dict(models=[a, b], structures=[list(structure_label(sa)), list(structure_label(sb))],
                                   ofv=[conv[a]['ofv'], conv[b]['ofv']]))
    # mark the tests made in the covariate context of the final model (all its other relationships present) and
    # in its stochastic model
    fm = (res.get('final_model') or {}).get('specification')
    if fm:
        rf = relationships(fm, trans, cols, ref_names)
        for t in tests:
            spec_with = conv[t['with_model']]['spec']
            rw = relationships(spec_with, trans, cols, ref_names)
            rel = tuple(t['relationship'].split('~'))
            t['final_context'] = set(rw) - {rel} == set(rf) - {rel}
            t['final_stochastic'] = rest(spec_with) == rest(fm)
    log = [json.loads(line) for line in (run_dir / 'tool_log.jsonl').read_text(encoding='utf-8').splitlines()]
    calls = {}
    for t in log:
        calls[t['tool']] = calls.get(t['tool'], 0) + 1
    fit_turns = [t['turn'] for t in log if t['tool'] in ('fit_models', 'covariate_search')]
    view_turns = [t['turn'] for t in log if t['tool'] == 'view_plots']
    boots = [dict(requested=t['result'].get('requested'), completed=t['result'].get('completed'),
                  converged=t['result'].get('converged'))
             for t in log if t['tool'] == 'resample_uncertainty' and isinstance(t.get('result'), dict)]
    final = res.get('final_model') or {}
    budget = res.get('budget') or {}
    statuses = [m.get('status') for m in models.values()]
    best = {}                                   # lowest OFV per structure among models without covariates
    for k, m in conv.items():
        if not m['spec'].get('covariates'):
            lab = '/'.join(map(str, structure_label(m['spec'])))
            best[lab] = min(best.get(lab, float('inf')), m['ofv'])
    return dict(dataset=ds, condition=parts[1], llm=parts[2], rep=parts[3], covariate_tests=tests,
                best_ofv_by_structure=best, compartment_tests=compartment_tests,
                structural_tests=structural, tool_calls=calls,
                plotted_data='plot_data' in calls, ran_nca='run_nca' in calls,
                screened_covariates='screen_covariates' in calls,
                first_plot_view_turn=min(view_turns) if view_turns else None,
                last_fit_turn=max(fit_turns) if fit_turns else None,
                fitted_after_viewing_plots=bool(view_turns and fit_turns and max(fit_turns) > min(view_turns)),
                bootstraps=boots, hours_left=budget.get('hours_left'),
                fits_total=len(models), fits_converged=statuses.count('converged'),
                final_uncertainty=(final.get('summary') or {}).get('uncertainty_status'),
                final_model=final.get('model_id'))


def main():
    out = []
    for path in sorted(RUNS.glob('*/*/*/rep*/results.json')):
        out.append(analyze(path.parent))
    (HERE / 'evaluation' / 'agent_tests.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    for r in out:
        tests = {}
        for t in r['covariate_tests']:
            tests.setdefault(t['relationship'], []).append(t['delta_ofv'])
        print(r['dataset'], r['condition'], r['llm'], r['rep'], {k: v for k, v in tests.items()},
              'struct', len(r['structural_tests']), 'plots-before-last-fit', r['fitted_after_viewing_plots'],
              'boot', r['bootstraps'], 'hours_left', r['hours_left'])


if __name__ == '__main__':
    main()
