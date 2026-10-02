"""Checks of the statements in paper/manuscript_cpt.md that are worded rather than filled from numbers.

Each check reads the run folders and evaluation outputs and fails when a worded claim no longer holds (for example
after more replicates). build_cpt.py runs these checks and refuses a final (non-draft) build when one fails.
Usage: python paper/check_claims.py
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from agent_tests import rest as _rest  # noqa: E402

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / 'benchmarks'
EVAL = BENCH / 'evaluation'


def runs_table():
    r = pd.read_csv(EVAL / 'runs.csv')
    return r[r['condition'].isin(['none', 'knowledge'])]


def results(r):
    return json.loads((BENCH / 'runs' / r['dataset'] / r['condition'] / r['llm'] / r['rep'] / 'results.json')
                      .read_text(encoding='utf-8'))


def report_text(r):
    rep = (results(r)['final_model'] or {}).get('report') or {}
    return ' '.join(str(v) for v in rep.values())


def covs(spec):
    return [(c['parameter'], c['covariate'], c['form'], c['coefficient']['fixed'], c['coefficient']['value'])
            for c in spec.get('covariates', [])]


def checks():
    runs = runs_table()
    tests = {(t['dataset'], t['condition'], t['llm'], t['rep']): t
             for t in json.loads((EVAL / 'agent_tests.json').read_text(encoding='utf-8'))}
    numbers = json.loads((HERE / 'build' / 'numbers.json').read_text(encoding='utf-8'))
    out = []

    def check(name, ok, detail=''):
        out.append((name, bool(ok), detail))

    pk = runs[(runs['dataset'] == 'pheno') & (runs['condition'] == 'knowledge')]
    check('phenobarbital expert runs reach OFV 871.147', all(abs(x - 871.147) < .01 for x in pk['ofv']), list(pk['ofv']))
    ok = True
    for _, r in pk.iterrows():
        c = covs(results(r)['final_model']['specification'])
        wt = [x for x in c if x[1] == 'WT']
        ok &= len(wt) == 2 and all(x[3] and x[4] == 1.0 for x in wt) and any(x[0] == 'V' and x[2] == 'categorical' for x in c)
    check('phenobarbital expert runs: weight exponents fixed at 1 and an Apgar indicator on V', ok)
    ok = all(any(abs(t['delta_ofv'] - 4.543) < .01 for t in tests[(r['dataset'], r['condition'], r['llm'], r['rep'])]
                 ['covariate_tests'] if t['relationship'] == 'V~APGR') for _, r in pk.iterrows())
    check('phenobarbital expert runs: Apgar test of 4.5 in the final model', ok)
    ok = all(re.search(r'6\.63|0\.01|retention|backward', report_text(r), re.I) for _, r in pk.iterrows())
    check('phenobarbital expert reports mention the retention criterion', ok)
    pn = runs[(runs['dataset'] == 'pheno') & (runs['condition'] == 'none')]
    ok = all(re.search(r'Apgar[^.]*cannot be (ruled out|excluded)', report_text(r), re.I)
             for _, r in pn[pn['llm'] == 'claude'].iterrows())
    check('Claude phenobarbital reports: an Apgar effect on V cannot be excluded', ok)
    check('some GPT phenobarbital run without knowledge chose or mentioned the Apgar cutoff of 5',
          not numbers.get('gpt_apgar_cutoff', 'none').startswith(('none', 'neither', 'not')))

    ok = all(any(set(b) == {'CL', 'V'} for b in results(r)['final_model']['specification'].get('iiv_blocks', []))
             for _, r in pn[pn['llm'] == 'claude'].iterrows())
    check('Claude phenobarbital runs without knowledge: CL-V correlation in the final model', ok)
    rn = runs[(runs['dataset'] == 'remifentanil') & (runs['condition'] == 'none')]
    ok = True
    for _, r in rn.iterrows():
        for p in (BENCH / 'runs' / r['dataset'] / r['condition'] / r['llm'] / r['rep'] / 'models').glob('*/spec.json'):
            c = covs(json.loads(p.read_text(encoding='utf-8')))
            ok &= not any(x[2] == 'linear' for x in c)
    check('remifentanil without knowledge: no linear forms tested', ok)
    check('remifentanil without knowledge: no GPT run added V3~AGE',
          numbers.get('remifentanil_none_V3_AGE_gpt', '').startswith(('not', 'neither', 'none')))
    check('with the statement, every run implemented every reference relationship (form family)',
          all(numbers.get(f'{ds}_knowledge_all_forms', '').startswith(('both', 'all', 'the run'))
              for ds in ('pheno', 'remifentanil')))
    check('without knowledge, every run omitted Apgar on V and age on V1',
          numbers.get('pheno_none_V_APGR', '').startswith(('none', 'neither', 'not'))
          and numbers.get('remifentanil_none_V1_AGE', '').startswith(('none', 'neither', 'not')))
    if numbers.get('strong_missed_desc'):
        ok = True
        for _, r in runs[(runs['condition'] == 'none')].iterrows():
            if r['dataset'] == 'remifentanil' and r['llm'] == 'claude' and int(r['fits']) <= 8:
                ok &= bool(re.search(r'time (budget|limit)', report_text(r), re.I))
        check('the run that omitted a strong effect cited the time budget', ok)
    pairs = []
    for key, t in tests.items():
        if key[0] != 'remifentanil':
            continue
        res = json.loads((BENCH / 'runs' / key[0] / key[1] / key[2] / key[3] / 'results.json').read_text(encoding='utf-8'))
        ofv = {m['model_id']: m for m in res['models'] if m.get('status') == 'converged'}
        specs = {}
        for mid in ofv:
            p = BENCH / 'runs' / key[0] / key[1] / key[2] / key[3] / 'models' / mid / 'spec.json'
            specs[mid] = json.loads(p.read_text(encoding='utf-8'))
        for a in specs:
            for b in specs:
                ca, cb = covs(specs[a]), covs(specs[b])
                if ca and len(ca) == len(cb) and {(x[0], x[1]) for x in ca} == {(x[0], x[1]) for x in cb} \
                        and {x[2] for x in ca} == {'linear'} and {x[2] for x in cb} == {'exponential'} \
                        and ofv[a]['n_estimated'] == ofv[b]['n_estimated'] and _rest(specs[a]) == _rest(specs[b]):
                    pairs.append(round(ofv[a]['ofv'] - ofv[b]['ofv'], 1))
    check('every direct linear vs exponential comparison favored exponential', pairs and all(x > 0 for x in pairs), pairs)
    check('GPT plotted the data and screened covariates in every run',
          numbers.get('gpt_plot_data', '').startswith('all') and numbers.get('gpt_screen', '').startswith('all'))
    check('runs near the time limit are GPT remifentanil runs only',
          set(re.findall(r'(GPT-6\.1 Sol|Claude Opus 5\.5) (\w+)', numbers.get('runs_near_time_limit_desc', ''))) <=
          {('GPT-6.1 Sol', 'remifentanil')})
    check('GPT fitted more models in every dataset and condition', numbers.get('gpt_more_fits_every_cell') == 'yes')
    v = numbers.get('rk_v3age_elderly', '0').replace('−', '-')
    lo, hi = (float(x) for x in v.split(' to ')) if ' to ' in v else (float(v), float(v))
    check('age effect on V3: about a quarter of the reference in older subjects', .18 <= lo and hi <= .32,
          numbers.get('rk_v3age_elderly'))
    mm = numbers.get('mm_vs_linear', '0').replace(',', '').split(' to ')[0]
    check('oral MM: MM vs linear about 800 or more', float(mm) >= 750, numbers.get('mm_vs_linear'))
    check('remifentanil V1~AGE is the weakest reference relationship (drop-one)',
          'evidence_remifentanil_V1~AGE' in numbers and float(numbers['evidence_remifentanil_V1~AGE']) < 3.84)
    import re as _re
    allr = pd.read_csv(EVAL / 'runs.csv')
    mis = allr[allr['condition'] == 'misleading']
    if len(mis):
        check('misleading: weight effects kept in every phenobarbital run',
              numbers.get('pm_weight_kept', '').startswith(('all', 'both')))
        check('misleading: saturable elimination kept in every oral MM run',
              numbers.get('om_mm_kept', '').startswith(('all', 'both')))
        check('misleading: two compartments adopted in at least one oral MM run',
              not numbers.get('om_2cmt_adopted', 'none').startswith(('none', 'neither', 'not')))
        check('misleading: every report flagged the conflict with the analyst',
              all(_re.search(r'analyst', report_text(r), re.I) and
                  _re.search(r'contradict|conflict|reconcil|not supported|disagree|reject', report_text(r), re.I)
                  for _, r in mis[mis['elimination'].notna()].iterrows()))
    gr = runs[(runs['dataset'] == 'remifentanil') & (runs['llm'] == 'gpt')]
    check('GPT remifentanil reports: backward checks not all repeated after the final covariance change',
          all(_re.search(r'repeat|complete backward', report_text(r), re.I) for _, r in gr.iterrows()))
    check('all runs near the time limit are the GPT remifentanil runs',
          numbers.get('runs_near_time_limit') == {6: 'six', 9: 'nine', 3: 'three'}.get(len(gr), str(len(gr))))
    check('Claude recalled the published datasets in every run', numbers.get('recall_claude', '').startswith('all'))
    check('GPT made no recall statement without knowledge',
          numbers.get('recall_gpt_none', '').startswith(('none', 'neither', 'not')))
    check('GPT recall statements name Minto', all('minto' in h['term'].lower() for k, v in json.loads(
        (HERE / 'build' / 'recall.json').read_text(encoding='utf-8')).items() if '/gpt/' in k for h in v))

    # remifentanil with the statement: the age effect on V1 where tested (weak support reported when below 6.63) and
    # the runs that never tested it (reports describe the data as supporting the statement)
    rk = runs[(runs['dataset'] == 'remifentanil') & (runs['condition'] == 'knowledge')]
    weak_ok, untested_ok, n_untested = True, True, 0
    for _, r in rk.iterrows():
        t = [x for x in tests[(r['dataset'], r['condition'], r['llm'], r['rep'])]['covariate_tests']
             if x['relationship'] == 'V1~AGE']
        txt = report_text(r)
        if not t:
            n_untested += 1
            untested_ok &= bool(re.search(r'data support(ed)? (age effects on all six|each part of the knowledge)', txt))
        elif max(t, key=lambda x: max(x['with_model'], x['without_model']))['delta_ofv'] < 6.63:
            weak_ok &= bool(re.search(r'(V1\W{0,2}age|AGE\W{0,2}V1|age\W{0,2}V1)[^.]{0,120}'
                                      r'(weak|borderline|includes 0|prior-supported|exception)|'
                                      r'(weak|borderline)[^.]{0,40}(V1\W{0,2}age|AGE\W{0,2}V1)', txt, re.I))
    check('remifentanil statement runs: weak V1-age support reported where tested and below 6.63', weak_ok)
    check('remifentanil statement runs that did not test V1-age describe the data as supporting the statement',
          untested_ok and numbers.get('rk_v1age_untested') == {0: 'no', 1: 'one', 2: 'two', 3: 'three'}.get(n_untested))
    weak_v3 = [r for _, r in rk.iterrows() if r['llm'] == 'gpt' and 'V3~AGE' in str(r.get('covariates_found', ''))]
    check('the weak V3-age effect kept with the statement is called analyst-supported in its report',
          numbers.get('rk_v3_weak') != 'yes' or any(re.search(r'analyst-supported', report_text(r), re.I) for r in weak_v3))
    # oral MM: the fourfold fall of apparent clearance with dose cited in the reports without knowledge
    on = runs[(runs['dataset'] == 'oral_mm') & (runs['condition'] == 'none')]
    check('oral MM runs without knowledge cite the fall of apparent clearance with dose',
          all(re.search(r'(CL/F|apparent clearance|clearance)[^.]{0,160}(fell|falls|decreas|declin|drop|lower)|'
                        r'(decreas|declin|fall)\w* (in )?apparent clearance|(fourfold|4-fold|4\.1)', report_text(r), re.I)
              for _, r in on.iterrows()))
    check('oral MM: dose effects tested only by GPT', numbers.get('oral_dose_all_gpt') == 'yes')
    check('oral MM: exponential residual error selected only by GPT', numbers.get('oral_mm_lognormal_llm') == 'GPT-6.1 Sol')
    # misleading statements
    if len(mis):
        check('misleading: unbounded Apgar-on-CL estimates opposite in every phenobarbital run',
              numbers.get('pm_apgar_cl_opposite', '').startswith(('all', 'both')))
        sec = numbers.get('om_2cmt_second', '99').replace('−', '-').split(' to ')
        check('misleading: the second compartment stayed below the inclusion threshold in every run',
              max(float(x) for x in sec) < 3.84, numbers.get('om_2cmt_second'))
        check('misleading: the stated second compartment was adopted in every oral MM run',
              numbers.get('om_2cmt_adopted', '').startswith(('all', 'both')))
    # tool use: GPT more often than Claude for each named behavior (36 main runs)
    prof = pd.DataFrame(json.loads((HERE / 'build' / 'run_profiles.json').read_text(encoding='utf-8')))
    prof['boot'] = prof['bootstrap'] != '–'
    cnt = {k: prof.groupby('llm')[k].sum().to_dict() for k in ('plot_data', 'fitted_after_plots', 'boot')}
    check('GPT plotted, refitted after plots, and bootstrapped more often than Claude',
          all(v['gpt'] > v['claude'] for v in cnt.values()), cnt)
    check('three runs per dataset, condition, and LLM', numbers.get('grid_complete') == 'yes')
    check('reference-fit VPCs exist', all(f'reffit_{ds}_vpc' in numbers for ds in ('pheno', 'remifentanil', 'oral_mm')))
    check('phenobarbital: a second compartment tested only by GPT',
          numbers.get('pheno_2cmt_claude', '').startswith(('none', 'neither')) and
          not numbers.get('pheno_2cmt_gpt', 'none').startswith(('none', 'neither')))
    # remifentanil V1-age without knowledge: one run's last test fell below 6.63 after an earlier test of 7.7
    flips = []
    for key, t in tests.items():
        if key[:2] != ('remifentanil', 'none'):
            continue
        v = [x for x in t['covariate_tests'] if x['relationship'] == 'V1~AGE']
        if v:
            last = max(v, key=lambda x: max(x['with_model'], x['without_model']))['delta_ofv']
            if last < 6.63 <= max(x['delta_ofv'] for x in v):
                flips.append(round(max(x['delta_ofv'] for x in v), 1))
    check('remifentanil without knowledge: one run had an earlier V1-age test of 7.7', flips == [7.7], flips)
    # remifentanil with the statement: the run without single V3 tests tested both V3 effects jointly (9.8 to 24.8)
    joint = []
    for _, r in rk.iterrows():
        key = (r['dataset'], r['condition'], r['llm'], r['rep'])
        if any(x['relationship'].startswith('V3~') for x in tests[key]['covariate_tests']):
            continue
        d = BENCH / 'runs' / key[0] / key[1] / key[2] / key[3]
        res = results(r)
        conv = {m['model_id']: m['ofv'] for m in res['models'] if m.get('status') == 'converged'}
        specs = {m: json.loads((d / 'models' / m / 'spec.json').read_text(encoding='utf-8')) for m in conv}
        for a in specs:
            for b_ in specs:
                ca, cb = set((x[0], x[1]) for x in covs(specs[a])), set((x[0], x[1]) for x in covs(specs[b_]))
                if cb - ca == {('V3', 'AGE'), ('V3', 'LBM')} and ca <= cb and _rest(specs[a]) == _rest(specs[b_]):
                    joint.append(round(conv[a] - conv[b_], 1))
    lo, hi = (float(x) for x in numbers['rk_v3_strong_support'].split(' to '))
    check('remifentanil statement: the V3 effects kept without single tests were tested jointly, within 9.8 to 24.8',
          numbers.get('rk_v3_untested') == 'one' and joint and all(lo <= x <= hi for x in joint), joint)
    check('oral MM: every run that fitted log-normal error selected it',
          numbers.get('oral_mm_lognormal_when_fitted') == 'yes')
    if len(mis):
        mis_tests = [round(c['delta_ofv'], 1) for t in json.loads((EVAL / 'agent_tests.json').read_text(encoding='utf-8'))
                     if t['dataset'] == 'oral_mm' and t['condition'] == 'misleading' for c in t['compartment_tests']
                     if c['compartments'] == [1, 2] and c.get('type') == 'michaelis_menten']
        check('misleading oral MM: the runs without the statement kept one compartment for the same OFV decrease',
              numbers.get('om_control_all_one') == 'yes' and numbers.get('om_control_dofv') in map(str, mis_tests),
              (numbers.get('om_control_dofv'), mis_tests))
    # the run whose final model PKAgent gave standard errors after finalization
    fin = []
    for _, r in runs.iterrows():
        d = BENCH / 'runs' / r['dataset'] / r['condition'] / r['llm'] / r['rep']
        final = results(r)['final_model']['model_id']
        calls = [json.loads(line) for line in (d / 'tool_log.jsonl').read_text(encoding='utf-8').splitlines()]
        fitted_without = any(c['tool'] == 'fit_models' and c['args'].get('standard_errors') is False
                             and final in [m.get('model_id') for m in (c.get('result') or {}).get('results', [])
                                           if isinstance(m, dict)] for c in calls)
        summ = json.loads((d / 'models' / final / 'summary.json').read_text(encoding='utf-8'))
        if fitted_without and summ.get('uncertainty_status') == 'computed':
            fin.append('/'.join((r['dataset'], r['condition'], r['llm'], r['rep'])))
    check('standard errors were added after finalization in exactly one run', len(fin) == 1, fin)
    return out


def main():
    out = checks()
    for name, ok, detail in out:
        print('ok  ' if ok else 'FAIL', name, '' if ok else detail)
    failed = [n for n, ok, _ in out if not ok]
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
