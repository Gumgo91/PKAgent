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
    return out


def main():
    out = checks()
    for name, ok, detail in out:
        print('ok  ' if ok else 'FAIL', name, '' if ok else detail)
    failed = [n for n, ok, _ in out if not ok]
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
