"""Deterministic baseline without a language model: stepwise covariate modeling (SCM) on the reference structure.

For each published dataset, the reference structural and stochastic model without covariates is fitted, and the
session's covariate_search (forward inclusion at P < 0.05, backward elimination at P < 0.01, the thresholds of the
agent's instructions) is run over the candidate relationships of the reference model in their reference functional
form, plus the alternatives an analyst would screen (Apgar on CL; LBM on V3, Q2 and Q3; age on V3). This shows which
reference relationships the stated criteria alone retain, independently of any LLM.
Writes evaluation/scm_baseline.json. Usage: python benchmarks/scm_baseline.py [pheno remifentanil]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
from pkagent.config import Budget, Settings          # noqa: E402
from pkagent.session import Session                   # noqa: E402

sys.path.insert(0, str(HERE))
from reference_fits import SPECS                      # noqa: E402

DATASETS = json.loads((HERE / 'datasets.json').read_text(encoding='utf-8'))

CANDIDATES = {
    'pheno': [dict(parameter='CL', covariate='WT', form='power', center=1.3, value=.75),
              dict(parameter='V', covariate='WT', form='power', center=1.3, value=1.),
              dict(parameter='V', covariate='APGR_LT5', form='categorical', level=1, value=.1),
              dict(parameter='CL', covariate='APGR_LT5', form='categorical', level=1, value=-.1)],
    'remifentanil': [dict(parameter=p, covariate=c, form='linear', center=dict(AGE=40, LBM=55)[c],
                          value=dict(AGE=-.005, LBM=.01)[c])
                     for p in ('CL', 'V1', 'V2', 'V3', 'Q2', 'Q3') for c in ('AGE', 'LBM')],
}


def base_spec(name):
    spec = json.loads(json.dumps(SPECS[name]['spec']))
    spec['covariates'] = []
    spec['name'] = f'{name} reference structure without covariates'
    return spec


def main(names):
    path = HERE / 'evaluation' / 'scm_baseline.json'
    out = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    for name in names:
        session = Session(HERE / 'data' / DATASETS[name]['file'], HERE / 'evaluation' / 'scm_baseline' / name,
                          Settings(workers=3, threads_per_worker=2, laplace_starts=1, fit_wall_seconds=1800.,
                                   budget=Budget(max_fits=200, max_hours=24.)), DATASETS[name]['description'])
        try:
            for col, expr, why in SPECS[name]['prepare']:
                session.dataset.add_column(col, expr, why)
                session.data_version += 1
                session._write_data()
            base = session.fit([base_spec(name)], uncertainty=False)[0]
            print(name, 'base', base.get('status'), base.get('ofv'), flush=True)
            result = session.covariate_search(base['model_id'], CANDIDATES[name])
            final = session.models[result['final_model_id']]
            out[name] = dict(base_ofv=base.get('ofv'), final_ofv=final.get('ofv'), included=result['included'],
                             history=result['history'], fits=len(session.models))
            print(name, json.dumps(out[name]['included']), final.get('ofv'), flush=True)
            path.write_text(json.dumps(out, indent=1, default=str), encoding='utf-8')
        finally:
            session.close()


if __name__ == '__main__':
    main(sys.argv[1:] or ['pheno', 'remifentanil'])
