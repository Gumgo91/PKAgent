"""Scripted session without an LLM: every tool on the phenobarbital data, then the report."""
import json
import sys
import time
from pathlib import Path

from pkagent.config import Budget, Settings
from pkagent.report import write_report
from pkagent.session import Session
from pkagent.tools import execute

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else 'output/smoke_session')


class FakeLLM:
    usage = dict(calls=0, prompt_tokens=0, completion_tokens=0, cached_tokens=0, reasoning_tokens=0, cost_usd=0.)


def call(session, name, args=None):
    t0 = time.time()
    out, images = execute(session, name, args or {})
    print(f'== {name} ({time.time() - t0:.1f} s) images={[p.name for p in images]}')
    print(json.dumps(out, default=str)[:1500])
    return out


def main():
    settings = Settings(workers=2, threads_per_worker=4, budget=Budget(max_fits=20))
    s = Session('benchmarks/data/pheno_sd.csv', OUT, settings,
                description='Phenobarbital in 59 neonates; AMT in mg (IV bolus doses), DV serum concentration in mg/L, '
                            'TIME in h, WT body weight in kg, APGR 5-minute Apgar score.')
    try:
        call(s, 'describe_data')
        call(s, 'run_nca')
        call(s, 'plot_data', dict(log_scale=True))
        call(s, 'add_data_column', dict(name='APGR_LT5', expression='APGR < 5', reason='low Apgar indicator'))
        base = dict(name='base 1cmt', structure=dict(compartments=1, absorption='iv'),
                    parameters=dict(CL=.006, V=1.3), iiv=dict(CL=.1, V=.1), residual=dict(proportional=.15))
        wt = json.loads(json.dumps(base))
        wt.update(name='WT linear on CL and V', covariates=[
            dict(parameter='CL', covariate='WT', form='power', center=1, value=1, fixed=True),
            dict(parameter='V', covariate='WT', form='power', center=1, value=1, fixed=True)])
        r = call(s, 'fit_models', dict(models=[base, wt], standard_errors=False))
        ids = [x['model_id'] for x in r['results']]
        call(s, 'compare_models', dict(model_ids=ids, reference_model_id=ids[0]))
        call(s, 'screen_covariates', dict(model_id=ids[1]))
        sc = call(s, 'covariate_search', dict(base_model_id=ids[1], candidates=[
            dict(parameter='V', covariate='APGR_LT5', form='categorical', level=1, value=.1),
            dict(parameter='CL', covariate='APGR', form='power', center=7, value=.01)]))
        final = sc['final_model_id']
        call(s, 'run_vpc', dict(model_id=final))
        call(s, 'view_plots', dict(model_id=final, plots=['gof', 'vpc']))
        call(s, 'finalize_model', dict(model_id=final, report=dict(summary='scripted test', development='scripted',
                                                                   evaluation='scripted', limitations='none')))
        print('report', write_report(s, FakeLLM(), settings))
    finally:
        s.close()


if __name__ == '__main__':
    main()
