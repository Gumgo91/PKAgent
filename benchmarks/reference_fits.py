"""Fit the published reference model of each benchmark dataset with PKPy2 (through a PKAgent session, no LLM).

The fits give the OFV of the reference structure on the same data and engine as the agent's final models, and
check that the specification language can express each reference model.
Usage: python benchmarks/reference_fits.py [pheno remifentanil nimotuzumab]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
from pkagent.config import Budget, Settings          # noqa: E402
from pkagent.session import Session                   # noqa: E402

DATASETS = json.loads((HERE / 'datasets.json').read_text(encoding='utf-8'))

SPECS = {
    'pheno': dict(
        prepare=[('APGR_LT5', 'APGR < 5', 'Apgar score below 5')],
        spec=dict(name='reference: Grasela and Donn / NONMEM example',
                  structure=dict(compartments=1, absorption='iv'),
                  parameters=dict(CL=.0047, V=.98), iiv=dict(CL=.03, V=.03),
                  covariates=[dict(parameter='CL', covariate='WT', form='power', center=1, value=1, fixed=True),
                              dict(parameter='V', covariate='WT', form='power', center=1, value=1, fixed=True),
                              dict(parameter='V', covariate='APGR_LT5', form='categorical', level=1, value=.15)],
                  residual=dict(proportional=.115))),
    'remifentanil': dict(
        prepare=[],
        spec=dict(name='reference: Minto 1997',
                  structure=dict(compartments=3, absorption='iv'),
                  parameters=dict(CL=2.6, V1=5.1, Q2=2.05, V2=9.82, Q3=.076, V3=5.42),
                  iiv=dict(CL=.05, V1=.1, Q2=.1, V2=.1, Q3=.1, V3=.1),
                  covariates=[dict(parameter='V1', covariate='AGE', form='linear', center=40, value=-.0201 / 5.1),
                              dict(parameter='V1', covariate='LBM', form='linear', center=55, value=.072 / 5.1),
                              dict(parameter='V2', covariate='AGE', form='linear', center=40, value=-.0811 / 9.82),
                              dict(parameter='V2', covariate='LBM', form='linear', center=55, value=.108 / 9.82),
                              dict(parameter='CL', covariate='AGE', form='linear', center=40, value=-.0162 / 2.6),
                              dict(parameter='CL', covariate='LBM', form='linear', center=55, value=.0191 / 2.6),
                              dict(parameter='Q2', covariate='AGE', form='linear', center=40, value=-.0301 / 2.05),
                              dict(parameter='Q3', covariate='AGE', form='linear', center=40, value=-.00113 / .076)],
                  residual=dict(proportional=.15))),
    'oral_mm': dict(
        prepare=[],
        spec=dict(name='reference: simulation model (nlmixr2data Oral_1CPTMM)',
                  structure=dict(type='michaelis_menten', compartments=1, absorption='first_order'),
                  parameters=dict(V=70., Ka=1., VMAX=1000., KM=250.),
                  iiv=dict(V=.09, Ka=.09, VMAX=.09, KM=.09),
                  residual=dict(proportional=.2))),
    'oral_mm2': dict(
        prepare=[],
        spec=dict(name='reference: simulation model (nlmixr2data Oral_2CPTMM)',
                  structure=dict(type='michaelis_menten', compartments=2, absorption='first_order'),
                  parameters=dict(V1=70., Q=4., V2=50., Ka=1., VMAX=1000., KM=250.),
                  iiv=dict(V1=.09, Q=.09, V2=.09, Ka=.09, VMAX=.09, KM=.09),
                  residual=dict(proportional=.2))),
    'nimotuzumab': dict(
        prepare=[],
        spec=dict(name='reference: Rodriguez-Vera 2015 QSS-TMDD',
                  structure=dict(type='tmdd', tmdd_kind='qss', compartments=2, absorption='iv'),
                  parameters=dict(CL=.0015, V1=1.43, Q=.05, V2=18.5, KSS=6.96, KINT=.148, KDEG=5.5, R0=.26),
                  iiv=dict(CL=.2, V1=.1, KSS=.3),
                  residual=dict(CP=dict(lognormal=.2)))),
}


def main(names):
    for name in names:
        d = DATASETS.get(name) or dict(file=f'{name}.csv', description=name)
        out = HERE / 'reference_fits' / name
        s = Session(HERE / 'data' / d['file'], out, Settings(workers=1, threads_per_worker=4, fit_wall_seconds=3600., laplace_starts=2,
                                                               budget=Budget(max_fits=10)), d['description'])
        try:
            for col, expr, why in SPECS[name]['prepare']:
                s.dataset.add_column(col, expr, why)
                s.data_version += 1
                s._write_data()
            rec = s.fit([SPECS[name]['spec']])[0]
            summary = s.compact(rec, detail=True)
            (out / 'reference_fit.json').write_text(json.dumps(summary, indent=1, default=str), encoding='utf-8')
            print(name, rec.get('status'), rec.get('ofv'), json.dumps(summary.get('parameters'))[:600])
        finally:
            s.close()


if __name__ == '__main__':
    main(sys.argv[1:] or list(SPECS))
