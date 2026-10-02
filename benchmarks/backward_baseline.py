"""Deterministic baseline for remifentanil: backward elimination from the reference covariate model.

Step 1 is the drop-one evidence of effect_evidence.py (every reference relationship removed in turn from the PKPy2 fit
of the reference model). The relationship with the smallest OFV increase is removed if that increase is below the
retention threshold of the agent's instructions (6.63, P < 0.01); then every remaining relationship is removed in turn
from the reduced model (refitted from its estimates), and so on until all remaining relationships pass.
Writes evaluation/backward_baseline.json. Usage: python benchmarks/backward_baseline.py [remifentanil]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
from pkagent.config import Budget, Settings          # noqa: E402
from pkagent.session import Session, _estimates_as_start, _iiv_start, _raw_spec  # noqa: E402

sys.path.insert(0, str(HERE))
from reference_fits import SPECS                      # noqa: E402

DATASETS = json.loads((HERE / 'datasets.json').read_text(encoding='utf-8'))
RETAIN = 6.63


def label(c):
    return f"{c['parameter']}~{c['covariate']}"


def with_estimates(spec, summary):
    """Agent-level specification of a fitted model with its estimates as starting values (next backward step)."""
    rec = dict(spec=spec, summary=summary)
    raw = _raw_spec(spec)
    raw['parameters'] = _estimates_as_start(rec)
    raw['iiv'] = _iiv_start(rec)
    coef = {c['effect']: c['coefficient'] for c in summary.get('covariate_effects', [])}
    for c in raw['covariates']:
        if not c.get('fixed') and label(c) in coef:
            c['value'] = coef[label(c)]
    return raw


def main(name='remifentanil'):
    ev = json.loads((HERE / 'evaluation' / 'effect_evidence.json').read_text(encoding='utf-8'))[name]
    removals = sorted(ev['removals'], key=lambda r: r['delta_ofv'])
    history = [dict(step=1, tests={r['effect']: r['delta_ofv'] for r in removals})]
    weakest = removals[0]
    out = dict(start='PKPy2 fit of the reference model', reference_ofv=ev['reference_ofv'], history=history)
    if weakest['delta_ofv'] >= RETAIN:
        out['removed'] = []
        out['retained'] = [r['effect'] for r in removals]
        (HERE / 'evaluation' / 'backward_baseline.json').write_text(json.dumps({name: out}, indent=1), encoding='utf-8')
        return
    # the reduced model: the drop-one fit without the weakest relationship
    base_dir = HERE / 'evaluation' / 'effect_evidence' / name / 'models'
    reduced = None
    for d in sorted(base_dir.iterdir()):
        spec = json.loads((d / 'spec.json').read_text(encoding='utf-8'))
        if not any(label(c) == weakest['effect'] for c in spec['covariates']):
            reduced = (spec, json.loads((d / 'summary.json').read_text(encoding='utf-8')))
            break
    removed = [weakest['effect']]
    spec, summary = reduced
    step = 2
    session = Session(HERE / 'data' / DATASETS[name]['file'], HERE / 'evaluation' / 'backward_baseline' / name,
                      Settings(workers=2, threads_per_worker=2, laplace_starts=1, fit_wall_seconds=3600.,
                               budget=Budget(max_fits=60, max_hours=24.)), DATASETS[name]['description'])
    try:
        for col, expr, why in SPECS[name]['prepare']:
            session.dataset.add_column(col, expr, why)
            session.data_version += 1
            session._write_data()
        while True:
            start = with_estimates(spec, summary)
            specs = []
            for k, c in enumerate(start['covariates']):
                s = json.loads(json.dumps(start))
                del s['covariates'][k]
                s['name'] = f"backward step {step}: without {label(c)}"
                specs.append((label(c), s))
            cur_ofv = summary['ofv']
            recs = session.fit([s for _, s in specs], uncertainty=False)
            tests = {lab: (None if r.get('ofv') is None else round(r['ofv'] - cur_ofv, 3)) for (lab, _), r in
                     zip(specs, recs)}
            history.append(dict(step=step, model_ofv=cur_ofv, tests=tests,
                                status={lab: r.get('status') for (lab, _), r in zip(specs, recs)}))
            print(step, tests, flush=True)
            valid = {k: v for k, v in tests.items() if v is not None}
            lab_min = min(valid, key=valid.get)
            if valid[lab_min] >= RETAIN:
                break
            removed.append(lab_min)
            i = [lab for lab, _ in specs].index(lab_min)
            mid = recs[i]['model_id']
            mdir = HERE / 'evaluation' / 'backward_baseline' / name / 'models' / mid
            spec = json.loads((mdir / 'spec.json').read_text(encoding='utf-8'))
            summary = json.loads((mdir / 'summary.json').read_text(encoding='utf-8'))
            step += 1
        out.update(removed=removed, retained=[label(c) for c in spec['covariates']])
    finally:
        session.close()
    (HERE / 'evaluation' / 'backward_baseline.json').write_text(json.dumps({name: out}, indent=1), encoding='utf-8')
    print('removed', removed, flush=True)


if __name__ == '__main__':
    main(*(sys.argv[1:] or ['remifentanil']))
