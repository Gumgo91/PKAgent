"""Evidence for each covariate relationship of the reference models in the benchmark data.

For pheno and remifentanil, every covariate effect of the reference model is removed in turn and the model is
refitted with PKPy2 (same Laplace settings as the agent's fits, starting at the reference-fit estimates, no standard
errors). The OFV increase over the reference fit is the likelihood-ratio evidence for that effect (3.84 and 6.63 are
the inclusion and retention thresholds of the agent's instructions).
Writes evaluation/effect_evidence.json and .csv.
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


def main(names):
    out = {}
    for name in names:
        ref = json.loads((HERE / 'reference_fits' / name / 'reference_fit.json').read_text(encoding='utf-8'))
        fitted = json.loads((HERE / 'reference_fits' / name / 'models' / 'M001' / 'summary.json').read_text(encoding='utf-8'))
        base = json.loads(json.dumps(SPECS[name]['spec']))
        # start at the reference-fit estimates
        for p in fitted['parameters']:
            base['parameters'][p['parameter']] = p['estimate']
        for r in fitted['iiv']:
            base['iiv'][r['parameter']] = r['variance']
        coef = {c['effect']: c['coefficient'] for c in fitted.get('covariate_effects', [])}
        for c in base.get('covariates', []):
            label = f"{c['parameter']}~{c['covariate']}" + (f"={float(c['level']):g}" if c.get('level') is not None else '')
            if not c.get('fixed') and label in coef:
                c['value'] = coef[label]
        specs = []
        for k, c in enumerate(base.get('covariates', [])):
            s = json.loads(json.dumps(base))
            del s['covariates'][k]
            s['name'] = f"reference without {c['parameter']}~{c['covariate']}"
            specs.append((f"{c['parameter']}~{c['covariate']}", s))
        session = Session(HERE / 'data' / DATASETS[name]['file'], HERE / 'evaluation' / 'effect_evidence' / name,
                          Settings(workers=min(len(specs), 4), threads_per_worker=2, laplace_starts=1,
                                   fit_wall_seconds=3600., budget=Budget(max_fits=len(specs) + 2)),
                          DATASETS[name]['description'])
        try:
            for col, expr, why in SPECS[name]['prepare']:
                session.dataset.add_column(col, expr, why)
                session.data_version += 1
                session._write_data()
            recs = session.fit([s for _, s in specs], uncertainty=False)
            rows = []
            for (label, _), r in zip(specs, recs):
                rows.append(dict(effect=label, status=r.get('status'), ofv=r.get('ofv'),
                                 delta_ofv=None if r.get('ofv') is None else round(r['ofv'] - ref['ofv'], 3)))
                print(name, rows[-1], flush=True)
            out[name] = dict(reference_ofv=ref['ofv'], removals=rows)
        finally:
            session.close()
    path = HERE / 'evaluation' / 'effect_evidence.json'
    old = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    old.update(out)
    path.write_text(json.dumps(old, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1:] or ['pheno', 'remifentanil'])
