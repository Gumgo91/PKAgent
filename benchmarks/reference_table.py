"""Table 3: published (or simulated) reference estimates against the PKPy2 fit of the same model.

Reads datasets.json and reference_fits/<dataset>/reference_fit.json; writes evaluation/reference_table.md and .csv.
Linear covariate slopes of the remifentanil model are compared as absolute slopes (published units, per year or
per kg): PKPy2 estimates the relative slope b of 1 + b (z - center), so the absolute slope is b times the typical value.
"""
import json
import math
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATASETS = json.loads((HERE / 'datasets.json').read_text(encoding='utf-8'))


def rows_pheno(ref, fit):
    est = {r['parameter']: r for r in fit['parameters']}
    iiv = {r['parameter']: r for r in fit['iiv']}
    cov = {r['effect']: r for r in fit.get('covariate_effects', [])}
    r = ref['estimates']
    out = [('CL (L/h/kg)', r['CL_per_kg'], est['CL']['estimate'], est['CL'].get('rse_percent')),
           ('V (L/kg)', r['V_per_kg'], est['V']['estimate'], est['V'].get('rse_percent')),
           ('V increase, Apgar < 5 (fraction)', r['APGR_LT5_fractional_increase_in_V'],
            math.exp(cov['V~APGR_LT5=1']['coefficient']) - 1, None),
           ('omega^2 CL', r['omega2_CL'], iiv['CL']['variance'], iiv['CL'].get('rse_percent')),
           ('omega^2 V', r['omega2_V'], iiv['V']['variance'], iiv['V'].get('rse_percent')),
           ('proportional residual SD', math.sqrt(r['sigma2_proportional']), fit['residual'][0]['sd'],
            fit['residual'][0].get('rse_percent'))]
    return out


def rows_remifentanil(ref, fit):
    est = {r['parameter']: r for r in fit['parameters']}
    cov = {r['effect']: r['coefficient'] for r in fit.get('covariate_effects', [])}
    r = ref['estimates']
    slopes = ref['linear_slopes_absolute']
    out = [(f'{p} ({"L/min" if p.startswith(("CL", "Q")) else "L"})', r[p], est[p]['estimate'], est[p].get('rse_percent'))
           for p in ('V1', 'V2', 'V3', 'CL', 'Q2', 'Q3')]
    for key, value in slopes.items():
        p, c = key.split('~')
        b = cov.get(f'{p}~{c}')
        out.append((f'{p} slope per unit {c}', value, None if b is None else b * est[p]['estimate'], None))
    return out


def rows_oral_mm(ref, fit):
    est = {r['parameter']: r for r in fit['parameters']}
    return [(f"{p} ({ref['units'][p]})", v, est[p]['estimate'], est[p].get('rse_percent'))
            for p, v in ref['estimates'].items()]


def main():
    out = []
    for name, fn in (('pheno', rows_pheno), ('remifentanil', rows_remifentanil), ('oral_mm', rows_oral_mm)):
        path = HERE / 'reference_fits' / name / 'reference_fit.json'
        if not path.exists():
            continue
        fit = json.loads(path.read_text(encoding='utf-8'))
        for label, published, estimate, rse in fn(DATASETS[name]['reference'], fit):
            out.append(dict(dataset=name, quantity=label, reference=published, pkpy2=estimate,
                            ratio=estimate / published if estimate is not None and published else None, rse_percent=rse))
        out.append(dict(dataset=name, quantity='OFV (Laplace, with constants)', reference=None, pkpy2=fit['ofv'],
                        ratio=None, rse_percent=None))
    table = pd.DataFrame(out)
    (HERE / 'evaluation').mkdir(exist_ok=True)
    table.to_csv(HERE / 'evaluation' / 'reference_table.csv', index=False)
    (HERE / 'evaluation' / 'reference_table.md').write_text(table.to_markdown(index=False, floatfmt='.4g'),
                                                           encoding='utf-8')
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
