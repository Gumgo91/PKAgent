"""Type of the residual error of the simulated oral MM data (nlmixr2data Oral_1CPTMM; the documentation gives only
"residual error at 20%").

For the first 80 subjects of the source file, the individual predictions are computed from each subject's true
parameters (KA, V, VM, KM columns) over its dosing history; exponential error gives symmetric log(DV/IPRED) and a
right-skewed DV/IPRED - 1, proportional error the reverse.
Usage: python benchmarks/residual_check.py     Writes benchmarks/evaluation/oral_mm_residual_check.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import skew

HERE = Path(__file__).resolve().parent


def main(n_subjects=80):
    d = pd.read_csv(HERE / 'data' / 'Oral_1CPTMM.csv')
    log_res, ratio_res = [], []
    for i, (_, g) in enumerate(d.groupby('ID', sort=False)):
        if i >= n_subjects:
            break
        p = g.iloc[0]
        V, VM, KM, KA = float(p['V']), float(p['VM']), float(p['KM']), float(p['KA'])
        doses = g[g['EVID'] == 1][['TIME', 'AMT']].to_numpy()
        obs = g[(g['EVID'] == 0) & (g['MDV'] == 0)]

        def rhs(t, y):
            c = y[1] / V
            return [-KA * y[0], KA * y[0] - VM * c / (KM + c)]
        y, t0, pred = np.zeros(2), 0., {}
        for t in sorted(set(doses[:, 0]) | set(obs['TIME'])):
            if t > t0:
                y = solve_ivp(rhs, (t0, t), y, rtol=1e-9, atol=1e-9, method='LSODA').y[:, -1]
                t0 = t
            y = y + np.array([doses[doses[:, 0] == t, 1].sum(), 0.])
            pred[t] = y[1] / V
        ipred = np.array([pred[t] for t in obs['TIME']])
        dv = obs['DV'].to_numpy(dtype=float)
        ok = (dv > 0) & (ipred > 0)
        log_res += list(np.log(dv[ok] / ipred[ok]))
        ratio_res += list(dv[ok] / ipred[ok] - 1)
    log_res, ratio_res = np.array(log_res), np.array(ratio_res)
    out = dict(subjects=n_subjects, observations=int(len(log_res)), log_sd=float(log_res.std()),
               log_skewness=float(skew(log_res)), ratio_sd=float(ratio_res.std()), ratio_skewness=float(skew(ratio_res)),
               expected_ratio_skewness_exponential=float((np.exp(.04) + 2) * np.sqrt(np.exp(.04) - 1)))
    (HERE / 'evaluation' / 'oral_mm_residual_check.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
