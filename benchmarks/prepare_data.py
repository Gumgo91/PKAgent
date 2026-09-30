"""NONMEM-format benchmark files from the R package data (exported as CSV by export_data.R).

remifentanil.csv  from nlme::Remifentanil (Minto et al. 1997): each run of records with the same positive
                  infusion rate becomes one dose record (TIME = start, RATE = rate in ug/min, AMT = rate x
                  duration in ug); observations keep conc (ng/mL = ug/L). Checks that the reconstructed amount
                  equals the sum of the Amt column (amount given in the interval that starts at each record).
pheno.csv         from nlmixr2data::pheno_sd (Grasela and Donn 1985), unchanged apart from column order.
oral_mm.csv       from nlmixr2data::Oral_1CPTMM (simulated, ACOP 2016; one-compartment model with first-order
                  absorption and Michaelis-Menten elimination, 30% IIV on every parameter, 20% residual error):
                  the first 10 subjects of each dose level (10, 20, 40, 80 mg; AMT in ug, DV in ug/L) with 25 of
                  the 58 samples (single-dose profile, troughs, profile after the last of 7 daily doses).
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent / 'data'


def remifentanil():
    d = pd.read_csv(HERE / 'remifentanil_nlme.csv', na_values=['.'])
    rows = []
    for sid, g in d.groupby('ID', sort=True):
        g = g.sort_values('Time').reset_index(drop=True)
        cov = g.iloc[0][['Age', 'Sex', 'Ht', 'Wt', 'BSA', 'LBM']].to_dict()
        t, rate = g['Time'].to_numpy(float), g['Rate'].to_numpy(float)
        amt_col = g['Amt'].to_numpy(float)
        given = 0.
        i = 0
        while i < len(g):
            if rate[i] > 0:
                j = i
                while j + 1 < len(g) and rate[j + 1] == rate[i]:
                    j += 1
                end = t[j + 1] if j + 1 < len(g) else t[j]
                amount = rate[i] * (end - t[i])
                given += amount
                rows.append(dict(ID=int(sid), TIME=t[i], AMT=round(amount, 6), RATE=rate[i], DV=np.nan, EVID=1, MDV=1, **cov))
                i = j + 1
            else:
                i += 1
        expected = float(np.nansum(amt_col))       # source Amt: amount given in the interval that starts at the record
        for _, r in g[g['conc'].notna()].iterrows():
            rows.append(dict(ID=int(sid), TIME=r['Time'], AMT=0., RATE=0., DV=r['conc'], EVID=0, MDV=0, **cov))
        if expected > 0 and abs(given - expected) / expected > .02:
            print(f'subject {sid}: reconstructed {given:.1f} ug vs Amt sum {expected:.1f} ug')
    out = pd.DataFrame(rows).sort_values(['ID', 'TIME', 'EVID'], ascending=[True, True, False])
    out = out[['ID', 'TIME', 'AMT', 'RATE', 'DV', 'EVID', 'MDV', 'AGE' if 'AGE' in out else 'Age', 'Sex', 'Ht', 'Wt', 'BSA',
               'LBM']].rename(columns={'Age': 'AGE', 'Sex': 'SEX', 'Ht': 'HT', 'Wt': 'WT'})
    out.to_csv(HERE / 'remifentanil.csv', index=False, na_rep='.')
    print('remifentanil', out['ID'].nunique(), 'subjects', int((out.EVID == 0).sum()), 'observations',
          int((out.EVID == 1).sum()), 'infusion records')


def pheno():
    d = pd.read_csv(HERE / 'pheno_sd.csv', na_values=['.'])
    d.loc[d['MDV'] == 1, 'DV'] = np.nan
    d[['ID', 'TIME', 'AMT', 'DV', 'EVID', 'MDV', 'WT', 'APGR']].to_csv(HERE / 'pheno.csv', index=False, na_rep='.')
    print('pheno', d['ID'].nunique(), 'subjects', int((d['EVID'] == 0).sum()), 'observations')


KEEP_TIMES = (0.5, 1, 2, 4, 8, 12, 24, 48, 71.99, 95.99, 119.99, 143.99, 167.99, 191.99, 215.99,
              216.5, 217, 218, 220, 224, 228, 240, 252, 264, 288)


def oral_mm(source='Oral_1CPTMM', out='oral_mm.csv', per_dose=10):
    d = pd.read_csv(HERE / f'{source}.csv', na_values=['.'])
    first = d.drop_duplicates('ID')
    ids = [i for _, g in first.sort_values('ID').groupby('DOSE') for i in g['ID'].head(per_dose)]
    d = d[d['ID'].isin(ids)].copy()
    obs = d['EVID'] == 0
    keep = ~obs | d['TIME'].round(2).isin([round(t, 2) for t in KEEP_TIMES])
    d = d[keep].copy()
    d.loc[d['EVID'] == 1, 'DV'] = np.nan
    d['MDV'] = (d['EVID'] == 1).astype(int)
    d['DOSE'] = d['DOSE'] / 1000.
    truth = d.drop_duplicates('ID')[[c for c in ('KA', 'V', 'V1', 'V2', 'Q', 'VM', 'KM') if c in d]]
    d = d[['ID', 'TIME', 'AMT', 'DV', 'EVID', 'MDV', 'CMT', 'DOSE']]
    d.to_csv(HERE / out, index=False, na_rep='.')
    geo = {c: round(float(np.exp(np.log(truth[c]).mean())), 4) for c in truth}
    print(out, d['ID'].nunique(), 'subjects', int((d['EVID'] == 0).sum()), 'observations;',
          'geometric means of the simulated individual parameters', geo)


if __name__ == '__main__':
    remifentanil()
    pheno()
    oral_mm()
