"""Non-compartmental summaries and starting values for the first model.

Per subject, the first dosing interval with at least three observations gives
Cmax, Tmax, AUC (linear-up/log-down), the terminal slope (best adjusted R² over
the last 3-6 points after Tmax), CL = dose/AUCinf and Vz = CL/lambda_z. The pooled,
dose-normalized profile after Tmax is compared with one and two exponentials to
flag multiphasic decline. When the data are too sparse for per-subject NCA, a
naive pooled one-compartment fit (no random effects) supplies starting values.
"""
import math

import numpy as np
from scipy.optimize import brentq, curve_fit


def _auc(t, c):
    auc = 0.
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if c[i] < c[i - 1] and c[i] > 0 and c[i - 1] > 0:
            auc += dt * (c[i - 1] - c[i]) / math.log(c[i - 1] / c[i])
        else:
            auc += dt * (c[i] + c[i - 1]) / 2
    return auc


def _lambda_z(t, c, i_max):
    best = None
    tail_t, tail_c = t[i_max + 1:], c[i_max + 1:]
    keep = tail_c > 0
    tail_t, tail_c = tail_t[keep], tail_c[keep]
    for k in range(3, min(6, len(tail_t)) + 1):
        x, y = tail_t[-k:], np.log(tail_c[-k:])
        slope, intercept = np.polyfit(x, y, 1)
        if slope >= 0:
            continue
        pred = intercept + slope * x
        ss_res, ss_tot = np.sum((y - pred) ** 2), np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.
        adj = 1 - (1 - r2) * (k - 1) / (k - 2)
        if best is None or adj > best['adj_r2'] + 1e-4:
            best = dict(lambda_z=-slope, adj_r2=adj, points=k)
    return best


def subject_nca(t, c, dose, extravascular):
    order = np.argsort(t)
    t, c = t[order], c[order]
    if len(t) < 3:
        return None
    i_max = int(np.argmax(c))
    row = dict(n=int(len(t)), dose=float(dose), cmax=float(c[i_max]), tmax=float(t[i_max]),
               first_sample_is_peak=bool(i_max == 0))
    tt, cc = (np.r_[0., t], np.r_[0., c]) if extravascular and t[0] > 0 else (t, c)
    if not extravascular and t[0] > 0 and len(t) >= 2 and c[1] < c[0] and c[1] > 0:
        # back-extrapolate C0 from the first two points (IV bolus)
        k = math.log(c[0] / c[1]) / (t[1] - t[0])
        tt, cc = np.r_[0., t], np.r_[c[0] * math.exp(k * t[0]), c]
    row['auc_last'] = float(_auc(tt, cc))
    lz = _lambda_z(t, c, i_max)
    if lz:
        row.update(lambda_z=lz['lambda_z'], half_life=math.log(2) / lz['lambda_z'], lambda_z_points=lz['points'],
                   lambda_z_adj_r2=lz['adj_r2'])
        auc_inf = row['auc_last'] + c[-1] / lz['lambda_z']
        row['auc_inf'] = float(auc_inf)
        row['extrapolated_pct'] = float(100 * (1 - row['auc_last'] / auc_inf))
        if dose > 0 and auc_inf > 0:
            row['cl'] = float(dose / auc_inf)
            row['vz'] = float(row['cl'] / lz['lambda_z'])
    return row


def _ka_from_tmax(tmax, k):
    """First-order absorption rate for which a one-compartment oral model peaks at tmax."""
    if tmax <= 0 or k <= 0:
        return None
    f = lambda ka: math.log(ka / k) / (ka - k) - tmax
    try:
        hi = k * 1e4
        if f(k * 1.0001) * f(hi) > 0:
            return None
        return float(brentq(f, k * 1.0001, hi))
    except (ValueError, ZeroDivisionError):
        return None


def _phases(t, y):
    """Compare one and two exponentials for the pooled dose-normalized decline (log scale, BIC)."""
    keep = y > 0
    t, y = t[keep], y[keep]
    if len(t) < 8:
        return None
    ly = np.log(y)
    b1 = np.polyfit(t, ly, 1)
    rss1 = np.sum((ly - np.polyval(b1, t)) ** 2)

    def two(x, a, la, b, lb):
        return np.log(np.exp(a - la * x) + np.exp(b - lb * x))
    try:
        import warnings
        p0 = [ly[0], max(-b1[0] * 4, 1e-3), ly[-1] + (-b1[0]) * t[-1], max(-b1[0], 1e-4)]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p, _ = curve_fit(two, t, ly, p0=p0, maxfev=20000)
        rss2 = np.sum((ly - two(t, *p)) ** 2)
    except (RuntimeError, ValueError):
        return dict(one_phase_slope=float(-b1[0]), two_phase_fit='failed')
    n = len(t)
    bic1 = n * math.log(rss1 / n) + 2 * math.log(n)
    bic2 = n * math.log(rss2 / n) + 4 * math.log(n)
    early, late = sorted([abs(p[1]), abs(p[3])], reverse=True)
    return dict(delta_bic_two_minus_one=float(bic2 - bic1), early_slope=float(early), terminal_slope=float(late),
                slope_ratio=float(early / late) if late > 0 else None)


def _pooled_shape(t, y):
    """Shape of the pooled dose-normalized profile over time-after-dose bins."""
    if len(t) < 6:
        return 'unknown'
    edges = np.unique(np.quantile(t, np.linspace(0, 1, 7)))
    if len(edges) < 3:
        return 'unknown'
    idx = np.clip(np.searchsorted(edges, t, side='right') - 1, 0, len(edges) - 2)
    med = np.array([np.median(y[idx == b]) if np.any(idx == b) else np.nan for b in range(len(edges) - 1)])
    first, peak = med[0], np.nanmax(med)
    if np.nanargmax(med) > 0 and first < .8 * peak:
        return 'rises to a peak'
    return 'declines from the first sample'


def run_nca(dataset, output=1):
    df = dataset.df
    dose_m, obs_m = dataset.dose_mask(), dataset.obs_mask()
    if 'DVID' in df:
        obs_m = obs_m & (df['DVID'].fillna(1).to_numpy() == output)
    rows, pooled_t, pooled_y, skipped = [], [], [], 0
    doses = df[dose_m]
    extravascular_votes = []
    first_rates = []
    for sid, g in df.groupby('ID', sort=False):
        gd = doses[doses['ID'] == sid].sort_values('TIME')
        go = g[obs_m[g.index]] if len(g) else g
        if not len(gd) or not len(go):
            skipped += 1
            continue
        t0, amount = float(gd['TIME'].iloc[0]), float(gd['AMT'].iloc[0])
        t_next = float(gd['TIME'].iloc[1]) if len(gd) > 1 else math.inf
        rate = float(gd['RATE'].iloc[0]) if 'RATE' in gd and not math.isnan(gd['RATE'].iloc[0]) else 0.
        first_rates.append((rate, amount))
        interval = go[(go['TIME'] > t0) & (go['TIME'] < t_next)]
        t = interval['TIME'].to_numpy(float) - t0
        c = interval['DV'].to_numpy(float)
        if len(t) >= 2:
            i_max = int(np.argmax(c))
            extravascular_votes.append(i_max > 0 or rate > 0)
        pooled_t.extend(t.tolist())
        pooled_y.extend((c / amount).tolist() if amount > 0 else [])
        r = subject_nca(t, c, amount, extravascular=True if len(t) < 2 else (int(np.argmax(c)) > 0 or rate > 0))
        if r is None:
            skipped += 1
            continue
        r['ID'] = sid if not isinstance(sid, np.generic) else sid.item()
        rows.append(r)
    t, y = np.asarray(pooled_t), np.asarray(pooled_y)
    if len(extravascular_votes) < 5:
        shape = 'undetermined (fewer than five subjects with two or more samples in the first dosing interval)'
    elif len(rows) >= 5:           # per-subject profiles: the majority of subjects decides
        shape = 'rises to a peak' if sum(extravascular_votes) > len(extravascular_votes) / 2 else \
            'declines from the first sample'
    else:
        shape = _pooled_shape(t, y)
    infusions = [(r, a) for r, a in first_rates if r > 0]
    out = dict(output=output, subjects_with_nca=len(rows), subjects_insufficient=skipped,
               first_doses=dict(infusions=len(infusions), bolus_or_extravascular=len(first_rates) - len(infusions),
                                note='times, rates and durations are in the units of the TIME column'),
               profile=shape,
               subjects_peaking_after_first_sample=int(sum(extravascular_votes)),
               subjects_with_two_or_more_samples=len(extravascular_votes))
    if infusions:
        out['first_doses']['median_infusion_duration'] = float(np.median([a / r for r, a in infusions]))
    if rows:
        med = lambda k, rs=rows: (float(np.median([r[k] for r in rs if r.get(k) is not None]))
                                  if any(r.get(k) is not None for r in rs) else None)
        out['median'] = {k: med(k) for k in ('cmax', 'auc_inf', 'half_life', 'cl', 'vz', 'extrapolated_pct')}
        # Tmax only from subjects observed before their peak
        early = [r for r in rows if r['tmax'] > 0 and r.get('first_sample_is_peak') is False]
        out['median']['tmax'] = med('tmax', early) if early else None
        out['median']['tmax_subjects'] = len(early)
        out['per_subject'] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                              for r in rows[:40]]
        levels = {}
        for r in rows:
            levels.setdefault(round(r['dose'], 6), []).append(r)
        if len(levels) >= 2 and all(len(v) >= 3 for v in levels.values()):     # dose proportionality
            out['by_dose'] = [dict(dose=d, subjects=len(v), median_cl=med('cl', v),
                                   median_auc_inf_per_dose=(float(np.median([r['auc_inf'] / d for r in v
                                                                             if r.get('auc_inf')]))
                                                            if any(r.get('auc_inf') for r in v) else None),
                                   median_cmax_per_dose=float(np.median([r['cmax'] / d for r in v])))
                              for d, v in sorted(levels.items())]
        cl, vz = out['median'].get('cl'), out['median'].get('vz')
        start = {}
        if cl and vz:
            start = dict(CL=round(cl, 4), V=round(vz, 4))
            k = cl / vz
            if 'rises' in out['profile'] and out['median']['tmax'] and not infusions:    # not for infusions
                ka = _ka_from_tmax(out['median']['tmax'], k)
                if ka:
                    start['Ka'] = round(ka, 4)
        out['suggested_starting_values'] = start
    if len(rows) >= 5 and len(t):              # pooled shape only from rich first-interval profiles
        if 'rises' in out['profile']:
            tmax = out.get('median', {}).get('tmax') or float(t[np.argmax(y)])
            sel = t >= tmax
        else:
            sel = np.ones_like(t, dtype=bool)
        ph = _phases(t[sel], y[sel])
        if ph:
            out['pooled_decline'] = ph
    if not rows:
        out['note'] = ('too few observations within a dosing interval for per-subject NCA; take starting values from '
                       'the literature or from a first fit_models run')
    return out
