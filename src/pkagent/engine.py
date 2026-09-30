"""Fitting jobs for worker processes, result summaries for the agent, and restoring fits.

A job is a plain dict (specification, data file, options), so it runs in any
process; the worker saves the PKPy2 fit (fit.json), diagnostics and plots in the
model's folder and returns a JSON-able summary.
"""
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np


def init_worker(threads):
    import numba
    numba.set_num_threads(max(1, min(int(threads), numba.config.NUMBA_NUM_THREADS)))
    import matplotlib
    matplotlib.use('Agg')


def load_data(data_csv, norm):
    import pkpy2
    from .spec import covariate_columns
    return pkpy2.read_nonmem(data_csv, covariates=covariate_columns(norm), occasion=norm.get('occasion_column') or None)


# Importance proposals for every bank (fit, audit, standard errors, diagnostics): t-mixtures with mode discovery,
# which keep the effective sample size up for bimodal or long-tailed individual posteriors (e.g. the weakly
# identified slow compartment of a three-compartment model); conditional modes also get a prior-draw restart.
INTEGRATION = dict(proposal='mixture', mode_search=True)


def restore(norm, data_csv, fit_json):
    """The PKPy2 result object of a saved fit (same data file and specification)."""
    from pkpy2._general.fit import GeneralFitResult
    from pkpy2._general.problem import GeneralProblem
    from .spec import compile_model
    d = json.loads(Path(fit_json).read_text(encoding='utf-8'))
    integration = (d.get('estimation') or {}).get('integration') or INTEGRATION
    problem = GeneralProblem(load_data(data_csv, norm), compile_model(norm), integration)
    if list(problem.labels) != d['coordinates']:
        raise ValueError('saved fit does not match the specification')
    return GeneralFitResult(problem, np.asarray(d['x'], dtype=float), float(d['ofv']), d['status'], d['audit'],
                            d.get('estimation', {}), float(d.get('seconds', 0.)), d.get('uncertainty'))


# ---------------------------------------------------------------------- jobs
def run_fit(job):
    """Fit one specification; returns the summary (or an error record)."""
    t0 = time.time()
    out = Path(job['out_dir'])
    out.mkdir(parents=True, exist_ok=True)
    try:
        import pkpy2
        from .spec import compile_model
        norm = job['spec']
        data = load_data(job['data_csv'], norm)
        model = compile_model(norm)
        start = job.get('start_x')
        progress = open(out / 'progress.jsonl', 'a', encoding='utf-8')

        def log(row):
            keep = {k: v for k, v in row.items() if k not in ('x', 'audit', 'score')}
            keep['wall_seconds'] = round(time.time() - t0, 1)
            progress.write(json.dumps(keep, default=_json) + '\n')
            progress.flush()
        method = job.get('estimation', 'laplace')
        wall = float(job.get('wall_seconds', 900.))
        options = dict(seed=int(job['seed']), integration=INTEGRATION, callback=log, method=method,
                       laplace_options=dict(scaled=True, gradient='exact', starts=int(job.get('starts', 2)),
                                            cpu_budget_seconds=1e9, wall_seconds=wall if method == 'laplace'
                                            else float(job.get('laplace_seconds', 300.))))
        if method == 'importance':
            options['refinement_options'] = dict(wall_budget_seconds=wall)
        else:
            options['laplace_tolerance'] = float(job.get('laplace_tolerance', .5))
        if start is not None:
            options['start'] = np.asarray(start)
        try:
            res = pkpy2.fit(data, model, **options)
        finally:
            progress.close()
        summary = summarize(res, norm)
        if res.converged and job.get('uncertainty', True):
            summary['uncertainty_status'] = _uncertainty(res, summary)
        res.save(out / 'fit.json')
        if job.get('diagnostics', True) and res.converged:
            try:
                summary['diagnostics'] = diagnostics(res, norm, out, npde_samples=int(job.get('npde_samples', 300)))
            except Exception as e:                              # noqa: BLE001 - reported to the agent
                summary['diagnostics'] = dict(error=f'{type(e).__name__}: {e}')
        summary['warnings'] = warnings(summary, norm)
        summary['seconds'] = round(time.time() - t0, 1)
        (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=_json), encoding='utf-8')
        return summary
    except Exception as e:                                      # noqa: BLE001
        rec = dict(status='error', error=f'{type(e).__name__}: {e}', seconds=round(time.time() - t0, 1),
                   traceback=traceback.format_exc()[-3000:])
        (out / 'summary.json').write_text(json.dumps(rec, indent=1, default=_json), encoding='utf-8')
        return rec


def run_vpc(job):
    """VPC of a saved fit: plot per output and the fraction of observed percentiles inside the intervals."""
    try:
        import pkpy2
        from pkpy2 import plots
        import matplotlib.pyplot as plt
        norm = job['spec']
        res = restore(norm, job['data_csv'], job['fit_json'])
        check = pkpy2.vpc(res, n=int(job.get('n', 500)), bins=int(job.get('bins', 8)),
                          prediction_corrected=bool(job.get('prediction_corrected', False)),
                          lloq=job.get('lloq'), seed=int(job.get('seed', 1)))
        out = Path(job['out_dir'])
        summary = {}
        for name, e in check.items():
            fig = plots.vpc(check, name, log=bool(job.get('log', False)))
            path = out / f"vpc_{name}{'_pc' if job.get('prediction_corrected') else ''}.png"
            fig.savefig(path, dpi=130)
            plt.close(fig)
            obs, lo, hi = (np.asarray(e[k], dtype=float) for k in ('observed', 'lower', 'upper'))
            inside = (obs >= lo) & (obs <= hi)
            bins = []
            for b, t in enumerate(np.asarray(e['bin_time'])):
                row = dict(time=round(float(t), 3))
                for j, q in enumerate(e['quantiles']):
                    row[f'p{int(round(100 * q))}'] = dict(observed=_r(obs[b, j]), interval=[_r(lo[b, j]), _r(hi[b, j])],
                                                          inside=bool(inside[b, j]))
                bins.append(row)
            summary[name] = dict(observed_percentiles_inside=f'{int(inside.sum())}/{int(np.isfinite(obs).sum())}',
                                 bins=bins, plot=str(path))
        return summary
    except Exception as e:                                      # noqa: BLE001
        return dict(status='error', error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc()[-2000:])


def run_standard_errors(job):
    """Standard errors (observed information and sandwich) of a saved fit that was fitted without them."""
    try:
        norm = job['spec']
        res = restore(norm, job['data_csv'], job['fit_json'])
        summary = json.loads(Path(job['summary_json']).read_text(encoding='utf-8'))
        summary['uncertainty_status'] = _uncertainty(res, summary)
        summary['warnings'] = warnings(summary, norm)
        res.save(job['fit_json'])
        Path(job['summary_json']).write_text(json.dumps(summary, indent=1, default=_json), encoding='utf-8')
        return summary
    except Exception as e:                                      # noqa: BLE001
        return dict(status='error', error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc()[-2000:])


def run_screen(job):
    """Empirical Bayes estimates of a saved fit against subject covariates (screening for covariate effects)."""
    try:
        import pandas as pd
        from scipy import stats
        import pkpy2
        import matplotlib.pyplot as plt
        norm = job['spec']
        res = restore(norm, job['data_csv'], job['fit_json'])
        ebe = pkpy2.individual_estimates(res, power=11)
        df = pd.read_csv(job['data_csv'], na_values=['.'])
        covs = [c for c in job['covariates'] if c in df.columns]
        base = df.groupby('ID')[covs].first()
        varying = {c: bool((df.groupby('ID')[c].nunique(dropna=True) > 1).any()) for c in covs}
        etas = pd.DataFrame([dict(ID=e['id'], **e['eta_mode']) for e in ebe]).set_index('ID')
        base = base.reindex(etas.index)
        rows = []
        for p in etas.columns:
            y = etas[p].to_numpy(float)
            for c in covs:
                z = base[c].to_numpy(float)
                ok = np.isfinite(y) & np.isfinite(z)
                levels = np.unique(z[ok])
                if ok.sum() < 5 or len(levels) < 2:
                    continue
                if len(levels) <= 4:
                    groups = [y[ok][z[ok] == v] for v in levels]
                    groups = [g for g in groups if len(g)]
                    stat = stats.kruskal(*groups) if len(groups) > 1 else None
                    rows.append(dict(parameter=p, covariate=c, type='categorical',
                                     mean_eta_by_level={_lvl(v): _r(np.mean(g), 3) for v, g in zip(levels, groups)},
                                     n_by_level={_lvl(v): int(len(g)) for v, g in zip(levels, groups)},
                                     p_value=_r(stat.pvalue, 3) if stat else None, time_varying=varying[c]))
                else:
                    rho, pv = stats.spearmanr(z[ok], y[ok])
                    row = dict(parameter=p, covariate=c, type='continuous', spearman_rho=_r(rho, 3), p_value=_r(pv, 3),
                               time_varying=varying[c])
                    if np.all(z[ok] > 0):
                        slope = np.polyfit(np.log(z[ok] / np.median(z[ok])), y[ok], 1)[0]
                        row['power_exponent_estimate'] = _r(slope, 3)
                    rows.append(row)
        rows.sort(key=lambda r: (r['p_value'] if r['p_value'] is not None else 1.))
        shrink = {}
        try:
            _, s = pkpy2.diagnostics(res, npde_samples=50)
            shrink = {k: _r(100 * v, 3) for k, v in (s.get('eta_shrinkage') or {}).items()}
        except Exception:                                      # noqa: BLE001
            pass
        # plot grid: parameters x covariates
        if len(etas.columns) and covs:
            fig, axes = plt.subplots(len(etas.columns), len(covs), figsize=(2.6 * len(covs), 2.2 * len(etas.columns)),
                                     squeeze=False, layout='constrained')
            for i, p in enumerate(etas.columns):
                for j, c in enumerate(covs):
                    ax = axes[i, j]
                    ax.scatter(base[c], etas[p], s=10, color='#0072B2')
                    ax.axhline(0, color='k', lw=.8)
                    if i == len(etas.columns) - 1:
                        ax.set_xlabel(c)
                    if j == 0:
                        ax.set_ylabel(f'eta {p}')
            path = Path(job['out_dir']) / 'eta_covariates.png'
            fig.savefig(path, dpi=100)
            plt.close(fig)
        return dict(eta_shrinkage_percent=shrink, relationships=rows[:30], plot='eta_covariates.png',
                    note='screening only: confirm any effect with a fit and a likelihood-ratio test')
    except Exception as e:                                      # noqa: BLE001
        return dict(status='error', error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc()[-2000:])


def _lvl(v):
    return str(int(v)) if float(v).is_integer() else str(v)


def run_uncertainty(job):
    """Bootstrap or SIR of a saved fit."""
    try:
        import pkpy2
        norm = job['spec']
        res = restore(norm, job['data_csv'], job['fit_json'])
        method = job['method']
        if method == 'bootstrap':
            b = pkpy2.bootstrap(res, n=int(job.get('n', 100)), seed=int(job.get('seed', 1)),
                                fit_options=dict(integration=INTEGRATION, laplace_options=dict(scaled=True),
                                                 refinement_options=dict(wall_budget_seconds=float(job.get('wall_seconds', 900.)))))
            return dict(method='bootstrap', requested=b['requested'], converged=b['converged'],
                        intervals=_interval_rows(b['summary']))
        if method == 'sir':
            unc = res.uncertainty_report or res.uncertainty()
            s = pkpy2.sir(res, samples=int(job.get('samples', 1000)), resamples=int(job.get('resamples', 500)),
                          iterations=int(job.get('iterations', 4)), covariance=np.asarray(unc['covariance']),
                          seed=int(job.get('seed', 1)))
            return dict(method='sir', effective_samples=float(s['effective_samples']), intervals=_interval_rows(s['summary']))
        raise ValueError(f'unknown method {method}')
    except Exception as e:                                      # noqa: BLE001
        return dict(status='error', error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc()[-2000:])


# ---------------------------------------------------------------------- summaries
def _r(v, d=4):
    v = float(v)
    if not math.isfinite(v):
        return None
    return float(f'{v:.{d}g}')


def _json(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return str(v)


def log_scale_jacobian(res):
    """2 sum(log DV) over the uncensored observations of outputs with a log-normal residual model. PKPy2 reports
    -2 log L of log(DV) for them; adding this term gives -2 log L of DV itself, comparable with additive,
    proportional and combined error models of the same data."""
    problem = res.problem
    lognormal = [o for o in range(problem.n_out) if problem.form[o] == 1]
    if not lognormal:
        return 0.
    total = 0.
    for s in problem.subjects:
        keep = np.asarray(s.obs_used, dtype=bool) & np.isin(s.obs_out, lognormal) & (np.asarray(s.obs_cens) == 0)
        total += 2. * float(np.sum(np.log(np.asarray(s.obs_dv, dtype=float)[keep])))
    return total


def summarize(res, norm):
    d = res.to_dict()
    status = 'converged' if res.converged else 'not converged'
    est = d.get('estimation', {}) or {}
    method = est.get('method', 'importance')
    message = est.get('message') if method == 'laplace' else (est.get('refinement') or {}).get('message')
    jac = log_scale_jacobian(res)
    out = dict(status=status, estimation=method, engine_status=d['status'], message=message,
               ofv=_r(d['ofv'] + jac, 8), n_estimated=int(d['n_param']), aic=_r(d['aic'] + jac, 8),
               bic=_r(d['bic'] + jac, 8))
    if jac:
        out['ofv_log_scale'] = _r(d['ofv'], 8)
        out['ofv_note'] = ('log-normal residual: the OFV includes 2*sum(log DV), so it is comparable with models '
                           'that use additive, proportional or combined error on the same data')
    audit = d.get('audit') or {}
    if method == 'laplace':
        out['audit'] = dict(newton_decrement=_r(audit.get('newton_decrement', math.nan), 3),
                            tolerance=audit.get('ofv_tolerance'))
        runs = (est.get('laplace_exploration') or {}).get('distinct_optima') or []
        if len(runs) > 1:
            out['distinct_local_optima_ofv'] = [_r(r['ofv'], 8) for r in runs]
    elif audit.get('replicas'):
        out['audit'] = dict(passed=bool(audit.get('passed')), ofv_range_between_banks=_r(audit.get('ofv_range', math.nan), 3),
                            max_scaled_score=_r(max((r.get('scaled_score_max', 0) or 0) for r in audit['replicas']), 3),
                            min_ess=_r(min((r.get('minimum_ess', 0) or 0) for r in audit['replicas']), 4))
    params = []
    for p, v in d['theta'].items():
        spec = norm['parameters'][p]
        row = dict(parameter=p, estimate=_r(v), fixed=spec['fixed'])
        if spec['lower'] is not None or spec['upper'] is not None:
            row['bounds'] = [spec['lower'], spec['upper']]
            if not spec['fixed']:
                if spec['lower'] is not None and abs(v - spec['lower']) <= 1e-3 * max(1., abs(spec['lower'])):
                    row['at_bound'] = 'lower'
                if spec['upper'] is not None and abs(v - spec['upper']) <= 1e-3 * max(1., abs(spec['upper'])):
                    row['at_bound'] = 'upper'
        params.append(row)
    out['parameters'] = params
    out['iiv'] = [dict(parameter=p, variance=_r(v), cv_percent=_r(100 * math.sqrt(math.exp(v) - 1)) if v < 20 else None,
                       fixed=norm['iiv'][p]['fixed']) for p, v in d['omega'].items()]
    if d.get('omega_covariance'):
        out['iiv_covariance'] = {k: _r(v) for k, v in d['omega_covariance'].items()}
        corr = {}
        for k, v in d['omega_covariance'].items():
            a, b = k.split(',')
            if a in d['omega'] and b in d['omega'] and d['omega'][a] > 0 and d['omega'][b] > 0:
                corr[k] = _r(v / math.sqrt(d['omega'][a] * d['omega'][b]), 3)
        out['iiv_correlation'] = corr
    if d.get('iov'):
        out['iov'] = {k: _r(v) for k, v in d['iov'].items()}
    if d.get('coefficients'):
        out['covariate_effects'] = [dict(effect=k, coefficient=_r(v)) for k, v in d['coefficients'].items()]
    out['residual'] = [dict(component=k, sd=_r(v)) for k, v in d['sigma'].items()]
    return out


def _uncertainty(res, summary):
    try:
        rep = res.uncertainty()
    except Exception as e:                                      # noqa: BLE001
        return f'failed: {type(e).__name__}: {e}'
    if rep.get('status') != 'computed' or not rep.get('intervals'):
        return rep.get('status', 'not computed')
    by_q = {r['quantity']: r for r in rep['intervals']}
    for row in summary['parameters']:
        q = by_q.get(f"theta:{row['parameter']}")
        if q:
            row['rse_percent'] = _r(q.get('rse_pct'), 3)
            row['ci95'] = [_r(q['interval'][0]), _r(q['interval'][1])]
    for row in summary['iiv']:
        q = by_q.get(f"omega:{row['parameter']}")
        if q:
            row['rse_percent'] = _r(q.get('rse_pct'), 3)
            row['ci95'] = [_r(q['interval'][0]), _r(q['interval'][1])]
    for row in summary.get('covariate_effects', []):
        q = by_q.get(f"beta:{row['effect']}")
        if q:
            row['se'] = _r(q.get('se'), 3)
            row['ci95'] = [_r(q['interval'][0]), _r(q['interval'][1])]
    for row in summary['residual']:
        q = by_q.get(f"sigma:{row['component']}")
        if q:
            row['rse_percent'] = _r(q.get('rse_pct'), 3)
    corr = rep.get('correlation')
    labels = rep.get('coordinates') or rep.get('labels')
    if corr is not None and labels:
        c = np.asarray(corr, dtype=float)
        high = [dict(pair=[labels[i], labels[j]], correlation=_r(c[i, j], 3))
                for i in range(len(labels)) for j in range(i) if abs(c[i, j]) > .9]
        if high:
            summary['high_estimate_correlations'] = high
    return 'computed'


def _bins(x, y, n=6):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 * n:
        n = max(2, len(x) // 5)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n + 1)))
    idx = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, len(edges) - 2)
    return [dict(x_range=[_r(edges[b], 3), _r(edges[b + 1], 3)], n=int(np.sum(idx == b)),
                 mean=_r(np.mean(y[idx == b]), 3)) for b in range(len(edges) - 1) if np.any(idx == b)]


def diagnostics(res, norm, out, npde_samples=300):
    import pkpy2
    from pkpy2 import plots
    import matplotlib.pyplot as plt
    table, s = pkpy2.diagnostics(res, npde_samples=npde_samples)
    result = dict(eta_shrinkage_percent={k: _r(100 * v, 3) for k, v in (s.get('eta_shrinkage') or {}).items()},
                  epsilon_shrinkage_percent=_r(100 * s['epsilon_shrinkage'], 3) if s.get('epsilon_shrinkage') is not None else None)
    outputs = [o for o in dict.fromkeys(np.asarray(table['OUTPUT']).tolist())]
    per = {}
    for o in outputs:
        k = (np.asarray(table['OUTPUT']) == o) & (np.asarray(table['CENS']) == 0)
        cw = np.asarray(table['CWRES'], float)[k]
        npde = np.asarray(table['NPDE'], float)[k]
        fin = np.isfinite(cw)
        entry = dict(n=int(k.sum()), cwres_mean=_r(np.mean(cw[fin]), 3), cwres_sd=_r(np.std(cw[fin], ddof=1), 3),
                     abs_cwres_above_3=int(np.sum(np.abs(cw[fin]) > 3)),
                     cwres_by_time=_bins(np.asarray(table['TIME'], float)[k], cw),
                     cwres_by_pred=_bins(np.asarray(table['PRED'], float)[k], cw))
        fn = np.isfinite(npde)
        if fn.any():
            entry.update(npde_mean=_r(np.mean(npde[fn]), 3), npde_variance=_r(np.var(npde[fn], ddof=1), 3))
        dv, ipred = np.asarray(table['DV'], float)[k], np.asarray(table['IPRED'], float)[k]
        ok = np.isfinite(dv) & np.isfinite(ipred) & (dv > 0)
        if ok.any():
            entry['median_abs_individual_error_percent'] = _r(100 * np.median(np.abs(ipred[ok] - dv[ok]) / dv[ok]), 3)
        name = o if isinstance(o, str) else str(o)
        fig = plots.gof(table, output=o)
        fig.savefig(out / f'gof_{name}.png', dpi=110)
        plt.close(fig)
        ids = list(dict.fromkeys(np.asarray(table['ID'])[k].tolist()))[:12]
        fig = plots.individual_fits(table, ids, output=o)
        fig.savefig(out / f'individual_{name}.png', dpi=110)
        plt.close(fig)
        per[name] = entry
    result['outputs'] = per
    with open(out / 'diagnostics.csv', 'w', encoding='utf-8') as f:
        cols = [c for c in ('ID', 'TIME', 'OUTPUT', 'DV', 'CENS', 'PRED', 'IPRED', 'IWRES', 'CWRES', 'NPDE') if c in table]
        f.write(','.join(cols) + '\n')
        for row in zip(*(np.asarray(table[c]) for c in cols)):
            f.write(','.join(str(v) for v in row) + '\n')
    return result


def warnings(summary, norm):
    w = []
    if summary.get('status') != 'converged':
        w.append(f"not converged: {summary.get('message')}")
    for row in summary.get('parameters', []):
        if row.get('at_bound'):
            w.append(f"{row['parameter']} is at its {row['at_bound']} bound")
        if (row.get('rse_percent') or 0) > 50:
            w.append(f"{row['parameter']} RSE {row['rse_percent']}%")
    for row in summary.get('iiv', []):
        if row['variance'] is not None and row['variance'] < 1e-3 and not row['fixed']:
            w.append(f"IIV of {row['parameter']} collapsed toward zero ({row['variance']})")
        if (row.get('rse_percent') or 0) > 100:
            w.append(f"IIV of {row['parameter']} RSE {row['rse_percent']}%")
    sh = (summary.get('diagnostics') or {}).get('eta_shrinkage_percent') or {}
    for p, v in sh.items():
        if v is not None and v > 40:
            w.append(f'eta shrinkage of {p} {v}%')
    if summary.get('uncertainty_status') not in (None, 'computed'):
        w.append(f"standard errors not available ({summary['uncertainty_status']})")
    for o, e in ((summary.get('diagnostics') or {}).get('outputs') or {}).items():
        worst = max((abs(b['mean']) for b in e.get('cwres_by_time', []) + e.get('cwres_by_pred', [])
                     if b['mean'] is not None), default=0)
        if worst > .75:
            w.append(f'{o}: binned mean CWRES reaches {worst:.2f} (trend against time or PRED)')
    return w


def _interval_rows(summary):
    rows = []
    for q, v in summary.items():
        if not isinstance(v, dict):
            continue
        iv = v.get('interval')
        row = dict(quantity=q, estimate=_r(v['estimate']) if v.get('estimate') is not None else None)
        if v.get('median') is not None:
            row['median'] = _r(v['median'])
        row['ci95'] = [_r(iv[0]), _r(iv[1])] if iv and None not in iv else None
        rows.append(row)
    return rows
