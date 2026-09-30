"""An analysis session: data versions, the registry of fitted models, parallel jobs and the budget.

Every tool call goes through Session.call, which records its arguments and
result in tool_log.jsonl. Models are identified as M001, M002, ...; each has a
folder with its specification, PKPy2 fit, diagnostics and plots.
"""
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from . import engine
from .config import Settings
from .data import Dataset
from .nca import run_nca
from .spec import SpecError, describe, normalize


class BudgetExhausted(RuntimeError):
    pass


class Session:
    def __init__(self, data_path, out_dir, settings=None, description=''):
        self.settings = settings or Settings()
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / 'models').mkdir(exist_ok=True)
        (self.out / 'data').mkdir(exist_ok=True)
        (self.out / 'plots').mkdir(exist_ok=True)
        self.dataset = Dataset(data_path, description)
        self.data_version = 0
        self._write_data()
        self.models = {}
        self.nca = None
        self.fits_used = 0
        self.t0 = time.time()
        self.final = None
        self.llm_cost = 0.
        self.turns = 0
        self._pool = None

    # ------------------------------------------------------------------ infrastructure
    @property
    def pool(self):
        if self._pool is None:
            s = self.settings
            self._pool = ProcessPoolExecutor(max_workers=s.workers, initializer=engine.init_worker,
                                             initargs=(s.threads_per_worker,))
        return self._pool

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def data_csv(self, version=None):
        return str(self.out / 'data' / f'data_v{self.data_version if version is None else version}.csv')

    def _write_data(self):
        self.dataset.write_normalized(self.data_csv())

    def log(self, record):
        with open(self.out / 'tool_log.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, default=engine._json) + '\n')

    def budget_status(self):
        b = self.settings.budget
        return dict(fits_used=self.fits_used, fits_left=b.max_fits - self.fits_used,
                    hours_used=round((time.time() - self.t0) / 3600, 2), hours_left=round(b.max_hours - (time.time() - self.t0) / 3600, 2),
                    llm_turns_used=self.turns, llm_turns_left=b.max_turns - self.turns,
                    llm_cost_usd=round(self.llm_cost, 2))

    def _budget_problem(self, fits=0):
        """Why `fits` more fits would exceed the budget, or None."""
        b = self.settings.budget
        if self.fits_used + fits > b.max_fits:
            return f'fit budget exhausted ({self.fits_used}/{b.max_fits} used); finalize with the best model'
        if (time.time() - self.t0) / 3600 > b.max_hours:
            return 'time budget exhausted; finalize with the best model'
        return None

    def _check_budget(self, fits=0):
        problem = self._budget_problem(fits)
        if problem:
            raise BudgetExhausted(problem)

    def _new_id(self):
        return f'M{len(self.models) + 1:03d}'

    def _suggestions(self):
        return (self.nca or {}).get('suggested_starting_values') or {}

    # ------------------------------------------------------------------ fitting
    def fit(self, specs, uncertainty=True, starts=None, estimation='laplace'):
        """Normalize and fit specifications in parallel; returns one record per specification."""
        if estimation not in ('laplace', 'importance'):
            raise ValueError("estimation must be 'laplace' or 'importance'")
        records, jobs = [], []
        ok = []
        for spec in specs:
            try:
                norm = normalize(spec, self._suggestions())
                ok.append(norm)
            except SpecError as e:
                records.append(dict(status='invalid specification', error=str(e), specification=spec))
                ok.append(None)
        n = sum(1 for x in ok if x is not None)
        self._check_budget(n)
        futures = {}
        for i, norm in enumerate(ok):
            if norm is None:
                continue
            if norm['parent'] and norm['parent'] not in self.models:
                norm['parent'] = None
            mid = self._new_id()
            folder = self.out / 'models' / mid
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'spec.json').write_text(json.dumps(norm, indent=1), encoding='utf-8')
            rec = dict(model_id=mid, name=norm['name'], parent=norm['parent'], description=describe(norm), spec=norm,
                       data_version=self.data_version, created=round(time.time() - self.t0, 1), status='running')
            self.models[mid] = rec
            job = dict(spec=norm, data_csv=self.data_csv(), out_dir=str(folder), seed=self.settings.seed,
                       wall_seconds=self.settings.fit_wall_seconds, laplace_seconds=self.settings.laplace_seconds,
                       uncertainty=uncertainty, estimation=estimation, starts=self.settings.laplace_starts,
                       laplace_tolerance=self.settings.laplace_tolerance,
                       start_x=(starts or {}).get(i))
            futures[self.pool.submit(engine.run_fit, job)] = (i, mid)
            self.fits_used += 1
        done, _ = wait(list(futures), timeout=self.settings.fit_wall_seconds * 3 + 600)
        results = {}
        for fut, (i, mid) in futures.items():
            if fut in done:
                try:
                    summary = fut.result()
                except Exception as e:                           # noqa: BLE001
                    summary = dict(status='error', error=f'{type(e).__name__}: {e}')
            else:
                summary = dict(status='error', error='fit did not finish within the time limit')
                fut.cancel()
            rec = self.models[mid]
            rec['summary'] = summary
            rec['status'] = summary.get('status')
            rec['ofv'] = summary.get('ofv')
            rec['n_estimated'] = summary.get('n_estimated')
            results[i] = rec
        out = []
        j = 0
        for i, norm in enumerate(ok):
            if norm is None:
                out.append(records[j]); j += 1
            else:
                out.append(results[i])
        return out

    def compact(self, rec, detail=False):
        """What the agent sees about a fitted model."""
        s = rec.get('summary') or {}
        row = dict(model_id=rec['model_id'], name=rec['name'], description=rec['description'], parent=rec['parent'],
                   status=s.get('status'))
        if s.get('status') == 'error':
            row['error'] = s.get('error')
            return row
        row.update(estimation=s.get('estimation'), ofv=s.get('ofv'), n_estimated=s.get('n_estimated'),
                   aic=s.get('aic'), bic=s.get('bic'))
        if s.get('distinct_local_optima_ofv'):
            row['distinct_local_optima_ofv'] = s['distinct_local_optima_ofv']
        if s.get('ofv_note'):
            row['ofv_note'] = s['ofv_note']
        if rec['parent'] and self.models.get(rec['parent'], {}).get('ofv') is not None and s.get('ofv') is not None:
            p = self.models[rec['parent']]
            row['delta_ofv_vs_parent'] = round(s['ofv'] - p['ofv'], 3)
            row['delta_n_estimated_vs_parent'] = (s.get('n_estimated') or 0) - (p.get('n_estimated') or 0)
            if p['data_version'] != rec['data_version']:
                row['note'] = 'parent was fitted to a different data version'
        if s.get('status') != 'converged':
            row['message'] = s.get('message')
        row['parameters'] = [_short(r, ('parameter', 'estimate', 'fixed', 'rse_percent', 'ci95', 'bounds', 'at_bound'))
                             for r in s.get('parameters', [])]
        diag = s.get('diagnostics') or {}
        shrink = diag.get('eta_shrinkage_percent') or {}
        row['iiv'] = [dict(_short(r, ('parameter', 'variance', 'cv_percent', 'fixed', 'rse_percent')),
                           shrinkage_percent=shrink.get(r['parameter'])) for r in s.get('iiv', [])]
        for k in ('iiv_correlation', 'iov', 'covariate_effects', 'high_estimate_correlations'):
            if s.get(k):
                row[k] = s[k]
        row['residual'] = [r for r in s.get('residual', []) if r.get('sd') not in (0, 0.)]
        if s.get('uncertainty_status') and s['uncertainty_status'] != 'computed':
            row['standard_errors'] = s['uncertainty_status']
        if diag.get('outputs'):
            row['diagnostics'] = {}
            for o, e in diag['outputs'].items():
                d = dict(n=e['n'], cwres_mean=e['cwres_mean'], cwres_sd=e['cwres_sd'],
                         abs_cwres_above_3=e['abs_cwres_above_3'],
                         cwres_mean_by_time_bin=[b['mean'] for b in e['cwres_by_time']],
                         time_bin_edges=[e['cwres_by_time'][0]['x_range'][0]] + [b['x_range'][1] for b in e['cwres_by_time']],
                         cwres_mean_by_pred_bin=[b['mean'] for b in e['cwres_by_pred']],
                         pred_bin_edges=[e['cwres_by_pred'][0]['x_range'][0]] + [b['x_range'][1] for b in e['cwres_by_pred']])
                for k in ('npde_mean', 'npde_variance', 'median_abs_individual_error_percent'):
                    if k in e:
                        d[k] = e[k]
                row['diagnostics'][o] = d
            if diag.get('epsilon_shrinkage_percent') is not None:
                row['epsilon_shrinkage_percent'] = diag['epsilon_shrinkage_percent']
        elif diag.get('error'):
            row['diagnostics_error'] = diag['error']
        row['warnings'] = s.get('warnings', [])
        row['fit_seconds'] = s.get('seconds')
        if detail:
            row['specification'] = rec['spec']
            row['audit'] = s.get('audit')
        return row

    def plots_of(self, model_id):
        folder = self.out / 'models' / model_id
        return sorted(p.name for p in folder.glob('*.png'))

    # ------------------------------------------------------------------ comparisons
    def compare(self, model_ids, reference=None):
        recs = [self.models[m] for m in model_ids if m in self.models]
        missing = [m for m in model_ids if m not in self.models]
        ref = self.models.get(reference) if reference else None
        rows = []
        for r in recs:
            row = dict(model_id=r['model_id'], description=r['description'], status=r['status'], ofv=r.get('ofv'),
                       n_estimated=r.get('n_estimated'), aic=(r.get('summary') or {}).get('aic'),
                       bic=(r.get('summary') or {}).get('bic'), data_version=r['data_version'])
            if ref and r is not ref and ref.get('ofv') is not None and r.get('ofv') is not None:
                d_ofv = r['ofv'] - ref['ofv']
                df = (r.get('n_estimated') or 0) - (ref.get('n_estimated') or 0)
                row['delta_ofv_vs_reference'] = round(d_ofv, 3)
                row['delta_parameters'] = df
                if df != 0:
                    big, small = (r, ref) if df > 0 else (ref, r)
                    delta = small['ofv'] - big['ofv']
                    row['lrt_p_value_if_nested'] = float(f'{chi2.sf(max(delta, 0), abs(df)):.3g}')
            rows.append(row)
        best_aic = min((x['aic'] for x in rows if x['aic'] is not None), default=None)
        for x in rows:
            if best_aic is not None and x['aic'] is not None:
                x['delta_aic_vs_best'] = round(x['aic'] - best_aic, 3)
        out = dict(models=rows)
        if len({x['data_version'] for x in rows}) > 1:
            out['warning'] = 'models were fitted to different data versions; their OFVs are not comparable'
        if missing:
            out['unknown_models'] = missing
        return out

    # ------------------------------------------------------------------ jobs on saved fits
    def _saved(self, model_id):
        rec = self.models.get(model_id)
        if rec is None:
            raise ValueError(f'unknown model {model_id}')
        fit_json = self.out / 'models' / model_id / 'fit.json'
        if rec['status'] != 'converged' or not fit_json.exists():
            raise ValueError(f'{model_id} has no converged fit')
        return rec, fit_json

    def vpc(self, model_id, prediction_corrected=False, log_scale=False, lloq=None, bins=8):
        rec, fit_json = self._saved(model_id)
        job = dict(spec=rec['spec'], data_csv=self.data_csv(rec['data_version']), fit_json=str(fit_json),
                   out_dir=str(fit_json.parent), prediction_corrected=prediction_corrected, log=log_scale, lloq=lloq,
                   bins=bins, seed=self.settings.seed)
        result = self.pool.submit(engine.run_vpc, job).result(timeout=3600)
        if isinstance(result, dict) and result.get('status') != 'error':
            rec['last_vpc'] = dict(prediction_corrected=prediction_corrected, result=result)
        return result

    def standard_errors(self, model_id):
        rec, fit_json = self._saved(model_id)
        job = dict(spec=rec['spec'], data_csv=self.data_csv(rec['data_version']), fit_json=str(fit_json),
                   summary_json=str(fit_json.parent / 'summary.json'))
        summary = self.pool.submit(engine.run_standard_errors, job).result(timeout=3600)
        if summary.get('status') == 'converged':
            rec['summary'] = summary
        return summary

    def finalize_outputs(self, model_id):
        """Standard errors and a VPC for the final model when they are missing."""
        rec, fit_json = self._saved(model_id)
        s = rec.get('summary') or {}
        if not s.get('uncertainty_status'):
            self.standard_errors(model_id)
        if not rec.get('last_vpc') or rec['last_vpc'].get('prediction_corrected'):
            self.vpc(model_id)
        rec['final_vpc'] = rec.get('last_vpc')
        return rec

    def screen(self, model_id, covariates=None):
        rec, fit_json = self._saved(model_id)
        covs = covariates or self.dataset.covariate_columns()
        job = dict(spec=rec['spec'], data_csv=self.data_csv(rec['data_version']), fit_json=str(fit_json),
                   out_dir=str(fit_json.parent), covariates=covs)
        return self.pool.submit(engine.run_screen, job).result(timeout=3600)

    def resample(self, model_id, method='bootstrap', n=100, **options):
        """Nonparametric bootstrap of a converged model: refits in parallel on the worker pool (settings of the model
        fits), stopped at the time limit (bootstrap_seconds, and the remaining time budget)."""
        if method != 'bootstrap':
            raise ValueError("only method='bootstrap' is available")
        rec, fit_json = self._saved(model_id)
        self._check_budget(0)
        n = max(1, min(int(n), 200))
        left = self.settings.budget.max_hours * 3600 - (time.time() - self.t0)
        seconds = max(60., min(self.settings.bootstrap_seconds, left - 300.))
        deadline = time.time() + seconds
        workers = max(1, self.settings.workers)
        sizes = [n // workers + (1 if k < n % workers else 0) for k in range(workers)]
        futures = [self.pool.submit(engine.run_bootstrap_chunk, dict(
            spec=rec['spec'], data_csv=self.data_csv(rec['data_version']), fit_json=str(fit_json), n=size,
            seed=self.settings.seed + 7919 * (k + 1), deadline=deadline,
            laplace_tolerance=self.settings.laplace_tolerance)) for k, size in enumerate(sizes) if size]
        chunks = []
        for fut in futures:
            chunk = fut.result(timeout=seconds + 3600)
            if chunk.get('status') == 'error':
                raise RuntimeError(chunk['error'])
            chunks.append(chunk)
        result = engine.bootstrap_summary(chunks, n)
        result['time_limit_minutes'] = round(seconds / 60, 1)
        result['note'] = ('bootstrap refits are not counted against the model-fit budget; replicates not started '
                          'before the time limit are omitted')
        rec.setdefault('resampling', {})['bootstrap'] = result
        return result

    # ------------------------------------------------------------------ covariate search
    def covariate_search(self, base_id, candidates, forward_p=.05, backward_p=.01):
        """Stepwise covariate modeling with the candidate fits of each step run in parallel."""
        base = self.models.get(base_id)
        if base is None or base['status'] != 'converged':
            raise ValueError(f'{base_id} is not a converged model')
        cand = []
        for c in candidates:
            c = dict(c)
            if c.get('form') == 'categorical' and c.get('level') is None:
                raise ValueError(f"categorical candidate {c.get('covariate')} needs level")
            cand.append(c)

        def label(c):
            level = f"={float(c['level']):g}" if c.get('level') is not None else ''
            return f"{c['parameter']}~{c['covariate']}{level}({c.get('form', 'power')})"
        spec0 = _raw_spec(base['spec'])
        current, history = base_id, []
        included = []
        remaining = list(cand)
        while remaining:
            problem = self._budget_problem(len(remaining))
            if problem:                                  # stop the search, keep what was found so far
                history.append(dict(step='stopped', reason=problem))
                break
            specs = []
            for c in remaining:
                s = _raw_spec(self.models[current]['spec'])
                s['covariates'] = s.get('covariates', []) + [c]
                s['name'] = f"SCM + {label(c)}"
                s['parent'] = current
                s['parameters'] = _estimates_as_start(self.models[current])
                s['iiv'] = _iiv_start(self.models[current])
                specs.append(s)
            recs = self.fit(specs, uncertainty=False)
            trials = []
            for c, r in zip(remaining, recs):
                if r.get('status') == 'converged' and r.get('ofv') is not None:
                    trials.append((self.models[current]['ofv'] - r['ofv'], c, r['model_id']))
                history.append(dict(step='forward', candidate=label(c), model_id=r.get('model_id'), status=r.get('status'),
                                    delta_ofv=round(self.models[current]['ofv'] - r['ofv'], 3) if r.get('ofv') is not None else None))
            if not trials:
                break
            delta, best, mid = max(trials, key=lambda t: t[0])
            p = chi2.sf(max(delta, 0), 1)
            if p >= forward_p:
                history.append(dict(step='forward stop', best=label(best), delta_ofv=round(delta, 3), p_value=float(f'{p:.3g}')))
                break
            included.append(best)
            remaining = [c for c in remaining if c is not best]
            current = mid
            history.append(dict(step='forward add', added=label(best), model_id=mid, delta_ofv=round(delta, 3),
                                p_value=float(f'{p:.3g}')))
        changed = True
        while changed and included:
            changed = False
            problem = self._budget_problem(len(included))
            if problem:
                history.append(dict(step='stopped', reason=problem))
                break
            specs = []
            for c in included:
                s = _raw_spec(self.models[current]['spec'])
                s['covariates'] = [k for k in s.get('covariates', []) if label(k) != label(c)]
                s['name'] = f"SCM - {label(c)}"
                s['parent'] = current
                s['parameters'] = _estimates_as_start(self.models[current])
                s['iiv'] = _iiv_start(self.models[current])
                specs.append(s)
            recs = self.fit(specs, uncertainty=False)
            trials = [(r['ofv'] - self.models[current]['ofv'], c, r['model_id']) for c, r in zip(included, recs)
                      if r.get('status') == 'converged' and r.get('ofv') is not None]
            for c, r in zip(included, recs):
                history.append(dict(step='backward', candidate=label(c), model_id=r.get('model_id'), status=r.get('status'),
                                    delta_ofv=round(r['ofv'] - self.models[current]['ofv'], 3) if r.get('ofv') is not None else None))
            if not trials:
                break
            delta, weakest, mid = min(trials, key=lambda t: t[0])
            p = chi2.sf(max(delta, 0), 1)
            if p >= backward_p:
                included = [c for c in included if c is not weakest]
                current = mid
                changed = True
                history.append(dict(step='backward remove', removed=label(weakest), model_id=mid, delta_ofv=round(delta, 3),
                                    p_value=float(f'{p:.3g}')))
        return dict(final_model_id=current, included=[label(c) for c in included], history=history,
                    note='the final model was fitted without standard errors; refit it (fit_models) to obtain them')

    # ------------------------------------------------------------------ plots of the data
    def plot_data(self, log_scale=True, color_by=None, output=1, max_subjects=60):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        df = self.dataset.df
        obs = self.dataset.obs_mask()
        if 'DVID' in df:
            obs = obs & (df['DVID'].fillna(1).to_numpy() == output)
        o = df[obs]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
        ids = list(dict.fromkeys(o['ID']))[:max_subjects]
        cmap = plt.get_cmap('viridis')
        cvals = None
        if color_by and color_by in df:
            first = df.groupby('ID')[color_by].first()
            lo, hi = float(first.min()), float(first.max())
            cvals = {i: (float(first[i]) - lo) / (hi - lo) if hi > lo else .5 for i in ids}
        dose = df[self.dataset.dose_mask()]
        for sid in ids:
            g = o[o['ID'] == sid]
            color = cmap(cvals[sid]) if cvals else None
            axes[0].plot(g['TIME'], g['DV'], '-o', ms=2.5, lw=.8, alpha=.7, color=color)
            # time after the most recent dose
            dt = dose.loc[dose['ID'] == sid, 'TIME'].to_numpy()
            tad = [t - dt[dt <= t].max() if np.any(dt <= t) else np.nan for t in g['TIME']]
            axes[1].plot(tad, g['DV'], 'o', ms=3, alpha=.6, color=color or '#0072B2')
        for ax, xl in zip(axes, ('Time', 'Time after the most recent dose')):
            ax.set_xlabel(xl)
            ax.set_ylabel('DV')
            if log_scale:
                ax.set_yscale('log')
        title = f"{len(ids)} subjects" + (f", colored by {color_by}" if cvals else '')
        fig.suptitle(title)
        path = self.out / 'plots' / f"data_{'log' if log_scale else 'lin'}_{output}{'_' + color_by if cvals else ''}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path


def _short(r, keys):
    return {k: r[k] for k in keys if k in r and r[k] is not None and not (k == 'fixed' and r[k] is False)}


def _raw_spec(norm):
    """Back from a normalized specification to agent-level JSON (for derived models)."""
    return dict(
        name=norm['name'], structure=dict(norm['structure']),
        parameters={p: {k: v for k, v in d.items() if v is not None} for p, d in norm['parameters'].items()},
        iiv={p: {k: v for k, v in d.items() if v is not None and k in ('value', 'fixed')} for p, d in norm['iiv'].items()},
        iiv_blocks=[list(b) for b in norm['iiv_blocks']],
        iov={p: {k: v for k, v in d.items() if v is not None and k in ('value', 'fixed')} for p, d in norm['iov'].items()},
        **({'occasion_column': norm['occasion_column']} if norm.get('occasion_column') else {}),
        covariates=[dict(parameter=c['parameter'], covariate=c['covariate'], form=c['form'], center=c['center'],
                         **({'level': c['level']} if c['level'] is not None else {}),
                         value=c['coefficient']['value'], fixed=c['coefficient']['fixed'],
                         **({'lower': c['coefficient']['lower']} if c['coefficient']['lower'] is not None else {}),
                         **({'upper': c['coefficient']['upper']} if c['coefficient']['upper'] is not None else {}))
                    for c in norm['covariates']],
        residual={o: {k: {kk: vv for kk, vv in v.items() if vv is not None and kk in ('value', 'fixed')}
                      for k, v in r.items()} for o, r in norm['residual'].items()})


def _estimates_as_start(rec):
    """Starting values for a derived model: the parent's estimates for estimated parameters."""
    spec = _raw_spec(rec['spec'])['parameters']
    est = {r['parameter']: r['estimate'] for r in (rec.get('summary') or {}).get('parameters', [])}
    for p, d in spec.items():
        if not d.get('fixed') and est.get(p) is not None:
            v = est[p]
            lo, hi = d.get('lower'), d.get('upper')
            if lo is not None:
                v = max(v, lo + 1e-6 * max(1, abs(lo)))
            if hi is not None:
                v = min(v, hi - 1e-6 * max(1, abs(hi)))
            if d.get('scale', 'log') == 'logit':
                v = min(max(v, 1e-4), 1 - 1e-4)
            d['value'] = v
    return spec


def _iiv_start(rec):
    """IIV starting variances for a derived model: the parent's estimates (at least 0.001)."""
    spec = _raw_spec(rec['spec'])['iiv']
    est = {r['parameter']: r['variance'] for r in (rec.get('summary') or {}).get('iiv', [])}
    for p, d in spec.items():
        if not d.get('fixed') and est.get(p) is not None:
            d['value'] = max(float(est[p]), 1e-3)
    return spec
