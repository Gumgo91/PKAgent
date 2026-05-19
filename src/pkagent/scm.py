"""Stepwise Covariate Modelling + ensemble — the numerical core PKAgent uses
to test candidate covariates and consensus across perturbed initialisations.

This module is the **PKPy-style numerical engine** for the SCM step.  It has
no drug-specific knowledge and no LLM calls — pure numerical PopPK.

Two key entry points:

  - `run_ensemble(subjects, drug, model_name, eta_params, n_runs)`
      Multiple perturbed-init fits, returns the best converged fit.

  - `stepwise_covariate_modeling(subjects, drug, model_name, candidates,
                                  base_fit, eta_params)`
      Full SCM (forward + backward, Wählby/Jonsson/Karlsson 2002).
"""

from __future__ import annotations

import math
import time as _time

import numpy as np

from .engine import run_fit, build_config, make_predict_fn
from .nlme import FitResult, Subject, fit_population
from .priors import CLASSES, get_class


# χ² critical values used as standard SCM thresholds
ALPHA_FWD = 3.84   # χ²₁ at α = 0.05  (forward inclusion)
ALPHA_BWD = 6.63   # χ²₁ at α = 0.01  (backward elimination — stricter)


def stepwise_covariate_modeling(
        drug_class: str | None, subjects: list[Subject], model_name: str,
        candidates: list[dict], base_ofv: float, base_fit: FitResult,
        eta_params: list[str],
        alpha_fwd: float = ALPHA_FWD, alpha_bwd: float = ALPHA_BWD,
        verbose: bool = True) -> tuple[dict, list[dict]]:
    """Full Stepwise Covariate Modelling (Wählby, Jonsson & Karlsson 2002).

    Forward phase at α_fwd: at each step, try every remaining candidate as
    a single-covariate addition; retain the best dOFV if it clears the
    χ²₁(α=0.05) = 3.84 threshold.  Each trial is warm-started from the
    current best fit's converged θ̂/ω̂, so adding a β=0 covariate yields a
    statistically valid LRT against the null model.

    Backward phase at α_bwd: drop each retained covariate in turn and
    refit (warm-start).  If the OFV degradation when dropping is below
    χ²₁(α=0.01) = 6.63, remove the covariate.  The stricter backward
    threshold controls type-I-error inflation from the forward search.
    """
    selected = {}
    history = []

    # ---- forward phase ----
    remaining = list(candidates)
    current_ofv = base_ofv
    current_fit = base_fit
    while remaining:
        best_delta = 0.0
        best_pick = None
        best_fit = None
        warm_theta = dict(current_fit.theta)
        warm_omega = dict(current_fit.omega_diag)
        for c in remaining:
            key = f"{c['param']}~{c['covariate']}"
            trial = dict(selected)
            trial[key] = (c["kind"], 0.0)
            t0 = _time.time()
            try:
                fit = run_fit(drug_class, subjects, model_name,
                              covariate_effects=trial,
                              eta_params=eta_params, use_priors=True,
                              compute_se=False,
                              init_theta_override=warm_theta,
                              init_omega_override=warm_omega)
            except Exception as e:
                history.append({"phase": "fwd", "trial": key, "status": f"error: {e}"})
                if verbose:
                    print(f"    fwd {key}: ERROR {e}", flush=True)
                continue
            d = current_ofv - fit.ofv
            history.append({"phase": "fwd", "trial": key, "dOFV": d, "ofv": fit.ofv})
            if verbose:
                print(f"    fwd {key}: dOFV={d:+.2f} ({_time.time()-t0:.1f}s)",
                      flush=True)
            if d > best_delta:
                best_delta = d
                best_pick = c
                best_fit = fit
        if best_pick and best_delta > alpha_fwd:
            key = f"{best_pick['param']}~{best_pick['covariate']}"
            selected[key] = (best_pick["kind"], best_fit.covariate_effects.get(key, 0.0))
            remaining.remove(best_pick)
            current_ofv = best_fit.ofv
            current_fit = best_fit
            history.append({"phase": "fwd", "accepted": key, "new_ofv": current_ofv})
        else:
            break

    # ---- backward elimination phase ----
    if selected:
        if verbose:
            print(f"    backward phase begins with {list(selected)}", flush=True)
        improved = True
        while improved and selected:
            improved = False
            worst_key = None
            worst_delta = float("inf")
            warm_theta_bwd = dict(current_fit.theta)
            warm_omega_bwd = dict(current_fit.omega_diag)
            worst_fit = None
            for key in list(selected):
                trial = {k: v for k, v in selected.items() if k != key}
                t0 = _time.time()
                try:
                    fit = run_fit(drug_class, subjects, model_name,
                                  covariate_effects=trial,
                                  eta_params=eta_params, use_priors=True,
                                  compute_se=False,
                                  init_theta_override=warm_theta_bwd,
                                  init_omega_override=warm_omega_bwd)
                except Exception as e:
                    history.append({"phase": "bwd", "drop": key, "status": f"error: {e}"})
                    continue
                d = fit.ofv - current_ofv
                history.append({"phase": "bwd", "drop_trial": key,
                                "dOFV_lost": d, "ofv": fit.ofv})
                if verbose:
                    print(f"    bwd drop {key}: ΔOFV={d:+.2f} ({_time.time()-t0:.1f}s)",
                          flush=True)
                if d < worst_delta:
                    worst_delta = d
                    worst_key = key
                    worst_fit = fit
            if worst_key is not None and worst_delta < alpha_bwd and worst_fit is not None:
                selected.pop(worst_key)
                current_ofv = worst_fit.ofv
                current_fit = worst_fit
                history.append({"phase": "bwd", "dropped": worst_key, "new_ofv": current_ofv})
                improved = True

    return selected, history


def run_ensemble(drug_class: str | None, subjects: list[Subject],
                  model_name: str, eta_params: list[str],
                  n_runs: int = 3, perturb: float = 0.2, seed: int = 0,
                  init_theta: dict | None = None,
                  init_omega: dict | None = None,
                  route: str | None = None,
                  verbose: bool = True,
                  fix_params: set[str] | None = None,
                  prior_sd_per_param: dict[str, float] | None = None,
                  ) -> tuple[FitResult, list[FitResult]]:
    """Run `n_runs` perturbed-init NLME fits; return best-OFV converged fit.

    A base θ̂ (default: catalog typical, override with `init_theta`) is
    perturbed by exp(N(0, perturb)) for each run.  The first run uses the
    unperturbed init (i.e. the data-driven starting point if `init_theta`
    came from the initial-value decision agent), so a good naive-pooled
    fit gets a clean first chance before the perturbed exploration.

    `fix_params` and `prior_sd_per_param` are pass-through plan-edits
    from the orchestrator (e.g. Clinician-requested "fix ALAG=0" or
    "tighten Ka prior").
    """
    fits = []
    rng = np.random.default_rng(seed)
    for i in range(n_runs):
        cfg = build_config(drug_class, model_name,
                            covariate_effects={},
                            use_priors=True,
                            eta_params=eta_params,
                            compute_se=False,
                            init_theta_override=init_theta,
                            init_omega_override=init_omega,
                            route=route,
                            fix_params=fix_params,
                            prior_sd_per_param=prior_sd_per_param)
        if i > 0:  # leave first run un-perturbed = pure NCA / catalog init
            for p in list(cfg.init_theta):
                cfg.init_theta[p] = cfg.init_theta[p] * float(np.exp(rng.normal(0, perturb)))
        predict_fn = make_predict_fn(model_name)
        t0 = _time.time()
        try:
            fit = fit_population(subjects, cfg, predict_fn)
            fits.append(fit)
            if verbose:
                print(f"    ensemble run {i+1}/{n_runs}: "
                      f"OFV={fit.ofv:.2f} ({_time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            if verbose:
                print(f"    ensemble run {i+1}/{n_runs}: FAIL {e}", flush=True)
    if not fits:
        raise RuntimeError("all ensemble runs failed")
    fits.sort(key=lambda f: f.ofv)
    best = fits[0]
    for f in fits:
        if f.converged:
            best = f
            break
    return best, fits
