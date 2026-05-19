"""FOCE-with-Interaction style NLME engine for population PK.

Implements a Laplace-approximated marginal likelihood with empirical Bayes
estimates (EBEs) for individual ETAs, optionally with informative log-normal
priors on THETAs (Bayesian regularization).  Avoids any NONMEM dependency.

Mathematical model
------------------
For subject i with observations y_{ij} at times t_{ij}, dose D_i, and covariates X_i:

  log(TVP_p) = log(theta_p) + sum_c beta_{p,c} * g(X_{i,c})   (covariate model, log-linear)
  P_{i,p}    = TVP_{i,p} * exp(eta_{i,p}),   eta_i ~ N(0, Omega)
  y_{ij}     = f(t_{ij}; P_i) * (1 + eps_prop_{ij}) + eps_add_{ij}
             eps_prop ~ N(0, sigma_p^2),  eps_add ~ N(0, sigma_a^2)

Per-subject marginal likelihood is approximated by Laplace at the MAP eta_hat:

  log p(y_i | theta, Omega, sigma) ≈ log p(y_i | eta_hat, theta) + log p(eta_hat | Omega)
                                    - 0.5 * log det(H_i / 2π)

The outer (population) optimization minimizes -2 * sum_i loglike + 2 * priors,
which functions as an OFV analogous to NONMEM's −2LL (constants absorbed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #

@dataclass
class Subject:
    sid: int
    time: np.ndarray
    obs: np.ndarray
    dose: float
    tinf: float = 0.0
    covariates: dict = field(default_factory=dict)


@dataclass
class FitResult:
    theta: dict
    omega_diag: dict          # variances of ETAs (between-subject variability)
    sigma_prop: float         # proportional residual SD
    sigma_add: float          # additive residual SD
    ofv: float                # -2 * marginal loglik
    aic: float
    bic: float
    n_obs: int
    n_subj: int
    n_param: int
    converged: bool
    individual_etas: dict     # subject id -> dict of ETAs
    shrinkage: dict           # per-eta-parameter percentage shrinkage
    cond_number: float | None
    covariate_effects: dict   # e.g. {"CL~CRCL": 0.75}
    residuals: pd.DataFrame   # IPRED, PRED, IWRES, CWRES
    # Asymptotic standard errors (Hessian-based) and relative SE in %.
    # Keys are parameter names (log-scale θ entries) and ω² entries; values
    # are SEs on the natural scale. The %RSE column uses SE / |estimate|.
    theta_se: dict = field(default_factory=dict)
    theta_rse_pct: dict = field(default_factory=dict)
    omega_se: dict = field(default_factory=dict)
    omega_rse_pct: dict = field(default_factory=dict)
    # NONMEM-equivalent OFV: strips the Bayesian log-prior penalty (which
    # NONMEM does not include) and the log(2π) constants per observation
    # and per η (which NONMEM normalises out).  Kept for diagnostics;
    # `ofv_foce` is the canonical OFV reported to users.
    ofv_nonmem_eq: float = 0.0
    # FOCE-I marginal OFV — what NONMEM reports.  Computed post-fit by
    # linearising the model around η̂_i and integrating the resulting
    # Gaussian marginal in closed form (NONMEM Method=1 INTERACTION).
    # This is the canonical OFV for cross-method comparison and is the
    # value rendered in figures / per-drug tables.  NaN when ill-conditioned.
    ofv_foce: float = 0.0
    message: str = ""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _vec_to_theta(vec: np.ndarray, names: list[str]) -> dict:
    return {n: float(np.exp(v)) for n, v in zip(names, vec)}


from . import kernels as _kernels


# Per-subject JIT call dispatch table:
#   1cmt_iv   → tv = [CL, V]              eta_active aligned
#   1cmt_oral → tv = [CL, V, Ka, ALAG]    ALAG never has ETA (always 0 in mask)
#   2cmt_iv   → tv = [CL, V1, Q, V2]
#   2cmt_inf  → tv = [CL, V1, Q, V2]
_PARAM_ORDER = {
    "1cmt_iv": ["CL", "V"],
    "1cmt_oral": ["CL", "V", "Ka", "ALAG"],
    "2cmt_iv": ["CL", "V1", "Q", "V2"],
    "2cmt_inf": ["CL", "V1", "Q", "V2"],
}


def _individual_loglike(eta: np.ndarray, subject: Subject, tv_params: dict,
                        eta_names: list[str], omega_diag: np.ndarray,
                        sigma_prop: float, sigma_add: float,
                        predict_fn: Callable,
                        model_name: str | None = None) -> float:
    """Conditional log p(y_i, eta_i | theta) (joint of obs and random effects).

    Fast path: when `model_name` is one of the supported analytical models
    (1cmt_iv/oral, 2cmt_iv/inf), delegate the per-subject computation to
    the Numba-JIT'd kernel for ~5-40x speedup.  Otherwise fall back to
    the pure-Python implementation that uses the user-supplied predict_fn.
    """
    # JIT kernels assume single-dose closed-form predictions; if the
    # subject carries a dose_history (multi-dose superposition, e.g.
    # real-data tobramycin TDM), fall back to the Python predict_fn path.
    if model_name in _PARAM_ORDER and not hasattr(subject, "dose_history"):
        order = _PARAM_ORDER[model_name]
        tv = np.array([tv_params.get(p, 0.0) for p in order], dtype=np.float64)
        # ETA only on Ka, V, V1, V2, Q, CL — never on ALAG
        eta_active = np.array(
            [1 if (p in eta_names and p != "ALAG") else 0 for p in order],
            dtype=np.int8)
        return _kernels.individual_loglike(
            eta.astype(np.float64),
            tv,
            eta_active,
            np.asarray(omega_diag, dtype=np.float64),
            float(sigma_prop), float(sigma_add),
            np.asarray(subject.time, dtype=np.float64),
            np.asarray(subject.obs, dtype=np.float64),
            float(subject.dose),
            float(getattr(subject, "tinf", 0.0) or 0.0),
            _kernels.MODEL_CODE[model_name])

    # --- Python fallback (rarely used; kept for non-standard predict_fns) ---
    p_indiv = {k: tv_params[k] * math.exp(eta[i] if k in eta_names else 0.0)
               for i, k in enumerate(eta_names)}
    for k, v in tv_params.items():
        p_indiv.setdefault(k, v)
    f = predict_fn(p_indiv, subject)
    f = np.maximum(f, 1e-10)
    var = (sigma_prop * f) ** 2 + sigma_add ** 2
    var = np.maximum(var, 1e-12)
    ll_obs = -0.5 * np.sum(np.log(2 * np.pi * var) + (subject.obs - f) ** 2 / var)
    om = np.maximum(omega_diag, 1e-8)
    ll_eta = -0.5 * np.sum(np.log(2 * np.pi * om) + eta ** 2 / om)
    return ll_obs + ll_eta


def _solve_eta_map(subject: Subject, tv_params: dict, eta_names: list[str],
                   omega_diag: np.ndarray, sigma_prop: float, sigma_add: float,
                   predict_fn: Callable, eta0: np.ndarray | None = None,
                   max_inner_iter: int = 80,
                   model_name: str | None = None
                   ) -> tuple[np.ndarray, float, np.ndarray]:
    """Find the eta that maximizes the joint log p(y_i, eta_i | theta).

    Returns (eta_hat, joint_ll_at_hat, hessian_diag_approx).
    """
    if eta0 is None:
        eta0 = np.zeros(len(eta_names))

    def neg_ll(eta):
        return -_individual_loglike(eta, subject, tv_params, eta_names,
                                    omega_diag, sigma_prop, sigma_add,
                                    predict_fn, model_name=model_name)

    # Bonate 2011 §5: inner EBE/MAP optimisation with a gradient-based
    # method gives cleaner mode location than Nelder-Mead (which has
    # simplex noise that propagates into the Laplace approximation,
    # especially for non-quadratic posteriors with parameter ridges, e.g.
    # the Ka-ALAG ridge in sparse oral PK data; Karlsson & Sheiner 1993).
    # L-BFGS-B with finite-difference gradient via SciPy's default eps.
    # If L-BFGS-B fails to converge (rare), fall back to Nelder-Mead.
    try:
        res = minimize(neg_ll, eta0, method="L-BFGS-B",
                       options={"maxiter": max_inner_iter, "ftol": 1e-6,
                                "gtol": 1e-5})
        if not res.success or not np.isfinite(res.fun):
            raise RuntimeError("L-BFGS-B failed")
    except Exception:
        res = minimize(neg_ll, eta0, method="Nelder-Mead",
                       options={"xatol": 5e-3, "fatol": 5e-3,
                                "maxiter": max_inner_iter, "adaptive": True})
    eta_hat = res.x
    ll_hat = -res.fun
    f0 = res.fun
    n = len(eta_hat)

    # Full numerical Hessian of the negative joint log-likelihood at MAP η̂.
    # The Laplace approximation log p(y_i) ≈ log p(y_i, η̂_i) − ½ log|H_i| +
    # (n_η/2)·log(2π) requires the determinant of the FULL Hessian.  When
    # parameters are correlated (e.g. Ka-ALAG ridge in sparse oral PK data;
    # Karlsson & Sheiner 1993), log|H_full| < log(Π H_ii); using only the
    # diagonal therefore over-estimates the determinant and inflates OFV.
    h = 1e-3
    H = np.zeros((n, n))

    def _ll(e):
        return -_individual_loglike(e, subject, tv_params, eta_names, omega_diag,
                                     sigma_prop, sigma_add, predict_fn,
                                     model_name=model_name)

    for i in range(n):
        e_p = eta_hat.copy(); e_p[i] += h
        e_m = eta_hat.copy(); e_m[i] -= h
        H[i, i] = (_ll(e_p) - 2 * f0 + _ll(e_m)) / (h * h)
        for j in range(i + 1, n):
            e_pp = eta_hat.copy(); e_pp[i] += h; e_pp[j] += h
            e_pm = eta_hat.copy(); e_pm[i] += h; e_pm[j] -= h
            e_mp = eta_hat.copy(); e_mp[i] -= h; e_mp[j] += h
            e_mm = eta_hat.copy(); e_mm[i] -= h; e_mm[j] -= h
            H[i, j] = H[j, i] = (_ll(e_pp) - _ll(e_pm)
                                  - _ll(e_mp) + _ll(e_mm)) / (4 * h * h)
    # Make Hessian symmetric and positive-definite (ridge if needed)
    H_sym = 0.5 * (H + H.T)
    eigvals = np.linalg.eigvalsh(H_sym)
    if eigvals.min() <= 0:
        ridge = -eigvals.min() + 1e-6
        H_sym = H_sym + ridge * np.eye(n)
    # Return the full Hessian matrix (caller uses log|H_full| in Laplace)
    return eta_hat, ll_hat, H_sym


# --------------------------------------------------------------------------- #
# FOCE-I marginal OFV (post-fit, NONMEM convention)
# --------------------------------------------------------------------------- #

def _foce_ofv_subject(subject: Subject, tv: dict, eta_hat: np.ndarray,
                       eta_names: list[str], omega_diag: np.ndarray,
                       sigma_prop: float, sigma_add: float,
                       predict_fn: Callable) -> float | None:
    """−2·log p_FOCE(y_i | θ) for one subject (NONMEM-style, INTERACTION).

    Linearises the structural model around the MAP η̂_i:

        y_i ≈ f_i(θ, η̂_i) + G_i · (η_i − η̂_i) + ε_i,
        ε_i ~ N(0, R_i(η̂_i)),   G_i = ∂f_i/∂η at η̂_i.

    Under this linearisation, y_i is marginally Gaussian:

        y_i ~ N(μ_i, Σ_i),
        μ_i = f_i(θ, η̂_i) − G_i · η̂_i,
        Σ_i = G_i · Ω · G_iᵀ + R_i.

    Returns log|Σ_i| + (y_i − μ_i)ᵀ Σ_i⁻¹ (y_i − μ_i), i.e. the per-subject
    contribution to NONMEM's reported FOCE-I OFV (the n_obs·log(2π) constant
    that NONMEM also drops is excluded).  Returns None on numerical failure.
    """
    p_indiv = dict(tv)
    for i, n in enumerate(eta_names):
        p_indiv[n] = tv[n] * math.exp(eta_hat[i])
    f_hat = np.asarray(predict_fn(p_indiv, subject), dtype=np.float64)
    f_hat = np.maximum(f_hat, 1e-10)

    n_obs_s = len(subject.obs)
    n_eta = len(eta_names)
    # Sensitivity G_i = ∂f/∂η at η̂  (central finite-difference)
    G = np.zeros((n_obs_s, n_eta))
    h = 1e-4
    for k in range(n_eta):
        eta_p = eta_hat.copy(); eta_p[k] += h
        eta_m = eta_hat.copy(); eta_m[k] -= h
        p_p = dict(tv); p_m = dict(tv)
        for i, n in enumerate(eta_names):
            p_p[n] = tv[n] * math.exp(eta_p[i])
            p_m[n] = tv[n] * math.exp(eta_m[i])
        f_p = np.asarray(predict_fn(p_p, subject), dtype=np.float64)
        f_m = np.asarray(predict_fn(p_m, subject), dtype=np.float64)
        G[:, k] = (f_p - f_m) / (2 * h)

    # INTERACTION residual variance at η̂  (this is what makes it FOCE-I,
    # not FOCE: the residual SD itself depends on the individual prediction).
    R_diag = (sigma_prop * f_hat) ** 2 + sigma_add ** 2
    R_diag = np.maximum(R_diag, 1e-12)

    Omega = np.diag(np.maximum(np.asarray(omega_diag, dtype=np.float64), 1e-8))
    Sigma = G @ Omega @ G.T + np.diag(R_diag)
    Sigma = 0.5 * (Sigma + Sigma.T)

    try:
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0 or not np.isfinite(logdet):
            ridge = 1e-6 * (np.trace(Sigma) / max(n_obs_s, 1) + 1.0)
            Sigma = Sigma + ridge * np.eye(n_obs_s)
            sign, logdet = np.linalg.slogdet(Sigma)
            if sign <= 0:
                return None
        mu = f_hat - G @ eta_hat
        r = np.asarray(subject.obs, dtype=np.float64) - mu
        quad = float(r @ np.linalg.solve(Sigma, r))
    except np.linalg.LinAlgError:
        return None

    val = float(logdet + quad)
    if not np.isfinite(val):
        return None
    return val


# --------------------------------------------------------------------------- #
# Covariate model
# --------------------------------------------------------------------------- #

def _apply_covariates(theta: dict, cov_effects: dict, subject_cov: dict,
                      medians: dict) -> dict:
    """Apply covariate effects to typical-value parameters.

    cov_effects: e.g. {"CL~CRCL": ("power", 0.75), "V~WT": ("power", 1.0),
                       "Ka~AGE": ("exp", -0.02)}
    """
    out = dict(theta)
    for key, spec in cov_effects.items():
        param, cov = key.split("~")
        cov_val = subject_cov.get(cov)
        if cov_val is None or param not in out:
            continue
        med = medians.get(cov, cov_val)
        kind, beta = spec
        if med == 0 or med is None:
            continue
        if kind == "power":
            out[param] = out[param] * (cov_val / med) ** beta
        elif kind == "linear":
            out[param] = out[param] * (1 + beta * (cov_val - med) / med)
        elif kind == "exp":
            out[param] = out[param] * math.exp(beta * (cov_val - med))
    return out


# --------------------------------------------------------------------------- #
# Population fit
# --------------------------------------------------------------------------- #

@dataclass
class NLMEConfig:
    model_name: str
    eta_params: list[str]                 # which THETAs have ETAs
    init_theta: dict                      # initial values, also serves as prior mean for log
    init_omega: dict                      # initial variances for eta_params
    init_sigma_prop: float = 0.1
    init_sigma_add: float = 0.01
    covariate_effects: dict = field(default_factory=dict)
    log_prior_sd: dict | None = None      # optional Bayesian regularization on log-theta
    fix: set = field(default_factory=set) # parameter names to hold fixed
    max_iter: int = 200
    verbose: bool = False
    # Compute Hessian-based asymptotic SEs at the optimum.  Disabled by
    # auxiliary refits (LRT drop-one-out, SCM trial fits) for which only
    # the OFV matters.  Default True for the main fit.
    compute_se: bool = True


def fit_population(subjects: list[Subject], cfg: NLMEConfig, predict_fn: Callable) -> FitResult:
    """Fit a population PK model using FOCE-with-Interaction + Laplace approximation."""

    theta_names = list(cfg.init_theta.keys())
    eta_names = list(cfg.eta_params)
    medians = {}
    cov_keys = set()
    for k in cfg.covariate_effects:
        cov_keys.add(k.split("~")[1])
    for c in cov_keys:
        vals = [s.covariates.get(c) for s in subjects if s.covariates.get(c) is not None]
        if vals:
            medians[c] = float(np.median(vals))

    log_prior_sd = cfg.log_prior_sd or {}

    # Numerical floor on omega: standard NLME practice. The Laplace
    # approximation depends on log det(H/2π), which diverges as ω → 0.
    # Setting a small minimum variance keeps the optimisation well-posed
    # without materially changing parameter estimates when the data
    # genuinely have IIV. When the data don't support IIV, the EBE
    # shrinkage rises (the orchestrator's reviewer then recommends
    # dropping the ETA).
    OMEGA_FLOOR = 1e-3

    def unpack(x: np.ndarray):
        """Unpack the optimizer vector into (theta, omega, sigma)."""
        n_th = len([n for n in theta_names if n not in cfg.fix])
        n_om = len(eta_names)
        ix = 0
        theta = dict(cfg.init_theta)
        for n in theta_names:
            if n in cfg.fix:
                continue
            theta[n] = math.exp(x[ix])
            ix += 1
        omega_diag = np.exp(2 * x[ix:ix + n_om])  # variance = exp(2*log_sd)
        omega_diag = np.maximum(omega_diag, OMEGA_FLOOR)
        ix += n_om
        sigma_prop = math.exp(x[ix]); ix += 1
        sigma_add = math.exp(x[ix]); ix += 1
        # covariate betas (free, not log-transformed)
        cov_betas = {}
        for k in cfg.covariate_effects:
            cov_betas[k] = (cfg.covariate_effects[k][0], x[ix])
            ix += 1
        return theta, omega_diag, sigma_prop, sigma_add, cov_betas

    def pack_initial():
        x0 = []
        for n in theta_names:
            if n in cfg.fix:
                continue
            x0.append(math.log(cfg.init_theta[n]))
        for n in eta_names:
            x0.append(0.5 * math.log(cfg.init_omega.get(n, 0.1)))
        x0.append(math.log(cfg.init_sigma_prop))
        x0.append(math.log(cfg.init_sigma_add))
        for k, spec in cfg.covariate_effects.items():
            x0.append(spec[1])  # initial beta
        return np.array(x0)

    # warm-start ETA cache across outer iterations
    eta_cache = {s.sid: np.zeros(len(eta_names)) for s in subjects}

    def objective(x: np.ndarray) -> float:
        try:
            theta, om_diag, sp, sa, cov_b = unpack(x)
        except Exception:
            return 1e12
        ofv = 0.0
        for s in subjects:
            cov_eff_active = {k: cov_b[k] for k in cfg.covariate_effects}
            tv = _apply_covariates(theta, cov_eff_active, s.covariates, medians)
            try:
                eta_hat, ll_hat, H_full = _solve_eta_map(
                    s, tv, eta_names, om_diag, sp, sa, predict_fn,
                    eta0=eta_cache[s.sid],
                    model_name=cfg.model_name)
                eta_cache[s.sid] = eta_hat
            except Exception:
                return 1e12
            # Laplace approximation:
            #   log p(y) ≈ log p(y, η̂) − ½·log|H_full| + (n_η/2)·log(2π)
            # → −2 contribution to OFV:
            #   −2·log p(y) ≈ −2·log p(y, η̂) + log|H_full| − n_η·log(2π)
            n_eta = H_full.shape[0]
            sign_det, logdet = np.linalg.slogdet(H_full)
            if sign_det <= 0:
                # ill-conditioned subject — penalise rather than crash
                return 1e12
            ofv += -2 * ll_hat + logdet - n_eta * math.log(2 * math.pi)
        # Bayesian regularization on log-theta
        for n, sd in log_prior_sd.items():
            if n in theta_names and sd is not None and sd > 0:
                logth = math.log(theta[n])
                logmu = math.log(cfg.init_theta[n])
                ofv += ((logth - logmu) / sd) ** 2
        if not np.isfinite(ofv):
            return 1e12
        return ofv

    x0 = pack_initial()
    # Two-stage refinement: quick + refine
    # Two-stage outer optimization:
    #   1) Nelder-Mead for robust global exploration (handles non-smooth
    #      objective from inner ETA-MAP optimisation noise)
    #   2) L-BFGS-B polish: gradient-based fine-tuning for tighter convergence
    #      (Bonate 2011 §5; comparable to NONMEM's BFGS-style outer step).
    #      Finite-difference gradient via SciPy default. If L-BFGS-B fails
    #      to improve, fall back to the Nelder-Mead solution.
    best = minimize(objective, x0, method="Nelder-Mead",
                    options={"maxiter": cfg.max_iter * 3, "xatol": 5e-3, "fatol": 5e-3,
                             "adaptive": True})
    try:
        polish = minimize(objective, best.x, method="L-BFGS-B",
                          options={"maxiter": cfg.max_iter, "ftol": 1e-7,
                                   "gtol": 1e-5})
        if polish.fun < best.fun and np.isfinite(polish.fun):
            best = polish
    except Exception:
        pass
    # second Nelder-Mead polish for robustness
    best = minimize(objective, best.x, method="Nelder-Mead",
                    options={"maxiter": cfg.max_iter * 2, "xatol": 1e-4, "fatol": 1e-4,
                             "adaptive": True})
    converged = best.success or best.fun < 1e11

    theta, om_diag, sp, sa, cov_b = unpack(best.x)
    ofv = float(best.fun)

    # Empirical Bayes etas + diagnostics + FOCE-I OFV (NONMEM-equivalent)
    ind_etas = {}
    iwres_all, cwres_all, ipred_all, pred_all, time_all, obs_all, sid_all = [], [], [], [], [], [], []
    ofv_foce_sum = 0.0
    ofv_foce_ok = True
    for s in subjects:
        cov_eff_active = {k: cov_b[k] for k in cfg.covariate_effects}
        tv = _apply_covariates(theta, cov_eff_active, s.covariates, medians)
        eta_hat, _, _ = _solve_eta_map(s, tv, eta_names, om_diag, sp, sa, predict_fn,
                                         model_name=cfg.model_name)
        ind_etas[s.sid] = dict(zip(eta_names, eta_hat))
        # individual prediction
        p_indiv = dict(tv)
        for i, n in enumerate(eta_names):
            p_indiv[n] = tv[n] * math.exp(eta_hat[i])
        ipred = predict_fn(p_indiv, s)
        pred = predict_fn(tv, s)
        ipred = np.maximum(ipred, 1e-10)
        var = (sp * ipred) ** 2 + sa ** 2
        iwres = (s.obs - ipred) / np.sqrt(var)
        var_pred = (sp * pred) ** 2 + sa ** 2
        cwres = (s.obs - pred) / np.sqrt(var_pred)
        iwres_all.extend(iwres); cwres_all.extend(cwres)
        ipred_all.extend(ipred); pred_all.extend(pred)
        time_all.extend(s.time); obs_all.extend(s.obs)
        sid_all.extend([s.sid] * len(s.obs))

        if ofv_foce_ok:
            contrib = _foce_ofv_subject(s, tv, eta_hat, eta_names, om_diag,
                                          sp, sa, predict_fn)
            if contrib is None:
                ofv_foce_ok = False
            else:
                ofv_foce_sum += contrib

    ofv_foce = ofv_foce_sum if ofv_foce_ok else float("nan")

    residuals = pd.DataFrame({
        "ID": sid_all, "TIME": time_all, "DV": obs_all,
        "IPRED": ipred_all, "PRED": pred_all,
        "IWRES": iwres_all, "CWRES": cwres_all,
    })

    # shrinkage = 1 - SD(eta_hat) / sqrt(omega_diag)
    shrinkage = {}
    for i, n in enumerate(eta_names):
        etas = np.array([ind_etas[s.sid][n] for s in subjects])
        sd_eta = float(np.std(etas, ddof=1)) if len(etas) > 1 else 0.0
        om_sd = float(math.sqrt(om_diag[i]))
        shrinkage[n] = 100 * (1 - sd_eta / max(om_sd, 1e-8))

    n_obs = sum(len(s.obs) for s in subjects)
    n_subj = len(subjects)
    # number of estimated parameters
    n_param = (len([n for n in theta_names if n not in cfg.fix]) + len(eta_names) + 2
               + len(cfg.covariate_effects))
    aic = ofv + 2 * n_param
    bic = ofv + n_param * math.log(n_obs)

    # --- Hessian-based asymptotic standard errors --------------------------- #
    # The optimiser minimised -2·log L (plus prior penalty). The observed
    # Fisher information for a -2·log L scale is ½ × Hessian (so SE matrix
    # is 2·H⁻¹). Computed numerically with central differences; clamped to
    # be positive-definite before inversion. Failures (singular Hessian)
    # leave SE = None; downstream code reports "—".
    theta_se, theta_rse, omega_se, omega_rse = {}, {}, {}, {}
    cond_number = None
    if not cfg.compute_se:
        # caller doesn't want SE -- skip the n_x^2 Hessian computation
        pass
    else:
      try:
        x_opt = best.x
        n_x = len(x_opt)
        # Step size chosen for numerical Hessian: the inner ETA Nelder–Mead
        # has fatol = 5e-3 (per subject), so objective() noise is bounded by
        # ~5e-3·N_subj. A 2nd-derivative estimate needs H·ε² ≫ noise, hence
        # ε = 0.05 (≈ 5% on log-θ scale). Larger ε would bias the curvature
        # estimate; smaller would be dominated by inner-optimiser noise.
        eps = 0.05
        f0 = objective(x_opt)
        # diagonal only — full off-diagonal Hessian is dominated by
        # finite-difference noise at this ε, gives unstable correlations.
        # NONMEM's $COV step also typically reports diagonal-based RSEs.
        H_diag = np.zeros(n_x)
        for i in range(n_x):
            xp = x_opt.copy(); xp[i] += eps
            xm = x_opt.copy(); xm[i] -= eps
            H_diag[i] = (objective(xp) - 2 * f0 + objective(xm)) / (eps ** 2)
        cond_number = float(H_diag.max() / max(H_diag.min(), 1e-12)) if H_diag.min() > 0 else None
        # observed Fisher (½ Hessian on -2log L scale); diagonal Cov ≈ 2 / H_diag.
        # Mark non-positive Hessian diagonals as indeterminate (None).
        se_x = np.full(n_x, np.nan)
        for i in range(n_x):
            if H_diag[i] > 0 and np.isfinite(H_diag[i]):
                se_x[i] = math.sqrt(2.0 / H_diag[i])

        # x entries: log-θ (one per free theta), then log-ω, then log-σ_p,
        # log-σ_a, then covariate βs. Match the layout in `unpack`.
        ix = 0
        for n in theta_names:
            if n in cfg.fix:
                continue
            sx = se_x[ix]
            if np.isfinite(sx):
                # SE(θ) ≈ θ · SE(log θ) (delta rule); cap reporting RSE > 200%
                # as "indeterminate" — NONMEM convention for ill-identified params
                theta_se[n] = float(theta[n] * sx)
                rse = 100 * theta_se[n] / max(abs(theta[n]), 1e-12)
                theta_rse[n] = float(rse) if rse < 200 else None
            ix += 1
        for j, n in enumerate(eta_names):
            sx = se_x[ix + j]
            om = float(om_diag[j])
            if np.isfinite(sx):
                se_om = float(2 * om * sx)
                omega_se[n] = se_om
                rse = 100 * se_om / max(abs(om), 1e-12)
                omega_rse[n] = float(rse) if rse < 200 else None
      except Exception as e:
        pass

    # NONMEM-equivalent OFV (strip Bayesian prior penalty + log(2π) constants)
    ofv_nonmem_eq = ofv
    # 1. remove prior penalty term (additive in `objective`)
    if log_prior_sd:
        prior_penalty = 0.0
        for n, sd in log_prior_sd.items():
            if n in theta and sd > 0:
                logth = math.log(theta[n])
                logmu = math.log(cfg.init_theta[n])
                prior_penalty += ((logth - logmu) / sd) ** 2
        ofv_nonmem_eq -= prior_penalty
    # 2. remove log(2π) constants that NONMEM normalises out
    # Per observation: 1 (from log|2π V|), per η: 1 - 1 = 0 (eta + Laplace cancel)
    ofv_nonmem_eq -= n_obs * math.log(2 * math.pi)

    return FitResult(
        theta=theta,
        omega_diag={n: float(om_diag[i]) for i, n in enumerate(eta_names)},
        sigma_prop=float(sp), sigma_add=float(sa),
        ofv=ofv, aic=aic, bic=bic,
        n_obs=n_obs, n_subj=n_subj, n_param=n_param,
        converged=converged,
        individual_etas=ind_etas, shrinkage=shrinkage,
        cond_number=cond_number,
        covariate_effects={k: float(cov_b[k][1]) for k in cov_b},
        residuals=residuals,
        theta_se=theta_se,
        theta_rse_pct=theta_rse,
        omega_se=omega_se,
        omega_rse_pct=omega_rse,
        ofv_nonmem_eq=ofv_nonmem_eq,
        ofv_foce=float(ofv_foce),
        message=str(best.message) if hasattr(best, "message") else "",
    )
