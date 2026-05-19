"""Numerical diagnostics for a PopPK fit — boundary LRT, summary stats, and
the deterministic acceptance/rejection rules.

The acceptance rules cite classical PopPK literature; they are NOT tuned
against the comparison numbers.

  - Boundary LRT for ω²=0      :  Stram & Lee 1994
  - Backward SCM α=0.01        :  Wählby, Jonsson & Karlsson 2002
  - n_obs/n_param ≥ 5          :  Bonate 2011 textbook §3.6
  - Shrinkage thresholds        :  Savic & Karlsson 2009 AAPS J 11(3):558-569
  - Ka > k_el identifiability   :  Pang & Rowland 1977; Bonate 2011 §4.3
  - 2-cmt V1/V2/Q physiology    :  Gibaldi & Perrier 1982 ch. 2;
                                    Rowland & Tozer 5th ed. ch. 19
  - NCA-derived V ≈ D/Cmax      :  Rowland & Tozer 5th ed. ch. 11
"""

from __future__ import annotations

import math

import numpy as np

from .engine import run_fit
from .nlme import FitResult, Subject
from .priors import CLASSES, get_class, physio_plausibility


OMEGA_FLOOR = 1e-3
LRT_CRIT_BOUNDARY = 2.706  # ½χ²₀ + ½χ²₁ at α = 0.05 (Stram & Lee 1994)


def fit_summary(fit: FitResult, drug_class: str | None,
                 cov_effects: dict | None,
                 eta_lrt: dict | None = None) -> dict:
    """JSON-serialisable summary of a FitResult used by evaluators and writers."""
    return {
        "drug_class": drug_class,
        "ofv": round(fit.ofv, 2),
        "aic": round(fit.aic, 2),
        "bic": round(fit.bic, 2),
        "theta": {k: round(v, 4) for k, v in fit.theta.items()},
        "omega_diag": {k: float(v) for k, v in fit.omega_diag.items()},
        "shrinkage_pct": {k: round(v, 1) for k, v in fit.shrinkage.items()},
        "sigma_prop": round(fit.sigma_prop, 4),
        "sigma_add": round(fit.sigma_add, 4),
        "n_subj": fit.n_subj, "n_obs": fit.n_obs, "n_param": fit.n_param,
        "converged": fit.converged,
        "covariate_betas": fit.covariate_effects,
        "selected_covariates": list(cov_effects.keys()) if cov_effects else [],
        "eta_lrt": eta_lrt or {},
    }


def compute_eta_boundary_lrt(drug_class: str | None, subjects: list[Subject],
                               model_name: str, eta_params: list[str],
                               full_fit: FitResult, cov_effects: dict) -> dict:
    """Boundary LRT statistic for every random effect.

    For each ETA k, refit the model with k removed from `eta_params`
    (i.e. ω²_k = 0 in the boundary sense) keeping every other parameter,
    covariate, and prior the same. The LRT statistic

        LRT_k = (−2·log L_reduced) − (−2·log L_full)

    is asymptotically distributed as a 50:50 mixture of χ²₀ and χ²₁ at the
    boundary (Stram & Lee 1994); the α = 0.05 critical value is 2.706.
    """
    out = {}
    for k in eta_params:
        reduced = [e for e in eta_params if e != k]
        if not reduced:
            out[k] = None
            continue
        try:
            warm_theta = dict(full_fit.theta)
            warm_omega = {kk: v for kk, v in full_fit.omega_diag.items() if kk in reduced}
            r = run_fit(drug_class, subjects, model_name,
                        covariate_effects=cov_effects, use_priors=True,
                        eta_params=reduced, compute_se=False,
                        init_theta_override=warm_theta,
                        init_omega_override=warm_omega)
            out[k] = float(r.ofv - full_fit.ofv)
        except Exception:
            out[k] = None
    return out


def statistical_acceptance(summary: dict) -> dict:
    """Pre-registered statistical acceptance criteria (deterministic rules).

    Acceptance requires the conjunction of:
      (i)   Optimiser converged (status flag + finite OFV).
      (ii)  For every ETA, boundary LRT > 2.706 OR ω² > 1.1×floor.
      (iii) n_obs/n_param ≥ 5 (Bonate 2011).
    """
    concerns, sugg = [], []
    dropped = set()
    eta_lrt = summary.get("eta_lrt", {})
    for k, om in summary["omega_diag"].items():
        lrt = eta_lrt.get(k)
        if lrt is not None and lrt < LRT_CRIT_BOUNDARY:
            concerns.append(
                f"boundary LRT for ω²[{k}] = {lrt:.2f} < critical 2.71 "
                f"(p > 0.05) → ETA not supported by the data")
            sugg.append(f"drop ETA on {k}")
            dropped.add(k)
            continue
        if om is not None and om <= 1.1 * OMEGA_FLOOR:
            concerns.append(
                f"ω²[{k}] = {om:.1e} at numerical floor → ETA collapsed")
            sugg.append(f"drop ETA on {k}")
            dropped.add(k)
    for k, v in summary["shrinkage_pct"].items():
        if k in dropped or v is None:
            continue
        if v > 50:
            concerns.append(
                f"shrinkage {v}% on η[{k}] — EBEs carry little subject-level "
                f"information (advisory, ETA kept)")
        elif v < -50:
            concerns.append(
                f"negative shrinkage on η[{k}] suggests ω² mis-estimated")
    if not summary["converged"]:
        concerns.append("optimiser did not converge")
        sugg.append("rerun with broader initial bracket")
    if summary["n_obs"] < 5 * summary["n_param"]:
        concerns.append(
            f"n_obs/n_param = {summary['n_obs']}/{summary['n_param']} < 5 "
            f"(Bonate 2011 minimum)")
    accept = (summary["converged"] and len(dropped) == 0
              and summary["n_obs"] >= 5 * summary["n_param"])
    return {"accept": accept, "concerns": concerns, "suggestions": sugg}


def data_anchors_from_subjects(subjects: list[Subject] | None) -> dict:
    """Compute data-only anchors for the clinical sanity check.

    Returns a dict with:
      cmax_geomean        : geometric mean of per-subject max(obs)
      v_apparent_geomean  : geometric mean of (total dose / max(obs))
      cl_apparent_geomean : geometric mean of (total dose / AUC)
      tmax_median         : median across subjects of arg-max obs time
      obs_window          : max observation time across the cohort
      n_obs_per_subject   : median observations per subject (TDM sparseness)
    Returns an empty dict if `subjects` is None or empty.
    """
    if not subjects:
        return {}
    cmaxes = []
    vs = []
    cls = []
    tmaxes = []
    obs_t_max = 0.0
    n_obs_each = []
    for s in subjects:
        obs = np.asarray(s.obs, dtype=float)
        t = np.asarray(s.time, dtype=float)
        if len(obs) == 0:
            continue
        n_obs_each.append(len(obs))
        cm = float(np.max(obs))
        if cm > 0:
            cmaxes.append(cm)
        hist = getattr(s, "dose_history", None)
        if hist:
            # Multi-dose: at steady state, single-dose-equivalent V is
            # estimated from the per-dose amount (median of the dose
            # history), not the total cumulative dose.  Using total dose
            # / Cmax inflates V by the number-of-doses factor, which
            # would otherwise trigger spurious NCA-mismatch warnings on
            # TDM data (Rowland & Tozer 5th ed. §10.5).
            per_dose = float(np.median([amt for _, amt in hist]))
            v_dose = per_dose
            total_dose = float(sum(amt for _, amt in hist))
        else:
            v_dose = float(s.dose)
            total_dose = float(s.dose)
        if cm > 0:
            vs.append(v_dose / cm)
        if len(t) >= 2:
            order = np.argsort(t)
            auc = float(np.trapz(obs[order], t[order]))
            if auc > 0:
                cls.append(total_dose / auc)
            tmaxes.append(float(t[int(np.argmax(obs))]))
        obs_t_max = max(obs_t_max, float(np.max(t)))

    def _gmean(arr):
        a = np.array([x for x in arr if x > 0 and np.isfinite(x)])
        if len(a) == 0:
            return float("nan")
        return float(math.exp(float(np.mean(np.log(a)))))

    return {
        "cmax_geomean":        _gmean(cmaxes),
        "v_apparent_geomean":  _gmean(vs),
        "cl_apparent_geomean": _gmean(cls),
        "tmax_median":         float(np.median(tmaxes)) if tmaxes else float("nan"),
        "obs_window":          obs_t_max,
        "n_obs_per_subject":   float(np.median(n_obs_each)) if n_obs_each else 0.0,
    }


# NOTE: clinical_acceptance (rule-based deterministic Clinician) was
# removed. The Clinician is now LLM-only and lives in
# decisions/evaluator.py; the data_anchors_from_subjects helper
# above feeds NCA Cmax/AUC/tmax/window/sparseness numbers verbatim
# to the LLM, which applies textbook PK constraints (Pang & Rowland
# 1977; Gibaldi & Perrier 1982; Rowland & Tozer 5th ed.; etc.).
# No silent fallback exists; missing API key raises in the evaluator.

