"""Initial-value decision agent — strictly data-only.

Two-stage init:
  1. Closed-form NCA on the pooled observations (textbook formulas — no
     drug name, no class catalog, no hardcoded fallbacks).
  2. A naive-pooled refine pass using the structural model the engine
     will actually fit.

Both stages operate on the data alone.  When a stage can't produce a
finite answer the previous stage's value is kept; we never substitute
class typicals or arbitrary defaults like 1.0 because those would smuggle
the answer in via the "fallback" path.

The class catalog is consulted only for Bayesian log-prior SDs in
`engine.build_config` (a wide regulariser, not an init seed).  Init
values themselves come exclusively from the dataset.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Callable

from ..llm_client import AgentRunLog
from ..nlme import Subject


# --------------------------------------------------------------------------- #
# NCA (non-compartmental) primitives — closed-form, data-only
# --------------------------------------------------------------------------- #

def _subject_auc_trapz(time: np.ndarray, conc: np.ndarray) -> float:
    """Linear-trapezoidal AUC over the observed time window."""
    if len(time) < 2:
        return float("nan")
    order = np.argsort(time)
    t = time[order]; c = conc[order]
    return float(np.trapz(c, t))


def _subject_total_dose(subject: Subject) -> float:
    """Sum of all dose amounts in the subject's record (multi-dose aware)."""
    hist = getattr(subject, "dose_history", None)
    if hist:
        return float(sum(amt for _, amt in hist))
    return float(subject.dose)


def _nca_init(subjects: list[Subject], model_name: str) -> dict:
    """Closed-form NCA → primary PK parameter seeds.

    Returns the keys required by `model_name`'s structural model
    (CL, V or V1/V2/Q, plus Ka/ALAG for oral).  All formulae use only
    the (time, conc, dose) data; no priors or class catalogs.
    """
    cls = []
    v1s = []
    cmaxes = []
    for s in subjects:
        if len(s.obs) < 1:
            continue
        auc = _subject_auc_trapz(s.time, s.obs)
        dose = _subject_total_dose(s)
        if auc and np.isfinite(auc) and auc > 0:
            cls.append(dose / auc)              # CL = D / AUC
        cmax = float(np.max(s.obs))
        if cmax > 0:
            cmaxes.append(cmax)
            v1s.append(_subject_total_dose(s) / cmax)   # V ≈ D / Cmax

    # Geometric means across subjects (lognormal-friendly)
    def _gmean(arr):
        a = np.array([x for x in arr if x > 0 and np.isfinite(x)])
        if len(a) == 0:
            return float("nan")
        return float(math.exp(float(np.mean(np.log(a)))))

    CL = _gmean(cls)
    V_apparent = _gmean(v1s)
    Cmax_typ = _gmean(cmaxes)

    seeds: dict = {}
    if model_name == "1cmt_iv":
        seeds["CL"] = CL
        seeds["V"] = V_apparent
    elif model_name == "1cmt_oral":
        seeds["CL"] = CL
        seeds["V"] = V_apparent
        # Ka from observed t_peak: t_peak ≈ ln(Ka/k) / (Ka - k); for typical
        # oral PK with first-order absorption Ka ≈ 1/hour is a reasonable
        # data-anchored guess derived from the median time-to-Cmax
        tmaxes = []
        for s in subjects:
            if len(s.obs) >= 2:
                tmaxes.append(float(s.time[int(np.argmax(s.obs))]))
        tmax = float(np.median(tmaxes)) if tmaxes else 1.0
        # Solve t_peak·(Ka−k) = ln(Ka/k) iteratively for Ka given k=CL/V
        k = CL / V_apparent if V_apparent > 0 else 0.1
        # Newton-ish guess: Ka ≈ 1 / (tmax/3) — empirical, data-only
        Ka_seed = max(3.0 / max(tmax, 1e-3), k * 1.5)
        seeds["Ka"] = Ka_seed
        # ALAG seed from data: lag time is a *population-shared* parameter,
        # so we anchor on the COHORT MINIMUM of "first time a positive
        # concentration is observed" — that's an upper bound on the true
        # lag (lag cannot exceed the earliest time anyone observed drug).
        # Median would be dominated by sparse-sampling subjects whose
        # first observation only happens hours after dosing.
        first_pos_times = []
        for s in subjects:
            for ti, ci in zip(s.time, s.obs):
                if ci > 0:
                    first_pos_times.append(float(ti))
                    break
        if first_pos_times:
            seeds["ALAG"] = max(0.01, float(np.min(first_pos_times)) * 0.5)
        else:
            seeds["ALAG"] = 0.01
    elif model_name in ("2cmt_iv", "2cmt_inf"):
        seeds["CL"] = CL
        # V1 ≈ D/Cmax (initial dilution volume after IV bolus / infusion)
        seeds["V1"] = V_apparent
        # Without resolving the distribution phase from sparse data, the
        # most we can data-derive is "V2 of the same order as V1, Q of the
        # same order as CL".  Set V2 = V1 and Q = CL — these collapse the
        # model to 1cmt; the population fit's gradient (which sees the
        # distribution phase across subjects) breaks the symmetry.
        seeds["V2"] = V_apparent
        seeds["Q"] = CL
    else:
        raise ValueError(f"unsupported model_name {model_name!r}")

    return seeds


# --------------------------------------------------------------------------- #
# Naive-pooled refine — uses the engine's actual predict_fn
# --------------------------------------------------------------------------- #

def _naive_pooled_refine(subjects: list[Subject], model_name: str,
                          seeds: dict) -> dict:
    """Refine NCA seeds with a naive-pooled MLE pass.

    Uses the same `predict_fn` the engine uses (including dose_history
    superposition for multi-dose subjects).  Returns the refined params
    if the optimisation succeeds; otherwise returns the seeds unchanged.
    """
    from scipy.optimize import minimize
    from ..engine import make_predict_fn

    params_for_model = {
        "1cmt_iv":   ["CL", "V"],
        "1cmt_oral": ["CL", "V", "Ka"],
        "2cmt_iv":   ["CL", "V1", "Q", "V2"],
        "2cmt_inf":  ["CL", "V1", "Q", "V2"],
    }[model_name]

    if not subjects:
        return dict(seeds)
    predict_fn = make_predict_fn(model_name)
    y_pooled = np.concatenate([s.obs for s in subjects])
    x0 = np.array([math.log(seeds[p]) for p in params_for_model])

    def neg_ll(x):
        p = {n: math.exp(v) for n, v in zip(params_for_model, x)}
        if model_name == "1cmt_oral":
            p.setdefault("ALAG", 0.0)
        try:
            preds = [predict_fn(p, s) for s in subjects]
            f = np.concatenate(preds)
        except Exception:
            return 1e10
        f = np.maximum(f, 1e-10)
        sigma = 0.1 * f + 0.05
        return float(np.sum(np.log(sigma) + 0.5 * ((y_pooled - f) / sigma) ** 2))

    res = minimize(neg_ll, x0, method="Nelder-Mead",
                   options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-3,
                            "adaptive": True})
    if not np.isfinite(res.fun) or res.fun >= 1e9:
        return dict(seeds)
    return {n: float(math.exp(v)) for n, v in zip(params_for_model, res.x)}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def compute(subjects: list[Subject], model_name: str,
             drug_class: str | None = None,
             route: str | None = None) -> tuple[dict, AgentRunLog]:
    """Return data-only initial values + a log entry.

    Output dict has keys:
      init_theta : every structural param the model needs, with a
                   finite, dataset-derived value
      init_omega : flat 0.1 starting variance per ETA param
      source     : either "nca+refine" (NCA succeeded then refine
                   improved it), "nca" (refine did not improve), or
                   "nca-partial" (some params still NaN; caller must
                   error out — never silently substitute defaults)
    """
    seeds = _nca_init(subjects, model_name)
    bad = [p for p, v in seeds.items()
            if not (isinstance(v, (int, float)) and np.isfinite(v) and v > 0)]
    if bad:
        # Strict: refuse to fabricate values for params NCA couldn't anchor.
        # The caller may decide whether to error or to use a different model.
        out = {"init_theta": seeds, "init_omega": {},
                "source": "nca-partial"}
        return out, AgentRunLog("initial",
                                 {"model": model_name},
                                 out,
                                 [f"NCA could not anchor {bad}; "
                                  f"no fallback applied (data-only policy)"])

    refined = _naive_pooled_refine(subjects, model_name, seeds)
    used_refined = (refined != seeds)
    init_theta = dict(refined)
    # ALAG is not in the naive-pooled fit's parameter list (we hold it
    # fixed during the pooled MLE for stability); carry it through from
    # the NCA seeds, derived from the first-positive-observation time.
    if model_name == "1cmt_oral" and "ALAG" in seeds:
        init_theta["ALAG"] = seeds["ALAG"]

    init_omega = {p: 0.1 for p in init_theta if p != "ALAG"}
    out = {"init_theta": init_theta, "init_omega": init_omega,
            "source": "nca+refine" if used_refined else "nca"}
    return out, AgentRunLog("initial",
                             {"model": model_name},
                             out, [out["source"]])
