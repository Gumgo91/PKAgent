"""Structural-model decision agent — DATA-ONLY detection.

Pure-data PK signature analysis to choose among:
  - 1cmt_iv   : 1-compartment IV bolus
  - 1cmt_oral : 1-compartment with first-order absorption (optional lag)
  - 2cmt_iv   : 2-compartment IV bolus
  - 2cmt_inf  : 2-compartment IV infusion

The decision pipeline (NO drug-name, NO drug-class fallback that defaults
to a specific drug type):

  1. **Route inference** — purely structural from the Subject dataclass:
        - tinf > 0   for any subject → IV infusion
        - else  Cmax at t = t_min → IV bolus
        - else  Cmax delayed       → Oral
     If `route` is provided by the user as a hint, that overrides the
     inference only when the inference is ambiguous.

  2. **Compartment-count inference** (1cmt vs 2cmt) — by fitting both
     candidates as naive-pooled OLS on log-concentration vs time, the
     model with lower AIC wins.

  3. **ALAG presence** (oral only) — checked by comparing AIC of oral
     with vs without lag.

  4. **LLM synthesis** — Gemini-3 reviews the numerical evidence (route,
     AICs, Tmax distribution, terminal-slope analysis, etc.) and either
     confirms the heuristic call or overrides with a rationale.

When no LLM key is configured, step 4 is skipped and the heuristic
decision from steps 1-3 is returned.

This file deliberately does NOT consult drug-specific catalogs.  The
class catalog is only used downstream for covariate hints — never to
pick a structural model.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Iterable

from ..llm_client import SHARED, AgentRunLog
from ..models import (one_comp_iv, one_comp_oral, two_comp_iv,
                       two_comp_iv_infusion)
from ..nlme import Subject


def _data_features(subjects: list[Subject]) -> dict:
    """Truth-agnostic data signature used by the structural decision."""
    has_infusion = any(getattr(s, "tinf", 0.0) > 0 for s in subjects)

    cmax_at_tmin_fraction = []  # what fraction of subjects have Cmax at min time?
    tmax_values = []
    tmin_values = []
    t_total = []
    early_c_fraction = []  # C(t_min) / Cmax — close to 1 = bolus, close to 0 = oral/absorption

    for s in subjects:
        if len(s.obs) < 2: continue
        t = np.asarray(s.time, dtype=float)
        y = np.asarray(s.obs, dtype=float)
        idx_peak = int(np.argmax(y))
        idx_min = int(np.argmin(t))
        tmax_values.append(float(t[idx_peak]))
        tmin_values.append(float(t[idx_min]))
        t_total.append(float(t.max() - t.min()))
        c_max = float(y[idx_peak])
        c_min_t = float(y[idx_min])
        if c_max > 0:
            early_c_fraction.append(c_min_t / c_max)
        cmax_at_tmin_fraction.append(1 if idx_peak == idx_min else 0)

    out = {
        "n_subj": len(subjects),
        "n_obs_total": sum(len(s.obs) for s in subjects),
        "has_infusion_field": has_infusion,
        "cmax_at_tmin_subj_fraction": (sum(cmax_at_tmin_fraction)
                                        / max(len(cmax_at_tmin_fraction), 1)),
        "median_tmax": float(np.median(tmax_values)) if tmax_values else 0.0,
        "median_tmin": float(np.median(tmin_values)) if tmin_values else 0.0,
        "median_t_total": float(np.median(t_total)) if t_total else 0.0,
        "median_early_c_fraction": (float(np.median(early_c_fraction))
                                     if early_c_fraction else 0.0),
        # bi-phasic evidence will be filled in by _detect_compartment_count
    }
    return out


def _infer_route_from_data(features: dict, route_hint: str | None) -> str:
    """Pure-data route detection.  Hint is consulted only when ambiguous."""
    if features["has_infusion_field"]:
        return "iv_infusion"
    # Cmax at t_min for most subjects + concentration at t_min ≥ ~70% of Cmax →
    # IV bolus signature
    if (features["cmax_at_tmin_subj_fraction"] >= 0.7
            and features["median_early_c_fraction"] >= 0.7):
        return "iv_bolus"
    # Cmax delayed (t_max > t_min + small fraction of total span) → oral
    if features["median_tmax"] > features["median_tmin"] + 0.1 * features["median_t_total"]:
        return "oral"
    # Ambiguous — consult user hint, else default to oral (most common)
    if route_hint:
        r = route_hint.lower()
        if "infusion" in r:
            return "iv_infusion"
        if r.startswith("iv"):
            return "iv_bolus"
        if r.startswith("o"):
            return "oral"
    return "oral"


def _naive_pooled_aic(subjects: list[Subject], model_fn,
                       init_params: list[float],
                       has_inf: bool = False,
                       has_lag: bool = False) -> tuple[float, int]:
    """Quick naive-pooled OLS fit on log-concentration; return (AIC, n_params).

    `model_fn(t, dose, *params, [tinf], [ALAG])` returns concentration.
    Used purely for **model selection** (AIC), not as a final fit.
    """
    from scipy.optimize import minimize

    # collect data
    all_t, all_y, all_dose, all_tinf = [], [], [], []
    for s in subjects:
        all_t.append(np.asarray(s.time, dtype=float))
        all_y.append(np.asarray(s.obs, dtype=float))
        all_dose.append(np.full_like(s.time, s.dose, dtype=float))
        all_tinf.append(np.full_like(s.time, getattr(s, "tinf", 0.0), dtype=float))
    if not all_t:
        return float("inf"), 0
    t = np.concatenate(all_t); y = np.concatenate(all_y)
    dose = np.concatenate(all_dose); tinf = np.concatenate(all_tinf)
    rep_dose = float(np.median(dose))
    rep_tinf = float(np.median(tinf))

    n_params = len(init_params)
    x0 = np.log(np.asarray(init_params, dtype=float))

    def neg_ll(x):
        p = np.exp(x)
        try:
            if has_inf:
                # 2cmt infusion expects (CL, V1, Q, V2)
                f = model_fn(t, rep_dose, rep_tinf, *p)
            elif has_lag:
                # 1cmt_oral with lag = (CL, V, Ka, ALAG)
                f = model_fn(t, rep_dose, *p[:3], ALAG=p[3])
            else:
                f = model_fn(t, rep_dose, *p)
        except Exception:
            return 1e10
        f = np.maximum(f, 1e-10)
        # combined error
        sigma = 0.1 * f + 0.05
        # negative log likelihood ignoring constants
        ll = -0.5 * float(np.sum(np.log(sigma**2) + ((y - f) / sigma) ** 2))
        return -ll

    res = minimize(neg_ll, x0, method="Nelder-Mead",
                   options={"maxiter": 300, "xatol": 1e-3, "fatol": 1e-3})
    nll = float(res.fun)
    aic = 2 * nll + 2 * n_params
    return aic, n_params


def _data_only_decision(subjects: list[Subject], route_hint: str | None
                         ) -> dict:
    """Heuristic data-driven decision: route → compartment count → ALAG."""
    features = _data_features(subjects)
    route = _infer_route_from_data(features, route_hint)

    # Run candidate naive-pooled fits and compare AIC
    candidate_aics = {}
    try:
        if route == "iv_infusion":
            aic1, _ = _naive_pooled_aic(subjects, one_comp_iv, [3.0, 30.0])
            aic2, _ = _naive_pooled_aic(subjects, two_comp_iv_infusion,
                                         [3.0, 5.0, 5.0, 15.0], has_inf=True)
            candidate_aics["1cmt_inf"] = aic1
            candidate_aics["2cmt_inf"] = aic2
            chosen = "2cmt_inf" if aic2 + 2 < aic1 else "1cmt_iv"
            if chosen == "1cmt_iv":
                # would still need infusion handling; if patient `tinf>0` use
                # 1-cmt analytical IV (acceptable approximation if tinf small)
                chosen = "1cmt_iv"
        elif route == "iv_bolus":
            aic1, _ = _naive_pooled_aic(subjects, one_comp_iv, [3.0, 30.0])
            aic2, _ = _naive_pooled_aic(subjects, two_comp_iv,
                                         [3.0, 5.0, 5.0, 15.0])
            candidate_aics["1cmt_iv"] = aic1
            candidate_aics["2cmt_iv"] = aic2
            chosen = "2cmt_iv" if aic2 + 2 < aic1 else "1cmt_iv"
        else:  # oral
            aic1, _ = _naive_pooled_aic(subjects, one_comp_oral, [3.0, 30.0, 1.0])
            aic_lag, _ = _naive_pooled_aic(subjects, one_comp_oral,
                                            [3.0, 30.0, 1.0, 0.5], has_lag=True)
            candidate_aics["1cmt_oral"] = aic1
            candidate_aics["1cmt_oral_with_lag"] = aic_lag
            chosen = "1cmt_oral"
            include_alag = aic_lag + 2 < aic1
            features["alag_aic_evidence"] = aic_lag - aic1  # negative = lag helps
            return {
                "structural_model": chosen,
                "include_alag": include_alag,
                "rationale": (
                    f"data-only: route={route} (Cmax at "
                    f"t_min in {features['cmax_at_tmin_subj_fraction']*100:.0f}% "
                    f"of subjects; early/peak ratio = "
                    f"{features['median_early_c_fraction']:.2f}); "
                    f"1cmt vs 1cmt+lag AIC: "
                    f"{aic1:.1f} vs {aic_lag:.1f} → "
                    f"{'include' if include_alag else 'omit'} ALAG"),
                "features": features,
                "candidate_aics": candidate_aics,
            }
    except Exception as e:
        chosen = "1cmt_oral"  # only safe fallback when AIC compute fails
        candidate_aics["error"] = str(e)

    return {
        "structural_model": chosen,
        "include_alag": False,
        "rationale": (
            f"data-only: route={route} "
            f"(infusion_field={features['has_infusion_field']}, "
            f"Cmax-at-tmin frac={features['cmax_at_tmin_subj_fraction']:.2f}); "
            f"AICs: {candidate_aics}"),
        "features": features,
        "candidate_aics": candidate_aics,
    }


LLM_SYSTEM_PROMPT = """You are the **Structural-Model Decision Agent** in
PKAgent — a fully data-driven PopPK selector.

You are given:
  - data-only features (n_subj, n_obs, route inference signatures,
    Tmax/Tmin patterns)
  - naive-pooled AICs for candidate compartment models
  - the heuristic call already made from those numbers

Your job: confirm or override the heuristic call, citing the data
evidence.  Return STRICT JSON only.

Available models:
  - "1cmt_iv"   (params: CL, V)
  - "1cmt_oral" (params: CL, V, Ka, optionally ALAG)
  - "2cmt_iv"   (params: CL, V1, Q, V2)
  - "2cmt_inf"  (params: CL, V1, Q, V2; for IV infusion data)

Schema:
{
  "structural_model": "...",
  "include_alag": true | false,
  "agree_with_heuristic": true | false,
  "rationale": "...",
  "warnings": ["..."]
}

Decision rules:
- Choose 2-cmt over 1-cmt only if ΔAIC ≥ 4 (Burnham & Anderson 2002).
- Include ALAG only if the AIC improvement ≥ 4 AND the data shows
  delayed onset of absorption (Cmax not at the very first sample) —
  UNLESS the user hint explicitly contradicts (e.g. hint says
  'rapid IR absorption, no lag' → set include_alag=false even if
  AIC narrowly favours the lag model).
- Never use drug-NAME knowledge.  Decisions must be justifiable from
  the data features and/or the explicit user hint."""


def decide(subjects: list[Subject] | None = None,
            drug_class: str | None = None,  # IGNORED for structural decision
            route: str | None = None,
            structural_hint: str | None = None,
            hint: str | None = None,
            n_subj: int | None = None
            ) -> tuple[dict, AgentRunLog]:
    """Pure-data structural decision (Gemini-augmented when available).

    `drug_class` is accepted for API compatibility but is **ignored** in
    the structural decision — the only legitimate inputs are the data
    itself and the user's clinical observation of the route.

    `structural_hint` is honoured as a user override (e.g. when the user
    knows the analytical form), bypassing automatic detection.

    `hint` is a natural-language user brief about the drug — e.g.
    "IR tablet, rapid absorption, no lag" or "enteric-coated, slow onset".
    The Gemini synthesis step (when enabled) reads this hint and can
    override the data-only `include_alag` decision accordingly.  Hint
    has no effect on the heuristic-only path.
    """
    if structural_hint:
        # user provided exact model; pass through
        return ({"structural_model": structural_hint,
                 "include_alag": "_with_lag" in structural_hint,
                 "rationale": f"user override: {structural_hint}",
                 "warnings": []},
                AgentRunLog("structural", {"hint": structural_hint},
                             {"structural_model": structural_hint},
                             ["user-hint"]))

    if subjects is None:
        raise ValueError("structural.decide() requires `subjects` for data-driven detection")

    # Step 1-3: data-only heuristic
    det = _data_only_decision(subjects, route)

    # Step 4: optional LLM synthesis
    if SHARED.enabled:
        hint_block = (
            f"User-provided hint about the drug/study (free-form natural "
            f"language): {hint!r}\n"
            f"If the hint clearly signals rapid no-lag absorption (e.g. "
            f"'IR tablet', 'oral solution', 'rapid absorption', "
            f"'no absorption lag'), prefer include_alag=false even if AIC "
            f"narrowly favours the lag model.  If the hint signals delayed "
            f"absorption (e.g. 'enteric-coated', 'slow onset', "
            f"'delayed absorption', 'absorption lag'), prefer "
            f"include_alag=true.  When the hint is silent on absorption "
            f"or absent, fall back to the AIC evidence.\n"
            if hint else ""
        )
        user = (
            f"Data features: {det['features']}\n"
            f"Naive-pooled AICs: {det['candidate_aics']}\n"
            f"Heuristic decision: structural_model={det['structural_model']}, "
            f"include_alag={det.get('include_alag', False)}\n"
            f"Heuristic rationale: {det['rationale']}\n"
            f"{hint_block}"
            "Return strict JSON only.")
        out = SHARED.chat_json([LLM_SYSTEM_PROMPT], user, max_tokens=1024)
        if out and "structural_model" in out:
            return out, AgentRunLog(
                "structural", {"features": det["features"]},
                out, ["llm-mode",
                      f"heuristic={det['structural_model']}",
                      f"llm={out.get('structural_model')}",
                      f"include_alag={out.get('include_alag', '?')}",
                      f"hint={'yes' if hint else 'no'}",
                      f"agree={out.get('agree_with_heuristic', '?')}"])

    return det, AgentRunLog("structural", {"features": det["features"]},
                             det, ["data-only-heuristic"])
