"""Fit-strategy decision agent (LLM-driven adaptive tuning).

PKAgent's NLME engine exposes a small but meaningful configuration
space of fitting "knobs":

  * `n_ensemble`             — # of perturbed-init NLME runs
  * `prior_strength`         — log_sd multiplier on θ priors (tight / mod / wide)
  * `inner_optimizer`        — "auto" (BFGS w/ NM fallback) | "nm" | "bfgs"
  * `omega_floor`            — numerical lower bound for ω²
  * `max_feedback_rounds`    — safety cap on the evaluator loop

The strategy agent uses Gemini (via SHARED.chat_json) to pick a starting
configuration from a *truth-agnostic* dataset summary (n_subj, n_obs,
obs/subj, conc-range, class label).  When the evaluator rejects a fit,
the agent re-examines the diagnostic summary and produces a refined
strategy — this is the "PKAgent automatically adapts to the data" loop.

Both the initial pick and the adapt step run through Gemini-3-flash-preview
when a key is present; otherwise a deterministic heuristic encodes the
same PopPK rules of thumb.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, asdict

from ..llm_client import SHARED, AgentRunLog
from ..nlme import Subject


@dataclass
class FitStrategy:
    n_ensemble: int = 3
    prior_strength: str = "wide"     # "tight" | "moderate" | "wide"
    inner_optimizer: str = "auto"    # "auto" | "nm" | "bfgs"
    omega_floor: float = 1e-3
    max_feedback_rounds: int = 5

    @property
    def prior_log_sd_multiplier(self) -> float:
        return {"tight": 0.5, "moderate": 1.0, "wide": 1.5}.get(
            self.prior_strength, 1.0)


def _dataset_summary(subjects: list[Subject]) -> dict:
    """Truth-agnostic data shape summary fed to the strategy LLM."""
    n_subj = len(subjects)
    obs_counts = [len(s.obs) for s in subjects]
    all_obs = np.concatenate([s.obs for s in subjects]) if subjects else np.array([])
    all_t = np.concatenate([s.time for s in subjects]) if subjects else np.array([])
    return {
        "n_subj": n_subj,
        "n_obs_total": int(sum(obs_counts)),
        "obs_per_subj_median": float(np.median(obs_counts)) if obs_counts else 0,
        "obs_per_subj_min": int(min(obs_counts)) if obs_counts else 0,
        "obs_per_subj_max": int(max(obs_counts)) if obs_counts else 0,
        "conc_min": float(all_obs.min()) if len(all_obs) else 0,
        "conc_max": float(all_obs.max()) if len(all_obs) else 0,
        "conc_log_range": float(np.log10(all_obs.max() / max(all_obs.min(), 1e-12)))
                          if len(all_obs) else 0,
        "time_max": float(all_t.max()) if len(all_t) else 0,
    }


SYSTEM_PROMPT = """You are the Fit-Strategy Decision Agent in PKAgent.

Given a truth-agnostic dataset summary and an optional drug-class label,
pick a FitStrategy configuration that will help the NLME engine succeed.

Return STRICT JSON:
{
  "n_ensemble": 2 | 3 | 5,
  "prior_strength": "tight" | "moderate" | "wide",
  "inner_optimizer": "auto" | "nm" | "bfgs",
  "omega_floor": 1e-3 | 1e-4,
  "max_feedback_rounds": 3 | 5,
  "rationale": "..."
}

Rules of thumb (Bonate 2011, Bauer 2017):
  - Sparse data (obs/subj < 5 or n_subj < 15): use TIGHTER prior_strength
    so the prior pulls the fit toward physiology when data is uninformative.
  - Dense data (obs/subj >= 8 and n_subj >= 25): use WIDER prior so the
    data dominates; the prior is just a soft regulariser.
  - Oral models with absorption lag: BFGS inner optimizer (Karlsson &
    Sheiner 1993) — Ka/ALAG ridge benefits from gradient-based mode location.
  - Large concentration dynamic range (>3 log10 units): more ensemble runs
    to escape local minima.
  - Default: n_ensemble=3, prior_strength="moderate", inner="auto",
    omega_floor=1e-3, max_feedback_rounds=5.
"""


ADAPT_SYSTEM_PROMPT = """You are the Fit-Strategy Adapter in PKAgent.
The previous fit was evaluated and the evaluator gave feedback below.
Propose a REFINED FitStrategy.

Return STRICT JSON in the same schema as the initial strategy.

Adaptation rules of thumb:
  - If shrinkage is HIGH on a primary parameter → increase prior_strength
    (tighten) and/or increase n_ensemble to find a better mode.
  - If omega collapsed → drop the offending ETA (handled elsewhere); strategy
    agent should LOWER omega_floor next round.
  - If multiple ensemble runs converged to very different OFVs (>10%
    spread) → increase n_ensemble.
  - If the optimizer reported lack of convergence → try the other inner
    optimizer.
"""


def _deterministic_initial(summary: dict, drug_class: str | None) -> FitStrategy:
    """Rule-based fallback when no LLM is available."""
    n_subj = summary.get("n_subj", 0)
    ops = summary.get("obs_per_subj_median", 0)
    conc_log = summary.get("conc_log_range", 0)
    s = FitStrategy()
    if ops < 5 or n_subj < 15:
        s.prior_strength = "moderate"   # one notch tighter for sparse data
    elif ops >= 8 and n_subj >= 25:
        s.prior_strength = "wide"
    else:
        s.prior_strength = "moderate"
    # absorption with lag → benefit from BFGS inner
    if drug_class and "with_lag" in drug_class:
        s.inner_optimizer = "bfgs"
    if conc_log > 3:
        s.n_ensemble = 5
    return s


def decide(subjects: list[Subject], drug_class: str | None = None,
            route: str | None = None, hint: str | None = None,
            ) -> tuple[FitStrategy, AgentRunLog]:
    """Pick an initial FitStrategy from a dataset summary + class metadata.

    `hint` is the optional natural-language user brief about the drug —
    e.g. "IR tablet, rapid absorption, hepatic CYP3A4 metabolism".  It
    is appended to the prompt so the LLM can adapt knobs (prior_strength,
    inner_optimizer, ensemble size) to the drug's profile.
    """
    summary = _dataset_summary(subjects)
    if SHARED.enabled:
        hint_block = (f"User-provided hint about the drug/study: {hint!r}\n"
                       if hint else "")
        user = (
            f"Dataset summary: {summary}\n"
            f"Drug class: {drug_class}\nRoute: {route}\n"
            f"{hint_block}"
            "Return strict JSON only.")
        out = SHARED.chat_json([SYSTEM_PROMPT], user, max_tokens=1024)
        # parse out the strategy fields
        if all(k in out for k in ("n_ensemble", "prior_strength",
                                     "inner_optimizer", "omega_floor")):
            try:
                s = FitStrategy(
                    n_ensemble=int(out["n_ensemble"]),
                    prior_strength=str(out["prior_strength"]),
                    inner_optimizer=str(out["inner_optimizer"]),
                    omega_floor=float(out["omega_floor"]),
                    max_feedback_rounds=int(out.get("max_feedback_rounds", 5)),
                )
                return s, AgentRunLog("strategy",
                                       {"summary": summary, "drug_class": drug_class},
                                       {"strategy": asdict(s),
                                        "rationale": out.get("rationale", "")},
                                       ["llm-mode"])
            except (ValueError, TypeError):
                pass
    s = _deterministic_initial(summary, drug_class)
    return s, AgentRunLog("strategy",
                           {"summary": summary, "drug_class": drug_class},
                           {"strategy": asdict(s)}, ["deterministic"])


def adapt(prev: FitStrategy, evaluator_reviews: dict,
           subjects: list[Subject],
           drug_class: str | None = None) -> tuple[FitStrategy, AgentRunLog]:
    """Refine the strategy based on the evaluator's feedback."""
    summary = _dataset_summary(subjects)
    if SHARED.enabled:
        user = (
            f"Previous strategy: {asdict(prev)}\n"
            f"Evaluator reviews: {evaluator_reviews}\n"
            f"Dataset summary: {summary}\n"
            f"Drug class: {drug_class}\n"
            "Return strict JSON only.")
        out = SHARED.chat_json([ADAPT_SYSTEM_PROMPT], user, max_tokens=1024)
        if all(k in out for k in ("n_ensemble", "prior_strength",
                                     "inner_optimizer", "omega_floor")):
            try:
                s = FitStrategy(
                    n_ensemble=int(out["n_ensemble"]),
                    prior_strength=str(out["prior_strength"]),
                    inner_optimizer=str(out["inner_optimizer"]),
                    omega_floor=float(out["omega_floor"]),
                    max_feedback_rounds=int(out.get("max_feedback_rounds",
                                                      prev.max_feedback_rounds)),
                )
                return s, AgentRunLog("strategy-adapt",
                                       {"prev": asdict(prev),
                                        "reviews": evaluator_reviews},
                                       {"strategy": asdict(s),
                                        "rationale": out.get("rationale", "")},
                                       ["llm-mode"])
            except (ValueError, TypeError):
                pass
    # deterministic adapt: tighten priors if shrinkage high
    stat_concerns = evaluator_reviews.get("statistician", {}).get("concerns", [])
    s = FitStrategy(**asdict(prev))
    if any("shrinkage" in c and ">" in c for c in stat_concerns):
        s.prior_strength = "tight" if prev.prior_strength == "moderate" else "moderate"
    if any("did not converge" in c for c in stat_concerns):
        s.inner_optimizer = "nm" if prev.inner_optimizer == "bfgs" else "bfgs"
    if s.n_ensemble < 5:
        s.n_ensemble = min(5, s.n_ensemble + 1)
    return s, AgentRunLog("strategy-adapt",
                           {"prev": asdict(prev)},
                           {"strategy": asdict(s)}, ["deterministic"])
