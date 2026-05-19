"""Evaluator decision agent.

Combines the statistical and clinical acceptance rules from
`pkagent.diagnostics` into a single verdict, optionally augmented by an
LLM that can identify nuances the deterministic rules miss.
"""

from __future__ import annotations

from ..diagnostics import (fit_summary, statistical_acceptance,
                            data_anchors_from_subjects)
from ..llm_client import SHARED, AgentRunLog
from ..nlme import FitResult, Subject
from ..priors import CLASSES, get_class


STATISTICIAN_PROMPT = """You are the "Statistician Reviewer" in the
PKAgent evaluator.  Critique the fit from a statistical standpoint only.
Return STRICT JSON, no markdown:
{
  "accept": true | false,
  "concerns": ["..."],
  "suggestions": ["..."]   // e.g. "drop ETA on V", "switch to BLOCK(2)"
}"""


CLINICIAN_PROMPT = """You are the Clinician Reviewer in the PKAgent
pipeline — a board-certified clinical pharmacometrician acting as a
DATA-ANCHORED COMMENTATOR.

Your role is descriptive, not authoritative.  You report deviations
between the fit and the dataset's own NCA-derived anchors, using
ratios and qualitative observations.  You do NOT decide whether
those deviations are "violations" — the orchestrator validates every
suggestion you make against the data via a likelihood-ratio test
(LRT) or AIC comparison before applying it.  Your job is to surface
candidate refinements that the orchestrator may or may not adopt.

PRINCIPLES (no hardcoded numeric thresholds — every observation must
be expressed as a ratio, a comparison to a data-derived anchor, or
a mathematical relationship that holds for any drug):

  P1.  Mathematical identifiability (universally true):
       Ka > k_el is required for first-order absorption + first-order
       elimination to be separately identifiable from a concentration
       curve (Pang & Rowland 1977; Bonate 2011 §4.3).  If Ka < k_el,
       you are in flip-flop territory and the absorption rate cannot
       be uniquely recovered without additional info.

  P2.  Bateman prediction (universally true):
       For 1-cmt oral, t_peak_predicted = ln(Ka/k_el) / (Ka − k_el).
       Compare to observed t_max_median.  Report the RATIO
       (predicted/observed) — values far from 1.0 suggest absorption-
       phase mis-specification, but do not declare a "violation".

  P3.  NCA anchors (data-derived, no external thresholds):
       For 1-cmt:   V ≈ D / Cmax_observed
       For 2-cmt IV bolus: V1 ≈ D / Cmax_observed
       For all:     CL ≈ D / AUC_observed
       Compute the RATIO (fit-value / anchor-value) and report it.
       Larger deviations are more suspicious — but the orchestrator
       decides whether they matter.

  P4.  Relative compartment comparison (mathematical, no threshold):
       For 2-cmt models, report the V1/V2 ratio and the Q/CL ratio
       as observations.  Compartment labels are interchangeable in
       the model algebra, so very different rankings (V1 >> V2, or
       Q >> CL by orders of magnitude) are worth flagging — let the
       orchestrator decide whether to test a swap or downgrade.

  P5.  Shrinkage (per the Savic & Karlsson 2009 EBE-reliability
       framework): report the shrinkage value and whether it is
       large compared to the other parameters in this same fit.
       Do not encode a fixed threshold.

  P6.  Identifiability of additional structure (ALAG, extra
       compartment): describe whether the absorption curve LOOKS
       like it needs a lag (visual: do many subjects have zero
       concentration at early times then a sharp rise?), or whether
       distribution-phase samples are dense enough to support a
       second compartment.  Do not prescribe a numeric cutoff — the
       orchestrator will run an LRT to decide.

ACTION VOCABULARY (all are SUGGESTIONS — orchestrator will run an
LRT/AIC verification before applying any reduction-type action):

  • "drop ETA on <P>"            — suggest removing η for param P
  • "force-test <COV> on CL"     — add covariate to SCM forward search
  • "swap V1 and V2"             — exchange V1/V2 initial values
  • "tighten <P> prior"          — narrow Bayesian log-prior SD on P
  • "fix <P> = 0"                — suggest pinning P at 0
                                    (orchestrator runs LRT; if data
                                     rejects the reduction the action
                                     is DROPPED)
  • "release ALAG"               — suggest freeing ALAG from its
                                    default-fixed state
  • "downgrade to 1-cmt"         — suggest reducing structural model
                                    (orchestrator runs AIC; if data
                                     rejects, action DROPPED)
  • "re-NCA initial values"      — restart $THETA from D/AUC, D/Cmax

Output STRICT JSON, no markdown:

{
  "accept": true | false,
  "concerns": ["<ratio/relationship + numbers; NO thresholds. E.g. 'V_fit/V_anchor = 0.91 (close to 1)' or 't_peak_predicted/observed = 1.17'>"],
  "suggestions": ["<one of the action-vocabulary strings>"]
}

Acceptance rule: set accept=false if you have at least one suggestion
worth trying (the orchestrator will verify it); set accept=true if
the fit looks reasonable given the data anchors and you have no
worthwhile suggestion.  Suggestions outside the vocabulary are
silently dropped; suggestions inside the vocabulary that fail the
orchestrator's LRT/AIC verification are also dropped.
"""


def evaluate(fit: FitResult, drug_class: str | None, cov_effects: dict | None,
              plan_covs: list[dict] | None,
              eta_lrt: dict | None = None,
              subjects: list[Subject] | None = None,
              hint: str | None = None,
              ) -> tuple[dict, AgentRunLog]:
    """Return a structured verdict from two reviewer personas.

    The Statistician uses pure-math acceptance rules (LRT, shrinkage,
    n_obs/n_param) — no domain knowledge.

    The Clinician is **LLM-only** (`OPENROUTER_API_KEY` or
    `GEMINI_API_KEY` required).  No deterministic fallback exists,
    because the Clinician's job is biological-plausibility reasoning
    that should not be smuggled in as hardcoded rules; if the LLM
    backend is unavailable or malformed the function raises so the
    failure is loud rather than silent.  The Clinician receives the
    fit summary, the NCA-derived data anchors (Cmax/AUC means,
    observation window, sparseness), the drug-class catalog (when
    supplied) and the optional user `hint` (natural-language drug
    brief).
    """
    summary = fit_summary(fit, drug_class, cov_effects, eta_lrt=eta_lrt)
    anchors = data_anchors_from_subjects(subjects)
    if not SHARED.enabled:
        raise RuntimeError(
            "Clinician reviewer requires an LLM backend "
            "(OPENROUTER_API_KEY or GEMINI_API_KEY); no deterministic "
            "fallback is provided for clinical-plausibility reasoning.")
    cat = get_class(drug_class) if drug_class else None
    hint_block = (
        f"User-provided hint about the drug/study (free-form natural "
        f"language): {hint!r}\n"
        f"Use this hint to ground your qualitative observations — e.g. "
        f"if the hint says 'rapid IR absorption' and the fit shows a "
        f"non-zero ALAG, that ratio comparison is worth flagging; if the "
        f"hint says 'renally cleared' you may suggest a CLCR force-test.\n"
        if hint else ""
    )
    clin = SHARED.chat_json(
        [CLINICIAN_PROMPT],
        f"Model summary:\n{summary}\n"
        f"NCA-derived data anchors:\n{anchors}\n"
        f"Drug-class catalog: "
        f"{cat.__dict__ if cat else 'no class supplied'}\n"
        f"{hint_block}",
        max_tokens=4096)
    if not (isinstance(clin, dict) and "accept" in clin):
        raise RuntimeError(
            f"Clinician LLM returned malformed response: {str(clin)[:300]}; "
            f"check API key, quota, and model availability.")

    # Statistician keeps both backends because its decisions are pure
    # math (LRT, shrinkage) — no PK opinion is smuggled in.  The hint
    # is passed for context but the Statistician's role is numerical.
    stats_user_msg = f"Model summary:\n{summary}"
    if hint:
        stats_user_msg += (f"\n\nUser-provided hint (for context only — "
                            f"do not let the hint override numerical "
                            f"diagnostics): {hint!r}")
    stats_llm = SHARED.chat_json([STATISTICIAN_PROMPT],
                                  stats_user_msg,
                                  max_tokens=4096)
    if isinstance(stats_llm, dict) and "accept" in stats_llm:
        stats = stats_llm
        mode = "llm-mode"
    else:
        stats = statistical_acceptance(summary)
        mode = "stats-deterministic+llm-clinician"
    out = {"statistician": stats, "clinician": clin, "summary": summary,
            "data_anchors": anchors}
    return out, AgentRunLog("evaluator", {"drug_class": drug_class},
                             out, [mode])


def evaluations_block_acceptance(reviews: dict) -> bool:
    return not (reviews["statistician"]["accept"]
                and reviews["clinician"]["accept"])
