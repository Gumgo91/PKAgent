"""Arbitrator decision agent — cross-round meta-reviewer.

Acts as the senior pharmacometrician (think: the PI signing off on a
PopPK manuscript) who reads the entire reviewer-feedback iteration
log and picks the ONE round whose fit should be reported as the
final answer.

This is the LLM analogue of the NONMEM analyst's notebook entry:
"I ran the LRT, saw ΔOFV = 38 favouring the lag model, but the lag
absorbed the absorption-rate variance and inflated Ka — I'm going
with the parsimonious model."  No single deterministic rule (lowest
OFV, LRT alone, last-round) captures that judgment, because the
trade-off is between statistical evidence (LRT/OFV) and biological
plausibility (C1-C9 PK constraints, NCA anchors).

LLM-only — no deterministic fallback for the cross-round judgment
itself.  If the LLM backend errors mid-call, callers fall back to
"last sane round" (the NONMEM-analyst default), and the failure is
logged so the user sees it happened.
"""

from __future__ import annotations

from ..llm_client import SHARED, AgentRunLog
from ..priors import get_class


ARBITRATOR_PROMPT = """You are the "Final Judge" in the PKAgent
pipeline — the senior pharmacometrician who reads the iteration
log and picks the round whose fit best represents the data, using
MATHEMATICAL INVARIANTS AND DATA-DERIVED ANCHORS ONLY.

You do NOT carry hardcoded numeric thresholds for "good" parameter
values.  Different drugs and study designs span many orders of
magnitude — what is typical for one drug is impossible for another.
Therefore every comparison you make must be either a likelihood-
ratio test (LRT) between nested fits, an AIC comparison, or a
ratio against this dataset's own NCA anchors.

DECISION FRAMEWORK (apply in order):

  J1. LIKELIHOOD-RATIO TEST for nested models (purely statistical):
      For two rounds with the same structural form that differ by
      Δdf free parameters, compute ΔOFV = OFV_reduced − OFV_full.
        • ΔOFV > 3.84·Δdf at α=0.05 → the data supports the full
          (more-parameters) model statistically.
        • ΔOFV ≤ 3.84·Δdf → the simpler model is statistically
          adequate; prefer it by parsimony.
      The 3.84 constant comes from χ²₁ at α=0.05 and is the only
      "magic number" — it is a mathematical fact about the χ²
      distribution, not a drug-specific threshold.

  J2. AIC for non-nested or differently structured rounds:
      Use AIC = OFV + 2·k (Burnham & Anderson 2002).  Smaller AIC
      wins.  If |ΔAIC| < 4, the two models are statistically
      indistinguishable — prefer the simpler model (smaller k).

  J3. NCA-ANCHOR RATIO COMPARISON when LRT/AIC are tied:
      Compute, for each round, |log(θ_fit / θ_anchor)| for the
      anchor-able parameters (V vs D/Cmax, CL vs D/AUC, predicted
      t_peak vs observed t_max).  Smaller |log-ratio| → closer to
      the dataset's own NCA estimate, indicating the fit is more
      grounded in the data.  Use this as a tiebreaker, NOT as a
      threshold for rejecting a fit.

  J4. SANITY (already enforced by the orchestrator):
      Degenerate fits (CL ≤ 0, NaN OFV) are filtered out before
      you see them.  Trust the input list.

  J5. NONMEM-ANALYST DEFAULT:
      When J1, J2, and J3 do not produce a clear winner, choose the
      LAST sane round in the trace — that is the model the analyst
      converged on after iterative refinement, the standard
      reporting convention.

WHAT YOU MUST NOT DO:
  • Do not invoke absolute parameter thresholds (Ka > 2, V > 1000,
    Q > 10·CL, etc.) — these encode drug-specific assumptions and
    do not generalize.
  • Do not override an LRT-supported model based on prose
    "biological plausibility" alone.  If you want to override an
    LRT result, you must cite an NCA-anchor RATIO that is large
    enough that the round being overridden is implausibly far from
    the data's own NCA estimate compared to the alternative.
  • Do not trust Clinician concerns wholesale — they are commentary,
    not arbiters.  Cross-check against the LRT/AIC/anchor numbers
    you compute yourself.

YOUR INPUT — for each sane round:
  - round_idx, model_name
  - θ̂ (all structural params)
  - OFV (FOCE-I marginal, NONMEM-comparable)
  - η-shrinkage per parameter
  - statistician_concerns, clinician_concerns (advisory text)
  - actions_applied (plan-edits applied AFTER this round)
  - n_free_params

You also receive the NCA-derived data anchors (Cmax_geomean,
V_apparent_geomean, CL_apparent_geomean, tmax_median, obs_window,
n_obs_per_subject) — use these for J3.

OUTPUT — STRICT JSON, no markdown:

{
  "chosen_round": <int>,                  // round_idx
  "primary_reason": "<J1|J2|J3|J5>",
  "rationale": "<2-4 sentences; cite LRT/AIC/anchor-ratio numbers>",
  "key_evidence": ["<bullet, with numbers>", ...],
  "rejected_alternatives": [{"round": <int>, "why": "<reason>"}]
}

Pick exactly one round.  When uncertain, default to J5 (last sane).
"""


def arbitrate(round_fits_summary: list[dict],
              data_anchors: dict,
              drug_class: str | None,
              hint: str | None = None,
              ) -> tuple[dict, AgentRunLog]:
    """Pick the final round across all sane feedback iterations.

    Parameters
    ----------
    round_fits_summary : list of dicts, one per sane round.
    data_anchors : dict from diagnostics.data_anchors_from_subjects.
    drug_class : optional class label.
    hint : optional natural-language user brief about the drug/study.
           The Arbitrator may consult the hint to break ties between
           statistically-equivalent rounds (|ΔAIC|<4) — e.g. if the hint
           says "rapid IR absorption", a round with non-zero ALAG looks
           less aligned with the user's biology.  The hint NEVER over-
           rides a decisive LRT (ΔOFV well outside χ²·Δdf).
    """
    if not round_fits_summary:
        raise ValueError("arbitrate() requires at least one round summary")
    if len(round_fits_summary) == 1:
        idx = round_fits_summary[0]["round_idx"]
        out = {"chosen_round": idx,
               "primary_reason": "single-round",
               "rationale": "Only one sane round produced; no arbitration.",
               "key_evidence": [],
               "rejected_alternatives": []}
        return out, AgentRunLog("arbitrator", {"drug_class": drug_class},
                                 out, ["trivial-single-round"])

    if not SHARED.enabled:
        raise RuntimeError(
            "Arbitrator requires an LLM backend (OPENROUTER_API_KEY or "
            "GEMINI_API_KEY); no deterministic fallback for cross-round "
            "comprehensive judgment.")

    cat = get_class(drug_class) if drug_class else None

    # Build a human-readable structured table for the LLM
    blocks = []
    for r in round_fits_summary:
        theta_str = ", ".join(f"{k}={v:g}" for k, v in r["theta"].items())
        shrink_str = ", ".join(f"{k}={v}%" for k, v in r["shrinkage"].items())
        stat_str = "; ".join(r["statistician_concerns"][:3]) or "(none)"
        clin_str = "; ".join(r["clinician_concerns"][:3]) or "(none)"
        actions_str = "; ".join(r["actions_applied"]) or "(none)"
        blocks.append(
            f"Round {r['round_idx']}\n"
            f"  model     : {r['model_name']}\n"
            f"  OFV       : {r['ofv_foce']:.2f}\n"
            f"  θ̂         : {theta_str}\n"
            f"  shrinkage : {shrink_str}\n"
            f"  n_free_θ  : {r['n_free_params']}\n"
            f"  stat_concerns: {stat_str}\n"
            f"  clin_concerns: {clin_str}\n"
            f"  actions_applied: {actions_str}"
        )
    rounds_text = "\n\n".join(blocks)

    hint_block = (
        f"=== User-provided hint about the drug/study ===\n{hint!r}\n\n"
        f"Use the hint as a tiebreaker when J1-J3 leave statistically "
        f"equivalent rounds (|ΔAIC|<4): prefer the round whose θ̂ aligns "
        f"with what the hint describes.  DO NOT use the hint to override "
        f"a decisive LRT or AIC verdict — the data still has primacy.\n\n"
        if hint else ""
    )
    user_msg = (
        f"=== Iteration log ({len(round_fits_summary)} sane rounds) ===\n\n"
        f"{rounds_text}\n\n"
        f"=== NCA-derived data anchors ===\n{data_anchors}\n\n"
        f"=== Drug-class catalog ===\n"
        f"{cat.__dict__ if cat else 'no class supplied'}\n\n"
        f"{hint_block}"
        f"Pick the round whose fit best represents the data, following "
        f"the J1-J5 framework.  Return STRICT JSON only."
    )

    verdict = SHARED.chat_json([ARBITRATOR_PROMPT], user_msg, max_tokens=4096)

    if not (isinstance(verdict, dict) and "chosen_round" in verdict):
        raise RuntimeError(
            f"Arbitrator LLM returned malformed response: "
            f"{str(verdict)[:300]}; "
            f"check API key, quota, and model availability.")

    # Validate chosen_round is in the input list; coerce to last sane if not
    valid_idx = [r["round_idx"] for r in round_fits_summary]
    if verdict["chosen_round"] not in valid_idx:
        verdict.setdefault("rationale", "")
        verdict["rationale"] += (
            f" [Note: LLM picked round {verdict['chosen_round']} which is "
            f"not in the sane list {valid_idx}; coerced to last sane.]")
        verdict["chosen_round"] = valid_idx[-1]
        verdict["primary_reason"] = "J5 (coerced — LLM out of range)"

    notes = [f"chose round {verdict['chosen_round']}",
              f"reason: {verdict.get('primary_reason', '?')}"]
    return verdict, AgentRunLog("arbitrator", {"drug_class": drug_class},
                                 verdict, notes)
