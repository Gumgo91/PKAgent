"""Candidate-covariate decision agent — DATA-DRIVEN.

Scans the actual `subjects.covariates` to discover which covariates are
present in the data, then asks the LLM (or applies a heuristic) to
propose biologically plausible parameter-covariate pairings.

No drug name, no drug_class catalog lookup.  The decision is based on:
  - data: which covariates exist in the dataset
  - LLM: well-known pharmacology (CLCR → renal CL, WT → allometric V/CL,
    AGE → CL, etc.) — covariate-specific, not drug-specific
  - SCM downstream: AIC-based selection from these candidates

The user can supply additional `covariate_hints` (e.g. ["CL~CLCR"]) when
they have clinical knowledge they want PKAgent to test.
"""

from __future__ import annotations

from ..llm_client import SHARED, AgentRunLog
from ..nlme import Subject


SYSTEM_PROMPT = """You are the Covariate-Selection Decision Agent in PKAgent.
Given the list of covariates that are actually recorded in the dataset
and the structural-model parameter names, propose candidate
parameter-covariate pairings to test in stepwise covariate modelling.

Return STRICT JSON:
{
  "candidates": [{"param": "...", "covariate": "...", "kind": "power|exp|linear",
                  "rationale": "..."}]
}

Pharmacology rules (covariate-specific, NOT drug-specific):
  - CLCR (creatinine clearance) → propose for CL with kind="power" (renal
    elimination scales with CLCR, well-established for renally-cleared drugs).
  - WT (body weight) → propose for clearance (CL) and central volume
    (V or V1) with kind="power" — allometric scaling (Anderson & Holford 2008).
  - AGE → optionally for CL with kind="exp" (age-related decline in
    clearance for hepatically-cleared drugs); only if AGE varies enough.
  - SEX → optionally for CL with kind="linear" (sex differences in some PK
    pathways).

Propose only pairings whose covariate exists in the provided list.  Do
NOT propose pairings based on drug-specific knowledge."""


def _discover_covariates(subjects: list[Subject]) -> set[str]:
    """Pure-data: which covariates appear in any subject record?"""
    available = set()
    for s in subjects:
        if hasattr(s, "covariates") and isinstance(s.covariates, dict):
            available.update(s.covariates.keys())
    return available


def _deterministic(available_covs: set[str], structural_params: list[str],
                    covariate_hints: list[str] | None) -> dict:
    """Deterministic pharmacology heuristic when no LLM is available.
    Applies textbook covariate-parameter rules to the covariates that
    actually exist in the data.
    """
    cands = []
    # CL pairings
    primary_clearance = "CL"
    if primary_clearance in structural_params:
        if "CLCR" in available_covs:
            cands.append({"param": "CL", "covariate": "CLCR", "kind": "power",
                          "rationale": "creatinine clearance → renal CL "
                                       "(Cockcroft-Gault, Bauer 2008)"})
        if "WT" in available_covs:
            cands.append({"param": "CL", "covariate": "WT", "kind": "power",
                          "rationale": "allometric scaling on CL "
                                       "(Anderson & Holford 2008)"})
        if "AGE" in available_covs:
            cands.append({"param": "CL", "covariate": "AGE", "kind": "exp",
                          "rationale": "age-related decline in clearance"})
    # Volume pairings
    primary_volume = "V1" if "V1" in structural_params else "V"
    if primary_volume in structural_params:
        if "WT" in available_covs:
            cands.append({"param": primary_volume, "covariate": "WT",
                          "kind": "power",
                          "rationale": f"allometric scaling on "
                                       f"{primary_volume} (body-size scaling)"})
    # user-supplied hints (override / add)
    for h in (covariate_hints or []):
        if "~" in h:
            p, c = h.split("~", 1)
            if (c in available_covs and p in structural_params and
                    not any(d["param"] == p and d["covariate"] == c for d in cands)):
                cands.append({"param": p, "covariate": c, "kind": "power",
                              "rationale": "user-supplied clinical hint"})
    return {"candidates": cands, "mandatory": []}


def propose(subjects: list[Subject] | None = None,
             model_name: str | None = None,
             route: str | None = None,
             drug_class: str | None = None,    # ignored (kept for API compat)
             covariate_hints: list[str] | None = None,
             hint: str | None = None,
             ) -> tuple[dict, AgentRunLog]:
    """Data-driven candidate-covariate proposal.

    Discovers available covariates from the data itself, then uses LLM
    pharmacology reasoning (or deterministic textbook heuristic) to
    propose parameter-covariate pairings.  `drug_class` is accepted only
    for API compatibility and is NOT used.

    `hint` is the optional natural-language user brief.  When the hint
    explicitly mentions an elimination pathway or covariate ("renally
    cleared", "hepatic CYP3A4", "CLCR-driven"), the LLM is more likely
    to propose the matching covariate pairing.
    """
    if subjects is None:
        # API-compat path: return empty proposal
        return ({"candidates": [], "mandatory": []},
                AgentRunLog("covariate", {}, {"candidates": []},
                             ["no-subjects-given"]))

    available = sorted(_discover_covariates(subjects))
    # extract structural params from model_name
    from ..models import REGISTRY
    structural_params = (REGISTRY[model_name].params
                         if model_name in REGISTRY else [])

    if SHARED.enabled and available and structural_params:
        hint_block = (
            f"User-provided hint about the drug/study: {hint!r}\n"
            f"If the hint names a covariate that is in the available list, "
            f"propose that pairing.  Stay covariate-specific, not drug-"
            f"specific: do not infer covariates from drug identity.\n"
            if hint else ""
        )
        user = (
            f"Available covariates in the dataset: {available}\n"
            f"Structural model parameters: {structural_params}\n"
            f"User clinical hints (optional): {covariate_hints}\n"
            f"{hint_block}"
            "Return strict JSON only.")
        out = SHARED.chat_json([SYSTEM_PROMPT], user, max_tokens=1024)
        if out and "candidates" in out:
            # filter: keep only candidates whose covariate is actually present
            out["candidates"] = [c for c in out["candidates"]
                                  if c.get("covariate") in available
                                  and c.get("param") in structural_params]
            out["mandatory"] = []   # data-driven: no drug-specific mandates
            return out, AgentRunLog("covariate",
                                     {"available": available,
                                      "params": structural_params},
                                     out, ["llm-mode"])
    det = _deterministic(set(available), structural_params, covariate_hints)
    return det, AgentRunLog("covariate",
                             {"available": available,
                              "params": structural_params},
                             det, ["deterministic"])
