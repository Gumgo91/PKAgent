"""Hint Interpreter — converts the user's natural-language brief into
authoritative structured constraints that every downstream stage
(LLM agents AND deterministic gates) must honour.

The interpreter is the *single point of truth* for what the user
asked for.  Once it runs, the resulting `HintConstraints` flows
through:

  • the Structural agent      (forced compartments, route, ALAG opt-in)
  • the Initial-value step    (forced fix_params at 0)
  • the Covariate agent       (favoured covariates from hint)
  • the Evaluator             (clinician/statistician see raw hint)
  • the Arbitrator            (tiebreaks aligned with hint)
  • the LRT-verifier          (cannot release a hint-fixed parameter)
  • the Action parser         (forbidden_actions are dropped on sight)

This is the architecture that makes hints behave as *ground truth*
instead of optional commentary — the LRT/AIC tests still run for
unhinted decisions, but a user-specified constraint cannot be
overruled by data-only statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

from ..llm_client import SHARED, AgentRunLog


@dataclass
class HintConstraints:
    """Structured ground-truth constraints derived from the user hint.

    Hard constraints (authoritative — orchestrator enforces unconditionally):
      force_fix_params:   parameters the user said must stay at 0
                           (e.g. {"ALAG"} for "rapid IR, no lag")
      forbidden_actions:  reviewer action phrases that must be dropped
                           on sight regardless of LRT/AIC support
                           (e.g. {"release ALAG"} for "no lag")
      force_route:        user-specified route ("oral"/"iv"/"iv_infusion")

    Soft constraints (LLM advisory — agents may use, but data has the
    final say where the hint is silent):
      suggested_compartments:  1 or 2, when the user names a model order
      suggested_covariates:   covariates whose pharmacology the hint
                              hints at (e.g. ["CLCR"] for renally cleared)
      absorption_speed:       "rapid" | "slow" | "delayed" | None

    raw_text holds the original hint string for LLM agents to read in
    full (the structured constraints are for deterministic gates).
    """
    force_fix_params: set[str] = field(default_factory=set)
    forbidden_actions: set[str] = field(default_factory=set)
    force_route: str | None = None
    suggested_compartments: int | None = None
    suggested_covariates: list[str] = field(default_factory=list)
    absorption_speed: str | None = None
    raw_text: str | None = None

    def is_empty(self) -> bool:
        return (not self.force_fix_params and not self.forbidden_actions
                and self.force_route is None
                and self.suggested_compartments is None
                and not self.suggested_covariates
                and self.absorption_speed is None)

    def to_log_dict(self) -> dict:
        return {
            "force_fix_params": sorted(self.force_fix_params),
            "forbidden_actions": sorted(self.forbidden_actions),
            "force_route": self.force_route,
            "suggested_compartments": self.suggested_compartments,
            "suggested_covariates": list(self.suggested_covariates),
            "absorption_speed": self.absorption_speed,
        }


INTERPRETER_PROMPT = """You are the Hint Interpreter for the PKAgent
pipeline.  The user has written a free-form natural-language brief
about the drug or study.  Convert that brief into STRUCTURED
CONSTRAINTS that downstream agents (and deterministic gates like the
LRT verifier) will treat as AUTHORITATIVE.

The user's words are ground truth.  Your job is to translate them
into machine-checkable directives — be conservative (only set
constraints the hint clearly supports) but explicit.

OUTPUT — STRICT JSON, no markdown:

{
  "force_fix_params": ["ALAG", ...],   // params the user wants fixed at 0
  "forbidden_actions": ["release ALAG", ...],  // reviewer actions to block
  "force_route": "oral" | "iv" | "iv_infusion" | null,
  "suggested_compartments": 1 | 2 | null,
  "suggested_covariates": ["CLCR", "WT", "AGE", ...],
  "absorption_speed": "rapid" | "slow" | "delayed" | null,
  "rationale": "short prose tying each constraint back to the hint"
}

TRANSLATION RULES (apply only when the hint clearly supports them):

  ABSORPTION:
  • "rapid absorption" / "IR tablet" / "oral solution" / "no lag" /
    "immediate-release" / "well absorbed"
       → absorption_speed="rapid"
       → force_fix_params includes "ALAG"
       → forbidden_actions includes "release ALAG"
  • "delayed absorption" / "absorption lag" / "enteric-coated" /
    "modified-release" / "coated tablet with lag" / "slow onset"
       → absorption_speed="delayed"
       → forbidden_actions includes "fix ALAG = 0"
       (do NOT force_fix; let the LRT release ALAG with confidence)
  • Silent on absorption → leave absorption_speed=null, no ALAG constraint

  ROUTE:
  • "oral" / "PO" / "by mouth" / "tablet" / "capsule" → force_route="oral"
  • "IV bolus" / "intravenous bolus" / "IV push" → force_route="iv"
  • "IV infusion" / "intravenous infusion" / "continuous infusion"
       → force_route="iv_infusion"
  • Silent → null

  MODEL ORDER:
  • "one-compartment" / "monoexponential" / "1-cmt" / "linear PK"
       → suggested_compartments=1
  • "two-compartment" / "biexponential" / "2-cmt" / "distribution phase"
       → suggested_compartments=2
  • Silent → null

  COVARIATES (only add to list when the hint clearly implicates):
  • "renally cleared" / "renal elimination" / "kidney" / "CrCL-dependent"
       → suggested_covariates += ["CLCR"]
  • "weight-dependent" / "allometric" / "pediatric"
       → suggested_covariates += ["WT"]
  • "age-related" / "elderly" / "geriatric"
       → suggested_covariates += ["AGE"]
  • "sex-dependent" / "gender-difference"
       → suggested_covariates += ["SEX"]

  WHAT NOT TO DO:
  • Do NOT invent constraints from the drug NAME (the hint may
    mention a drug; do not infer beyond what the hint explicitly says).
  • Do NOT set numeric thresholds (no "Ka > 2", no "V < 100").
  • If the hint is silent on a field, leave it null/empty.
  • If the hint is missing or empty, return all-empty constraints.

Return STRICT JSON.  The orchestrator will treat your output as the
user's authoritative directives.
"""


def interpret(hint: str | None) -> tuple[HintConstraints, AgentRunLog]:
    """Convert a natural-language user hint into HintConstraints.

    Returns an empty HintConstraints (with raw_text=None) if hint is
    None or empty.  Otherwise calls the LLM hint interpreter; if the
    LLM backend is unavailable, returns an empty-but-raw-text-bearing
    constraints object (so LLM agents still see the raw hint).
    """
    if not hint or not hint.strip():
        return HintConstraints(), AgentRunLog(
            "hint_interpreter", {"hint": None},
            {"constraints": "empty"}, ["no-hint"])

    if not SHARED.enabled:
        out = HintConstraints(raw_text=hint)
        return out, AgentRunLog(
            "hint_interpreter", {"hint": hint},
            {"constraints": out.to_log_dict()},
            ["llm-unavailable; raw-text-only"])

    response = SHARED.chat_json([INTERPRETER_PROMPT], hint, max_tokens=1024)
    if not isinstance(response, dict):
        out = HintConstraints(raw_text=hint)
        return out, AgentRunLog(
            "hint_interpreter", {"hint": hint},
            {"constraints": out.to_log_dict(),
             "llm_response": str(response)[:200]},
            ["llm-malformed; raw-text-only"])

    def _safe_list(v):
        if v is None: return []
        if isinstance(v, list): return [str(x) for x in v if x]
        if isinstance(v, str) and v: return [v]
        return []

    def _safe_int(v):
        if v is None: return None
        try:
            iv = int(v)
            return iv if iv in (1, 2) else None
        except (ValueError, TypeError):
            return None

    def _safe_str(v, allowed=None):
        if v is None: return None
        s = str(v).strip().lower()
        if not s or s in ("none", "null"): return None
        if allowed and s not in allowed:
            return None
        return s

    constraints = HintConstraints(
        force_fix_params=set(_safe_list(response.get("force_fix_params"))),
        forbidden_actions=set(_safe_list(response.get("forbidden_actions"))),
        force_route=_safe_str(response.get("force_route"),
                               allowed={"oral", "iv", "iv_infusion"}),
        suggested_compartments=_safe_int(response.get("suggested_compartments")),
        suggested_covariates=_safe_list(response.get("suggested_covariates")),
        absorption_speed=_safe_str(response.get("absorption_speed"),
                                     allowed={"rapid", "slow", "delayed"}),
        raw_text=hint,
    )
    return constraints, AgentRunLog(
        "hint_interpreter", {"hint": hint},
        {"constraints": constraints.to_log_dict(),
         "rationale": response.get("rationale", "")},
        ["llm-mode"])
