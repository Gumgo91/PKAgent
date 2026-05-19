"""Top-level PKAgent API — data-in / fit-out, no drug-name dependency.

User-facing API:

    from pkagent import fit
    output = fit(subjects,
                 route="oral",              # required metadata
                 drug_class="oral_small_molecule_with_lag",  # optional class label
                 covariate_hints=["CL~WT"]) # optional user knowledge

What the fit() function does NOT receive:
  * the drug NAME (e.g. "warfarin").  PKAgent never looks up drug-specific
    parameter values.  Only the drug CLASS (broad pharmacological group)
    is consulted, and class priors are deliberately wide so they do not
    encode the answer to any particular benchmark drug.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .decisions import structural as _structural
from .decisions import initial as _initial
from .decisions import covariate as _covariate
from .decisions import evaluator as _evaluator
from .decisions import arbitrator as _arbitrator
from .decisions import strategy as _strategy
from .decisions import hint_interpreter as _hint_interpreter
from .diagnostics import compute_eta_boundary_lrt, data_anchors_from_subjects
from .engine import run_fit
from .llm_client import AgentRunLog
from .models import REGISTRY
from .nlme import FitResult, Subject
from .scm import run_ensemble, stepwise_covariate_modeling


# χ²₁ at α=0.05 — the only universal "constant" in the LRT verifier.
# Per Δdf the critical value is df × this (Wilks-style scaling for
# closely-related nested models); for the rare Δdf > 1 cases we use
# 3.84·Δdf as the threshold (deliberately conservative — favours
# keeping the simpler model unless data clearly supports the fuller one).
LRT_CRIT_1DF_A05 = 3.84


def _lrt_verify_reduction(
        action_label: str,
        current_fit: "FitResult",
        drug_class: str | None,
        subjects: list["Subject"],
        target_model_name: str,
        target_eta_params: list[str],
        target_init_theta: dict,
        target_init_omega: dict,
        target_fix_params: set[str],
        target_covariate_effects: dict,
        target_prior_sd_per_param: dict,
        route: str | None,
        ddf: int,
        verbose: bool = False,
        ) -> tuple[bool, dict]:
    """Run one quick fit under the PROPOSED reduction and compare to
    the current fit via LRT.

    The reduction is statistically supported (= the simpler model is
    acceptable) when ΔOFV = OFV_reduced − OFV_current ≤ 3.84·Δdf.
    Otherwise the data REJECT the reduction and the action is dropped.

    Returns `(passes, info)`.  `info["test_fit"]` is the candidate
    FitResult when passes=True so callers can reuse it instead of
    re-fitting.
    """
    try:
        test_fit = run_fit(
            drug_class, subjects, target_model_name,
            covariate_effects=target_covariate_effects,
            eta_params=target_eta_params, use_priors=True,
            compute_se=False,
            init_theta_override=target_init_theta,
            init_omega_override=target_init_omega,
            route=route,
            fix_params=target_fix_params,
            prior_sd_per_param=target_prior_sd_per_param,
        )
        ofv_curr = float(getattr(current_fit, "ofv_foce",
                                  getattr(current_fit, "ofv", float("nan"))))
        ofv_test = float(getattr(test_fit, "ofv_foce",
                                  getattr(test_fit, "ofv", float("nan"))))
        d_ofv = ofv_test - ofv_curr
        crit = LRT_CRIT_1DF_A05 * max(1, int(ddf))
        passes = bool(np.isfinite(d_ofv) and d_ofv <= crit)
        info = {
            "action": action_label,
            "ofv_current": ofv_curr,
            "ofv_test": ofv_test,
            "d_ofv": d_ofv,
            "ddf": ddf,
            "crit": crit,
            "verdict": "pass" if passes else "reject (data supports current/fuller)",
            "test_fit": test_fit if passes else None,
        }
        if verbose:
            print(f"    [LRT-verify] {action_label}: "
                  f"ΔOFV={d_ofv:+.2f} (df={ddf}, crit={crit:.2f}) → "
                  f"{info['verdict']}")
        return passes, info
    except Exception as e:
        return False, {"action": action_label, "error": str(e),
                       "verdict": "reject (verifier error)"}


@dataclass
class RoundTrace:
    """One row of the reviewer-feedback iteration log.

    Captures the state immediately AFTER the round's fit finished and
    BEFORE the orchestrator applied any of the round's actions.  The
    list of these (returned in `FitOutput.feedback_trace`) is the
    PK-expert-style narrative of how the fit evolved.
    """
    round: int                              # 0 = initial fit, 1+ = post-feedback re-fits
    ofv_foce: float                         # NONMEM-comparable FOCE-I OFV at this round
    model_name: str                         # structural model at this round
    eta_params: list[str]                   # active η parameters
    theta: dict                             # θ̂ at this round (rounded)
    omega_diag: dict                        # ω² at this round
    shrinkage: dict                         # η-shrinkage % at this round
    statistician_concerns: list[str]
    statistician_suggestions: list[str]
    clinician_concerns: list[str]
    clinician_suggestions: list[str]
    actions_applied: list[str]              # plan-edits the orchestrator made *after* this round
    accepted: bool                          # both reviewers accepted (terminal round)
    final_selected: bool = False            # True for the round whose fit became FitOutput.final_fit (post-OFV-revert)
    rejected_actions: list[dict] = field(default_factory=list)  # LRT/AIC-rejected reviewer suggestions


@dataclass
class FitOutput:
    """End-to-end fit result."""
    final_fit: FitResult
    model_name: str
    eta_params: list[str]
    selected_covariates: dict
    reviews: dict
    logs: list[AgentRunLog]
    feedback_trace: list[RoundTrace] = field(default_factory=list)
    arbitrator_verdict: dict | None = None    # LLM "Final Judge" rationale
                                              # (None if only one sane round)


def fit(subjects: list[Subject],
         route: str | None = None,
         drug_class: str | None = None,
         covariate_hints: list[str] | None = None,
         structural_hint: str | None = None,
         hint: str | None = None,
         max_feedback_rounds: int = 5,
         n_ensemble: int = 3,
         label: str | None = None,   # opaque tag, only used for logs/print
         verbose: bool = True) -> FitOutput:
    """End-to-end fit: dataset in → fitted population model out.

    Parameters
    ----------
    subjects : list[Subject]
        PopPK dataset (one entry per subject).  This is the only input
        that conveys actual PK information; everything else is metadata.
    route : str
        "oral" / "iv" / "iv_infusion" — the user's clinical knowledge of
        how the drug was administered.
    drug_class : str, optional
        Pharmacological class label (NOT a drug name).  Examples:
          'oral_small_molecule_with_lag', 'iv_aminoglycoside'.
        Used to pick the structural model and the broad parameter priors.
    covariate_hints : list[str], optional
        Extra covariate-parameter pairs to test, e.g. ["CL~CLCR"].  The
        user might supply these from clinical knowledge of the drug being
        studied (e.g. renally cleared drugs).
    structural_hint : str, optional
        Pre-specified compartment model name (skips structural decision).
    hint : str, optional
        Free-form natural-language brief about the drug/study that ALL
        LLM decision agents will read alongside the data.  Examples:
          "Oral IR tablet, rapid absorption, no lag. Hepatic metabolism,
           t½≈8h."
          "IV infusion aminoglycoside, renally cleared, CLCR a major
           covariate, narrow therapeutic window."
        The hint is a soft user knowledge channel — it informs the LLM
        agents' reasoning but does not bypass the data-driven LRT/AIC
        guards. The Structural agent can choose `include_alag=False`
        if the hint indicates rapid IR absorption, in which case the
        orchestrator skips the ALAG-release LRT entirely.
    label : str, optional
        Opaque identifier used only in print/log lines (does NOT enter
        any algorithmic decision).

    Returns
    -------
    FitOutput with the converged NLME fit, the decisions taken, and the
    evaluator's verdict.
    """
    logs: list[AgentRunLog] = []
    print_tag = label or drug_class or route or "fit"

    # -1) Hint Interpreter — the single point of truth for user
    #     directives.  Returns structured constraints that EVERY
    #     downstream stage (LLM agents AND deterministic gates like
    #     the LRT verifier and action parser) will honour as
    #     authoritative.  Hard constraints (force_fix_params,
    #     forbidden_actions, force_route) cannot be overridden by
    #     data-only statistics; soft constraints (suggested compartments
    #     / covariates) are advisory.
    hint_constraints, hint_log = _hint_interpreter.interpret(hint)
    logs.append(hint_log)
    if verbose and not hint_constraints.is_empty():
        print(f"  [{print_tag}] hint constraints: {hint_constraints.to_log_dict()}")
    # User-forced route from hint (only when caller didn't supply one)
    if route is None and hint_constraints.force_route:
        route = hint_constraints.force_route

    # 0) Gemini-driven strategy decision — picks knobs (n_ensemble,
    #    prior_strength, inner_optimizer, omega_floor) based on dataset
    #    shape + class metadata.  This is PKAgent's main adaptive layer.
    strategy, log = _strategy.decide(subjects, drug_class=drug_class,
                                       route=route, hint=hint)
    logs.append(log)
    if verbose:
        print(f"  [{print_tag}] strategy: n_ens={strategy.n_ensemble}, "
              f"prior={strategy.prior_strength} (×{strategy.prior_log_sd_multiplier}), "
              f"inner={strategy.inner_optimizer}, "
              f"ω_floor={strategy.omega_floor:.1e}")
        if hint:
            print(f"  [{print_tag}] user hint: {hint!r}")
    n_ensemble = strategy.n_ensemble
    max_feedback_rounds = min(max_feedback_rounds, strategy.max_feedback_rounds)

    # 1) structural decision — DATA-DRIVEN heuristic + LLM synthesis that
    #    sees the natural-language hint.  The Structural agent can set
    #    `include_alag=False` when the hint signals no-lag absorption
    #    (e.g. "IR tablet, rapid absorption") — in that case the
    #    orchestrator will skip the ALAG-release LRT below and keep
    #    ALAG fixed at 0 throughout.
    s_dec, log = _structural.decide(subjects=subjects,
                                      route=route,
                                      structural_hint=structural_hint,
                                      hint=hint,
                                      n_subj=len(subjects))
    logs.append(log)
    model_name = s_dec["structural_model"]
    structural_include_alag = bool(s_dec.get("include_alag", True))

    sm = REGISTRY[model_name]
    eta_params = [p for p in sm.params if p != "ALAG"]

    # 2) initial values (data-driven naive-pooled)
    init_dec, log = _initial.compute(subjects, model_name,
                                       drug_class=drug_class, route=route)
    logs.append(log)
    init_theta = init_dec["init_theta"]
    init_omega = init_dec["init_omega"]

    # 3) covariate candidates — DATA-DRIVEN: discovers covariates from
    #    subjects + LLM/heuristic pharmacology rules (no drug name lookup).
    c_dec, log = _covariate.propose(subjects=subjects,
                                      model_name=model_name,
                                      route=route,
                                      covariate_hints=covariate_hints,
                                      hint=hint)
    logs.append(log)
    candidates = c_dec.get("candidates", [])

    # --- ALAG policy ---
    # ALAG always STARTS fixed at 0 (NONMEM-style opt-in).  Whether the
    # orchestrator then attempts to release it via an LRT depends on,
    # in priority order:
    #
    #   1. USER HINT (hint_constraints.force_fix_params):
    #      If the hint resolved to "ALAG" ∈ force_fix_params (e.g. user
    #      said "no lag" / "IR rapid absorption"), ALAG stays fixed at
    #      0 throughout — no LRT attempt, no reviewer override.  This
    #      is the authoritative path.
    #
    #   2. STRUCTURAL AGENT's include_alag flag:
    #      When the hint is silent, the Structural agent's AIC pre-
    #      screen + LLM judgement (which itself sees the hint as
    #      context) decides whether to attempt LRT release.
    #
    #   3. DATA-only LRT:
    #      If both upstream gates allow it, run the standard LRT
    #      (ΔOFV > 3.84) below and let data decide.
    alag_seed_for_lrt = None
    fix_params: set[str] = set(hint_constraints.force_fix_params)
    user_alag_fixed = "ALAG" in hint_constraints.force_fix_params
    if model_name == "1cmt_oral" and "ALAG" in init_theta:
        if user_alag_fixed:
            alag_seed_for_lrt = None  # user said no ALAG; never try to release
            if verbose:
                print(f"  [{print_tag}] user hint forces ALAG fixed at 0 "
                      f"(authoritative); skipping LRT-release")
        elif structural_include_alag:
            alag_seed_for_lrt = float(init_theta["ALAG"])
        else:
            alag_seed_for_lrt = None
            if verbose:
                print(f"  [{print_tag}] Structural agent says include_alag=False "
                      f"→ ALAG kept fixed at 0, skipping LRT-release")
        init_theta["ALAG"] = 1e-6   # numerically fixed at "zero"
        fix_params.add("ALAG")

    if verbose:
        print(f"  [{print_tag}] structural={model_name}, etas={eta_params}, "
              f"init θ={ {k: round(v,3) for k,v in init_theta.items()} } "
              f"({init_dec['source']}), "
              f"candidates={[c['covariate'] for c in candidates]}"
              + (f"  [ALAG opt-in: fixed=0, LRT to release]"
                 if alag_seed_for_lrt is not None else ""))

    # 4) ensemble + SCM
    best_fit, ensemble = run_ensemble(drug_class, subjects, model_name,
                                        eta_params,
                                        n_runs=n_ensemble,
                                        init_theta=init_theta,
                                        init_omega=init_omega,
                                        route=route, verbose=verbose,
                                        fix_params=fix_params)

    selected_cov = {}
    if candidates:
        selected_cov, hist = stepwise_covariate_modeling(
            drug_class, subjects, model_name, candidates,
            best_fit.ofv, best_fit, eta_params, verbose=verbose)

    # 5) final refit
    final = run_fit(drug_class, subjects, model_name,
                    covariate_effects=selected_cov,
                    eta_params=eta_params, use_priors=True,
                    compute_se=True,
                    init_theta_override=dict(best_fit.theta),
                    init_omega_override=dict(best_fit.omega_diag),
                    route=route,
                    fix_params=fix_params)

    # 5.5) ALAG release LRT — one extra fit with ALAG free; accept only
    #      if ΔOFV(FOCE-I) > 3.84 (χ²₁ at α=0.05, 1 df = ALAG).  This
    #      is the structural-model-selection step that decides whether
    #      the data supports an absorption lag.
    if alag_seed_for_lrt is not None and "ALAG" in fix_params:
        try:
            init_release = dict(final.theta)
            init_release["ALAG"] = max(alag_seed_for_lrt, 0.01)
            best_release, _ = run_ensemble(
                drug_class, subjects, model_name, eta_params,
                n_runs=n_ensemble,
                init_theta=init_release,
                init_omega=dict(final.omega_diag),
                route=route, verbose=verbose,
                fix_params=set())   # no fix → ALAG free
            cov_release = {}
            if candidates:
                cov_release, _ = stepwise_covariate_modeling(
                    drug_class, subjects, model_name, candidates,
                    best_release.ofv, best_release, eta_params,
                    verbose=verbose)
            final_release = run_fit(
                drug_class, subjects, model_name,
                covariate_effects=cov_release,
                eta_params=eta_params, use_priors=True,
                compute_se=True,
                init_theta_override=dict(best_release.theta),
                init_omega_override=dict(best_release.omega_diag),
                route=route,
                fix_params=set())
            ofv_fixed = float(getattr(final, "ofv_foce",
                                       getattr(final, "ofv", float("nan"))))
            ofv_free = float(getattr(final_release, "ofv_foce",
                                      getattr(final_release, "ofv", float("nan"))))
            d = ofv_fixed - ofv_free
            passed = (np.isfinite(d) and d > 3.84)
            if verbose:
                print(f"  [{print_tag}] ALAG-release LRT: OFV {ofv_fixed:.2f} "
                      f"(ALAG=0) vs {ofv_free:.2f} (ALAG free), "
                      f"ΔOFV={d:+.2f} → {'release' if passed else 'keep fixed'}")
            if passed:
                final = final_release
                selected_cov = cov_release
                fix_params = set()
            # else: keep ALAG fixed; `final` already reflects that
        except Exception as e:
            if verbose:
                print(f"  [{print_tag}] ALAG-release LRT failed ({e}); "
                      f"keeping ALAG fixed at 0")

    # 6) evaluator feedback loop
    #
    # The reviewer agents' suggestions are translated into plan-edits
    # that mirror what a NONMEM user would do in the control stream:
    #   - drop ETA on k        → remove from $OMEGA  (eta_params)
    #   - force-test cov X     → add to SCM candidates
    #   - swap V1↔V2 init      → reorder $THETA initial estimates
    #   - tighten <P> prior    → narrow $PRIOR THETA SD for parameter P
    #   - fix <P> = 0          → $THETA (0 FIX) for parameter P
    #   - downgrade to 1-cmt   → change $SUBROUTINES + redo $PK
    #   - re-init from NCA     → recompute $THETA from the data
    # The orchestrator carries these as state between rounds and feeds
    # them to run_ensemble / run_fit; the NLME core then minimises a
    # different objective (same FOCE-I formula, different plan).
    prev_plan_fingerprint = None
    reviews = None
    # fix_params already initialised by the ALAG opt-in step above (may be
    # empty if the LRT released ALAG, or {"ALAG"} if the LRT kept it fixed).
    prior_sd_per_param: dict[str, float] = {}
    feedback_trace: list[RoundTrace] = []
    # Track every round's fit so we can revert to the best-OFV one if a
    # later reviewer-driven plan-edit produced a worse fit.  Real PK
    # experts do the same: if my edit raised OFV by 20 units, I undo
    # the edit rather than commit to the worse fit.
    round_fits: list[tuple[FitResult, str, list[str], dict]] = [
        (final, model_name, list(eta_params), dict(selected_cov)),
    ]
    for round_idx in range(max_feedback_rounds + 1):
        cov_effects_full = {k: ("power", v) for k, v in final.covariate_effects.items()}
        eta_lrt = compute_eta_boundary_lrt(drug_class, subjects, model_name,
                                            eta_params, final, cov_effects_full)
        reviews, log = _evaluator.evaluate(final, drug_class, cov_effects_full,
                                            candidates, eta_lrt=eta_lrt,
                                            subjects=subjects, hint=hint)
        log.notes.append(
            f"eta_lrt={{ {', '.join(f'{k}: {round(v,2) if v is not None else None}' for k,v in eta_lrt.items())} }}")
        # Log the Clinician's concerns so the run log makes its activity
        # visible (previously only the LRT dict was logged).
        clin_concerns = reviews["clinician"].get("concerns", [])
        if clin_concerns:
            log.notes.append(
                "clinician concerns: " + "; ".join(clin_concerns[:5]))
        logs.append(log)

        # Capture the round trace BEFORE we apply any plan-edits.
        # `actions_applied` is filled in just after the actions are
        # decided (a few lines below); we append `trace` to the list
        # only at the end of this iteration's branch to keep the order.
        trace = RoundTrace(
            round=round_idx,
            ofv_foce=float(getattr(final, "ofv_foce", float("nan"))),
            model_name=model_name,
            eta_params=list(eta_params),
            theta={k: round(v, 4) for k, v in final.theta.items()},
            omega_diag={k: float(v) for k, v in final.omega_diag.items()},
            shrinkage={k: round(v, 1) for k, v in final.shrinkage.items()},
            statistician_concerns=list(reviews["statistician"].get("concerns", [])),
            statistician_suggestions=list(reviews["statistician"].get("suggestions", [])),
            clinician_concerns=list(reviews["clinician"].get("concerns", [])),
            clinician_suggestions=list(reviews["clinician"].get("suggestions", [])),
            actions_applied=[],     # filled in below
            accepted=False,          # filled in below
        )

        if not _evaluator.evaluations_block_acceptance(reviews):
            trace.accepted = True
            feedback_trace.append(trace)
            break
        if round_idx == max_feedback_rounds:
            trace.actions_applied = ["max-feedback-rounds cap"]
            feedback_trace.append(trace)
            break

        stat_sugg = reviews["statistician"]["suggestions"]
        clin_sugg = reviews["clinician"]["suggestions"]
        actions_applied: list[str] = []
        # rejected_actions accumulates LRT-rejected reviewer suggestions for
        # transparency (reported in feedback_trace + FitOutput).
        rejected_actions: list[dict] = []

        # --- PHASE 1: PARSE suggestions into candidate actions (no apply yet) ---
        # Helper: reject an action immediately when the user hint forbids it
        # (authoritative — bypasses LRT/AIC verification).
        def _hint_forbids(action_label: str) -> bool:
            if not hint_constraints.forbidden_actions:
                return False
            al = action_label.lower()
            for fb in hint_constraints.forbidden_actions:
                if fb.lower() in al or al in fb.lower():
                    return True
            return False

        new_eta = list(eta_params)
        candidate_fix_targets: list[str] = []   # params to test fixing at 0
        candidate_release_alag = False
        candidate_swap_v1_v2 = False
        candidate_downgrade_to: str | None = None
        candidate_renca_init = False

        # Statistician: drop ETA — safe (boundary LRT pre-validated)
        for s in stat_sugg:
            if "drop ETA on" in s:
                p = s.split("drop ETA on")[-1].strip()
                if p in new_eta:
                    new_eta.remove(p)
                    actions_applied.append(f"drop ETA on {p}")

        # Clinician
        for c in clin_sugg:
            c_lower = c.lower()

            # (a) force-test covariate — safe (becomes SCM candidate)
            if "force-test" in c:
                tok = c.split("force-test")[1].split("on")[0].strip()
                if tok and tok not in [cc["covariate"] for cc in candidates]:
                    candidates.append({"param": "CL", "covariate": tok,
                                       "kind": "power",
                                       "rationale": "reviewer-forced"})
                    actions_applied.append(f"force-test {tok} on CL")

            # (b) swap V1↔V2 — reparameterisation; safe (next refit decides)
            if (("v1" in c_lower and "v2" in c_lower and "swap" in c_lower)
                    or "swap v1" in c_lower):
                candidate_swap_v1_v2 = True

            # (c) tighten <P> prior — soft regularisation; safe
            if ("tighten" in c_lower and "prior" in c_lower) \
                    or "narrower prior" in c_lower:
                for p in list(init_theta.keys()):
                    if p.lower() in c_lower:
                        prior_sd_per_param[p] = min(
                            prior_sd_per_param.get(p, 1.0), 0.3)
                        actions_applied.append(f"tighten {p} prior (×0.3)")

            # (d) fix <P> = 0 — RISKY: requires LRT verification
            #     (unless user-hint already FORCES the fix, in which case
            #      it's already in fix_params at orchestrator entry)
            for p in list(init_theta.keys()):
                if (("fix" in c_lower or "remove" in c_lower)
                        and p.lower() in c_lower
                        and ("at 0" in c_lower or "= 0" in c_lower
                              or "to 0" in c_lower or "= 0.0" in c_lower
                              or f"{p.lower()}=0" in c_lower.replace(" ", ""))):
                    if p not in fix_params and p not in candidate_fix_targets:
                        if _hint_forbids(f"fix {p} = 0"):
                            rejected_actions.append({
                                "action": f"fix {p} = 0",
                                "verdict": "blocked by user hint "
                                           "(forbidden_actions)"})
                        else:
                            candidate_fix_targets.append(p)

            # (d2) release ALAG — RISKY: requires LRT verification
            #     ALSO blocked when the user hint forbids it (e.g. "no lag"
            #     resolves to forbidden_actions ⊇ {"release ALAG"}).
            if ("release alag" in c_lower or "unfix alag" in c_lower
                    or "free alag" in c_lower):
                if "ALAG" in fix_params:
                    if _hint_forbids("release ALAG") or "ALAG" in hint_constraints.force_fix_params:
                        rejected_actions.append({
                            "action": "release ALAG",
                            "verdict": "blocked by user hint "
                                       "(forbidden_actions or force_fix_params)"})
                    else:
                        candidate_release_alag = True

            # (e) downgrade to 1-cmt — RISKY: requires AIC verification
            #     Blocked when user hint suggests a higher compartment
            #     count (e.g. "two-compartment kinetics").
            if ("downgrade" in c_lower
                    or "1-cmt" in c_lower or "1-compartment" in c_lower
                    or "one-compartment" in c_lower
                    or "1cmt" in c_lower):
                if hint_constraints.suggested_compartments == 2:
                    rejected_actions.append({
                        "action": "downgrade to 1-cmt",
                        "verdict": "blocked by user hint "
                                   "(suggested_compartments=2)"})
                elif _hint_forbids("downgrade to 1-cmt"):
                    rejected_actions.append({
                        "action": "downgrade to 1-cmt",
                        "verdict": "blocked by user hint (forbidden_actions)"})
                else:
                    if "oral" in (route or model_name or ""):
                        candidate_downgrade_to = "1cmt_oral"
                    else:
                        candidate_downgrade_to = "1cmt_iv"

            # (f) re-NCA init — safe (just resets initials)
            if (("nca" in c_lower and "re" in c_lower)
                    or "recompute init" in c_lower
                    or "redo initial" in c_lower):
                candidate_renca_init = True

        # --- PHASE 2: VERIFY risky candidates via LRT/AIC before applying ---

        # (d) fix <P> = 0 verification: fit with the param fixed; reject
        #     if the reduction is statistically rejected (ΔOFV > 3.84).
        for p in candidate_fix_targets:
            trial_fix = set(fix_params) | {p}
            trial_init = dict(init_theta)
            trial_init[p] = 1e-3   # numerically "zero"
            passes, info = _lrt_verify_reduction(
                action_label=f"fix {p} = 0",
                current_fit=final,
                drug_class=drug_class,
                subjects=subjects,
                target_model_name=model_name,
                target_eta_params=eta_params,
                target_init_theta=trial_init,
                target_init_omega=init_omega,
                target_fix_params=trial_fix,
                target_covariate_effects=selected_cov,
                target_prior_sd_per_param=prior_sd_per_param,
                route=route,
                ddf=1,
                verbose=verbose,
            )
            if passes:
                fix_params.add(p)
                init_theta[p] = 1e-3
                actions_applied.append(f"fix {p} = 0 (LRT-verified)")
            else:
                rejected_actions.append(info)

        # (d2) release ALAG verification: a release is an ADDITION (1 df
        # more), so the LRT direction is reversed.  We fit ALAG-free and
        # check whether ΔOFV(free vs current) > 3.84 in favour of free.
        if candidate_release_alag:
            trial_fix = set(fix_params) - {"ALAG"}
            trial_init = dict(init_theta)
            if alag_seed_for_lrt is not None:
                trial_init["ALAG"] = max(alag_seed_for_lrt, 0.01)
            else:
                trial_init["ALAG"] = 0.05
            try:
                test_fit = run_fit(
                    drug_class, subjects, model_name,
                    covariate_effects=selected_cov,
                    eta_params=eta_params, use_priors=True,
                    compute_se=False,
                    init_theta_override=trial_init,
                    init_omega_override=init_omega,
                    route=route,
                    fix_params=trial_fix,
                    prior_sd_per_param=prior_sd_per_param)
                ofv_curr = float(getattr(final, "ofv_foce",
                                          getattr(final, "ofv", float("nan"))))
                ofv_free = float(getattr(test_fit, "ofv_foce",
                                          getattr(test_fit, "ofv", float("nan"))))
                d_ofv = ofv_curr - ofv_free  # positive = free is better
                if np.isfinite(d_ofv) and d_ofv > LRT_CRIT_1DF_A05:
                    fix_params.discard("ALAG")
                    init_theta["ALAG"] = trial_init["ALAG"]
                    actions_applied.append("release ALAG (LRT-verified)")
                    if verbose:
                        print(f"    [LRT-verify] release ALAG: "
                              f"ΔOFV={d_ofv:+.2f} > {LRT_CRIT_1DF_A05} → pass")
                else:
                    rejected_actions.append({
                        "action": "release ALAG",
                        "ofv_current": ofv_curr, "ofv_test": ofv_free,
                        "d_ofv_in_favour_of_release": d_ofv,
                        "crit": LRT_CRIT_1DF_A05,
                        "verdict": "reject (data does not support adding ALAG)",
                    })
                    if verbose:
                        print(f"    [LRT-verify] release ALAG: "
                              f"ΔOFV={d_ofv:+.2f} ≤ {LRT_CRIT_1DF_A05} → reject")
            except Exception as e:
                rejected_actions.append({"action": "release ALAG",
                                          "error": str(e)})

        # (e) downgrade verification via AIC: a downgrade is allowed if the
        # simpler-model fit's AIC is no worse than the current AIC.
        downgrade_to = None
        if candidate_downgrade_to and candidate_downgrade_to != model_name:
            try:
                # Refresh init values for the new structural model
                init_dec_test, _log_test = _initial.compute(
                    subjects, candidate_downgrade_to,
                    drug_class=drug_class, route=route)
                new_sm = REGISTRY[candidate_downgrade_to]
                new_eta_test = [p for p in new_sm.params if p != "ALAG"]
                test_fit = run_fit(
                    drug_class, subjects, candidate_downgrade_to,
                    covariate_effects={},
                    eta_params=new_eta_test, use_priors=True,
                    compute_se=False,
                    init_theta_override=init_dec_test["init_theta"],
                    init_omega_override=init_dec_test["init_omega"],
                    route=route,
                    fix_params=set(),
                )
                aic_curr = float(getattr(final, "aic", float("nan")))
                aic_test = float(getattr(test_fit, "aic", float("nan")))
                d_aic = aic_test - aic_curr
                # AIC selection (Burnham & Anderson 2002): smaller AIC wins
                if np.isfinite(d_aic) and d_aic <= 0:
                    downgrade_to = candidate_downgrade_to
                    actions_applied.append(
                        f"downgrade {model_name} → {candidate_downgrade_to} "
                        f"(AIC-verified, ΔAIC={d_aic:+.2f})")
                    if verbose:
                        print(f"    [AIC-verify] downgrade: "
                              f"ΔAIC={d_aic:+.2f} ≤ 0 → pass")
                else:
                    rejected_actions.append({
                        "action": f"downgrade {model_name} → {candidate_downgrade_to}",
                        "aic_current": aic_curr, "aic_test": aic_test,
                        "d_aic": d_aic,
                        "verdict": "reject (current model has lower AIC)",
                    })
                    if verbose:
                        print(f"    [AIC-verify] downgrade: "
                              f"ΔAIC={d_aic:+.2f} > 0 → reject")
            except Exception as e:
                rejected_actions.append({
                    "action": f"downgrade {model_name} → {candidate_downgrade_to}",
                    "error": str(e),
                })

        # Soft actions that don't need LRT verification
        swap_v1_v2 = candidate_swap_v1_v2
        renca_init = candidate_renca_init

        # --- Apply structural-level edits (model downgrade) ---
        if downgrade_to and downgrade_to != model_name:
            if verbose:
                print(f"  [{print_tag}] Clinician requested downgrade "
                      f"{model_name} → {downgrade_to}")
            actions_applied.append(f"downgrade {model_name} → {downgrade_to}")
            model_name = downgrade_to
            sm = REGISTRY[model_name]
            # Reset both eta_params AND new_eta — the snapshot taken
            # before the downgrade still carries pre-downgrade ETA
            # names (V1/Q/V2 for 2-cmt) and would crash the next
            # ensemble on a 1-cmt model.
            eta_params = [p for p in sm.params if p != "ALAG"]
            new_eta = list(eta_params)
            # Refresh init values from NCA for the new model.
            init_dec, log_init = _initial.compute(
                subjects, model_name, drug_class=drug_class, route=route)
            init_theta = init_dec["init_theta"]
            init_omega = init_dec["init_omega"]
            logs.append(log_init)
            renca_init = False   # already done
            fix_params = {p for p in fix_params if p in init_theta}
            prior_sd_per_param = {
                p: v for p, v in prior_sd_per_param.items() if p in init_theta}

        # --- Apply init-level edits ---
        if swap_v1_v2 and "V1" in init_theta and "V2" in init_theta:
            init_theta["V1"], init_theta["V2"] = (init_theta["V2"],
                                                    init_theta["V1"])
            actions_applied.append(
                f"swap V1↔V2 init (V1={init_theta['V1']:.3g}, "
                f"V2={init_theta['V2']:.3g})")
            if verbose:
                print(f"  [{print_tag}] Clinician requested V1↔V2 swap → "
                      f"new init θ_V1={init_theta['V1']:.3g}, "
                      f"θ_V2={init_theta['V2']:.3g}")
        if renca_init:
            init_dec, log_init = _initial.compute(
                subjects, model_name, drug_class=drug_class, route=route)
            init_theta = init_dec["init_theta"]
            init_omega = init_dec["init_omega"]
            # Preserve "fixed at 0" semantics: re-NCA reseeds with
            # data-derived values (incl. ALAG≈0.1h from first-positive-obs),
            # which would silently un-fix any param currently in fix_params.
            # Restore the 1e-3 sentinel for fixed params so they remain
            # numerically pinned to zero.
            for p in fix_params:
                if p in init_theta:
                    init_theta[p] = 1e-3
            logs.append(log_init)
            actions_applied.append("re-NCA initial values")
            if verbose:
                print(f"  [{print_tag}] Clinician requested re-NCA init → "
                      f"new init θ={ {k: round(v,3) for k,v in init_theta.items()} }")

        if verbose and (fix_params or prior_sd_per_param):
            print(f"  [{print_tag}] active plan-edits: "
                  f"fix={sorted(fix_params)}, "
                  f"prior_tighten={prior_sd_per_param}")

        # Record the actions we decided on into the trace now that we
        # know everything that will fire this round.
        trace.actions_applied = list(actions_applied)
        trace.rejected_actions = list(rejected_actions)
        feedback_trace.append(trace)

        plan_fingerprint = (model_name,
                            tuple(new_eta),
                            tuple(sorted(c["covariate"] for c in candidates)),
                            swap_v1_v2,
                            tuple(sorted(fix_params)),
                            tuple(sorted(prior_sd_per_param.items())))
        if plan_fingerprint == prev_plan_fingerprint:
            break
        prev_plan_fingerprint = plan_fingerprint
        eta_params = new_eta or eta_params

        best_fit, _ = run_ensemble(drug_class, subjects, model_name, eta_params,
                                     n_runs=n_ensemble,
                                     init_theta=init_theta,
                                     init_omega=init_omega,
                                     route=route, verbose=verbose,
                                     fix_params=fix_params,
                                     prior_sd_per_param=prior_sd_per_param)
        selected_cov = {}
        if candidates:
            selected_cov, _ = stepwise_covariate_modeling(
                drug_class, subjects, model_name, candidates,
                best_fit.ofv, best_fit, eta_params, verbose=verbose)
        final = run_fit(drug_class, subjects, model_name,
                        covariate_effects=selected_cov,
                        eta_params=eta_params, use_priors=True,
                        compute_se=True,
                        init_theta_override=dict(best_fit.theta),
                        init_omega_override=dict(best_fit.omega_diag),
                        route=route,
                        fix_params=fix_params,
                        prior_sd_per_param=prior_sd_per_param)
        round_fits.append(
            (final, model_name, list(eta_params), dict(selected_cov)))

    # ---- Final-round selection: LLM Arbitrator + sanity guard ----
    #
    # Step 1 (sanity guard): reject clearly degenerate fits (CL≤1e-6,
    # NaN OFV).  These are optimiser collapses (e.g. tobramycin's
    # post-downgrade 1-cmt CL=0), not legitimate analyst-chosen models.
    #
    # Step 2 (LLM Arbitrator): a senior-pharmacometrician "Final Judge"
    # weighs the entire iteration log — statistical evidence (OFV, LRT,
    # shrinkage) AND biological evidence (C1-C9 PK constraints, NCA
    # anchors) — and picks the round whose fit is publishable.  This
    # is the LLM analogue of an analyst's notebook conclusion.  No
    # single deterministic rule captures the trade-off (lowest-OFV
    # undoes biology-motivated reductions; last-round ignores wrong-
    # direction edits; LRT alone misses Ka–ALAG coupling like C9).
    #
    # Step 3 (Arbitrator failure fallback): if the LLM errors mid-call,
    # fall back to "last sane round" — the NONMEM-analyst convention.
    # This is recovery from LLM error, not a hardcoded judgment rule.
    def _is_sane(fr: FitResult) -> bool:
        if not np.isfinite(getattr(fr, "ofv_foce", float("nan"))):
            return False
        for p, v in fr.theta.items():
            if not np.isfinite(v) or v <= 1e-6:
                return False
        return True

    sane_pairs = [(i, t) for i, t in enumerate(round_fits) if _is_sane(t[0])]
    arbitrator_verdict: dict | None = None
    if sane_pairs:
        if len(sane_pairs) == 1:
            chosen_idx, chosen = sane_pairs[0]
        else:
            # ---- LLM Arbitrator across all sane rounds ----
            anchors = data_anchors_from_subjects(subjects)
            round_summaries = []
            for idx, t in sane_pairs:
                tr = feedback_trace[idx] if idx < len(feedback_trace) else None
                fr = t[0]
                # n_free = structural θ - fixed + ω-diag (one per ETA)
                n_theta = len(fr.theta)
                n_fixed = len(getattr(fr, "fixed_params", set()) or set())
                n_eta = len(t[2])
                round_summaries.append({
                    "round_idx": idx,
                    "model_name": t[1],
                    "theta": {k: round(v, 4) for k, v in fr.theta.items()},
                    "ofv_foce": float(getattr(fr, "ofv_foce",
                                                getattr(fr, "ofv", float("nan")))),
                    "shrinkage": {k: round(v, 1)
                                   for k, v in (fr.shrinkage or {}).items()},
                    "statistician_concerns": (list(tr.statistician_concerns)
                                                if tr else []),
                    "clinician_concerns": (list(tr.clinician_concerns)
                                            if tr else []),
                    "actions_applied": list(tr.actions_applied) if tr else [],
                    "n_free_params": n_theta - n_fixed + n_eta,
                })
            try:
                arbitrator_verdict, alog = _arbitrator.arbitrate(
                    round_summaries, anchors, drug_class, hint=hint)
                logs.append(alog)
                chosen_idx = arbitrator_verdict["chosen_round"]
                # Look up the chosen round in sane_pairs
                chosen = None
                for idx, t in sane_pairs:
                    if idx == chosen_idx:
                        chosen = t
                        break
                if chosen is None:
                    # Shouldn't happen (arbitrator validates) but stay safe
                    chosen_idx, chosen = sane_pairs[-1]
                if verbose:
                    print(f"  [{print_tag}] Arbitrator → round {chosen_idx} "
                          f"(model={chosen[1]}, OFV={chosen[0].ofv_foce:.2f}, "
                          f"reason={arbitrator_verdict.get('primary_reason','?')})")
                    rat = arbitrator_verdict.get("rationale", "")
                    if rat:
                        print(f"    rationale: {rat[:300]}")
            except Exception as e:
                # LLM failure → NONMEM-analyst default (last sane round)
                if verbose:
                    print(f"  [{print_tag}] Arbitrator unavailable ({e}); "
                          f"falling back to last sane round")
                chosen_idx, chosen = sane_pairs[-1]
                arbitrator_verdict = {
                    "chosen_round": chosen_idx,
                    "primary_reason": "J5 (Arbitrator unavailable — fallback)",
                    "rationale": f"LLM call failed: {e}",
                    "key_evidence": [],
                    "rejected_alternatives": [],
                }
        # Apply the chosen round as the final
        if chosen[0] is not final:
            final, model_name, eta_params, selected_cov = chosen
        # Tag the trace entry whose fit became FitOutput.final_fit.
        for idx, (fr, *_rest) in enumerate(round_fits):
            if fr is final and idx < len(feedback_trace):
                feedback_trace[idx].final_selected = True
                break

    return FitOutput(final_fit=final, model_name=model_name,
                      eta_params=eta_params,
                      selected_covariates=selected_cov,
                      reviews=reviews or {},
                      logs=logs,
                      feedback_trace=feedback_trace,
                      arbitrator_verdict=arbitrator_verdict)



def print_report(out: "FitOutput", label: str | None = None) -> str:
    """Render a human-readable PK-Expert iteration log from FitOutput.

    Returns the report as a string (also prints to stdout).  Shows the
    round-by-round progression of θ, OFV, reviewer concerns, and the
    plan-edits applied — the standard "what did the agents change"
    narrative a pharmacometrician expects to see after running NONMEM.
    """
    lines: list[str] = []
    tag = label or out.model_name
    lines.append(f"\n=== PKAgent iteration log: {tag} ===")
    lines.append(f"final model: {out.model_name},  selected covariates: "
                  f"{list(out.selected_covariates) or '-'}")
    if not out.feedback_trace:
        lines.append("(no reviewer-feedback rounds recorded)")
    else:
        for tr in out.feedback_trace:
            verdict = "ACCEPTED" if tr.accepted else "modified"
            lines.append(
                f"\n  round {tr.round}  ({verdict})  model={tr.model_name}  "
                f"OFV={tr.ofv_foce:.2f}")
            theta_str = ", ".join(f"{k}={v:g}" for k, v in tr.theta.items())
            lines.append(f"    θ̂: {theta_str}")
            sh_str = ", ".join(f"{k}={v}%" for k, v in tr.shrinkage.items())
            if sh_str:
                lines.append(f"    shrinkage: {sh_str}")
            for c in tr.statistician_concerns[:4]:
                lines.append(f"    [stat] {c}")
            for c in tr.clinician_concerns[:4]:
                lines.append(f"    [clin] {c}")
            if tr.actions_applied:
                lines.append("    → actions: " + "; ".join(tr.actions_applied))
            elif tr.accepted:
                lines.append("    → no further plan-edits required")
            if tr.rejected_actions:
                for ra in tr.rejected_actions:
                    if "d_ofv" in ra:
                        lines.append(
                            f"    ✗ rejected '{ra['action']}': "
                            f"ΔOFV={ra.get('d_ofv', '?')} "
                            f"(crit={ra.get('crit', '?')}) → "
                            f"{ra.get('verdict', '?')}")
                    elif "d_aic" in ra:
                        lines.append(
                            f"    ✗ rejected '{ra['action']}': "
                            f"ΔAIC={ra.get('d_aic', '?')} → "
                            f"{ra.get('verdict', '?')}")
                    elif "d_ofv_in_favour_of_release" in ra:
                        lines.append(
                            f"    ✗ rejected '{ra['action']}': "
                            f"ΔOFV(release-current)={ra.get('d_ofv_in_favour_of_release', '?')} "
                            f"→ {ra.get('verdict', '?')}")
                    else:
                        lines.append(
                            f"    ✗ rejected '{ra['action']}': "
                            f"{ra.get('verdict', ra.get('error', '?'))}")
    if out.arbitrator_verdict:
        v = out.arbitrator_verdict
        lines.append(f"\nArbitrator (cross-round Final Judge):")
        lines.append(f"  chose round {v.get('chosen_round','?')} "
                      f"({v.get('primary_reason','?')})")
        if v.get("rationale"):
            lines.append(f"  rationale: {v['rationale']}")
        if v.get("key_evidence"):
            for ev in v["key_evidence"]:
                lines.append(f"    - {ev}")
        if v.get("rejected_alternatives"):
            for ra in v["rejected_alternatives"]:
                lines.append(f"  rejected round {ra.get('round','?')}: "
                              f"{ra.get('why','?')}")
    lines.append(f"\nfinal FOCE-I OFV: "
                  f"{getattr(out.final_fit, 'ofv_foce', float('nan')):.2f}")
    lines.append(f"final θ̂: "
                  f"{ {k: round(v,4) for k,v in out.final_fit.theta.items()} }")
    text = "\n".join(lines)
    print(text)
    return text
