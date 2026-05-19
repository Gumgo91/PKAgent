"""High-level engine: wire NLME + structural models + class priors.

This module is keyed on **drug class**, never on drug name.  Callers
identify the pharmacological class (e.g. 'oral_small_molecule_with_lag'
or 'iv_aminoglycoside') and the engine looks up the corresponding broad
prior from priors.CLASSES.
"""
from __future__ import annotations

from typing import Callable
import numpy as np

from .models import REGISTRY, one_comp_iv, one_comp_oral, two_comp_iv, two_comp_iv_infusion
from .nlme import NLMEConfig, Subject, fit_population, FitResult
from .priors import CLASSES, get_class, physio_plausibility
from .kernels import predict_2cmt_iv_multidose as _jit_multidose_2cmt_iv


def _dose_history_arrays(subject):
    """Materialise Subject.dose_history into two parallel float64 arrays.

    Caches the result on the Subject instance so the conversion happens
    once per subject, not once per predict_fn call.
    """
    arr = getattr(subject, "_dose_arrays", None)
    if arr is not None:
        return arr
    history = subject.dose_history
    times = np.array([d[0] for d in history], dtype=np.float64)
    amts  = np.array([d[1] for d in history], dtype=np.float64)
    arr = (times, amts)
    subject._dose_arrays = arr   # type: ignore[attr-defined]
    return arr


def _multidose_2cmt_iv(times, subject, CL, V1, Q, V2):
    """JIT-compiled multi-dose 2cmt IV-bolus superposition."""
    dose_times, dose_amts = _dose_history_arrays(subject)
    return _jit_multidose_2cmt_iv(np.asarray(times, dtype=np.float64),
                                    dose_times, dose_amts,
                                    float(CL), float(V1), float(Q), float(V2))


def make_predict_fn(model_name: str) -> Callable:
    """Return a predict_fn(theta_dict, subject) -> ndarray suitable for nlme.fit_population."""
    if model_name == "1cmt_iv":
        return lambda p, s: one_comp_iv(np.asarray(s.time), s.dose, p["CL"], p["V"])
    if model_name == "1cmt_oral":
        return lambda p, s: one_comp_oral(np.asarray(s.time), s.dose,
                                          p["CL"], p["V"], p["Ka"],
                                          ALAG=p.get("ALAG", 0.0))
    if model_name == "2cmt_iv":
        # Supports both single-dose (uses Subject.dose) and multi-dose via
        # the optional Subject.dose_history list of (time, amt) pairs.
        # Multi-dose uses a JIT-compiled superposition kernel (~30× the
        # Python loop) so 97-subject TDM datasets stay tractable.
        def _predict(p, s):
            if getattr(s, "dose_history", None):
                return _multidose_2cmt_iv(s.time, s,
                                           p["CL"], p["V1"], p["Q"], p["V2"])
            return two_comp_iv(np.asarray(s.time), s.dose,
                                p["CL"], p["V1"], p["Q"], p["V2"])
        return _predict
    if model_name == "2cmt_inf":
        return lambda p, s: two_comp_iv_infusion(np.asarray(s.time), s.dose, s.tinf,
                                                 p["CL"], p["V1"], p["Q"], p["V2"])
    raise ValueError(model_name)


def build_config(drug_class: str | None, model_name: str,
                  covariate_effects: dict | None = None,
                  use_priors: bool = True,
                  eta_params: list[str] | None = None,
                  compute_se: bool = True,
                  init_theta_override: dict | None = None,
                  init_omega_override: dict | None = None,
                  route: str | None = None,
                  prior_log_sd_multiplier: float = 1.0,
                  omega_floor: float | None = None,
                  fix_params: set[str] | None = None,
                  prior_sd_per_param: dict[str, float] | None = None,
                  ) -> NLMEConfig:
    """Construct an NLMEConfig.

    Initial values come *exclusively* from `init_theta_override` (which
    the caller produces via the data-only NCA + naive-pooled init in
    `decisions.initial.compute`).  The class catalog is consulted only
    for the Bayesian log-prior SD (a wide regulariser, not an init
    seed) — using class typicals as init would silently bias the
    optimiser toward the known answer.

    Raises ValueError if `init_theta_override` is missing any structural
    parameter the model needs; we refuse to fall back to class typicals
    or hardcoded defaults.

    `init_theta_override` / `init_omega_override` are also used by SCM
    to warm-start step-wise refits (Wählby/Jonsson/Karlsson 2002).
    """
    cat = get_class(drug_class, route=route)
    sm = REGISTRY[model_name]
    if not init_theta_override:
        raise ValueError(
            f"build_config requires init_theta_override (model={model_name}); "
            f"no class-typical fallback allowed (data-only init policy)")
    init_theta = dict(init_theta_override)
    # ALAG starts at 0 when the oral model needs it but the caller didn't
    # provide one — the engine then optimises it from data.  Not a class
    # fallback: 0 is the "no lag" neutral starting point.
    if model_name == "1cmt_oral":
        init_theta.setdefault("ALAG", 0.0)
    missing = [p for p in sm.params if p not in init_theta]
    if missing:
        raise ValueError(
            f"init_theta_override missing required params {missing} "
            f"for model {model_name!r} (data-only init policy: no defaults)")

    if eta_params is None:
        eta_params = [p for p in sm.params if p != "ALAG"]
    init_omega = {p: 0.1 for p in eta_params}
    if init_omega_override:
        for k, v in init_omega_override.items():
            if k in init_omega:
                init_omega[k] = v

    log_sd = None
    if use_priors:
        # Class log_sd is a *wide* Bayesian regulariser, not an init seed.
        # Falling back to 2.0 (≈e^2 ≈ 7× spread) when class doesn't list
        # a param is still wide enough that it does not encode the answer.
        log_sd = {p: cat.log_sd.get(p, 2.0) * prior_log_sd_multiplier
                  for p in init_theta}
        # Per-parameter override: when the Clinician reviewer asks to
        # "tighten the Ka prior" the orchestrator multiplies the
        # specific parameter's SD here.  This is the engine-level
        # analogue of editing `$PRIOR THETA` for a single parameter
        # in a NONMEM control stream.
        if prior_sd_per_param:
            for p, mult in prior_sd_per_param.items():
                if p in log_sd:
                    log_sd[p] = log_sd[p] * float(mult)

    cfg = NLMEConfig(
        model_name=model_name,
        eta_params=eta_params,
        init_theta=init_theta,
        init_omega=init_omega,
        init_sigma_prop=0.15,
        init_sigma_add=0.05,
        covariate_effects=covariate_effects or {},
        log_prior_sd=log_sd,
        fix=set(fix_params) if fix_params else set(),
        max_iter=80,
        compute_se=compute_se,
    )
    if omega_floor is not None:
        cfg.omega_floor = omega_floor
    return cfg


# Backwards-compat alias for callers that still use the old name.
build_config_for_drug = build_config


def run_fit(drug_class: str | None, subjects: list[Subject], model_name: str,
            covariate_effects: dict | None = None,
            use_priors: bool = True,
            eta_params: list[str] | None = None,
            compute_se: bool = True,
            init_theta_override: dict | None = None,
            init_omega_override: dict | None = None,
            route: str | None = None,
            fix_params: set[str] | None = None,
            prior_sd_per_param: dict[str, float] | None = None,
            ) -> FitResult:
    cfg = build_config(drug_class, model_name, covariate_effects, use_priors,
                        eta_params, compute_se=compute_se,
                        init_theta_override=init_theta_override,
                        init_omega_override=init_omega_override,
                        route=route,
                        fix_params=fix_params,
                        prior_sd_per_param=prior_sd_per_param)
    predict_fn = make_predict_fn(model_name)
    fit = fit_population(subjects, cfg, predict_fn)
    return fit
