"""Numba-JIT'd numerical kernels for PKAgent's hot path.

Encapsulates the innermost computations that are called O(10^4-10^6) times
during a fit:

  - structural-model concentration prediction (one_comp_iv/oral, two_comp_*)
  - per-subject joint log p(y, η | θ) used by ETA-MAP and Laplace
  - per-subject diagonal Hessian by finite difference (for Laplace correction)

All functions are @njit(fastmath=True, cache=True) and operate on flat
np.float64 arrays so Numba can fully type-infer.

The Python-facing API in models.py / nlme.py forwards into these kernels;
the rest of the codebase does not need to know Numba exists.
"""

from __future__ import annotations

import math
import numpy as np

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _maybe_njit(*args, **kwargs):
    """Return @njit if numba is available, else a no-op decorator."""
    def deco(fn):
        if HAS_NUMBA:
            return numba.njit(*args, **kwargs)(fn)
        return fn
    return deco


# ---------------------------------------------------------------------- #
# Structural-model concentration predictions
# ---------------------------------------------------------------------- #

@_maybe_njit(fastmath=True, cache=True)
def predict_1cmt_iv(t: np.ndarray, dose: float, CL: float, V: float) -> np.ndarray:
    k = CL / V
    out = np.empty_like(t)
    factor = dose / V
    for i in range(t.shape[0]):
        out[i] = factor * math.exp(-k * t[i])
    return out


@_maybe_njit(fastmath=True, cache=True)
def predict_1cmt_oral(t: np.ndarray, dose: float, CL: float, V: float,
                       Ka: float, ALAG: float) -> np.ndarray:
    k = CL / V
    out = np.empty_like(t)
    if abs(Ka - k) < 1e-8:
        coef = dose * Ka / V
        for i in range(t.shape[0]):
            tt = t[i] - ALAG
            if tt <= 0:
                out[i] = 0.0
            else:
                out[i] = coef * tt * math.exp(-k * tt)
    else:
        coef = dose * Ka / (V * (Ka - k))
        for i in range(t.shape[0]):
            tt = t[i] - ALAG
            if tt <= 0:
                out[i] = 0.0
            else:
                out[i] = coef * (math.exp(-k * tt) - math.exp(-Ka * tt))
    return out


@_maybe_njit(fastmath=True, cache=True)
def predict_2cmt_iv(t: np.ndarray, dose: float, CL: float, V1: float,
                     Q: float, V2: float) -> np.ndarray:
    k10 = CL / V1
    k12 = Q / V1
    k21 = Q / V2
    s = k10 + k12 + k21
    disc = s * s - 4.0 * k10 * k21
    if disc < 1e-12:
        disc = 1e-12
    sq = math.sqrt(disc)
    alpha = 0.5 * (s + sq)
    beta = 0.5 * (s - sq)
    A = dose / V1 * (alpha - k21) / (alpha - beta)
    B = dose / V1 * (k21 - beta) / (alpha - beta)
    out = np.empty_like(t)
    for i in range(t.shape[0]):
        out[i] = A * math.exp(-alpha * t[i]) + B * math.exp(-beta * t[i])
    return out


@_maybe_njit(fastmath=True, cache=True)
def predict_2cmt_iv_multidose(t: np.ndarray, dose_times: np.ndarray,
                                dose_amts: np.ndarray,
                                CL: float, V1: float, Q: float, V2: float) -> np.ndarray:
    """Multi-dose 2-compartment IV bolus by linear-time-invariance superposition.

    c(t) = Σ_k 1{t ≥ d_k_t} · c_single(t − d_k_t; d_k_amt).

    Used for real-data tobramycin TDM (q8h dosing).  Hot path: O(n_obs ·
    n_doses); JIT compilation gives ~30× speed-up vs the Python loop.
    """
    k10 = CL / V1
    k12 = Q / V1
    k21 = Q / V2
    s = k10 + k12 + k21
    disc = s * s - 4.0 * k10 * k21
    if disc < 1e-12:
        disc = 1e-12
    sq = math.sqrt(disc)
    alpha = 0.5 * (s + sq)
    beta = 0.5 * (s - sq)
    # per-unit-dose coefficients (multiplied by amt per dose inside the loop)
    a_coef = (alpha - k21) / (V1 * (alpha - beta))
    b_coef = (k21 - beta)  / (V1 * (alpha - beta))
    n = t.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for k in range(dose_times.shape[0]):
        dt = dose_times[k]
        amt = dose_amts[k]
        for i in range(n):
            rel = t[i] - dt
            if rel > 0.0:
                out[i] += amt * (a_coef * math.exp(-alpha * rel)
                                  + b_coef * math.exp(-beta * rel))
    return out


@_maybe_njit(fastmath=True, cache=True)
def predict_2cmt_inf(t: np.ndarray, dose: float, tinf: float,
                      CL: float, V1: float, Q: float, V2: float) -> np.ndarray:
    if tinf <= 0:
        return predict_2cmt_iv(t, dose, CL, V1, Q, V2)
    R0 = dose / tinf
    k10 = CL / V1
    k12 = Q / V1
    k21 = Q / V2
    s = k10 + k12 + k21
    disc = s * s - 4.0 * k10 * k21
    if disc < 1e-12:
        disc = 1e-12
    sq = math.sqrt(disc)
    alpha = 0.5 * (s + sq)
    beta = 0.5 * (s - sq)
    A = (1.0 / V1) * (alpha - k21) / (alpha - beta)
    B = (1.0 / V1) * (k21 - beta) / (alpha - beta)
    out = np.empty_like(t)
    for i in range(t.shape[0]):
        ti = t[i]
        if ti <= 0:
            out[i] = 0.0
        elif ti <= tinf:
            # during-infusion: F(ti)
            out[i] = R0 * (A * (1.0 - math.exp(-alpha * ti)) / alpha
                            + B * (1.0 - math.exp(-beta * ti)) / beta)
        else:
            tau = ti - tinf
            f1 = (A * (1.0 - math.exp(-alpha * ti)) / alpha
                  + B * (1.0 - math.exp(-beta * ti)) / beta)
            f2 = (A * (1.0 - math.exp(-alpha * tau)) / alpha
                  + B * (1.0 - math.exp(-beta * tau)) / beta)
            out[i] = R0 * (f1 - f2)
    return out


# ---------------------------------------------------------------------- #
# Per-subject log p(y, η | θ) — the bottleneck of the NLME inner loop.
#
# Dispatch by model_code (integer enum) so Numba can infer types cleanly:
#   0 = 1cmt_iv,  1 = 1cmt_oral,  2 = 2cmt_iv,  3 = 2cmt_inf
#
# `eta_active` is an int8 mask (1 = ETA on this param, 0 = no ETA).  This
# avoids dict lookups inside the JIT'd hot loop.
# ---------------------------------------------------------------------- #

@_maybe_njit(fastmath=True, cache=True)
def _apply_eta_4param(tv: np.ndarray, eta: np.ndarray,
                       eta_active: np.ndarray) -> np.ndarray:
    """Element-wise multiply tv by exp(eta) where eta is active."""
    out = np.empty_like(tv)
    j = 0
    for i in range(tv.shape[0]):
        if eta_active[i] == 1:
            out[i] = tv[i] * math.exp(eta[j])
            j += 1
        else:
            out[i] = tv[i]
    return out


@_maybe_njit(fastmath=True, cache=True)
def individual_loglike(eta: np.ndarray, tv: np.ndarray,
                        eta_active: np.ndarray, omega_diag: np.ndarray,
                        sigma_prop: float, sigma_add: float,
                        time: np.ndarray, obs: np.ndarray, dose: float,
                        tinf: float, model_code: int) -> float:
    """Joint log p(y, η | θ) at given eta.

    `tv` is the ordered typical-value array.  Indexing:
      - 1cmt_iv:   tv = [CL, V]
      - 1cmt_oral: tv = [CL, V, Ka, ALAG]   (ALAG element required, even if 0)
      - 2cmt_iv:   tv = [CL, V1, Q, V2]
      - 2cmt_inf:  tv = [CL, V1, Q, V2]
    """
    # Apply ETAs to the structural (non-ALAG) parameters
    p_indiv = _apply_eta_4param(tv, eta, eta_active)

    if model_code == 0:    # 1cmt_iv
        f = predict_1cmt_iv(time, dose, p_indiv[0], p_indiv[1])
    elif model_code == 1:  # 1cmt_oral
        ALAG = tv[3] if tv.shape[0] >= 4 else 0.0   # ALAG carries no ETA
        f = predict_1cmt_oral(time, dose, p_indiv[0], p_indiv[1],
                              p_indiv[2], ALAG)
    elif model_code == 2:  # 2cmt_iv
        f = predict_2cmt_iv(time, dose, p_indiv[0], p_indiv[1],
                            p_indiv[2], p_indiv[3])
    else:                  # 2cmt_inf
        f = predict_2cmt_inf(time, dose, tinf, p_indiv[0], p_indiv[1],
                             p_indiv[2], p_indiv[3])

    # Residual log-likelihood (Gaussian, combined error)
    ll_obs = 0.0
    for j in range(obs.shape[0]):
        fj = f[j] if f[j] > 1e-10 else 1e-10
        sig = sigma_prop * fj + 0.0  # will combine with add below
        var = (sigma_prop * fj) ** 2 + sigma_add ** 2
        if var < 1e-12:
            var = 1e-12
        ll_obs += -0.5 * (math.log(2 * math.pi * var)
                          + (obs[j] - fj) ** 2 / var)

    # ETA prior (only over active ETAs)
    ll_eta = 0.0
    for j in range(omega_diag.shape[0]):
        om = omega_diag[j] if omega_diag[j] > 1e-8 else 1e-8
        ll_eta += -0.5 * (math.log(2 * math.pi * om)
                          + eta[j] ** 2 / om)

    return ll_obs + ll_eta


MODEL_CODE = {
    "1cmt_iv": 0,
    "1cmt_oral": 1,
    "2cmt_iv": 2,
    "2cmt_inf": 3,
}
