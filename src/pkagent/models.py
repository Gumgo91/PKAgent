"""Analytical PK structural models.

All concentrations are at unit dose; multiply by Dose for actual dose.
Time t is hours, concentrations are mg/L (or whatever consistent with V units).
Numerically stable: uses log-sum-exp / l'Hopital around singularities.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


def one_comp_iv(t: np.ndarray, dose: float, CL: float, V: float) -> np.ndarray:
    k = CL / V
    return (dose / V) * np.exp(-k * t)


def one_comp_oral(t: np.ndarray, dose: float, CL: float, V: float, Ka: float, F: float = 1.0,
                  ALAG: float = 0.0) -> np.ndarray:
    k = CL / V
    tt = np.maximum(t - ALAG, 0.0)
    if abs(Ka - k) < 1e-8:
        # l'Hopital limit when Ka ~ k
        return (F * dose * Ka / V) * tt * np.exp(-k * tt)
    return (F * dose * Ka / (V * (Ka - k))) * (np.exp(-k * tt) - np.exp(-Ka * tt))


def two_comp_iv(t: np.ndarray, dose: float, CL: float, V1: float, Q: float, V2: float) -> np.ndarray:
    k10 = CL / V1
    k12 = Q / V1
    k21 = Q / V2
    s = k10 + k12 + k21
    disc = s * s - 4.0 * k10 * k21
    disc = max(disc, 1e-12)
    sq = np.sqrt(disc)
    alpha = 0.5 * (s + sq)
    beta = 0.5 * (s - sq)
    A = dose / V1 * (alpha - k21) / (alpha - beta)
    B = dose / V1 * (k21 - beta) / (alpha - beta)
    return A * np.exp(-alpha * t) + B * np.exp(-beta * t)


def two_comp_iv_infusion(t: np.ndarray, dose: float, tinf: float,
                         CL: float, V1: float, Q: float, V2: float) -> np.ndarray:
    """IV infusion of duration tinf for a 2-compartment model.

    Uses superposition: during infusion, c(t) = R0 * f(t); after, c(t) = R0 * (f(t) - f(t-tinf)).
    Here f(t) integrates the unit-rate IV response over [0,t].
    """
    if tinf <= 0:
        return two_comp_iv(t, dose, CL, V1, Q, V2)
    R0 = dose / tinf
    k10 = CL / V1
    k12 = Q / V1
    k21 = Q / V2
    s = k10 + k12 + k21
    disc = max(s * s - 4.0 * k10 * k21, 1e-12)
    sq = np.sqrt(disc)
    alpha = 0.5 * (s + sq)
    beta = 0.5 * (s - sq)
    # Coefficients per unit dose IV bolus
    A = (1.0 / V1) * (alpha - k21) / (alpha - beta)
    B = (1.0 / V1) * (k21 - beta) / (alpha - beta)

    def F(tau):
        tau = np.asarray(tau)
        out = np.where(tau > 0,
                       A * (1.0 - np.exp(-alpha * tau)) / alpha + B * (1.0 - np.exp(-beta * tau)) / beta,
                       0.0)
        return out

    t = np.asarray(t, dtype=float)
    during = (t > 0) & (t <= tinf)
    after = t > tinf
    c = np.zeros_like(t, dtype=float)
    c[during] = R0 * F(t[during])
    c[after] = R0 * (F(t[after]) - F(t[after] - tinf))
    return c


@dataclass
class StructuralModel:
    name: str            # "1cmt_iv", "1cmt_oral", "2cmt_iv", "2cmt_oral", "2cmt_inf"
    params: list         # ordered list of parameter names

    def predict(self, theta: dict, t, dose, **kw):
        if self.name == "1cmt_iv":
            return one_comp_iv(np.asarray(t), dose, theta["CL"], theta["V"])
        if self.name == "1cmt_oral":
            return one_comp_oral(np.asarray(t), dose, theta["CL"], theta["V"], theta["Ka"],
                                 ALAG=theta.get("ALAG", 0.0))
        if self.name == "2cmt_iv":
            return two_comp_iv(np.asarray(t), dose, theta["CL"], theta["V1"], theta["Q"], theta["V2"])
        if self.name == "2cmt_inf":
            tinf = kw.get("tinf", 0.5)
            return two_comp_iv_infusion(np.asarray(t), dose, tinf,
                                        theta["CL"], theta["V1"], theta["Q"], theta["V2"])
        raise ValueError(f"unknown model {self.name}")


REGISTRY = {
    "1cmt_iv":   StructuralModel("1cmt_iv",   ["CL", "V"]),
    "1cmt_oral": StructuralModel("1cmt_oral", ["CL", "V", "Ka"]),
    "2cmt_iv":   StructuralModel("2cmt_iv",   ["CL", "V1", "Q", "V2"]),
    "2cmt_inf":  StructuralModel("2cmt_inf",  ["CL", "V1", "Q", "V2"]),
}
