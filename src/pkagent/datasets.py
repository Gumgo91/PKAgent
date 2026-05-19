"""Public PKGPT benchmark datasets — verbatim NONMEM-style CSVs.

Downloaded from https://github.com/Gumgo91/PKGPT/tree/main/dataset into
`data/pkgpt_real/`.  These are the actual clinical datasets fit by the
Human-Expert and PKGPT-Agent NONMEM runs reported in PKGPT paper
Tables 1-3, so any OFV computed here is directly comparable.

Available drugs (use `get_dataset(name)`):
  - "theophylline" — single oral dose, ~4 mg/kg
  - "warfarin"     — single oral dose 100 mg, with WT + AGE covariates
  - "tobramycin"   — multi-dose IV bolus q8h TDM, with WT/AGE/CLCR

Tobramycin's repeated dosing is encoded as a per-subject
`dose_history: list[(time, amt)]` attribute on the returned Subjects.
The engine's `2cmt_iv` predict_fn detects this attribute and dispatches
to a JIT-compiled superposition kernel.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .nlme import Subject

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "pkgpt_real"


# --------------------------------------------------------------------------- #
# Theophylline — classic NONMEM example data set, single oral dose
# --------------------------------------------------------------------------- #

def load_theophylline_real() -> list[Subject]:
    """Load the real Theophylline dataset (Boeckmann et al., 1992).

    AMT is recorded in mg/kg; we multiply by WT for the absolute dose so
    that downstream models use absolute parameterisation (CL in L/h,
    V in L) consistent with how the synthetic generator works.
    """
    df = pd.read_csv(_DATA_DIR / "theo.csv")
    subs = []
    for sid, sub in df.groupby("ID"):
        dose_rows = sub[sub["AMT"].notna() & (sub["AMT"] > 0)]
        if dose_rows.empty:
            continue
        dose_mg_per_kg = float(dose_rows["AMT"].iloc[0])
        wt = float(sub["WT"].iloc[0])
        dose_mg = dose_mg_per_kg * wt
        obs = sub[sub["DV"].notna()]
        if obs.empty:
            continue
        times = obs["TIME"].to_numpy(dtype=float)
        dv = obs["DV"].to_numpy(dtype=float)
        subs.append(Subject(
            sid=int(sid), time=times, obs=dv,
            dose=dose_mg, tinf=0.0,
            covariates={"WT": wt}))
    return subs


# --------------------------------------------------------------------------- #
# Warfarin — single oral dose ~100 mg, n=32, WT + AGE covariates
# --------------------------------------------------------------------------- #

def load_warfarin_real() -> list[Subject]:
    df = pd.read_csv(_DATA_DIR / "wafarin.csv" if (_DATA_DIR/"wafarin.csv").exists()
                       else _DATA_DIR / "warfarin.csv")
    subs = []
    for sid, sub in df.groupby("ID"):
        dose_rows = sub[(sub["EVID"] == 1) & (sub["AMT"].notna()) & (sub["AMT"] > 0)]
        if dose_rows.empty:
            continue
        dose_mg = float(dose_rows["AMT"].iloc[0])
        # observations: EVID=0 & MDV=0 & DV present
        obs = sub[(sub["EVID"] == 0) & (sub["MDV"] == 0) & sub["DV"].notna()]
        if obs.empty:
            continue
        # drop the t=0 zero-conc row if present (NONMEM-style placeholder)
        obs = obs[obs["TIME"] > 0]
        if obs.empty:
            continue
        times = obs["TIME"].to_numpy(dtype=float)
        dv = obs["DV"].to_numpy(dtype=float)
        wt = float(sub["WT"].iloc[0])
        age = float(sub["AGE"].iloc[0])
        subs.append(Subject(
            sid=int(sid), time=times, obs=dv,
            dose=dose_mg, tinf=0.0,
            covariates={"WT": wt, "AGE": age}))
    return subs


# --------------------------------------------------------------------------- #
# Tobramycin — multi-dose IV bolus q8h with peak/trough TDM sampling.
# `dose_history` on each Subject is the list of (time, amount) pairs;
# the JIT-compiled superposition kernel in engine.py handles the rest.
# --------------------------------------------------------------------------- #

def load_tobramycin_real() -> list[Subject]:
    """Real tobramycin TDM dataset (multi-dose IV bolus q8h)."""
    df = pd.read_csv(_DATA_DIR / "tobramycin.csv", comment="#",
                     names=["ID", "TIME", "AMT", "RATE", "DV", "MDV", "EVID",
                            "CMT", "WT", "AGE", "SEX", "CLCR"], skiprows=1)
    subs = []
    for sid, sub in df.groupby("ID"):
        dose_rows = sub[(sub["EVID"] == 1) & (sub["AMT"] > 0)]
        obs_rows = sub[(sub["EVID"] == 0) & (sub["MDV"] == 0) & (sub["DV"] > 0)]
        if dose_rows.empty or obs_rows.empty:
            continue
        dose_history = list(zip(dose_rows["TIME"].astype(float).tolist(),
                                  dose_rows["AMT"].astype(float).tolist()))
        times = obs_rows["TIME"].to_numpy(dtype=float)
        dv = obs_rows["DV"].to_numpy(dtype=float)
        # representative dose = first AMT (for engine's nominal-dose field;
        # actual prediction uses the full history via dose_history)
        first_dose = float(dose_rows["AMT"].iloc[0])
        wt = float(sub["WT"].iloc[0])
        age = float(sub["AGE"].iloc[0])
        clcr = float(sub["CLCR"].iloc[0])
        s = Subject(
            sid=int(sid), time=times, obs=dv,
            dose=first_dose, tinf=0.0,
            covariates={"WT": wt, "AGE": age, "CLCR": clcr})
        s.dose_history = dose_history   # type: ignore[attr-defined]
        subs.append(s)
    return subs


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

_LOADERS = {
    "theophylline": load_theophylline_real,
    "warfarin":     load_warfarin_real,
    "tobramycin":   load_tobramycin_real,
}


def get_dataset(name: str) -> list[Subject]:
    """Load a PKGPT benchmark dataset by name → list of `Subject`."""
    if name not in _LOADERS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(_LOADERS)}")
    return _LOADERS[name]()


def available_datasets() -> list[str]:
    return sorted(_LOADERS)
