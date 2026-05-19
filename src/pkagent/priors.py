"""Drug-CLASS priors and physiological bounds (data-driven, not drug-specific).

PKAgent does NOT receive the drug name.  At fit time the user supplies the
route (oral / iv / iv_infusion) and optionally a drug-CLASS label that
identifies a small group of pharmacologically-similar drugs (e.g.
'oral_small_molecule' or 'iv_aminoglycoside').  The class catalog
encodes broad pharmacology-textbook priors and the well-known mandatory
covariates for the class — never the truth values of any particular
benchmark drug.

Design principles
-----------------
1. **No drug name in the catalog.**  Keys are class names that apply to
   many drugs, not 'warfarin' / 'theophylline' / 'tobramycin'.
2. **Wide priors.**  log-SD = 1.0 → 1.5 (95% interval ≈ ±200-500%) so the
   prior regularises but does not encode the answer.  The data-driven
   NCA init step is what carries the actual starting estimates.
3. **Physiological bounds remain loose.**  Wider than the truth value so
   they only catch egregious failures (e.g. the PKGPT V2=149 L for an
   aminoglycoside), not normal variation around literature.
4. **Mandatory covariates are class-level**, not drug-specific.  Renally
   cleared drugs → CL ~ CLCR.  Body-weight-scaled disposition → allometric
   WT.  These are textbook (Bauer 2008, Bonate 2011).

References (general PopPK literature, NOT tuned to the benchmark drugs):
  - Bonate 2011 *Pharmacokinetic-Pharmacodynamic Modeling and Simulation*
  - Bauer 2008 *Population PK methodology and applications*
  - Anderson & Holford 2008 *Mechanism-based concepts of size and maturity
    in pharmacokinetics* (allometric WT)
  - Gilbert 2018 *Sanford Guide* (aminoglycoside Vss ≈ 0.25 L/kg ECF)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DrugClass:
    """A pharmacologically-similar group of drugs sharing structural model,
    typical parameter ranges, and mandatory covariates.
    """
    name: str
    structural_hint: str
    typical: dict                     # geometric mean (broad, NOT tuned to truth)
    log_sd: dict                      # WIDE log-SD (≥ 1.0)
    bounds: dict                      # very loose plausibility bounds
    mandatory_covariates: list        # standard for the class
    expected_covariate_signs: dict    # textbook directions


# Class-level catalog.  Typical values are LITERATURE TEXTBOOK MEDIANS for
# the entire pharmacological class — explicitly chosen so they would not
# precisely match the truth of any individual drug.
CLASSES = {
    # Oral small-molecule with first-order absorption and lag.  Covers
    # most oral drugs (analgesics, antibiotics, anticoagulants etc.).
    # CL covers 0.01-1000 L/h, V covers 1-5000 L — three orders of magnitude.
    "oral_small_molecule_with_lag": DrugClass(
        name="oral_small_molecule_with_lag",
        structural_hint="1cmt_oral_with_lag",
        typical={"Ka": 1.0, "CL": 10.0, "V": 50.0, "ALAG": 0.5},
        log_sd={"Ka": 1.0, "CL": 2.0, "V": 1.8, "ALAG": 1.0},
        bounds={"Ka": (0.01, 30.0), "CL": (0.001, 5000.0),
                "V": (0.5, 50000.0), "ALAG": (0.0, 8.0)},
        mandatory_covariates=[],
        expected_covariate_signs={"CL~WT": "+", "V~WT": "+"},
    ),

    # Oral small-molecule WITHOUT absorption lag (rapid absorption).
    "oral_small_molecule": DrugClass(
        name="oral_small_molecule",
        structural_hint="1cmt_oral",
        typical={"Ka": 1.0, "CL": 10.0, "V": 50.0},
        log_sd={"Ka": 1.0, "CL": 2.0, "V": 1.8},
        bounds={"Ka": (0.01, 30.0), "CL": (0.001, 5000.0),
                "V": (0.5, 50000.0)},
        mandatory_covariates=[],
        expected_covariate_signs={"CL~WT": "+", "V~WT": "+"},
    ),

    # IV aminoglycoside (renally cleared, two-compartment, narrow Vss).
    # Vss ≈ 0.25 L/kg (Gilbert Sanford Guide) so V1+V2 ≈ 17-20 L for 70 kg.
    # CLCR is THE mandatory covariate (renal clearance).
    "iv_aminoglycoside": DrugClass(
        name="iv_aminoglycoside",
        structural_hint="2cmt_inf",
        typical={"CL": 5.0, "V1": 10.0, "Q": 5.0, "V2": 20.0},
        log_sd={"CL": 1.5, "V1": 1.5, "Q": 1.5, "V2": 1.5},
        bounds={"CL": (0.1, 50.0), "V1": (0.5, 60.0),
                "Q": (0.1, 50.0), "V2": (1.0, 80.0)},
        mandatory_covariates=["CLCR"],
        expected_covariate_signs={"CL~CLCR": "+", "V1~WT": "+"},
    ),

    # Generic IV bolus single-compartment drug — fallback for unknown IV drugs.
    "iv_small_molecule": DrugClass(
        name="iv_small_molecule",
        structural_hint="1cmt_iv",
        typical={"CL": 10.0, "V": 50.0},
        log_sd={"CL": 2.0, "V": 1.8},
        bounds={"CL": (0.001, 5000.0), "V": (0.5, 50000.0)},
        mandatory_covariates=[],
        expected_covariate_signs={"CL~WT": "+", "V~WT": "+"},
    ),
}


# Backward-compat alias for code paths that still call this CATALOG.  This
# name does NOT refer to drug names; it is just the class registry.
CATALOG = CLASSES


def get_class(drug_class: str | None, route: str | None = None) -> DrugClass:
    """Look up a class by name, with a route-based fallback.

    If `drug_class` is registered, return it.  Otherwise fall back to a
    sensible default by route:
      - "oral"          → oral_small_molecule_with_lag
      - "iv_infusion"   → iv_aminoglycoside (only 2-cmt IV in catalog)
      - "iv"            → iv_small_molecule
      - anything else   → oral_small_molecule
    """
    if drug_class and drug_class in CLASSES:
        return CLASSES[drug_class]
    if route:
        r = route.lower()
        if "infusion" in r:
            return CLASSES["iv_aminoglycoside"]
        if r.startswith("iv"):
            return CLASSES["iv_small_molecule"]
        if r.startswith("o"):
            return CLASSES["oral_small_molecule_with_lag"]
    return CLASSES["oral_small_molecule"]


def physio_plausibility(drug_class: str, theta: dict
                          ) -> tuple[bool, list[str]]:
    """Hard plausibility gate keyed by drug class (not drug name).

    Returns (is_plausible, list_of_violations).
    """
    if drug_class not in CLASSES:
        return True, []
    cat = CLASSES[drug_class]
    violations = []
    for p, (lo, hi) in cat.bounds.items():
        v = theta.get(p)
        if v is None:
            continue
        if v < lo or v > hi:
            violations.append(f"{p}={v:.3g} outside plausible [{lo},{hi}]")
    return (len(violations) == 0), violations
