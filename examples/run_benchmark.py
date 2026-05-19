"""Run the six-run PKAgent evaluation reported in the manuscript.

Three drugs (warfarin, theophylline, tobramycin), each analyzed under
with-hint and without-hint scenarios. The hint sentences are exactly
those quoted in Methods section 2.7 of the manuscript.

Usage
-----
1. Install:                    pip install -e .
2. Set your LLM key in .env    (copy .env.example and fill one of the
                                  two keys).
3. Run:                        python examples/run_benchmark.py
                               python examples/run_benchmark.py --drug theophylline
                               python examples/run_benchmark.py --without-hint-only

Output goes to stdout. Final converged FOCE-I OFV and theta estimates
are printed per run; no files are written.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Load .env into os.environ if present, without overriding existing keys.
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# Make `import pkagent` work whether the package is installed or
# being run from a source checkout.
_repo_root = Path(__file__).resolve().parent.parent
_src_dir = _repo_root / "src"
if _src_dir.exists():
    sys.path.insert(0, str(_src_dir))

from pkagent import datasets as dsets
from pkagent import fit as run_pkagent
from pkagent.fit import print_report


# Per-drug fit metadata — route, structural-model seed, covariate
# search candidates, and the natural-language hint exactly as it
# appears in Methods section 2.7 of the manuscript.
DRUGS = {
    "theophylline": {
        "route": "oral",
        "structural_hint": "1cmt_oral",
        "covariate_hints": [],
        "paper_hint": (
            "Theophylline is an oral drug with rapid absorption and no lag."
        ),
    },
    "warfarin": {
        "route": "oral",
        "structural_hint": "1cmt_oral",
        "covariate_hints": [],
        "paper_hint": (
            "Warfarin is an oral drug with a measurable absorption lag."
        ),
    },
    "tobramycin": {
        "route": "iv",
        "structural_hint": "2cmt_iv",
        "covariate_hints": [],
        "paper_hint": (
            "Tobramycin is given by IV infusion and distributes into a "
            "peripheral compartment with renal elimination."
        ),
    },
}


def run_one(drug: str, mode: str, hint: str | None) -> None:
    """Run PKAgent once on `drug` under the given hint mode and print a report."""
    meta = DRUGS[drug]
    bar = "-" * 72
    print(f"\n{bar}\n[{drug} / {mode}] starting\n{bar}", flush=True)
    if hint:
        print(f"  hint: {hint}", flush=True)
    t0 = time.time()
    out = run_pkagent(
        dsets.get_dataset(drug),
        route=meta["route"],
        covariate_hints=meta["covariate_hints"],
        structural_hint=meta["structural_hint"],
        hint=hint,
        label=f"{drug}_{mode}",
    )
    elapsed = time.time() - t0
    ofv = getattr(out.final_fit, "ofv_foce", out.final_fit.ofv)
    theta_rounded = {k: round(v, 4) for k, v in out.final_fit.theta.items()}
    print(
        f"  done in {elapsed:.1f}s OFV={ofv:.2f} theta={theta_rounded}",
        flush=True,
    )
    print_report(out, label=f"{drug} ({mode})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the six-run PKAgent evaluation from the manuscript."
    )
    parser.add_argument(
        "--drug",
        choices=list(DRUGS) + ["all"],
        default="all",
        help="Drug to evaluate (default: all three).",
    )
    parser.add_argument(
        "--without-hint-only",
        action="store_true",
        help="Skip the with-hint scenario.",
    )
    parser.add_argument(
        "--with-hint-only",
        action="store_true",
        help="Skip the without-hint scenario.",
    )
    args = parser.parse_args()

    if not (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    ):
        print(
            "WARNING: no LLM API key found in the environment "
            "(set OPENROUTER_API_KEY or GEMINI_API_KEY in .env). "
            "PKAgent will fall back to its rule-based deterministic mode.",
            flush=True,
        )

    drugs = list(DRUGS) if args.drug == "all" else [args.drug]
    scenarios: list[tuple[str, bool]] = []
    if not args.with_hint_only:
        scenarios.append(("without_hint", False))
    if not args.without_hint_only:
        scenarios.append(("with_hint", True))

    for drug in drugs:
        for mode, send_hint in scenarios:
            hint = DRUGS[drug]["paper_hint"] if send_hint else None
            run_one(drug, mode, hint)


if __name__ == "__main__":
    main()
