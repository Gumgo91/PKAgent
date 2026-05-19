"""PKAgent: AI-driven population PK modeling, NONMEM-free.

PKAgent is a PKPy-style PopPK library with automated decision-making at
the structural-model / initial-value / covariate-selection / evaluation
points.  Single-call API:

    from pkagent import fit
    output = fit(subjects, drug="warfarin")
    print(output.final_fit.theta)
"""
__version__ = "0.2.0"

from .fit import fit, FitOutput
from .nlme import Subject, FitResult
from . import decisions, datasets

__all__ = ["fit", "FitOutput", "Subject", "FitResult", "decisions", "datasets"]
