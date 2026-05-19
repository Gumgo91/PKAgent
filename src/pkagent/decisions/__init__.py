"""Decision agents — PKAgent's internal automation.

The decision agents drive PKAgent's end-to-end automation given a
dataset:

  - `structural.decide(...)`     → which compartment/absorption model
  - `initial.compute(...)`       → data-driven initial θ̂ and ω̂
  - `covariate.propose(...)`     → candidate covariates to test
  - `evaluator.evaluate(...)`    → per-round Statistician+Clinician verdict
  - `arbitrator.arbitrate(...)`  → cross-round "Final Judge" picks the
                                   publishable round from the iteration log

The structural/initial/covariate agents have deterministic rule-based
paths; the evaluator's Clinician and the arbitrator are LLM-only (no
deterministic fallback) because their judgments encode pharmacological
reasoning that must not be smuggled in as hardcoded rules.
"""
