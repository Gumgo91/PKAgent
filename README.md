# PKAgent

A Python-native agentic workflow for population pharmacokinetic (PopPK)
modeling with structured natural-language priors.

PKAgent combines a FOCE-I-inspired Laplace approximation (full-Hessian
inner step, log-parameter regularization, Numba-JIT multi-dose
prediction kernel) with an LLM-based agentic control loop. A free-form
analyst sentence is compiled into a typed constraint object with hard
(forbidden actions, fixed parameters, route overrides) and soft
(compartment preferences, ordinal absorption-speed priors) layers, and
that object is consulted by every model-building agent **and** by the
deterministic likelihood-ratio and AIC selection rules.

This repository contains the source code and the three public datasets
needed to reproduce the six-run evaluation described in the manuscript.

## Repository layout

```
pkagent/
├── src/pkagent/             # the package
├── data/pkgpt_real/         # public datasets (warfarin, theophylline, tobramycin)
├── examples/run_benchmark.py  # six-run evaluation from the manuscript
├── pyproject.toml
├── .env.example             # template for LLM API key
└── README.md
```

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/Gumgo91/PKAgent.git
cd PKAgent
pip install -e .
```

This installs the core dependencies (numpy, scipy, pandas, numba,
matplotlib, requests). If you want to use the Google Gemini direct API
or Anthropic Claude instead of the default OpenRouter routing, install
the corresponding extra:

```bash
pip install -e ".[gemini]"      # adds google-generativeai
pip install -e ".[anthropic]"   # adds anthropic
pip install -e ".[all]"         # both
```

## Configure an LLM backend

Copy the template and add one key. PKAgent picks a backend in this
priority order: OpenRouter > Anthropic > Gemini direct > deterministic
rule-based fallback (used automatically if no key is set).

```bash
cp .env.example .env
# then edit .env and fill in one key
```

## Run the six-run evaluation

The benchmark script runs three drugs (warfarin, theophylline,
tobramycin) under with-hint and without-hint scenarios — exactly the
six runs reported in the manuscript. The hint sentences match those
quoted in Methods section 2.7 verbatim.

```bash
python examples/run_benchmark.py
```

Useful options:

```bash
python examples/run_benchmark.py --drug theophylline    # one drug only
python examples/run_benchmark.py --without-hint-only    # skip with-hint
python examples/run_benchmark.py --with-hint-only       # skip without-hint
```

Per-run wall-clock on commodity hardware:

| Drug         | Approximate run time |
| ------------ | -------------------- |
| Theophylline | 4–5 min              |
| Warfarin     | 14–15 min            |
| Tobramycin   | 2.5–3 hr             |

Tobramycin time is dominated by the two-compartment forward
covariate search; reducing the covariate-search depth shortens it.

## Hint sentences used in the manuscript

| Drug | Scenario | Sentence |
| ---- | -------- | -------- |
| Theophylline | with-hint    | "Theophylline is an oral drug with rapid absorption and no lag." |
| Warfarin     | with-hint    | "Warfarin is an oral drug with a measurable absorption lag." |
| Tobramycin   | with-hint    | "Tobramycin is given by IV infusion and distributes into a peripheral compartment with renal elimination." |
| all          | without-hint | (no sentence — the agent runs from the dataset and NCA anchors alone) |

These exact strings are wired into `examples/run_benchmark.py` so the
benchmark output is reproducible against the manuscript without any
manual configuration.

## Data

The three public datasets live in `data/pkgpt_real/` in NONMEM-style
tabular form (columns: ID, TIME, DV, AMT, EVID, MDV, RATE, CMT, and
demographic covariates where applicable). They are the same Monolix
Suite reference datasets used by the PKGPT benchmark.

| File | Subjects | Route | Sampling |
| ---- | -------- | ----- | -------- |
| `theo.csv`       | 12  | oral          | sparse (≤11 samples/subject, 132 total observations) |
| `warfarin.csv`   | 32  | oral          | rich (≥8 samples/subject)                            |
| `tobramycin.csv` | 97  | IV infusion   | sparse peak-and-trough (2–3 samples/subject)         |

## Citation

If you use PKAgent in academic work, please cite the manuscript (citation
will appear here on publication).

## License

MIT. See `LICENSE`.
