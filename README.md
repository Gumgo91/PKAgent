# PKAgent

PKAgent is an LLM agent that develops population pharmacokinetic (PopPK) and PK/PD models. The language model works
like a pharmacometrician: it looks at the data, proposes models, fits them, reads the diagnostics and decides what to
test next, entirely through tool calls. All estimation, diagnostics and statistics are computed by
[PKPy2](https://github.com/Gumgo91/PKPy2), a validated Python engine for nonlinear mixed-effects models. Models are
fitted with the Laplace (FOCE-I-type) objective by default; the importance-sampled marginal likelihood with PKPy2's
two-bank convergence audit is available on request. No NONMEM installation is needed.

An analyst can add expert knowledge in plain language (for example "clearance and volume scale linearly with body
weight"). The agent encodes it in the model as structure, fixed values, bounds or covariate relationships, tests it
against the data, and reports the evidence.

## How it works

The agent (any tool-calling model on [OpenRouter](https://openrouter.ai), for example `openai/gpt-6.1-sol` or
`anthropic/claude-opus-5.5`) calls these tools:

| Tool | What it does |
|---|---|
| `describe_data`, `plot_data`, `run_nca` | data summary, concentration-time plots, non-compartmental analysis and starting values |
| `add_data_column` | derived columns (indicators, unit conversions) with a recorded expression |
| `fit_models` | fit up to six JSON model specifications in parallel with PKPy2 |
| `list_models`, `get_model`, `compare_models` | model registry, likelihood-ratio tests, AIC/BIC |
| `view_plots`, `run_vpc`, `screen_covariates` | goodness-of-fit and individual plots (as images), visual predictive checks, eta-covariate screening |
| `covariate_search` | stepwise covariate modeling with the fits of each step run in parallel |
| `resample_uncertainty` | parallel bootstrap of the final model within a time limit |
| `finalize_model` | final model and written report |

Model specifications are JSON (structure from the PKPy2 library: 1-3 compartments, first-order, zero-order or transit
absorption, lag, bioavailability, Michaelis-Menten elimination, TMDD, parent-metabolite, direct, effect-compartment
and indirect-response PD; IIV with correlated blocks, interoccasion variability, power/exponential/linear/categorical
covariates, residual models per output, fixed values and bounds). The agent never runs code.

Every session writes a folder with `report.md` (final model table, the agent's report, all models, figures),
`results.json`, `transcript.jsonl` (every LLM turn), `tool_log.jsonl` (every tool call with its result) and one folder
per fitted model (specification, PKPy2 fit, diagnostics, plots).

## Installation

Python 3.13.

```bash
git clone https://github.com/Gumgo91/PKAgent.git
cd PKAgent
python -m pip install .
```

PKAgent needs an OpenRouter API key in the environment or in a `.env` file: `OPENROUTER_API_KEY=...`

## Usage

```bash
pkagent run data.csv --description description.txt --knowledge "Clearance scales with creatinine clearance." \
    --model claude --out results/run1
```

`--model` accepts `gpt`, `claude` or any OpenRouter model id. Budgets: `--max-fits`, `--max-turns`, `--max-hours`,
`--max-cost` (USD). Parallel fitting: `--workers` processes with `--threads` Numba threads each.

```python
from pkagent import run, Settings
run('data.csv', 'results/run1', description='...', knowledge='...', settings=Settings(model='openai/gpt-6.1-sol'))
```

## Benchmarks

`benchmarks/run_benchmark.py` runs public datasets with published reference models, without and with a one-sentence
expert statement, for each LLM. See `benchmarks/README.md`.

## Paper

How each table, figure and supplementary file of the CPT paper is built from the benchmark results: `paper/README.md`.

## License

MIT. See `LICENSE`.
