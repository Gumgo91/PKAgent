# PKAgent

PKAgent is a large language model (LLM) agent that develops population pharmacokinetic (PopPK) and PK/PD models. The
language model works only through tool calls: it summarizes and plots the data, declares models as JSON
specifications, has them fitted, reads the diagnostics, decides what to test next, and ends the session with a final
model and a written report. No number comes from the language model: [PKPy2](https://github.com/Gumgo91/PKPy2), an
open-source Python engine for nonlinear mixed-effects models, fits every model and computes the estimates, standard
errors, residual diagnostics and simulations, and PKAgent's own Python code adds the non-compartmental analysis (NCA),
the covariate screening statistics and the likelihood-ratio tests. No NONMEM installation is needed. An analyst can
add expert knowledge in plain language (for example "clearance and volume of distribution are proportional to body
weight"); the agent encodes it in the model, tests it against the data and reports the evidence. Every LLM response and
every tool call is logged.

![Graphical abstract. Three public datasets (neonatal phenobarbital, adult remifentanil, and a simulated oral drug) and three conditions (no knowledge, expert statement, misleading statement) go into PKAgent, in which a language model (GPT-6.1 Sol or Claude Opus 5.5) works only through tools of the open-source PKPy2 estimation engine. Strong evidence was recovered: the reference structure in all 36 runs, saturable elimination inferred from the data, and strongly supported covariate effects kept in all but one run. Weak evidence (the Apgar score effect on phenobarbital volume and the age effect on remifentanil central volume) was left out without knowledge, like stepwise covariate selection, and included with the expert statement. With a misleading statement, claims the data contradicted were rejected but an unsupported second compartment was adopted. Bottom line: where data are strong, PKAgent recovers the model and rejects statements the data contradict; where they are weak, the model follows what the analyst states, including unsupported claims.](docs/images/graphical_abstract.png)

## How it works

The LLM receives a task message with the data description, the optional expert knowledge and analysis objective, and
the budget of the session. It never runs code. It calls 14 tools in five groups (data, models, diagnostics, covariates
and uncertainty, report; definitions in [src/pkagent/tools.py](src/pkagent/tools.py)), and the session has PKPy2 fit
every model.

![Block diagram of PKAgent. A task (study description, optional expert statement, and a budget per run of 40 fits, 80 responses and 6 hours) goes to the language model (GPT-6.1 Sol or Claude Opus 5.5), and a data file goes into the PKAgent session. The language model writes no code; its system prompt sets the OFV decrease needed to add a parameter (at least 3.84) and to keep an effect (at least 6.63). It sends tool calls to the session and receives results and plots. The session offers 14 tools in five groups (data 4, models 4, diagnostics 3, covariates and uncertainty 2, report 1) and keeps versioned data, a model registry, the budgets with a 25 US dollar fee limit, and a log. The PKPy2 engine fits every model (Laplace FOCE-I-type estimation with L-BFGS-B; standard errors, CWRES and NPDE; VPC simulation). Outputs: the final model and report, an audit trail, and the tokens, fees and time used.](docs/images/architecture.png)

| Group | Tools | What they do |
|---|---|---|
| Data | `describe_data`, `plot_data`, `run_nca`, `add_data_column` | summary of subjects, doses, sampling, outputs and covariates; concentration-time plots; non-compartmental analysis with starting values for a first model; derived columns (indicators, unit conversions) from a recorded expression |
| Models | `fit_models`, `list_models`, `get_model`, `compare_models` | fit up to six specifications in parallel; the model registry; the full summary and specification of one model; OFV, AIC, BIC and likelihood-ratio tests |
| Diagnostics | `view_plots`, `screen_covariates`, `run_vpc` | goodness-of-fit and individual plots, sent to the LLM as images; screening of empirical Bayes estimates against covariates; visual predictive checks (500 simulations, optionally prediction-corrected) |
| Covariates and uncertainty | `covariate_search`, `resample_uncertainty` | stepwise covariate modeling (forward inclusion at p < 0.05, backward elimination at p < 0.01 by default) with every trial fit registered; nonparametric bootstrap of the final model within a time limit |
| Report | `finalize_model` | the final model and the written report; ends the session |

**Model specifications.** A specification is JSON ([src/pkagent/spec.py](src/pkagent/spec.py)) and every structure
comes from the PKPy2 library: one to three compartments; intravenous bolus or infusion, first-order, zero-order or
transit absorption, lag time and bioavailability; linear or Michaelis-Menten elimination; target-mediated drug
disposition (full or quasi-steady-state); parent-metabolite models; direct and effect-compartment PD (Emax, sigmoid
Emax, Imax, sigmoid Imax, linear) and indirect-response models I to IV. The statistical model has interindividual
variability with correlated blocks, interoccasion variability, power, exponential, linear and categorical covariate
effects, additive, proportional, combined or log-normal residual error per output, and fixed values and bounds.

**Estimation and decisions.** Fits use the Laplace (FOCE-I-type) objective by default; `fit_models` can request
PKPy2's importance-sampled marginal likelihood with its two-bank convergence audit instead, which is slower. The system
prompt ([src/pkagent/prompts.py](src/pkagent/prompts.py)) tells the agent to add a parameter only when the OFV drops by
at least 3.84 (p < 0.05) and, in the final backward check, to keep an effect only when its removal raises the OFV by at
least 6.63 (p < 0.01). It asks the agent to encode the analyst's knowledge, compare it with the data-driven
alternative and report the OFV difference.

**Budget.** A session is limited to 40 fits, 80 LLM responses, 6 hours and 25 US dollars of LLM fees by default
(`Budget` in [src/pkagent/config.py](src/pkagent/config.py)). Every tool result reports the fits, responses and hours
left and the fees spent; when a limit is reached, the agent is asked to finalize the best converged model.

## Installation

PKAgent requires Python 3.13 (`requires-python = ">=3.13,<3.14"`) and PKPy2. PKPy2 is not on PyPI: `pyproject.toml`
declares it as `pkpy2 @ git+https://github.com/Gumgo91/PKPy2.git`, and pip installs it from its repository (git must
be installed).

```bash
git clone https://github.com/Gumgo91/PKAgent.git
cd PKAgent
python -m pip install -e .
```

The editable install (`-e`) runs PKAgent from the cloned folder, so it also reads a `.env` file in the repository root
(see [Configuration](#configuration)) and records the PKAgent git commit in `run.json`. The other dependencies (NumPy,
SciPy, pandas, Matplotlib and the `openai` client, which PKAgent uses for the OpenRouter API) are installed by pip;
pandas and openai are kept below version 3. Tested with Python 3.13.5, NumPy 2.2.6, SciPy 1.16.1,
pandas 2.3.1, Matplotlib 3.10.5 and openai 2.6.1.

Tests of the model specifications (no LLM calls, no fits): `python -m pip install -e ".[test]"`, then
`python -m pytest`. `python tests/smoke_session.py` runs every tool in a scripted session without an LLM on the
phenobarbital data (it fits models and needs the benchmark data, see [Benchmark](#benchmark)). The benchmark figures
(`benchmarks/figures.py`) also need Pillow (tested 11.1.0), which writes their CMYK TIFF copies:
`python -m pip install -e ".[figures]"`.

## Configuration

PKAgent calls the LLM through [OpenRouter](https://openrouter.ai) and needs an API key in `OPENROUTER_API_KEY`. Set it
in the environment, or copy [.env.example](.env.example) to `.env` and replace the placeholder. PKAgent reads `.env`
from the working directory, or else, with the editable install above, from the repository root; a variable already set
in the environment takes precedence. `.env` is git-ignored. Without a key, a run stops with an error.

## Usage

```bash
pkagent run data.csv --description description.txt \
    --knowledge "Clearance and volume of distribution are proportional to body weight." \
    --model gpt --out results/run1
```

| Option | Default | Meaning |
|---|---|---|
| `DATA` | (required) | analysis data, a NONMEM-format CSV file |
| `--description` | (required) | data description: study design, units and column meanings (text, or a path to a text file) |
| `--knowledge` | none | expert knowledge from the analyst (text or file) |
| `--objective` | none | analysis objective (text or file) |
| `--model` | `claude` | `gpt` (`openai/gpt-6.1-sol`), `claude` (`anthropic/claude-opus-5.5`) or any OpenRouter model id |
| `--out` | (required) | output folder |
| `--max-fits`, `--max-turns`, `--max-hours`, `--max-cost` | 40, 80, 6, 25 | budget: model fits, LLM responses, hours, LLM fees in US dollars |
| `--workers`, `--threads` | 2, 4 | parallel fitting processes, and Numba threads in each |
| `--reasoning` | `medium` | reasoning effort of the LLM: `low`, `medium`, `high` or `none` |
| `--seed` | 20261001 | random seed of the fits, simulations and bootstrap (LLM sampling is not seeded) |
| `--no-images` | off | do not send plots to the LLM as images |

The data use NONMEM column names. `ID`, `TIME` and `DV` are required; `AMT`, `EVID`, `MDV`, `CMT`, `RATE`, `II`,
`SS`, `ADDL`, `DVID`, `CENS` and `LIMIT` are read as event-record columns, and `.` marks a missing value. The other
numeric columns are candidate covariates, except common NONMEM names such as `OCC`, `BLQ`, `LLOQ` or `TAD`; the
occasion column for interoccasion variability is named in the model specification.

From Python, `run` takes the description and knowledge as text and returns the path of the report:

```python
from pathlib import Path

from pkagent import Budget, Settings, run

if __name__ == '__main__':          # fits run in worker processes
    report = run('data.csv', 'results/run1',
                 description=Path('description.txt').read_text(encoding='utf-8'),
                 knowledge='Clearance and volume of distribution are proportional to body weight.',
                 settings=Settings(model='openai/gpt-6.1-sol', budget=Budget(max_fits=20)))
    print(report)                   # results/run1/report.md
```

`Settings` ([src/pkagent/config.py](src/pkagent/config.py)) also sets the LLM temperature and output limit, the
number of fitting processes and threads, the time limit of one fit and of the bootstrap, and the number of Laplace
starts.

## Outputs of a session

| Path in the output folder | Content |
|---|---|
| `report.md` | final model table (estimates, RSE, 95% CI, IIV with shrinkage, covariate effects, residual error), the agent's report (summary, model development, use of the expert knowledge, evaluation, limitations), diagnostic figures, all models with OFV and AIC, and the data transformations |
| `results.json` | the same in machine-readable form, with LLM calls, tokens, fees, fits and hours |
| `run.json` | task inputs (data path, description, knowledge, objective), LLM, seed, reasoning effort, budget and start time, and the PKAgent and PKPy2 versions, with the PKAgent git commit when PKAgent runs from a source checkout (the editable install) |
| `transcript.jsonl` | every LLM response with its tool calls, the reasoning summary where the provider returns one, and token usage |
| `tool_log.jsonl` | every tool call with its arguments and result |
| `messages.json` | the full conversation (images replaced by placeholders) |
| `models/<model id>/` | for each model: specification (`spec.json`), PKPy2 fit (`fit.json`), `summary.json`, `diagnostics.csv`, fit progress and diagnostic plots |
| `final/` | a copy of the final model's folder |
| `data/data_v<k>.csv` | each version of the analysis data (a new version after every derived column) |
| `plots/` | plots of the data |

## Benchmark

[benchmarks/](benchmarks/README.md) tests whether the agent reproduces reference models on three public datasets:
phenobarbital in 59 preterm neonates (nlmixr2data `pheno_sd`), remifentanil in 65 adults (nlme `Remifentanil`) and a
simulated oral drug with Michaelis-Menten elimination in 40 subjects (nlmixr2data `Oral_1CPTMM`). GPT-6.1 Sol
(`openai/gpt-6.1-sol`) and Claude Opus 5.5 (`anthropic/claude-opus-5.5`) analyzed each dataset without knowledge and
with a short expert statement of the reference model, three times each (36 runs), and with deliberately misleading
statements on phenobarbital and the oral drug (8 runs).

Results:

- **Structure and cost.** The final model had the reference structure in 36 of 36 runs. Over these 36 runs, a run
  took a median of 1.2 hours and 0.28 US dollars of LLM fees.
- **No knowledge.** The agents kept the strongly supported covariate effects (those whose removal from the PKPy2 fit
  of the reference model raises the OFV by at least 6.63) in 11 of 12 runs on phenobarbital and remifentanil. The
  effect with the weakest support in each of these two reference models was included in none of the six runs: the
  Apgar score effect on phenobarbital volume, whose removal raises the OFV by 4.5, and the age effect on remifentanil
  central volume, whose removal raises it by 2.7.
- **Expert statement.** With the statement, which the system prompt tells the agent to encode and test, every
  reference covariate relationship was in the final model in all six runs on phenobarbital and in all six runs on
  remifentanil. Over both conditions, the final model reproduced the reference model in 19 of 36 runs.
- **Misleading statements.** On phenobarbital, the claim of no weight effects and the claim of an Apgar effect on
  clearance were each adopted in none of the four runs. On the oral drug, the claim of linear elimination was adopted
  in none of the four runs, and the claim of a second compartment, which the data did not support, in all four runs.

![Grid of the covariate relationships of the phenobarbital and remifentanil reference models (rows, each labeled with the OFV increase on removal from the reference fit) against 24 runs (columns: per drug, six runs without knowledge and six with the expert statement; G1 to G3 GPT-6.1 Sol, C1 to C3 Claude Opus 5.5). Dark cells mark a relationship present in the reference form, hatched cells one present in another form, and white cells one that is absent; a last row counts relationships not in the reference model. With the expert statement every cell is dark. Without knowledge, all phenobarbital runs include both weight effects and lack the Apgar effect (OFV 4.5); all remifentanil runs lack the age effect on V1 (OFV 2.7), one lacks the lean body mass effect on V2, and the other effects are present in another form.](docs/images/covariate_relationships.png)

*Covariate relationships of the reference models in the final models of the 24 phenobarbital and remifentanil runs.
ΔOFV: increase in OFV when the relationship is removed from the PKPy2 fit of the reference model. G1 to G3: GPT-6.1
Sol; C1 to C3: Claude Opus 5.5. Dark: present in the reference form; hatched: present in another form; white: absent.
Other relationships: number of covariate relationships in the final model that are not in the reference model.*

**Reproducing the benchmark.** The scripts and the order in which to run them are in
[benchmarks/README.md](benchmarks/README.md).

1. Data: the source datasets are not redistributed. [benchmarks/export_data.R](benchmarks/export_data.R) exports them
   from the R packages nlme 3.1-168 and nlmixr2data 2.0.10, and `benchmarks/prepare_data.py` writes the analysis files:
   `Rscript benchmarks/export_data.R`, then `python benchmarks/prepare_data.py`.
2. Runs ([benchmarks/run_benchmark.py](benchmarks/run_benchmark.py), needs `OPENROUTER_API_KEY`):
   ```bash
   python benchmarks/run_benchmark.py --all --models gpt claude --reps 3 --jobs 2 --workers 3 --threads 2
   python benchmarks/run_benchmark.py --all --datasets pheno oral_mm --conditions misleading --models gpt claude \
       --reps 2 --jobs 2 --workers 3 --threads 2
   ```
   New runs differ from the runs reported here, because LLM sampling is not seeded.
3. Reference fits, evaluation, figures and an archive of the outputs: `benchmarks/reference_fits.py`,
   `benchmarks/evaluate.py`, `benchmarks/figures.py` and `benchmarks/export_archive.py`, in the order given in
   [benchmarks/README.md](benchmarks/README.md).

## Paper

The benchmark is described in a manuscript in preparation. Until it is published, please cite the software (see
[Citation](#citation)).

## Repository layout

| Folder | Content |
|---|---|
| `src/pkagent/` | the agent: session and tools, model specifications, prompts, LLM client, report |
| `benchmarks/` | benchmark definition (`datasets.json`) and the scripts for the runs, reference fits, evaluation, Figures 2 to 4 and an archive of the outputs |
| `docs/` | the images of this README |
| `tests/` | tests of the model specifications and a scripted session without an LLM |

## Citation

Citation metadata are in [CITATION.cff](CITATION.cff) (GitHub shows them under "Cite this repository"). Please also
cite [PKPy2](https://github.com/Gumgo91/PKPy2).

## License

MIT. See [LICENSE](LICENSE).
