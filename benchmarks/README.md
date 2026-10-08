# PKAgent benchmark

Can an LLM agent that drives an open-source estimation engine (PKPy2) reproduce published population
pharmacokinetic models, and what does a short expert statement change?

## Datasets and reference models

| Dataset | Data | Design | Reference model |
|---|---|---|---|
| `pheno` | nlmixr2data 2.0.10 `pheno_sd` (phenobarbital, Grasela and Donn 1985) | 59 preterm neonates, IV bolus loading and maintenance doses, 155 sparse concentrations | NONMEM example model with its NONMEM 7.4.2 FOCE-I estimates: 1-cmt; CL and V proportional to birth weight; V 15.9% larger when the 5-minute Apgar score is below 5; exponential IIV on CL and V; proportional error |
| `remifentanil` | nlme 3.1-168 `Remifentanil` (Minto et al. 1997) | 65 adults aged 20-85 years, one 4-20 min infusion, 1,992 arterial concentrations | Minto et al. 1997: 3-cmt; V1, V2 and CL linear in age (centered at 40 years) and lean body mass (centered at 55 kg), Q2 and Q3 linear in age, V3 constant; no sex effect. The additive terms are fitted as a product of linear terms; the variability model (exponential IIV on all six parameters, proportional error) is not from the publication |
| `oral_mm` | nlmixr2data 2.0.10 `Oral_1CPTMM` (simulated, ACOP 2016), subset | 40 subjects (the first 10 at each dose level), 10/20/40/80 mg as a single dose and then 7 daily oral doses, 25 of the 58 samples per subject, 1,000 concentrations | simulation model: 1-cmt, first-order absorption, Michaelis-Menten elimination, 30% IIV on Ka, V, VMAX and KM, 20% exponential (log-normal) residual error |

The source documentation of `Oral_1CPTMM` gives 20% residual error without its type. `residual_check.py` showed,
after all runs had finished, that the error is exponential, so the reference model uses log-normal error; the
benchmark definition frozen before the runs (commit f5a4263) used proportional error, and that reference fit is kept
in `reference_fits/oral_mm_proportional` (Supplementary Material S2).

`datasets.json` holds, for each dataset, the description given to the agent (study design and column meanings; it
names the drug where there is one but not the source publication), the expert statement (one sentence; two for
remifentanil), for `pheno` and `oral_mm` the misleading statement and the claims it makes, and the reference model with
its estimates. `export_data.R` exports the source datasets from the R packages to `data/` (they are not
redistributed), and `prepare_data.py` builds the NONMEM-format files `data/pheno.csv`, `data/remifentanil.csv` and
`data/oral_mm.csv` from them. `excluded_nimotuzumab.json` records why the nimotuzumab TMDD dataset was dropped (no
verifiable reference model).

## Conditions

- `none` (no knowledge): the agent receives the data file and the description.
- `knowledge` (expert statement): the same, plus the expert statement of `datasets.json` in the section "Expert
  knowledge from the analyst" of the task message. The system prompt instructs the agent to encode such knowledge,
  compare it with the data-driven alternative and report the OFV difference, and not to discard it only because an
  alternative fits slightly better.
- `misleading` (misleading statement; `pheno` and `oral_mm` only): the same as `knowledge`, with the deliberately
  wrong statement of `datasets.json` in place of the expert statement.

The runs of the paper use `openai/gpt-6.1-sol` and `anthropic/claude-opus-5.5` through OpenRouter: three replicates
per dataset, condition (`none`, `knowledge`) and LLM (36 runs), and two replicates per LLM of the `misleading`
condition on `pheno` and `oral_mm` (8 runs). Replicates differ in LLM sampling (provider defaults, not seeded) and in
the seed that PKAgent passes to the engine (20261001 + replicate). Limits per run (`pkagent.config.Budget`): 40 fits
(bootstrap refits excluded), 80 LLM responses and 6 hours, which the task message states, and 25 USD of LLM fees,
which the session enforces; every tool result reports the remaining fits, responses and hours and the fees spent.

```bash
python benchmarks/run_benchmark.py --dataset pheno --condition knowledge --model claude --rep 1
python benchmarks/run_benchmark.py --all --models gpt claude --reps 3 --jobs 2 --workers 3 --threads 2
python benchmarks/run_benchmark.py --all --datasets pheno oral_mm --conditions misleading --models gpt claude \
    --reps 2 --jobs 2 --workers 3 --threads 2
```

Runs need `OPENROUTER_API_KEY` and write to `runs/<dataset>/<condition>/<model>/rep<k>/`; a run whose `results.json`
exists is skipped. New runs differ from those of the paper, which are in the release archive
(`export_archive.py`).

## Reference fits

`reference_fits.py pheno remifentanil oral_mm` fits every reference model with PKPy2 through the same session code the
agent uses (no LLM; two starts, the published values and one perturbation; 60-minute limit), which checks that the
specification language expresses it and gives the OFV of the reference model on the same data and engine. Results:
`reference_fits/<dataset>/reference_fit.json`. `reference_table.py` compares the estimates with the published values
(`evaluation/reference_table.csv`, Table S2), and `reference_vpc.py` runs the VPCs of the reference fits with the
settings of the final models (500 simulations, 8 bins, the default seed; `reference_fits/<dataset>/vpc.json`).

## Evaluation

`standard_vpc.py` runs the standard final VPC for the runs in which PKAgent kept the agent's own last VPC of the final
model (`evaluation/standard_vpc.json`); `evaluate.py` reads it, so it runs first. `evaluate.py` then compares the final
model of every run with the reference:

- structure: number of compartments and elimination type;
- covariate relationships: (parameter, covariate) pairs and their functional form, with derived covariates traced to
  their source columns through the data transformations the agent recorded;
- typical values: for every subject, the typical value of each reference parameter under the final model divided
  by the reference model's value at the same covariates (median and range across subjects), which compares
  covariate models independently of their parameterization;
- likelihood: OFV and AIC relative to the PKPy2 fit of the reference model (on the data scale for log-normal error);
- evaluation and process: VPC coverage (without prediction correction), CWRES, largest RSE, shrinkage; fits, LLM
  responses, tokens, cost and hours.

Outputs: `evaluation/runs.csv`, `evaluation/summary.csv`, `evaluation/evaluation.json`,
`evaluation/reference_fit_ratios.json`.

Further scripts:

- `agent_tests.py`: what each agent tested (covariate and structural tests from its model registry) and its tool use
  (`evaluation/agent_tests.json`; no refitting).
- `effect_evidence.py`: the increase in OFV when each covariate relationship is removed from the reference fit
  (`evaluation/effect_evidence.json`, Table S3).
- `scm_baseline.py pheno`: stepwise covariate modeling without an LLM (`evaluation/scm_baseline.json`).
- `backward_baseline.py`: backward elimination from the remifentanil reference model
  (`evaluation/backward_baseline.json`).
- `residual_check.py`: the residual error type of the oral MM simulation (`evaluation/oral_mm_residual_check.json`).
- `recall.py`: the search for statements that refer to prior knowledge of a dataset (used by `evaluate.py` and the
  paper scripts).
- `figures.py`: Figures 2 to 4 of the paper.
- `export_archive.py`: the release archive of the runs, reference fits and evaluation
  (`dist/PKAgent_benchmark_archive.zip`).

The order of all commands, from the data to the submission files, and the paper item each one produces are in
`paper/README.md`.
