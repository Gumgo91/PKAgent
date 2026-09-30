# PKAgent benchmark

Can an LLM agent that drives an open-source estimation engine (PKPy2) reproduce published population
pharmacokinetic models, and what does one sentence of expert knowledge change?

## Datasets and reference models

| Dataset | Data | Design | Reference model |
|---|---|---|---|
| `pheno` | nlmixr2data `pheno_sd` (phenobarbital, Grasela and Donn 1985) | 59 preterm neonates, IV bolus loading and maintenance doses, 155 sparse concentrations | 1-cmt; CL and V proportional to birth weight; V 15.9% larger when the 5-minute Apgar score is below 5; exponential IIV on CL and V; proportional error (NONMEM example) |
| `remifentanil` | nlme `Remifentanil` (Minto et al. 1997) | 65 adults aged 20-85 years, one 4-20 min infusion, 1,992 arterial concentrations | 3-cmt; V1, V2, CL linear in age (centred 40 y) and lean body mass (centred 55 kg), Q2 and Q3 linear in age; no sex effect |
| `oral_mm` | nlmixr2data `Oral_1CPTMM` (simulated, ACOP 2016), subset | 40 subjects, 10/20/40/80 mg single then 7 daily oral doses, 1,000 concentrations | simulation model: 1-cmt, first-order absorption, Michaelis-Menten elimination, 30% IIV on every parameter, 20% proportional error |

`datasets.json` holds the agent-facing description of each dataset (study design and column meanings; it does not
cite the source publication), the one-sentence expert knowledge, and the reference model with its published
estimates. `prepare_data.py` builds the NONMEM-format files in `data/` from the R package exports.
`excluded_nimotuzumab.json` records why the nimotuzumab TMDD dataset was dropped (no verifiable reference model).

## Conditions

- `none`: the agent receives the data file, the description and the standard instructions.
- `knowledge`: the same, plus the expert sentence of `datasets.json`, given as analyst knowledge that the agent
  must weigh against the data.

Each condition is run with `openai/gpt-6.1-sol` and `anthropic/claude-opus-5.5` (OpenRouter), with independent
replicates (different seeds; LLM sampling at provider defaults). Budgets per run: 40 fits, 80 LLM turns,
6 hours, USD 25.

```bash
python benchmarks/run_benchmark.py --dataset pheno --condition knowledge --model claude --rep 1
python benchmarks/run_benchmark.py --all --models gpt claude --reps 3
```

## Reference fits

`reference_fits.py` fits every reference model with PKPy2 through the same session code the agent uses (no LLM),
which checks that the specification language expresses it and gives the OFV of the reference structure on the same
data and engine. Results: `reference_fits/<dataset>/reference_fit.json`.

## Evaluation

`evaluate.py` compares the final model of every run with the reference:

- structure: number of compartments and elimination type;
- covariate relationships: (parameter, covariate) pairs, with derived covariates traced to their source columns
  through the data transformations the agent recorded;
- typical values: for every subject, the typical value of each reference parameter under the final model divided
  by the reference model's value at the same covariates (median and range across subjects), which compares
  covariate models independently of their parameterization;
- likelihood: OFV and AIC relative to the PKPy2 fit of the reference model;
- evaluation and process: VPC coverage, CWRES, largest RSE, shrinkage; fits, LLM turns, tokens, cost and hours.

Outputs: `evaluation/runs.csv`, `evaluation/summary.csv`, `evaluation/evaluation.json`.
