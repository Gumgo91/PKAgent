# Simulation experiment: analyst statements, evidence in the data and instruction wording

In the benchmark, the final model followed the analyst's statement for the two reference effects that the data
support only weakly, and an unsupported two-compartment statement was adopted in every run that received it. Those
results rest on few runs, on published datasets, and on a system prompt that tells the agent to respect analyst
knowledge. This folder holds a pre-specified experiment (A) that measures, on new simulated data with known truth,
how the agent weighs an analyst's statement against the evidence in the data and how much of it is due to the
wording of the prompt, and a second experiment (B) that repeats the no-knowledge runs on the two published datasets
with the names masked. `PROTOCOL.md` is the plan; section numbers below refer to it. The new runs are never pooled
with the 44 benchmark runs, and nothing under `src/` or in the benchmark folders is changed.

| Module | Question | Data | Agent runs |
|---|---|---|---|
| A1 | How does the probability that the final model contains a covariate effect depend on the evidence in the data, the analyst's statement (none, asserts, denies) and the instruction paragraph of the system prompt (benchmark wording, neutral wording, neutral wording plus an explicit rule)? | 28 simulated datasets: one compartment, intravenous infusion, 60 subjects, a binary comedication effect on clearance calibrated to a reference dOFV between 0 and 40 | 192 |
| A2 | Is an unsupported two-compartment statement adopted, under the benchmark wording and under the explicit rule? | 8 of the 28 datasets | 32 |
| B | Do the no-knowledge results on phenobarbital and remifentanil depend on the drug and file names, and on the example `"APGR < 5"` in a tool description? | the two benchmark data files, byte-identical, as `pk_data.csv` with a masked description | 22 |
| Probe | Can the LLMs name the drug and the source from a description and a data summary? | 5 descriptions | 50 single LLM calls |
| Pilot | Technical check of runner, prompt variants, tool patch, scoring, time and fees | 2 further simulated datasets, blinded phenobarbital | 10, not analyzed |
| N1 | Does the instruction paragraph change model selection when there is no statement? | 8 of the 28 datasets | 16, if fees and time allow |

Every run is one dataset under one condition with one LLM (`openai/gpt-6.1-sol` or `anthropic/claude-opus-5.5`, the
settings of the benchmark). The primary endpoint is whether the typical clearance of the final model depends on the
target comedication; contrasts are paired within dataset and tested with an exact sign-flip test on the datasets.

## Files

| File | Content |
|---|---|
| `PROTOCOL.md`, `DEVIATIONS.md` | the plan; the log of deviations and of the decisions of sections 10.1 and 10.3 |
| `strings.json` | the exact texts given to the agent (data description, statements, instruction paragraphs, tool example, blinded descriptions, probe message) with their sha256 |
| `common.py` | shared definitions: paths, settings, seeds, slot table and dataset IDs, the cells of every module with their run order, prompt and tool variants, acceptance of attempts, fees spent, columns of `runs.csv`; `--check` verifies every hash and the cell counts |
| `generate.py` | simulated datasets: covariates, sampling times, true model, simulation, CSV writer, generator test |
| `calibrate.py` | engine freeze, calibration of the evidence (steps 0 to 7 of section 5), pilot datasets; writes `design.json` |
| `design.json` | the 28 datasets with slot, seed, effect size and reference evidence, the sha256 of every data file, the engine manifest and the run order |
| `data/<ID>/pk_data.csv` | the 28 simulated datasets |
| `run_grid.py` | runner of the pilot, A, B, N1 and the probe: one child process per run, checks before and after, sidecar `condition.json`, `served.jsonl`, failure classes and reruns, fee guards, watchdog, resume; stage 0 (replay, dry run, load test) |
| `replay_llm.py` | scripted stand-in for the LLM (stage 0; no key, no fees) |
| `score.py` | endpoints of every run of A (`evaluation/runs.csv`) and the excerpts for the coded misreport endpoint |
| `code_reports.py` | coding of the misreport excerpts by the two LLMs of the study: one call per excerpt and coder with the message of section 4.6, resume, fee guard; `--check` verifies the message against its sha256 and the protocol |
| `analyze.py`, `sim_figures.py` | statistical analysis (`evaluation/summary.json`, `evaluation/tables/`) and figures |
| `test_scoring.py` | tests of the scoring code, of the coding (with scripted answers) and of the rule for the parallel runs |
| `blind/evaluate_blind.py` | evaluation of B and of the probe, next to the 12 named no-knowledge runs of the benchmark |
| `coding/misreport_excerpts.csv`, `coder_calls.jsonl`, `misreport_codes.csv`, `misreport_recode.csv` | excerpts of the reports under random codes, every call of the two coders with its answer and fees, the sheet with both categories and the misreport, and the sheet for the optional check by a person |

Not tracked (`.gitignore`): `_engine/` (frozen copy of PKPy2), `calibration/`, `pilot/`, `runs/`, `runs_failed/`,
`evaluation/`, `blind/data/`, `blind/runs/`, `blind/evaluation/` and `smoke/` (output of tests). The runner writes
`load.csv` (CPU load once a minute) and `grid_log.jsonl` into the run root of a grid (`runs/`, `pilot/runs/`,
`blind/runs/`), the probe writes `blind/runs/probe.jsonl`, and the key that maps the excerpt codes to runs is
`evaluation/misreport_key.csv`.

## Commands

Run from the repository root with `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`. Paid runs and calls need
`OPENROUTER_API_KEY` and start only with `--execute`; everything else calls no LLM. The fee guards of the scripts
stop the experiment below 100 USD; a credit limit of 100 USD on the key is optional and would be a second stop.

### 1. Engine and tests

```bash
python benchmarks/simulation/calibrate.py engine          # freeze PKPy2 to _engine/, refit the phenobarbital reference
python benchmarks/simulation/common.py --check            # hashes of strings.json, slot table, cell counts
python benchmarks/simulation/generate.py --test 300       # generator test of section 2.3
python benchmarks/simulation/generate.py --mock           # mock design and data for the tests, in smoke/
python benchmarks/simulation/test_scoring.py [--fit]      # scoring against the benchmark and synthetic final models;
                                                          #   coding with scripted answers; rule for the parallel runs
python benchmarks/simulation/analyze.py --self-test       # tests, interval, multiplicity, Firth fit, power, figures
python benchmarks/simulation/blind/evaluate_blind.py --check
python benchmarks/simulation/code_reports.py --check      # message to the coders: its sha256 and PROTOCOL.md
```

`calibrate.py engine` copies the installed PKPy2 package to `_engine/pkpy2-<version>-<hash8>/` with a manifest of
its files. Calibration and runs import PKPy2 from this copy only, and every run checks the manifest.

### 2. Stage 0: the runner without an LLM

```bash
python benchmarks/simulation/run_grid.py --replay benchmarks/runs/pheno/knowledge/claude/rep1
python benchmarks/simulation/run_grid.py --replay benchmarks/runs/pheno/knowledge/claude/rep1 --variant never_finalize
python benchmarks/simulation/run_grid.py --replay benchmarks/runs/pheno/knowledge/claude/rep1 --variant api_error
python benchmarks/simulation/run_grid.py --replay benchmarks/runs/pheno/knowledge/claude/rep1 --variant hang \
    --stall-seconds 60
python benchmarks/simulation/replay_llm.py --script benchmarks/simulation/smoke/script
python benchmarks/simulation/run_grid.py --module A --blocks 2 --design smoke/design_mock.json \
    --stand-in benchmarks/simulation/smoke/script --jobs 1 --max-launches 2
```

`--replay` runs one cell that replays the tool calls of a recorded benchmark run through the real runner and
compares the OFVs with the recorded ones (output in `smoke/replay`, or the folder below `smoke/` named with `--out`;
a replay that is already complete there is not run again); the variants must end in the failure classes of
section 8.7. `--stand-in` runs cells of the mock design with a script in place of the LLM; `score.py`, `analyze.py`
and `sim_figures.py` accept `--design smoke/design_mock.json` to follow them through, and
`code_reports.py --design smoke/design_mock.json --stand-in <file of scripted answers>` codes their excerpts without
an LLM (it writes below `smoke/` only).

### 3. Calibration and pilot datasets

```bash
python benchmarks/simulation/calibrate.py run --jobs 2    # steps 1 to 5 and 7; writes data/ and design.json
python benchmarks/simulation/calibrate.py pilot --jobs 2  # pilot datasets P1 and P2
python benchmarks/simulation/generate.py --verify         # the files equal the generator output and design.json
python benchmarks/simulation/common.py --check
python benchmarks/simulation/run_grid.py --module A       # dry run: prompt hash, tool-list hash, task message
python benchmarks/simulation/run_grid.py --module B       #   of every cell (also pilot, N1, probe)
python benchmarks/simulation/run_grid.py --load-test      # fits per hour with 1 and 2 parallel runs (time estimate)
```

The calibration screens candidate seeds without an effect, fills the 28 slots, solves the effect size of every
dataset for its evidence target and recomputes the evidence of the final files through the agent's fit path. It can
be stopped and started again; finished fits are read from `calibration/fits/`. Every candidate seed is listed with
its reason in `calibration/candidates.csv`. `calibrate.py options --remove-a2` or `--remove-named-controls` applies
the two decisions that section 10.1 leaves to the freeze.

### 4. Pilot

```bash
python benchmarks/simulation/run_grid.py --module pilot --jobs 2 --execute
python benchmarks/simulation/run_grid.py --module pilot --status      # go criteria of section 3.9
python benchmarks/simulation/score.py --module pilot
python benchmarks/simulation/blind/evaluate_blind.py --pilot
```

The protocol, the scripts, `strings.json`, `design.json` and the 28 data files are committed and pushed before the
pilot, and again after the pilot and before the first run of A (the freeze, section 10.1); the pushed commits are the
public time stamp of the plan.

### 5. Runs

```bash
python benchmarks/simulation/run_grid.py --module A --blocks 1 2 --jobs 2 --execute
python benchmarks/simulation/run_grid.py --module A --status          # states, fees, served models, projection
python benchmarks/simulation/run_grid.py --module A --blocks 3 4 --jobs 2 --execute
python benchmarks/simulation/run_grid.py --module B --jobs 1 --execute
python benchmarks/simulation/run_grid.py --module probe --execute
python benchmarks/simulation/run_grid.py --module N1 --jobs 2 --execute
python benchmarks/simulation/calibrate.py baseline --jobs 2           # step 6, before the runs of A or after B
```

A cell is complete when its accepted attempt exists; an interrupted, failed or invalid attempt is moved to
`runs_failed/<same path>/attempt<k>/` and the cell is run again as section 8.7 allows. The runner starts no run of
the pilot or of A at 83 USD of fees or more and no run of B, N1 or the probe at 95 USD or more, stops after an HTTP
status 401 to 404, after two consecutive infrastructure failures, after three consecutive failures, and when a run
was served another model identifier than the runs before it; started again, it continues with the open cells.
The machine is shared with other work, which is not stopped: the runner accepts 1 or 2 parallel runs for the pilot,
A and N1 and exactly 1 for B (section 6); without `--jobs` it takes 2, and 1 for B. While a grid runs,
`<run root>/jobs.txt` (one integer) lowers the number of parallel runs and a file `<run root>/STOP` stops new
launches. `--reduction 1` (without block 4) and `--reduction 2` (also without the deny-neutral cells of block 3) apply
the reduction of section 10.3.

### 6. Scoring, coding and analysis

No run of A or B is scored before all grids are closed, the N1 extension included or waived (section 10.2):
`score.py`, `code_reports.py`, `analyze.py` and `blind/evaluate_blind.py` stop while a cell is open. `--reduction 1|2`
and `--n1-waived` name what was decided under section 10.3.

```bash
python benchmarks/simulation/score.py                     # evaluation/runs.csv, coding/misreport_excerpts.csv
python benchmarks/simulation/code_reports.py              # dry run: excerpts, calls, fees spent, the first message
python benchmarks/simulation/code_reports.py --execute    # coding/coder_calls.jsonl, coding/misreport_codes.csv
#   optional check by a person: S, Q or N for the 30 excerpts of coding/misreport_recode.csv
python benchmarks/simulation/analyze.py                   # evaluation/summary.json, evaluation/tables/*.csv
python benchmarks/simulation/sim_figures.py               # evaluation/figures/
python benchmarks/simulation/blind/evaluate_blind.py      # blind/evaluation/: runs.csv, tables.md, matches.csv
python benchmarks/simulation/blind/evaluate_blind.py --probe
```

`code_reports.py` sends every excerpt to the two LLMs of the study, each as one user message without tools (the
template of section 4.6 with the statement, the sentences that name its subject and the report section on the
analyst's statements), and reads the category S, Q or N from the answer. An answer without a category is requested
again, at most three calls for an excerpt and coder; after that the excerpt has no category from that coder. The
script can be stopped and started again, makes no call at 97 USD of fees or more, and writes `misreport_codes.csv`
when every excerpt has the result of both coders. A misreport is S from both coders. The calls and the sheet are
committed before `analyze.py` is run; `analyze.py` stops while an excerpt lacks the result of a coder, and with
`--coding-not-made` (the fee guard prevented the coding) it reports the rule-based score alone. A person who fills in
`coding/misreport_recode.csv` does so without the categories of the coders; `analyze.py` then reports the agreement
with each coder and with the misreport classification, and otherwise states that no check by a person was made.

## Outputs

- `runs/<ID>/<statement>-<instruction>/<llm>/rep1/` and `blind/runs/<arm>/none/<llm>/rep<k>/`: the run folder of
  PKAgent (`results.json`, `report.md`, `transcript.jsonl`, `models/`), plus `condition.json` (cell, hashes of
  prompt, tool list and data, settings, versions, checks), `served.jsonl` (served model identifier of every response)
  and `exit.json`.
- `evaluation/runs.csv`: one row per cell of A with the primary endpoint `Y`, the size of the adopted effect, twin
  adoption, the two-compartment endpoint, background recovery, the run's own tests of the target effect, the
  rule-based report score and process measures; `evaluation/runs_columns.csv` explains every column and
  `evaluation/attempts.csv` lists every attempt with its failure class.
- `evaluation/summary.json` and `evaluation/tables/`: the paired contrasts with exact tests and intervals, the gate
  and Holm decisions of the three families, the readings fixed in section 1.4, the curve model, controls,
  sensitivity analyses, descriptive tables and the coded misreport endpoint (by arm, with the agreement of the two
  coders and their categories by the LLM that wrote the report); `evaluation/figures/`: adoption against the evidence.
- `blind/evaluation/`: every run of B next to the named runs of October, with recall, identification of the drug and
  the dichotomization of the Apgar score; `probe_calls.csv` and `probe_summary.csv` for the probe.
