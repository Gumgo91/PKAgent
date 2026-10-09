"""One archive of the benchmark outputs (runs, reference fits, evaluation) without data files or local paths.

Contents: runs (every run: prompts, transcripts with reasoning summaries, tool calls and results, fitted models,
plots, reports), reference_fits and evaluation (incl. agent tests, stepwise baseline, drop-one evidence). The data
files are not redistributed (folders named 'data' are left out), although the per-model diagnostics include the
observations; benchmarks/export_data.R and prepare_data.py recreate them from the R packages nlme and nlmixr2data.
Local absolute paths are removed from all text files.
--text-only also leaves out the plots (*.png) and the per-model diagnostics (diagnostics.csv, one row per observation)
and puts a README.txt on the layout and the files at the root of the archive.
Writes dist/PKAgent_benchmark_archive.zip, or the file given with --out.
Usage: python benchmarks/export_archive.py [--text-only] [--out PATH]
"""
import argparse
import json
import re
import textwrap
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FOLDERS = ['runs', 'reference_fits', 'evaluation']
TEXT = {'.json', '.jsonl', '.csv', '.md', '.log', '.txt'}
PREFIX = re.compile(r'[A-Za-z]:(?:\\\\|\\|/)Users(?:\\\\|\\|/)[^\\/"]+(?:\\\\|\\|/)Desktop(?:\\\\|\\|/)PKAgent(?:\\\\|\\|/)')
HOME = re.compile(r'[A-Za-z]:(?:\\\\|\\|/)Users(?:\\\\|\\|/)[^\\/"\s]+')
REPOSITORY = 'https://github.com/Gumgo91/PKAgent'
CONDITIONS = dict(none='no knowledge', knowledge='expert statement', misleading='misleading statement')


LOCAL = re.compile(r'~(?:\\\\|\\|/)Desktop(?:\\\\|\\|/)(?:pkpy2(?:\\\\|\\|/)packages(?:\\\\|\\|/)pkpy2(?:\\\\|\\|/)src(?:\\\\|\\|/))?')


def sanitize(text):
    text = PREFIX.sub('', text)
    return LOCAL.sub('', HOME.sub('~', text))       # PKPy2 source paths in console logs become 'pkpy2\...'


def left_out(p, text_only):
    """None if the file goes into the archive, otherwise why not ('data', 'plot', 'diagnostics', 'other')."""
    if p.suffix == '.pyc' or 'data' in p.relative_to(HERE).parts[1:-1]:
        return 'data'
    if not text_only:
        return None
    if p.suffix == '.png':
        return 'plot'
    if p.name == 'diagnostics.csv':
        return 'diagnostics'
    return None if p.suffix in TEXT else 'other'


def readme(skipped):
    """README.txt of the text-only archive: layout, contents of the files, and what was left out and why."""
    runs = sorted((HERE / 'runs').glob('*/*/*/rep*/run.json'))
    by_condition = Counter(p.parts[-4] for p in runs)
    models = {p.parts[-3]: json.loads(p.read_text(encoding='utf-8')).get('model') for p in runs}
    datasets = json.loads((HERE / 'datasets.json').read_text(encoding='utf-8'))
    sources = '; '.join(f"{k}: {re.match(r'[^ ;(]+', v.get('source', '')).group(0)}" for k, v in datasets.items()
                        if (HERE / 'runs' / k).is_dir())
    counts = ', '.join(f'{by_condition[c]} {v}' for c, v in CONDITIONS.items() if c in by_condition)
    llms = '; '.join(f'{k}: {v}' for k, v in sorted(models.items()))
    conditions = '; '.join(f'{k}: {v}' for k, v in CONDITIONS.items() if k in by_condition)
    intro = textwrap.fill(
        f'This archive holds the text files of the PKAgent benchmark: the logs of all {len(runs)} runs ({counts}), '
        'the PKPy2 fits of the reference models (without a language model), and the evaluation of the final models. '
        'PKAgent, the benchmark definition (benchmarks/datasets.json) and the scripts that ran and evaluated the '
        f'benchmark are available at {REPOSITORY} (described in benchmarks/README.md there). Local file paths were '
        'removed from all files. JSON files are UTF-8 text; JSONL files hold one JSON object per line.', 116)
    left = '\n'.join(textwrap.fill(t, 116, initial_indent='- ', subsequent_indent='  ') for t in [
        'Data files (folders named data): the source data are not redistributed. They are Remifentanil from the R '
        'package nlme 3.1-168 and pheno_sd and Oral_1CPTMM from the R package nlmixr2data 2.0.10 (CRAN); '
        'benchmarks/export_data.R and benchmarks/prepare_data.py in the PKAgent repository rebuild the data files of '
        'the runs.',
        f"diagnostics.csv of every model ({skipped['diagnostics']} files): predictions and residuals per observation, "
        'which include the observations themselves.',
        f"Plots (*.png, {skipped['plot']} files): images of the data, goodness of fit, individual fits and VPCs, left "
        'out to keep the archive small.',
        'The pilot runs made while the tools were developed.'])
    return f"""PKAgent benchmark: logs of the runs, reference fits and evaluation outputs (text files)

{intro}

LAYOUT
benchmarks/runs/<dataset>/<condition>/<llm>/rep<k>/              one run, replicate k (files below)
benchmarks/runs/<dataset>/<condition>/<llm>/rep<k>.console.log   console output of that run (abridged)
benchmarks/runs/grid*.log                                        start and end times of the runs
benchmarks/reference_fits/<dataset>/                             PKPy2 fit of the reference model
benchmarks/evaluation/                                           comparison with the reference models, baselines
  <dataset>    {sources}
  <condition>  {conditions}
  <llm>        {llms}

FILES OF A RUN
run.json          settings: language model, data file name, dataset description and statement given to the agent,
                  seed, limits on fits, responses, hours and fees, reasoning effort, start time, PKAgent and PKPy2
                  versions and PKAgent commit
messages.json     the conversation with the language model: system prompt, task message, assistant messages with
                  tool calls and reasoning details, and tool results ("<png>" marks an image that was sent)
transcript.jsonl  one line per language-model response: text, tool calls, finish reason, reasoning summary (when
                  the provider returned one) and token usage
tool_log.jsonl    one line per tool call: tool, arguments, result and seconds
models/M<nnn>/    every fitted model: spec.json (model specification), fit.json (estimates, OFV, convergence check,
                  conditional modes, and standard errors when computed), summary.json (the summary returned to the
                  agent), progress.jsonl (phases of the minimization)
final/            the same files for the final model
results.json      resources used, the final model (specification, summary, VPC and report), the registry of all
                  models and the data transformations
report.md         the final report

REFERENCE FITS (benchmarks/reference_fits/<dataset>/)
reference_fit.json  status, OFV and estimates of the fit; vpc.json  its VPCs without and with prediction correction;
models/M<nnn>/      the fitted model (files as above). oral_mm_proportional is the earlier oral MM reference fit with
                    proportional instead of log-normal residual error.

EVALUATION (benchmarks/evaluation/; written by the scripts in benchmarks/ of the repository named in parentheses)
runs.csv, evaluation.json   per run: structure, covariate relationships, typical values relative to the reference
                            model, OFV and AIC against the reference fit, diagnostics and resources (evaluate.py)
summary.csv                 medians by dataset, condition and language model (evaluate.py)
reference_fit_ratios.json   typical values of the reference fits relative to the reference values (evaluate.py)
agent_tests.json            covariate and structural tests in each run's model registry, and its tool use
                            (agent_tests.py)
effect_evidence.json, effect_evidence/      increase in OFV when each reference covariate relationship is removed
                                            from the reference fit, and the refitted models (effect_evidence.py)
scm_baseline.json, scm_baseline/            stepwise covariate modeling without a language model (scm_baseline.py)
backward_baseline.json, backward_baseline/  backward elimination from the remifentanil reference model
                                            (backward_baseline.py)
standard_vpc.json           the standard final VPC of runs in which the agent's own VPC of the final model was kept
                            (standard_vpc.py)
reference_table.csv, .md    reference values and the PKPy2 fits of the reference models (reference_table.py)
oral_mm_residual_check.json residual error type of the oral MM simulation (residual_check.py)

LEFT OUT
{left}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--text-only', action='store_true',
                    help='leave out plots (*.png) and per-model diagnostics.csv; add README.txt')
    ap.add_argument('--out', type=Path, default=ROOT / 'dist' / 'PKAgent_benchmark_archive.zip',
                    help='archive to write (default: dist/PKAgent_benchmark_archive.zip)')
    args = ap.parse_args()
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    n, skipped = 0, Counter()
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for folder in FOLDERS:
            for p in sorted((HERE / folder).rglob('*')):
                if not p.is_file():
                    continue
                why = left_out(p, args.text_only)
                if why:
                    skipped[why] += 1
                    continue
                arc = Path('benchmarks') / p.relative_to(HERE)
                if p.suffix in TEXT:
                    z.writestr(str(arc).replace('\\', '/'), sanitize(p.read_text(encoding='utf-8', errors='replace')))
                else:
                    z.write(p, str(arc).replace('\\', '/'))
                n += 1
        if args.text_only:
            z.writestr('README.txt', readme(skipped))
    with zipfile.ZipFile(out) as z:                       # check that no local path or API key is left
        texts = {i.filename: z.read(i).decode('utf-8', 'replace') for i in z.infolist()
                 if Path(i.filename).suffix in TEXT}
    left = [f for f, t in texts.items() if re.search(r'Users[\\/]+[^\\/]+[\\/]+Desktop|~[\\/]+Desktop', t)]
    keys = [f for f, t in texts.items() if re.search(r'sk-or-|OPENROUTER_API_KEY\s*=', t)]
    print(f'wrote {out} ({n} files, {out.stat().st_size / 1e6:.1f} MB); files with local paths: {len(left)}; '
          f'with API keys: {len(keys)}' + (f"; left out: {dict(skipped)}" if args.text_only else ''))


if __name__ == '__main__':
    main()
