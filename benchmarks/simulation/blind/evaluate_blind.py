"""Evaluation of experiment B, the name-blinded sensitivity runs (PROTOCOL.md, section 9).

The runs of B (blind/runs/<arm>/none/<llm>/rep<k>/) are scored with the definitions of benchmarks/evaluate.py, the
arms being mapped to the benchmark datasets pheno and remifentanil, next to the 12 named no-knowledge runs of the
benchmark (benchmarks/runs/pheno/none and remifentanil/none, October 2026). For every run:
- structure, strong and weak reference relationships retained, extra relationships, typical values against the
  reference model, OFV and AIC against the reference fit, and the agent's own tests (benchmarks/agent_tests.py);
- recall (the expression of benchmarks/recall.py), identification of the drug or its source, and the dichotomization
  of the Apgar score, searched in the assistant text, the tool-call arguments and the reasoning summaries;
- fits, responses, fees and hours, the served model identifiers (served.jsonl) and the CPU load during the run
  (load.csv of the runner).
Output under blind/evaluation/: runs.csv (one row per run), matches.csv (every match of the three expressions with
context), transformations.csv (every recorded data transformation), comparison.csv and tables.md (one table per
dataset and LLM, blinded against named runs), evaluation.json; probe_calls.csv and probe_summary.csv (--probe);
benchmark_check.csv (--check).

Usage, from the repository root:
  python benchmarks/simulation/blind/evaluate_blind.py --prepare   copy the two data files to blind/data/
  python benchmarks/simulation/blind/evaluate_blind.py [--reduction 1|2] [--n1-waived]
                                                                   evaluate B once all grids are closed (section 10.2)
  python benchmarks/simulation/blind/evaluate_blind.py --partial   evaluate the accepted runs that exist (tests)
  python benchmarks/simulation/blind/evaluate_blind.py --pilot     the blinded pilot runs (pilot/blind_evaluation/)
  python benchmarks/simulation/blind/evaluate_blind.py --probe     score the recognition probe (probe.jsonl)
  python benchmarks/simulation/blind/evaluate_blind.py --check     tests of section 7.3 on the 12 named benchmark runs
Options: --design PATH (default design.json, or the slot table while it does not exist), --load PATH (another
load.csv than that of the runner), --probe-file PATH (another probe.jsonl).
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common                                                                                    # noqa: E402

common.add_paths()
import agent_tests                                                                               # noqa: E402
from evaluate import (REFERENCE_COVARIATES, _match, _sources, covariate_relationships,           # noqa: E402
                      form_agreement, reference_typical, typical_values)
from recall import RECALL, recalled, turn_text                                                   # noqa: E402

DATASETS = common.benchmark_datasets()                  # benchmarks/datasets.json
SOURCES = common.B_SOURCES                              # benchmark dataset of an arm: common.B_ARMS[arm]['source']
LABEL = {'pheno': 'Phenobarbital', 'remifentanil': 'Remifentanil'}
OCTOBER = common.BENCHMARKS / 'runs'                    # named runs of the benchmark: <source>/none/<llm>/rep<k>/
# the weak reference relationship of each dataset (reference-fit OFV rise 4.5 and 2.7); the others are strong
WEAK = {'pheno': ('V', 'APGR'), 'remifentanil': ('V1', 'AGE')}
STRONG = {s: REFERENCE_COVARIATES[s] - {WEAK[s]} for s in SOURCES}
PARAMETERS = ('CL', 'V', 'V1', 'V2', 'V3', 'Q2', 'Q3')  # reference parameters of the two datasets

# rows of the tables: the benchmark runs of October and the arms of B
GROUPS = {'named_october': 'named, October', 'named_concurrent': 'named, concurrent', 'blinded': 'blinded',
          'blinded_example': 'blinded, original example'}
GROUP_OF = {'pheno_named': 'named_concurrent', 'pheno_blind': 'blinded', 'pheno_blind_ex': 'blinded_example',
            'remifentanil_named': 'named_concurrent', 'remifentanil_blind': 'blinded'}

# expressions of section 9, searched without regard to case as the recall expression is
IDENTIFICATION = re.compile(r'phenobarb|barbitur|remifentanil|opioid|fentanyl|esterase|anticonvuls|antiepilep|seizure'
                            r'|anesth|anaesth|grasela|donn\b|minto', re.IGNORECASE)
APGR_AT_5 = re.compile(r'APGR\s*<\s*5|APGR\s*<=\s*4', re.IGNORECASE)
PROBE_DRUG = re.compile(r'phenobarb|remifentanil', re.IGNORECASE)
PROBE_SOURCE = re.compile(r'grasela|donn\b|minto|nlmixr|nlme|nonmem', re.IGNORECASE)
# the terms of the two probe expressions that are correct for a dataset: its drug, its authors, or a package
PROBE_TERMS = {'pheno': dict(drug='phenobarb', authors=r'grasela|donn\b'),
               'remifentanil': dict(drug='remifentanil', authors='minto')}
PROBE_PACKAGES = 'nlmixr|nlme|nonmem'
PARTS = ('text', 'tool_calls', 'reasoning')             # the parts of a turn that are searched

RUN_COLUMNS = (
    ['set', 'dataset', 'arm', 'group', 'llm', 'rep', 'run', 'tools', 'accepted', 'state', 'attempt', 'n_attempts',
     'reason', 'started', 'ended', 'tools_sha256', 'data_sha256', 'served', 'finalized', 'model_id', 'description',
     'status',
     'compartments', 'elimination', 'structure_match', 'covariates_found', 'strong_retained', 'strong_total',
     'strong_missing', 'weak_relationship', 'weak_retained', 'extra', 'n_extra', 'covariates_exact', 'form_agreement']
    + [f'ratio_{p}_{k}' for p in PARAMETERS for k in ('median', 'range')]
    + ['parameters_missing', 'max_abs_log_ratio', 'all_within_20pct', 'reproduced', 'ofv', 'n_estimated', 'aic',
       'delta_ofv_vs_reference', 'delta_aic_vs_reference', 'weak_tested', 'weak_d_max', 'weak_d_final',
       'covariate_tests', 'compartment_tests', 'recall', 'recall_first_turn', 'recall_terms', 'identification',
       'identification_first_turn', 'identification_terms', 'apgr_at_5', 'apgr_at_5_turn', 'apgr_other',
       'apgr_other_turn', 'apgr_columns', 'apgr_text_first_turn', 'fits', 'responses', 'prompt_tokens',
       'completion_tokens', 'cost_usd', 'hours', 'cpu_load_median', 'cpu_load_minutes'])
INTEGERS = ('rep', 'attempt', 'n_attempts', 'compartments', 'strong_retained', 'strong_total', 'n_extra', 'n_estimated',
            'covariate_tests', 'recall_first_turn', 'identification_first_turn', 'apgr_at_5_turn', 'apgr_other_turn',
            'apgr_text_first_turn', 'fits', 'responses', 'prompt_tokens', 'completion_tokens', 'cpu_load_minutes')
# endpoints of the comparison of blinded with named runs (tables.md, comparison.csv)
COMPARED = (('structure_match', 'reference structure'), ('strong_retained', 'strong relationships retained'),
            ('weak_retained', 'weak relationship retained'), ('n_extra', 'extra relationships'),
            ('all_within_20pct', 'typical values within 0.80 to 1.25'),
            ('max_abs_log_ratio', 'largest absolute log ratio'), ('reproduced', 'reproduced'),
            ('delta_ofv_vs_reference', 'OFV minus reference fit'),
            ('delta_aic_vs_reference', 'AIC minus reference fit'),
            ('recall', 'recall'), ('identification', 'identification'), ('apgr_at_5', 'APGR dichotomized at 5'),
            ('fits', 'fits'), ('hours', 'hours'))


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _rel(path):
    """Path relative to the repository, with forward slashes."""
    return Path(path).resolve().relative_to(common.REPO).as_posix()


def load_design(path=None):
    """The design named on the command line (absolute, relative to the working directory or to
    benchmarks/simulation), else design.json, else the slot table, which is enough for the cells of B."""
    if path:
        return common.load_design(common.design_path(path))
    return common.load_design() if common.DESIGN.exists() else common.template_design()


# ---------------------------------------------------------------------- blinded data copies
def prepare(design, report=print):
    """Write blind/data/<source>_blind/pk_data.csv, byte-identical copies of benchmarks/data/pheno.csv and
    remifentanil.csv, and record their sha256 in blind/data/sha256.json (common.prepare_blind, which the runner
    also calls before the first run of B). An existing copy with other bytes is an error and is not replaced."""
    problems = common.prepare_blind(design, write=True, report=report)
    common.require(not problems, '; '.join(problems))
    return common.blind_copies(design)


# ---------------------------------------------------------------------- endpoints of benchmarks/evaluate.py
def reference_fit(source):
    """OFV and AIC of the PKPy2 fit of the reference model (benchmarks/reference_fits.py), or None."""
    path = common.BENCHMARKS / 'reference_fits' / source / 'reference_fit.json'
    return _json(path) if path.exists() else None


def final_data(run_dir, final):
    """The data file of the final model: the file of that name in the run folder (results.json holds the absolute
    path at the time of the run, which evaluate.py reads)."""
    local = run_dir / 'data' / Path(final['data_file'].replace('\\', '/')).name
    return local if local.exists() else Path(final['data_file'])


def endpoints(run_dir, source):
    """Endpoints of one run as benchmarks/evaluate.evaluate_run defines them, for a run on the data of `source`
    (pheno or remifentanil). Returns (row, detail); a run without a final model has the process columns only."""
    res = _json(run_dir / 'results.json')
    run = res['run']
    row = dict(finalized=run['finalized'], fits=run['fits'], responses=run['llm_calls'],
               prompt_tokens=run['prompt_tokens'], completion_tokens=run['completion_tokens'],
               cost_usd=run['cost_usd'], hours=run['hours'])
    final = res.get('final_model')
    if not final:
        return row, None
    spec, s = final['specification'], final['summary']
    ref = DATASETS[source]['reference']['structure']
    st = spec['structure']
    kind = st.get('type', 'pk')
    row.update(model_id=final['model_id'], description=final['description'], status=s.get('status'),
               compartments=st.get('compartments', 1), ofv=s.get('ofv'), n_estimated=s.get('n_estimated'),
               aic=s.get('aic'))
    if kind == 'michaelis_menten':
        row['elimination'] = 'Michaelis-Menten' + (' + linear' if st.get('linear_clearance') else '')
    else:
        row['elimination'] = 'linear' if kind == 'pk' else kind
    row['structure_match'] = bool(row['compartments'] == ref['compartments']
                                  and row['elimination'] == ref['elimination'])
    data = pd.read_csv(final_data(run_dir, final), na_values=['.'])
    base_columns = set(pd.read_csv(common.BENCHMARKS / 'data' / DATASETS[source]['file'], nrows=1).columns)
    transformations = res.get('data_transformations', [])
    subjects = data.groupby('ID').first().reset_index()
    ref_tv = reference_typical(source, subjects)
    found = covariate_relationships(spec, transformations, base_columns, set(ref_tv))
    target = REFERENCE_COVARIATES[source]

    def names(pairs):
        return ';'.join(f'{p}~{c}' for p, c in sorted(pairs))
    row.update(covariates_found=names(found), strong_retained=len(found & STRONG[source]),
               strong_total=len(STRONG[source]), strong_missing=names(STRONG[source] - found),
               weak_relationship='~'.join(WEAK[source]), weak_retained=WEAK[source] in found,
               extra=names(found - target), n_extra=len(found - target), covariates_exact=found == target,
               form_agreement=form_agreement(spec, transformations, base_columns, set(ref_tv), source))
    agent_tv = typical_values(spec, s, subjects)
    ratios = {}
    for p, ref_values in ref_tv.items():
        q = _match(p, agent_tv)
        if q is None:
            continue
        r = agent_tv[q] / ref_values
        ratios[p] = dict(median=float(np.median(r)), min=float(np.min(r)), max=float(np.max(r)))
        row[f'ratio_{p}_median'] = ratios[p]['median']
        row[f'ratio_{p}_range'] = f"{ratios[p]['min']:.3f}-{ratios[p]['max']:.3f}"
    row['parameters_missing'] = ';'.join(p for p in ref_tv if p not in ratios)
    if ratios:
        row['max_abs_log_ratio'] = max(abs(math.log(v['median'])) for v in ratios.values())
        row['all_within_20pct'] = (not row['parameters_missing']
                                   and all(.8 <= v['median'] <= 1.25 for v in ratios.values()))
    row['reproduced'] = bool(row['structure_match'] and row['covariates_exact'] and row.get('all_within_20pct'))
    rf = reference_fit(source)
    if rf and rf.get('ofv') is not None and s.get('ofv') is not None:
        row['delta_ofv_vs_reference'] = s['ofv'] - rf['ofv']
        row['delta_aic_vs_reference'] = s['aic'] - rf['aic']
    return row, dict(ratios=ratios, covariates=sorted(found), description=final['description'])


def own_tests(run_dir, runs_root, source):
    """benchmarks/agent_tests.analyze for a run below `runs_root` (common.own_tests): the first level of the run
    folder (an arm of B) gets the reference parameter names of its dataset; the definitions are those of the
    benchmark."""
    return common.own_tests(run_dir, runs_root, agent_tests.REFERENCE_NAMES[source])


def weak_tests(tests, source):
    """The run's own evidence for the weak reference relationship: tested (a pair of converged models that differ in
    it only), the largest OFV difference, and the difference in the covariate and stochastic context of the final
    model (the last one if several)."""
    pairs = [t for t in tests['covariate_tests'] if t['relationship'] == '~'.join(WEAK[source])]
    final = [t['delta_ofv'] for t in pairs if t.get('final_context') and t.get('final_stochastic')]
    return dict(weak_tested=bool(pairs), weak_d_max=max(t['delta_ofv'] for t in pairs) if pairs else None,
                weak_d_final=final[-1] if final else None)


# ---------------------------------------------------------------------- expressions in the transcript
def turn_parts(turn):
    """Assistant text, tool calls and reasoning summary of a turn, the parts that recall.turn_text joins by a blank."""
    parts = [turn.get('content') or '', json.dumps(turn.get('tool_calls') or '', ensure_ascii=False)]
    reasoning = turn.get('reasoning')
    if reasoning:
        parts.append(reasoning if isinstance(reasoning, str) else json.dumps(reasoning, ensure_ascii=False))
    return parts


def matches(run_dir, pattern, context=120):
    """Matches of an expression in a run, found as benchmarks/recall.recalled finds them (the text of
    recall.turn_text for every turn of transcript.jsonl): turn, part of the turn, matched term, surrounding text."""
    out = []
    for turn in common.read_jsonl(run_dir / 'transcript.jsonl'):
        text, parts = turn_text(turn), turn_parts(turn)
        common.require(' '.join(parts) == text, 'turn_parts differs from recall.turn_text')
        ends = np.cumsum([len(p) + 1 for p in parts])
        for m in pattern.finditer(text):
            part = PARTS[min(int(np.searchsorted(ends, m.start(), side='right')), len(PARTS) - 1)]
            out.append(dict(turn=turn['turn'], part=part, term=m.group(0),
                            context=text[max(0, m.start() - context):m.end() + context].replace('\\n', ' ')))
    return out


def by_term(hits):
    """'term@turn,turn;...', terms in lower case (the format of recall_terms in benchmarks/evaluation/runs.csv)."""
    turns = {}
    for h in hits:
        turns.setdefault(h['term'].lower(), set()).add(h['turn'])
    return ';'.join(f'{k}@{",".join(map(str, sorted(v)))}' for k, v in sorted(turns.items()))


def first_turn(hits):
    return min((h['turn'] for h in hits), default=None)


# ---------------------------------------------------------------------- recorded data transformations
def transformations(run_dir, source):
    """Every recorded data transformation of a run (results.json) with the turn of its add_data_column call
    (tool_log.jsonl), its source columns, the number of levels of the new column in the data version it created
    and, for two levels and one source column, the range of the source column at each level.

    apgr_at_5: the expression matches the dichotomization of the reference model (APGR < 5 or APGR <= 4).
    apgr_other: any other dichotomization of APGR, a column with two levels that is derived from APGR."""
    history = _json(run_dir / 'results.json').get('data_transformations', [])
    base = set(pd.read_csv(common.BENCHMARKS / 'data' / DATASETS[source]['file'], nrows=0).columns)
    calls = [t for t in common.read_jsonl(run_dir / 'tool_log.jsonl') if t['tool'] == 'add_data_column'
             and isinstance(t.get('result'), dict) and 'data_version' in t['result']]
    if len(calls) != len(history):                      # the log does not match the record: no turns
        calls = [None] * len(history)
    out = []
    for i, (t, call) in enumerate(zip(history, calls)):
        sources = sorted(_sources(t['column'], history[:i + 1], base))
        levels = split = None
        path = run_dir / 'data' / f"data_v{call['result']['data_version']}.csv" if call else None
        if path is not None and path.exists():
            df = pd.read_csv(path, na_values=['.'])
            if t['column'] in df:
                levels = int(df[t['column']].nunique())
                if levels == 2 and len(sources) == 1 and sources[0] in df:
                    g = df.groupby(t['column'])[sources[0]].agg(['min', 'max'])
                    split = '; '.join(f'{k:g}: {sources[0]} {lo:g} to {hi:g}' for k, (lo, hi) in g.iterrows())
        at_5 = bool(APGR_AT_5.search(t['expression']))
        binary = levels == 2 if levels is not None else bool(re.search(r'[<>]|==|!=', t['expression']))
        out.append(dict(turn=call['turn'] if call else None, column=t['column'], expression=t['expression'],
                        reason=t.get('reason'), replaced=t.get('replaced'), sources=';'.join(sources), levels=levels,
                        split=split, apgr_at_5=at_5, apgr_other='APGR' in sources and binary and not at_5))
    return out


# ---------------------------------------------------------------------- CPU load
def read_load(path):
    """load.csv of the runner (columns common.LOAD_COLUMNS, one line a minute) as a Series of the CPU load in
    percent indexed by local time, or None if the file cannot be read so."""
    try:
        df = pd.read_csv(path)
        return pd.Series(df['cpu_percent'].to_numpy(float), index=pd.to_datetime(df['time'])).sort_index()
    except (OSError, ValueError, KeyError):
        return None


def served_models(run_dir):
    """The served model identifiers of a run as '<model> (<provider>)', separated by ';' (common.served_models);
    empty for a run without served.jsonl, as the benchmark runs of October."""
    return ';'.join(f'{model} ({provider})' for model, provider in sorted(common.served_models(run_dir)))


def load_during(load, started, ended, hours):
    """Median CPU load and number of samples between the start and the end of a run (end: recorded, or start plus
    the hours of the run); empty when the table has no sample in that interval."""
    if load is None or not started:
        return None, None
    try:
        start = pd.Timestamp(started)
        end = pd.Timestamp(ended) if ended else start + pd.Timedelta(hours=hours or 0)
    except ValueError:
        return None, None
    x = load[(load.index >= start) & (load.index <= end)]
    return (float(x.median()), int(len(x))) if len(x) else (None, None)


# ---------------------------------------------------------------------- runs
def score_run(run_dir, runs_root, source, load=None):
    """All endpoints of one run: (row, detail, matches, transformations)."""
    row, detail = endpoints(run_dir, source)
    tests = own_tests(run_dir, runs_root, source)
    row.update(weak_tests(tests, source), covariate_tests=len(tests['covariate_tests']),
               compartment_tests=';'.join(f"{t['compartments'][0]}-{t['compartments'][1]}:{t['delta_ofv']:g}"
                                          for t in tests['compartment_tests']))
    found = []
    expressions = dict(recall=RECALL, identification=IDENTIFICATION)
    if source == 'pheno':
        expressions['apgr'] = APGR_AT_5
    for kind, pattern in expressions.items():
        hits = matches(run_dir, pattern)
        found += [dict(kind=kind, **h) for h in hits]
        if kind == 'apgr':
            row['apgr_text_first_turn'] = first_turn(hits)
        else:
            row.update({kind: bool(hits), f'{kind}_first_turn': first_turn(hits), f'{kind}_terms': by_term(hits)})
    made = transformations(run_dir, source)
    if source == 'pheno':
        for key in ('apgr_at_5', 'apgr_other'):
            turns = [t['turn'] for t in made if t[key]]
            row[key] = bool(turns)
            row[f'{key}_turn'] = min((x for x in turns if x is not None), default=None)
        row['apgr_columns'] = '; '.join(f"{t['column']} = {t['expression']} (turn {t['turn']})" for t in made
                                        if t['apgr_at_5'] or t['apgr_other'])
    cond = _json(run_dir / 'condition.json') if (run_dir / 'condition.json').exists() else {}
    started = cond.get('started') or (_json(run_dir / 'run.json').get('started')
                                      if (run_dir / 'run.json').exists() else None)
    row.update(started=started, ended=cond.get('ended'), tools_sha256=cond.get('tools_sha256'),
               data_sha256=cond.get('data_sha256'), served=served_models(run_dir))
    row['cpu_load_median'], row['cpu_load_minutes'] = load_during(load, started, cond.get('ended'), row['hours'])
    return row, dict(detail or {}, own_tests=tests), found, made


def october_runs():
    """The named no-knowledge runs of the benchmark: (source, llm, replicate, run folder)."""
    return [(source, llm, int(p.parent.name[3:]), p.parent) for source in SOURCES for llm in sorted(common.LLMS)
            for p in sorted((OCTOBER / source / 'none' / llm).glob('rep*/results.json'))]


def blind_cells(design, pilot=False):
    """The cells to evaluate: those of B, or the blinded phenobarbital runs of the pilot."""
    if pilot:
        return [c for c in common.cells_pilot(design) if c['arm'] == common.PILOT_BLIND_ARM]
    return common.cells_b(design)


def collect(design, pilot=False, load=None):
    """Rows, details, matches and transformations of the October runs and of every cell of B (or of the pilot). A
    cell without an accepted attempt (section 8.7) gets a row without outcomes."""
    rows, details, found, made = [], {}, [], []

    def add(labels, run_dir, runs_root):
        row, detail, hits, trans = score_run(run_dir, runs_root, labels['dataset'], load)
        rows.append(dict(labels, **row))
        details[labels['run']] = detail
        keys = {k: labels[k] for k in ('set', 'dataset', 'arm', 'llm', 'rep')}
        found.extend(dict(keys, **h) for h in hits)
        made.extend(dict(keys, **t) for t in trans)
    for source, llm, rep, run_dir in october_runs():
        add(dict(set='october', dataset=source, arm=f'{source}_october', group='named_october', llm=llm, rep=rep,
                 run=_rel(run_dir), tools='original', accepted=True, state='complete', attempt=1, n_attempts=1,
                 reason=''), run_dir, OCTOBER)
    for c in blind_cells(design, pilot):
        status = common.cell_status(c)
        last = status['attempts'][-1] if status['attempts'] else {}
        labels = dict(set='pilot' if pilot else 'B', dataset=common.B_ARMS[c['arm']]['source'], arm=c['arm'],
                      group=GROUP_OF[c['arm']], llm=c['llm'], rep=c['rep'], run=_rel(c['run_dir']), tools=c['tools'],
                      accepted=status['state'] == 'complete', state=status['state'], attempt=last.get('attempt'),
                      n_attempts=len(status['attempts']), reason=last.get('reason', 'no attempt'))
        if labels['accepted']:
            add(labels, c['run_dir'], c['run_root'])
        else:
            rows.append(labels)
    return rows, details, found, made


# ---------------------------------------------------------------------- tables
def _show(v, digits=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    if isinstance(v, (bool, np.bool_)):
        return 'yes' if v else 'no'
    if isinstance(v, (float, np.floating)):
        return f'{v:.{digits}g}'
    return str(v)


def _table(header, lines):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    return '\n'.join(out + ['| ' + ' | '.join(str(x).replace('|', '/') for x in line) + ' |' for line in lines])


def _scored(rows):
    """Accepted runs with a final model (the analysis set)."""
    return [r for r in rows if r.get('accepted') and r.get('compartments') is not None]


def identified(rows):
    """Blinded runs in which the drug or its source was identified (a match of the identification expression; the
    named runs are told the drug)."""
    return [r for r in _scored(rows) if r['group'] in ('blinded', 'blinded_example') and r['identification']]


def comparison(rows):
    """For every dataset, LLM and endpoint: the values by group, whether the concurrent named control lies within
    the range of the October runs, and whether the blinded runs lie within the range of the October runs and of all
    named runs. A yes/no endpoint lies within a range when its value occurs among the reference runs."""
    def inside(values, reference):
        if not values or not reference:
            return None
        return all(min(reference) <= v <= max(reference) for v in values)
    out = []
    for source in SOURCES:
        for llm in sorted(common.LLMS):
            runs = [r for r in _scored(rows) if r['dataset'] == source and r['llm'] == llm]
            for key, label in COMPARED:
                if key == 'apgr_at_5' and source != 'pheno':
                    continue
                v = {g: [r[key] for r in runs if r['group'] == g and r.get(key) is not None] for g in GROUPS}
                x = {g: [float(a) for a in v[g]] for g in GROUPS}
                named = x['named_october'] + x['named_concurrent']
                line = dict(dataset=source, llm=llm, endpoint=key, label=label)
                line.update({g: ' '.join(_show(a) for a in v[g]) for g in GROUPS})
                line.update(concurrent_in_october_range=inside(x['named_concurrent'], x['named_october']),
                            blinded_in_october_range=inside(x['blinded'], x['named_october']),
                            blinded_in_named_range=inside(x['blinded'], named),
                            example_in_october_range=inside(x['blinded_example'], x['named_october']),
                            example_in_named_range=inside(x['blinded_example'], named))
                out.append(line)
    return out


def run_tables(rows, compared):
    """Markdown: for every dataset and LLM, the table of its runs and the comparison of the groups."""
    header = ['Runs', 'Rep', 'Compartments', 'Reference structure', 'Strong retained', 'Weak retained', 'Extra',
              'Typical values within 0.80-1.25', 'Largest abs log ratio', 'Reproduced', 'OFV - reference',
              'AIC - reference', 'Own test of weak (final; largest)', 'Recall', 'Identification', 'APGR dichotomized',
              'Fits', 'Responses', 'USD', 'Hours', 'CPU load %']
    out = []
    for source in SOURCES:
        for llm in sorted(common.LLMS):
            lines = []
            for r in sorted((r for r in rows if r['dataset'] == source and r['llm'] == llm),
                            key=lambda r: (list(GROUPS).index(r['group']), r['rep'])):
                head = [GROUPS[r['group']], r['rep']]
                if not r.get('accepted') or r.get('compartments') is None:
                    lines.append(head + [f"no accepted run ({r.get('state')}: {r.get('reason')})"]
                                 + [''] * (len(header) - 3))
                    continue
                if source != 'pheno':
                    apgr = 'no APGR column'
                elif r['apgr_at_5']:
                    apgr = f"at 5, turn {r['apgr_at_5_turn']}"
                else:
                    apgr = f"other, turn {r['apgr_other_turn']}: {r['apgr_columns']}" if r['apgr_other'] else 'no'
                own = f"{_show(r['weak_d_final']) or '-'}; {_show(r['weak_d_max']) or '-'}" if r['weak_tested'] \
                    else 'not tested'
                lines.append(head + [
                    r['compartments'], _show(r['structure_match']), f"{r['strong_retained']}/{r['strong_total']}",
                    _show(r['weak_retained']), r['extra'] or '-', _show(r.get('all_within_20pct')),
                    _show(r.get('max_abs_log_ratio')), _show(r['reproduced']),
                    _show(r.get('delta_ofv_vs_reference'), 4), _show(r.get('delta_aic_vs_reference'), 4), own,
                    r['recall_terms'] or 'no',
                    f"turn {r['identification_first_turn']}: {r['identification_terms']}" if r['identification']
                    else 'no',
                    apgr, r['fits'], r['responses'], _show(r['cost_usd']), _show(r['hours']),
                    _show(r.get('cpu_load_median'))])
            weak = '~'.join(WEAK[source])
            out += [f'## {LABEL[source]}, {common.MODELS[llm]}', '',
                    f"Strong relationships: {', '.join(f'{p}~{c}' for p, c in sorted(STRONG[source]))}; weak: {weak}.",
                    '',
                    _table(header, lines), '']
            sub = [c for c in compared if c['dataset'] == source and c['llm'] == llm]
            keys = ('concurrent_in_october_range', 'blinded_in_october_range', 'blinded_in_named_range',
                    'example_in_october_range', 'example_in_named_range')
            names = ['Endpoint'] + list(GROUPS.values()) + ['Concurrent within October range',
                                                             'Blinded within October range',
                                                             'Blinded within range of all named',
                                                             'Original example within October range',
                                                             'Original example within range of all named']
            out += [_table(names, [[c['label']] + [c[g] or '-' for g in GROUPS] + [_show(c[k]) or '-' for k in keys]
                                   for c in sub]), '']
    return out


def tables(rows):
    """tables.md: the tables of the Analysis paragraph of section 9."""
    out = ['# Experiment B: named and blinded no-knowledge runs', '',
           f"Written by blind/evaluate_blind.py, {time.strftime('%Y-%m-%d %H:%M')}. Named, October: the no-knowledge "
           'runs of the benchmark; the other rows are the arms of B (blinded: masked names; for phenobarbital the tool '
           'example is "CRCL < 60" in the blinded arm and "APGR < 5" in the arm with the original example). Recall and '
           'identification list the matched terms with their turns. A blinded run lies within a range when its value '
           'is between the smallest and the largest value of the named runs. Remifentanil OFV differences below 0.5 '
           'are not interpreted.', '']
    out += run_tables(rows, comparison(rows))
    found = identified(rows)
    out += ['## Blinded runs in which the drug or its source was identified', '']
    out += [f"- {r['run']}: first turn {r['identification_first_turn']}, {r['identification_terms']}" for r in found] \
        or ['None.']
    if found:
        keep = [r for r in rows if r not in found]
        out += ['', '# The same tables without these runs', ''] + run_tables(keep, comparison(keep))
    return '\n'.join(out) + '\n'


def evaluate(design, out, pilot=False, partial=False, load_path=None, report=print, reduction=0, n1_waived=False):
    """Evaluate the runs and write the output files; returns the exit code. B is evaluated when all grids are closed
    (section 10.2: its own cells and, for a design that is not a test design, those of A and of the N1 extension;
    reduction and n1_waived as in common.open_cells); the blinded runs of the pilot when they are closed."""
    cells = blind_cells(design, pilot)
    states = {c['key']: common.cell_status(c)['state'] for c in cells}
    still_open = [k for k, s in states.items() if s == 'open']
    if still_open and not partial:
        report(f'{len(still_open)} of {len(cells)} cells are not closed (first: {still_open[0]}); nothing was '
               'evaluated. Use --partial for tests.')
        return 2
    elsewhere = [] if pilot or partial or design.get('mock') else common.open_cells(design, reduction, n1_waived)
    if elsewhere:
        report(common.closure_message(elsewhere) + ' Nothing was evaluated.')
        return 2
    path = Path(load_path) if load_path else common.load_file(design, 'pilot' if pilot else 'B')
    path = path if path.exists() else None
    load = read_load(path) if path else None
    if path and load is None:
        report(f'{path} could not be read as a load table; the CPU load is left empty')
    rows, details, found, made = collect(design, pilot, load)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=RUN_COLUMNS)
    frame[list(INTEGERS)] = frame[list(INTEGERS)].astype('Int64')
    frame.to_csv(out / 'runs.csv', index=False)
    for h in found:
        h['context'] = ' '.join(h['context'].split())
    pd.DataFrame(found, columns=['set', 'dataset', 'arm', 'llm', 'rep', 'kind', 'turn', 'part', 'term', 'context']
                 ).to_csv(out / 'matches.csv', index=False)
    frame = pd.DataFrame(made, columns=['set', 'dataset', 'arm', 'llm', 'rep', 'turn', 'column', 'expression', 'reason',
                                        'replaced', 'sources', 'levels', 'split', 'apgr_at_5', 'apgr_other'])
    frame[['turn', 'levels']] = frame[['turn', 'levels']].astype('Int64')
    frame.to_csv(out / 'transformations.csv', index=False)
    compared = comparison(rows)
    pd.DataFrame(compared).to_csv(out / 'comparison.csv', index=False)
    (out / 'tables.md').write_text(tables(rows), encoding='utf-8')
    (out / 'evaluation.json').write_text(json.dumps(dict(
        protocol_version=common.PROTOCOL_VERSION, written=time.strftime('%Y-%m-%d %H:%M:%S'),
        cells='pilot' if pilot else 'B', partial=bool(still_open), cell_states=states,
        data=common.blind_copies(design), benchmark_data=design.get('benchmark_data'),
        load_file=str(path) if path else None, machine_load=common.load_summary(path) if path else None,
        expressions=dict(recall=RECALL.pattern, identification=IDENTIFICATION.pattern, apgr=APGR_AT_5.pattern),
        reference_fits={s: {k: (reference_fit(s) or {}).get(k) for k in ('ofv', 'aic')} for s in SOURCES},
        runs=rows, identified=[r['run'] for r in identified(rows)], details=details), indent=1, default=str),
        encoding='utf-8')
    scored = _scored(rows)
    report(f"{len(rows)} rows: {sum(r['set'] == 'october' for r in rows)} benchmark runs of October, "
           f"{sum(r['set'] != 'october' and r['accepted'] for r in rows)} accepted runs of "
           f"{len(cells)} cells of {'the pilot' if pilot else 'B'} ({len(still_open)} open); "
           f'{len(scored)} with a final model')
    for kind in ('recall', 'identification', 'apgr'):
        hits = [h for h in found if h['kind'] == kind]
        report(f"{kind}: {len(hits)} matches in {len({(h['set'], h['arm'], h['llm'], h['rep']) for h in hits})} runs")
    report(f'written to {_rel(out)}: runs.csv, matches.csv, transformations.csv, comparison.csv, tables.md, '
           'evaluation.json')
    return 0


# ---------------------------------------------------------------------- recognition probe
def parse_answer(text):
    """The JSON object of a probe answer, or None: the text itself, or its part from the first '{' to the last '}'
    (an answer inside a code fence or a sentence)."""
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    for candidate in (text, text[text.find('{'):text.rfind('}') + 1]):
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _confidence(v):
    """A confidence as a number (0 to 100), or None."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r'-?\d+(?:\.\d+)?', str(v))
    return float(m.group(0)) if m else None


def _terms(pattern, text):
    return ';'.join(sorted({m.group(0).lower() for m in pattern.finditer(text or '')}))


def score_probe(records):
    """Score the calls of the recognition probe, one dict per line of probe.jsonl (common.probe_file) with the LLM
    (llm: gpt or claude), the input (input: pheno_named, pheno_blind, remifentanil_named, remifentanil_blind or
    simulated), the number of the call (call) and the answer text (content; absent for a failed call).

    drug_correct: the drug field names the drug of the input. source_correct: the source field names the authors
    of the input's dataset or a package of the source expression. Both are empty for the simulated study, which
    has no drug and no source; drug_terms and source_terms hold every match of the two expressions. An answer
    that is not a JSON object is scored on its whole text (parsed false)."""
    out = []
    for r in records:
        name, answer = r.get('input'), r.get('content')
        if name is None:
            continue
        obj = parse_answer(answer)
        if obj is not None:
            drug, source = obj.get('drug'), obj.get('source')
        else:
            drug = source = answer if isinstance(answer, str) else None
        drug, source = (None if x is None else str(x) for x in (drug, source))
        dataset = next((s for s in SOURCES if str(name).startswith(s)), None)
        row = dict(llm=r.get('llm'), input=name, call=r.get('call'),
                   answered=answer is not None, parsed=obj is not None, drug=drug if obj is not None else None,
                   drug_confidence=_confidence((obj or {}).get('drug_confidence')),
                   source=source if obj is not None else None,
                   source_confidence=_confidence((obj or {}).get('source_confidence')),
                   drug_terms=_terms(PROBE_DRUG, drug), source_terms=_terms(PROBE_SOURCE, source),
                   drug_correct=None, source_correct=None, reason=(obj or {}).get('reason'))
        if dataset and answer is not None:
            t = PROBE_TERMS[dataset]
            row['drug_correct'] = bool(re.search(t['drug'], drug or '', re.IGNORECASE))
            row['source_correct'] = bool(re.search(f"{t['authors']}|{PROBE_PACKAGES}", source or '', re.IGNORECASE))
        out.append(row)
    return out


def probe_summary(rows):
    """Per LLM and input: calls, answers that were JSON objects, drug and source correct, answers that named a
    drug or a source (parsed answers), and median, smallest and largest confidence."""
    out = []
    for (llm, name), g in pd.DataFrame(rows).groupby(['llm', 'input'], sort=True):
        line = dict(llm=llm, input=name, calls=len(g), answered=int(g['answered'].sum()), parsed=int(g['parsed'].sum()))
        for key in ('drug_correct', 'source_correct'):
            x = [v for v in g[key] if v is not None and not (isinstance(v, float) and math.isnan(v))]
            line[key] = int(sum(bool(v) for v in x)) if x else None
        line.update(drug_named=int(g['drug'].notna().sum()), source_named=int(g['source'].notna().sum()))
        for key in ('drug_confidence', 'source_confidence'):
            x = g[key].dropna().astype(float)
            line.update({f'{key}_median': float(x.median()) if len(x) else None,
                         f'{key}_min': float(x.min()) if len(x) else None,
                         f'{key}_max': float(x.max()) if len(x) else None})
        out.append(line)
    return out


def probe(design, out, given=None, report=print):
    """Score probe.jsonl and write probe_calls.csv and probe_summary.csv; returns the exit code. Of the lines of a
    call (LLM, input, number), the answered one is scored; a call that only failed is listed without an answer."""
    path = Path(given) if given else common.probe_file(design)
    if not path.exists():
        report(f'{path} does not exist: run_grid.py --module probe writes it (or name a file with --probe-file)')
        return 2
    calls = {}
    for r in common.read_jsonl(path):
        key = (r.get('llm'), r.get('input'), r.get('call'))
        if key not in calls or not r.get('error'):
            calls[key] = r
    rows = score_probe(calls.values())
    known = {i['name'] for i in common.probe_inputs(design)}
    for name in sorted({r['input'] for r in rows} - known):
        report(f'unknown input {name}')
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'probe_calls.csv', index=False)
    summary = pd.DataFrame(probe_summary(rows))
    summary[['drug_correct', 'source_correct']] = summary[['drug_correct', 'source_correct']].astype('Int64')
    summary.to_csv(out / 'probe_summary.csv', index=False)
    report(summary.to_string(index=False))
    expected = len(known) * common.PROBE_CALLS * len(common.LLMS)
    report(f'{len(rows)} calls scored ({expected} planned) from {path}; written to {_rel(out)}: probe_calls.csv, '
           'probe_summary.csv')
    return 0


# ---------------------------------------------------------------------- tests (section 7.3)
REQUIRED = ('covariates_found', 'structure_match', 'all_within_20pct', 'reproduced')     # and the ratio medians
ALSO = ('compartments', 'elimination', 'covariates_exact', 'form_agreement', 'max_abs_log_ratio', 'parameters_missing',
        'ofv', 'aic', 'delta_ofv_vs_reference', 'delta_aic_vs_reference', 'recall_terms', 'fits', 'llm_calls',
        'cost_usd', 'hours')
PROBE_TEST = (          # (input, answer, expected parsed, drug correct, source correct, drug and source confidence)
    ('pheno_named', '{"drug": "phenobarbital", "drug_confidence": 95, "source": "Grasela and Donn 1985, a NONMEM '
     'example", "source_confidence": 90, "reason": "neonates"}', True, True, True, 95., 90.),
    ('pheno_blind', '```json\n{"drug": "Phenobarbital", "drug_confidence": 70, "source": null, '
     '"source_confidence": 0, "reason": "x"}\n```', True, True, False, 70., 0.),
    ('pheno_blind', '{"drug": "gentamicin", "drug_confidence": 40, "source": "Minto et al. 1997", '
     '"source_confidence": 20, "reason": "x"}', True, False, False, 40., 20.),
    ('remifentanil_named', '{"drug": "remifentanil", "drug_confidence": 99, "source": "Minto 1997 '
     '(nlme::Remifentanil)", "source_confidence": 80, "reason": "x"}', True, True, True, 99., 80.),
    ('remifentanil_blind', 'Here is my answer: {"drug": "an opioid such as fentanyl", "drug_confidence": "60", '
     '"source": "the nlme package", "source_confidence": 30.5, "reason": "x"}', True, False, True, 60., 30.5),
    ('remifentanil_blind', '{"drug": "phenobarbital", "drug_confidence": 10, "source": "Grasela and Donn", '
     '"source_confidence": 10, "reason": "x"}', True, False, False, 10., 10.),
    ('simulated', '{"drug": null, "drug_confidence": 0, "source": null, "source_confidence": 0, "reason": "x"}',
     True, None, None, 0., 0.),
    ('simulated', '{"drug": "phenobarbital", "drug_confidence": 30, "source": "Madonna 2001", '
     '"source_confidence": 5, "reason": "x"}', True, None, None, 30., 5.),
    ('pheno_named', 'This is the phenobarbital dataset of Grasela and Donn.', False, True, True, None, None),
    ('pheno_named', None, False, None, None, None, None))


def _same(a, b):
    """Equality of a value of benchmarks/evaluation/runs.csv with a value computed here (empty = missing)."""
    def missing(x):
        return x is None or (isinstance(x, str) and x == '') or (isinstance(x, (float, np.floating)) and math.isnan(x))
    if missing(a) or missing(b):
        return missing(a) and missing(b)
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)) or isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)


def check(design, out, report=print):
    """The tests of this script; returns the number of failures.

    1. Section 7.3: the endpoints of the 12 named no-knowledge benchmark runs equal benchmarks/evaluation/runs.csv
       (covariates_found, structure_match, all_within_20pct, reproduced, the ratio medians, and further columns).
    2. The recorded transformations of the six phenobarbital runs give the statement of section 9: APGR < 5 was
       created in 3 of 3 runs of Claude and 1 of 3 of GPT.
    3. Recall and identification matches in the 12 runs; the recall matches equal those of recall.recalled.
    4. The blinded data copies, if prepared, have the sha256 of the originals.
    5. Synthetic probe answers are scored as expected.
    """
    failures = 0

    def test(ok, text):
        nonlocal failures
        failures += not ok
        report(('ok      ' if ok else 'FAILED  ') + text)
    test(set(GROUP_OF) == set(common.B_ARMS), 'every arm of B has a row group')
    test({s: len(STRONG[s]) for s in SOURCES} == {'pheno': 2, 'remifentanil': 7} and
         all(WEAK[s] in REFERENCE_COVARIATES[s] for s in SOURCES), 'strong relationships: 2 and 7; one weak each')
    test('|'.join(PROBE_TERMS[s]['drug'] for s in SOURCES) == PROBE_DRUG.pattern and
         '|'.join([PROBE_TERMS['pheno']['authors'], PROBE_TERMS['remifentanil']['authors'], PROBE_PACKAGES])
         == PROBE_SOURCE.pattern, 'the terms by dataset are those of the two probe expressions')

    report('\n1. benchmark runs against benchmarks/evaluation/runs.csv')
    bench = pd.read_csv(common.BENCHMARKS / 'evaluation' / 'runs.csv')
    runs = october_runs()
    test(len(runs) == 12, f'{len(runs)} named no-knowledge runs found (12 expected)')
    scored, lines = [], []
    for source, llm, rep, run_dir in runs:
        row, _, found, made = score_run(run_dir, OCTOBER, source)
        scored.append((source, llm, rep, run_dir, row, found, made))
        b = bench[(bench.dataset == source) & (bench.condition == 'none') & (bench.llm == llm)
                  & (bench.rep == f'rep{rep}')]
        if len(b) != 1:
            test(False, f'{source}/none/{llm}/rep{rep}: {len(b)} rows in runs.csv')
            continue
        b = b.iloc[0]
        ratio = [c for c in bench.columns if c.startswith('ratio_')
                 and not (_same(b[c], None) and _same(row.get(c), None))]
        medians = [c for c in ratio if c.endswith('_median')]
        differ = {}
        for c in list(REQUIRED) + medians + list(ALSO) + [c for c in ratio if c.endswith('_range')]:
            mine = row.get('responses' if c == 'llm_calls' else c)
            same = _same(b[c], mine)
            lines.append(dict(run=f'{source}/none/{llm}/rep{rep}', column=c, required=c in REQUIRED or c in medians,
                              benchmark=b[c], here=mine, agree=same))
            if not same:
                differ[c] = (b[c], mine)
        test(not differ, f"{source}/none/{llm}/rep{rep}: covariates_found {row['covariates_found']}; structure_match "
             f"{row['structure_match']}; all_within_20pct {row['all_within_20pct']}; reproduced {row['reproduced']}; "
             f"{len(medians)} ratio medians ({', '.join(f'{c[6:-7]} {row.get(c, float('nan')):.4f}' for c in medians)})"
             + (f'; DIFFERENT: {differ}' if differ else '; all compared columns agree'))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(lines).to_csv(out / 'benchmark_check.csv', index=False)
    report(f"{sum(x['agree'] for x in lines)} of {len(lines)} compared values agree "
           f"({sum(x['agree'] for x in lines if x['required'])} of {sum(x['required'] for x in lines)} of the columns "
           f'named in section 7.3); written to {_rel(out / "benchmark_check.csv")}')

    report('\n2. dichotomization of APGR in the six phenobarbital runs (protocol: Claude 3 of 3, GPT 1 of 3)')
    counts = {}
    for source, llm, rep, run_dir, row, found, made in scored:
        if source != 'pheno':
            continue
        counts[llm] = counts.get(llm, 0) + bool(row['apgr_at_5'])
        report(f"        pheno/none/{llm}/rep{rep}: at 5 {row['apgr_at_5']}, turn {row['apgr_at_5_turn']}; other "
               f"{row['apgr_other']}; expression in the transcript from turn {row['apgr_text_first_turn']}; "
               f"{row['apgr_columns'] or 'no column'}")
    test(counts == {'claude': 3, 'gpt': 1}, f'runs that created the column: {counts}')

    report('\n3. recall and identification in the 12 runs')
    for source, llm, rep, run_dir, row, found, made in scored:
        mine = [(h['turn'], h['term'], h['context']) for h in found if h['kind'] == 'recall']
        theirs = [(h['turn'], h['term'], h['context']) for h in recalled(run_dir)]
        n = {k: sum(h['kind'] == k for h in found) for k in ('recall', 'identification')}
        test(mine == theirs, f"{source}/none/{llm}/rep{rep}: recall {n['recall']} matches ({row['recall_terms'] or '-'}"
             f"); identification {n['identification']} matches, first turn {row['identification_first_turn']} "
             f"({row['identification_terms'] or '-'})")
    for kind in ('recall', 'identification'):
        by = {}
        for source, llm, rep, run_dir, row, found, made in scored:
            by.setdefault((source, llm), []).append(bool(row[kind]))
        report(f'        runs with {kind}: ' + ', '.join(f'{s} {m} {sum(v)} of {len(v)}' for (s, m), v in by.items()))

    report('\n4. blinded data copies')
    for name, r in common.blind_copies(design).items():
        if r['sha256_copy'] is None:
            report(f"        {name}: {r['copy']} does not exist (run --prepare)")
        else:
            test(r['identical'], f"{name}: sha256 {r['sha256_copy']} = {r['source']}")

    report('\n5. scoring of synthetic probe answers')
    records = [dict(llm='gpt' if i % 2 else 'claude', input=name, call=i, content=answer)
               for i, (name, answer, *_) in enumerate(PROBE_TEST)]
    rows = score_probe([json.loads(json.dumps(r)) for r in records])
    for (name, answer, *expected), r in zip(PROBE_TEST, rows):
        got = [r['parsed'], r['drug_correct'], r['source_correct'], r['drug_confidence'], r['source_confidence']]
        test(got == expected, f'{name}: {str(answer)[:60]!r} -> parsed, drug correct, source correct, confidences {got}'
             + ('' if got == expected else f', expected {expected}'))
    by = {(s['llm'], s['input']): s for s in probe_summary(rows)}
    test(by[('claude', 'pheno_named')]['calls'] == 2 and by[('claude', 'pheno_named')]['drug_correct'] == 2
         and by[('claude', 'simulated')]['drug_correct'] is None and by[('claude', 'simulated')]['drug_named'] == 0
         and by[('gpt', 'simulated')]['drug_named'] == 1 and rows[7]['source_terms'] == ''
         and rows[2]['source_terms'] == 'minto', 'probe summary by LLM and input; "Madonna" is no match of donn\\b')
    report('\ncheck passed' if not failures else f'\ncheck FAILED ({failures})')
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(description='Evaluation of experiment B (PROTOCOL.md, section 9).')
    ap.add_argument('--prepare', action='store_true', help='write the blinded data copies and record their sha256')
    ap.add_argument('--check', action='store_true', help='tests on the 12 named benchmark runs and of the scoring')
    ap.add_argument('--probe', action='store_true', help='score the recognition probe')
    ap.add_argument('--pilot', action='store_true', help='evaluate the blinded runs of the pilot')
    ap.add_argument('--partial', action='store_true', help='tests only: evaluate although cells are still open')
    ap.add_argument('--reduction', type=int, default=0, choices=(0, 1, 2),
                    help='the reduction of section 10.3 that was applied to A')
    ap.add_argument('--n1-waived', action='store_true', help='the N1 extension is not run (section 10.3)')
    ap.add_argument('--design', help='design file (default: design.json, or the slot table if it does not exist)')
    ap.add_argument('--load', help='load.csv of the runner')
    ap.add_argument('--probe-file', help='probe.jsonl of the runner')
    a = ap.parse_args(argv)
    design = load_design(a.design)
    f = common.folders(design)
    out = f['pilot'] / 'blind_evaluation' if a.pilot else f['blind'] / 'evaluation'
    if a.prepare:
        prepare(design)
        return 0
    if a.check:
        return 1 if check(design, out) else 0
    if a.probe:
        return probe(design, out, a.probe_file)
    return evaluate(design, out, pilot=a.pilot, partial=a.partial, load_path=a.load, reduction=a.reduction,
                    n1_waived=a.n1_waived)


if __name__ == '__main__':
    sys.exit(main())
