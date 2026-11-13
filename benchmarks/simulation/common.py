"""Shared constants and helpers of the simulation experiment (PROTOCOL.md).

Paths and file names, LLMs and run settings, seeds, limits of the runner, the slot table with the dataset IDs,
loaders for strings.json and design.json, the cells of every module with their run order, the prompt and tool
variants, the data files of experiment B, the fields of condition.json and of evaluation/runs.csv, the acceptance
rule for attempts, the fees spent and the rule that scoring waits until all grids are closed. generate.py,
calibrate.py, run_grid.py, score.py, code_reports.py, analyze.py, sim_figures.py and blind/evaluate_blind.py import
this module; importing it reads no file and changes nothing.

Usage:
  python benchmarks/simulation/common.py --check [--design smoke/design_mock.json]
verifies every hash of strings.json against pkagent and benchmarks/datasets.json, checks the slot table and the
design, and prints the cell counts. Without --design it uses design.json, or the slot table when design.json does
not exist yet.
"""
import copy
import functools
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent          # benchmarks/simulation
REPO = ROOT.parents[1]
BENCHMARKS = ROOT.parent
DATA = ROOT / 'data'                            # data/<ID>/pk_data.csv (tracked)
RUNS = ROOT / 'runs'                            # runs/<ID>/<statement>-<instruction>/<llm>/rep1/
RUNS_FAILED = ROOT / 'runs_failed'              # attempts that were not accepted
EVALUATION = ROOT / 'evaluation'
CALIBRATION = ROOT / 'calibration'
PILOT = ROOT / 'pilot'
BLIND = ROOT / 'blind'                          # blind/data, blind/runs/<arm>/none/<llm>/rep<k>/, blind/evaluation
ENGINE = ROOT / '_engine'                       # frozen copy of PKPy2
SMOKE = ROOT / 'smoke'                          # throwaway output of tests
STRINGS = ROOT / 'strings.json'
DESIGN = ROOT / 'design.json'
PROTOCOL = ROOT / 'PROTOCOL.md'
DATA_FILE = 'pk_data.csv'                       # the same basename for every dataset (describe_data shows it)
# files of the runner, in the run root of a grid (runs/, pilot/runs/ or blind/runs/; see run_root)
LOAD_FILE = 'load.csv'                          # CPU load once a minute while a grid runs
LOAD_COLUMNS = ('time', 'module', 'cpu_percent', 'memory_percent', 'runs', 'processes', 'fees_usd')
GRID_LOG = 'grid_log.jsonl'                     # one line per event of the runner
PROBE_FILE = 'probe.jsonl'                      # one line per call of the recognition probe, in blind/runs/
LOCK_FILE = 'run_grid.lock'                     # in runs_failed/ while a runner is active
# files of the coded misreport endpoint (section 7.2, S5): the key in evaluation/, the others in coding/
MISREPORT_KEY, MISREPORT_EXCERPTS = 'misreport_key.csv', 'misreport_excerpts.csv'
MISREPORT_CODES, MISREPORT_RECODE = 'misreport_codes.csv', 'misreport_recode.csv'

PROTOCOL_VERSION = '1.1'
SRC_TREE = '7993100e9aea71df5e29aef3d2372d224ae3510c'     # git rev-parse HEAD:src, the tree of commit f5a4263


def add_paths():
    """Put REPO/src and REPO/benchmarks at the front of sys.path (pkagent is not installed as a package)."""
    for p in (BENCHMARKS, REPO / 'src'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def folders(design=None):
    """The folders of a design: dict with root, data, runs, runs_failed, evaluation, coding, calibration, pilot and
    blind.

    For design.json (root '.') these are the module constants. A test design names another root relative to this
    folder (the mock design: 'smoke'), and all its data and outputs then stay below that root.
    """
    root = (ROOT / (design or {}).get('root', '.')).resolve()
    require(root == ROOT or ROOT in root.parents, f'design root {root} is outside {ROOT}')
    return dict(root=root, data=root / 'data', runs=root / 'runs', runs_failed=root / 'runs_failed',
                evaluation=root / 'evaluation', coding=root / 'coding', calibration=root / 'calibration',
                pilot=root / 'pilot', blind=root / 'blind')


def run_root(design, module):
    """Root of the run folders of a grid, where the runner also writes load.csv and grid_log.jsonl: runs/ for A (A1,
    A2) and N1, pilot/runs/ for the pilot, blind/runs/ for B and the probe."""
    f = folders(design)
    return f['pilot'] / 'runs' if module == 'pilot' else f['blind'] / 'runs' if module in ('B', 'probe') else f['runs']


def load_file(design, module):
    """load.csv of a grid (columns LOAD_COLUMNS, one line a minute)."""
    return run_root(design, module) / LOAD_FILE


def load_summary(path):
    """The CPU load of the machine while a grid ran (section 6: the median is reported for A and for B), from the
    lines of a load.csv with at least one run in progress: dict(file, minutes, cpu_percent_median,
    memory_percent_median), or None when the file does not exist or has no such line."""
    import pandas as pd
    path = Path(path)
    if not path.exists():
        return None
    table = pd.read_csv(path)
    require(list(table.columns) == list(LOAD_COLUMNS), f'{path} does not have the columns {LOAD_COLUMNS}')
    table = table[table['runs'] > 0]
    if table.empty:
        return None
    return dict(file=str(path), minutes=int(len(table)), cpu_percent_median=float(table['cpu_percent'].median()),
                memory_percent_median=float(table['memory_percent'].median()))


def probe_file(design):
    """probe.jsonl of the recognition probe: one line per call with time, llm, model, input, call, message_sha256,
    content, finish_reason, reasoning, served, unix_time and usage {prompt_tokens, completion_tokens, cost_usd}; a
    failed call has error and http_status in place of the answer."""
    return run_root(design, 'probe') / PROBE_FILE


def lock_file(design):
    """The lock of the runner: one runner at a time per design, whatever the module."""
    return folders(design)['runs_failed'] / LOCK_FILE


def design_path(given=None):
    """The design file named on a command line: absolute, relative to the working directory, or relative to
    benchmarks/simulation; design.json when no file is named."""
    if not given:
        return DESIGN
    path = Path(given)
    return path if path.is_absolute() or path.exists() else ROOT / path


def require(condition, message):
    """Raise AssertionError(message) unless the condition holds (also under python -O)."""
    if not condition:
        raise AssertionError(message)


def sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def read_json(path):
    """The content of a JSON file, or None when it does not exist or cannot be read."""
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def read_jsonl(path):
    """The records of a JSON-lines file (empty list when it does not exist). A line that is not complete JSON, as
    the last line of a file that is being written can be, is left out."""
    out = []
    try:
        lines = Path(path).read_text(encoding='utf-8').splitlines()
    except OSError:
        return out
    for line in lines:
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def write_json(path, value, default=None):
    """Write a JSON file through a temporary file, so that neither a reader nor a stopped process meets a partial
    file. default: function for values that JSON cannot hold (str for records that hold paths)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8', newline='\n') as out:
        out.write(json.dumps(value, indent=1, ensure_ascii=False, default=default) + '\n')
    os.replace(tmp, path)


def append_json(path, value):
    """Append one record to a JSON-lines file."""
    with open(path, 'a', encoding='utf-8', newline='\n') as out:
        out.write(json.dumps(value, default=str) + '\n')


# ---------------------------------------------------------------------- LLMs and run settings (sections 3.4, 6)
LLMS = ('gpt', 'claude')                        # keys of benchmarks/run_benchmark.py and pkagent.config.MODELS
MODELS = {'gpt': 'openai/gpt-6.1-sol', 'claude': 'anthropic/claude-opus-5.5'}
REASONING_EFFORT = 'medium'
WORKERS, THREADS_PER_WORKER = 3, 2
ENGINE_SEED = 20261001                          # Settings.seed = ENGINE_SEED + rep
BUDGET = dict(max_turns=80, max_fits=40, max_hours=6.0)
MAX_COST_USD = dict(A=3.0, B=6.0)               # fee limit per run: A (A1, A2, pilot, N1) and B
INCLUDE, RETAIN = 3.84, 6.63                    # OFV thresholds of the system prompt (p < 0.05, p < 0.01)

SEED_GENERATOR = 20261112                       # covariates: default_rng([SEED_GENERATOR, s]); simulation: SEED_GENERATOR + s
SEED_IDS = 20261113                             # dataset IDs
SEED_ORDER_A = 20261114                         # run order of block r: SEED_ORDER_A + r
SEED_BOOTSTRAP = 20261116                       # bootstrap of the curve model (section 8.4)
SEED_ORDER_B = 20261117
SEED_EXCERPTS = 20261118                        # codes and order of the misreport excerpts (section 7.2, S5)
SEED_ORDER_N1 = 20261119

# limits of the runner (sections 6, 8.7 and 10.3)
FEE_LIMIT = {'pilot': 83., 'A': 83., 'B': 95., 'probe': 95., 'N1': 95.}   # no new run at this amount or more (USD)
FEE_RESERVE = 12.                               # reserve for B and the probe: 95 - 83
N1_START_LIMIT = 88.                            # N1 is started only if the fees spent are at most this
WATCHDOG_SECONDS = 6.75 * 3600                  # a child process is terminated after this time
STALL_SECONDS = {'A': 60 * 60, 'B': 150 * 60}   # ... or when its logs got no line for this time (pilot, N1: as A)
FATAL_HTTP = (401, 402, 403, 404)               # statuses of the LLM client that stop the launches at once
MAX_INFRASTRUCTURE = 2                          # consecutive infrastructure failures that stop the launches
MAX_CONSECUTIVE = 3                             # consecutive failures of any class that pause the grid


def budget_kwargs(cell):
    """Keyword arguments of pkagent.config.Budget for a cell (section 6)."""
    return dict(BUDGET, max_cost_usd=MAX_COST_USD['B' if cell['module'] == 'B' else 'A'])


def settings_kwargs(cell):
    """Keyword arguments of pkagent.config.Settings for a cell, without the budget (section 6); the other fields
    keep their defaults. Use Settings(budget=Budget(**budget_kwargs(cell)), **settings_kwargs(cell))."""
    return dict(model=MODELS[cell['llm']], workers=WORKERS, threads_per_worker=THREADS_PER_WORKER,
                reasoning_effort=REASONING_EFFORT, seed=ENGINE_SEED + cell['rep'])


# ---------------------------------------------------------------------- slots and dataset IDs (sections 2.6, 3.1, 3.5)
SLOT_TYPES = ('E0+', 'E0-', 'W-low', 'W-mid', 'S-a', 'S-b', 'S-c')      # column order of the table of section 3.5
STRATUM = {'E0+': 'E0', 'E0-': 'E0', 'W-low': 'W-low', 'W-mid': 'W-mid', 'S-a': 'S', 'S-b': 'S', 'S-c': 'S'}
ZONE = {'E0': 'weak', 'W-low': 'weak', 'W-mid': 'weak', 'S': 'strong'}
BLOCKS = (1, 2, 3, 4)
# (evidence target, number of the target comedication) by block (rows) and slot type (columns); None: beta = 0
SLOT_TABLE = (
    ((None, 1), (None, 2), (2.0, 1), (5.4, 2), (8.5, 1), (17.0, 2), (24.0, 1)),
    ((None, 2), (None, 1), (3.2, 2), (4.4, 1), (11.0, 2), (12.5, 1), (34.0, 2)),
    ((None, 1), (None, 2), (1.5, 2), (5.9, 1), (7.5, 1), (20.0, 2), (29.0, 2)),
    ((None, 2), (None, 1), (2.6, 1), (4.9, 2), (9.5, 2), (14.5, 1), (40.0, 1)),
)
E0_DOFV_MAX = 0.5                               # E0: dOFV_ref at most 0.5 and z_ref in the range of the sign stratum
E0_Z_RANGE = {'E0+': (0.25, 0.71), 'E0-': (-0.71, -0.25)}
COLUMNS = ('COMED1', 'COMED2')


def tolerance(target):
    """Largest accepted |dOFV_ref - target| of a W or S slot (section 3.1)."""
    return max(0.25, 0.05 * target)


def default_a2(block, slot_type):
    """Default A2 role (section 3.7): in every block the S-c dataset and one E0 dataset (E0+ in blocks 1 and 3,
    E0- in blocks 2 and 4). design.json carries the final flag after calibration."""
    return slot_type == 'S-c' or slot_type == ('E0+' if block in (1, 3) else 'E0-')


def dataset_ids():
    """IDs of the 28 slots in sorted order (block, position in the table of section 3.5): with
    perm = numpy.random.default_rng(20261113).permutation(28), the slot at sorted position i is 'D%02d' % (perm[i] + 1)."""
    perm = np.random.default_rng(SEED_IDS).permutation(28)
    return ['D%02d' % (perm[i] + 1) for i in range(28)]


def slots():
    """The 28 slots sorted by (block, position in the table of section 3.5).

    Each slot is a dict: index (0 to 27), id, block (1 to 4), position (0 to 6), slot_type (E0+, E0-, W-low, W-mid,
    S-a, S-b, S-c), stratum (E0, W-low, W-mid, S), zone (weak, strong), evidence_target (None for E0), tolerance
    (max(0.25, 0.05 x target); None for E0), z_range (E0 only: allowed range of z_ref, with dOFV_ref <= 0.5),
    target_column, twin_column and a2_default.
    """
    ids, out = dataset_ids(), []
    for b, row in zip(BLOCKS, SLOT_TABLE):
        for p, (slot_type, (target, k)) in enumerate(zip(SLOT_TYPES, row)):
            stratum = STRATUM[slot_type]
            out.append(dict(index=len(out), id=ids[len(out)], block=b, position=p, slot_type=slot_type,
                            stratum=stratum, zone=ZONE[stratum], evidence_target=target,
                            tolerance=None if target is None else tolerance(target),
                            z_range=E0_Z_RANGE.get(slot_type), target_column=f'COMED{k}',
                            twin_column=f'COMED{3 - k}', a2_default=default_a2(b, slot_type)))
    return out


def slot_of(dataset_id):
    """The slot of a dataset ID."""
    return next(s for s in slots() if s['id'] == dataset_id)


def signed_root(dofv, estimate):
    """z = sign(-estimate) x sqrt(dOFV): positive when the estimated coefficient points to lower clearance with
    the comedication (section 5)."""
    return float(np.sign(-estimate) * np.sqrt(max(float(dofv), 0.)))


def in_tolerance(slot, dofv_ref, z_ref):
    """Whether the evidence of a dataset meets the rule of its slot (sections 3.1 and 5): E0, dOFV_ref <= 0.5 and
    z_ref in the range of the sign stratum; W and S, |dOFV_ref - target| <= max(0.25, 0.05 x target)."""
    if slot['evidence_target'] is None:
        lo, hi = slot['z_range']
        return bool(dofv_ref <= E0_DOFV_MAX and lo <= z_ref <= hi)
    return bool(abs(dofv_ref - slot['evidence_target']) <= slot['tolerance'])


# ---------------------------------------------------------------------- conditions (sections 3.2, 3.3, 3.6, 3.7, 3.9, 9)
STATEMENTS = ('none', 'assert', 'deny', 'twocmt')
INSTRUCTIONS = ('respect', 'neutral', 'criteria')
_WEAK = (('none', 'respect'), ('assert', 'respect'), ('assert', 'neutral'), ('assert', 'criteria'))
_NEAR = (('none', 'respect'), ('deny', 'respect'), ('deny', 'neutral'))
A1_ARMS = {'E0+': _WEAK, 'E0-': _WEAK, 'W-low': _WEAK, 'W-mid': _WEAK, 'S-a': _NEAR, 'S-b': _NEAR,
           'S-c': (('none', 'respect'), ('deny', 'respect'))}       # (statement, instruction) by slot type
A2_ARMS = (('twocmt', 'respect'), ('twocmt', 'criteria'))
N1_ARM, N1_STRATA = ('none', 'neutral'), ('W-low', 'W-mid')
# pilot datasets (never part of the analysis): default generator seed, evidence target, target column, arms
PILOT_DATASETS = (dict(id='P1', seed=9001, evidence_target=10.0, target_column='COMED1', twin_column='COMED2',
                       arms=(('none', 'respect'), ('deny', 'neutral'))),
                  dict(id='P2', seed=9002, evidence_target=2.5, target_column='COMED2', twin_column='COMED1',
                       arms=(('assert', 'criteria'), ('twocmt', 'respect'))))
PILOT_BLIND_ARM = 'pheno_blind'                 # one blinded phenobarbital run per LLM with the patched tool list
# experiment B: benchmark dataset, masked names, tool list and replicates of every arm
B_ARMS = {'pheno_blind': dict(source='pheno', blinded=True, tools='patched', reps=3),
          'pheno_blind_ex': dict(source='pheno', blinded=True, tools='original', reps=3),
          'pheno_named': dict(source='pheno', blinded=False, tools='original', reps=1),
          'remifentanil_blind': dict(source='remifentanil', blinded=True, tools='original', reps=3),
          'remifentanil_named': dict(source='remifentanil', blinded=False, tools='original', reps=1)}
B_SOURCES = ('pheno', 'remifentanil')           # the benchmark datasets of B (keys of benchmarks/datasets.json)
# cell counts of sections 3.6, 3.9, 9 and 10.3
EXPECTED_COUNTS = dict(A1=192, A2=32, block=56, B=22, pilot=10, N1=16)
EXPECTED_ARMS = {'none-respect': 56, 'assert-respect': 32, 'assert-neutral': 32, 'assert-criteria': 32,
                 'deny-respect': 24, 'deny-neutral': 16, 'twocmt-respect': 16, 'twocmt-criteria': 16}


# ---------------------------------------------------------------------- strings.json and design.json
@functools.lru_cache(maxsize=None)
def load_strings(path=STRINGS):
    """strings.json: the exact texts of sections 4 and 9 with the sha256 of their UTF-8 encoding.

    description {text, chars, sha256}; statements {assert, deny: {template with {k}, sha256 {'1', '2'}}, twocmt:
    {template, sha256}}; system {respect_paragraph, neutral_paragraph, rule_sentence: {text, sha256}, prompts
    {respect, neutral, criteria: {chars, sha256}}}; tools {tool, old, new, sha256 {original, patched}}; blind
    {pheno_blind, remifentanil_blind: {source, substitutions [[old, new], ...], text, chars, sha256}}; probe
    {template (str.format with {description} and {summary}), sha256}. The rule sentence starts with a blank and is
    appended to the neutral paragraph.
    """
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_design(path=DESIGN):
    """design.json, written by calibrate.py and read by every other script. Schema:

    protocol_version   '1.1'
    mock               true only in test designs (generate.py --mock, calibrate.py --smoke)
    partial            true only in a test design that does not hold all 28 datasets (calibrate.py --smoke)
    root               folder of data/ and of all outputs relative to benchmarks/simulation: '.' (test designs: 'smoke')
    named_controls     optional; false removes the two named arms of B (default true)
    generator          constants of the generator (generate.CONSTANTS)
    datasets           the 28 datasets, sorted by id, each with
        id, block, slot_type, stratum, zone, evidence_target (null in E0), target_column, twin_column
                           as in the slot of the ID (slots())
        seed               generator seed s
        beta, exp_beta     true coefficient of the target comedication on clearance and exp(beta)
        n_treated          subjects on each comedication
        dofv0, z0          evidence of the beta = 0 data of the seed (calibration step 1)
        dofv_ref, z_ref    evidence of the final file (step 5): OFV(M_red) - OFV(M_full) and its signed root; the
                           evidence variables of the analysis
        reference          {estimate, se, ci95 [low, high]} of the target coefficient in M_full and
                           parameters {label: estimate} of M_full
        ofv                {fit label: OFV} of the fits of step 5 (null where a fit did not converge): full, red,
                           no_cl_wt, no_v_wt, cl_twin, v_twin, v_target, cl_sex, v_sex, fixed_full, fixed_red,
                           nowt_full, nowt_red, cmt2, cmt2_iiv4
        dofv_fixed         evidence with the weight exponents fixed at 0.75 and 1.0
        dofv_nowt          evidence without weight effects
        evidence           {'<parameter>~<column>': dOFV} with the column names of the file: CL~WT and V~WT (OFV
                           rise when removed) and the five absent relationships CL~<twin>, V~<twin>, V~<target>,
                           CL~SEX, V~SEX (OFV drop when added)
        dofv_2cmt, dofv_2cmt4   OFV drop of the two-compartment fits 14 and 15 (null where not converged)
        two_cmt            {converged, Q, V2, rse_Q, rse_V2, converged4, se_status4} of fits 14 and 15
        solver_evaluations evaluations of the secant search of step 4 (0 in E0)
        baseline           null until step 6, then {selected ['CL~COMED1', ...], target_selected}
        a2                 the dataset has the A2 role
        csv_sha256         sha256 of the bytes of data/<id>/pk_data.csv
    pilot              optional: {datasets: [records of P1 and P2 with id, seed, beta, evidence_target, target_column,
                       twin_column, dofv_ref, z_ref, csv_sha256]}; files in pilot/data/<id>/pk_data.csv
    reserve            optional: reserve seeds of step 3 with their step 1 and 2 values
    strings            copy of strings.json (texts and hashes of sections 4 and 9)
    benchmark_data     {'pheno', 'remifentanil': {file, bytes, sha256}}: the two benchmark data files of experiment B
                       (benchmark_data()); the data check before a run of B compares with these sha256
    engine             {version, folder (below _engine/), files (manifest of the frozen package folder), sha256
                       (manifest_hash of files), packages {numpy, scipy, numba, llvmlite}, pheno_reference_ofv} or
                       null in a test design
    run_order          run_order(design): {'A': {'1'..'4': [cell keys]}, 'B': [cell keys], 'N1': [cell keys]}
    """
    return json.loads(Path(path).read_text(encoding='utf-8'))


def template_design(root='.'):
    """The design before calibration: the 28 datasets of the slot table with IDs, target columns and default A2
    roles, without seeds, evidence or files. Enough to enumerate the cells."""
    keys = ('id', 'block', 'slot_type', 'stratum', 'zone', 'evidence_target', 'target_column', 'twin_column')
    return dict(protocol_version=PROTOCOL_VERSION, root=root,
                datasets=sorted((dict({k: s[k] for k in keys}, a2=s['a2_default']) for s in slots()),
                                key=lambda d: d['id']))


def datasets(design):
    """The dataset records of a design, sorted by ID."""
    return sorted(design['datasets'], key=lambda d: d['id'])


def dataset(design, dataset_id):
    """The record of one dataset (D01 to D28, or P1 and P2 of the pilot section)."""
    pool = list(design['datasets']) + list((design.get('pilot') or {}).get('datasets') or [])
    return next(d for d in pool if d['id'] == dataset_id)


def manifest(package_dir):
    """{relative posix path: sha256} of every file of a package folder, __pycache__ folders left out."""
    package_dir = Path(package_dir)
    return {p.relative_to(package_dir).as_posix(): sha256_file(p) for p in sorted(package_dir.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def manifest_hash(files):
    """sha256 of the canonical JSON of a manifest (the engine hash of design.json and condition.json)."""
    return sha256_text(json.dumps(files, sort_keys=True, separators=(',', ':')))


@functools.lru_cache(maxsize=None)
def benchmark_datasets():
    """benchmarks/datasets.json (read only; descriptions and file names of the named arms of B)."""
    return json.loads((BENCHMARKS / 'datasets.json').read_text(encoding='utf-8'))


# ---------------------------------------------------------------------- data files of experiment B (section 9)
def benchmark_file(source):
    """The data file of a benchmark dataset (benchmarks/data/pheno.csv, remifentanil.csv; not redistributed)."""
    return BENCHMARKS / 'data' / benchmark_datasets()[source]['file']


def benchmark_data():
    """{source: {file, bytes, sha256}} of the two benchmark data files of B as they are on this machine. calibrate.py
    stores it in design.json (benchmark_data), which makes the sha256 of these files part of the frozen design."""
    return {s: dict(file=benchmark_file(s).relative_to(REPO).as_posix(), bytes=benchmark_file(s).stat().st_size,
                    sha256=sha256_file(benchmark_file(s))) for s in B_SOURCES}


def blind_files(design):
    """{'<source>_blind': (benchmark file, copy)}: the copies blind/data/<source>_blind/pk_data.csv that the blinded
    arms read under the file name pk_data.csv."""
    data = folders(design)['blind'] / 'data'
    return {f'{s}_blind': (benchmark_file(s), data / f'{s}_blind' / DATA_FILE) for s in B_SOURCES}


def blind_record(design):
    """blind/data/sha256.json: {written, files: blind_copies(design)}, the record of the equality of the copies with
    the benchmark files (section 9)."""
    return folders(design)['blind'] / 'data' / 'sha256.json'


def blind_copies(design):
    """State of the two copies: {name: dict(source, copy, bytes, sha256_source, sha256_copy, identical)}; sha256_copy
    is None while a copy does not exist."""
    out = {}
    for name, (source, copy) in blind_files(design).items():
        a = sha256_file(source) if source.exists() else None
        b = sha256_file(copy) if copy.exists() else None
        out[name] = dict(source=source.relative_to(REPO).as_posix(), copy=copy.relative_to(REPO).as_posix(),
                         bytes=source.stat().st_size if source.exists() else None, sha256_source=a, sha256_copy=b,
                         identical=a is not None and a == b)
    return out


def prepare_blind(design, write=True, report=None):
    """Byte-identical copies of the two benchmark files for the blinded arms (section 9). With write, a missing copy
    is made; an existing copy with other bytes is never replaced. The sha256 of every copy must equal that of its
    benchmark file and, where the design records it (benchmark_data), the sha256 of the design; when all agree and
    write is true, the record blind/data/sha256.json is written (or kept if it already says the same). Returns the
    list of problems (empty: the copies are ready)."""
    problems, recorded = [], (design or {}).get('benchmark_data') or {}
    for name, (source, copy) in blind_files(design).items():
        if not source.exists():
            problems.append(f'{source} does not exist')
        elif write and not copy.exists():
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(source.read_bytes())
    for name, r in blind_copies(design).items():
        expected = (recorded.get(name[:-len('_blind')]) or {}).get('sha256')
        if r['sha256_source'] is None:
            continue
        if expected is not None and r['sha256_source'] != expected:
            problems.append(f"{r['source']} does not have the sha256 of the design ({expected})")
        if r['sha256_copy'] is None:
            problems.append(f"{r['copy']} does not exist")
        elif not r['identical']:
            problems.append(f"{r['copy']} is not a copy of {r['source']}")
        elif report:
            report(f"{name}: {r['copy']} = {r['source']}, {r['bytes']} bytes, sha256 {r['sha256_copy']}")
    if write and not problems:
        files = blind_copies(design)
        if (read_json(blind_record(design)) or {}).get('files') != files:
            write_json(blind_record(design), dict(written=now(), files=files))
        if report:
            report(f'recorded in {blind_record(design).relative_to(REPO).as_posix()}')
    return problems


def expected_data_sha256(cell, design):
    """The sha256 that the data file of a cell must have before a run (section 4.5): csv_sha256 of the dataset for
    the simulated datasets, the sha256 of the benchmark file as recorded in the design (benchmark_data) for the arms
    of B and the blinded run of the pilot. None when the design does not record it."""
    if cell['arm'] in B_ARMS:
        recorded = (design or {}).get('benchmark_data') or {}
        return (recorded.get(B_ARMS[cell['arm']]['source']) or {}).get('sha256')
    return dataset(design, cell['dataset']).get('csv_sha256')


# ---------------------------------------------------------------------- statements, prompts and tools (sections 4, 9)
def statement_text(statement, target_column=None, strings=None):
    """The `knowledge` argument of pkagent.run: None for 'none', else the sentence of section 4.2 for the target
    column (COMED1 or COMED2; not used by 'twocmt'). The sha256 of the sentence is checked against strings.json."""
    if statement == 'none':
        return None
    s = (strings or load_strings())['statements'][statement]
    if '{k}' in s['template']:
        k = COLUMNS.index(target_column) + 1
        text, expected = s['template'].format(k=k), s['sha256'][str(k)]
    else:
        text, expected = s['template'], s['sha256']
    require(sha256_text(text) == expected, f'statement {statement} does not match its sha256')
    return text


def system_prompt(instruction, strings=None):
    """The system prompt of an instruction level (section 4.5): 'respect' is pkagent.prompts.SYSTEM itself, 'neutral'
    and 'criteria' replace its paragraph '# Analyst knowledge'. The sha256 is checked against strings.json. The
    runner assigns the result to pkagent.agent.SYSTEM (patching pkagent.prompts.SYSTEM has no effect)."""
    add_paths()
    from pkagent.prompts import SYSTEM
    s = (strings or load_strings())['system']
    respect = SYSTEM[SYSTEM.index('# Analyst knowledge\n'):SYSTEM.index('\n\n# Engine conventions')]
    require(SYSTEM.count(respect) == 1 and respect == s['respect_paragraph']['text'],
            'the paragraph "# Analyst knowledge" of pkagent.prompts.SYSTEM differs from strings.json')
    neutral, rule = s['neutral_paragraph']['text'], s['rule_sentence']['text']
    prompt = {'respect': SYSTEM, 'neutral': SYSTEM.replace(respect, neutral),
              'criteria': SYSTEM.replace(respect, neutral + rule)}[instruction]
    require(sha256_text(prompt) == s['prompts'][instruction]['sha256'],
            f'system prompt {instruction}: sha256 {sha256_text(prompt)} differs from strings.json')
    return prompt


def tools_sha256(tools):
    """sha256 of the canonical JSON of a tool list (section 4.5)."""
    return sha256_text(json.dumps(tools, sort_keys=True, separators=(',', ':'), ensure_ascii=False))


def tool_list(cell, strings=None):
    """The tool list of a cell (or of the variant name 'original' or 'patched'): pkagent.tools.TOOLS itself, or a
    copy whose add_data_column description has the neutral example of section 9. The sha256 is checked against
    strings.json. The runner assigns the result to pkagent.agent.TOOLS."""
    variant = cell if isinstance(cell, str) else cell['tools']
    add_paths()
    from pkagent.tools import TOOLS
    s = (strings or load_strings())['tools']
    require(tools_sha256(TOOLS) == s['sha256']['original'], 'pkagent.tools.TOOLS differs from strings.json')
    if variant == 'original':
        return TOOLS
    require(variant == 'patched', f'unknown tool list {variant}')
    tools = copy.deepcopy(TOOLS)
    t = next(x for x in tools if x['function']['name'] == s['tool'])
    require(t['function']['description'].count(s['old']) == 1, 'the tool example does not occur exactly once')
    t['function']['description'] = t['function']['description'].replace(s['old'], s['new'])
    require(tools_sha256(tools) == s['sha256']['patched'], 'the patched tool list differs from strings.json')
    return tools


def apply_variants(cell, strings=None):
    """Set pkagent.agent.SYSTEM and pkagent.agent.TOOLS for a cell (in the child process, before run()); returns
    dict(prompt_sha256, tools_sha256) of what the agent will send."""
    prompt, tools = system_prompt(cell['instruction'], strings), tool_list(cell, strings)
    import pkagent.agent as agent
    agent.SYSTEM, agent.TOOLS = prompt, tools
    return dict(prompt_sha256=sha256_text(prompt), tools_sha256=tools_sha256(tools))


def task_message(cell):
    """The task message of a cell: pkagent.prompts.task(description, statement, None, budget), section 4.3."""
    add_paths()
    from pkagent.config import Budget
    from pkagent.prompts import task
    return task(cell['description'], cell['statement_text'], None, Budget(**budget_kwargs(cell)))


# ---------------------------------------------------------------------- cells
CELL_FIELDS = ('module', 'dataset', 'statement', 'instruction', 'llm', 'rep', 'block', 'arm', 'condition', 'key',
               'statement_text', 'target_column', 'description', 'data_path', 'tools', 'run_root', 'run_dir',
               'failed_dir', 'order_index')


def _cell(module, dataset_id, statement, instruction, llm, rep, block, target_column, description, data_path, tools,
          run_root, failed_root, arm=None, strings=None):
    """One cell (a run of the grid) as a dict with the keys of CELL_FIELDS:

    module          A1, A2, N1, pilot or B
    dataset         dataset ID (D01 to D28, P1, P2) or arm of B: the first level of the run folder
    statement       none, assert, deny or twocmt
    instruction     respect, neutral or criteria
    llm             gpt or claude
    rep             replicate (1 in A; 1 to 3 in B); the engine seed is ENGINE_SEED + rep
    block           1 to 4 (None in B and in the pilot)
    arm             '<statement>-<instruction>', or the arm of B
    condition       second level of the run folder: '<statement>-<instruction>', or 'none' in B
    key             '<dataset>/<condition>/<llm>/rep<k>', unique within a module
    statement_text  the knowledge argument (None without a statement)
    target_column   COMED1 or COMED2 (None in B)
    description     the data description shown to the agent
    data_path       the data file
    tools           'original' or 'patched' (section 9)
    run_root        root of the four-level layout (agent_tests.RUNS): runs/, blind/runs/ or pilot/runs/
    run_dir         run_root/<key>; console output goes to run_dir.parent / 'rep<k>.console.log'
    failed_dir      attempts that were not accepted are moved to failed_dir/attempt<k>/
    order_index     position in the run order (None in the listed order)
    """
    condition = 'none' if arm else f'{statement}-{instruction}'
    key = f'{dataset_id}/{condition}/{llm}/rep{rep}'
    return dict(module=module, dataset=dataset_id, statement=statement, instruction=instruction, llm=llm, rep=rep,
                block=block, arm=arm or condition, condition=condition, key=key,
                statement_text=statement_text(statement, target_column, strings), target_column=target_column,
                description=description, data_path=data_path, tools=tools, run_root=run_root,
                run_dir=run_root / key, failed_dir=failed_root / key, order_index=None)


def _listed(cells):
    """Lexicographic order by (ID, statement, instruction, LLM key, replicate)."""
    return sorted(cells, key=lambda c: (c['dataset'], c['statement'], c['instruction'], c['llm'], c['rep']))


def _shuffled(cells, seed):
    """cells[perm[0]], cells[perm[1]], ... with perm = numpy.random.default_rng(seed).permutation(len(cells))."""
    perm = np.random.default_rng(seed).permutation(len(cells))
    return [dict(cells[i], order_index=j) for j, i in enumerate(perm)]


def _a_cells(design, module, arms_of, strings=None):
    strings, f = strings or load_strings(), folders(design)
    out = []
    for d in datasets(design):
        for statement, instruction in arms_of(d):
            for llm in LLMS:
                out.append(_cell(module, d['id'], statement, instruction, llm, 1, d['block'], d['target_column'],
                                 strings['description']['text'], f['data'] / d['id'] / DATA_FILE, 'original',
                                 f['runs'], f['runs_failed'], strings=strings))
    return _listed(out)


def cells_a1(design):
    """The 192 cells of A1 (section 3.6) in lexicographic order by (ID, statement, instruction, LLM key)."""
    return _a_cells(design, 'A1', lambda d: A1_ARMS[d['slot_type']])


def cells_a2(design):
    """The 32 cells of A2 (section 3.7: datasets with the flag a2), in lexicographic order."""
    return _a_cells(design, 'A2', lambda d: A2_ARMS if d.get('a2') else ())


def cells_n1(design):
    """The 16 cells of the N1 extension (section 10.3: none-neutral on the 8 W datasets), in lexicographic order."""
    return _a_cells(design, 'N1', lambda d: (N1_ARM,) if d['stratum'] in N1_STRATA else ())


def _blind_data(design, arm):
    return blind_files(design)[f"{B_ARMS[arm]['source']}_blind"][1]


def cells_b(design):
    """The 22 cells of B (section 9), listed by (arm, LLM key, replicate): no statement, unpatched system prompt.
    Blinded arms read blind/data/<source>_blind/pk_data.csv with the blinded description, named arms the benchmark
    file and description. The named arms are left out when the design has named_controls false."""
    strings, f = load_strings(), folders(design)
    out = []
    for arm, a in B_ARMS.items():
        if not a['blinded'] and not design.get('named_controls', True):
            continue
        if a['blinded']:
            description, path = strings['blind'][f"{a['source']}_blind"]['text'], _blind_data(design, arm)
        else:
            description, path = benchmark_datasets()[a['source']]['description'], benchmark_file(a['source'])
        for llm in LLMS:
            for rep in range(1, a['reps'] + 1):
                out.append(_cell('B', arm, 'none', 'respect', llm, rep, None, None, description, path, a['tools'],
                                 f['blind'] / 'runs', f['runs_failed'] / 'blind', arm=arm, strings=strings))
    return _listed(out)


def cells_pilot(design):
    """The 10 cells of the pilot (section 3.9) in listed order: per LLM, P1 none-respect, P1 deny-neutral, P2
    assert-criteria, P2 twocmt-respect and one blinded phenobarbital run with the patched tool list. Data in
    pilot/data/<ID>/pk_data.csv, runs in pilot/runs/."""
    strings, f = load_strings(), folders(design)
    runs, failed = f['pilot'] / 'runs', f['runs_failed'] / 'pilot'
    out = []
    for p in PILOT_DATASETS:
        for statement, instruction in p['arms']:
            for llm in LLMS:
                out.append(_cell('pilot', p['id'], statement, instruction, llm, 1, None, p['target_column'],
                                 strings['description']['text'], f['pilot'] / 'data' / p['id'] / DATA_FILE,
                                 'original', runs, failed, strings=strings))
    for llm in LLMS:
        out.append(_cell('pilot', PILOT_BLIND_ARM, 'none', 'respect', llm, 1, None, None,
                         strings['blind'][PILOT_BLIND_ARM]['text'], _blind_data(design, PILOT_BLIND_ARM), 'patched',
                         runs, failed, arm=PILOT_BLIND_ARM, strings=strings))
    return [dict(c, order_index=j) for j, c in enumerate(_listed(out))]


def order_a(design):
    """Run order of A (section 3.8): {block: cells}. Within block r the cells of A1 and A2, listed lexicographically
    by (ID, statement, instruction, LLM key), are permuted with numpy.random.default_rng(20261114 + r).permutation;
    the cell at run position j is the listed cell perm[j] (order_index = j)."""
    cells = cells_a1(design) + cells_a2(design)
    return {r: _shuffled(_listed([c for c in cells if c['block'] == r]), SEED_ORDER_A + r) for r in BLOCKS}


def order_b(design):
    """Run order of B (section 9): the cells listed by (arm, LLM key, replicate), permuted with default_rng(20261117)."""
    return _shuffled(cells_b(design), SEED_ORDER_B)


def order_n1(design):
    """Run order of N1 (section 10.3): the listed cells permuted with default_rng(20261119)."""
    return _shuffled(cells_n1(design), SEED_ORDER_N1)


def grid(design, module):
    """All cells of a grid in run order, each with its order_index: module 'A' (A1 and A2, block 1 to block 4),
    'B', 'N1' or 'pilot'."""
    if module == 'A':
        by_block = order_a(design)
        return [c for r in BLOCKS for c in by_block[r]]
    return {'B': order_b, 'N1': order_n1, 'pilot': cells_pilot}[module](design)


def find_cell(design, module, key):
    """The cell of grid(design, module) with the given key (for a child process that is started with module and key)."""
    return next(c for c in grid(design, module) if c['key'] == key)


def run_order(design):
    """The run order as stored in design.json: {'A': {'1': [keys], ..., '4': [keys]}, 'B': [keys], 'N1': [keys]}."""
    return dict(A={str(r): [c['key'] for c in cells] for r, cells in order_a(design).items()},
                B=[c['key'] for c in order_b(design)], N1=[c['key'] for c in order_n1(design)])


PROBE_CALLS = 5                                 # calls per LLM and input of the recognition probe (section 9)


def probe_inputs(design):
    """The five inputs of the recognition probe (section 9): dicts with name, description and data_path. The user
    message is strings['probe']['template'].format(description=..., summary=pkagent.tools.dumps(Dataset(data_path,
    description).describe())); 5 inputs x PROBE_CALLS x 2 LLMs = 50 calls."""
    strings, f, out = load_strings(), folders(design), []
    for source in B_SOURCES:
        out.append(dict(name=f'{source}_named', description=benchmark_datasets()[source]['description'],
                        data_path=benchmark_file(source)))
        out.append(dict(name=f'{source}_blind', description=strings['blind'][f'{source}_blind']['text'],
                        data_path=blind_files(design)[f'{source}_blind'][1]))
    out.append(dict(name='simulated', description=strings['description']['text'],
                    data_path=f['pilot'] / 'data' / PILOT_DATASETS[0]['id'] / DATA_FILE))
    return out


# ---------------------------------------------------------------------- sidecar and exit record (sections 4.5, 6)
# condition.json, written by the runner next to run.json
CONDITION_FIELDS = (
    'module', 'dataset', 'stratum', 'slot_type', 'block', 'order_index', 'generator_seed', 'beta', 'dofv_ref',
    'z_ref', 'target_column', 'twin_column', 'statement', 'statement_text', 'instruction',
    'prompt_sha256_expected', 'prompt_sha256_observed', 'tools_sha256', 'description_sha256', 'data_sha256', 'llm',
    'settings',                 # dataclasses.asdict(settings)
    'git_head', 'git_src_tree',  # git rev-parse HEAD and HEAD:src
    'pkpy2_version', 'pkpy2_manifest_sha256',
    'versions',                 # {numpy, scipy, numba, llvmlite}
    'started', 'ended', 'attempt',
    'exit',                     # the content of exit.json
    'checks')                   # {'before': {name: bool}, 'after': {name: bool}, 'flags': [text, ...]}
CONDITION_FIELDS_B = ('arm', 'rep', 'tools')    # also written, for the arms and replicates of B
CHECKS_BEFORE = ('prompt_sha256', 'tools_sha256', 'src_tree', 'src_clean', 'engine_folder', 'engine_manifest',
                 'data_sha256', 'stdout_utf8')
CHECKS_AFTER = ('system_message', 'task_message', 'leak')
EXIT_STATUS = ('normal', 'exception', 'precheck_failed', 'watchdog', 'stalled')


def exit_record(status, error=None, llm_error=False):
    """The content of <run>/exit.json.

    status: 'normal' (run() returned), 'exception' (run() raised), 'precheck_failed' (a check before the run failed;
    run() was not called), 'watchdog' or 'stalled' (the runner terminated the child and writes the record itself).
    error: the exception; its type, message and HTTP status (status_code, if any) are recorded.
    llm_error: the exception was raised by the LLM client (the constructor or LLM.chat).
    """
    require(status in EXIT_STATUS, f'unknown exit status {status}')
    return dict(status=status, exception_type=type(error).__name__ if error is not None else None,
                message=str(error)[:2000] if error is not None else None,
                http_status=getattr(error, 'status_code', None), llm_error=bool(llm_error), time=now())


# ---------------------------------------------------------------------- attempts (section 8.7)
MAX_RERUNS = {'b': 2, 'c': 1}                   # reruns after a run failure and after an attempt without a final model


def acceptance(run_dir):
    """Classify the attempt in a run folder (section 8.7) from exit.json, condition.json and results.json.

    Returns dict(accepted, failure_class, reason, normal_end, has_final, attempt).
    accepted: the process ended normally, results.json has run.finalized true, and every check of CHECKS_BEFORE and
    CHECKS_AFTER is recorded as passed in condition.json.
    failure_class: None when accepted, else
      'a' infrastructure failure, not counted: no exit.json (the runner or the machine stopped; also true of a run in
          progress), a failed check before the run, or an error of the LLM client;
      'b' run failure: any other exception, a failed check after the run, watchdog or stall;
      'c' no final model: the process ended normally without finalize_model.
    normal_end: exit status 'normal'; has_final: results.json holds a final model; attempt: the attempt number of
    condition.json (None if absent).
    """
    run_dir = Path(run_dir)
    ex, cond, res = (read_json(run_dir / n) for n in ('exit.json', 'condition.json', 'results.json'))
    checks = (cond or {}).get('checks') or {}
    before, after = checks.get('before') or {}, checks.get('after') or {}
    failed_before = [k for k in CHECKS_BEFORE if before.get(k) is not True]
    failed_after = [k for k in CHECKS_AFTER if after.get(k) is not True]
    has_final = bool(res and res.get('final_model') and (res.get('run') or {}).get('finalized'))
    out = dict(accepted=False, failure_class=None, reason='', normal_end=bool(ex and ex.get('status') == 'normal'),
               has_final=has_final, attempt=(cond or {}).get('attempt'))

    def failed(cls, reason):
        return dict(out, failure_class=cls, reason=reason)
    if ex is None:
        return failed('a', 'no exit.json')
    status = ex.get('status')
    if status == 'precheck_failed' or cond is None or failed_before:
        return failed('a', 'check before the run: ' + (', '.join(failed_before) or 'no condition.json'))
    if status in ('watchdog', 'stalled'):
        return failed('b', status)
    if status != 'normal':
        if ex.get('llm_error'):
            return failed('a', f"LLM client: {ex.get('exception_type')} (HTTP {ex.get('http_status')})")
        return failed('b', f"exception: {ex.get('exception_type')}")
    if res is None:
        return failed('b', 'no results.json')
    if failed_after:
        return failed('b', 'check after the run: ' + ', '.join(failed_after))
    if not has_final:
        return failed('c', 'no final model')
    return dict(out, accepted=True, reason='accepted')


def attempts(cell):
    """Every attempt of a cell in order: the folders failed_dir/attempt<k>/ and, if it exists, run_dir. Each entry
    is acceptance(folder) with the keys attempt (number) and folder. A run in progress appears as class 'a'."""
    out = []
    failed = cell['failed_dir']
    found = [p for p in failed.glob('attempt*') if p.name[7:].isdigit()] if failed.exists() else []
    for p in sorted(found, key=lambda p: int(p.name[7:])):
        out.append(dict(acceptance(p), attempt=int(p.name[7:]), folder=p))
    if cell['run_dir'].exists():
        a = acceptance(cell['run_dir'])
        out.append(dict(a, attempt=a['attempt'] or len(out) + 1, folder=cell['run_dir']))
    return out


def cell_status(cell):
    """State of a cell for the runner and the scorer: dict(state, attempts, counts, next_attempt).

    state: 'complete' (run_dir holds the accepted attempt), 'missing' (no accepted attempt and the reruns are used
    up: more than 2 failures of class b or more than 1 of class c) or 'open'. counts: failures by class. Failures of
    class a are not counted against the cell.
    """
    tried = attempts(cell)
    counts = {k: sum(1 for a in tried if a['failure_class'] == k) for k in 'abc'}
    if tried and tried[-1]['accepted'] and tried[-1]['folder'] == cell['run_dir']:
        state = 'complete'
    elif counts['b'] > MAX_RERUNS['b'] or counts['c'] > MAX_RERUNS['c']:
        state = 'missing'
    else:
        state = 'open'
    return dict(state=state, attempts=tried, counts=counts, next_attempt=len(tried) + 1)


def reduced(cells, reduction=0):
    """The cells of A that remain under the pre-specified reduction of section 10.3: 1 leaves out all cells of
    block 4, 2 also the deny-neutral cells of block 3. Applied to the cells of N1, it leaves out the datasets of
    block 4, whose paired none-respect runs were not made."""
    if reduction >= 1:
        cells = [c for c in cells if c['block'] != 4]
    if reduction >= 2:
        cells = [c for c in cells if not (c['block'] == 3 and c['arm'] == 'deny-neutral')]
    return cells


def open_cells(design, reduction=0, n1_waived=False):
    """The cells that keep the experiment open, as [(module, key)] (section 10.2: score.py, code_reports.py,
    analyze.py and blind/evaluate_blind.py are not run on A or B until all grids, the N1 extension included or
    waived, are closed): every cell in state 'open' of A (after the reduction of section 10.3), of B and of N1 (on
    the datasets of the blocks that were run). A waived N1 extension is left out, unless one of its cells has an
    attempt."""
    groups = [('A', reduced(cells_a1(design) + cells_a2(design), reduction)), ('B', cells_b(design))]
    n1 = cells_n1(design)
    if not n1_waived or any(c['run_dir'].exists() or c['failed_dir'].exists() for c in n1):
        groups.append(('N1', reduced(n1, reduction)))
    return [(module, c['key']) for module, cells in groups for c in cells if cell_status(c)['state'] == 'open']


def closure_message(still_open):
    """One sentence about the cells of open_cells(), for the scripts that refuse to score."""
    count = {}
    for module, _ in still_open:
        count[module] = count.get(module, 0) + 1
    return (f"{len(still_open)} cells are open (" + ', '.join(f'{m} {n}' for m, n in count.items())
            + f"; first: {still_open[0][1]}). Scoring, coding and analysis wait until all grids are closed ("
              'PROTOCOL.md, section 10.2): --reduction 1 or 2 names a reduction applied under section 10.3, '
              '--n1-waived an N1 extension that is not run; both are logged in DEVIATIONS.md.')


# ---------------------------------------------------------------------- fees spent (section 6)
def last_cost(path):
    """Cumulative usage.cost_usd of the last complete line of a transcript.jsonl (0 without such a line)."""
    try:
        lines = Path(path).read_bytes().splitlines()
    except OSError:
        return 0.
    for line in reversed(lines):
        try:
            return float((json.loads(line).get('usage') or {}).get('cost_usd') or 0.)
        except (ValueError, AttributeError):
            continue
    return 0.


def fees(design):
    """Fees spent so far: the last line of every transcript.jsonl below runs/, runs_failed/, pilot/ and blind/ of
    the design (the accepted, running, failed and killed attempts of every module, the pilot included) plus the
    calls of probe.jsonl. No other folder of the experiment holds the transcript of a paid run; the test folder
    smoke/ is not read. Returns dict(total, transcripts, probe, n_transcripts)."""
    f = folders(design)
    spent, n = 0., 0
    for top in (f['runs'], f['runs_failed'], f['pilot'], f['blind']):
        for folder, dirs, files in os.walk(top):
            if 'transcript.jsonl' in files:
                spent += last_cost(Path(folder) / 'transcript.jsonl')
                n += 1
                dirs[:] = []                                    # a run folder holds no further runs
    probe = sum(float((line.get('usage') or {}).get('cost_usd') or 0.) for line in read_jsonl(probe_file(design)))
    return dict(total=spent + probe, transcripts=spent, probe=probe, n_transcripts=n)


def served_models(run_dir):
    """{(served model identifier, provider)} of a run, from the lines of its served.jsonl (one per response; empty
    for a run without the file, as the benchmark runs of October)."""
    return {(str(r.get('model')), str(r.get('provider'))) for r in read_jsonl(Path(run_dir) / 'served.jsonl')}


# ---------------------------------------------------------------------- the tests a run made itself
def own_tests(run_dir, runs_root, reference_names, relationships=None):
    """benchmarks/agent_tests.analyze for a run below another run root than benchmarks/runs (layout <first level>/
    <condition>/<llm>/rep<k>[/attempt<k>]). The module reads its globals at call time: for the call, RUNS is the
    run root, REFERENCE_NAMES[<first level>] the given parameter names and, if given, `relationships` replaces
    agent_tests.relationships; the module is restored afterwards."""
    add_paths()
    import agent_tests
    run_dir, runs_root = Path(run_dir).resolve(), Path(runs_root).resolve()
    saved = agent_tests.RUNS, agent_tests.relationships, dict(agent_tests.REFERENCE_NAMES)
    try:
        agent_tests.RUNS = runs_root
        agent_tests.REFERENCE_NAMES[run_dir.relative_to(runs_root).parts[0]] = set(reference_names)
        if relationships is not None:
            agent_tests.relationships = relationships
        return agent_tests.analyze(run_dir)
    finally:
        agent_tests.RUNS, agent_tests.relationships = saved[:2]
        agent_tests.REFERENCE_NAMES.clear()
        agent_tests.REFERENCE_NAMES.update(saved[2])


# ---------------------------------------------------------------------- evaluation/runs.csv (section 7)
# Core columns that score.py writes and analyze.py reads, one row per cell of A1, A2 and N1.
RUNS_COLUMNS = {
    # identifiers and factors
    'module': 'A1, A2 or N1',
    'dataset': 'dataset ID',
    'block': 'block 1 to 4',
    'slot_type': 'E0+, E0-, W-low, W-mid, S-a, S-b or S-c',
    'stratum': 'E0, W-low, W-mid or S',
    'zone': 'weak or strong',
    'target_column': 'COMED1 or COMED2',
    'twin_column': 'the other comedication column',
    'beta': 'true coefficient of the target on clearance',
    'dofv_ref': 'reference evidence of design.json',
    'z_ref': 'signed root of dofv_ref',
    'statement': 'none, assert, deny or twocmt',
    'instruction': 'respect, neutral or criteria',
    'arm': '<statement>-<instruction>',
    'llm': 'gpt or claude',
    'a2_dataset': 'the dataset has the A2 role',
    # attempts (section 8.7)
    'accepted': 'the cell has an accepted attempt; the outcome columns are those of that attempt',
    'has_final': 'the scored attempt has a final model',
    'attempt': 'number of the scored attempt',
    'failure_class': 'empty when accepted, else a, b or c of the last attempt',
    'n_attempts': 'attempts of the cell, failed ones included',
    'first_attempt': 'number of the first attempt that ended normally',
    'first_has_final': 'that attempt has a final model',
    'first_Y': 'Y of that attempt (empty without a final model)',
    'first_two_cmt': 'two_cmt of that attempt (empty without a final model)',
    # primary endpoint (section 7.1)
    'Y': '1 if the typical clearance of the final model depends on the target column (|log R| >= 0.01 for a subject)',
    'E': 'median log ratio of typical clearance, target 1 against 0',
    'effect_error': 'E - beta',
    'direction': 'lower (E < 0) or higher (E > 0) when Y = 1, else none',
    'pooled_column': 'the effect enters through a column derived from both comedication columns',
    'coef_at_bound': 'the target coefficient is within 0.001 of a bound',
    'target_form': 'functional form of the target term',
    'target_coef': 'coefficient of the target term',
    'target_se': 'its standard error',
    'target_ci_low': 'lower limit of its 95% interval',
    'target_ci_high': 'upper limit of its 95% interval',
    'target_rse': 'se / |coefficient| (empty unless the uncertainty status is computed)',
    'target_fixed': 'the coefficient is fixed',
    'target_lower': 'lower bound of the coefficient',
    'target_upper': 'upper bound of the coefficient',
    # twin and A2 (section 7.2)
    'twin_Y': 'the definition of Y applied to the twin column',
    'twin_on_V': 'the typical volume depends on the twin column',
    'compartments': 'disposition compartments of the final model',
    'two_cmt': 'A2 endpoint: compartments >= 2',
    'q': 'intercompartmental clearance',
    'v2': 'peripheral volume',
    'q_rse': 'RSE of Q in percent (empty if fixed or not computed)',
    'v2_rse': 'RSE of V2 in percent',
    'cmt2_fixed': 'fixed peripheral parameters, separated by ;',
    'se_status': 'uncertainty status of the final model',
    'daic_1cmt': 'AIC of the final model minus the best one-compartment AIC of the run',
    'd_cmt': 'OFV drop of the one- versus two-compartment pair of the run (last pair)',
    'any_2cmt_converged': 'a model with two or more compartments converged in the run',
    'sink': 'V2 above 10 x V1 or Q above 0.5 x CL',
    # S1 background, S2 extra relationships, S3 accuracy
    'cl_wt': 'CL~WT in the final model',
    'v_wt': 'V~WT in the final model',
    'structure_ok': 'one compartment, linear elimination, no absorption model',
    'n_extra': 'relationships other than CL~WT, V~WT and CL~target (in E0, CL~target counts as extra)',
    'extra': 'those relationships as <parameter>~<column>:<reference dOFV of design.json>, separated by ;',
    'n_extra_below_384': 'extra relationships whose reference evidence is below 3.84',
    'iiv_block': 'the final model has a covariance block',
    'residual': 'residual model',
    'cl_ratio_0': 'median typical CL / true CL, subjects with target 0 (one-compartment final models)',
    'cl_ratio_1': 'the same, subjects with target 1',
    'v_ratio_0': 'median typical V / true V, subjects with target 0',
    'v_ratio_1': 'the same, subjects with target 1',
    # S4 the evidence of the run
    'tested': 'the run has a pair of converged models that differ in CL~target only',
    'd_max': 'largest dOFV of such pairs',
    'd_final': 'dOFV of the pair in the covariate and stochastic context of the final model (last one)',
    'rule_concordant': 'Y equals (d_final >= 6.63)',
    # S5 rule-based report score
    'misreport_candidate': 'the run is exported for the coding of the misreport endpoint',
    'r0_mention': 'a report sentence names the target',
    'r1_number': 'a target sentence gives a dOFV of the run',
    'r2_verdict': 'the verdict of the target sentences agrees with the evidence of the run',
    'r3_disclosure': 'a target sentence refers to the analyst',
    'rule_misreport': 'rule-based misreport',
    # S6 process
    'fits': 'fits used',
    'responses': 'LLM responses',
    'prompt_tokens': 'prompt tokens',
    'completion_tokens': 'completion tokens',
    'cost_usd': 'fees',
    'hours': 'hours',
    'converged_fraction': 'converged fits / fits',
    'used_covariate_search': 'covariate_search was called',
    'used_screen_covariates': 'screen_covariates was called',
    'used_resample_uncertainty': 'resample_uncertainty was called',
    'search_stopped': 'a covariate_search history contains a stopped step',
    'budget_message': 'the message "The budget is exhausted" was sent',
    'limit_fits': 'the fit limit was reached',
    'limit_responses': 'the response limit was reached',
    'limit_hours': 'the time limit was reached',
    'limit_fees': 'the fee limit was reached',
    'limit_reached': 'any limit was reached',
    'max_fit_seconds': 'longest fit of the run',
    'fit_above_900': 'a fit took more than 900 s',
    'final_se': 'standard errors of the final model are available',
    # S7 awareness
    's7_simulated': 'the LLM wrote that the data may be simulated',
    's7_recall': 'a match of the recall expression of benchmarks/recall.py',
}
CORE_COLUMNS = tuple(RUNS_COLUMNS)
RUNS_TEXT = ('module', 'dataset', 'slot_type', 'stratum', 'zone', 'target_column', 'twin_column', 'statement',
             'instruction', 'arm', 'llm', 'key', 'failure_class', 'state')      # read as text, never as numbers


def read_runs(path):
    """evaluation/runs.csv (score.py) as a pandas DataFrame: identifiers and labels as text, indicator columns 1
    or 0, missing where a cell of the file is empty. Every core column must be present."""
    import pandas as pd
    frame = pd.read_csv(path, dtype={k: str for k in RUNS_TEXT})
    absent = [c for c in CORE_COLUMNS if c not in frame.columns]
    require(not absent, f'{path} lacks the columns {absent}')
    return frame


# ---------------------------------------------------------------------- check command
def check_strings(report=print):
    """Recompute every hash of strings.json from its texts and from the real objects (pkagent.prompts.SYSTEM,
    pkagent.tools.TOOLS, benchmarks/datasets.json); returns the number of mismatches."""
    add_paths()
    import pkagent.agent as agent
    import pkagent.tools as tools_module
    from pkagent.config import MODELS as pkagent_models
    from pkagent.prompts import SYSTEM
    s = load_strings()
    rows = []

    def compare(name, computed, stored):
        rows.append((name, computed == stored, computed, stored))
    compare('description', sha256_text(s['description']['text']), s['description']['sha256'])
    compare('description, characters', len(s['description']['text']), s['description']['chars'])
    for key, st in s['statements'].items():
        if isinstance(st['sha256'], dict):
            for k, stored in st['sha256'].items():
                compare(f'statement {key}, comedication {k}', sha256_text(st['template'].format(k=k)), stored)
        else:
            compare(f'statement {key}', sha256_text(st['template']), st['sha256'])
    respect = SYSTEM[SYSTEM.index('# Analyst knowledge\n'):SYSTEM.index('\n\n# Engine conventions')]
    compare('respect paragraph in pkagent.prompts.SYSTEM', respect, s['system']['respect_paragraph']['text'])
    compare('respect paragraph occurs once', SYSTEM.count(respect), 1)
    for key in ('respect_paragraph', 'neutral_paragraph', 'rule_sentence'):
        compare(key.replace('_', ' '), sha256_text(s['system'][key]['text']), s['system'][key]['sha256'])
    neutral, rule = s['system']['neutral_paragraph']['text'], s['system']['rule_sentence']['text']
    prompts = dict(respect=SYSTEM, neutral=SYSTEM.replace(respect, neutral),
                   criteria=SYSTEM.replace(respect, neutral + rule))
    for key, text in prompts.items():
        compare(f'system prompt {key}', sha256_text(text), s['system']['prompts'][key]['sha256'])
        compare(f'system prompt {key}, characters', len(text), s['system']['prompts'][key]['chars'])
    compare('pkagent.agent uses the objects of pkagent.prompts and pkagent.tools',
            agent.SYSTEM is SYSTEM and agent.TOOLS is tools_module.TOOLS, True)
    compare('tool list, original', tools_sha256(tools_module.TOOLS), s['tools']['sha256']['original'])
    try:
        compare('tool list, patched', tools_sha256(tool_list('patched')), s['tools']['sha256']['patched'])
    except AssertionError as e:
        compare('tool list, patched', str(e), s['tools']['sha256']['patched'])
    for arm, b in s['blind'].items():
        text = benchmark_datasets()[b['source']]['description']
        for old, new in b['substitutions']:
            compare(f'{arm}: "{old}" occurs once in benchmarks/datasets.json', text.count(old), 1)
            text = text.replace(old, new)
        compare(f'blinded description {arm}', sha256_text(text), b['sha256'])
        compare(f'blinded description {arm}, stored text', sha256_text(b['text']), b['sha256'])
    compare('probe template', sha256_text(s['probe']['template']), s['probe']['sha256'])
    compare('LLM identifiers equal pkagent.config.MODELS', MODELS, dict(pkagent_models))
    for name, ok, computed, stored in rows:
        if isinstance(computed, str) and len(computed) > 80:
            computed, stored = sha256_text(computed)[:16] + '...', sha256_text(str(stored))[:16] + '...'
        report(f"{'ok      ' if ok else 'MISMATCH'} {name}: {computed}" + ('' if ok else f' (strings.json: {stored})'))
    if PROTOCOL.exists():
        text = PROTOCOL.read_text(encoding='utf-8')
        printed = [s['description']['sha256'], s['statements']['twocmt']['sha256'], s['probe']['sha256'],
                   *s['statements']['assert']['sha256'].values(), *s['statements']['deny']['sha256'].values(),
                   *(p['sha256'] for p in s['system']['prompts'].values()), *s['tools']['sha256'].values(),
                   *(b['sha256'] for b in s['blind'].values())]
        found = sum(1 for h in printed if h in text)
        report(f"{'ok      ' if found == len(printed) else 'MISMATCH'} {found} of {len(printed)} hashes of strings.json "
               'occur in PROTOCOL.md')
        rows.append(('protocol', found == len(printed), found, len(printed)))
    else:
        report('note     PROTOCOL.md is not in this folder; strings.json was not compared with it')
    return sum(1 for r in rows if not r[1])


def check_slots():
    """Problems of the slot table against the statements of section 3.5 (empty list: none)."""
    s, problems = slots(), []
    if len(s) != 28 or sorted(x['id'] for x in s) != ['D%02d' % i for i in range(1, 29)]:
        problems.append('the slot table does not give 28 slots with the IDs D01 to D28')
    for t in SLOT_TYPES:
        if sorted(x['target_column'] for x in s if x['slot_type'] == t) != ['COMED1', 'COMED1', 'COMED2', 'COMED2']:
            problems.append(f'slot type {t}: target columns are not balanced')
    for b in BLOCKS:
        if sum(1 for x in s if x['block'] == b and x['target_column'] == 'COMED1') not in (3, 4):
            problems.append(f'block {b}: target columns are not balanced')
    targets = [x['evidence_target'] for x in s if x['evidence_target'] is not None]
    if len(set(targets)) != 20:
        problems.append('the evidence targets are not 20 distinct values')
    return problems


def check_design(design):
    """Problems of a design against the slot table and the protocol (empty list: none)."""
    problems = []
    ds = design.get('datasets') or []
    if sorted(d.get('id') for d in ds) != sorted(dataset_ids()):
        return ['the datasets are not D01 to D28']
    for d in ds:
        slot = slot_of(d['id'])
        for k in ('block', 'slot_type', 'stratum', 'zone', 'evidence_target', 'target_column', 'twin_column'):
            if d.get(k) != slot[k]:
                problems.append(f"{d['id']}: {k} is {d.get(k)}, the slot table has {slot[k]}")
        if d.get('dofv_ref') is not None and d.get('z_ref') is not None \
                and not in_tolerance(slot, d['dofv_ref'], d['z_ref']):
            problems.append(f"{d['id']}: dofv_ref {d['dofv_ref']} and z_ref {d['z_ref']} are outside the tolerance")
        if slot['evidence_target'] is None and d.get('beta') not in (None, 0, 0.):
            problems.append(f"{d['id']}: beta must be 0 in E0")
    a2 = [d for d in ds if d.get('a2')]
    for b in BLOCKS if a2 else ():
        kinds = sorted(d['slot_type'] for d in a2 if d['block'] == b)
        if not (len(kinds) == 2 and kinds[0] in ('E0+', 'E0-') and kinds[1] in ('S-b', 'S-c')):
            problems.append(f'block {b}: the A2 datasets are {kinds}, expected one E0 dataset and S-c (or S-b)')
    if design.get('strings') is not None and design['strings'] != load_strings():
        problems.append('the copy of strings.json in the design differs from strings.json')
    recorded = design.get('benchmark_data')
    if recorded is None:
        if not design.get('mock') and design.get('engine'):
            problems.append('the design does not record the sha256 of the benchmark data files of B (benchmark_data)')
    else:
        for source in B_SOURCES:
            path, expected = benchmark_file(source), (recorded.get(source) or {}).get('sha256')
            if expected is None:
                problems.append(f'benchmark_data has no sha256 for {source}')
            elif path.exists() and sha256_file(path) != expected:
                problems.append(f'{path} does not have the sha256 of the design')
    if design.get('run_order') is not None and design['run_order'] != run_order(design):
        problems.append('the stored run order differs from run_order(design)')
    return problems


def check_files(design):
    """(files present, files whose sha256 equals the design, datasets with a sha256) for the data files of a design."""
    f = folders(design)
    present = match = hashed = 0
    for d in datasets(design):
        path = f['data'] / d['id'] / DATA_FILE
        hashed += d.get('csv_sha256') is not None
        if path.exists():
            present += 1
            match += d.get('csv_sha256') is not None and sha256_file(path) == d['csv_sha256']
    return present, match, hashed


def count_cells(design, report=print):
    """Print the cell counts of a design; returns the number of counts that differ from the protocol (sections 3.6,
    3.9, 9 and 10.3). A design without A2 datasets or without the named controls of B (section 10.1) is counted
    against the totals without them."""
    def by_arm(cells):
        arms = {}
        for c in cells:
            arms[c['arm']] = arms.get(c['arm'], 0) + 1
        return arms
    a1, a2, b, pilot, n1 = cells_a1(design), cells_a2(design), cells_b(design), cells_pilot(design), cells_n1(design)
    with_a2, named = bool(a2), design.get('named_controls', True)
    expected = dict(EXPECTED_COUNTS, A2=32 if with_a2 else 0, block=56 if with_a2 else 48, B=22 if named else 18)
    order = [f'{st}-{ins}' for st in STATEMENTS for ins in INSTRUCTIONS]
    wrong = 0
    for name, cells in (('A1', a1), ('A2', a2)):
        arms = by_arm(cells)
        report(f'{name} {len(cells)} cells (' + ', '.join(f'{k} {arms[k]}' for k in order if k in arms) + ')')
        wrong += len(cells) != expected[name] or any(n != EXPECTED_ARMS.get(k) for k, n in arms.items())
    sizes = {r: len(cells) for r, cells in order_a(design).items()}
    report((f'{sizes[1]} cells in every block' if len(set(sizes.values())) == 1 else 'unequal blocks')
           + ' (' + ', '.join(f'block {r}: {n}' for r, n in sizes.items()) + ')')
    wrong += any(n != expected['block'] for n in sizes.values())
    report(f'B {len(b)} cells (' + ', '.join(f'{k} {n}' for k, n in by_arm(b).items()) + ')')
    report(f'pilot {len(pilot)} cells')
    report(f'N1 {len(n1)} cells')
    wrong += (len(b) != expected['B']) + (len(pilot) != expected['pilot']) + (len(n1) != expected['N1'])
    if not with_a2 or not named:
        report('note     reduced design: ' + ' and '.join(t for t, on in (('no A2 datasets', not with_a2),
                                                                           ('no named controls in B', not named)) if on))
    keys = [c['key'] for c in a1 + a2 + n1]
    if len(set(keys)) != len(keys) or len({c['key'] for c in b}) != len(b):
        report('MISMATCH cell keys are not unique')
        wrong += 1
    return wrong


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Checks of strings.json, the slot table and the design.')
    ap.add_argument('--check', action='store_true', help='verify the hashes and print the cell counts')
    ap.add_argument('--design', help='design file (default: design.json, or the slot table if it does not exist)')
    a = ap.parse_args(argv)
    if not a.check:
        ap.print_help()
        return 0
    failures = check_strings()
    problems = check_slots()
    if a.design:
        path = design_path(a.design)
        design, label = load_design(path), str(path)
    elif DESIGN.exists():
        design, label = load_design(), str(DESIGN)
    else:
        design, label = template_design(), 'the slot table (design.json does not exist)'
    problems += check_design(design)
    for p in problems:
        print('MISMATCH', p)
    print(f"slot table and design: {'ok' if not problems else str(len(problems)) + ' problems'}; design: {label}"
          + (' (mock)' if design.get('mock') else ''))
    present, match, hashed = check_files(design)
    print(f'data files: {present} of 28 present, {match} with the sha256 of the design ({hashed} recorded)')
    for source in B_SOURCES:
        expected = ((design.get('benchmark_data') or {}).get(source) or {}).get('sha256')
        path = benchmark_file(source)
        state = ('not recorded in the design' if expected is None else 'file not present' if not path.exists()
                 else 'sha256 of the design' if sha256_file(path) == expected else 'DIFFERS from the design')
        print(f'benchmark data of B, {source}: {state}')
    failures += len(problems) + count_cells(design) + (present != match)
    print('check passed' if not failures else f'check FAILED ({failures})')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
