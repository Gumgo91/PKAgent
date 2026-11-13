"""Calibration of the evidence strength (PROTOCOL.md, section 5, steps 0 to 7) and the pilot datasets (section 3.9).
No LLM. Every command can be stopped and started again: finished fits are read from calibration/fits/ and are never
fitted twice.

Usage (from the repository root; python benchmarks/simulation/calibrate.py COMMAND [--jobs N] [--smoke]):
  engine     Step 0. Copies the PKPy2 package folder (without __pycache__) to _engine/pkpy2-<version>-<hash8>/pkpy2,
             writes _engine/pkpy2-<version>-<hash8>.zip and the record _engine/pkpy2-<version>-<hash8>.json (manifest
             with the sha256 of every file, package versions), checks that this process, a child process and a fit
             worker import pkpy2 from the frozen copy, and refits the phenobarbital reference model as
             benchmarks/reference_fits.py does (output in _engine/pkpy2-<version>-<hash8>-check/). The refit runs in
             one worker process and is the first fit on the frozen copy, so the compiled cache exists before
             parallel fits start; its OFV must be 871.147. A frozen copy with the same manifest is reused; if the
             working tree differs from the frozen copy the command stops.
  run        Steps 1 to 5 and 7. Screens candidate seeds 1, 2, 3, ... at beta = 0 (steps 1 and 2), fills the 28 slots
             and the reserve (step 3), solves beta for the W and S slots (step 4), writes data/<ID>/pk_data.csv, fits
             the 15 reference models of every file through pkagent.session.Session.fit and applies the hard checks
             (step 5). A dataset that fails step 4 or a hard check is replaced from the reserve and the A2 roles are
             set (section 3.7). Writes calibration/candidates.csv (every candidate seed with the reason),
             calibration/search.json (slots, reserve, solver points, checks), calibration/evidence.csv,
             calibration/calibrate.log, the files of step 7 and design.json.
  null       Step 7 alone, from calibration/search.json: dOFV0 of all candidates of step 1 and the twin's dOFV of all
             candidates of step 2 against the chi-square distribution with one degree of freedom. Writes
             calibration/null_calibration.json, .csv and .png (quantile plot).
  baseline   Step 6, on the files of design.json: session.covariate_search from M_red with the CL and V effects of
             COMED1, COMED2 and SEX. Writes calibration/baseline/ and the field baseline of every dataset in
             design.json. Run it before the agent runs or after B, never while agent runs are in progress.
  pilot      The pilot datasets P1 and P2: generator seeds 9001 and 9002 or the next seeds that pass steps 1 and 2,
             solved to their evidence targets, fits 1 and 2 of step 5 on the files. Writes
             pilot/data/<ID>/pk_data.csv, pilot/pilot.json, pilot/candidates.csv and the pilot section of
             design.json (if design.json exists; run adds it otherwise).
  options    The two decisions that section 10.1 leaves to the freeze, before any agent run: --remove-a2 takes
             the A2 role from every dataset, --remove-named-controls removes the two named arms of B. design.json is
             rewritten with the run order of the remaining cells. Without an option the present state is printed.
             (calibrate.py run restores both parts from the cached fits.)
  identity   Prints where this process imports pkpy2 from (used by engine for the child process).
Options:
  --jobs N   parallel fitting sessions of 3 workers x 2 threads (default 2). Steps 1 to 4 use one pool of 3 N
             workers that call pkagent.engine.run_fit; steps 5 and 6 and the pilot files use N sessions.
  --smoke    test run: every output goes to smoke/calibration_test/ (the engine stays in _engine/), the slots are
             the W-low and the S-a slot of block 1, there is no reserve, at most 12 candidate seeds are screened and
             step 5 is run for one dataset.

Fit records: calibration/fits/s<seed>/b<beta>/<model>.json, one per (seed, beta, model), with status, OFV, estimates
and fit seconds. A fit that did not converge, failed or passed its wall limit is recorded as such and is handled by
the rules of section 5; it is fitted again only after its record is deleted. The work files of the fits are next to
the records (steps 1 to 4) or in calibration/step5/ and calibration/baseline/ (session folders). A fit that had
finished when the process was stopped, but had no record yet, is recorded from its work folder at the next start.

run_grid.py calls engine_record(design), engine_parent(engine), child_env(engine), engine_identity(engine),
engine_problems(engine) and reference_specs() to run on the frozen engine, to check it and to fit the reference
models of the load test.
"""
import argparse
import copy
import csv
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import psutil

import common
import generate

PHENO_OFV = 871.147                        # OFV of the phenobarbital reference model on the frozen engine (step 0)
PACKAGES = dict(numpy='2.2.6', scipy='1.16.1', numba='0.65.1', llvmlite='0.47.0')     # versions of section 5
NULL_MAX = 0.5                             # step 1: dOFV0 at most
TWIN_MAX = 2.0                             # steps 2 and 5: OFV drop of CL~twin below
V_TARGET_MAX = common.INCLUDE              # steps 2 and 5: OFV drop of V~target below
WEIGHT_MIN = 10.0                          # steps 2 and 5: OFV rise when a weight effect is removed, at least
CMT2_MAX = common.INCLUDE                  # step 5: OFV drop of a converged two-compartment fit below
RESERVE, RESERVE_E0 = 8, 2                 # step 3: reserve seeds, and how many of them qualify for E0+ and for E0-
SLOPE = 0.07                               # step 4: beta per unit of the signed root (first guess)
MAX_EVALUATIONS = 6                        # step 4
START_AGREEMENT = 0.05                     # step 5: the two Laplace starts of fits 1 and 2 agree within (OFV)
SEED = common.ENGINE_SEED + 1              # engine seed of the agent runs of A (replicate 1)
BATCH = 48                                 # candidate seeds screened at a time, at most

SEARCH = ('red', 'full')                                    # fits of steps 1 and 4 (internal columns CA and CB)
STEP2 = ('cl_twin', 'v_target', 'no_cl_wt', 'no_v_wt')      # fits of step 2
EFFECT = 'CL~CA=1'                                          # label of the target coefficient in those fits
# labels of the 15 fits of step 5, in the order of the table of section 5
LABELS = ('full', 'red', 'no_cl_wt', 'no_v_wt', 'cl_twin', 'v_twin', 'v_target', 'cl_sex', 'v_sex', 'fixed_full',
          'fixed_red', 'nowt_full', 'nowt_red', 'cmt2', 'cmt2_iiv4')
TWO_STARTS = ('full', 'red')               # fits 1 and 2: two Laplace starts in their own session
WITH_SE = ('full', 'cmt2', 'cmt2_iiv4')    # fits with standard errors
IIV_START = 0.04                           # starting variance of Q and V2 in fit 15

SMOKE = dict(root='smoke/calibration_test', slots=((1, 'W-low'), (1, 'S-a')), reserve=0, reserve_e0=0, max_seeds=12,
             datasets=1, mock=True)
CANDIDATE_COLUMNS = ('seed', 'n_treated', 'draws', 'ofv_red', 'ofv_full', 'estimate0', 'dofv0', 'z0', 'step1',
                     'dofv_twin', 'dofv_v_target', 'dofv_cl_wt', 'dofv_v_wt', 'step2', 'accepted', 'role', 'slot',
                     'beta', 'evaluations', 'fit_seconds', 'reason')
DESIGN_FIELDS = ('id', 'block', 'slot_type', 'stratum', 'zone', 'evidence_target', 'target_column', 'twin_column',
                 'seed', 'beta', 'exp_beta', 'n_treated', 'dofv0', 'z0', 'dofv_ref', 'z_ref', 'reference', 'ofv',
                 'dofv_fixed', 'dofv_nowt', 'evidence', 'dofv_2cmt', 'dofv_2cmt4', 'two_cmt', 'solver_evaluations',
                 'baseline', 'a2', 'csv_sha256')


# ---------------------------------------------------------------------- small helpers
def _clear(folder):
    """Remove the work folder of a fit that was not finished."""
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
        common.require(not folder.exists(), f'{folder} cannot be removed (is another calibration process running?)')


def _parallel(function, items, workers):
    """function(item) for every item in threads; the results are in the order of the items."""
    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), len(items)))) as ex:
        return list(ex.map(function, items))


class Log:
    """Progress lines with a time stamp, printed and appended to a file."""

    def __init__(self, path=None):
        self.path, self.lock = path, threading.Lock()

    def __call__(self, text):
        line = f'[{common.now()}] {text}'
        with self.lock:
            print(line, flush=True)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, 'a', encoding='utf-8') as out:
                    out.write(line + '\n')


# ---------------------------------------------------------------------- step 0: frozen engine
def engine_source():
    """The PKPy2 package folder that Python finds outside _engine/ (the working tree). Nothing is imported."""
    engine = common.ENGINE.resolve()
    paths = [p for p in sys.path if p and engine not in Path(p).resolve().parents and Path(p).resolve() != engine]
    spec = importlib.machinery.PathFinder.find_spec('pkpy2', paths)
    common.require(spec is not None and spec.origin, 'pkpy2 is not installed')
    return Path(spec.origin).resolve().parent


def package_versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def freeze(report=print):
    """Freeze the engine (step 0) and return its record. A frozen copy with the manifest of the working tree is
    reused; if the working tree differs from the frozen copy, AssertionError."""
    source = engine_source()
    match = re.search(r'''__version__\s*=\s*["']([^"']+)["']''', (source / '__init__.py').read_text(encoding='utf-8'))
    common.require(match is not None, f'no __version__ in {source / "__init__.py"}')
    version, files = match.group(1), common.manifest(source)
    digest = common.manifest_hash(files)
    name = f'pkpy2-{version}-{digest[:8]}'
    existing = sorted(p.name for p in common.ENGINE.glob('pkpy2-*') if p.is_dir() and (p / 'pkpy2').is_dir())
    target, record_path = common.ENGINE / name / 'pkpy2', common.ENGINE / f'{name}.json'
    if name in existing:
        common.require(common.manifest(target) == files,
                       f'the frozen copy {target} no longer has the manifest of its name; move _engine/{name} away '
                       'and freeze again')
        report(f'engine: frozen copy {name} has the manifest of the working tree and is reused')
    else:
        common.require(not existing,
                       f'the PKPy2 working tree {source} (version {version}, manifest {digest[:8]}) differs from the '
                       f'frozen copy _engine/{existing[0] if existing else ""}. The frozen copy stays in use by the '
                       'other commands; to freeze the present working tree, move the frozen copy away first '
                       '(calibration and runs made with it are then not comparable).')
        shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__'))
        common.require(common.manifest(target) == files, f'the copy {target} differs from {source}')
        with zipfile.ZipFile(common.ENGINE / f'{name}.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for rel in sorted(files):
                z.write(target / rel, f'pkpy2/{rel}')
        report(f'engine: {len(files)} files of PKPy2 {version} copied to _engine/{name}/pkpy2, zip written')
    record = common.read_json(record_path) or {}
    if record.get('sha256') != digest:
        record = dict(version=version, folder=name, files=files, sha256=digest, packages=package_versions(),
                      python=sys.version.split()[0], frozen=common.now(), pheno_reference_ofv=None)
        common.write_json(record_path, record)
    return record


def engine_record(design=None):
    """The record of the frozen engine: the engine section of a design, or the record of the one frozen copy in
    _engine/ (dict with version, folder, files, sha256, packages, pheno_reference_ofv)."""
    if design and design.get('engine'):
        return design['engine']
    records = sorted(common.ENGINE.glob('pkpy2-*.json'))
    common.require(len(records) == 1, (f'no engine record in {common.ENGINE}; run "calibrate.py engine" first'
                                       if not records else f'more than one engine record in {common.ENGINE}'))
    return common.read_json(records[0])


def design_engine(record):
    """The engine section of design.json."""
    return {k: record.get(k) for k in ('version', 'folder', 'files', 'sha256', 'packages', 'pheno_reference_ofv')}


def engine_parent(engine):
    """The folder that goes first on PYTHONPATH: _engine/<folder> (it holds the package folder pkpy2)."""
    return common.ENGINE / engine['folder']


def child_env(engine, env=None, stand_in=False):
    """Environment of a child process (sections 5 and 6): UTF-8 console output, as the agent prints characters that
    the default console encoding of this machine cannot encode, and the frozen parent folder first on PYTHONPATH
    (fit workers inherit it). engine None (a test design without a frozen engine) leaves PYTHONPATH as it is. A
    child that runs the scripted stand-in gets no key."""
    env = dict(os.environ if env is None else env)
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8:backslashreplace')
    if engine:
        parent = str(engine_parent(engine))
        rest = [p for p in env.get('PYTHONPATH', '').split(os.pathsep) if p and p != parent]
        env['PYTHONPATH'] = os.pathsep.join([parent] + rest)
    if stand_in:
        env.pop('OPENROUTER_API_KEY', None)
    return env


def engine_problems(engine):
    """Problems of the frozen copy of an engine record, found without importing it (empty list: none)."""
    package = engine_parent(engine) / 'pkpy2'
    if not package.is_dir():
        return [f'the frozen engine {package} does not exist']
    files = common.manifest(package)
    if files != engine['files'] or common.manifest_hash(files) != engine['sha256']:
        return [f'the files of {package} differ from the manifest of the design']
    return []


def engine_identity(engine):
    """Where this process finds pkpy2 and whether it is the frozen copy of the record: dict(package, version,
    engine_folder, engine_manifest, packages, packages_ok). engine_folder and engine_manifest are the checks of
    section 4.5. pkpy2 is imported only when it is found in the frozen folder."""
    module = sys.modules.get('pkpy2')
    origin = module.__file__ if module is not None else getattr(importlib.util.find_spec('pkpy2'), 'origin', None)
    package = Path(origin).resolve().parent if origin else None
    frozen = (engine_parent(engine) / 'pkpy2').resolve()
    versions = package_versions()
    out = dict(package=str(package), version=None, engine_folder=package == frozen, engine_manifest=False,
               packages=versions, packages_ok=versions == engine.get('packages'))
    if out['engine_folder']:
        files = common.manifest(package)
        out['engine_manifest'] = files == engine['files'] and common.manifest_hash(files) == engine['sha256']
        import pkpy2
        out['version'] = pkpy2.__version__
        out['engine_folder'] = Path(pkpy2.__file__).resolve().parent == frozen
    return out


def assert_engine(engine):
    """AssertionError unless pkpy2 is imported from the frozen copy and the manifest matches; returns the identity."""
    identity = engine_identity(engine)
    common.require(identity['engine_folder'], f"pkpy2 is found in {identity['package']}, not in the frozen copy "
                                              f"{engine_parent(engine) / 'pkpy2'}")
    common.require(identity['engine_manifest'], f"the files of {identity['package']} differ from the manifest "
                                                f"{engine['sha256'][:8]} of the engine record")
    return identity


def use_engine(engine):
    """Make this process and its children run on the frozen engine: the frozen parent folder goes first on sys.path
    and on PYTHONPATH (spawned fit workers get both). Call it before anything imports pkpy2."""
    parent = str(engine_parent(engine))
    os.environ['PYTHONPATH'] = child_env(engine)['PYTHONPATH']
    sys.path[:] = [parent] + [p for p in sys.path if p != parent]
    return assert_engine(engine)


def reference_check(engine, report=print):
    """Refit the phenobarbital reference model on the frozen engine as benchmarks/reference_fits.py does (one worker
    with 4 threads, two Laplace starts, standard errors) and return dict(ofv, status, seconds, worker). The worker
    entry is the engine identity of the fit worker."""
    common.add_paths()
    import reference_fits
    from pkagent.config import Budget, Settings
    from pkagent.session import Session
    d = common.benchmark_datasets()['pheno']
    out = common.ENGINE / f"{engine['folder']}-check" / 'pheno'
    _clear(out)
    s = Session(common.BENCHMARKS / 'data' / d['file'], out,
                Settings(workers=1, threads_per_worker=4, fit_wall_seconds=3600., laplace_starts=2,
                         budget=Budget(max_fits=10)), d['description'])
    try:
        for col, expr, why in reference_fits.SPECS['pheno']['prepare']:
            s.dataset.add_column(col, expr, why)
            s.data_version += 1
            s._write_data()
        worker = s.pool.submit(engine_identity, engine).result(timeout=600)
        report('engine: phenobarbital reference fit in one worker process (first fit on the frozen copy) ...')
        rec = s.fit([reference_fits.SPECS['pheno']['spec']])[0]
        (out / 'reference_fit.json').write_text(json.dumps(s.compact(rec, detail=True), indent=1, default=str),
                                                encoding='utf-8')
    finally:
        s.close()
    return dict(ofv=rec.get('ofv'), status=rec.get('status'), seconds=(rec.get('summary') or {}).get('seconds'),
                worker=worker)


def step0(report=print):
    """Step 0: freeze, identity in this process, in a child process and in a fit worker, reference fit. Returns the
    engine record; AssertionError when a check fails."""
    record = freeze(report)
    path = common.ENGINE / f"{record['folder']}.json"
    for name, version in record['packages'].items():
        if version != PACKAGES[name]:
            report(f'engine: note, {name} {version} is installed; section 5 names {PACKAGES[name]}')
    here = use_engine(record)
    report(f"engine: this process imports pkpy2 {here['version']} from {here['package']} (manifest "
           f"{record['sha256'][:8]}, {len(record['files'])} files)")
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'identity'], env=child_env(record),
                           capture_output=True, text=True, encoding='utf-8')
    try:
        child_identity = json.loads(child.stdout)
    except ValueError:
        child_identity = dict(engine_folder=False, engine_manifest=False, error=(child.stdout + child.stderr)[-500:])
    child_ok = bool(child.returncode == 0 and child_identity['engine_folder'] and child_identity['engine_manifest'])
    report(f'engine: a child process imports pkpy2 from the frozen copy with the manifest of the record: {child_ok}')
    common.require(child_ok, f'the child process does not run on the frozen engine: {child_identity}')
    record['identity'] = dict(record.get('identity') or {}, process=here, child=child_identity)
    if not (record.get('pheno_reference') or {}).get('passed'):
        check = reference_check(record, report)
        worker = check.pop('worker')
        worker_ok = bool(worker['engine_folder'] and worker['engine_manifest'])
        report(f'engine: the fit worker imports pkpy2 from the frozen copy with the manifest of the record: '
               f'{worker_ok}')
        passed = bool(worker_ok and check['status'] == 'converged' and check['ofv'] is not None
                      and round(check['ofv'], 3) == PHENO_OFV)
        record['identity']['worker'] = worker
        record.update(pheno_reference_ofv=check['ofv'],
                      pheno_reference=dict(check, expected=PHENO_OFV, passed=passed, time=common.now()))
        common.write_json(path, record)
        report(f"engine: phenobarbital reference fit {check['status']}, OFV "
               f"{'%.3f' % check['ofv'] if check['ofv'] is not None else None} (required {PHENO_OFV:.3f}), "
               f"{check['seconds']} s: {'passed' if passed else 'FAILED'}")
        common.require(worker_ok, f'the fit worker does not run on the frozen engine: {worker}')
        common.require(passed, f'the phenobarbital reference fit on the frozen engine does not give OFV '
                               f'{PHENO_OFV:.3f}; the calibration must not start')
    else:
        common.write_json(path, record)
        report(f"engine: the phenobarbital reference fit passed before: OFV {record['pheno_reference_ofv']:.3f}")
    return record


def checked_engine(design=None):
    """The engine record for the commands that fit; AssertionError unless step 0 has passed on it."""
    record = engine_record(design)
    stored = common.read_json(common.ENGINE / f"{record['folder']}.json") or {}
    common.require(stored.get('sha256') == record['sha256'] and (stored.get('pheno_reference') or {}).get('passed'),
                   f"step 0 has not passed on the engine {record['folder']}; run \"calibrate.py engine\" first")
    return stored


# ---------------------------------------------------------------------- reference models
def _term(parameter, covariate):
    """A categorical effect (level 1) that starts at -0.01, as the target coefficient at beta = 0."""
    return dict(parameter=parameter, covariate=covariate, form='categorical', level=1, value=-0.01)


def _plus(spec, name, *terms):
    s = copy.deepcopy(spec)
    s['name'] = name
    s['covariates'] = s['covariates'] + [dict(t) for t in terms]
    return s


def _minus(spec, name, *pairs):
    s = copy.deepcopy(spec)
    s['name'] = name
    s['covariates'] = [c for c in s['covariates'] if (c['parameter'], c['covariate']) not in pairs]
    return s


def search_models(beta):
    """Specifications of steps 1, 2 and 4 by name, for a file with the internal columns CA (effect) and CB (twin):
    M_red, M_full, and M_red with CL~CB, with V~CA, without CL~WT and without V~WT."""
    red, full = generate.reference_models(beta, 'CA')
    return dict(red=red, full=full,
                cl_twin=_plus(red, 'M_red + CL~CB', _term('CL', 'CB')),
                v_target=_plus(red, 'M_red + V~CA', _term('V', 'CA')),
                no_cl_wt=_minus(red, 'M_red without CL~WT', ('CL', 'WT')),
                no_v_wt=_minus(red, 'M_red without V~WT', ('V', 'WT')))


def reference_specs(beta, target, twin):
    """The 15 specifications of step 5 by label, for a final file with the target and twin columns."""
    red, full = generate.reference_models(beta, target)
    true = generate.TRUE

    def fixed(spec, name):
        s = copy.deepcopy(spec)
        s['name'] = name
        for c in s['covariates']:
            if c['covariate'] == 'WT':
                c['fixed'] = True
        return s

    def two(name, four):
        s = copy.deepcopy(full)
        s['name'] = name
        s['structure'] = dict(compartments=2, absorption='iv')
        s['parameters'] = dict(CL=true['CL'], V1=true['V'], Q=2.0, V2=20.0)
        s['iiv'] = dict(CL=true['iiv_CL'], V1=true['iiv_V'], **(dict(Q=IIV_START, V2=IIV_START) if four else {}))
        for c in s['covariates']:
            if c['parameter'] == 'V':
                c['parameter'] = 'V1'
        return s
    weight = (('CL', 'WT'), ('V', 'WT'))
    return dict(
        full=full, red=red,
        no_cl_wt=_minus(full, 'M_full without CL~WT', weight[0]),
        no_v_wt=_minus(full, 'M_full without V~WT', weight[1]),
        cl_twin=_plus(full, f'M_full + CL~{twin}', _term('CL', twin)),
        v_twin=_plus(full, f'M_full + V~{twin}', _term('V', twin)),
        v_target=_plus(full, f'M_full + V~{target}', _term('V', target)),
        cl_sex=_plus(full, 'M_full + CL~SEX', _term('CL', 'SEX')),
        v_sex=_plus(full, 'M_full + V~SEX', _term('V', 'SEX')),
        fixed_full=fixed(full, 'M_full, weight exponents fixed'),
        fixed_red=fixed(red, 'M_red, weight exponents fixed'),
        nowt_full=_minus(full, 'M_full without weight effects', *weight),
        nowt_red=_minus(red, 'M_red without weight effects', *weight),
        cmt2=two('M_full, two compartments', False),
        cmt2_iiv4=two('M_full, two compartments, IIV on CL, V1, Q and V2', True))


# ---------------------------------------------------------------------- fit records
def _start_ofvs(folder):
    """OFV of every Laplace start of a fit, from progress.jsonl of its folder."""
    out = []
    try:
        for line in (Path(folder) / 'progress.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row.get('phase') == 'laplace_exploration' and row.get('ofv') is not None:
                out.append(row['ofv'])
    except (OSError, ValueError, TypeError):
        pass
    return out


def fit_record(seed, beta, model, summary, folder=None, base=None, wall=None):
    """The record of one fit from the summary of pkagent.engine.run_fit."""
    s = summary or {}
    effects = s.get('covariate_effects') or []
    status = s.get('status') or 'error'
    seconds = s.get('seconds')
    late = 'did not finish within the time limit' in str(s.get('error'))
    return dict(
        seed=seed, beta=beta, model=model, status=status, ofv=s.get('ofv'), n_estimated=s.get('n_estimated'),
        seconds=seconds, message=s.get('message'), error=s.get('error'),
        wall_limit=bool(status != 'converged' and (late or (wall and seconds and seconds >= wall))),
        newton_decrement=(s.get('audit') or {}).get('newton_decrement'),
        parameters={r['parameter']: r['estimate'] for r in s.get('parameters') or []},
        rse_percent={r['parameter']: r['rse_percent'] for r in s.get('parameters') or []
                     if r.get('rse_percent') is not None},
        iiv={r['parameter']: r['variance'] for r in s.get('iiv') or []},
        coefficients={r['effect']: r['coefficient'] for r in effects},
        se={r['effect']: r['se'] for r in effects if r.get('se') is not None},
        ci95={r['effect']: r['ci95'] for r in effects if r.get('ci95') is not None},
        residual={r['component']: r['sd'] for r in s.get('residual') or []},
        uncertainty_status=s.get('uncertainty_status'),
        start_ofvs=_start_ofvs(folder) if folder is not None else [],
        folder=None if folder is None else (Path(folder).relative_to(base).as_posix() if base else str(folder)),
        finished=common.now())


def converged(record):
    return bool(record and record.get('status') == 'converged' and record.get('ofv') is not None)


def _ofv(record):
    return record['ofv'] if converged(record) else None


def _status(record):
    """Short text of a fit that did not converge."""
    text = record.get('status') or 'error'
    if record.get('wall_limit'):
        text += ' (wall limit)'
    return text


def _fit_job(job):
    """One fit in a worker process of the pool of steps 1 to 4."""
    from pkagent import engine
    return engine.run_fit(job)


def _init_worker(threads, engine):
    """Initializer of the pool of steps 1 to 4: the worker must run on the frozen engine."""
    common.add_paths()
    if engine is not None:
        assert_engine(engine)
    from pkagent import engine as fit_engine
    fit_engine.init_worker(threads)


# ---------------------------------------------------------------------- hard checks and evidence table (step 5)
def evidence_table(slot, candidate, beta, evaluations, fits, csv_sha256):
    """The record of a dataset in design.json (without a2 and baseline) from the 15 fits of step 5."""
    target, twin = slot['target_column'], slot['twin_column']
    ofv = {label: _ofv(fits[label]) for label in LABELS}

    def diff(a, b):
        return None if ofv[a] is None or ofv[b] is None else round(ofv[a] - ofv[b], 4)
    full = fits['full']
    key = f'CL~{target}=1'
    estimate = full['coefficients'].get(key) if converged(full) else None
    dofv_ref = diff('red', 'full')
    z_ref = None if dofv_ref is None or estimate is None else round(common.signed_root(dofv_ref, estimate), 4)
    parameters = {}
    if converged(full):
        parameters.update(full['parameters'])
        parameters.update({k.split('=')[0]: v for k, v in full['coefficients'].items()})
        parameters.update({f'omega_{k}': v for k, v in full['iiv'].items()})
        parameters.update({f'sigma_{k}': v for k, v in full['residual'].items() if v})
    two, four = fits['cmt2'], fits['cmt2_iiv4']
    return dict(
        id=slot['id'], block=slot['block'], slot_type=slot['slot_type'], stratum=slot['stratum'], zone=slot['zone'],
        evidence_target=slot['evidence_target'], target_column=target, twin_column=twin, seed=candidate['seed'],
        beta=beta, exp_beta=round(math.exp(beta), 4), n_treated=candidate['n_treated'], dofv0=candidate['dofv0'],
        z0=candidate['z0'], dofv_ref=dofv_ref, z_ref=z_ref,
        reference=dict(estimate=estimate, se=full['se'].get(key), ci95=full['ci95'].get(key), parameters=parameters),
        ofv=ofv, dofv_fixed=diff('fixed_red', 'fixed_full'), dofv_nowt=diff('nowt_red', 'nowt_full'),
        evidence={'CL~WT': diff('no_cl_wt', 'full'), 'V~WT': diff('no_v_wt', 'full'),
                  f'CL~{twin}': diff('full', 'cl_twin'), f'V~{twin}': diff('full', 'v_twin'),
                  f'V~{target}': diff('full', 'v_target'), 'CL~SEX': diff('full', 'cl_sex'),
                  'V~SEX': diff('full', 'v_sex')},
        dofv_2cmt=diff('full', 'cmt2'), dofv_2cmt4=diff('full', 'cmt2_iiv4'),
        two_cmt=dict(converged=converged(two), Q=two['parameters'].get('Q') if converged(two) else None,
                     V2=two['parameters'].get('V2') if converged(two) else None,
                     rse_Q=two['rse_percent'].get('Q') if converged(two) else None,
                     rse_V2=two['rse_percent'].get('V2') if converged(two) else None,
                     converged4=converged(four), se_status4=four.get('uncertainty_status')),
        solver_evaluations=evaluations, baseline=None, a2=False, csv_sha256=csv_sha256)


def _side(dofv):
    """0 below 3.84, 1 between 3.84 and 6.63, 2 from 6.63."""
    return int(dofv >= common.INCLUDE) + int(dofv >= common.RETAIN)


def hard_checks(slot, table, fits):
    """The hard checks of step 5 except the convergence of fits 14 and 15 on A2 datasets (a2_roles): a list of
    dict(name, passed, detail). A check whose quantities are missing is failed."""
    out = []

    def add(name, passed, detail):
        out.append(dict(name=name, passed=bool(passed), detail=detail))
    twin, target = slot['twin_column'], slot['target_column']
    bad = [f'{label} {_status(fits[label])}' for label in LABELS[:9] if not converged(fits[label])]
    add('fits 1 to 9 converged', not bad, ', '.join(bad) or 'all converged')
    for label in TWO_STARTS:
        starts = fits[label].get('start_ofvs') or []
        add(f'two starts of {label} agree', len(starts) == 2 and abs(starts[0] - starts[1]) <= START_AGREEMENT,
            'start OFVs ' + ', '.join(f'{v:.4f}' for v in starts) if starts else 'no start OFVs')
    d, z = table['dofv_ref'], table['z_ref']
    rule = (f"dOFV_ref <= {common.E0_DOFV_MAX} and z_ref in {list(slot['z_range'])}" if slot['evidence_target'] is None
            else f"within {slot['tolerance']:g} of {slot['evidence_target']:g}")
    add('dOFV_ref in the tolerance of the slot', d is not None and z is not None and common.in_tolerance(slot, d, z),
        f'dOFV_ref {d}, z_ref {z}; {rule}')
    f = table['dofv_fixed']
    add('dOFV_fixed on the same side of 3.84 and 6.63', d is not None and f is not None and _side(d) == _side(f),
        f'dOFV_fixed {f}, dOFV_ref {d}')
    e = table['evidence']
    for key in ('CL~WT', 'V~WT'):
        add(f'{key} at least {WEIGHT_MIN:g}', e[key] is not None and e[key] >= WEIGHT_MIN, f'{e[key]}')
    add(f'CL~twin below {TWIN_MAX:g}', e[f'CL~{twin}'] is not None and e[f'CL~{twin}'] < TWIN_MAX, f"{e[f'CL~{twin}']}")
    add(f'V~target below {V_TARGET_MAX:g}', e[f'V~{target}'] is not None and e[f'V~{target}'] < V_TARGET_MAX,
        f"{e[f'V~{target}']}")
    for label, key in (('cmt2', 'dofv_2cmt'), ('cmt2_iiv4', 'dofv_2cmt4')):
        if converged(fits[label]):
            add(f'{key} below {CMT2_MAX:g}', table[key] is not None and table[key] < CMT2_MAX, f'{table[key]}')
    return out


# ---------------------------------------------------------------------- the calibration
def scope(smoke=False):
    """What a run covers: the protocol (28 slots, reserve of 8 with 2 seeds for each E0 slot type) or the test run
    of --smoke."""
    slots = common.slots()
    if not smoke:
        return dict(root='.', slots=slots, reserve=RESERVE, reserve_e0=RESERVE_E0, max_seeds=None, datasets=None,
                    mock=False)
    wanted = [s for s in slots if (s['block'], s['slot_type']) in SMOKE['slots']]
    return dict(SMOKE, slots=wanted)


class Calibration:
    """Steps 1 to 5 for the slots of a scope, with all fits cached on disk. The state (candidates, slots, reserve,
    solutions, evidence tables) is rebuilt from the cached fits when the command is started again."""

    def __init__(self, scope, jobs=2, engine=None, report=None):
        self.scope, self.jobs, self.engine = scope, max(1, int(jobs)), engine
        self.f = common.folders(dict(root=scope['root']))
        self.cal = self.f['calibration']
        self.cal.mkdir(parents=True, exist_ok=True)
        self.report = report or Log(self.cal / 'calibrate.log')
        self.slots = list(scope['slots'])
        self.candidates = {}                # seed -> candidate record (CANDIDATE_COLUMNS)
        self.assigned = {}                  # slot ID -> seed
        self.reserve = []                   # accepted seeds without a slot, ascending
        self.solution = {}                  # slot ID -> result of step 4 for the seed of the slot
        self.table = {}                     # slot ID -> evidence table of step 5 for the seed of the slot
        self.checks = {}                    # slot ID -> hard checks
        self.replaced = []                  # replacements: dict(slot, seed, step, reason, by)
        self.next_seed, self.initial = 1, True
        self.count = dict(fitted=0, cached=0, seconds=0.)
        self.lock = threading.Lock()
        self._pool = None

    # ------------------------------------------------------------------ fits of steps 1 to 4 (one process pool)
    @property
    def pool(self):
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.jobs * common.WORKERS, initializer=_init_worker,
                                             initargs=(common.THREADS_PER_WORKER, self.engine))
        return self._pool

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def fit_dir(self, seed, beta):
        return self.cal / 'fits' / f's{seed:05d}' / f'b{beta + 0.:+.4f}'

    def cached(self, seed, beta, model):
        """The record of a finished fit, or None."""
        record = common.read_json(self.fit_dir(seed, beta) / f'{model}.json')
        if record is not None:
            with self.lock:
                self.count['cached'] += 1
        return record

    def store(self, seed, beta, model, summary, folder=None, wall=None):
        record = fit_record(seed, beta, model, summary, folder, self.cal, wall)
        common.write_json(self.fit_dir(seed, beta) / f'{model}.json', record)
        with self.lock:
            self.count['fitted'] += 1
            self.count['seconds'] += record['seconds'] or 0.
        ofv = 'no OFV' if record['ofv'] is None else f"OFV {record['ofv']:.3f}"
        self.report(f"fit   seed {seed} beta {beta:g} {model}: {_status(record)}, {ofv}, {record['seconds']} s")
        return record

    def search_data(self, seed, beta):
        """The data file of a seed and beta with the internal columns, as a Session hands it to the fit workers."""
        folder = self.fit_dir(seed, beta)
        normalized = folder / 'data_v0.csv'
        with self.lock:
            if not normalized.exists():
                from pkagent.data import Dataset
                folder.mkdir(parents=True, exist_ok=True)
                (folder / common.DATA_FILE).write_bytes(generate.csv_bytes(generate.simulate(seed, beta)))
                tmp = folder / 'data_v0.tmp'
                Dataset(folder / common.DATA_FILE, '').write_normalized(tmp)
                os.replace(tmp, normalized)
        return normalized

    def fits(self, seed, beta, names):
        """{name: record} of the models of search_models(beta) on the data of (seed, beta): cached records, and
        pkagent.engine.run_fit in the pool for the others (the agent's fit settings: Laplace, one start, tolerance
        0.5, no standard errors; no diagnostic plots)."""
        out = {name: self.cached(seed, beta, name) for name in names}
        todo = [name for name in names if out[name] is None]
        if not todo:
            return out
        from pkagent.config import Settings
        from pkagent.spec import normalize
        settings = Settings(seed=SEED)
        data_csv, specs, futures = self.search_data(seed, beta), search_models(beta), {}
        wall = settings.fit_wall_seconds
        for name in todo:
            work = self.fit_dir(seed, beta) / name
            summary = common.read_json(work / 'summary.json')
            if summary is not None:                         # finished before the process was stopped
                out[name] = self.store(seed, beta, name, summary, work, wall)
                continue
            _clear(work)
            work.mkdir(parents=True)
            norm = normalize(specs[name], {})
            (work / 'spec.json').write_text(json.dumps(norm, indent=1), encoding='utf-8')
            job = dict(spec=norm, data_csv=str(data_csv), out_dir=str(work), seed=settings.seed, wall_seconds=wall,
                       laplace_seconds=settings.laplace_seconds, uncertainty=False, estimation='laplace',
                       starts=settings.laplace_starts, laplace_tolerance=settings.laplace_tolerance, start_x=None,
                       diagnostics=False)
            futures[self.pool.submit(_fit_job, job)] = (name, work)
        pending, started = set(futures), {}
        while pending:                                      # records are written as the fits finish
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                name, work = futures[future]
                out[name] = self.store(seed, beta, name, future.result(), work, wall)
            now = time.time()
            for future in list(pending):                    # the limit of Session.fit, counted from the start of a fit
                if future.running():
                    started.setdefault(future, now)
                if now - started.get(future, now) > wall * 3 + 600:
                    pending.discard(future)
                    name, work = futures[future]
                    out[name] = self.store(seed, beta, name, dict(
                        status='error', error='fit did not finish within the time limit'), work, wall)
        return out

    # ------------------------------------------------------------------ steps 1 and 2
    def screen(self, seed):
        """Steps 1 and 2 for a candidate seed: the candidate record with the reason."""
        c = dict.fromkeys(CANDIDATE_COLUMNS)
        c.update(seed=seed, step1=False, accepted=False, role='rejected', slot='', fit_seconds=0.)
        try:
            cov = generate.covariates(seed)
        except RuntimeError as e:
            return dict(c, reason=f'generator: {e}')
        c.update(n_treated=cov['n_treated'], draws=cov['draws'])
        first = self.fits(seed, 0., SEARCH)
        c['fit_seconds'] = round(sum(r['seconds'] or 0. for r in first.values()), 1)
        bad = [f'{name} {_status(first[name])}' for name in SEARCH if not converged(first[name])]
        estimate = first['full']['coefficients'].get(EFFECT)
        if bad or estimate is None:
            return dict(c, reason='step 1: ' + (', '.join(bad) or 'no estimate of the target coefficient'))
        red = first['red']['ofv']
        dofv0 = round(red - first['full']['ofv'], 4)
        c.update(ofv_red=red, ofv_full=first['full']['ofv'], estimate0=estimate, dofv0=dofv0,
                 z0=round(common.signed_root(dofv0, estimate), 4))
        if dofv0 > NULL_MAX:
            return dict(c, reason=f'step 1: dOFV0 {dofv0:.3f} above {NULL_MAX}')
        c['step1'] = True
        second = self.fits(seed, 0., STEP2)
        c.update(step2=False, fit_seconds=round(c['fit_seconds'] + sum(r['seconds'] or 0. for r in second.values()), 1))
        bad = [f'{name} {_status(second[name])}' for name in STEP2 if not converged(second[name])]
        if bad:
            return dict(c, reason='step 2: ' + ', '.join(bad))
        c.update(dofv_twin=round(red - second['cl_twin']['ofv'], 4),
                 dofv_v_target=round(red - second['v_target']['ofv'], 4),
                 dofv_cl_wt=round(second['no_cl_wt']['ofv'] - red, 4),
                 dofv_v_wt=round(second['no_v_wt']['ofv'] - red, 4))
        problems = []
        if not c['dofv_twin'] < TWIN_MAX:
            problems.append(f"twin dOFV {c['dofv_twin']:.3f} not below {TWIN_MAX:g}")
        if not c['dofv_v_target'] < V_TARGET_MAX:
            problems.append(f"V~target dOFV {c['dofv_v_target']:.3f} not below {V_TARGET_MAX:g}")
        for key, name in (('dofv_cl_wt', 'CL~WT'), ('dofv_v_wt', 'V~WT')):
            if not c[key] >= WEIGHT_MIN:
                problems.append(f'{name} dOFV {c[key]:.3f} below {WEIGHT_MIN:g}')
        if problems:
            return dict(c, reason='step 2: ' + '; '.join(problems))
        return dict(c, step2=True, accepted=True, role='accepted', reason='accepted')

    # ------------------------------------------------------------------ step 3
    @staticmethod
    def e0_type(z0):
        """E0+ or E0- when z0 is in the range of that slot type, else None."""
        return next((k for k, (lo, hi) in common.E0_Z_RANGE.items() if lo <= z0 <= hi), None)

    def open_slots(self):
        return [s for s in self.slots if s['id'] not in self.assigned]

    def place(self, seed):
        """Step 3 for an accepted seed: the first open E0 slot of its sign, else the first open W or S slot, else
        the reserve."""
        c = self.candidates[seed]
        kind = self.e0_type(c['z0'])
        slot = next((s for s in self.open_slots() if s['slot_type'] == kind), None) if kind else None
        if slot is None:
            slot = next((s for s in self.open_slots() if s['evidence_target'] is not None), None)
        if slot is None:
            self.reserve.append(seed)
            c.update(role='reserve', reason='accepted: reserve (no open slot)')
        else:
            self.assigned[slot['id']] = seed
            c.update(role='dataset', slot=slot['id'],
                     reason=f"accepted: slot {slot['id']} (block {slot['block']}, {slot['slot_type']})")

    def need(self):
        """Lower bound of the accepted seeds still needed: open slots and, in the first pass, the missing reserve
        seeds. Screening stops when it is 0, and never more than this number of seeds is screened at a time, so no
        seed is screened beyond the one that completes the slots and the reserve."""
        short = 0
        if self.initial:
            kinds = [self.e0_type(self.candidates[s]['z0']) for s in self.reserve]
            e0 = sum(max(0, self.scope['reserve_e0'] - kinds.count(k)) for k in common.E0_Z_RANGE)
            short = max(self.scope['reserve'] - len(self.reserve), e0, 0)
        return len(self.open_slots()) + short

    def search(self):
        """Steps 1 to 3: screen candidate seeds in ascending order until the slots are filled (and, in the first
        pass, the reserve holds its seeds)."""
        limit = self.scope['max_seeds']
        while self.need() > 0:
            last = self.next_seed + min(self.need(), BATCH) - 1
            if limit is not None:
                last = min(last, limit)
            if last < self.next_seed:
                self.report(f'steps 1 to 3: the limit of {limit} candidate seeds is reached with '
                            f'{len(self.open_slots())} open slots')
                break
            seeds = list(range(self.next_seed, last + 1))
            self.report(f'steps 1 and 2: candidate seeds {seeds[0]} to {seeds[-1]} '
                        f'({len(self.open_slots())} open slots, {len(self.reserve)} reserve seeds)')
            for c in _parallel(self.screen, seeds, len(seeds)):
                self.candidates[c['seed']] = c
                if c['accepted']:
                    self.place(c['seed'])
                self.report(f"seed  {c['seed']}: dOFV0 {c['dofv0']}, z0 {c['z0']}, twin {c['dofv_twin']}: {c['reason']}")
            self.next_seed = last + 1
            self.write_state()
        self.initial = False

    def replace(self, slot, step, reason):
        """Take the seed of a slot out and give the slot to the first reserve seed that qualifies for it (E0: z0 in
        the range of the slot type; W and S: any reserve seed). Without such a seed the slot stays open and search()
        screens further candidates."""
        old = self.assigned.pop(slot['id'])
        for store in (self.solution, self.table, self.checks):
            store.pop(slot['id'], None)
        c = self.candidates[old]
        c.update(role='replaced', reason=f"{c['reason']}; {step} failed: {reason}")
        kind = slot['slot_type'] if slot['evidence_target'] is None else None
        new = next((s for s in self.reserve if kind is None or self.e0_type(self.candidates[s]['z0']) == kind), None)
        self.replaced.append(dict(slot=slot['id'], seed=old, step=step, reason=reason, by=new))
        if new is None:
            self.report(f"slot  {slot['id']}: seed {old} failed {step} ({reason}); no reserve seed qualifies, "
                        'further candidates are screened')
            return
        self.reserve.remove(new)
        self.assigned[slot['id']] = new
        self.candidates[new].update(role='dataset', slot=slot['id'],
                                    reason=f"accepted: reserve, then slot {slot['id']} in place of seed {old}")
        self.report(f"slot  {slot['id']}: seed {old} failed {step} ({reason}); replaced by reserve seed {new}")

    # ------------------------------------------------------------------ step 4
    def evidence(self, seed, beta):
        """One evaluation of step 4: simulation with beta and fits of M_red and M_full."""
        f = self.fits(seed, beta, SEARCH)
        estimate = f['full']['coefficients'].get(EFFECT)
        if not (converged(f['red']) and converged(f['full'])) or estimate is None:
            return dict(converged=False, detail=', '.join(f'{n} {_status(f[n])}' for n in SEARCH))
        dofv = f['red']['ofv'] - f['full']['ofv']
        return dict(converged=True, dofv=dofv, z=common.signed_root(dofv, estimate), estimate=estimate,
                    seconds=sum(r['seconds'] or 0. for r in f.values()))

    def solve(self, seed, target, z0):
        """Step 4: secant search on the signed root for beta < 0 with |dOFV(beta) - target| within the tolerance.
        Returns dict(ok, beta, evaluations, dofv, z, estimate, points, reason)."""
        root, tolerance = math.sqrt(target), common.tolerance(target)
        points = [(0., z0)]
        beta = round(-(root - z0) * SLOPE, 4)

        def failed(reason, k):
            return dict(ok=False, beta=None, evaluations=k, points=points, reason=reason)
        for k in range(1, MAX_EVALUATIONS + 1):
            if not beta < 0:
                return failed(f'the search left beta < 0 (beta {beta:g})', k - 1)
            if any(b == beta for b, _ in points):
                return failed(f'the search returned to beta {beta:g}', k - 1)
            e = self.evidence(seed, beta)
            if not e['converged']:
                return failed(f"beta {beta:g}: {e['detail']}", k)
            points.append((beta, e['z']))
            self.report(f"solve seed {seed} target {target:g}: evaluation {k}, beta {beta:g}, dOFV {e['dofv']:.3f}, "
                        f"z {e['z']:.3f}")
            if abs(e['dofv'] - target) <= tolerance:
                return dict(ok=True, beta=beta, evaluations=k, dofv=round(e['dofv'], 4), z=round(e['z'], 4),
                            estimate=e['estimate'], points=points, reason='')
            (b1, z1), (b2, z2) = sorted(points, key=lambda p: abs(p[1] - root))[:2]
            if z1 == z2:
                return failed('the two closest points have the same signed root', k)
            beta = round(b2 + (root - z2) * (b1 - b2) / (z1 - z2), 4)
        return failed(f'no beta within the tolerance in {MAX_EVALUATIONS} evaluations', MAX_EVALUATIONS)

    def solve_slot(self, slot):
        seed = self.assigned[slot['id']]
        if slot['evidence_target'] is None:                 # E0 datasets keep beta = 0
            return dict(ok=True, beta=0., evaluations=0, points=[], reason='')
        return self.solve(seed, slot['evidence_target'], self.candidates[seed]['z0'])

    # ------------------------------------------------------------------ step 5 (sessions)
    def session(self, csv_path, folder, starts, fits):
        """A pkagent Session with the settings of the agent's fits, on the frozen engine."""
        from pkagent.config import Budget, Settings
        from pkagent.session import Session
        k = 1
        while (folder / f'session{k}').exists():
            k += 1
        s = Session(csv_path, folder / f'session{k}',
                    Settings(workers=common.WORKERS, threads_per_worker=common.THREADS_PER_WORKER,
                             laplace_starts=starts, seed=SEED, budget=Budget(max_fits=fits, max_hours=1e6)),
                    common.load_strings()['description']['text'])
        if self.engine is not None:
            identity = s.pool.submit(engine_identity, self.engine).result(timeout=600)
            if not (identity['engine_folder'] and identity['engine_manifest']):
                s.close()
                raise AssertionError(f'the fit workers do not run on the frozen engine: {identity}')
        return s

    def reference_fits(self, tag, seed, beta, target, csv_path, labels=LABELS):
        """{label: record} of the fits of step 5 on a final file, through Session.fit: fits 1 and 2 with two
        Laplace starts in their own session, the others with one start in a second session; standard errors for the
        labels of WITH_SE. Records are cached under the model name 'ref.<target column>.<label>'; fits that an
        interrupted session finished are taken from its folder."""
        from pkagent import engine as fit_engine
        twin = common.COLUMNS[1 - common.COLUMNS.index(target)]
        specs = reference_specs(beta, target, twin)
        folder = self.cal / 'step5' / f'{tag}_s{seed:05d}_b{beta + 0.:+.4f}'

        def key(label):
            return f'ref.{target}.{label}'
        out = {label: self.cached(seed, beta, key(label)) for label in labels}
        by_name = {specs[label]['name']: label for label in labels}
        for model in sorted(folder.glob('session*/models/M*')):
            spec, summary = common.read_json(model / 'spec.json'), common.read_json(model / 'summary.json')
            label = by_name.get((spec or {}).get('name'))
            if label is None or summary is None or out[label] is not None:
                continue
            if label in WITH_SE and summary.get('status') == 'converged' and not summary.get('uncertainty_status'):
                continue                                    # stopped before its standard errors: fitted again
            out[label] = self.store(seed, beta, key(label), summary, model)
        for starts, group in ((2, [x for x in labels if x in TWO_STARTS]),
                              (1, [x for x in labels if x not in TWO_STARTS])):
            todo = [label for label in group if out[label] is None]
            if not todo:
                continue
            s = self.session(csv_path, folder, starts, len(todo) + 2)
            try:
                records = s.fit([specs[label] for label in todo], uncertainty=False)
                done = {}
                for label, r in zip(todo, records):
                    summary = r.get('summary') or dict(status=r.get('status') or 'error', error=r.get('error'))
                    model = s.out / 'models' / r['model_id'] if r.get('model_id') else None
                    done[label] = [summary, model, None]
                    if label in WITH_SE and summary.get('status') == 'converged':
                        done[label][2] = s.pool.submit(fit_engine.run_standard_errors, dict(
                            spec=r['spec'], data_csv=s.data_csv(r['data_version']), fit_json=str(model / 'fit.json'),
                            summary_json=str(model / 'summary.json')))
                for label, (summary, model, future) in done.items():
                    if future is not None:
                        with_se = future.result(timeout=3600)
                        if with_se.get('status') == 'converged':
                            summary = with_se
                        else:
                            summary = dict(summary, uncertainty_status=f"failed: {with_se.get('error')}")
                    out[label] = self.store(seed, beta, key(label), summary, model, s.settings.fit_wall_seconds)
            finally:
                s.close()
        return out

    def reference(self, slot):
        """Step 5 for the dataset of a slot: the final file, the 15 fits, the evidence table and the hard checks."""
        seed, sol = self.assigned[slot['id']], self.solution[slot['id']]
        path = self.f['data'] / slot['id'] / common.DATA_FILE
        with self.lock:
            sha = generate.write_dataset(seed, sol['beta'], slot['target_column'], path)
        self.report(f"step 5 {slot['id']} ({slot['slot_type']}, seed {seed}, beta {sol['beta']:g}, "
                    f"{slot['target_column']}): {path.relative_to(self.f['root']).as_posix()} sha256 {sha[:16]}")
        fits = self.reference_fits(slot['id'], seed, sol['beta'], slot['target_column'], path)
        table = evidence_table(slot, self.candidates[seed], sol['beta'], sol['evaluations'], fits, sha)
        checks = hard_checks(slot, table, fits)
        failed = [c for c in checks if not c['passed']]
        self.report(f"step 5 {slot['id']}: dOFV_ref {table['dofv_ref']}, z_ref {table['z_ref']}, dOFV_fixed "
                    f"{table['dofv_fixed']}, dOFV_noWT {table['dofv_nowt']}, 2cmt {table['dofv_2cmt']} / "
                    f"{table['dofv_2cmt4']}; " + ('all hard checks passed' if not failed else 'FAILED: ' + '; '.join(
                        f"{c['name']} ({c['detail']})" for c in failed)))
        return table, checks, sum(r['seconds'] or 0. for r in fits.values())

    def a2_roles(self, done):
        """The A2 roles among the datasets with an evidence table (section 3.7): in every block the S-c dataset and
        one E0 dataset keep the role when fits 14 and 15 converged; else the role passes to the other dataset of
        the same block and stratum if its fits 14 and 15 converged; else the dataset is to be replaced. Returns
        ({slot ID: True}, [(slot that gives the role up, slot that takes it)], [slots to replace])."""
        by_key = {(s['block'], s['slot_type']): s for s in done}

        def both(slot):
            t = self.table[slot['id']]['two_cmt']
            return t['converged'] and t['converged4']
        roles, moved, replace = {}, [], []
        for s in done:
            if not s['a2_default']:
                continue
            other = by_key.get((s['block'], {'E0+': 'E0-', 'E0-': 'E0+', 'S-c': 'S-b'}[s['slot_type']]))
            if both(s):
                roles[s['id']] = True
            elif other is not None and both(other):
                roles[other['id']] = True
                moved.append((s, other))
            else:
                replace.append(s)
        return roles, moved, replace

    # ------------------------------------------------------------------ the whole run
    def run(self):
        """Steps 1 to 5 with the replacements; returns the design (also written to design.json)."""
        self.search()
        while True:
            todo = [s for s in self.slots if s['id'] in self.assigned and s['id'] not in self.solution]
            failed = False
            for slot, r in zip(todo, _parallel(self.solve_slot, todo, len(todo))):
                seed = self.assigned[slot['id']]
                self.candidates[seed].update(beta=r['beta'], evaluations=r['evaluations'])
                if r['ok']:
                    self.solution[slot['id']] = dict(r, seed=seed)
                else:
                    self.replace(slot, 'step 4', r['reason'])
                    failed = True
            self.write_state()
            if failed:
                self.search()
                continue
            done = [s for s in self.slots if s['id'] in self.solution][:self.scope['datasets']]
            todo = [s for s in done if s['id'] not in self.table]
            if todo:
                self.close()                                # the pool of steps 1 to 4 is not needed during step 5
            for slot, (table, checks, seconds) in zip(todo, _parallel(self.reference, todo, self.jobs)):
                self.table[slot['id']], self.checks[slot['id']] = dict(table, fit_seconds=round(seconds, 1)), checks
            for slot in done:
                bad = [c for c in self.checks[slot['id']] if not c['passed']]
                if bad:
                    self.replace(slot, 'step 5', '; '.join(f"{c['name']} ({c['detail']})" for c in bad))
                    failed = True
            if not failed:
                roles, moved, replace = self.a2_roles(done)
                for slot in replace:
                    self.replace(slot, 'step 5', 'A2 dataset: fits 14 and 15 did not both converge, and neither '
                                                 'did those of the other dataset of the block and stratum')
                    failed = True
            self.write_state()
            if not failed:
                break
            self.search()
        self.close()
        for slot, other in moved:
            self.report(f"A2    {slot['id']} ({slot['slot_type']}, block {slot['block']}): fits 14 and 15 did not both "
                        f"converge; the A2 role passes to {other['id']} ({other['slot_type']})")
        design = self.write_design(done, roles)
        self.report(f"fits: {self.count['fitted']} fitted now ({self.count['seconds'] / 60:.1f} fit minutes), "
                    f"{self.count['cached']} records read from the cache")
        return design

    # ------------------------------------------------------------------ outputs
    def write_state(self):
        """calibration/candidates.csv and calibration/search.json."""
        tmp = self.cal / 'candidates.tmp'
        with open(tmp, 'w', encoding='utf-8', newline='') as out:
            w = csv.DictWriter(out, CANDIDATE_COLUMNS, lineterminator='\n')
            w.writeheader()
            w.writerows(self.candidates[s] for s in sorted(self.candidates))
        os.replace(tmp, self.cal / 'candidates.csv')
        common.write_json(self.cal / 'search.json', dict(
            time=common.now(), root=self.scope['root'],
            slots=[dict(s, seed=self.assigned.get(s['id'])) for s in self.slots],
            reserve=self.reserve, candidates=[self.candidates[s] for s in sorted(self.candidates)],
            solutions=self.solution, replaced=self.replaced, checks=self.checks))

    def reserve_records(self):
        keys = ('seed', 'n_treated', 'dofv0', 'z0', 'dofv_twin', 'dofv_v_target', 'dofv_cl_wt', 'dofv_v_wt')
        used = {r['by']: r['slot'] for r in self.replaced if r['by'] is not None}
        return [dict({k: self.candidates[s][k] for k in keys}, qualifies_for=self.e0_type(self.candidates[s]['z0']),
                     used_for=used.get(s)) for s in sorted(set(self.reserve) | set(used))]

    def baseline_of(self, d):
        record = common.read_json(self.fit_dir(d['seed'], d['beta']) / f"baseline.{d['target_column']}.json")
        if not record or record.get('status') != 'completed':
            return None
        return dict(selected=record['selected'], target_selected=record['target_selected'])

    def write_design(self, done, roles):
        """design.json (schema: common.load_design) and calibration/evidence.csv for the datasets with an evidence
        table. A design that does not hold all 28 datasets is marked mock and partial."""
        datasets = []
        for slot in done:
            d = {k: self.table[slot['id']][k] for k in DESIGN_FIELDS}
            d.update(a2=bool(roles.get(slot['id'])), baseline=self.baseline_of(d))
            datasets.append(d)
        with open(self.cal / 'evidence.csv', 'w', encoding='utf-8', newline='') as out:
            w = csv.writer(out, lineterminator='\n')
            w.writerow(['id', 'block', 'slot_type', 'target_column', 'seed', 'beta', 'evidence_target', 'dofv_ref',
                        'z_ref', 'estimate', 'se', 'dofv_fixed', 'dofv_nowt', 'CL~WT', 'V~WT', 'CL~twin', 'V~twin',
                        'V~target', 'CL~SEX', 'V~SEX', 'dofv_2cmt', 'dofv_2cmt4', 'a2', 'solver_evaluations',
                        'fit_seconds'])
            for d in datasets:
                e, t, w2 = d['evidence'], d['target_column'], d['twin_column']
                w.writerow([d['id'], d['block'], d['slot_type'], t, d['seed'], d['beta'], d['evidence_target'],
                            d['dofv_ref'], d['z_ref'], d['reference']['estimate'], d['reference']['se'],
                            d['dofv_fixed'], d['dofv_nowt'], e['CL~WT'], e['V~WT'], e[f'CL~{w2}'], e[f'V~{w2}'],
                            e[f'V~{t}'], e['CL~SEX'], e['V~SEX'], d['dofv_2cmt'], d['dofv_2cmt4'], d['a2'],
                            d['solver_evaluations'], self.table[d['id']]['fit_seconds']])
        complete = len(datasets) == len(common.slots())
        design = dict(protocol_version=common.PROTOCOL_VERSION)
        if self.scope['mock'] or not complete:
            design.update(mock=True, partial=not complete)
        design.update(root=self.scope['root'], generator=generate.CONSTANTS,
                      datasets=sorted(datasets, key=lambda d: d['id']))
        pilot = common.read_json(self.f['pilot'] / 'pilot.json')
        if pilot:
            design['pilot'] = dict(datasets=pilot['datasets'])
        design.update(reserve=self.reserve_records(), strings=common.load_strings(),
                      benchmark_data=common.benchmark_data(),
                      engine=design_engine(self.engine) if self.engine else None)
        design['run_order'] = common.run_order(design)
        path = self.f['root'] / 'design.json'
        common.write_json(path, design)
        self.report(f'design: {len(datasets)} datasets written to {path}' + ('' if complete else ' (partial, mock)'))
        if complete:
            problems = common.check_design(design)
            present, match, hashed = common.check_files(design)
            for p in problems:
                self.report(f'design: MISMATCH {p}')
            self.report(f'design: {len(problems)} problems; data files {present} present, {match} with the sha256 of '
                        f'the design ({hashed} recorded)')
            common.require(not problems and present == match == hashed == len(datasets),
                           'design.json does not pass the checks of common.py')
        return design

    # ------------------------------------------------------------------ pilot datasets (section 3.9)
    def pilot(self):
        """P1 and P2: for each, the first seed from its default seed (and after the seed of the previous dataset)
        that passes steps 1 and 2, step 4 and the tolerance of fits 1 and 2 on the final file."""
        folder, records, log = self.f['pilot'], [], []
        seed = 0
        for p in common.PILOT_DATASETS:
            seed = max(seed + 1, p['seed'])
            while True:
                c = self.screen(seed)
                self.report(f"pilot {p['id']} seed {seed}: dOFV0 {c['dofv0']}, z0 {c['z0']}, twin {c['dofv_twin']}: "
                            f"{c['reason']}")
                record = None
                if c['accepted']:
                    r = self.solve(seed, p['evidence_target'], c['z0'])
                    c.update(beta=r['beta'], evaluations=r['evaluations'])
                    if not r['ok']:
                        c.update(role='rejected', reason=f"accepted; step 4 failed: {r['reason']}")
                    else:
                        record = self.pilot_file(folder, p, seed, r['beta'])
                        ok = (record['dofv_ref'] is not None
                              and abs(record['dofv_ref'] - p['evidence_target']) <= common.tolerance(p['evidence_target'])
                              and record.pop('starts_agree'))
                        c.update(role='dataset' if ok else 'rejected', slot=p['id'] if ok else '',
                                 reason=f"accepted: {p['id']}" if ok else
                                 f"accepted; fits 1 and 2 on the file failed (dOFV_ref {record['dofv_ref']})")
                        record = record if ok else None
                log.append(c)
                if record is not None:
                    records.append(record)
                    break
                seed += 1
        self.close()
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / 'candidates.csv', 'w', encoding='utf-8', newline='') as out:
            w = csv.DictWriter(out, CANDIDATE_COLUMNS, lineterminator='\n')
            w.writeheader()
            w.writerows(log)
        common.write_json(folder / 'pilot.json', dict(time=common.now(), datasets=records))
        path = self.f['root'] / 'design.json'
        design = common.read_json(path)
        if design is not None:
            design['pilot'] = dict(datasets=records)
            common.write_json(path, design)
        self.report(f"pilot: {', '.join(r['id'] + ' seed ' + str(r['seed']) for r in records)} written to {folder}"
                    + ('' if design is None else ' and to the pilot section of design.json'))
        return records

    def pilot_file(self, folder, p, seed, beta):
        path = folder / 'data' / p['id'] / common.DATA_FILE
        with self.lock:
            sha = generate.write_dataset(seed, beta, p['target_column'], path)
        fits = self.reference_fits(p['id'], seed, beta, p['target_column'], path, TWO_STARTS)
        full, red = fits['full'], fits['red']
        estimate = full['coefficients'].get(f"CL~{p['target_column']}=1") if converged(full) else None
        dofv = round(red['ofv'] - full['ofv'], 4) if converged(full) and converged(red) else None
        starts = all(len(f['start_ofvs']) == 2 and abs(f['start_ofvs'][0] - f['start_ofvs'][1]) <= START_AGREEMENT
                     for f in (full, red))
        return dict(id=p['id'], seed=seed, beta=beta, evidence_target=p['evidence_target'],
                    target_column=p['target_column'], twin_column=p['twin_column'], dofv_ref=dofv,
                    z_ref=None if dofv is None or estimate is None else round(common.signed_root(dofv, estimate), 4),
                    csv_sha256=sha, starts_agree=starts)

    # ------------------------------------------------------------------ step 6
    def baseline(self, d):
        """Step 6 for one dataset of the design: covariate_search from M_red with the candidates CL and V effects
        of COMED1, COMED2 and SEX (the agent's thresholds). The result is cached as model 'baseline.<target>'."""
        key = f"baseline.{d['target_column']}"
        record = self.cached(d['seed'], d['beta'], key)
        if record is not None:
            return record
        from pkagent.config import Budget, Settings
        from pkagent.session import Session
        path = self.f['data'] / d['id'] / common.DATA_FILE
        common.require(common.sha256_file(path) == d['csv_sha256'], f'{path} does not have the sha256 of the design')
        folder = self.cal / 'baseline' / f"{d['id']}_s{d['seed']:05d}_b{d['beta'] + 0.:+.4f}"
        k = 1
        while (folder / f'session{k}').exists():
            k += 1
        s = Session(path, folder / f'session{k}',
                    Settings(workers=common.WORKERS, threads_per_worker=common.THREADS_PER_WORKER, laplace_starts=1,
                             seed=SEED, budget=Budget(max_fits=500, max_hours=1e6)),
                    common.load_strings()['description']['text'])
        try:
            if self.engine is not None:
                identity = s.pool.submit(engine_identity, self.engine).result(timeout=600)
                common.require(identity['engine_folder'] and identity['engine_manifest'],
                               f'the fit workers do not run on the frozen engine: {identity}')
            base = s.fit([generate.reference_models(d['beta'], d['target_column'])[0]], uncertainty=False)[0]
            record = dict(seed=d['seed'], beta=d['beta'], model=key, status='base model ' + str(base.get('status')),
                          base_ofv=base.get('ofv'))
            if base.get('status') == 'converged':
                candidates = [dict(parameter=p, covariate=c, form='categorical', level=1)
                              for c in (*common.COLUMNS, 'SEX') for p in ('CL', 'V')]
                result = s.covariate_search(base['model_id'], candidates, forward_p=.05, backward_p=.01)
                selected = [label.split('=')[0] for label in result['included']]
                stopped = [h for h in result['history'] if h.get('step') == 'stopped']
                record.update(status='stopped' if stopped else 'completed', selected=selected,
                              target_selected=f"CL~{d['target_column']}" in selected,
                              final_ofv=s.models[result['final_model_id']].get('ofv'), history=result['history'])
            record.update(fits=len(s.models), finished=common.now(), folder=s.out.relative_to(self.cal).as_posix(),
                          seconds=round(sum((m.get('summary') or {}).get('seconds') or 0. for m in s.models.values()), 1))
        finally:
            s.close()
        common.write_json(self.fit_dir(d['seed'], d['beta']) / f'{key}.json', record)
        self.report(f"step 6 {d['id']}: {record['status']}, selected {record.get('selected')}, CL~target selected "
                    f"{record.get('target_selected')}, {record['fits']} fits, {record['seconds']} fit seconds")
        return record

    def baselines(self):
        """Step 6 for every dataset of design.json; writes the field baseline of the design. It is not started
        while a runner holds the lock of the design: step 6 runs before the agent runs or after B."""
        path = self.f['root'] / 'design.json'
        design = common.load_design(path)
        lock = common.read_json(common.lock_file(design))
        common.require(lock is None or not psutil.pid_exists(lock.get('pid') or -1),
                       f'a runner is active ({common.lock_file(design)}): step 6 is run before the agent runs or '
                       'after B, never while agent runs are in progress')
        records = _parallel(self.baseline, common.datasets(design), self.jobs)
        design = common.load_design(path)
        for d, record in zip(common.datasets(design), records):
            d['baseline'] = (dict(selected=record['selected'], target_selected=record['target_selected'])
                             if record.get('status') == 'completed' else None)
        common.write_json(path, design)
        done = [r for r in records if r.get('status') == 'completed']
        self.report(f"step 6: {len(done)} of {len(records)} datasets completed, CL~target selected in "
                    f"{sum(1 for r in done if r['target_selected'])}; design.json updated")
        return records


# ---------------------------------------------------------------------- step 7
def null_calibration(folder, candidates, report=print):
    """Step 7: dOFV0 of the candidates of step 1 and the twin's dOFV of the candidates of step 2 against the
    chi-square distribution with one degree of freedom. Writes null_calibration.json, .csv and .png to the folder
    and returns the summary."""
    from scipy.stats import chi2, kstest
    values = {'target (dOFV0, candidates of step 1)': [(c['seed'], c['dofv0']) for c in candidates
                                                       if c.get('dofv0') is not None],
              'twin (candidates of step 2)': [(c['seed'], c['dofv_twin']) for c in candidates
                                              if c.get('dofv_twin') is not None]}
    summary, rows = {}, []
    for name, pairs in values.items():
        raw = np.array([v for _, v in pairs], dtype=float)
        x = np.sort(np.clip(raw, 0., None))
        entry = dict(n=int(len(x)), negative=int(np.sum(raw < 0)))
        if len(x) >= 2:
            test = kstest(x, 'chi2', args=(1,))
            entry.update(mean=float(x.mean()), median=float(np.median(x)),
                         fraction_above_2=float(np.mean(x > 2.0)), expected_above_2=float(chi2.sf(2.0, 1)),
                         fraction_above_384=float(np.mean(x > 3.84)), expected_above_384=float(chi2.sf(3.84, 1)),
                         fraction_at_most_05=float(np.mean(x <= 0.5)), expected_at_most_05=float(chi2.cdf(0.5, 1)),
                         ks_statistic=float(test.statistic), ks_p=float(test.pvalue),
                         clear_departure=bool(test.pvalue < 0.01))
            report(f"step 7 {name}: n {entry['n']}, above 2.0 {entry['fraction_above_2']:.3f} (expected "
                   f"{entry['expected_above_2']:.3f}), above 3.84 {entry['fraction_above_384']:.3f} (expected "
                   f"{entry['expected_above_384']:.3f}), Kolmogorov-Smirnov D {entry['ks_statistic']:.3f}, P "
                   f"{entry['ks_p']:.3f}" + (': CLEAR DEPARTURE (P < 0.01), to be examined before the freeze'
                                             if entry['clear_departure'] else ''))
        else:
            report(f"step 7 {name}: {entry['n']} values, no test")
        summary[name] = entry
        rows += [(name, seed, v) for seed, v in pairs]
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    common.write_json(folder / 'null_calibration.json', summary)
    with open(folder / 'null_calibration.csv', 'w', encoding='utf-8', newline='') as out:
        w = csv.writer(out, lineterminator='\n')
        w.writerow(['quantity', 'seed', 'dofv'])
        w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), layout='constrained')
    for ax, (name, pairs) in zip(axes, values.items()):
        x = np.sort(np.clip(np.array([v for _, v in pairs], dtype=float), 0., None))
        if len(x):
            q = chi2.ppf((np.arange(1, len(x) + 1) - 0.5) / len(x), 1)
            top = max(float(q[-1]), float(x[-1]), 4.) * 1.05
            ax.plot([0, top], [0, top], color='0.5', lw=1)
            ax.plot(q, x, 'o', ms=4, color='#0072B2')
            for limit in (2.0, 3.84):
                ax.axhline(limit, color='0.8', lw=.8, ls='--')
            ax.set_xlim(0, top)
            ax.set_ylim(0, top)
        entry = summary[name]
        ax.set_title(f"{name}\nn = {entry['n']}" + (f", KS P = {entry['ks_p']:.2f}" if 'ks_p' in entry else ''),
                     fontsize=9)
        ax.set_xlabel('Quantile of the chi-square distribution (1 df)')
        ax.set_ylabel('Observed dOFV at beta = 0')
    fig.savefig(folder / 'null_calibration.png', dpi=150)
    plt.close(fig)
    return summary


# ---------------------------------------------------------------------- options of section 10.1
def options(root, remove_a2=False, remove_named_controls=False, report=print):
    """Remove module A2 (no dataset keeps the A2 role) or the named controls of B from the design.json of a root,
    as section 10.1 allows at the freeze, and store the run order of the remaining cells. Nothing
    is changed once a run folder of A or B exists. Returns the design."""
    path = Path(root) / 'design.json'
    design = common.read_json(path)
    common.require(design is not None, f'no design.json in {root}')
    if remove_a2 or remove_named_controls:
        started = [c['key'] for c in common.grid(design, 'A') + common.grid(design, 'B')
                   if c['run_dir'].exists() or c['failed_dir'].exists()]
        common.require(not started, f'{len(started)} cells of A or B have a run folder: the design is frozen')
        if remove_a2:
            for d in design['datasets']:
                d['a2'] = False
        if remove_named_controls:
            design['named_controls'] = False
        design['run_order'] = common.run_order(design)
        common.write_json(path, design)
    report(f"design {path}: module A2 {'kept' if any(d.get('a2') for d in design['datasets']) else 'removed'}, "
           f"named controls of B {'kept' if design.get('named_controls', True) else 'removed'}")
    common.count_cells(design, report)
    return design


# ---------------------------------------------------------------------- command line
def main(argv=None):
    ap = argparse.ArgumentParser(description='Calibration of the evidence strength (steps 0 to 7) and pilot datasets.')
    ap.add_argument('command', choices=('engine', 'run', 'null', 'baseline', 'pilot', 'options', 'identity'))
    ap.add_argument('--jobs', type=int, default=2, help='parallel sessions of 3 workers x 2 threads (default 2)')
    ap.add_argument('--smoke', action='store_true', help='test run into smoke/calibration_test/')
    ap.add_argument('--remove-a2', action='store_true', help='options: remove module A2 from the design')
    ap.add_argument('--remove-named-controls', action='store_true', help='options: remove the named arms of B')
    a = ap.parse_args(argv)
    if a.command == 'identity':
        identity = engine_identity(engine_record())
        print(json.dumps(identity))
        return 0 if identity['engine_folder'] and identity['engine_manifest'] else 1
    if a.command == 'engine':
        common.ENGINE.mkdir(parents=True, exist_ok=True)
        step0(Log(common.ENGINE / 'engine.log'))
        return 0
    sc = scope(a.smoke)
    folders = common.folders(dict(root=sc['root']))
    if a.command == 'options':
        options(folders['root'], a.remove_a2, a.remove_named_controls)
        return 0
    if a.command == 'null':
        state = common.read_json(folders['calibration'] / 'search.json')
        common.require(state is not None, f"no search.json in {folders['calibration']}; run the calibration first")
        null_calibration(folders['calibration'], state['candidates'], Log(folders['calibration'] / 'calibrate.log'))
        return 0
    design = common.read_json(folders['root'] / 'design.json') if a.command == 'baseline' else None
    common.require(a.command != 'baseline' or design is not None, f"no design.json in {folders['root']}")
    engine = checked_engine(design)
    use_engine(engine)
    common.add_paths()
    calibration = Calibration(sc, a.jobs, engine)
    calibration.report(f"{a.command}: engine {engine['folder']}, {a.jobs} jobs, output below {folders['root']}")
    try:
        if a.command == 'run':
            calibration.run()
            null_calibration(calibration.cal, [calibration.candidates[s] for s in sorted(calibration.candidates)],
                             calibration.report)
        elif a.command == 'baseline':
            calibration.baselines()
        else:
            calibration.pilot()
    finally:
        calibration.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
