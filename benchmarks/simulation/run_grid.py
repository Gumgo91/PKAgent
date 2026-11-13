"""Runner of the simulation experiment (PROTOCOL.md, sections 3.8, 3.9, 4.5, 6, 8.7, 9 and 10.3): the grids of the
pilot, A (A1 and A2), B and N1, the recognition probe, and stage 0 (replay, dry run, load test).

Every cell is one child process. The child sets the system prompt and the tool list of the cell in pkagent.agent,
installs an LLM subclass that writes served.jsonl, makes the checks before the run, calls pkagent.run with the
settings of section 6, makes the checks after the run and writes condition.json and, as its last action, exit.json.
The parent starts the children in run order, keeps the fee guards, the watchdog and the stall rule, classifies every
attempt (common.acceptance), moves attempts that are not accepted to runs_failed/<same path>/attempt<k>/, reruns
cells as section 8.7 allows, stops after repeated failures and writes the CPU load once a minute.

Usage (run as: PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python benchmarks/simulation/run_grid.py ...):
  --module A [--blocks 1 2] [--reduction 1]    dry run (the default): prints for every cell in run order the prompt
  --module pilot | B | N1 | probe              hash, the tool-list hash and the task message, then the cell counts;
                                               no LLM is called and nothing is written
  --module A --blocks 1 2 --jobs 2 --execute   paid runs: only with --execute
  --module A --status                          state of every cell, fees spent, projection of the remaining cells
  --replay <recorded run folder> [--variant replay|never_finalize|api_error|hang] [--turns N] [--out DIR]
                                               stage 0(a): one cell that replays a recorded benchmark run with the
                                               scripted stand-in of replay_llm.py (no LLM, no fees); output below
                                               smoke/, the replayed OFVs are compared with the recorded ones
  --module pilot --stand-in <recorded run folder> --variant api_error --design smoke/design_mock.json
                                               the cells of a test design with the stand-in in place of the LLM
  --load-test [--load-jobs 1 2] [--load-fits 10]
                                               stage 0(c): reference fits of pilot dataset P1 in 1 and 2
                                               simulated jobs; reports the fits per hour
  --design PATH selects another design file (default design.json). --jobs N: parallel runs (default 2, for module B
  1). With --execute the runner accepts 1 or 2 for A, the pilot and N1 and exactly 1 for B (section 6: the machine
  is shared with other work). The test options --watchdog-seconds, --stall-seconds, --poll-seconds, --fee-limit and
  --max-launches are refused together with --execute.
While a grid runs, <run root>/jobs.txt (one integer) lowers the number of parallel runs and a file <run root>/STOP
stops new launches; the runs in progress finish. Exit code of a grid: 0 all cells complete; 1 cells missing or
open (or a replay that did not reproduce the recorded OFVs); 2 stopped after an HTTP status 401 to 404 of the LLM
client or two consecutive infrastructure failures; 3 paused after three consecutive failures; 4 fee guard, stop
file or test limit; 130 interrupted. After a stop the runner is started again by the operator and continues with
the open cells.

Files written: the run folder of every cell (cell['run_dir']: run.json, transcript.jsonl, tool_log.jsonl,
messages.json, results.json, report.md, models/ from pkagent; condition.json, served.jsonl, exit.json from this
script), <cell parent>/rep<k>.console.log, failed attempts in runs_failed/<same path>/attempt<k>/ (with console.log),
in the run root of the module (common.run_root: runs/, pilot/runs/ or blind/runs/) load.csv and grid_log.jsonl, and
runs_failed/run_grid.lock while a grid runs. B and the pilot also write the copies
blind/data/<source>_blind/pk_data.csv with blind/data/sha256.json; the probe writes blind/runs/probe.jsonl. The key
file is read by pkagent.config.load_env only. The fee guards, the watchdog and stall limits and the failure rules
are constants of common.py; the fees spent are those of code_reports.fees (the transcripts of all runs, the probe and
the calls of the coders); the frozen engine is handled by calibrate.py.
"""
import argparse
import functools
import json
import os
import re
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil

import calibrate
import code_reports
import common
from common import append_json, now, read_json

HERE = Path(__file__).resolve()

ACTIVITY_FILES = ('transcript.jsonl', 'tool_log.jsonl')      # and models/*/progress.jsonl (stall rule, section 6)
MOVE_RETRY_SECONDS = 60                         # the move of a failed attempt is retried for this time
POLL_SECONDS = 5
LOAD_SECONDS = 60                               # interval of load.csv
JOBS = {'A': (1, 2), 'B': (1, 1)}               # parallel runs with --execute, section 6 (pilot and N1 as A)
GRIDS = ('pilot', 'A', 'B', 'N1')
FIT_MODULES = ('A1', 'A2', 'N1', 'pilot')       # cell modules whose data are simulated datasets of the design
EXIT_COMPLETE, EXIT_MISSING, EXIT_STOPPED, EXIT_PAUSED, EXIT_GUARD = 0, 1, 2, 3, 4
# flagged for inspection in tool and user messages (section 4.5); a match does not invalidate the attempt
FLAGS = (('benchmarks', re.compile(r'benchmarks')), ('a dataset ID', re.compile(r'\bD\d\d\b')),
         ('-respect', re.compile(r'-respect')), ('-neutral', re.compile(r'-neutral')),
         ('-criteria', re.compile(r'-criteria')))
LOAD_BATCH = 5                                  # specifications per fit call of a simulated job of the load test
# the reference models of section 5 that a simulated job of the load test fits (labels of calibrate.reference_specs)
LOAD_MODELS = ('red', 'full', 'no_cl_wt', 'no_v_wt', 'cl_twin', 'v_twin', 'v_target', 'cl_sex', 'v_sex', 'cmt2')
PROBE_STAND_IN = ('{"drug": null, "drug_confidence": 0, "source": null, "source_confidence": 0, '
                  '"reason": "scripted stand-in"}')


# ---------------------------------------------------------------------- small helpers
write_json = functools.partial(common.write_json, default=str)      # the records of the runner hold paths


def say(*parts):
    print(time.strftime('%H:%M:%S'), *parts, flush=True)


def git(*args):
    """Output of a read-only git command in the repository, or None if it failed."""
    try:
        r = subprocess.run(['git', *args], cwd=common.REPO, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def utf8_stdout():
    return (getattr(sys.stdout, 'encoding', None) or '').lower().startswith('utf')


def inside(path, folder):
    path, folder = Path(path).resolve(), Path(folder).resolve()
    return path == folder or folder in path.parents


def load_design(a, required=True):
    """(design, path) of --design, else of design.json. When there is none and it is not required (a replay before
    the calibration): (a design with nothing but the engine section of the frozen engine, None), or (None, None)
    when no engine is frozen yet."""
    path = common.design_path(a.design)
    if path.exists():
        return common.load_design(path), path.resolve()
    common.require(not required and not a.design, f'{path} does not exist')
    try:
        return dict(engine=calibrate.design_engine(calibrate.engine_record())), None
    except AssertionError:                                      # no frozen engine yet
        return None, None


# ---------------------------------------------------------------------- frozen engine (section 5, step 0)
def engine_of(design):
    """The engine record of a design (calibrate.py, step 0): the frozen copy of PKPy2 that every child runs on.
    None for a test design (mock) without an engine section and for a replay before any engine is frozen, which
    use the importable PKPy2."""
    engine = (design or {}).get('engine')
    if not engine:
        common.require(design is None or design.get('mock'), 'the design has no engine section (calibrate.py, step 0)')
        return None
    folder = calibrate.engine_parent(engine).resolve()
    common.require(design.get('mock') or inside(folder, common.ENGINE), f'engine folder {folder} is outside _engine/')
    return engine


def engine_checks(design):
    """The two engine checks before a run (section 4.5), as calibrate.engine_identity makes them: pkpy2 is imported
    from the frozen folder, and the sha256 of every file of the package equals the manifest of the design. Returns
    (checks, info, flags)."""
    import pkpy2
    package = Path(pkpy2.__file__).resolve().parent
    info = dict(pkpy2_version=pkpy2.__version__, pkpy2_file=str(package),
                pkpy2_manifest_sha256=common.manifest_hash(common.manifest(package)))
    engine = engine_of(design)
    if engine is None:
        return (dict(engine_folder=True, engine_manifest=True), info,
                ['no frozen engine in this test design: the engine checks were not applied'])
    identity = calibrate.engine_identity(engine)
    return (dict(engine_folder=bool(identity['engine_folder']), engine_manifest=bool(identity['engine_manifest'])),
            info, [])


def engine_problems(design):
    """Problems of the frozen engine seen from the parent, without importing it."""
    engine = engine_of(design)
    return calibrate.engine_problems(engine) if engine else []


def child_env(design, stand_in=False):
    """Environment of a child (section 6): UTF-8 console output and the frozen engine first on PYTHONPATH. A child
    that runs the stand-in gets no key."""
    return calibrate.child_env(engine_of(design), stand_in=stand_in)


# ---------------------------------------------------------------------- cells
def replay_cell(recorded, out):
    """The cell of stage 0(a): the data, description, statement and seed of a recorded benchmark run
    (<dataset>/<condition>/<llm>/rep<k>), with the benchmark prompt, the unchanged tool list and the settings of
    section 6. Output: <out>/runs/<same four levels>/."""
    recorded, out = Path(recorded).resolve(), Path(out).resolve()
    run = read_json(recorded / 'run.json')
    common.require(run is not None and (recorded / 'transcript.jsonl').exists(), f'{recorded} is not a recorded run')
    llm = next((k for k, v in common.MODELS.items() if v == run['model']), None)
    common.require(llm is not None, f"{run['model']} is not an LLM of this experiment")
    rep = int(recorded.name[3:]) if re.fullmatch(r'rep\d+', recorded.name) else int(run['seed']) - common.ENGINE_SEED
    common.require(run['seed'] == common.ENGINE_SEED + rep, 'the recorded seed is not 20261001 + replicate')
    dataset, condition = recorded.parents[2].name, recorded.parents[1].name
    key = f'{dataset}/{condition}/{llm}/rep{rep}'
    return dict(module='replay', dataset=dataset, statement='none' if not run.get('knowledge') else condition,
                instruction='respect', llm=llm, rep=rep, block=None, arm=condition, condition=condition, key=key,
                statement_text=run.get('knowledge'), target_column=None, description=run['description'],
                data_path=Path(run['data']), tools='original', run_root=out / 'runs', run_dir=out / 'runs' / key,
                failed_dir=out / 'runs_failed' / key, order_index=0)


def select_cells(design, a):
    """The cells of --module in run order; A with --blocks and the reduction of section 10.3, N1 without the
    datasets of a block that the reduction removed."""
    if a.module == 'N1':
        return common.reduced(common.grid(design, 'N1'), a.reduction)
    if a.module != 'A':
        return common.grid(design, a.module)
    order = common.order_a(design)
    return common.reduced([c for r in (a.blocks or common.BLOCKS) for c in order[r]], a.reduction)


def console_log(cell):
    return cell['run_dir'].parent / f"rep{cell['rep']}.console.log"


def runner_file(cell):
    return cell['run_dir'].parent / f"rep{cell['rep']}.runner.json"


# ---------------------------------------------------------------------- dry run
def utf8_child(design):
    """Start a child as the grid does and return (stdout encoding it reports, ok)."""
    r = subprocess.run([sys.executable, str(HERE), '--utf8-check'], env=child_env(design), capture_output=True,
                       cwd=str(common.REPO), timeout=120)
    return r.stdout.decode('utf-8', 'replace').strip(), r.returncode == 0


def dry_run(design, cells, a):
    """Print the prompt hash, the tool-list hash and the task message of every cell in run order, and the counts."""
    common.require(utf8_stdout(), 'stdout is not UTF-8: run with PYTHONUTF8=1 PYTHONIOENCODING=utf-8')
    encoding, ok = utf8_child(design)
    print(f"dry run of module {a.module}: {len(cells)} cells in run order; no LLM is called, nothing is written")
    print(f"stdout encoding of a child started as in the grid: {encoding} ({'ok' if ok else 'NOT UTF-8'})")
    prompts, tools, arms = {}, {}, {}
    for i, c in enumerate(cells, 1):
        prompt = common.sha256_text(common.system_prompt(c['instruction']))
        tool = common.tools_sha256(common.tool_list(c))
        prompts.setdefault((c['instruction'], prompt), []).append(c)
        tools.setdefault((c['tools'], tool), []).append(c)
        arms[c['arm']] = arms.get(c['arm'], 0) + 1
        where = f"block {c['block']}, " if c['block'] else ''
        print(f"\n[{i}] {c['key']}  ({c['module']}, {where}order {c['order_index']}, {common.MODELS[c['llm']]})")
        print(f"    data    {c['data_path']}{'' if Path(c['data_path']).exists() else '  (not present)'}")
        print(f"    prompt  {c['instruction']} {prompt}")
        print(f"    tools   {c['tools']} {tool}")
        print('    task message:')
        print('\n'.join('    | ' + line for line in common.task_message(c).splitlines()))
    print('\nsystem prompts:')
    for (name, digest), group in sorted(prompts.items()):
        print(f'  {name:9s} {digest}  {len(group)} cells')
    print('tool lists:')
    for (name, digest), group in sorted(tools.items()):
        print(f'  {name:9s} {digest}  {len(group)} cells')
    print('cells by arm: ' + ', '.join(f'{k} {n}' for k, n in sorted(arms.items())))
    by_llm = {k: sum(1 for c in cells if c['llm'] == k) for k in common.LLMS}
    print(f"selected: {len(cells)} cells ({', '.join(f'{k} {n}' for k, n in by_llm.items())})")
    print('cell counts of the design:')
    wrong = common.count_cells(design, report=lambda line: print('  ' + line))
    print('counts as in the protocol' if not wrong else f'{wrong} counts differ from the protocol')
    return 0 if ok and not wrong else 1


# ---------------------------------------------------------------------- the child process: one run
def served_fields(resp):
    """What the provider says about a response: served model identifier, provider, response id, creation time."""
    extra = getattr(resp, 'model_extra', None) or {}
    return dict(model=getattr(resp, 'model', None), provider=extra.get('provider') or getattr(resp, 'provider', None),
                id=getattr(resp, 'id', None), created=getattr(resp, 'created', None))


def recording_llm(base, run_dir, state):
    """A subclass of the LLM class `base` whose _account() also appends the served model identifier, the provider,
    the response id, its creation time and the wall-clock time to <run>/served.jsonl (section 4.5). Requests are
    unchanged. An error raised by the constructor or by chat() is kept in state['llm_error'], so that the exit
    record can tell an error of the LLM client from an error inside a tool."""
    class RecordingLLM(base):
        def __init__(self, *args, **kwargs):
            try:
                super().__init__(*args, **kwargs)
            except Exception as e:
                state['llm_error'] = e
                raise

        def chat(self, messages, tools):
            try:
                return super().chat(messages, tools)
            except Exception as e:
                state['llm_error'] = e
                raise

        def _account(self, resp):
            super()._account(resp)
            append_json(Path(run_dir) / 'served.jsonl', dict(call=self.usage['calls'], **served_fields(resp),
                                                             time=now(), unix_time=round(time.time(), 3)))
    return RecordingLLM


def _chain(error):
    """An exception and the exceptions it replaced (context and cause)."""
    seen = []
    while error is not None and error not in seen:
        seen.append(error)
        error = error.__cause__ or error.__context__
    return seen


def _plain(text):
    """Lower case with every run of slashes and backslashes as one slash, so that a path is found however it is
    escaped in a message."""
    return re.sub(r'[\\/]+', '/', text).lower()


def message_text(message):
    content = message.get('content')
    if isinstance(content, list):
        return ' '.join(str(p.get('text', '')) for p in content if isinstance(p, dict))
    return content if isinstance(content, str) else ''


def leak_check(messages, run_dir):
    """(leak, flags): leak is true when a message of role tool or user after the task message contains the
    run-folder path; matches of the expressions of FLAGS are listed for inspection only (section 4.5)."""
    needle, leak, flags = _plain(str(Path(run_dir).resolve())), False, []
    for i, m in enumerate(messages[2:], 2):
        if m.get('role') not in ('tool', 'user'):
            continue
        text = message_text(m)
        if needle in _plain(text):
            leak = True
            flags.append(f"message {i} ({m['role']}) contains the run-folder path")
        for name, pattern in FLAGS:
            found = pattern.search(text)
            if found:
                context = text[max(0, found.start() - 40):found.end() + 40].replace('\n', ' ')
                flags.append(f"message {i} ({m['role']}) matches {name}: ...{context}...")
    return leak, flags


def checks_before(cell, design, state, stand_in):
    """Apply the variants of the cell, install the LLM class and make the checks before the run (section 4.5).
    Returns (checks, fields for condition.json, flags)."""
    import pkagent.agent as agent
    strings, flags = common.load_strings(), []
    try:
        common.apply_variants(cell)
    except AssertionError as e:
        flags.append(f'variants: {e}')
    expected = dict(prompt=strings['system']['prompts'][cell['instruction']]['sha256'],
                    tools=strings['tools']['sha256'][cell['tools']])
    observed = dict(prompt=common.sha256_text(agent.SYSTEM), tools=common.tools_sha256(agent.TOOLS))
    agent.LLM = recording_llm(stand_in or agent.LLM, cell['run_dir'], state)
    if stand_in is not None:
        agent.load_env = lambda path=None: None                 # the stand-in needs no key: the key file is not read
    engine, info, engine_flags = engine_checks(design)
    data_path = Path(cell['data_path'])
    data = common.sha256_file(data_path) if data_path.exists() else None
    if cell['module'] == 'replay':
        data_expected = data
        flags.append('replay: the data file is the one named in the recorded run.json')
    else:
        data_expected = common.expected_data_sha256(cell, design)
    import llvmlite
    import numba
    import numpy
    import scipy
    versions = dict(numpy=numpy.__version__, scipy=scipy.__version__, numba=numba.__version__,
                    llvmlite=llvmlite.__version__)
    recorded = ((design or {}).get('engine') or {}).get('packages')
    if recorded and any(recorded.get(k) != v for k, v in versions.items()):
        flags.append(f'package versions {versions} differ from the design {recorded}')
    checks = dict(prompt_sha256=observed['prompt'] == expected['prompt'],
                  tools_sha256=observed['tools'] == expected['tools'],
                  src_tree=git('rev-parse', 'HEAD:src') == common.SRC_TREE,
                  src_clean=git('status', '--porcelain', '--', 'src') == '',
                  engine_folder=engine['engine_folder'], engine_manifest=engine['engine_manifest'],
                  data_sha256=data is not None and data == data_expected, stdout_utf8=utf8_stdout())
    fields = dict(prompt_sha256_expected=expected['prompt'], prompt_sha256_observed=observed['prompt'],
                  tools_sha256=observed['tools'], tools_sha256_expected=expected['tools'], data_sha256=data,
                  data_sha256_expected=data_expected, versions=versions, **info)
    return checks, fields, flags + engine_flags


def checks_after(cell, expected_prompt):
    """The checks after the run on messages.json (section 4.5). Returns (checks, observed prompt sha256, flags)."""
    messages = read_json(cell['run_dir'] / 'messages.json')
    if not isinstance(messages, list) or len(messages) < 2:
        return {k: False for k in common.CHECKS_AFTER}, None, ['messages.json is missing or incomplete']
    system = messages[0].get('content')
    observed = common.sha256_text(system) if isinstance(system, str) else None
    leak, flags = leak_check(messages, cell['run_dir'])
    return (dict(system_message=observed == expected_prompt,
                 task_message=messages[1].get('content') == common.task_message(cell), leak=not leak),
            observed, flags)


def sidecar(cell, design, attempt, settings, a):
    """The fields of condition.json that are known before the checks (section 4.5)."""
    import dataclasses
    simulated = design is not None and cell['module'] in FIT_MODULES and cell['arm'] not in common.B_ARMS
    d = common.dataset(design, cell['dataset']) if simulated else {}
    return dict(
        module=cell['module'], dataset=cell['dataset'], stratum=d.get('stratum'), slot_type=d.get('slot_type'),
        block=cell['block'], order_index=cell['order_index'], generator_seed=d.get('seed'), beta=d.get('beta'),
        dofv_ref=d.get('dofv_ref'), z_ref=d.get('z_ref'), target_column=cell['target_column'],
        twin_column=d.get('twin_column'), statement=cell['statement'], statement_text=cell['statement_text'],
        instruction=cell['instruction'], prompt_sha256_expected=None, prompt_sha256_observed=None, tools_sha256=None,
        description_sha256=common.sha256_text(cell['description']), data_sha256=None, llm=cell['llm'],
        settings=dataclasses.asdict(settings), git_head=git('rev-parse', 'HEAD'),
        git_src_tree=git('rev-parse', 'HEAD:src'), pkpy2_version=None, pkpy2_manifest_sha256=None, versions=None,
        started=now(), ended=None, attempt=attempt, exit=None, checks=None,
        arm=cell['arm'], rep=cell['rep'], tools=cell['tools'], key=cell['key'],
        stand_in=dict(recorded=str(Path(a.replay or a.stand_in).resolve()), variant=a.variant, turns=a.turns,
                      api_status=a.api_status) if (a.replay or a.stand_in) else None)


def child_cell(design, a):
    if a.replay:
        return replay_cell(a.replay, a.out)
    return common.find_cell(design, a.module, a.key)


def child_main(a):
    """One run: checks, pkagent.run, checks, condition.json and exit.json (the last action)."""
    design, _ = load_design(a, required=not a.replay)
    cell = child_cell(design, a)
    run_dir = cell['run_dir']
    scripted = bool(a.replay or a.stand_in)
    common.require(scripted != bool(a.execute), 'a child runs with --execute or with the stand-in, not both')
    common.require(not scripted or inside(run_dir, common.SMOKE), 'stand-in runs are written below smoke/ only')
    run_dir.mkdir(parents=True, exist_ok=True)
    common.require(not any(run_dir.iterdir()), f'{run_dir} is not empty')
    common.add_paths()
    from pkagent.config import Budget, Settings
    settings = Settings(budget=Budget(**common.budget_kwargs(cell)), **common.settings_kwargs(cell))
    stand_in = None
    if scripted:
        import replay_llm
        script = replay_llm.load_script(a.replay or a.stand_in)
        stand_in = replay_llm.stand_in(script, a.variant, a.turns, a.api_status)
    side = sidecar(cell, design, a.attempt, settings, a)
    state, status, error, ran = {}, 'precheck_failed', None, False
    before, after, flags = {k: False for k in common.CHECKS_BEFORE}, {k: False for k in common.CHECKS_AFTER}, []
    print(f"=== {cell['key']} ({cell['module']}), attempt {a.attempt}, {now()}", flush=True)
    try:
        before, fields, flags = checks_before(cell, design, state, stand_in)
        side.update(fields)
        side['checks'] = dict(before=before, after=after, flags=flags)
        write_json(run_dir / 'condition.json', side)
        if all(before.values()):
            import pkagent.agent as agent
            ran, status = True, 'exception'
            agent.run(cell['data_path'], run_dir, cell['description'], cell['statement_text'], settings, None)
            status = 'normal'
        else:
            print('checks before the run failed:', ', '.join(k for k, v in before.items() if not v), flush=True)
    except Exception as e:                                      # noqa: BLE001 - recorded in exit.json
        error = e
        traceback.print_exc()
    llm_error = state.get('llm_error') if state.get('llm_error') in _chain(error) else None
    record = common.exit_record(status, llm_error or error, llm_error is not None)
    if llm_error is not None and llm_error is not error:
        record['message'] = f"{record['message']} | then {type(error).__name__}: {error}"[:2000]
    if ran:
        after, observed, more = checks_after(cell, side['prompt_sha256_expected'])
        side['prompt_sha256_observed'] = observed or side['prompt_sha256_observed']
        flags = flags + more
    else:
        flags = flags + ['the run did not start: the checks after the run were not made']
    side.update(ended=now(), exit=record, checks=dict(before=before, after=after, flags=flags))
    common.require(set(common.CONDITION_FIELDS + common.CONDITION_FIELDS_B) <= set(side),
                   'condition.json: a field is missing')
    write_json(run_dir / 'condition.json', side)
    for line in flags:
        print('flag:', line, flush=True)
    print(f"=== exit: {status}; checks after the run: {after}; {now()}", flush=True)
    write_json(run_dir / 'exit.json', record)
    return 0 if status == 'normal' else 1


# ---------------------------------------------------------------------- processes
def process_of(pid, create_time=None):
    """The psutil process of a pid, or None when it is gone (or, with create_time, when the pid was reused)."""
    if pid is None:
        return None
    try:
        p = psutil.Process(pid)
        if create_time is not None and abs(p.create_time() - create_time) > 0.01:
            return None
        return p if p.is_running() else None
    except psutil.Error:
        return None


def kill(processes, grace=0.):
    """Kill psutil processes that still run (after a grace period); returns how many were killed."""
    processes = [p for p in processes if p is not None]
    if grace:
        psutil.wait_procs(processes, timeout=grace)
    killed = []
    for p in processes:
        try:
            if p.is_running():
                p.kill()
                killed.append(p)
        except psutil.Error:
            pass
    psutil.wait_procs(killed, timeout=10)
    return len(killed)


def remember_tree(item):
    """Add the present descendants of a child to the processes remembered for it (fit workers outlive a child that
    is killed, and they hold files of the run folder)."""
    if item['handle'] is None:
        return
    try:
        found = item['handle'].children(recursive=True)
    except psutil.Error:
        return
    new = [p for p in found if p.pid not in item['tree']]
    for p in new:
        item['tree'][p.pid] = p
    if new:
        write_runner_file(item)


def write_runner_file(item):
    """<cell parent>/rep<k>.runner.json: the processes of the attempt, for a runner that is started after this one
    stopped without ending its children."""
    rows = []
    for p in [item['handle']] + list(item['tree'].values()):
        try:
            rows.append(dict(pid=p.pid, create_time=p.create_time()))
        except (psutil.Error, AttributeError):
            pass
    write_json(runner_file(item['cell']), dict(runner_pid=os.getpid(), key=item['cell']['key'],
                                               attempt=item['attempt'], started=item['started'], processes=rows))


def terminate(item):
    """Kill the process tree of a child (section 6): its descendants, then the child. The child is suspended first,
    so that it cannot start anything while its descendants are killed."""
    remember_tree(item)
    if item['handle'] is not None:
        try:
            item['handle'].suspend()
        except psutil.Error:
            pass
    kill(list(item['tree'].values()))
    kill([item['handle']])
    try:
        item['proc'].wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def activity(run_dir):
    """Sizes of the files whose growth counts as activity (section 6): transcript.jsonl, tool_log.jsonl and every
    models/*/progress.jsonl."""
    paths = [run_dir / n for n in ACTIVITY_FILES] + sorted((run_dir / 'models').glob('*/progress.jsonl'))
    out = []
    for p in paths:
        try:
            out.append((p.name if p.parent == run_dir else p.parent.name, p.stat().st_size))
        except OSError:
            pass
    return tuple(out)


# ---------------------------------------------------------------------- the parent: grid of child processes
def limits(module, a):
    """Watchdog, stall, poll and fee limits of a module (constants of common.py; sections 6 and 10.3); a replay
    spends no fees and has no fee guard (None)."""
    stall = common.STALL_SECONDS['B' if module == 'B' else 'A']
    guard = common.FEE_LIMIT.get(module) if a.fee_limit is None else a.fee_limit
    return dict(watchdog=a.watchdog_seconds or common.WATCHDOG_SECONDS, stall=a.stall_seconds or stall,
                poll=a.poll_seconds or POLL_SECONDS, fees=guard)


def child_command(cell, a, design_path, attempt):
    cmd = [sys.executable, '-u', str(HERE), '--child', '--attempt', str(attempt)]
    if design_path is not None:
        cmd += ['--design', str(design_path)]
    if cell['module'] == 'replay':
        cmd += ['--replay', str(Path(a.replay).resolve()), '--out', str(Path(a.out).resolve())]
    else:
        cmd += ['--module', a.module, '--key', cell['key']]
        if a.stand_in:
            cmd += ['--stand-in', str(Path(a.stand_in).resolve())]
    if a.replay or a.stand_in:
        cmd += ['--variant', a.variant, '--api-status', str(a.api_status)]
        if a.turns is not None:
            cmd += ['--turns', str(a.turns)]
    if a.execute:
        cmd += ['--execute']
    return cmd


def launch(cell, a, design, design_path, attempt):
    """Start the child of a cell; console output goes to <cell parent>/rep<k>.console.log."""
    common.require(not cell['run_dir'].exists(), f"{cell['run_dir']} exists")
    cell['run_dir'].parent.mkdir(parents=True, exist_ok=True)
    log = open(console_log(cell), 'w', encoding='utf-8')
    proc = subprocess.Popen(child_command(cell, a, design_path, attempt), stdout=log, stderr=subprocess.STDOUT,
                            cwd=str(common.REPO), env=child_env(design, bool(a.replay or a.stand_in)),
                            creationflags=getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0))
    item = dict(cell=cell, proc=proc, handle=process_of(proc.pid), log=log, attempt=attempt, t0=time.time(),
                started=now(), last=time.time(), sizes=(), tree={}, killed=None)
    write_runner_file(item)
    return item


def poll(item, lim):
    """True when the child has ended, by itself or because the watchdog or the stall rule terminated it."""
    remember_tree(item)
    if item['proc'].poll() is not None:
        return True
    t, sizes = time.time(), activity(item['cell']['run_dir'])
    if sizes != item['sizes']:
        item['sizes'], item['last'] = sizes, t
    if t - item['t0'] > lim['watchdog']:
        item['killed'] = ('watchdog', f"terminated by the runner after {t - item['t0']:.0f} s")
    elif t - item['last'] > lim['stall']:
        item['killed'] = ('stalled', f"terminated by the runner: no line in transcript.jsonl, tool_log.jsonl or "
                                     f"models/*/progress.jsonl for {t - item['last']:.0f} s")
    else:
        return False
    terminate(item)
    return True


def move_attempt(cell, attempt):
    """Move the run folder of an attempt that was not accepted to runs_failed/<same path>/attempt<k>/ (retried for
    60 s: a process that is ending can still hold a file), with its console log. Nothing is deleted."""
    target = cell['failed_dir'] / f'attempt{attempt}'
    while target.exists():
        attempt += 1
        target = cell['failed_dir'] / f'attempt{attempt}'
    target.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + MOVE_RETRY_SECONDS
    while True:
        try:
            os.rename(cell['run_dir'], target)
            break
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(2)
    for source, name in ((console_log(cell), 'console.log'), (runner_file(cell), 'runner.json')):
        if source.exists():
            os.replace(source, target / name)
    return target


def close_killed(item):
    """After the runner terminated a child: write exit.json with 'watchdog' or 'stalled' and complete
    condition.json, unless the child had already written its exit record."""
    run_dir = item['cell']['run_dir']
    if read_json(run_dir / 'exit.json') is not None or not run_dir.exists():
        return
    record = common.exit_record(item['killed'][0])
    record['message'] = item['killed'][1]
    side = read_json(run_dir / 'condition.json')
    if side is not None:
        side.update(ended=now(), exit=record)
        write_json(run_dir / 'condition.json', side)
    write_json(run_dir / 'exit.json', record)


def scorer_hook(cell, design, a):
    """After an accepted stand-in run (stage 0(a)): a replayed benchmark run is compared with the recorded one
    (<run>/replay_check.json; returns whether it was reproduced), and a run on a simulated dataset of the design is
    scored with score.score_run (<run>/score_check.json). Paid runs are never scored here: score.py is run when all
    grids are closed (section 10.2)."""
    reproduced = None
    if a.replay and a.variant == 'replay' and a.turns is None:
        import replay_llm
        check = replay_llm.compare(a.replay, cell['run_dir'])
        write_json(cell['run_dir'] / 'replay_check.json', check)
        print('replayed against the recorded run:')
        for r in check['models']:
            d = '' if r['difference'] is None else f"{r['difference']:+.5f}"
            print(f"  {r['model_id']}  recorded {r['recorded_status']} {r['recorded_ofv']}  replayed "
                  f"{r['replayed_status']} {r['replayed_ofv']}  difference {d}")
        print(f"  final model: recorded {check['final_model']['recorded']}, replayed "
              f"{check['final_model']['replayed']}; data files identical: {check['data_identical']}; largest "
              f"|difference| {check['max_abs_difference']}; reproduced: {check['reproduced']}", flush=True)
        reproduced = check['reproduced']
    if design is not None and cell['module'] in FIT_MODULES and cell['target_column']:
        try:
            import score
            columns, sentences = score.score_run(cell['run_dir'], design, cell)
            write_json(cell['run_dir'] / 'score_check.json', dict(columns=columns, sentences=sentences))
            shown = {k: columns.get(k) for k in ('Y', 'E', 'twin_Y', 'compartments', 'fits', 'responses')}
            print(f"  score.score_run: {len(columns)} columns, {shown}", flush=True)
        except Exception as e:                                  # noqa: BLE001 - a test of the scorer, reported
            print(f'  score.score_run failed: {type(e).__name__}: {e}', flush=True)
    return reproduced


def served_change(design, cell):
    """Section 10.2: the experiment stops when served.jsonl shows that an LLM identifier is served by another model.
    Returns a text when the accepted run of a cell holds a served model identifier that no other run of the same
    LLM holds (among the runs of A, B and N1; for a pilot run, among those of the pilot) or, as the first run of
    its LLM, more than one identifier; else None. After such a stop the operator decides and restarts the runner,
    which then knows the identifier."""
    own = {model for model, _ in common.served_models(cell['run_dir'])}
    known = set()
    for module in (('pilot',) if cell['module'] == 'pilot' else ('A', 'B', 'N1')):
        for c in common.grid(design, module):
            if c['llm'] == cell['llm'] and c['run_dir'] != cell['run_dir']:
                known |= {model for model, _ in common.served_models(c['run_dir'])}
    if known and own - known:
        return (f"{cell['key']} was served {sorted(own - known)} for {cell['llm']}; the other runs were served "
                f"{sorted(known)}")
    if not known and len(own) > 1:
        return f"{cell['key']} was served more than one model identifier for {cell['llm']}: {sorted(own)}"
    return None


def finish(item, a, state, queue, log_path):
    """Classify the attempt of a child that has ended (section 8.7), move it if it was not accepted, decide on a
    rerun and update the counters of consecutive failures."""
    cell = item['cell']
    killed = kill(list(item['tree'].values()), grace=0. if item['killed'] else 3.)
    item['log'].close()
    if item['killed']:
        close_killed(item)
    result = common.acceptance(cell['run_dir'])
    hours = (time.time() - item['t0']) / 3600
    event = dict(time=now(), event='finished', key=cell['key'], attempt=item['attempt'], accepted=result['accepted'],
                 failure_class=result['failure_class'], reason=result['reason'], hours=round(hours, 3),
                 returncode=item['proc'].returncode, orphans_killed=killed)
    if result['accepted']:
        state['consecutive'] = state['infrastructure'] = 0
        runner_file(cell).unlink(missing_ok=True)
        say('finished', cell['key'], f"attempt {item['attempt']}, {hours:.2f} h")
        append_json(log_path, event)
        change = served_change(state['design'], cell) if cell['module'] != 'replay' else None
        if change:
            state['stop'] = (EXIT_STOPPED, 'served model identifier (section 10.2): ' + change)
        if a.replay or a.stand_in:
            if scorer_hook(cell, state['design'], a) is False:
                state['not_reproduced'] = True
        return
    record = read_json(cell['run_dir'] / 'exit.json') or {}
    target = move_attempt(cell, item['attempt'])
    cls = result['failure_class']
    state['consecutive'] += 1
    state['infrastructure'] = state['infrastructure'] + 1 if cls == 'a' else 0
    status = common.cell_status(cell)
    if status['state'] == 'open':
        queue.insert(0, cell)                                   # the cell is run again next
    say(f"FAILED ({cls}: {result['reason']})", cell['key'], f"attempt {item['attempt']}, {hours:.2f} h,",
        f"moved to {target};", 'rerun' if status['state'] == 'open' else 'no rerun left: the cell is missing',
        f'({killed} orphan processes ended)' if killed else '')
    append_json(log_path, dict(event, moved_to=str(target), cell_state=status['state'],
                               http_status=record.get('http_status')))
    if cls == 'a' and record.get('http_status') in common.FATAL_HTTP:
        state['stop'] = (EXIT_STOPPED, f"HTTP {record['http_status']} from the LLM client")
    elif state['infrastructure'] >= common.MAX_INFRASTRUCTURE:
        state['stop'] = (EXIT_STOPPED, f'{common.MAX_INFRASTRUCTURE} consecutive infrastructure failures')
    elif state['consecutive'] >= common.MAX_CONSECUTIVE:
        state['stop'] = (EXIT_PAUSED, f'{common.MAX_CONSECUTIVE} consecutive failures')


def recover(cells, log_path):
    """Before the first launch: end processes that an earlier runner left behind and move every run folder that
    does not hold an accepted attempt (interrupted, failed or invalid) to runs_failed/."""
    for cell in cells:
        left = read_json(runner_file(cell))
        if left is not None:
            n = kill([process_of(p['pid'], p['create_time']) for p in left.get('processes', [])])
            if n:
                say(f"ended {n} processes left by an earlier runner for {cell['key']}")
        run_dir = cell['run_dir']
        if not run_dir.exists():
            runner_file(cell).unlink(missing_ok=True)
            continue
        if not any(run_dir.iterdir()):
            run_dir.rmdir()
            runner_file(cell).unlink(missing_ok=True)
            continue
        result = common.acceptance(run_dir)
        if result['accepted']:
            runner_file(cell).unlink(missing_ok=True)
            continue
        attempt = result['attempt'] or (left or {}).get('attempt') or common.cell_status(cell)['next_attempt'] - 1
        target = move_attempt(cell, attempt)
        say(f"earlier attempt of {cell['key']} was not accepted ({result['failure_class']}: {result['reason']}); "
            f'moved to {target}')
        append_json(log_path, dict(time=now(), event='recovered', key=cell['key'], attempt=attempt,
                                   failure_class=result['failure_class'], reason=result['reason'],
                                   moved_to=str(target)))


def take_lock(path):
    """One runner at a time per design (the lock is runs_failed/run_grid.lock for every module): a second one would
    move the folders of runs in progress and share the processor with them."""
    old = read_json(path)
    if old is not None and process_of(old.get('pid'), old.get('create_time')) is not None:
        raise SystemExit(f"another runner (pid {old['pid']}, started {old.get('started')}) is running: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(pid=os.getpid(), create_time=psutil.Process().create_time(), started=now()))


def write_load(path, module, active, spent):
    """One line of load.csv: mean CPU load of the machine since the last line, memory, runs and processes."""
    new = not path.exists()
    row = dict(time=now(), module=module, cpu_percent=f'{psutil.cpu_percent(None):.1f}',
               memory_percent=f'{psutil.virtual_memory().percent:.1f}', runs=len(active),
               processes=sum(1 + len(i['tree']) for i in active), fees_usd=f'{spent:.4f}')
    with open(path, 'a', encoding='utf-8', newline='\n') as f:
        if new:
            f.write(','.join(common.LOAD_COLUMNS) + '\n')
        f.write(','.join(str(row[k]) for k in common.LOAD_COLUMNS) + '\n')


def default_jobs(module):
    """--jobs when it is not given: the number of section 6, 2 for A, the pilot and N1 and 1 for B."""
    return JOBS['B' if module == 'B' else 'A'][1]


def jobs_problem(module, jobs):
    """The rule of section 6 for paid runs: 1 or 2 parallel runs for A, the pilot and N1, exactly 1 for B. Returns
    the problem as a text, or None."""
    low, high = JOBS['B' if module == 'B' else 'A']
    if low <= jobs <= high:
        return None
    allowed = str(low) if low == high else f'{low} or {high}'
    return f'--jobs must be {allowed} for module {module} (section 6)'


def allowed_jobs(a, run_root):
    """--jobs, lowered by <run root>/jobs.txt if that file holds a smaller positive integer."""
    try:
        return max(1, min(a.jobs, int((run_root / 'jobs.txt').read_text().strip())))
    except (OSError, ValueError):
        return a.jobs


def run_cells(design, design_path, cells, a, module):
    """Run the open cells in order with a.jobs children in parallel. Returns the exit code: 0 all cells complete,
    1 cells missing, 2 stopped (infrastructure), 3 paused (three consecutive failures), 4 fee guard or stop file."""
    run_root, lim = cells[0]['run_root'], limits(module, a)
    run_root.mkdir(parents=True, exist_ok=True)
    lock = run_root.parent / 'runs_failed' / common.LOCK_FILE if module == 'replay' else common.lock_file(design)
    log_path = run_root / common.GRID_LOG
    take_lock(lock)
    state = dict(consecutive=0, infrastructure=0, stop=None, design=design, launched=0)
    active, queue = [], []

    def spent_now():
        return 0. if module == 'replay' else code_reports.fees(design)['total']
    try:
        recover(cells, log_path)
        states = {c['key']: common.cell_status(c)['state'] for c in cells}
        queue = [c for c in cells if states[c['key']] == 'open']
        spent = spent_now()
        guard = 'no fee guard' if lim['fees'] is None else f"no launch at {lim['fees']:g} USD or more"
        say(f"module {module}: {len(cells)} cells, {sum(s == 'complete' for s in states.values())} complete, "
            f"{sum(s == 'missing' for s in states.values())} missing, {len(queue)} to run; {a.jobs} jobs; "
            f"fees spent {spent:.2f} USD ({guard})")
        append_json(log_path, dict(time=now(), event='start', module=module, cells=len(cells), to_run=len(queue),
                                   jobs=a.jobs, fees_usd=round(spent, 4), limits=lim, execute=bool(a.execute)))
        psutil.cpu_percent(None)
        next_load = time.time() + LOAD_SECONDS
        while active or (queue and not state['stop']):
            while queue and not state['stop'] and len(active) < allowed_jobs(a, run_root):
                spent = spent_now()
                if lim['fees'] is not None and spent >= lim['fees']:
                    state['stop'] = (EXIT_GUARD, f"fee guard: {spent:.2f} USD spent, limit {lim['fees']:g} USD")
                elif (run_root / 'STOP').exists():
                    state['stop'] = (EXIT_GUARD, f"stop file {run_root / 'STOP'}")
                elif a.max_launches is not None and state['launched'] >= a.max_launches:
                    state['stop'] = (EXIT_GUARD, f'test limit of {a.max_launches} launches')
                else:
                    cell = queue.pop(0)
                    attempt = common.cell_status(cell)['next_attempt']
                    state['launched'] += 1
                    active.append(launch(cell, a, design, design_path, attempt))
                    say('started', cell['key'], f"attempt {attempt}, pid {active[-1]['proc'].pid}")
                    append_json(log_path, dict(time=now(), event='started', key=cell['key'], attempt=attempt,
                                               pid=active[-1]['proc'].pid, fees_usd=round(spent, 4)))
            time.sleep(lim['poll'])
            for item in list(active):
                if poll(item, lim):
                    active.remove(item)
                    finish(item, a, state, queue, log_path)
            if state['stop'] and not state.get('announced'):
                state['announced'] = True
                say(f"no further launches: {state['stop'][1]}"
                    + (f'; {len(active)} runs in progress will finish' if active else ''))
                append_json(log_path, dict(time=now(), event='stop', reason=state['stop'][1], open=len(queue)))
            if time.time() >= next_load:
                write_load(run_root / common.LOAD_FILE, module, active, spent_now())
                next_load += LOAD_SECONDS
    except KeyboardInterrupt:
        for item in active:
            terminate(item)
            item['log'].close()
        say(f'interrupted: {len(active)} runs in progress were terminated; they are moved at the next start')
        append_json(log_path, dict(time=now(), event='interrupted', keys=[i['cell']['key'] for i in active]))
        return 130
    finally:
        lock.unlink(missing_ok=True)
    states = {c['key']: common.cell_status(c)['state'] for c in cells}
    n = {s: sum(v == s for v in states.values()) for s in ('complete', 'missing', 'open')}
    code = state['stop'][0] if state['stop'] else EXIT_MISSING if n['missing'] or n['open'] else EXIT_COMPLETE
    if state.get('not_reproduced') and code == EXIT_COMPLETE:
        code = EXIT_MISSING
    reason = state['stop'][1] if state['stop'] else 'all cells were processed'
    say(('STOPPED' if state['stop'] else 'done') + f": {reason}; {n['complete']} complete, {n['missing']} missing, "
        f"{n['open']} open" + ('; restart the runner to continue' if state['stop'] else ''))
    append_json(log_path, dict(time=now(), event='end', code=code, reason=reason, **n))
    return code


def closed(design, module):
    """Whether every cell of a grid is complete or missing."""
    return all(common.cell_status(c)['state'] != 'open' for c in common.grid(design, module))


def preflight(design, cells, a, module):
    """Problems that forbid the start of a grid (empty list: none)."""
    problems = []
    scripted = bool(a.replay or a.stand_in)
    if not utf8_stdout():
        problems.append('stdout is not UTF-8: run with PYTHONUTF8=1 PYTHONIOENCODING=utf-8')
    encoding, ok = utf8_child(design)
    if not ok:
        problems.append(f'the stdout of a child is {encoding}, not UTF-8')
    if common.check_strings(report=lambda line: None):
        problems.append('strings.json does not match pkagent (python common.py --check)')
    if git('rev-parse', 'HEAD:src') != common.SRC_TREE or git('status', '--porcelain', '--', 'src') != '':
        problems.append(f'the source tree src/ is not the clean tree {common.SRC_TREE}')
    if design is not None:
        problems += engine_problems(design)
        if module != 'replay':
            problems += common.check_design(design)
    if a.execute:
        tests = (a.watchdog_seconds, a.stall_seconds, a.poll_seconds, a.fee_limit, a.max_launches)
        if scripted or any(v is not None for v in tests):
            problems.append('--execute cannot be combined with the stand-in or with test options')
        if design is None or design.get('mock') or not design.get('engine'):
            problems.append('--execute needs design.json with the frozen engine, not a test design')
        if jobs_problem(module, a.jobs):
            problems.append(jobs_problem(module, a.jobs))
        if module in ('N1', 'probe') and not closed(design, 'B'):
            problems.append(f'{module} is run when B is closed (sections 9 and 10.3)')
        if module == 'N1' and code_reports.fees(design)['total'] > common.N1_START_LIMIT:
            problems.append(f'N1 is run only if the fees spent are at most {common.N1_START_LIMIT:g} USD '
                            '(section 10.3)')
    elif module != 'replay' and not (design or {}).get('mock'):
        problems.append('the stand-in runs the cells of a test design only (mock: true)')
    if scripted and not all(inside(c['run_dir'], common.SMOKE) for c in cells):
        problems.append('stand-in runs are written below smoke/ only')
    if any(c['arm'] in common.B_ARMS for c in cells):
        problems += common.prepare_blind(design, write=True)
    for path in sorted({Path(c['data_path']) for c in cells}):
        cell = next(c for c in cells if Path(c['data_path']) == path)
        if not path.exists():
            problems.append(f'{path} does not exist')
        elif module != 'replay' and common.sha256_file(path) != common.expected_data_sha256(cell, design):
            problems.append(f'{path} does not have the sha256 of the design')
    return problems


def status(design, cells, a):
    """State of every selected cell, fees spent, the served model identifiers of the accepted runs and, for the
    reduction rule of section 10.3, the fees and hours of the open cells projected from the accepted runs. No
    outcome of a run is read (section 10.2)."""
    rows, cost, hours, served = {}, [], [], {}
    for c in cells:
        s = common.cell_status(c)
        rows.setdefault(s['state'], []).append(c)
        failures = [f"attempt {t['attempt']}: {t['failure_class']} ({t['reason']})" for t in s['attempts']
                    if not t['accepted']]
        if s['state'] != 'open' or failures:
            print(f"{s['state']:8s} {c['key']}" + ('  ' + '; '.join(failures) if failures else ''))
        if s['state'] == 'complete':
            run = (read_json(c['run_dir'] / 'results.json') or {}).get('run') or {}
            cost.append(float(run.get('cost_usd') or 0.))
            hours.append(float(run.get('hours') or 0.))
            try:
                lines = (c['run_dir'] / 'served.jsonl').read_text(encoding='utf-8').splitlines()
            except OSError:
                lines = []
            for line in lines:
                r = json.loads(line)
                key = (c['llm'], str(r.get('model')), str(r.get('provider')))
                served[key] = served.get(key, 0) + 1
    n = {k: len(rows.get(k, [])) for k in ('complete', 'missing', 'open')}
    spent = code_reports.fees(design)
    print(f"module {a.module}: {len(cells)} cells, {n['complete']} complete, {n['missing']} missing, {n['open']} open")
    print(f"fees spent: {spent['total']:.2f} USD ({spent['n_transcripts']} transcripts {spent['transcripts']:.2f}, "
          f"probe {spent['probe']:.2f}, coding {spent['coding']:.2f}); no new run of {a.module} at "
          f"{common.FEE_LIMIT[a.module]:g} USD or more")
    for (llm, model, provider), responses in sorted(served.items()):
        print(f'served to {llm}: {model} by {provider}, {responses} responses')
    for llm in common.LLMS:
        models = sorted({model for (k, model, _) in served if k == llm})
        if len(models) > 1:
            print(f'NOTE: {llm} was served {len(models)} model identifiers ({models}); see section 10.2')
    if cost:
        mean_cost, mean_hours = sum(cost) / len(cost), sum(hours) / len(hours)
        print(f"accepted runs: mean {mean_cost:.3f} USD and {mean_hours:.2f} h per run; projection for the "
              f"{n['open']} open cells: {n['open'] * mean_cost:.2f} USD and {n['open'] * mean_hours:.1f} run-hours, "
              f"total fees {spent['total'] + n['open'] * mean_cost:.2f} USD")
        if a.module == 'A':
            print(f"reduction rule of section 10.3: applied if the projected fees of the pilot and A exceed "
                  f"{common.FEE_LIMIT['A']:g} USD (--reduction 1: without block 4; 2: also without the deny-neutral "
                  'cells of block 3)')
    if a.module == 'pilot':
        pilot_go(design, rows.get('complete', []), len(cells), served, spent['total'])
    return 0


def pilot_go(design, complete, n_cells, served, spent):
    """The go criteria of the pilot (section 3.9), which rest on technical quantities only: runs finalized, checks
    of section 4.5, attempts that the leak check invalidated, fits used and limits reached (the process columns of
    score.py; no outcome is read), served model identifiers, and the projected fees and run-hours of A."""
    import score
    rows, flagged = [], 0
    for c in complete:
        run_dir = c['run_dir']
        results, condition = read_json(run_dir / 'results.json'), read_json(run_dir / 'condition.json') or {}
        p = score.process(run_dir, results, condition, score.registry(run_dir, results),
                          common.read_jsonl(run_dir / 'tool_log.jsonl'), None)
        rows.append(dict(p, simulated=bool(c['target_column'])))
        flagged += bool((condition.get('checks') or {}).get('flags'))
    leaks = [t for c in common.grid(design, 'pilot') for t in common.cell_status(c)['attempts']
             if 'leak' in t['reason']]
    fits = [r['fits'] or 0 for r in rows]
    simulated = [r for r in rows if r['simulated']]
    n_a = len(common.grid(design, 'A'))
    print('go criteria of the pilot (section 3.9):')
    print(f"  runs finalized with all checks of section 4.5 passed: {len(complete)} of {n_cells}")
    print(f'  attempts invalidated by the leak check: {len(leaks)} (each is inspected: a false alarm fails the '
          f'criterion); accepted runs with flags for inspection: {flagged}')
    print(f"  largest number of fits: {max(fits, default=None)} (criterion: no run above 32); runs at a limit: "
          f"{sum(r['limit_reached'] for r in rows)}")
    for llm in common.LLMS:
        models = sorted({model for (k, model, _) in served if k == llm})
        print(f"  served model identifiers of {llm}: {len(models)} {models} (criterion: one)")
    if simulated:
        cost = sum(r['cost_usd'] or 0. for r in simulated) / len(simulated)
        hours = sum(r['hours'] or 0. for r in simulated) / len(simulated)
        total = spent + n_a * cost + common.FEE_RESERVE
        print(f"  projection for the {n_a} runs of A from the {len(simulated)} pilot runs on simulated data: "
              f"{n_a * cost:.2f} USD and {n_a * hours:.0f} run-hours; with the {spent:.2f} USD spent and the reserve "
              f"of {common.FEE_RESERVE:g} USD for B and the probe: {total:.2f} USD (criterion: at most 95)")


# ---------------------------------------------------------------------- recognition probe (section 9)
def probe_calls(design):
    """The calls of the probe in listed order (LLM, input, call): dicts with llm, input, call, message and
    message_sha256 (message None when the data file of the input does not exist)."""
    common.add_paths()
    from pkagent.data import Dataset
    from pkagent.tools import dumps
    probe = common.load_strings()['probe']
    common.require(common.sha256_text(probe['template']) == probe['sha256'],
                   'the probe template differs from its sha256')
    messages = {}
    for item in common.probe_inputs(design):
        if Path(item['data_path']).exists():
            summary = dumps(Dataset(item['data_path'], item['description']).describe())
            messages[item['name']] = probe['template'].format(description=item['description'], summary=summary)
        else:
            messages[item['name']] = None
    return [dict(llm=llm, input=name, call=k, message=m, message_sha256=common.sha256_text(m) if m else None)
            for llm in common.LLMS for name, m in messages.items() for k in range(1, common.PROBE_CALLS + 1)]


def probe_request(llm, message):
    """One call without tools: the request of LLM.chat without the tool fields, with its retry rule."""
    body = dict(model=llm.model, messages=[dict(role='user', content=message)], max_tokens=llm.max_output_tokens)
    if llm.temperature is not None:
        body['temperature'] = llm.temperature
    extra = dict(usage=dict(include=True))
    if llm.reasoning_effort:
        extra['reasoning'] = dict(effort=llm.reasoning_effort)
    delay, last = 5., None
    for attempt in range(8):
        try:
            resp = llm.client.chat.completions.create(**body, extra_body=extra, timeout=900)
            if not resp.choices:
                raise RuntimeError(f'empty response: {getattr(resp, "error", None)}')
            return resp
        except Exception as e:                                  # noqa: BLE001 - retried, then raised
            last = e
            if getattr(e, 'status_code', None) in (400, 401, 402, 403, 404) and attempt >= 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 120.)
    raise RuntimeError(f'LLM request failed after retries: {last}')


def probe_main(design, a):
    """Dry run, or the 50 single calls of the probe; answers go to blind/runs/probe.jsonl, one line per call. A
    call that is already answered there is not repeated."""
    calls = probe_calls(design)
    path = common.probe_file(design)
    done = {(line['llm'], line['input'], line['call']) for line in common.read_jsonl(path) if not line.get('error')}
    if not (a.execute or a.stand_in):
        common.require(utf8_stdout(), 'stdout is not UTF-8: run with PYTHONUTF8=1 PYTHONIOENCODING=utf-8')
        print(f'dry run of the probe: {len(calls)} single calls without tools; no LLM is called, nothing is written')
        for name in dict.fromkeys(c['input'] for c in calls):
            c = next(x for x in calls if x['input'] == name)
            print(f"\ninput {name}: message sha256 {c['message_sha256']}")
            text = c['message'] or '(the data file does not exist)'
            print('\n'.join('    | ' + line for line in text.splitlines()))
        print('\ncalls: ' + ', '.join(f"{llm} {sum(1 for c in calls if c['llm'] == llm)}" for llm in common.LLMS)
              + f"; {len(done)} already answered in {path}")
        expected = len(common.LLMS) * 5 * common.PROBE_CALLS
        print(f"{'counts as in the protocol' if len(calls) == expected == 50 else 'counts differ from the protocol'}")
        return 0 if len(calls) == expected else 1
    problems = preflight(design, [], a, 'probe') + common.prepare_blind(design, write=True)
    problems += [f"input {c['input']}: the data file does not exist" for c in calls if c['message'] is None][:5]
    if problems:
        print('\n'.join('problem: ' + p for p in problems))
        return 1
    common.add_paths()
    if a.stand_in:
        import replay_llm
        base = replay_llm.stand_in([], 'replay', turns=0, idle=PROBE_STAND_IN)
    else:
        from pkagent.config import load_env
        from pkagent.llm import LLM as base
        load_env()
    path.parent.mkdir(parents=True, exist_ok=True)
    limit = common.FEE_LIMIT['probe'] if a.fee_limit is None else a.fee_limit
    clients = {}
    for c in calls:
        if (c['llm'], c['input'], c['call']) in done:
            continue
        spent = code_reports.fees(design)['total']
        if spent >= limit:
            say(f'fee guard: {spent:.2f} USD spent, limit {limit:g} USD; the probe stops')
            return EXIT_GUARD
        if c['llm'] not in clients:
            clients[c['llm']] = base(common.MODELS[c['llm']], 8000, None, common.REASONING_EFFORT)
        llm = clients[c['llm']]
        before = llm.usage['cost_usd']
        line = dict(time=now(), llm=c['llm'], model=llm.model, input=c['input'], call=c['call'],
                    message_sha256=c['message_sha256'])
        try:
            resp = probe_request(llm, c['message'])
        except Exception as e:                                  # noqa: BLE001 - recorded; the probe stops
            append_json(path, dict(line, error=f'{type(e).__name__}: {e}'[:2000],
                                   http_status=getattr(e, 'status_code', None)))
            say(f"probe call failed ({type(e).__name__}: {e}); restart to continue")
            return EXIT_STOPPED
        llm._account(resp)
        message = resp.choices[0].message
        usage = getattr(resp, 'usage', None)
        append_json(path, dict(line, content=message.content, finish_reason=resp.choices[0].finish_reason,
                               reasoning=(getattr(message, 'model_extra', None) or {}).get('reasoning'),
                               served=served_fields(resp), unix_time=round(time.time(), 3),
                               usage=dict(prompt_tokens=getattr(usage, 'prompt_tokens', None),
                                          completion_tokens=getattr(usage, 'completion_tokens', None),
                                          cost_usd=llm.usage['cost_usd'] - before)))
        say(f"probe {c['llm']} {c['input']} call {c['call']}: {(message.content or '')[:100]!r}")
    say(f"probe complete: {len(common.read_jsonl(path))} lines in {path}, fees "
        f"{code_reports.fees(design)['probe']:.2f} USD")
    return 0


# ---------------------------------------------------------------------- load test (section 3.9, stage 0(c))
def load_specs(design, n):
    """The first n of the reference models of section 5 for pilot dataset P1 (calibrate.reference_specs): M_red,
    M_full, M_full without CL~WT and without V~WT, M_full plus CL~twin, V~twin, V~target, CL~SEX and V~SEX, and its
    two-compartment version. Starting values are the true values, as in the calibration."""
    d = common.dataset(design, common.PILOT_DATASETS[0]['id'])
    specs = calibrate.reference_specs(d['beta'], d['target_column'], d['twin_column'])
    return [specs[label] for label in LOAD_MODELS[:n]]


def load_job(a):
    """One simulated job of the load test: a session with the settings of section 6 that fits the reference models
    in calls of LOAD_BATCH specifications without standard errors, as an agent run does without its LLM."""
    design, _ = load_design(a)
    common.add_paths()
    from pkagent.config import Budget, Settings
    from pkagent.session import Session
    d = common.dataset(design, common.PILOT_DATASETS[0]['id'])
    specs = load_specs(design, a.load_fits)
    out = Path(a.out)
    session = Session(common.folders(design)['pilot'] / 'data' / d['id'] / common.DATA_FILE, out,
                      Settings(workers=common.WORKERS, threads_per_worker=common.THREADS_PER_WORKER,
                               seed=common.ENGINE_SEED + 1, budget=Budget(max_fits=len(specs))),
                      common.load_strings()['description']['text'])
    t0, rows = time.time(), []
    try:
        for i in range(0, len(specs), LOAD_BATCH):
            for r in session.fit(specs[i:i + LOAD_BATCH], uncertainty=False):
                rows.append(dict(model_id=r.get('model_id'), name=r.get('name'), status=r.get('status'),
                                 ofv=r.get('ofv'), seconds=(r.get('summary') or {}).get('seconds')))
                print(rows[-1], flush=True)
    finally:
        session.close()
    write_json(out / 'load_job.json', dict(seconds=round(time.time() - t0, 1), fits=rows))
    return 0


def load_test(a):
    """Stage 0(c): for each number of jobs, start that many simulated jobs at once and measure the fits per hour,
    for the time estimate of section 11. The test chooses nothing: the number of parallel runs is fixed in section 6
    (JOBS). Output: calibration/load_test/<time>/ of the design. Returns 0 when every job completed its fits."""
    design, design_path = load_design(a)
    problems = engine_problems(design)
    pilot = common.folders(design)['pilot'] / 'data' / common.PILOT_DATASETS[0]['id'] / common.DATA_FILE
    if not pilot.exists():
        problems.append(f'{pilot} does not exist')
    if problems:
        print('\n'.join('problem: ' + p for p in problems))
        return 1
    root = common.folders(design)['calibration'] / 'load_test' / time.strftime('%Y%m%d-%H%M%S')
    n_specs = len(load_specs(design, a.load_fits))
    say(f"load test: {n_specs} fits per job, {common.WORKERS} workers x {common.THREADS_PER_WORKER} threads, "
        f"jobs {a.load_jobs}; output {root}")
    rows = []
    for jobs in a.load_jobs:
        psutil.cpu_percent(None)
        t0, children = time.time(), []
        for j in range(1, jobs + 1):
            out = root / f'jobs{jobs}' / f'job{j}'
            out.mkdir(parents=True)
            log = open(out.parent / f'job{j}.console.log', 'w', encoding='utf-8')
            cmd = [sys.executable, '-u', str(HERE), '--load-job', '--design', str(design_path), '--out', str(out),
                   '--load-fits', str(a.load_fits)]
            children.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(common.REPO),
                                              env=child_env(design, True)), log, out))
        for proc, log, _ in children:
            proc.wait()
            log.close()
        wall = time.time() - t0
        fits = [f for _, _, out in children for f in (read_json(out / 'load_job.json') or {}).get('fits', [])]
        seconds = sorted(f['seconds'] for f in fits if f.get('seconds') is not None)
        rows.append(dict(jobs=jobs, fits=len(fits), converged=sum(f['status'] == 'converged' for f in fits),
                         wall_seconds=round(wall, 1), fits_per_hour=round(len(fits) / wall * 3600, 1),
                         median_fit_seconds=round(statistics.median(seconds), 1) if seconds else None,
                         cpu_percent=psutil.cpu_percent(None), complete=len(fits) == jobs * n_specs))
        say(f"{jobs} jobs: {rows[-1]}")
    write_json(root / 'load_test.json', dict(time=now(), design=str(design_path), fits_per_job=n_specs,
                                             workers=common.WORKERS, threads_per_worker=common.THREADS_PER_WORKER,
                                             results=rows))
    print('jobs  fits  converged  wall s  fits/hour  median fit s  CPU %')
    for r in rows:
        print(f"{r['jobs']:4d}  {r['fits']:4d}  {r['converged']:9d}  {r['wall_seconds']:6.0f}  "
              f"{r['fits_per_hour']:9.1f}  {str(r['median_fit_seconds']):>12s}  {r['cpu_percent']:5.1f}")
    incomplete = [r['jobs'] for r in rows if not r['complete']]
    print('fits per hour for the time estimate of section 11; the number of parallel runs is fixed in section 6'
          + (f'. Not all fits were completed with {incomplete} jobs' if incomplete else ''))
    return 1 if incomplete else 0


# ---------------------------------------------------------------------- command line
def arguments(argv=None):
    ap = argparse.ArgumentParser(description='Runner of the simulation experiment; the default is a dry run.')
    ap.add_argument('--module', choices=GRIDS + ('probe',))
    ap.add_argument('--blocks', type=int, nargs='+', choices=common.BLOCKS, help='blocks of A (default: all)')
    ap.add_argument('--reduction', type=int, default=0, choices=(0, 1, 2),
                    help='section 10.3: 1 leaves out block 4 (of A and of N1), 2 also the deny-neutral cells of '
                         'block 3')
    ap.add_argument('--jobs', type=int,
                    help='runs in parallel (default 2, for module B 1; with --execute 1 or 2, for module B 1)')
    ap.add_argument('--design', help='design file (default design.json)')
    ap.add_argument('--execute', action='store_true', help='start paid runs (without it: dry run)')
    ap.add_argument('--status', action='store_true', help='state of the cells and fees spent')
    ap.add_argument('--replay', help='recorded run folder to replay with the stand-in (stage 0(a))')
    ap.add_argument('--stand-in',
                    help='recorded run folder whose script replaces the LLM in the cells of a test design')
    ap.add_argument('--variant', default='replay', choices=('replay', 'never_finalize', 'api_error', 'hang'))
    ap.add_argument('--turns', type=int, help='recorded turns replayed before the variant acts')
    ap.add_argument('--api-status', type=int, default=400, help='HTTP status of the variant api_error')
    ap.add_argument('--out', help='output folder of --replay (default smoke/replay or smoke/replay_<variant>)')
    ap.add_argument('--load-test', action='store_true', help='stage 0(c)')
    ap.add_argument('--load-jobs', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--load-fits', type=int, default=10)
    ap.add_argument('--watchdog-seconds', type=float, help='test only')
    ap.add_argument('--stall-seconds', type=float, help='test only')
    ap.add_argument('--poll-seconds', type=float, help='test only')
    ap.add_argument('--fee-limit', type=float, help='test only')
    ap.add_argument('--max-launches', type=int, help='test only: start at most this many runs')
    ap.add_argument('--dry-run', action='store_true', help='the default: print the cells, start nothing')
    for name in ('--child', '--load-job', '--utf8-check'):       # started by this script
        ap.add_argument(name, action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--key', help=argparse.SUPPRESS)
    ap.add_argument('--attempt', type=int, default=1, help=argparse.SUPPRESS)
    return ap, ap.parse_args(argv)


def main(argv=None):
    ap, a = arguments(argv)
    if a.utf8_check:
        print(sys.stdout.encoding)
        return 0 if utf8_stdout() else 1
    common.require(not (a.dry_run and (a.execute or a.stand_in or a.replay)),
                   '--dry-run excludes --execute, --stand-in and --replay')
    if a.jobs is None:
        a.jobs = default_jobs(a.module)
    common.require(a.jobs >= 1, '--jobs must be at least 1')
    if a.replay and not a.out:
        plain = a.variant == 'replay' and a.turns is None
        a.out = str(common.SMOKE / ('replay' if plain else f'replay_{a.variant}'))
    if a.child:
        return child_main(a)
    if a.load_job:
        return load_job(a)
    if a.load_test:
        return load_test(a)
    if a.replay:
        common.require(not a.execute and not a.module, '--replay is a stage 0 test: no --execute, no --module')
        common.require(inside(a.out, common.SMOKE), 'the output of --replay must be below smoke/')
        design, design_path = load_design(a, required=False)
        cells = [replay_cell(a.replay, a.out)]
        problems = preflight(design, cells, a, 'replay')
        if problems:
            print('\n'.join('problem: ' + p for p in problems))
            return 1
        say(f"replay of {Path(a.replay).resolve()} (variant {a.variant}) into {cells[0]['run_dir']}; engine: "
            f"{(engine_of(design) or {}).get('folder') or 'the importable PKPy2 (no frozen engine)'}")
        return run_cells(design, design_path, cells, a, 'replay')
    if not a.module:
        ap.print_help()
        return 1
    design, design_path = load_design(a)
    if a.module == 'probe':
        return probe_main(design, a)
    cells = select_cells(design, a)
    if a.status:
        return status(design, cells, a)
    if not (a.execute or a.stand_in):
        return dry_run(design, cells, a)
    problems = preflight(design, cells, a, a.module)
    if problems:
        print('\n'.join('problem: ' + p for p in problems))
        return 1
    return run_cells(design, design_path, cells, a, a.module)


if __name__ == '__main__':
    sys.exit(main())
