"""Benchmark runs: each dataset without and with expert knowledge, for each LLM.

Usage:
  python benchmarks/run_benchmark.py --dataset pheno --condition knowledge --model claude --rep 1
  python benchmarks/run_benchmark.py --all --models gpt claude --reps 3 --jobs 2
  python benchmarks/run_benchmark.py --all --datasets pheno oral_mm --conditions none --models claude --reps 1
Outputs go to benchmarks/runs/<dataset>/<condition>/<model>/rep<k>/ (report.md, results.json, logs); a run whose
results.json exists is skipped, so an interrupted grid resumes where it stopped. With --jobs N, N runs proceed in
parallel as separate processes (console output in each run directory's console.log).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
from pkagent import MODELS, Budget, Settings, run          # noqa: E402

DATASETS = json.loads((HERE / 'datasets.json').read_text(encoding='utf-8'))
CONDITIONS = ('none', 'knowledge')


def out_dir(dataset, condition, model, rep):
    return HERE / 'runs' / dataset / condition / model / f'rep{rep}'


def run_one(dataset, condition, model, rep, budget, workers, threads, reasoning):
    d = DATASETS[dataset]
    out = out_dir(dataset, condition, model, rep)
    if (out / 'results.json').exists():
        print('exists:', out)
        return out
    settings = Settings(model=MODELS.get(model, model), workers=workers, threads_per_worker=threads,
                        seed=20261001 + rep, reasoning_effort=reasoning, budget=budget)
    knowledge = d['knowledge'] if condition == 'knowledge' else None
    print(f'=== {dataset} / {condition} / {model} / rep {rep}', flush=True)
    return run(HERE / 'data' / d['file'], out, d['description'], knowledge, settings, d.get('objective'))


def grid(a):
    cells = []
    for rep in range(1, a.reps + 1):                    # replicate-major: every cell gets rep 1 before rep 2
        for ds in a.datasets:
            for cond in a.conditions:
                for m in a.models:
                    cells.append((ds, cond, m, rep))
    return [c for c in cells if not (out_dir(*c) / 'results.json').exists()]


def launch(cell, a):
    ds, cond, m, rep = cell
    out = out_dir(*cell)
    if out.exists():                                    # an interrupted run restarts from scratch
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--dataset', ds, '--condition', cond, '--model', m,
           '--rep', str(rep), '--max-fits', str(a.max_fits), '--max-turns', str(a.max_turns),
           '--max-hours', str(a.max_hours), '--max-cost', str(a.max_cost), '--workers', str(a.workers),
           '--threads', str(a.threads), '--reasoning', a.reasoning]
    log = open(out.parent / f'{out.name}.console.log', 'w', encoding='utf-8')
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE.parent)), log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(DATASETS))
    ap.add_argument('--condition', choices=CONDITIONS)
    ap.add_argument('--model', default='claude')
    ap.add_argument('--rep', type=int, default=1)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS))
    ap.add_argument('--conditions', nargs='+', default=list(CONDITIONS))
    ap.add_argument('--models', nargs='+', default=['gpt', 'claude'])
    ap.add_argument('--reps', type=int, default=1)
    ap.add_argument('--jobs', type=int, default=1, help='runs in parallel (separate processes)')
    ap.add_argument('--max-fits', type=int, default=40)
    ap.add_argument('--max-turns', type=int, default=80)
    ap.add_argument('--max-hours', type=float, default=6.)
    ap.add_argument('--max-cost', type=float, default=25.)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--reasoning', default='medium')
    a = ap.parse_args()
    budget = Budget(max_turns=a.max_turns, max_fits=a.max_fits, max_hours=a.max_hours, max_cost_usd=a.max_cost)
    reasoning = None if a.reasoning == 'none' else a.reasoning
    if not a.all:
        run_one(a.dataset, a.condition, a.model, a.rep, budget, a.workers, a.threads, reasoning)
        return
    queue = grid(a)
    print(f'{len(queue)} runs to do', flush=True)
    active = []
    while queue or active:
        while queue and len(active) < a.jobs:
            cell = queue.pop(0)
            proc, log = launch(cell, a)
            active.append((cell, proc, log, time.time()))
            print(time.strftime('%H:%M:%S'), 'started', '/'.join(map(str, cell)), flush=True)
        time.sleep(20)
        for item in list(active):
            cell, proc, log, t0 = item
            if proc.poll() is not None:
                log.close()
                active.remove(item)
                ok = (out_dir(*cell) / 'results.json').exists()
                print(time.strftime('%H:%M:%S'), 'finished' if ok else f'FAILED (exit {proc.returncode})',
                      '/'.join(map(str, cell)), f'{(time.time() - t0) / 3600:.2f} h', flush=True)


if __name__ == '__main__':
    main()
