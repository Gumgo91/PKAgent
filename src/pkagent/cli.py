"""Command line: pkagent run DATA --description FILE_OR_TEXT [--knowledge TEXT] --model claude|gpt|<id> --out DIR"""
import argparse
from pathlib import Path

from .config import MODELS, Budget, Settings


def _text(value):
    if value and Path(value).is_file():
        return Path(value).read_text(encoding='utf-8')
    return value


def main(argv=None):
    ap = argparse.ArgumentParser(prog='pkagent')
    sub = ap.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run', help='run one analysis')
    r.add_argument('data', help='NONMEM-format CSV')
    r.add_argument('--description', required=True, help='data description (text or a file)')
    r.add_argument('--knowledge', help='expert knowledge from the analyst (text or a file)')
    r.add_argument('--objective', help='analysis objective (text or a file)')
    r.add_argument('--model', default='claude', help=f'LLM: {", ".join(MODELS)} or an OpenRouter model id')
    r.add_argument('--out', required=True)
    r.add_argument('--max-fits', type=int, default=40)
    r.add_argument('--max-turns', type=int, default=80)
    r.add_argument('--max-hours', type=float, default=6.)
    r.add_argument('--max-cost', type=float, default=25.)
    r.add_argument('--workers', type=int, default=2)
    r.add_argument('--threads', type=int, default=4)
    r.add_argument('--reasoning', default='medium', help='reasoning effort: low, medium, high or none')
    r.add_argument('--seed', type=int, default=20261001)
    r.add_argument('--no-images', action='store_true')
    a = ap.parse_args(argv)
    from .agent import run
    settings = Settings(model=MODELS.get(a.model, a.model), workers=a.workers, threads_per_worker=a.threads,
                        seed=a.seed, images=not a.no_images,
                        reasoning_effort=None if a.reasoning == 'none' else a.reasoning,
                        budget=Budget(max_turns=a.max_turns, max_fits=a.max_fits, max_hours=a.max_hours,
                                      max_cost_usd=a.max_cost))
    path = run(a.data, a.out, _text(a.description), _text(a.knowledge), settings, _text(a.objective))
    print(f'report: {path}')


if __name__ == '__main__':
    main()
