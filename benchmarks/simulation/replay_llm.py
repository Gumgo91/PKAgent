"""Scripted stand-in for the LLM of PKAgent (PROTOCOL.md, section 3.9, stage 0): no LLM, no key, no fees.

stand_in() returns a subclass of pkagent.llm.LLM whose client answers from the recorded turns of a benchmark
transcript.jsonl. Requests still pass through the unchanged LLM.chat (message preparation, retry rule, accounting),
so that `run_grid.py --replay` exercises the real runner, sidecar, checks and served.jsonl writer. With one Laplace
start a fit is deterministic: replaying a recorded run must reproduce its OFVs, which compare() verifies.

Variants (`turns` = number of recorded turns that are replayed first):
  replay          the recorded turns as recorded (default: all)
  never_finalize  the recorded turns without their finalize_model calls, then answers without tool calls, so that
                  the run ends normally without a final model (default: all)
  api_error       the recorded turns, then every request raises an HTTP error (default: 2 turns, error at turn 3)
  hang            the recorded turns, then the request never returns (default: 2 turns, hang at turn 3)
Beyond the replayed turns the variants replay and never_finalize answer without tool calls.

The stand-in is used through the runner:
  python benchmarks/simulation/run_grid.py --replay benchmarks/runs/pheno/knowledge/claude/rep3
  python benchmarks/simulation/run_grid.py --replay <recorded run folder> --variant api_error
  python benchmarks/simulation/run_grid.py --module A --design smoke/design_mock.json --stand-in <script folder>
This file alone lists the turns and the models of a recorded run, or writes a script for the simulated datasets,
which no recorded run exists for before the pilot:
  python benchmarks/simulation/replay_llm.py <recorded run folder>
  python benchmarks/simulation/replay_llm.py --script benchmarks/simulation/smoke/script
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import common

VARIANTS = ('replay', 'never_finalize', 'api_error', 'hang')
PROVIDER = 'replay_llm'                         # provider and model prefix that the stand-in reports as served
TOKENS = ('prompt_tokens', 'completion_tokens', 'cached_tokens', 'reasoning_tokens')
IDLE = dict(content='No further tool calls.', tool_calls=[], finish_reason='stop', reasoning=None, tokens={})
OFV_TOLERANCE = 0.0005                          # a replayed OFV must equal the recorded one to three decimals


class APIStatusError(Exception):
    """Error of the scripted client that carries an HTTP status, as the errors of the OpenAI client do."""

    def __init__(self, status_code, message=None):
        super().__init__(message or f'scripted API error (HTTP {status_code})')
        self.status_code = status_code


def load_script(run_dir):
    """The turns of <run_dir>/transcript.jsonl in order: content, tool_calls, finish_reason, reasoning and the
    tokens of each call (differences of the cumulative usage)."""
    turns, previous = [], {}
    for line in (Path(run_dir) / 'transcript.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        usage = t.get('usage') or {}
        turns.append(dict(content=t.get('content') or '', tool_calls=t.get('tool_calls') or [],
                          finish_reason=t.get('finish_reason'), reasoning=t.get('reasoning'),
                          tokens={k: int(usage.get(k) or 0) - int(previous.get(k) or 0) for k in TOKENS}))
        previous = usage
    return turns


def _without_finalize(turn):
    calls = [c for c in turn['tool_calls'] if c['function']['name'] != 'finalize_model']
    return dict(turn, tool_calls=calls, finish_reason=turn['finish_reason'] if calls else 'stop')


def _response(turn, number, model):
    """One turn in the shape of a chat completion of the OpenAI client (the attributes that LLM.chat and
    LLM._account read, and those that the runner writes to served.jsonl). The cost is 0: no fees are spent."""
    calls = [SimpleNamespace(id=c['id'], function=SimpleNamespace(name=c['function']['name'],
                                                                 arguments=c['function']['arguments']))
             for c in turn['tool_calls']]
    message = SimpleNamespace(content=turn['content'], tool_calls=calls or None,
                              model_extra=dict(reasoning=turn.get('reasoning')))
    tokens = turn.get('tokens') or {}
    usage = SimpleNamespace(prompt_tokens=tokens.get('prompt_tokens', 0),
                            completion_tokens=tokens.get('completion_tokens', 0),
                            prompt_tokens_details=SimpleNamespace(cached_tokens=tokens.get('cached_tokens', 0)),
                            completion_tokens_details=SimpleNamespace(
                                reasoning_tokens=tokens.get('reasoning_tokens', 0)),
                            model_extra=dict(cost=0.))
    return SimpleNamespace(id=f'replay-{number:03d}', created=int(time.time()), model=f'{PROVIDER}/{model}',
                           provider=PROVIDER, model_extra=dict(provider=PROVIDER), usage=usage,
                           choices=[SimpleNamespace(message=message,
                                                    finish_reason=turn.get('finish_reason') or 'stop')])


class _Client:
    """The part of the OpenAI client that LLM.chat uses (client.chat.completions.create), answered from a script."""

    def __init__(self, script, variant, turns, status, idle):
        self.script, self.variant, self.turns, self.status, self.idle = script, variant, turns, status, idle
        self.answered = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **request):
        number = self.answered + 1
        if number > self.turns:
            if self.variant == 'api_error':
                raise APIStatusError(self.status)
            if self.variant == 'hang':
                while True:
                    time.sleep(60)
        turn = self.script[number - 1] if number <= min(self.turns, len(self.script)) else self.idle
        self.answered = number
        return _response(turn, number, request.get('model'))


def stand_in(script, variant='replay', turns=None, status=400, idle=None):
    """A subclass of pkagent.llm.LLM that answers from `script` (turns as returned by load_script) and needs no key.

    variant and turns: see the module docstring. status: HTTP status of the api_error variant (LLM.chat raises 400
    to 404 at the second attempt and retries other errors eight times). idle: content of the answers beyond the
    replayed turns.
    """
    common.require(variant in VARIANTS, f'unknown variant {variant}')
    common.add_paths()
    from pkagent.llm import LLM
    script = [_without_finalize(t) for t in script] if variant == 'never_finalize' else list(script)
    if turns is None:
        turns = len(script) if variant in ('replay', 'never_finalize') else 2
    answer = dict(IDLE, content=idle) if idle else IDLE

    class StandIn(LLM):
        def __init__(self, model, max_output_tokens=8000, temperature=None, reasoning_effort='medium'):
            self.client = _Client(script, variant, int(turns), int(status), answer)
            self.model = model
            self.max_output_tokens = max_output_tokens
            self.temperature = temperature
            self.reasoning_effort = reasoning_effort
            self.usage = dict(calls=0, prompt_tokens=0, completion_tokens=0, cached_tokens=0, reasoning_tokens=0,
                              cost_usd=0.)
    return StandIn


def write_script(folder):
    """Write <folder>/transcript.jsonl, a script of three turns for any simulated dataset of the experiment:
    describe_data, one fit_models call with the reference models M_red and M_full of section 5 (weight on CL and
    V; M_full with COMED1 on CL), and finalize_model with M_full. Returns the path."""
    import generate
    red, full = generate.reference_models(-0.2, 'COMED1')
    report = dict(summary='One-compartment model with weight on CL and V and COMED1 on CL.',
                  development='Scripted stand-in: the reduced and the full reference model were fitted together.',
                  evaluation='Not evaluated (scripted stand-in).', limitations='Scripted stand-in.')
    calls = (('describe_data', {}), ('fit_models', dict(models=[red, full], standard_errors=False)),
             ('finalize_model', dict(model_id='M002', report=report)))
    path = Path(folder) / 'transcript.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as out:
        for i, (name, arguments) in enumerate(calls, 1):
            call = dict(id=f'call_{i}', type='function', function=dict(name=name, arguments=json.dumps(arguments)))
            out.write(json.dumps(dict(
                turn=i, content=f'scripted turn {i}', tool_calls=[call], finish_reason='tool_calls', reasoning=None,
                usage=dict(calls=i, prompt_tokens=1000 * i, completion_tokens=100 * i, cached_tokens=0,
                           reasoning_tokens=0, cost_usd=0.))) + '\n')
    return path


def recorded_models(run_dir):
    """The models of <run_dir>/results.json in the order of fitting: model_id, status, ofv and data_version."""
    results = json.loads((Path(run_dir) / 'results.json').read_text(encoding='utf-8'))
    return [dict(model_id=m['model_id'], status=m.get('status'), ofv=m.get('ofv'), data_version=m.get('data_version'))
            for m in results.get('models', [])]


def compare(recorded_dir, replayed_dir, tolerance=OFV_TOLERANCE):
    """Compare a replayed run with the recorded one: status and OFV of every model, the final model and the
    normalized data files. Returns dict(recorded, models, n_models, max_abs_difference, tolerance, final_model,
    data_identical, reproduced). reproduced: the same models were fitted with the same status, every OFV agrees
    within the tolerance and the final model is the same."""
    recorded_dir, replayed_dir = Path(recorded_dir), Path(replayed_dir)
    old, new = recorded_models(recorded_dir), recorded_models(replayed_dir)
    rows = []
    for i in range(max(len(old), len(new))):
        a, b = (old[i] if i < len(old) else {}), (new[i] if i < len(new) else {})
        both = a.get('ofv') is not None and b.get('ofv') is not None
        rows.append(dict(model_id=a.get('model_id') or b.get('model_id'), recorded_status=a.get('status'),
                         replayed_status=b.get('status'), recorded_ofv=a.get('ofv'), replayed_ofv=b.get('ofv'),
                         difference=b['ofv'] - a['ofv'] if both else None))
    final = []
    for folder in (recorded_dir, replayed_dir):
        results = json.loads((folder / 'results.json').read_text(encoding='utf-8'))
        final.append((results.get('final_model') or {}).get('model_id'))
    names = sorted(p.name for p in (recorded_dir / 'data').glob('data_v*.csv'))
    identical = names == sorted(p.name for p in (replayed_dir / 'data').glob('data_v*.csv')) and all(
        (recorded_dir / 'data' / n).read_bytes() == (replayed_dir / 'data' / n).read_bytes() for n in names)

    def agrees(r):
        if r['recorded_status'] != r['replayed_status']:
            return False
        if r['recorded_ofv'] is None or r['replayed_ofv'] is None:
            return r['recorded_ofv'] is None and r['replayed_ofv'] is None
        return abs(r['difference']) <= tolerance
    differences = [abs(r['difference']) for r in rows if r['difference'] is not None]
    return dict(recorded=str(recorded_dir), models=rows, n_models=[len(old), len(new)],
                max_abs_difference=max(differences) if differences else None, tolerance=tolerance,
                final_model=dict(recorded=final[0], replayed=final[1]), data_identical=bool(identical),
                reproduced=bool(rows and len(old) == len(new) and all(agrees(r) for r in rows)
                                and final[0] == final[1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description='Scripted stand-in for the LLM: list a recorded run or write a script.')
    ap.add_argument('run', nargs='?', help='recorded run folder (or script folder) to list')
    ap.add_argument('--script', metavar='FOLDER', help='write a script for the simulated datasets to this folder')
    a = ap.parse_args(argv)
    if a.script:
        print(f'wrote {write_script(a.script)}')
    folder = a.run or a.script
    if not folder:
        ap.print_help()
        return 1
    for i, t in enumerate(load_script(folder), 1):
        calls = ', '.join(c['function']['name'] for c in t['tool_calls']) or '(no tool call)'
        print(f"turn {i}: {calls}; finish_reason {t['finish_reason']}; tokens {t['tokens']}")
    if (Path(folder) / 'results.json').exists():
        for m in recorded_models(folder):
            print(f"{m['model_id']} {m['status']} OFV {m['ofv']} (data version {m['data_version']})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
