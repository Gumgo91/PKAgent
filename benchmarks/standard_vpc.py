"""The standard final VPC (500 simulations, 8 bins, the run's seed, no prediction correction) for runs in which
PKAgent kept the agent's own last VPC of the final model instead of running it after finalization.

session.finalize_outputs reuses the agent's last VPC of the final model when it was not prediction-corrected, so such
a run would be scored with the agent's own bins. This script runs the standard VPC for those runs (simulation only, no
fit; the run folders are not changed) so that every run is scored with the same VPC.
Usage: python benchmarks/standard_vpc.py     Writes benchmarks/evaluation/standard_vpc.json (read by evaluate.py).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
from pkagent import engine                  # noqa: E402


def own_final_vpc(run, final_id):
    """The agent's last VPC call on the final model if it was not prediction-corrected (PKAgent then kept it)."""
    last = None
    for line in (run / 'tool_log.jsonl').read_text(encoding='utf-8').splitlines():
        x = json.loads(line)
        if x['tool'] == 'run_vpc' and x['args'].get('model_id') == final_id and isinstance(x.get('result'), dict) \
                and x['result'].get('status') != 'error' and 'error' not in x['result']:
            last = x['args']
    return None if last is None or last.get('prediction_corrected') else last


def main():
    out, folder = {}, HERE / 'evaluation' / 'standard_vpc'
    for res_path in sorted((HERE / 'runs').glob('*/*/*/rep*/results.json')):
        run = res_path.parent
        res = json.loads(res_path.read_text(encoding='utf-8'))
        final = res['final_model']['model_id']
        own = own_final_vpc(run, final)
        if own is None:
            continue
        key = '/'.join(run.relative_to(HERE / 'runs').parts)
        seed = json.loads((run / 'run.json').read_text(encoding='utf-8'))['seed']
        model = run / 'models' / final
        data = run / 'data' / Path(res['final_model']['data_file'].replace('\\', '/')).name
        target = folder / key.replace('/', '_')
        target.mkdir(parents=True, exist_ok=True)
        job = dict(spec=json.loads((model / 'spec.json').read_text(encoding='utf-8')), data_csv=str(data),
                   fit_json=str(model / 'fit.json'), out_dir=str(target), prediction_corrected=False, bins=8, seed=seed)
        result = engine.run_vpc(job)
        if result.get('status') == 'error':
            raise RuntimeError(f"{key}: {result['error']}")
        for v in result.values():
            v['plot'] = Path(v['plot']).relative_to(HERE).as_posix()
        out[key] = dict(model_id=final, seed=seed, agent_vpc=own, result=result)
        print(key, final, {k: v['observed_percentiles_inside'] for k, v in result.items()}, 'agent bins', own.get('bins'))
    (HERE / 'evaluation' / 'standard_vpc.json').write_text(json.dumps(out, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
