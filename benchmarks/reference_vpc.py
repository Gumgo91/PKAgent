"""VPCs of the PKPy2 fits of the reference models, with the settings PKAgent uses for the final models of the runs
(500 simulations, 8 bins, the default seed), without and with prediction correction.

Gives the coverage the reference model itself reaches, against which the coverage of the agents' final models is read.
Usage: python benchmarks/reference_vpc.py [pheno remifentanil oral_mm]   (after benchmarks/reference_fits.py)
Writes benchmarks/reference_fits/<name>/vpc.json.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
from pkagent import engine                  # noqa: E402
from pkagent.config import Settings          # noqa: E402


def coverage(result):
    inside = total = misses = upper = 0
    for out in result.values():
        a, b = map(int, out['observed_percentiles_inside'].split('/'))
        inside, total = inside + a, total + b
        for row in out['bins']:
            for q in ('p5', 'p50', 'p95'):
                if q in row and row[q]['observed'] is not None and not row[q]['inside']:
                    misses += 1
                    upper += q == 'p95'
    return dict(inside_percent=100. * inside / total, inside=inside, total=total, misses=misses, misses_p95=upper)


def main(names):
    seed = Settings().seed
    for name in names:
        folder = HERE / 'reference_fits' / name
        model = folder / 'models' / 'M001'
        data = sorted((folder / 'data').glob('data_v*.csv'))[-1]
        spec = json.loads((model / 'spec.json').read_text(encoding='utf-8'))
        out = dict(settings=dict(n=500, bins=8, seed=seed, data=data.name))
        for pc in (False, True):
            job = dict(spec=spec, data_csv=str(data), fit_json=str(model / 'fit.json'), out_dir=str(model),
                       prediction_corrected=pc, bins=8, seed=seed)
            result = engine.run_vpc(job)
            if result.get('status') == 'error':
                raise RuntimeError(f"{name}: {result['error']}\n{result.get('traceback', '')}")
            out['prediction_corrected' if pc else 'plain'] = dict(coverage(result), result=result)
        (folder / 'vpc.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
        print(name, {k: round(v['inside_percent'], 1) for k, v in out.items() if k != 'settings'})


if __name__ == '__main__':
    main(sys.argv[1:] or ['pheno', 'remifentanil', 'oral_mm'])
