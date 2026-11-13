"""Simulated datasets of the experiment (PROTOCOL.md, sections 2.2 to 2.6): covariates and sampling times, the true
model, the simulation with PKPy2 and the CSV writer. No fits.

Usage:
  python benchmarks/simulation/generate.py              write data/<ID>/pk_data.csv for the datasets of design.json and
                                                        verify the sha256 of every file
  python benchmarks/simulation/generate.py --verify     the same comparison without writing
  python benchmarks/simulation/generate.py --test [N]   generator test of section 2.3 over N seeds (default 300)
  python benchmarks/simulation/generate.py --mock       write the mock design of the tests to smoke/
  --design PATH selects another design file.
calibrate.py calls covariates(), truth(), reference_models(), simulate(), csv_bytes() and write_dataset() for a
candidate seed, beta and target column.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common

BASE_SEED = common.SEED_GENERATOR                      # 20261112
N = 60                                                  # subjects
DOSE, T_INF = 100., 1.                                  # mg, hours
NOMINAL = (0.5, 1, 1.5, 3, 6, 10, 16, 24)               # nominal sampling times, hours after the start of the infusion
R_MAX, MAX_DRAWS = 0.10, 5000                           # allocation of the comedications
INTERNAL = ['ID', 'TIME', 'AMT', 'RATE', 'DV', 'EVID', 'MDV', 'WT', 'SEX', 'CA', 'CB']
COLUMNS = ['ID', 'TIME', 'AMT', 'RATE', 'DV', 'EVID', 'MDV', 'WT', 'SEX', 'COMED1', 'COMED2']
DV_FORMAT = '%.4g'
TRUE = dict(CL=5.0, V=50.0, wt_center=70, wt_exponent_CL=0.75, wt_exponent_V=1.0, iiv_CL=0.0625, iiv_V=0.04,
            proportional=0.15)
# recorded in design.json
CONSTANTS = dict(base_seed=BASE_SEED, n_subjects=N, dose_mg=DOSE, infusion_hours=T_INF, nominal_times=list(NOMINAL),
                 time_jitter=0.08, men=[24, 36], weight=dict(median_women=66.0, log_effect_men=0.18, log_sd=0.22,
                                                             limits=[45, 130], decimals=1),
                 treated=[20, 28], max_abs_correlation=R_MAX, max_allocation_draws=MAX_DRAWS, truth=TRUE,
                 columns=COLUMNS, dv_format=DV_FORMAT)


# ---------------------------------------------------------------------- covariates and design (section 2.3)
def covariates(seed):
    """Covariates and sampling times of dataset seed s, drawn from numpy.random.default_rng([20261112, s]) in the
    order of section 2.3. Returns dict(sex, wt, ca, cb: arrays of 60; times: 60 x 8; n_treated; draws)."""
    rng = np.random.default_rng([BASE_SEED, seed])
    while True:                                         # 1. sex, redrawn as a whole until 24 to 36 men
        sex = (rng.random(N) < 0.5).astype(int)
        if 24 <= sex.sum() <= 36:
            break
    wt = np.round(np.clip(np.exp(np.log(66.) + 0.18 * sex + rng.normal(0, 0.22, N)), 45, 130), 1)    # 2. weight
    n1 = int(rng.integers(20, 29))                      # 3. subjects on each comedication
    log_wt = np.log(wt)
    for draws in range(1, MAX_DRAWS + 1):               # 4. CA (carries the effect) and CB (twin)
        ca, cb = np.zeros(N, dtype=int), np.zeros(N, dtype=int)
        ca[rng.permutation(N)[:n1]] = 1
        cb[rng.permutation(N)[:n1]] = 1
        r = [abs(np.corrcoef(a, b)[0, 1]) for a, b in ((ca, cb), (ca, log_wt), (ca, sex), (cb, log_wt), (cb, sex))]
        if max(r) < R_MAX:
            break
    else:
        raise RuntimeError(f'seed {seed}: no allocation with all correlations below {R_MAX} in {MAX_DRAWS} draws')
    times = np.round(np.array(NOMINAL)[None, :] * (1 + rng.uniform(-0.08, 0.08, (N, len(NOMINAL)))), 2)   # 5. times
    return dict(sex=sex, wt=wt, ca=ca, cb=cb, times=times, n_treated=n1, draws=draws)


def design_table(seed):
    """Event records of dataset seed s without concentrations (section 2.4): per subject one dose record and eight
    observation records at the actual times; columns ID, TIME, AMT, RATE, DV ('.'), EVID, MDV, WT, SEX, CA, CB."""
    c = covariates(seed)
    rows = []
    for i in range(N):
        base = dict(ID=i + 1, WT=float(c['wt'][i]), SEX=int(c['sex'][i]), CA=int(c['ca'][i]), CB=int(c['cb'][i]))
        rows.append(dict(base, TIME=0., AMT=DOSE, RATE=DOSE / T_INF, DV='.', EVID=1, MDV=1))
        rows += [dict(base, TIME=float(t), AMT=0., RATE=0., DV='.', EVID=0, MDV=1) for t in c['times'][i]]
    return pd.DataFrame(rows)[INTERNAL]


# ---------------------------------------------------------------------- true model and simulation (sections 2.2, 2.5)
def truth(beta=0., covariate='CA'):
    """PKAgent specification of the data-generating model with the coefficient beta of the comedication on
    clearance; the term is left out when beta = 0. With the column name of a data file as `covariate` the same
    specification is a reference model that starts at the true values."""
    cov = [dict(parameter='CL', covariate='WT', form='power', center=TRUE['wt_center'], value=TRUE['wt_exponent_CL']),
           dict(parameter='V', covariate='WT', form='power', center=TRUE['wt_center'], value=TRUE['wt_exponent_V'])]
    if beta != 0:
        cov.append(dict(parameter='CL', covariate=covariate, form='categorical', level=1, value=float(beta)))
    return dict(structure=dict(compartments=1, absorption='iv'), parameters=dict(CL=TRUE['CL'], V=TRUE['V']),
                iiv=dict(CL=TRUE['iiv_CL'], V=TRUE['iiv_V']), covariates=cov,
                residual=dict(proportional=TRUE['proportional']))


def reference_models(beta, target='CA'):
    """(M_red, M_full) of section 5 for a file whose target column is `target`: starting values are the true
    values, and the target coefficient starts at beta, or at -0.01 when beta = 0."""
    return dict(truth(0.), name='M_red'), dict(truth(beta if beta != 0 else -0.01, target), name='M_full')


def simulate(seed, beta=0.):
    """Simulate dataset seed s with the true coefficient beta (section 2.5). Returns the table of design_table()
    with DV filled on the observation records (NaN on dose records) and MDV 0 there. The simulation seed is
    20261112 + s, so the same s with another beta reuses the same random numbers."""
    common.add_paths()
    import pkpy2
    from pkagent.spec import compile_model, normalize
    table = design_table(seed)
    design = pkpy2.read_nonmem(table, covariates=['WT', 'SEX', 'CA', 'CB'])
    sim = pkpy2.simulate_data(compile_model(normalize(truth(beta))), design, seed=BASE_SEED + seed)
    dv = np.concatenate([ind.dv for ind in sim])
    common.require(len(dv) == len(table) and np.array_equal(np.concatenate([ind.time for ind in sim]),
                                                           table['TIME'].to_numpy()),
                   'the simulated records are not in the order of the design table')
    obs = (table['EVID'] == 0).to_numpy()
    table['DV'] = np.where(obs, dv, np.nan)
    table['MDV'] = np.where(obs, 0, 1)
    return table


# ---------------------------------------------------------------------- data file (section 2.6)
def csv_bytes(table, target_column=None):
    """The bytes of the data file of a simulated table. With target_column COMED1 or COMED2 the internal column CA
    is written under that name and CB under the other (column order ID, TIME, AMT, RATE, DV, EVID, MDV, WT, SEX,
    COMED1, COMED2); with None the internal names CA and CB are kept (calibration search). DV is written with
    '%.4g' and '.' on dose records; lines end with a line feed."""
    if target_column is None:
        header, source = INTERNAL, {}
    else:
        twin = common.COLUMNS[1 - common.COLUMNS.index(target_column)]
        header, source = COLUMNS, {target_column: 'CA', twin: 'CB'}
    columns = [table[source.get(name, name)].tolist() for name in header]
    lines = [','.join(header)]
    for row in zip(*columns):
        cells = []
        for name, v in zip(header, row):
            if name == 'DV':
                cells.append('.' if v != v else DV_FORMAT % v)
            elif name in ('TIME', 'AMT', 'RATE', 'WT'):
                cells.append(repr(float(v)))
            else:
                cells.append('%d' % v)
        lines.append(','.join(cells))
    return ('\n'.join(lines) + '\n').encode('ascii')


def write_dataset(seed, beta, target_column, path):
    """Simulate and write one data file; returns the sha256 of its bytes."""
    data = csv_bytes(simulate(seed, beta), target_column)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def regenerate(design, write=True, report=print):
    """Regenerate the files of a design (the 28 datasets and, if present, the pilot datasets) and compare the
    sha256 of the bytes with the design and with the files on disk. A file is written only when the regenerated
    bytes have the sha256 of the design (or the design has none yet), so that a tracked file is never replaced by
    different content. Returns the number of problems."""
    f = common.folders(design)
    todo = [(d, f['data']) for d in common.datasets(design)]
    todo += [(d, f['pilot'] / 'data') for d in (design.get('pilot') or {}).get('datasets') or []]
    problems = 0
    for d, folder in todo:
        path = folder / d['id'] / common.DATA_FILE
        if d.get('seed') is None or d.get('beta') is None:
            report(f"{d['id']}: no seed or beta in the design")
            problems += 1
            continue
        data = csv_bytes(simulate(d['seed'], d['beta']), d['target_column'])
        digest, expected = hashlib.sha256(data).hexdigest(), d.get('csv_sha256')
        ok = expected is None or digest == expected
        note = 'no sha256 in the design' if expected is None else 'ok' if ok else f'MISMATCH with the design ({expected})'
        if path.exists() and path.read_bytes() != data:
            on_disk = path.read_bytes()
            note += ('; the file differs in line endings only' if on_disk.replace(b'\r\n', b'\n') == data
                     else '; the file on disk differs')
            ok = False
        elif not path.exists():
            note += '; written' if write and ok else '; no file'
            ok = ok and write
        if write and ok and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        problems += not ok
        report(f"{d['id']} seed {d['seed']} beta {d['beta']:g} {d['target_column']} sha256 {digest} {note}")
    report(f'{len(todo)} datasets, {problems} problems')
    return problems


# ---------------------------------------------------------------------- generator test (section 2.3)
def _log_clearance(table):
    """log(dose / AUC) of every subject: linear trapezoids from time 0 and a log-linear tail from the samples
    after 5 hours."""
    out = []
    for _, g in table[table['EVID'] == 0].groupby('ID', sort=True):
        t = np.concatenate([[0.], g['TIME'].to_numpy()])
        c = np.concatenate([[0.], g['DV'].to_numpy()])
        auc = np.sum((t[1:] - t[:-1]) * (c[1:] + c[:-1]) / 2)
        late = g[g['TIME'] >= 5]
        k = -np.polyfit(late['TIME'], np.log(late['DV']), 1)[0]
        out.append(np.log(DOSE / (auc + c[-1] / k)))
    return np.array(out)


def generator_test(n_seeds=300, report=print):
    """Generator test without fits: for seeds 1 to n_seeds at beta = 0, regress log(dose/AUC) of the 60 subjects on
    log weight and both binary covariates and collect the z values of CA and CB. Mean 0 and SD 1 are expected when
    the covariates are independent of the random numbers of the simulation. Returns {CA, CB: dict(mean, sd)}."""
    z = []
    for seed in range(1, n_seeds + 1):
        table = simulate(seed, 0.)
        subjects = table.groupby('ID', sort=True).first()
        y = _log_clearance(table)
        x = np.column_stack([np.ones(N), np.log(subjects['WT'].to_numpy()), subjects['CA'].to_numpy(),
                             subjects['CB'].to_numpy()])
        b = np.linalg.lstsq(x, y, rcond=None)[0]
        s2 = ((y - x @ b) ** 2).sum() / (N - x.shape[1])
        se = np.sqrt(np.diag(s2 * np.linalg.inv(x.T @ x)))
        z.append((b[2] / se[2], b[3] / se[3]))
    z = np.array(z)
    out = {}
    for j, name in enumerate(('CA', 'CB')):
        v = z[:, j]
        out[name] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)))
        report(f'{name}: {len(v)} seeds, z mean {v.mean():.2f}, SD {v.std(ddof=1):.2f}, '
               f'|z| > 1.96 in {np.mean(np.abs(v) > 1.96):.3f}, z^2 <= 0.5 in {np.mean(v ** 2 <= 0.5):.3f} '
               f'(0.52 expected), z^2 < 2 in {np.mean(v ** 2 < 2):.3f} (0.84 expected)')
    report(f'correlation of the two z values {np.corrcoef(z[:, 0], z[:, 1])[0, 1]:.3f}')
    report('protocol, section 2.3 (prototype, 300 seeds): CA mean 0.02, SD 0.92; CB mean -0.10, SD 1.08')
    return out


# ---------------------------------------------------------------------- mock design for tests
def mock_design(report=print):
    """Write smoke/design_mock.json and its data files: the 28 slots filled with generator seeds 1 to 28 (in slot
    order), a typical beta (-0.075 per unit of the square root of the evidence target; 0 in E0), the evidence
    fields set to the slot targets (E0: z_ref +0.5 or -0.5) and placeholder values for every field that needs a
    fit. Top-level flag "mock": true and root 'smoke', so that everything derived from it stays below smoke/."""
    design = common.template_design(root='smoke')
    design.update(mock=True, generator=CONSTANTS)
    f = common.folders(design)

    def record(base, seed, beta, dofv, z):
        target, twin = base['target_column'], base['twin_column']
        estimate = beta if beta else -0.07 * z
        d = dict(base, seed=seed, beta=beta, exp_beta=round(float(np.exp(beta)), 4),
                 n_treated=covariates(seed)['n_treated'], dofv0=0.25 if beta == 0 else 0.0, z0=z if beta == 0 else 0.0,
                 dofv_ref=dofv, z_ref=z,
                 reference=dict(estimate=round(estimate, 4), se=0.07, ci95=[round(estimate - 0.137, 4),
                                                                            round(estimate + 0.137, 4)],
                                parameters={}),
                 ofv={}, dofv_fixed=dofv, dofv_nowt=dofv,
                 evidence={'CL~WT': 26.0, 'V~WT': 38.0, f'CL~{twin}': 0.5, f'V~{twin}': 0.5, f'V~{target}': 0.5,
                           'CL~SEX': 0.5, 'V~SEX': 0.5},
                 dofv_2cmt=0.7, dofv_2cmt4=1.1,
                 two_cmt=dict(converged=True, Q=2.0, V2=20.0, rse_Q=None, rse_V2=None, converged4=True,
                              se_status4=None),
                 solver_evaluations=0,
                 baseline=dict(selected=[f'CL~{target}'] if dofv >= common.RETAIN else [],
                               target_selected=bool(dofv >= common.RETAIN)))
        return d
    for d in design['datasets']:
        slot = common.slot_of(d['id'])
        target = slot['evidence_target']
        if target is None:
            beta, dofv, z = 0.0, 0.25, 0.5 if slot['slot_type'] == 'E0+' else -0.5
        else:
            beta, dofv, z = -round(0.075 * float(np.sqrt(target)), 4), target, round(float(np.sqrt(target)), 4)
        d.update(record(dict(d), slot['index'] + 1, beta, dofv, z))
        d['csv_sha256'] = write_dataset(d['seed'], d['beta'], d['target_column'], f['data'] / d['id'] / common.DATA_FILE)
        report(f"{d['id']} block {d['block']} {d['slot_type']} seed {d['seed']} beta {d['beta']:g} "
               f"{d['target_column']} n_treated {d['n_treated']} sha256 {d['csv_sha256'][:16]}")
    pilot = []
    for p in common.PILOT_DATASETS:
        t = p['evidence_target']
        d = dict(id=p['id'], seed=p['seed'], beta=-round(0.075 * float(np.sqrt(t)), 4), evidence_target=t,
                 target_column=p['target_column'], twin_column=p['twin_column'], dofv_ref=t,
                 z_ref=round(float(np.sqrt(t)), 4))
        d['csv_sha256'] = write_dataset(d['seed'], d['beta'], d['target_column'],
                                        f['pilot'] / 'data' / d['id'] / common.DATA_FILE)
        pilot.append(d)
        report(f"{d['id']} seed {d['seed']} beta {d['beta']:g} {d['target_column']} sha256 {d['csv_sha256'][:16]}")
    design.update(pilot=dict(datasets=pilot), strings=common.load_strings(),
                  benchmark_data=common.benchmark_data(), engine=None)
    design['run_order'] = common.run_order(design)
    path = f['root'] / 'design_mock.json'
    common.write_json(path, design)
    report(f'wrote {path}')
    return design


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Simulated datasets of the experiment (no fits).')
    ap.add_argument('--design', help='design file (default: design.json)')
    ap.add_argument('--verify', action='store_true', help='compare with the design and the files, write nothing')
    ap.add_argument('--test', nargs='?', type=int, const=300, metavar='N', help='generator test over N seeds')
    ap.add_argument('--mock', action='store_true', help='write the mock design to smoke/')
    a = ap.parse_args(argv)
    if a.test:
        generator_test(a.test)
        return 0
    if a.mock:
        mock_design()
        return 0
    path = common.design_path(a.design)
    if not path.exists():
        print(f'{path} does not exist (calibrate.py writes design.json)')
        return 1
    return 1 if regenerate(common.load_design(path), write=not a.verify) else 0


if __name__ == '__main__':
    sys.exit(main())
