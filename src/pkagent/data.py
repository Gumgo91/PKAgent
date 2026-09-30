"""Analysis data: NONMEM-format tables, their summary, and derived columns.

The table is kept as a pandas DataFrame with NONMEM column names. PKPy2 reads a
normalized copy (pkpy2.read_nonmem), so every transformation the agent makes is
recorded and reproducible.
"""
import ast
import math
import operator
from pathlib import Path

import numpy as np
import pandas as pd

STANDARD = ['ID', 'TIME', 'AMT', 'DV', 'EVID', 'MDV', 'CMT', 'RATE', 'II', 'SS', 'ADDL', 'DVID', 'CENS', 'LIMIT']
NON_COVARIATES = set(STANDARD) | {'OCC', 'BLQ', 'LLOQ', 'TAD', 'NTIME', 'NOMTIME', 'C', 'COMMENT', 'DATE', 'DAT1',
                                   'DAT2', 'DAT3', 'L2', 'PRED', 'IPRED', 'RES', 'WRES', 'CWRES'}


def read_table(path):
    df = pd.read_csv(path, na_values=['.', ''], keep_default_na=True, skipinitialspace=True)
    df.columns = [str(c).strip().lstrip('#').strip() for c in df.columns]
    upper = {c: c.upper() for c in df.columns if c.upper() in STANDARD or c.upper() == 'OCC'}
    return df.rename(columns=upper)


class Dataset:
    """A NONMEM-format analysis table with a record of the transformations applied to it."""

    def __init__(self, path, description=''):
        self.source = Path(path)
        self.df = read_table(path)
        self.description = description
        self.history = []
        missing = [c for c in ('ID', 'TIME', 'DV') if c not in self.df.columns]
        if missing:
            raise ValueError(f'the data need columns {missing} (NONMEM names ID, TIME, DV, AMT, ...)')
        if 'AMT' not in self.df.columns:
            self.df['AMT'] = 0.

    # ------------------------------------------------------------------ records
    def _evid(self):
        if 'EVID' in self.df:
            return self.df['EVID'].fillna(0).to_numpy()
        return np.where(self.df['AMT'].fillna(0).to_numpy() > 0, 1, 0)

    def dose_mask(self):
        return np.isin(self._evid(), (1, 4)) & (self.df['AMT'].fillna(0).to_numpy() > 0)

    def obs_mask(self):
        evid = self._evid()
        mdv = self.df['MDV'].fillna(0).to_numpy() if 'MDV' in self.df else np.zeros(len(self.df))
        return (evid == 0) & (mdv == 0) & self.df['DV'].notna().to_numpy()

    def covariate_columns(self):
        cols = []
        for c in self.df.columns:
            if c.upper() in NON_COVARIATES:
                continue
            if pd.api.types.is_numeric_dtype(self.df[c]):
                cols.append(c)
        return cols

    # ------------------------------------------------------------------ summary
    def describe(self):
        df = self.df
        obs, dose = self.obs_mask(), self.dose_mask()
        ids = df['ID'].unique()
        per_subject_obs = df[obs].groupby('ID').size().reindex(ids, fill_value=0)
        per_subject_dose = df[dose].groupby('ID').size().reindex(ids, fill_value=0)
        out = dict(file=self.source.name, description=self.description, records=int(len(df)), subjects=int(len(ids)),
                   observations=int(obs.sum()), dose_records=int(dose.sum()),
                   observations_per_subject=_range(per_subject_obs), doses_per_subject=_range(per_subject_dose),
                   time_range=[float(df['TIME'].min()), float(df['TIME'].max())],
                   columns=list(df.columns))
        d = df[dose]
        out['doses'] = dict(amounts=_levels(d['AMT']), compartments=_levels(d['CMT']) if 'CMT' in d else [1],
                            infusions=int((d['RATE'].fillna(0) != 0).sum()) if 'RATE' in d else 0,
                            steady_state_records=int((d['SS'].fillna(0) != 0).sum()) if 'SS' in d else 0,
                            additional_doses=int(d['ADDL'].fillna(0).sum()) if 'ADDL' in d else 0)
        if out['doses']['infusions']:
            rates = d['RATE'].fillna(0)
            dur = (d['AMT'] / rates.where(rates > 0)).dropna()
            out['doses']['infusion_durations_h'] = _levels(dur.round(3)) if len(dur) else []
            out['doses']['modeled_rate_records'] = int((rates < 0).sum())
        o = df[obs]
        out['observations_summary'] = dict(dv_range=[float(o['DV'].min()), float(o['DV'].max())],
                                           nonpositive_dv=int((o['DV'] <= 0).sum()),
                                           censored=int((o['CENS'].fillna(0) != 0).sum()) if 'CENS' in o else 0,
                                           outputs=_levels(o['DVID']) if 'DVID' in o else [1])
        if 'DVID' in o and o['DVID'].nunique() > 1:
            out['observations_summary']['by_output'] = {
                str(int(k)): dict(n=int(len(g)), dv_range=[float(g['DV'].min()), float(g['DV'].max())])
                for k, g in o.groupby('DVID')}
        # time after the most recent dose, pooled: where sampling happens
        tad = _time_after_dose(df, dose, obs)
        if len(tad):
            out['sampling_time_after_dose'] = dict(quantiles=dict(zip(['min', 'q25', 'median', 'q75', 'max'],
                                                                      np.round(np.quantile(tad, [0, .25, .5, .75, 1]), 3).tolist())),
                                                   distinct_times=int(len(np.unique(np.round(tad, 2)))))
        out['covariates'] = {c: self._covariate_summary(c) for c in self.covariate_columns()}
        if 'OCC' in df:
            out['occasions'] = _levels(df['OCC'])
        out['transformations'] = self.history
        return out

    def _covariate_summary(self, c):
        s = self.df[c]
        first = self.df.groupby('ID')[c].first()
        varies = bool((self.df.groupby('ID')[c].nunique(dropna=True) > 1).any())
        levels = s.dropna().unique()
        info = dict(missing_records=int(s.isna().sum()), time_varying=varies)
        if len(levels) <= 6:
            counts = first.value_counts(dropna=False).sort_index()
            info.update(type='categorical', subjects_by_level={_key(k): int(v) for k, v in counts.items()})
        else:
            q = np.nanquantile(first.to_numpy(dtype=float), [0, .5, 1])
            info.update(type='continuous', subject_min=float(q[0]), subject_median=float(q[1]), subject_max=float(q[2]))
        return info

    # ------------------------------------------------------------------ derived columns
    def add_column(self, name, expression, reason=''):
        """name = expression over existing columns (+, -, *, /, **, log, exp, sqrt, abs, min, max, where)."""
        name = str(name).strip()
        if not name.isidentifier():
            raise ValueError('the new column name must be an identifier')
        values = _evaluate(expression, self.df)
        values = pd.Series(np.broadcast_to(values, (len(self.df),)).astype(float), index=self.df.index)
        replaced = name in self.df.columns
        self.df[name] = values
        self.history.append(dict(column=name, expression=expression, replaced=replaced, reason=reason))
        return dict(column=name, replaced=replaced, summary=_range(values.dropna()), missing=int(values.isna().sum()))

    # ------------------------------------------------------------------ PKPy2 view
    def write_normalized(self, path):
        df = self.df.copy()
        if 'EVID' not in df:
            df['EVID'] = self._evid()
        if 'MDV' not in df:
            df['MDV'] = np.where((df['EVID'] != 0) | df['DV'].isna(), 1, 0)
        df['DV'] = df['DV'].where(df['MDV'] == 0)          # dose-record DV values are not observations
        df.to_csv(path, index=False, na_rep='.')
        return path

    def to_pkpy2(self, path, covariates=(), occasion=None):
        import pkpy2
        self.write_normalized(path)
        covariates = [c for c in covariates if c]
        unknown = [c for c in covariates if c not in self.df.columns]
        if unknown:
            raise ValueError(f'unknown covariate columns {unknown}; available: {self.covariate_columns()}')
        return pkpy2.read_nonmem(path, covariates=list(dict.fromkeys(covariates)),
                                 occasion=occasion if occasion else None)


# ---------------------------------------------------------------------- helpers
def _range(s):
    s = pd.Series(s).astype(float)
    if not len(s):
        return {}
    return dict(min=float(s.min()), median=float(s.median()), max=float(s.max()))


def _levels(s, limit=12):
    v = sorted(pd.Series(s).dropna().unique().tolist())
    v = [int(x) if float(x).is_integer() else float(x) for x in v]
    return v if len(v) <= limit else v[:limit] + [f'... ({len(v)} distinct values)']


def _key(k):
    if isinstance(k, float) and math.isnan(k):
        return 'missing'
    return str(int(k)) if float(k).is_integer() else str(k)


def _time_after_dose(df, dose, obs):
    out = []
    for _, g in df.assign(_dose=dose, _obs=obs).groupby('ID', sort=False):
        dt = g.loc[g['_dose'], 'TIME'].to_numpy()
        for t in g.loc[g['_obs'], 'TIME'].to_numpy():
            prior = dt[dt <= t]
            if len(prior):
                out.append(t - prior.max())
    return np.asarray(out)


_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow}
_CMP = {ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Lt: operator.lt, ast.LtE: operator.le, ast.Eq: operator.eq,
        ast.NotEq: operator.ne}
_FUN = {'log': np.log, 'exp': np.exp, 'sqrt': np.sqrt, 'abs': np.abs, 'min': np.minimum, 'max': np.maximum,
        'where': np.where}


def _evaluate(expression, df):
    """Arithmetic over columns without Python evaluation (a small AST interpreter)."""
    tree = ast.parse(str(expression), mode='eval')

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in df.columns:
                raise ValueError(f'unknown column {node.id}')
            return df[node.id].to_numpy(dtype=float)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            return _BIN[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMP:
            return _CMP[type(node.ops[0])](ev(node.left), ev(node.comparators[0])).astype(float)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUN and not node.keywords:
            return _FUN[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f'unsupported expression element: {ast.dump(node)[:60]}')
    with np.errstate(all='ignore'):
        return ev(tree)
