"""Model specifications: the JSON the agent writes, checked and compiled into a PKPy2 Model.

A specification never contains code. Every structural model comes from the PKPy2
library; the agent chooses among its options and gives starting values, fixed
terms, bounds, random effects, covariate effects and residual models.
"""
import copy
import json

STRUCTURE_TYPES = ('pk', 'michaelis_menten', 'indirect_response', 'tmdd', 'parent_metabolite')
ABSORPTION = {'iv': None, 'intravenous': None, 'bolus': None, 'infusion': None, None: None,
              'first_order': 'first_order', 'zero_order': 'zero_order', 'transit': 'transit'}
PD_TYPES = (None, 'emax', 'sigmoid_emax', 'imax', 'sigmoid_imax', 'linear')
LOGIT = {'F', 'FM', 'IMAX'}
COVARIATE_FORMS = ('power', 'exponential', 'linear', 'categorical')

SPEC_SCHEMA = {
    'type': 'object',
    'description': 'A population model. Unspecified structural parameters get starting values from NCA or defaults.',
    'properties': {
        'name': {'type': 'string', 'description': 'short label, e.g. "2cmt oral lag, WT on CL"'},
        'parent': {'type': 'string', 'description': 'model_id of the model this one modifies (for the development log)'},
        'structure': {
            'type': 'object',
            'properties': {
                'type': {'type': 'string', 'enum': list(STRUCTURE_TYPES), 'default': 'pk'},
                'compartments': {'type': 'integer', 'enum': [1, 2, 3], 'default': 1},
                'absorption': {'type': 'string', 'enum': ['iv', 'first_order', 'zero_order', 'transit'],
                               'description': 'iv = doses enter the central compartment (bolus, or infusion from RATE)'},
                'transit_compartments': {'type': 'integer', 'minimum': 1},
                'lag': {'type': 'boolean'},
                'bioavailability': {'type': 'boolean', 'description': 'estimate F (logit scale); otherwise F = 1'},
                'infusion_parameter': {'type': 'string', 'enum': ['rate', 'duration'],
                                       'description': 'modeled infusion rate R1 or duration D1 for RATE=-1/-2 records'},
                'pd': {'type': 'string', 'enum': [p for p in PD_TYPES if p],
                       'description': 'direct PD output E driven by plasma (or effect-compartment) concentration'},
                'effect_compartment': {'type': 'boolean', 'description': 'PD acts on an effect compartment (KE0)'},
                'linear_clearance': {'type': 'boolean', 'description': 'michaelis_menten: keep a parallel linear CL'},
                'response_type': {'type': 'integer', 'enum': [1, 2, 3, 4],
                                  'description': 'indirect_response I-IV (I inhibition of production, II inhibition of '
                                                 'loss, III stimulation of production, IV stimulation of loss)'},
                'sigmoid': {'type': 'boolean', 'description': 'indirect_response: Hill coefficient GAMMA'},
                'tmdd_kind': {'type': 'string', 'enum': ['full', 'qss']},
                'metabolite_compartments': {'type': 'integer', 'enum': [1, 2]},
            },
        },
        'parameters': {'type': 'object', 'description': 'typical values: {"CL": 3.1} or {"CL": {"value": 3.1, "lower": 0.1, '
                                                        '"upper": 50, "fixed": false, "scale": "log"}}',
                       'additionalProperties': True},
        'iiv': {'type': 'object', 'description': 'interindividual VARIANCES (log scale for log-normal parameters), e.g. '
                                                 '{"CL": 0.1, "V": 0.1}; {"Ka": {"value": 0.5, "fixed": true}} fixes one',
                'additionalProperties': True},
        'iiv_blocks': {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'string'}},
                       'description': 'groups of parameters with correlated random effects, e.g. [["CL", "V"]]'},
        'iov': {'type': 'object', 'description': 'interoccasion variances (needs occasion_column)', 'additionalProperties': True},
        'occasion_column': {'type': 'string'},
        'covariates': {'type': 'array', 'items': {
            'type': 'object',
            'properties': {
                'parameter': {'type': 'string'}, 'covariate': {'type': 'string'},
                'form': {'type': 'string', 'enum': list(COVARIATE_FORMS),
                         'description': 'power: (z/center)^b; exponential: exp(b (z-center)); linear: 1 + b (z-center); '
                                        'categorical: exp(b) when z == level'},
                'center': {'type': 'number', 'description': 'reference value (continuous) '},
                'level': {'type': 'number', 'description': 'category level that receives the effect (categorical)'},
                'value': {'type': 'number', 'description': 'starting (or fixed) coefficient'},
                'fixed': {'type': 'boolean'}, 'lower': {'type': 'number'}, 'upper': {'type': 'number'},
            },
            'required': ['parameter', 'covariate', 'form']}},
        'residual': {'type': 'object', 'description': 'per output (CP, E, R, CM, RTOT ...): {"CP": {"proportional": 0.2, '
                                                      '"additive": 0.1}} or {"CP": {"lognormal": 0.2}}; with one output '
                                                      '{"proportional": 0.2} is accepted. Values are SDs.',
                     'additionalProperties': True},
    },
    'required': ['structure'],
}


class SpecError(ValueError):
    pass


def _param(v, name=''):
    """Accept a number or {value, fixed, lower, upper, scale}."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return dict(value=float(v), fixed=False, lower=None, upper=None, scale=None)
    if isinstance(v, dict) and 'value' in v:
        extra = set(v) - {'value', 'fixed', 'lower', 'upper', 'scale', 'estimate'}
        if extra:
            raise SpecError(f'{name}: unknown keys {sorted(extra)} (use value, fixed, lower, upper, scale)')
        out = dict(value=float(v['value']), fixed=bool(v.get('fixed', False)),
                   lower=None if v.get('lower') is None else float(v['lower']),
                   upper=None if v.get('upper') is None else float(v['upper']), scale=v.get('scale'))
        if out['lower'] is not None and out['upper'] is not None and out['lower'] >= out['upper']:
            raise SpecError(f'{name}: lower must be below upper')
        return out
    raise SpecError(f'{name}: give a number or an object with "value"')


def build_structure(s):
    from pkpy2 import structures as S
    s = dict(s or {})
    kind = s.get('type', 'pk')
    if kind not in STRUCTURE_TYPES:
        raise SpecError(f'structure.type must be one of {STRUCTURE_TYPES}')
    cmt = int(s.get('compartments', 1))
    if cmt not in (1, 2, 3):
        raise SpecError('structure.compartments must be 1, 2 or 3')
    if s.get('absorption') not in ABSORPTION:
        raise SpecError('structure.absorption must be iv, first_order, zero_order or transit')
    absorption = ABSORPTION[s.get('absorption')]
    lag, bio = bool(s.get('lag', False)), bool(s.get('bioavailability', False))
    try:
        if kind == 'pk':
            pd = s.get('pd')
            if pd not in PD_TYPES:
                raise SpecError(f'structure.pd must be one of {PD_TYPES[1:]}')
            return S.pk(cmt, absorption, transit=int(s.get('transit_compartments', 0) or 0), lag=lag,
                        bioavailability=bio, effect='compartment' if s.get('effect_compartment') else None, pd=pd,
                        infusion_parameter=s.get('infusion_parameter'))
        if kind == 'michaelis_menten':
            return S.michaelis_menten(cmt, absorption, lag=lag, bioavailability=bio,
                                      linear_clearance=bool(s.get('linear_clearance', False)))
        if kind == 'indirect_response':
            if s.get('response_type') not in (1, 2, 3, 4):
                raise SpecError('indirect_response needs structure.response_type 1-4')
            base = S.pk(cmt, absorption, transit=int(s.get('transit_compartments', 0) or 0), lag=lag,
                        bioavailability=bio)
            return S.indirect_response(int(s['response_type']), base, sigmoid=bool(s.get('sigmoid', False)))
        if kind == 'tmdd':
            return S.tmdd(s.get('tmdd_kind', 'full'), cmt, absorption, lag=lag, bioavailability=bio)
        return S.parent_metabolite(cmt, absorption, metabolite_cmt=int(s.get('metabolite_compartments', 1)), lag=lag,
                                   bioavailability=bio)
    except SpecError:
        raise
    except (ValueError, TypeError) as e:
        raise SpecError(f'structure: {e}') from e


def structure_info(s):
    st = build_structure(s)
    return dict(parameters=list(st.parameters), outputs=[o.name for o in st.outputs], name=st.name)


GENERIC_DEFAULTS = dict(CL=1., V=10., V1=10., Q=1., V2=20., Q2=1., Q3=.5, V3=20., Ka=1., ALAG=.25, F=.7, D1=1.,
                        MTT=1., R1=10., VMAX=10., KM=1., KE0=.5, E0=1., EMAX=1., EC50=1., IMAX=.8, IC50=1.,
                        GAMMA=1., SLOPE=.1, R0=100., KOUT=.1, KON=1., KOFF=1., KINT=.1, KDEG=.1, KSS=1., FM=1.,
                        CLM=1., VM=10., QM=1., VM2=20.)


def default_values(parameters, suggestions):
    """Starting values for parameters the agent left out: NCA suggestions, then generic values."""
    s = dict(suggestions or {})
    if 'V' in s:
        s.setdefault('V1', s['V'])
    if 'CL' in s:
        s.setdefault('Q', s['CL']); s.setdefault('Q2', s['CL']); s.setdefault('Q3', s['CL'] / 2)
    if 'V1' in s:
        s.setdefault('V2', 2 * s['V1']); s.setdefault('V3', 2 * s['V1'])
    return {p: float(s.get(p, GENERIC_DEFAULTS.get(p, 1.))) for p in parameters}


def normalize(spec, suggestions=None):
    """Validated, fully explicit copy of a specification (missing starting values filled in)."""
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError as e:
            raise SpecError(f'specification is not valid JSON: {e}') from e
    if not isinstance(spec, dict):
        raise SpecError('specification must be a JSON object')
    known = set(SPEC_SCHEMA['properties'])
    extra = set(spec) - known
    if extra:
        raise SpecError(f'unknown specification keys {sorted(extra)}; allowed: {sorted(known)}')
    spec = copy.deepcopy(spec)
    info = structure_info(spec.get('structure', {}))
    params, outputs = info['parameters'], info['outputs']
    given = spec.get('parameters') or {}
    unknown = sorted(set(given) - set(params))
    if unknown:
        raise SpecError(f'parameters {unknown} are not in this structure; its parameters are {params}')
    defaults = default_values(params, suggestions)
    theta, defaulted = {}, []
    for p in params:
        if p in given:
            theta[p] = _param(given[p], p)
        else:
            theta[p] = _param(defaults[p], p)
            defaulted.append(p)
        if theta[p]['scale'] is None:
            theta[p]['scale'] = 'logit' if p in LOGIT else 'log'
        if theta[p]['scale'] not in ('log', 'logit', 'identity'):
            raise SpecError(f'{p}: scale must be log, logit or identity')
        v = theta[p]['value']
        if theta[p]['scale'] == 'log' and v <= 0:
            raise SpecError(f'{p}: a log-scale parameter needs a positive value (got {v})')
        if theta[p]['scale'] == 'logit' and not 0 < v < 1:
            raise SpecError(f'{p}: a logit-scale parameter needs a value in (0, 1) (got {v})')
    iiv = {}
    for p, v in (spec.get('iiv') or {}).items():
        if p not in params:
            raise SpecError(f'iiv on unknown parameter {p}; parameters: {params}')
        iiv[p] = _param(v, f'iiv.{p}')
        if iiv[p]['value'] <= 0:
            raise SpecError(f'iiv.{p}: a variance must be positive')
    blocks = [list(b) for b in (spec.get('iiv_blocks') or [])]
    for b in blocks:
        if len(b) < 2 or any(p not in iiv for p in b):
            raise SpecError(f'iiv_blocks {b}: each block needs two or more parameters that have iiv')
    iov = {p: _param(v, f'iov.{p}') for p, v in (spec.get('iov') or {}).items()}
    if iov and not spec.get('occasion_column'):
        raise SpecError('iov needs occasion_column (the data column with occasion numbers)')
    for p in iov:
        if p not in params:
            raise SpecError(f'iov on unknown parameter {p}')
    covs = []
    for c in spec.get('covariates') or []:
        c = dict(c)
        if c.get('parameter') not in params:
            raise SpecError(f'covariate effect on unknown parameter {c.get("parameter")}; parameters: {params}')
        form = c.get('form', 'power')
        if form not in COVARIATE_FORMS:
            raise SpecError(f'covariate form must be one of {COVARIATE_FORMS}')
        if form == 'categorical':
            if c.get('level') is None:
                raise SpecError(f'categorical effect of {c.get("covariate")} needs "level" (the category with the effect)')
            c['center'] = c.get('center', 0.)
        elif c.get('center') is None:
            raise SpecError(f'{form} effect of {c.get("covariate")} needs "center" (for example the median)')
        if form == 'power' and float(c['center']) <= 0:
            raise SpecError('a power effect needs a positive center')
        coef = _param(dict(value=c.get('value', .01 if form != 'power' else .5), fixed=c.get('fixed', False),
                           lower=c.get('lower'), upper=c.get('upper')), f'covariate {c["parameter"]}~{c["covariate"]}')
        covs.append(dict(parameter=c['parameter'], covariate=str(c['covariate']), form=form, center=float(c['center']),
                         level=None if c.get('level') is None else float(c['level']), coefficient=coef))
    keys = [(c['parameter'], c['covariate'], c['level']) for c in covs]
    if len(set(keys)) != len(keys):
        raise SpecError('the same covariate effect is declared twice')
    res = spec.get('residual') or {'proportional': .2}
    if set(res) <= {'proportional', 'additive', 'lognormal'}:
        if len(outputs) > 1:
            raise SpecError(f'this structure has outputs {outputs}; give a residual model per output, e.g. '
                            f'{{"{outputs[0]}": {{"proportional": 0.2}}}}; outputs without observations may be omitted')
        res = {outputs[0]: res}
    residual = {}
    for o, r in res.items():
        if o not in outputs:
            raise SpecError(f'residual for unknown output {o}; outputs: {outputs}')
        if not isinstance(r, dict) or not r or set(r) - {'proportional', 'additive', 'lognormal'}:
            raise SpecError(f'residual.{o} must use proportional, additive and/or lognormal')
        if 'lognormal' in r and len(r) > 1:
            raise SpecError(f'residual.{o}: lognormal cannot be combined with other components')
        residual[o] = {k: _param(v, f'residual.{o}.{k}') for k, v in r.items()}
    missing_outputs = [o for o in outputs if o not in residual]
    if missing_outputs and len(outputs) > 1:
        # outputs without data (e.g. a response that is not observed) keep a nominal fixed error
        for o in missing_outputs:
            residual[o] = {'additive': dict(value=1., fixed=True, lower=None, upper=None, scale=None)}
    return dict(name=spec.get('name') or '', parent=spec.get('parent'), structure=dict(spec.get('structure', {})),
                parameters=theta, iiv=iiv, iiv_blocks=blocks, iov=iov, occasion_column=spec.get('occasion_column'),
                covariates=covs, residual=residual, defaulted=defaulted, outputs=outputs)


def compile_model(norm):
    """pkpy2.Model from a normalized specification."""
    from pkpy2 import Model, Residual, Covariate, Parameter as P

    def par(d):
        return P(d['value'], d['fixed'], d['lower'], d['upper'])
    structure = build_structure(norm['structure'])
    transforms = {p: d['scale'] for p, d in norm['parameters'].items() if d['scale'] != 'log'}
    residual = {o: Residual(**{k: par(v) for k, v in r.items()}) for o, r in norm['residual'].items()}
    try:
        return Model(structure,
                     theta={p: par(d) for p, d in norm['parameters'].items()},
                     omega={p: par(d) for p, d in norm['iiv'].items()},
                     omega_blocks=tuple(tuple(b) for b in norm['iiv_blocks']),
                     iov={p: par(d) for p, d in norm['iov'].items()},
                     covariates=tuple(Covariate(c['parameter'], c['covariate'], c['center'], par(c['coefficient']),
                                                c['form'], c['level']) for c in norm['covariates']),
                     residual=residual if len(residual) > 1 else next(iter(residual.values())),
                     transforms=transforms)
    except (ValueError, TypeError) as e:
        raise SpecError(str(e)) from e


def covariate_columns(norm):
    return sorted({c['covariate'] for c in norm['covariates']})


def describe(norm):
    s = norm['structure']
    kind = s.get('type', 'pk')
    parts = [f"{s.get('compartments', 1)}-cmt", {None: 'IV', 'iv': 'IV'}.get(s.get('absorption'), s.get('absorption'))]
    if s.get('transit_compartments'):
        parts.append(f"{s['transit_compartments']} transit")
    if s.get('lag'):
        parts.append('lag')
    if s.get('bioavailability'):
        parts.append('F')
    if kind != 'pk':
        parts.append(kind + (f" type {s.get('response_type')}" if kind == 'indirect_response' else ''))
    if s.get('pd'):
        parts.append(f"PD {s['pd']}" + (' (effect cmt)' if s.get('effect_compartment') else ''))
    text = ', '.join(p for p in parts if p)
    iiv = list(norm['iiv'])
    text += f"; IIV {'+'.join(iiv) if iiv else 'none'}"
    if norm['iiv_blocks']:
        text += ' block(' + '; '.join(','.join(b) for b in norm['iiv_blocks']) + ')'
    if norm['iov']:
        text += f"; IOV {'+'.join(norm['iov'])}"
    text += '; error ' + '; '.join(f"{o}:{'+'.join(r)}" for o, r in norm['residual'].items())
    if norm['covariates']:
        text += '; covariates ' + ', '.join(f"{c['covariate']}{'=' + str(c['level']) if c['level'] is not None else ''}"
                                            f"~{c['parameter']}({c['form']})" for c in norm['covariates'])
    fixed = [p for p, d in norm['parameters'].items() if d['fixed']]
    if fixed:
        text += '; fixed ' + ', '.join(f"{p}={norm['parameters'][p]['value']:g}" for p in fixed)
    return text
