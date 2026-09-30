"""The tools the agent calls (OpenAI/OpenRouter function-calling format) and their execution."""
import json

from .session import BudgetExhausted
from .spec import SPEC_SCHEMA, SpecError


def _fn(name, description, properties=None, required=()):
    return dict(type='function', function=dict(name=name, description=description, parameters=dict(
        type='object', properties=properties or {}, required=list(required))))


MODEL_IDS = dict(type='array', items=dict(type='string'))

TOOLS = [
    _fn('describe_data', 'Summary of the analysis data: subjects, records, doses (routes, infusions, steady state), '
                         'observations per subject, time-after-dose sampling, outputs, covariates (type, range, '
                         'time-varying) and data transformations so far.'),
    _fn('run_nca', 'Non-compartmental analysis of the first dosing interval of each subject (Cmax, Tmax, AUC, terminal '
                   'half-life, CL/F, V/F), pooled check for multiphasic decline, and starting values for a first model. '
                   'Sparse or multiple-dose data may not allow per-subject NCA.',
        dict(output=dict(type='integer', description='DVID of the output (default 1)'))),
    _fn('plot_data', 'Concentration-time plot of the observations (by time and by time after dose); you receive the image.',
        dict(log_scale=dict(type='boolean'), color_by=dict(type='string', description='covariate column for coloring'),
             output=dict(type='integer'))),
    _fn('add_data_column', 'Add (or replace) a column computed from existing columns, e.g. an indicator "APGR < 5", a '
                           'derived covariate "WT/HT**2", or a unit conversion. Allowed: + - * / ** comparisons log exp '
                           'sqrt abs min max where. Models fitted before a change of existing columns are not comparable '
                           'with later ones.',
        dict(name=dict(type='string'), expression=dict(type='string'), reason=dict(type='string')),
        ('name', 'expression', 'reason')),
    _fn('fit_models', 'Fit one or more model specifications with PKPy2. Default estimation is the Laplace '
                      '(FOCE-I-type) objective, minimized from the specified starting values; a fit is converged when '
                      'the remaining achievable OFV decrease (Newton decrement) is small. Several specifications are '
                      'fitted in parallel; each fit takes about 1-30 minutes. Returns for each: model_id, OFV (-2 log L with '
                      'normal constants), AIC, BIC, estimates with RSE% and 95% CI, IIV (variance, CV%, shrinkage), '
                      'residual SDs, CWRES/NPDE summaries and warnings. Use "parent" to record which model a new one '
                      'modifies.',
        dict(models=dict(type='array', items=SPEC_SCHEMA, minItems=1, maxItems=6),
             standard_errors=dict(type='boolean', description='compute RSE and CI (default true; false is faster)'),
             estimation=dict(type='string', enum=['laplace', 'importance'],
                             description='laplace (default) or importance: importance-sampled exact marginal '
                                         'likelihood with a two-bank audit, much slower and not comparable in OFV '
                                         'with Laplace fits')),
        ('models',)),
    _fn('list_models', 'All fitted models with status, OFV, number of estimated parameters, AIC and parent.'),
    _fn('get_model', 'Full summary of one fitted model, including its complete specification (useful as a template '
                     'for the next model).', dict(model_id=dict(type='string')), ('model_id',)),
    _fn('compare_models', 'Compare fitted models: OFV, parameters, AIC, BIC, and for each model versus the reference the '
                          'OFV difference and the likelihood-ratio p-value if the models are nested.',
        dict(model_ids=MODEL_IDS, reference_model_id=dict(type='string')), ('model_ids',)),
    _fn('view_plots', 'Look at diagnostic plots of a fitted model: "gof" (DV vs PRED/IPRED, CWRES vs time/PRED), '
                      '"individual" (observed, PRED, IPRED for up to 12 subjects), "vpc" (after run_vpc), '
                      '"eta_covariates" (after screen_covariates).',
        dict(model_id=dict(type='string'), plots=dict(type='array', items=dict(
            type='string', enum=['gof', 'individual', 'vpc', 'eta_covariates']))), ('model_id', 'plots')),
    _fn('screen_covariates', 'Empirical Bayes estimates (etas) of a fitted model against subject covariates: Spearman '
                             'correlations or level differences, p-values, and an approximate power exponent. Screening '
                             'only; confirm effects by fitting.',
        dict(model_id=dict(type='string'), covariates=dict(type='array', items=dict(type='string'))), ('model_id',)),
    _fn('covariate_search', 'Stepwise covariate modeling from a converged base model: forward inclusion (p < forward_p) '
                            'then backward elimination (p < backward_p to stay), one degree of freedom per effect. Every '
                            'trial fit is registered and counts against the fit budget.',
        dict(base_model_id=dict(type='string'),
             candidates=dict(type='array', items=SPEC_SCHEMA['properties']['covariates']['items']),
             forward_p=dict(type='number'), backward_p=dict(type='number')), ('base_model_id', 'candidates')),
    _fn('run_vpc', 'Visual predictive check of a fitted model (500 simulations): observed 5th/50th/95th percentiles '
                   'against the 95% intervals of the simulated percentiles, per time bin and output.',
        dict(model_id=dict(type='string'), prediction_corrected=dict(type='boolean'), log_scale=dict(type='boolean'),
             bins=dict(type='integer')), ('model_id',)),
    _fn('resample_uncertainty', 'Nonparametric bootstrap (refits on resampled subjects) or sampling importance '
                                'resampling (SIR) for the uncertainty of a final model. Slow; use once at the end.',
        dict(model_id=dict(type='string'), method=dict(type='string', enum=['bootstrap', 'sir']),
             n=dict(type='integer', description='bootstrap replicates (default 100)')), ('model_id', 'method')),
    _fn('finalize_model', 'Submit the final model and the written report. Ends the analysis.',
        dict(model_id=dict(type='string'),
             report=dict(type='object', properties=dict(
                 summary=dict(type='string', description='the final model in a few sentences'),
                 development=dict(type='string', description='how the model was built: key steps, tests and decisions'),
                 expert_knowledge=dict(type='string', description='how the analyst information was used, and whether '
                                                                  'the data supported it'),
                 evaluation=dict(type='string', description='goodness of fit, VPC, precision, shrinkage'),
                 limitations=dict(type='string')),
                 required=['summary', 'development', 'evaluation', 'limitations'])),
        ('model_id', 'report')),
]


def execute(session, name, args):
    """Run one tool; returns (result for the agent, list of image paths)."""
    images = []
    try:
        if name == 'describe_data':
            out = session.dataset.describe()
        elif name == 'run_nca':
            from .nca import run_nca
            session.nca = run_nca(session.dataset, output=int(args.get('output', 1)))
            out = dict(session.nca)
            if 'per_subject' in out:
                out['per_subject'] = out['per_subject'][:15]
                out['per_subject_note'] = 'first 15 subjects shown'
        elif name == 'plot_data':
            path = session.plot_data(log_scale=args.get('log_scale', True), color_by=args.get('color_by'),
                                     output=int(args.get('output', 1)))
            out = dict(plot=path.name)
            images.append(path)
        elif name == 'add_data_column':
            out = session.dataset.add_column(args['name'], args['expression'], args.get('reason', ''))
            session.data_version += 1
            session._write_data()
            out['data_version'] = session.data_version
        elif name == 'fit_models':
            models = args.get('models') or []
            if not isinstance(models, list) or not models:
                raise ValueError('models must be a non-empty list of specifications')
            if len(models) > 6:
                raise ValueError('at most 6 models per call')
            recs = session.fit(models, uncertainty=args.get('standard_errors', True),
                               estimation=args.get('estimation') or 'laplace')
            out = dict(results=[session.compact(r) if 'model_id' in r else r for r in recs])
        elif name == 'list_models':
            out = dict(models=[dict(model_id=r['model_id'], name=r['name'], parent=r['parent'], status=r['status'],
                                    ofv=r.get('ofv'), n_estimated=r.get('n_estimated'),
                                    aic=(r.get('summary') or {}).get('aic'), description=r['description'])
                               for r in session.models.values()])
        elif name == 'get_model':
            rec = session.models.get(args['model_id'])
            if rec is None:
                raise ValueError(f"unknown model {args['model_id']}")
            out = session.compact(rec, detail=True)
            out['plots_available'] = session.plots_of(rec['model_id'])
        elif name == 'compare_models':
            out = session.compare(args['model_ids'], args.get('reference_model_id'))
        elif name == 'view_plots':
            mid = args['model_id']
            folder = session.out / 'models' / mid
            found, missing = [], []
            for kind in args.get('plots', []):
                paths = sorted(folder.glob({'gof': 'gof_*.png', 'individual': 'individual_*.png', 'vpc': 'vpc_*.png',
                                            'eta_covariates': 'eta_covariates.png'}.get(kind, '__none__')))
                (found if paths else missing).extend([p.name for p in paths] if paths else [kind])
                images.extend(paths)
            out = dict(plots=found, missing=missing)
            if missing:
                out['hint'] = 'vpc needs run_vpc first; eta_covariates needs screen_covariates first'
        elif name == 'screen_covariates':
            out = session.screen(args['model_id'], args.get('covariates'))
        elif name == 'covariate_search':
            out = session.covariate_search(args['base_model_id'], args['candidates'], float(args.get('forward_p', .05)),
                                           float(args.get('backward_p', .01)))
            final = session.models.get(out['final_model_id'])
            if final:
                out['final_model'] = session.compact(final)
        elif name == 'run_vpc':
            out = session.vpc(args['model_id'], bool(args.get('prediction_corrected', False)),
                              bool(args.get('log_scale', False)), bins=int(args.get('bins', 8)))
            for v in out.values():
                if isinstance(v, dict) and v.get('plot'):
                    images.append(session.out / 'models' / args['model_id'] / v['plot'].split('\\')[-1].split('/')[-1])
                    v['plot'] = images[-1].name
        elif name == 'resample_uncertainty':
            out = session.resample(args['model_id'], args['method'], **({'n': int(args['n'])} if args.get('n') else {}))
        elif name == 'finalize_model':
            rec = session.models.get(args['model_id'])
            if rec is None or rec['status'] != 'converged':
                raise ValueError('finalize_model needs a converged model_id')
            session.final = dict(model_id=args['model_id'], report=args.get('report') or {})
            out = dict(accepted=True, model_id=args['model_id'])
        else:
            raise ValueError(f'unknown tool {name}')
    except BudgetExhausted as e:
        out = dict(error=str(e), budget_exhausted=True)
    except (SpecError, ValueError, KeyError, TypeError) as e:
        out = dict(error=f'{type(e).__name__}: {e}')
    except Exception as e:                                        # noqa: BLE001 - never stop the session
        out = dict(error=f'{type(e).__name__}: {e}')
    if isinstance(out, dict):
        out['budget'] = session.budget_status()
    return out, [p for p in images if p.exists()]


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'), default=str)
