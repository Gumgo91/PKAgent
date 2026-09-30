"""The final outputs of a session: report.md, final/ (specification, fit, plots) and results.json."""
import json
import shutil
import time
from pathlib import Path


def _fmt(v, d=4):
    if v is None:
        return '-'
    if isinstance(v, float):
        return f'{v:.{d}g}'
    return str(v)


def write_report(session, llm, settings, knowledge=None):
    out = Path(session.out)
    final = session.final
    lines = ['# PKAgent analysis report', '']
    info = dict(llm=settings.model, llm_calls=llm.usage['calls'], prompt_tokens=llm.usage['prompt_tokens'],
                completion_tokens=llm.usage['completion_tokens'], cached_tokens=llm.usage['cached_tokens'],
                cost_usd=round(llm.usage['cost_usd'], 4), fits=session.fits_used, models=len(session.models),
                hours=round((time.time() - session.t0) / 3600, 3), finalized=final is not None)
    lines += ['| Item | Value |', '|---|---|'] + [f'| {k} | {_fmt(v)} |' for k, v in info.items()] + ['']
    lines += ['## Data', '', session.dataset.description.strip() or session.dataset.source.name, '']
    if knowledge:
        lines += ['## Expert knowledge given to the agent', '', knowledge.strip(), '']
    result = dict(run=info, final_model=None, models=[])
    if final:
        mid = final['model_id']
        rec = session.models[mid]
        try:
            session.finalize_outputs(mid)
        except Exception as e:                                   # noqa: BLE001
            lines += [f'(final outputs incomplete: {type(e).__name__}: {e})', '']
        s = rec.get('summary') or {}
        lines += [f'## Final model: {mid}', '', f"**{rec['description']}**", '',
                  f"OFV {_fmt(s.get('ofv'), 8)} (-2 log L with normal constants), {s.get('n_estimated')} estimated "
                  f"parameters, AIC {_fmt(s.get('aic'), 8)}, BIC {_fmt(s.get('bic'), 8)}; status {s.get('status')}.", '']
        lines += ['| Parameter | Estimate | RSE (%) | 95% CI | Note |', '|---|---|---|---|---|']
        for r in s.get('parameters', []):
            note = 'fixed' if r.get('fixed') else (f"bounds {r['bounds']}" if r.get('bounds') else '')
            ci = f"{_fmt(r['ci95'][0])} to {_fmt(r['ci95'][1])}" if r.get('ci95') else '-'
            lines.append(f"| {r['parameter']} | {_fmt(r['estimate'])} | {_fmt(r.get('rse_percent'), 3)} | {ci} | {note} |")
        shrink = ((s.get('diagnostics') or {}).get('eta_shrinkage_percent')) or {}
        for r in s.get('iiv', []):
            ci = f"{_fmt(r['ci95'][0])} to {_fmt(r['ci95'][1])}" if r.get('ci95') else '-'
            lines.append(f"| IIV {r['parameter']} (variance) | {_fmt(r['variance'])} | {_fmt(r.get('rse_percent'), 3)} | {ci} | "
                         f"CV {_fmt(r.get('cv_percent'), 3)}%, shrinkage {_fmt(shrink.get(r['parameter']), 3)}% |")
        for r in s.get('covariate_effects', []):
            ci = f"{_fmt(r['ci95'][0])} to {_fmt(r['ci95'][1])}" if r.get('ci95') else '-'
            lines.append(f"| {r['effect']} | {_fmt(r['coefficient'])} | - | {ci} | coefficient |")
        for r in s.get('residual', []):
            if r.get('sd'):
                lines.append(f"| {r['component']} (SD) | {_fmt(r['sd'])} | {_fmt(r.get('rse_percent'), 3)} | - | |")
        lines.append('')
        report = final.get('report') or {}
        for key, title in (('summary', 'Summary'), ('development', 'Model development'),
                           ('expert_knowledge', 'Use of the expert knowledge'), ('evaluation', 'Evaluation'),
                           ('limitations', 'Limitations')):
            if report.get(key):
                lines += [f'### {title}', '', report[key].strip(), '']
        fdir = out / 'final'
        fdir.mkdir(exist_ok=True)
        src = out / 'models' / mid
        for p in src.iterdir():
            shutil.copy2(p, fdir / p.name)
        figs = sorted(p.name for p in fdir.glob('*.png'))
        if figs:
            lines += ['### Figures', ''] + [f'![{f}](final/{f})' for f in figs] + ['']
        result['final_model'] = dict(model_id=mid, description=rec['description'], summary=s, report=report,
                                     specification=rec['spec'], data_file=session.data_csv(rec['data_version']),
                                     vpc=rec.get('final_vpc'))
    else:
        lines += ['## No final model', '', 'The session ended without finalize_model.', '']
    lines += ['## All models', '', '| Model | Parent | Description | Status | OFV | dOFV vs parent | Estimated | AIC |',
              '|---|---|---|---|---|---|---|---|']
    for rec in session.models.values():
        s = rec.get('summary') or {}
        parent = session.models.get(rec['parent']) if rec['parent'] else None
        d = (s.get('ofv') - parent['ofv']) if parent and parent.get('ofv') is not None and s.get('ofv') is not None else None
        lines.append(f"| {rec['model_id']} | {rec['parent'] or ''} | {rec['description']} | {rec['status']} | "
                     f"{_fmt(s.get('ofv'), 8)} | {_fmt(d, 4)} | {s.get('n_estimated', '-')} | {_fmt(s.get('aic'), 8)} |")
        result['models'].append(dict(model_id=rec['model_id'], parent=rec['parent'], name=rec['name'],
                                     description=rec['description'], status=rec['status'], ofv=s.get('ofv'),
                                     n_estimated=s.get('n_estimated'), aic=s.get('aic'), bic=s.get('bic'),
                                     data_version=rec['data_version']))
    result['data_transformations'] = list(session.dataset.history or [])
    result['budget'] = session.budget_status()
    lines += ['', '## Data transformations', '']
    for h in session.dataset.history or []:
        lines.append(f"- {h['column']} = {h['expression']} ({h.get('reason', '')})")
    if not session.dataset.history:
        lines.append('none')
    path = out / 'report.md'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (out / 'results.json').write_text(json.dumps(result, indent=1, default=str), encoding='utf-8')
    return path
