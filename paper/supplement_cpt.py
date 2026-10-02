"""Supplementary material of the CPT submission, generated from the code, the benchmark definitions and the runs.

S1  system prompt, task message, tools (Table S1), model specification schema, dataset descriptions and sentences.
S2  estimation and run settings; PKPy2 fits of the reference models (Table S2); likelihood evidence of each reference
    covariate relationship (Table S3).
S3  results of every run (Table S4), statements referring to prior knowledge (recall), and the final report of
    every run.
Outputs: paper/submission_cpt/Supplementary_Material_S1.docx, _S2.docx, _S3.docx (captions inside the files).
Run benchmarks/evaluate.py, benchmarks/reference_table.py and paper/manuscript_numbers.py first.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from docx.enum.text import WD_LINE_SPACING
from docx.shared import Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(HERE))
from build_cpt import TITLE, add_inline, add_table, base_document, heading   # noqa: E402
from pkagent.config import Budget, Settings                           # noqa: E402
from pkagent.prompts import SYSTEM, task                              # noqa: E402
from pkagent.spec import SPEC_SCHEMA                                  # noqa: E402
from pkagent.tools import TOOLS                                       # noqa: E402

OUT = HERE / 'submission_cpt'
BENCH = ROOT / 'benchmarks'
DATASETS = json.loads((BENCH / 'datasets.json').read_text(encoding='utf-8'))
LABEL = dict(pheno='Phenobarbital', remifentanil='Remifentanil', oral_mm='Oral MM (simulated)')
LLM = dict(gpt='GPT-6.1 Sol', claude='Claude Opus 5.5')
COND = dict(none='no knowledge', knowledge='expert statement', misleading='misleading statement')


def single(p, size=None):
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    if size:
        for r in p.runs:
            r.font.size = Pt(size)
    return p


def text(d, t, size=None):
    return single(add_inline(d.add_paragraph(), t), size)


def code(d, t, size=8):
    for line in t.rstrip().splitlines():
        p = d.add_paragraph()
        run = p.add_run(line if line else ' ')
        run.font.name = 'Consolas'
        run.font.size = Pt(size)
        single(p)
        p.paragraph_format.space_after = Pt(0)


def new_doc(title):
    d = base_document()
    p = d.add_paragraph()
    add_inline(p, f'**{title}**')
    text(d, f'*{TITLE}*')
    return d


def s1():
    d = new_doc('Supplementary Material S1. System prompt, task message, and tools of PKAgent')
    heading(d, 'S1.1 System prompt', 3)
    text(d, 'Given verbatim to the language model at the start of every run.')
    code(d, SYSTEM)
    heading(d, 'S1.2 Task message', 3)
    text(d, 'The first user message of a run. The expert-knowledge section appears only in the expert-knowledge and '
            'misleading conditions; the budget lines show the settings of the benchmark.')
    code(d, task('<dataset description>', '<expert statement>', None, Budget()))
    heading(d, 'S1.3 Tools', 3)
    rows = []
    for t in TOOLS:
        f = t['function']
        params = ', '.join(f['parameters'].get('properties', {}))
        rows.append([f['name'], f['description'], params or '–'])
    add_table(d, f'**Table S1.** The {len(TOOLS)} tools available to the language model (function-calling '
                 'definitions sent with every request; descriptions verbatim)',
              ['Tool', 'Description', 'Parameters'], rows, 'The model specification accepted by fit_models is '
              'given in S1.4.', [1.3, 4.0, 1.2], size=8)
    heading(d, 'S1.4 Model specification schema', 3)
    text(d, 'JSON schema of one model specification (an item of fit_models.models). Specifications are validated '
            'against this schema and additional semantic checks before fitting.')
    code(d, json.dumps(SPEC_SCHEMA, indent=1), size=7)
    heading(d, 'S1.5 Benchmark datasets: descriptions and sentences given to the agent', 3)
    for name in LABEL:
        v = DATASETS[name]
        text(d, f'**{LABEL[name]}**')
        text(d, '*Description given to the agent:* ' + v['description'])
        text(d, '*Expert statement (expert-knowledge condition):* ' + v['knowledge'])
        if v.get('misleading_knowledge'):
            text(d, '*Misleading statement (misleading condition):* ' + v['misleading_knowledge'])
        d.add_paragraph()
    d.save(OUT / 'Supplementary_Material_S1.docx')


def s2():
    d = new_doc('Supplementary Material S2. Run and estimation settings, and PKPy2 fits of the reference models')
    st, b = Settings(), Budget()
    heading(d, 'S2.1 Settings of the benchmark runs', 3)
    items = [
        ('Language models', 'openai/gpt-6.1-sol and anthropic/claude-opus-5.5 through OpenRouter; reasoning effort '
                            f'"{st.reasoning_effort}"; provider-default temperature and routing (not seeded, not '
                            f'recorded); at most {st.max_output_tokens:,} output tokens per response; explicit prompt '
                            'caching for Claude Opus 5.5'),
        ('Plot images', 'the plots of one round of tool calls (at most six) form one message; the images of the two most '
                        'recent such messages are kept, older ones are replaced by a note'),
        ('Budgets per run', f'{b.max_fits} fits (bootstrap refits excluded), {b.max_turns} LLM responses, '
                            f'{b.max_hours:g} hours, {b.max_cost_usd:g} USD of LLM fees; the task message states the '
                            'first three, and every tool result reports the remaining fits, responses and hours and the '
                            'fees spent'),
        ('Computing', 'one desktop computer (AMD Ryzen 5 5600, 6 cores, 12 threads, 32 GB memory, Windows 11); two '
                      'runs at a time, each with three fitting processes of two Numba threads; other computations ran '
                      'on the same computer'),
        ('Estimation', "PKPy2 0.2.1, method='laplace': Laplace approximation of the marginal likelihood with "
                       'expected-information (FOCE-I-type) individual curvature'),
        ('Optimizer', 'one start (the agent\'s values); L-BFGS-B, at most 150 iterations (ftol 1e-10, gtol 1e-4); '
                      'covariate coefficients scaled by the standard deviation of their covariate; linear covariate '
                      'coefficients bounded so that 1 + β(z − center) stays positive for every subject (margin 0.98)'),
        ('Gradient', 'central finite differences of the Laplace objective (relative step 1e-4) with the conditional modes '
                     're-solved at every difference point; warm starts anchored at the best point'),
        ('Conditional modes', 'Newton search; a mode solved without a warm start receives a second search from the best '
                              'of 256 scrambled Sobol draws from N(0, Ω), and the mode with the higher joint density is '
                              'kept; at the end of a minimization, cold and warm modes are compared and the minimization '
                              'continues (at most twice) when the cold modes lower the OFV; modes are saved with the fit'),
        ('Time limit', f'{st.fit_wall_seconds / 60:g} minutes for the L-BFGS-B minimization; the polish (at most 20 '
                       'iterations), convergence check, standard errors and diagnostics are not time-limited'),
        ('Convergence', f'Newton decrement of the OFV at most {st.laplace_tolerance}'),
        ('Standard errors', 'covariance 2H⁻¹, with H from central differences of the gradient (modes re-solved) at '
                            'relative steps 0.002 and 0.001; reported when H is positive definite, the two Hessians '
                            'differ by at most 10% (relative norm), the standard errors agree within 20%, each Hessian '
                            'is at most 10% asymmetric, and no step crosses a bound'),
        ('Bootstrap (on request)', f'parallel Laplace refits, at most 200 samples, time limit '
                                   f'{st.bootstrap_seconds / 60:g} minutes'),
        ('Log-normal residual models', 'PKAgent reports the OFV on the data scale (PKPy2 OFV plus twice the sum of the '
                                       'log observations)'),
        ('Reference fits', 'same session code without an LLM; two starts (the published values and one perturbation); '
                           '60-minute limit'),
        ('Removal of reference relationships', 'one start from the reference estimates; 60-minute limit; no standard '
                                               'errors; the phenobarbital weight exponents are fixed, so their OFV '
                                               'changes are descriptive'),
        ('Stepwise baseline', 'reference structure without covariates; forward inclusion at P < 0.05 and backward '
                              'elimination at P < 0.01 over the reference form families (weight as a power function '
                              'with estimated exponent, an indicator for Apgar score below 5, linear age and LBM '
                              'effects) and plausible alternatives (Apgar on CL; age and LBM on every remifentanil '
                              'parameter)'),
        ('Development', '16 pilot runs (10 completed) on the phenobarbital and remifentanil datasets under earlier tool '
                        'versions were discarded; the system prompt was not changed after the first pilot; the code '
                        'was frozen at PKAgent commit f5a4263 and PKPy2 0.2.1 before the benchmark; covariate recall, '
                        'precision and form agreement and the misleading condition were added to the evaluation after '
                        'the first replicate'),
    ]
    for k, v in items:
        text(d, f'**{k}.** {v}', 10)
    heading(d, 'S2.2 PKPy2 fits of the reference models', 3)
    ref = pd.read_csv(BENCH / 'evaluation' / 'reference_table.csv')
    rows = []
    for _, r in ref.iterrows():
        def f(x, digits=4):
            return '–' if pd.isna(x) else f'{x:.{digits}g}'
        rows.append([LABEL.get(r['dataset'], r['dataset']), r['quantity'], f(r['reference']), f(r['pkpy2'], 5),
                     f(r['ratio'], 3), f(r['rse_percent'], 3)])
    add_table(d, '**Table S2.** Reference values (phenobarbital: FOCE-I estimates of the NONMEM example model from a '
                 'NONMEM 7.4.2 run distributed with the Pharmpy test data, pheno_real.mod and pheno_real.ext, '
                 'https://github.com/pharmpy/pharmpy; remifentanil: Minto et al.; oral MM: nominal simulation values) '
                 'and the PKPy2 fit of the same model to the benchmark data',
              ['Dataset', 'Quantity', 'Reference', 'PKPy2', 'Ratio', 'RSE (%)'], rows,
              'The NONMEM objective of the phenobarbital model (586.276) differs from the PKPy2 OFV by the constant '
              '155·ln(2π). Remifentanil slopes are absolute changes per year of age or per kg of LBM; the additive '
              'published model was fitted as a product of linear terms, and its variability model (exponential IIV on '
              'all six parameters, proportional error) is not from the publication; its fit had two optima 0.6 apart '
              'and no standard errors; its values are at age 40 years and LBM 55 kg, whereas the text gives medians over '
              'subjects. Oral MM: exponential IIV on Ka, V, VMAX, and KM and proportional error, as in the simulation; '
              'geometric means of the simulated individual values in the subset were '
              'Ka 0.98 1/h, V 67.2 L, VMAX 981 µg/h, KM 232 µg/L. IIV, interindividual variability; LBM, lean body '
              'mass; MM, Michaelis–Menten; OFV, objective function value; RSE, relative standard error (– when not '
              'available).', [1.2, 2.2, .9, .9, .6, .7], size=8.5)
    ev_path = BENCH / 'evaluation' / 'effect_evidence.json'
    if ev_path.exists():
        ev = json.loads(ev_path.read_text(encoding='utf-8'))
        rows = [[LABEL.get(ds, ds), r['effect'].replace('APGR_LT5', 'APGR < 5'), r['status'],
                 '–' if r['delta_ofv'] is None else f"{r['delta_ofv']:.2f}"]
                for ds, v in ev.items() for r in v['removals']]
        heading(d, 'S2.3 Likelihood evidence of the reference covariate relationships', 3)
        add_table(d, '**Table S3.** Increase in OFV when one covariate relationship is removed from the reference '
                     'model and the model is refitted',
                  ['Dataset', 'Relationship removed', 'Fit status', 'ΔOFV'], rows,
                  'Thresholds of the agent: 3.84 for inclusion (P < 0.05) and 6.63 for retention (P < 0.01). The '
                  'phenobarbital weight exponents are fixed at 1, so their OFV changes are descriptive. OFV, objective '
                  'function value.', [1.5, 2.0, 1.2, 1.0], size=8.5)
    d.save(OUT / 'Supplementary_Material_S2.docx')


def s3():
    d = new_doc('Supplementary Material S3. Final models, tests, tool use, recall statements, and reports of every '
                'run')
    runs = pd.read_csv(BENCH / 'evaluation' / 'runs.csv')
    runs = runs.sort_values(['dataset', 'condition', 'llm', 'rep'])
    prof = {(p['dataset'], p['condition'], p['llm'], p['rep']): p
            for p in json.loads((HERE / 'build' / 'run_profiles.json').read_text(encoding='utf-8'))}
    tests = {(t['dataset'], t['condition'], t['llm'], t['rep']): t
             for t in json.loads((BENCH / 'evaluation' / 'agent_tests.json').read_text(encoding='utf-8'))}

    def f(x, digits=1):
        return '–' if x is None or (isinstance(x, float) and pd.isna(x)) else f'{x:,.{digits}f}'.replace('-', '−')
    rows = []
    for _, r in runs.iterrows():
        p = prof.get((r['dataset'], r['condition'], r['llm'], r['rep']), {})
        rows.append([LABEL[r['dataset']], COND[r['condition']], LLM[r['llm']], r['rep'][3:], r['description'],
                     f(r['delta_ofv_vs_reference']), f(r['delta_aic_vs_reference']),
                     f(r['vpc_inside_fraction'] * 100 if pd.notna(r['vpc_inside_fraction']) else None, 0),
                     f(r['max_rse_structural']), f(r['max_eta_shrinkage'], 0),
                     'yes' if p.get('final_se') == 'computed' else 'no', p.get('bootstrap', '–'),
                     int(r['fits']), int(r['llm_calls']), f(r['hours'], 2), f(p.get('hours_left'), 2),
                     f(r['cost_usd'], 2)])
    add_table(d, '**Table S4.** Final model, diagnostics, and resources of every run',
              ['Dataset', 'Condition', 'LLM', 'Rep', 'Final model', 'ΔOFV', 'ΔAIC', 'VPC (%)', 'Max RSE (%)',
               'Max η-shr. (%)', 'SEs', 'Bootstrap', 'Fits', 'LLM resp.', 'Hours', 'Hours left', 'USD'], rows,
              'ΔOFV and ΔAIC, final model minus the PKPy2 fit of the reference model; VPC, percentage of observed 5th, '
              '50th and 95th percentiles inside their simulated 95% intervals (not prediction-corrected); Max RSE, '
              'largest relative standard error of a typical value; SEs, standard errors of the final model available; '
              'Bootstrap, converged/requested replicates within the 30-minute tool limit; Hours left, of the 6-hour '
              'budget at finalization. AIC, Akaike information criterion; cmt, compartment; IIV, interindividual '
              'variability; LLM, large language model; MM, Michaelis–Menten; OFV, objective function value; USD, US '
              'dollars; VPC, visual predictive check.',
              [.7, .6, .6, .3, 1.9, .5, .5, .4, .4, .4, .3, .5, .3, .3, .4, .4, .4], size=6.5)
    rows = []
    for _, r in runs.iterrows():
        key = (r['dataset'], r['condition'], r['llm'], r['rep'])
        p, t = prof.get(key, {}), tests.get(key, {})
        ev = {}
        for x in t.get('covariate_tests', []):
            ev.setdefault(x['relationship'], []).append(x['delta_ofv'])
        ev_text = '; '.join(f"{k} {', '.join(f(v, 1) for v in vs)}" for k, vs in sorted(ev.items())) or '–'
        rows.append([LABEL[r['dataset']], COND[r['condition']], LLM[r['llm']], r['rep'][3:],
                     'yes' if p.get('plot_data') else 'no', 'yes' if p.get('run_nca') else 'no',
                     'yes' if p.get('screen_covariates') else 'no', 'yes' if p.get('fitted_after_plots') else 'no',
                     ev_text])
    add_table(d, '**Table S5.** Tool use and the covariate tests of every run',
              ['Dataset', 'Condition', 'LLM', 'Rep', 'Plotted data', 'NCA', 'Screened covariates',
               'Fitted after viewing plots', 'Covariate tests (ΔOFV)'], rows,
              'Covariate tests: OFV differences between two converged models of the run that differ only by the '
              'relationship (positive values favor the relationship), extracted from the model registry. LLM, large '
              'language model; NCA, noncompartmental analysis; OFV, objective function value.',
              [.8, .7, .7, .3, .5, .4, .6, .6,3.0], size=7)
    scm_path = BENCH / 'evaluation' / 'scm_baseline.json'
    if scm_path.exists():
        heading(d, 'S3.1 Deterministic stepwise covariate baseline', 3)
        for ds, v in json.loads(scm_path.read_text(encoding='utf-8')).items():
            steps = [h for h in v['history'] if h['step'] in ('forward add', 'forward stop', 'backward remove')]
            desc = '; '.join(f"{h['step']} {h.get('added') or h.get('removed') or h.get('best')} (ΔOFV "
                             f"{h['delta_ofv']})" for h in steps)
            text(d, f"**{LABEL[ds]}.** Base OFV {v['base_ofv']:.2f}; final OFV {v['final_ofv']:.2f}; retained: "
                    f"{', '.join(v['included']) or 'none'}; {v['fits']} fits. Steps: {desc}.", 9)
    bb_path = BENCH / 'evaluation' / 'backward_baseline.json'
    if bb_path.exists():
        for ds, v in json.loads(bb_path.read_text(encoding='utf-8')).items():
            steps = '; '.join(f"step {h['step']}: " + ', '.join(f"{k} {x:.1f}" for k, x in h['tests'].items() if x is not None)
                              for h in v['history'])
            text(d, f"**{LABEL[ds]} (backward elimination from the reference model).** Removed: "
                    f"{', '.join(v.get('removed', [])) or 'none'}; retained: {', '.join(v.get('retained', []))}. OFV "
                    f"increase on removal at each step: {steps}.", 9)
    heading(d, 'S3.2 Statements referring to prior knowledge of the data or their analysis', 3)
    text(d, 'Matches of the regular expression \\bclassic\\b|well[- ]known|textbook|nonmem example|published|literature|'
            'grasela|donn\\b|minto (case-insensitive) in the assistant messages, tool-call arguments and reasoning '
            'summaries of the runs on the two published datasets, with 120 characters of context.', 10)
    recall = json.loads((HERE / 'build' / 'recall.json').read_text(encoding='utf-8'))
    for key, hits in recall.items():
        if not hits:
            continue
        ds, cond, llm, rep = key.split('/')
        text(d, f'**{LABEL[ds]}, {COND[cond]}, {LLM[llm]}, replicate {rep[3:]}**', 10)
        for h in hits:
            text(d, f"Turn {h['turn']} ({h['term']}): …{h['context']}…", 9)
    heading(d, 'S3.3 Final reports', 3)
    text(d, 'The structured report submitted by the agent with its final model (verbatim).', 10)
    for _, r in runs.iterrows():
        res = json.loads((BENCH / 'runs' / r['dataset'] / r['condition'] / r['llm'] / r['rep'] / 'results.json')
                         .read_text(encoding='utf-8'))
        rep = (res.get('final_model') or {}).get('report') or {}
        text(d, f"**{LABEL[r['dataset']]}, {COND[r['condition']]}, {LLM[r['llm']]}, replicate {r['rep'][3:]}**", 10)
        for k in ('summary', 'development', 'expert_knowledge', 'evaluation', 'limitations'):
            if rep.get(k):
                text(d, f"*{k.replace('_', ' ').capitalize()}.* {rep[k]}", 9)
    d.save(OUT / 'Supplementary_Material_S3.docx')


def main():
    OUT.mkdir(exist_ok=True)
    s1()
    s2()
    s3()
    print('wrote', *sorted(p.name for p in OUT.glob('Supplementary_Material_S*.docx')))


if __name__ == '__main__':
    main()
