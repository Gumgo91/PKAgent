"""Build the Clinical Pharmacology & Therapeutics submission of the PKAgent manuscript.

Input: paper/manuscript_cpt.md (body), paper/build/numbers.json and table2.json (python paper/manuscript_numbers.py),
paper/misleading_text.json (prose about the misleading-sentence runs, once they exist), benchmarks/datasets.json
(expert statements of Table 1), paper/figures/Figure_<n>.pdf.
Output: paper/submission_cpt/ with the manuscript (DOCX: title page, abstract, text, study highlights, statements,
references, tables, figure legends), the figures as separate files, and the cover letter.

CPT conventions applied: 12-point Times New Roman, double spacing, 1-inch margins, US Letter, page and line numbers;
citations as superscript numbers after punctuation, numbered in order of first citation; tables after the references,
one per page; figure legends after the tables.
"""
import json
import re
import shutil
import sys
from pathlib import Path

import docx
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_BREAK, WD_COLOR_INDEX, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from references_cpt import REFERENCES                      # noqa: E402

BUILD = HERE / 'build'
OUT = HERE / 'submission_cpt'
DATASETS = json.loads((HERE.parent / 'benchmarks' / 'datasets.json').read_text(encoding='utf-8'))

TITLE = ('A Large Language Model Agent for Population Pharmacokinetic Model Development on an Open-Source Engine: '
         'Agreement With Reference Models Without and With Expert Knowledge')
AUTHORS = [('Hyunseung Kong', 1)]
AFFILIATIONS = {1: 'Interdisciplinary Program in Bioinformatics, Seoul National University, Seoul, Republic of Korea'}
CORRESPONDING = ('Hyunseung Kong, Interdisciplinary Program in Bioinformatics, Seoul National University, 1 Gwanak-ro, '
                 'Gwanak-gu, Seoul 08826, Republic of Korea. Email: hskong@snu.ac.kr')
KEYWORDS = ['population pharmacokinetics', 'large language models', 'artificial intelligence agents',
            'model development', 'covariate modeling', 'nonlinear mixed-effects models']
FUNDING = 'No funding was received for this work.'
COI = 'The authors declared no competing interests for this work.'
CONTRIBUTIONS = ('H.K. wrote the manuscript; H.K. designed the research; H.K. performed the research; H.K. analyzed '
                 'the data; H.K. contributed new reagents/analytical tools.')
AI_DISCLOSURE = ('[To be completed by the authors: disclosure of any use of artificial intelligence tools in preparing '
                 'this manuscript, as required by the journal (tool name and version, date of use, role, and how the '
                 'authors reviewed the output). The language models evaluated in this study are described in Methods.]')

CITE = re.compile(r'\s*\[([A-Za-z0-9]+(?:;\s*[A-Za-z0-9]+)*)\]([.,;:])?')


# ------------------------------------------------------------------ text processing
def conditional_blocks(text, flags):
    """Keep '<!-- if KEY -->...<!-- endif -->' when KEY is set and '<!-- ifnot KEY -->...' when it is not."""
    def keep(m):
        on = bool(flags.get(m.group(2)))
        return m.group(3) if on != bool(m.group(1)) else ''
    return re.sub(r'<!-- if(not)? (\S+) -->(.*?)<!-- endif -->', keep, text, flags=re.S)


def fill(text, numbers, draft=False):
    """Replace {{key}} by its value; values may contain placeholders themselves (prose such as misleading_results),
    so the replacement is repeated until none is left."""
    for _ in range(3):
        missing = sorted(set(re.findall(r'\{\{([^}]+)\}\}', text)) - set(numbers))
        if missing and not draft:
            raise KeyError(f'numbers missing for: {missing} (use --draft to mark them as pending)')
        if missing:
            print('pending numbers:', missing)
        text = re.sub(r'\{\{([^}]+)\}\}', lambda m: str(numbers.get(m.group(1), '⟦HL⟧pending⟦/HL⟧')), text)
        if '{{' not in text:
            break
    return text


class Citations:
    """Numbers references in order of first citation; '[A; B].' becomes '.' followed by superscript '1,2'."""

    def __init__(self):
        self.order = []

    def numbers(self, keys):
        for k in keys:
            if k not in REFERENCES:
                raise KeyError(f'unknown reference key {k}')
            if k not in self.order:
                self.order.append(k)
        nums = sorted({self.order.index(k) + 1 for k in keys})
        parts, run = [], [nums[0]]
        for n in nums[1:]:
            if n == run[-1] + 1:
                run.append(n)
            else:
                parts.append(run)
                run = [n]
        parts.append(run)
        return ','.join(f'{r[0]}–{r[-1]}' if len(r) > 2 else ','.join(map(str, r)) for r in parts)

    def markup(self, text):
        """Replace citations by '^{...}' markers placed after the following punctuation."""
        def repl(m):
            keys = [k.strip() for k in m.group(1).split(';')]
            return (m.group(2) or '') + '^{' + self.numbers(keys) + '}'
        return CITE.sub(repl, text)


# ------------------------------------------------------------------ DOCX helpers
def base_document():
    d = docx.Document()
    st = d.styles['Normal']
    st.font.name = 'Times New Roman'
    st.font.size = Pt(12)
    st.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    pf = st.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    pf.space_after = Pt(0)
    pf.space_before = Pt(0)
    sec = d.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    for side in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'):
        setattr(sec, side, Inches(1))
    line_numbers(sec)
    page_numbers(sec)
    return d


def line_numbers(section):
    ln = OxmlElement('w:lnNumType')
    ln.set(qn('w:countBy'), '1')
    ln.set(qn('w:restart'), 'continuous')
    ln.set(qn('w:distance'), '360')
    section._sectPr.append(ln)


def page_numbers(section):
    p = section.footer.paragraphs[0]
    p.alignment = 1
    run = p.add_run()
    for tag, text in (('begin', None), (None, 'PAGE'), ('end', None)):
        if tag:
            el = OxmlElement('w:fldChar')
            el.set(qn('w:fldCharType'), tag)
        else:
            el = OxmlElement('w:instrText')
            el.set(qn('xml:space'), 'preserve')
            el.text = text
        run._r.append(el)


INLINE = re.compile(r'(\*\*[^*]+\*\*|\*[^*]+\*|\^\{[^}]+\}|<[^>]+>|⟦HL⟧.*?⟦/HL⟧)')


def add_inline(p, text, size=None, bold=False):
    """Add text with **bold**, *italic*, ^{superscript} and ⟦HL⟧highlight⟦/HL⟧ markup to a paragraph."""
    for part in INLINE.split(text):
        if not part:
            continue
        kw = {}
        if part.startswith('**') and part.endswith('**'):
            part, kw['bold'] = part[2:-2], True
        elif part.startswith('*') and part.endswith('*') and len(part) > 2:
            part, kw['italic'] = part[1:-1], True
        elif part.startswith('^{'):
            part, kw['sup'] = part[2:-1], True
        elif part.startswith('⟦HL⟧'):
            part, kw['hl'] = part[4:-5], True
        run = p.add_run(part)
        run.bold = kw.get('bold') or bold or None
        run.italic = kw.get('italic')
        if kw.get('sup'):
            run.font.superscript = True
        if kw.get('hl'):
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        if size:
            run.font.size = Pt(size)
    return p


def heading(d, text, level):
    p = d.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    if level == 3:
        run.italic = False
    p.paragraph_format.keep_with_next = True
    return p


def page_break(d):
    d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ------------------------------------------------------------------ tables
def table1_rows():
    design = {
        'pheno': '59 preterm neonates; IV bolus loading and maintenance doses; 155 concentrations (1–6 per infant); '
                 'covariates: birth weight, 5-minute Apgar score',
        'remifentanil': '65 adults aged 20–85 years; one 4- to 20-minute IV infusion; 1,992 arterial concentrations; '
                        'covariates: age, sex, height, weight, body surface area, LBM',
        'oral_mm': 'Simulated (subset of Oral_1CPTMM): 40 subjects, 10 per dose level of 10, 20, 40, and 80 mg; single '
                   'oral dose, then seven daily doses; 25 samples each',
    }
    reference = {
        'pheno': 'NONMEM example model: one compartment; CL and V proportional to birth weight (exponents fixed at 1); '
                 'V 15.9% larger when Apgar < 5; exponential IIV on CL and V; proportional error',
        'remifentanil': 'Minto et al.: three compartments; V1, V2, and CL linear in age (centered at 40 years) and LBM '
                        '(centered at 55 kg); Q2 and Q3 linear in age; V3 constant. Variability model not taken from '
                        'the publication (PKPy2 fit: exponential IIV on all parameters, proportional error)',
        'oral_mm': 'Simulation model: one compartment; first-order absorption; MM elimination; 30% IIV on Ka, V, VMAX, '
                   'and KM; 20% proportional error',
    }
    label = dict(pheno='Phenobarbital', remifentanil='Remifentanil', oral_mm='Oral MM (simulated)')
    return [[label[k], design[k], reference[k], f'"{DATASETS[k]["knowledge"]}"'] for k in label]


def add_table(d, title, header, rows, footnote, widths, size=9):
    p = d.add_paragraph()
    add_inline(p, title)
    t = d.add_table(rows=1, cols=len(header))
    t.style = 'Table Grid'
    for cell, h in zip(t.rows[0].cells, header):
        cell.paragraphs[0].text = ''
        add_inline(cell.paragraphs[0], h, size=size, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for cell, v in zip(cells, row):
            add_inline(cell.paragraphs[0], str(v), size=size)
    for row in t.rows:
        for cell, w in zip(row.cells, widths):
            cell.width = Inches(w)
            for par in cell.paragraphs:
                par.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    p = d.add_paragraph()
    add_inline(p, footnote, size=10)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE


def landscape_section(d):
    sec = d.add_section()
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = Inches(11), Inches(8.5)
    return sec


def portrait_section(d):
    sec = d.add_section()
    sec.orientation = WD_ORIENT.PORTRAIT
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    return sec


# ------------------------------------------------------------------ build
def sections(text):
    """Split the body into (level, title, paragraphs) blocks by '## ' and '### ' headings."""
    blocks, cur = [], None
    for line in text.splitlines():
        if line.startswith('## ') or line.startswith('### '):
            level = 2 if line.startswith('## ') else 3
            cur = [level, line.split(' ', 1)[1].strip(), []]
            blocks.append(cur)
        elif cur is not None:
            cur[2].append(line)
    out = []
    for level, title, lines in blocks:
        paras, buf = [], []
        for line in lines + ['']:
            if line.strip():
                buf.append(line.strip())
            elif buf:
                paras.append(' '.join(buf))
                buf = []
        out.append((level, title, paras))
    return out


def words(text):
    text = re.sub(r'\^\{[^}]+\}', '', text)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.,/–-]*", text))


def main():
    numbers = json.loads((BUILD / 'numbers.json').read_text(encoding='utf-8'))
    table2 = json.loads((BUILD / 'table2.json').read_text(encoding='utf-8'))
    misleading = any(r['condition'].startswith('Misleading') for r in table2)
    extra = HERE / 'misleading_text.json'
    if misleading and extra.exists():
        numbers.update(json.loads(extra.read_text(encoding='utf-8')))
    numbers['ai_disclosure'] = f'⟦HL⟧{AI_DISCLOSURE}⟦/HL⟧'
    numbers['author_contributions'] = CONTRIBUTIONS

    text = (HERE / 'manuscript_cpt.md').read_text(encoding='utf-8')
    text = conditional_blocks(text, dict(numbers, misleading=misleading))
    text = re.sub(r'<!--.*?-->', '', text, flags=re.S)
    text = fill(text, numbers, draft='--draft' in sys.argv)
    cites = Citations()
    body = sections(text)
    body = [(lvl, title, [cites.markup(p) for p in paras]) for lvl, title, paras in body]

    main_text = [p for lvl, title, paras in _between(body, 'INTRODUCTION', 'STUDY HIGHLIGHTS') for p in paras]
    n_words = sum(words(p) for p in main_text)
    abstract = next(paras for lvl, title, paras in body if title == 'ABSTRACT')
    n_abstract = sum(words(p) for p in abstract)
    highlights = next(paras for lvl, title, paras in body if title == 'STUDY HIGHLIGHTS')
    n_highlights = sum(words(p) for p in highlights if not p.startswith('**'))
    n_refs = len(cites.order)
    n_fig, n_tab = 4, 2
    print(f'main text {n_words} words; abstract {n_abstract}; highlights {n_highlights}; references {n_refs}; '
          f'figures {n_fig}; tables {n_tab}')
    problems = []
    if n_words > 4000:
        problems.append(f'main text {n_words} > 4000 words')
    if n_abstract > 250:
        problems.append(f'abstract {n_abstract} > 250 words')
    if n_highlights >= 250:
        problems.append(f'study highlights {n_highlights} >= 250 words')
    if n_refs > 50:
        problems.append(f'{n_refs} references > 50')
    if n_fig + n_tab > 7:
        problems.append('more than 7 figures and tables')
    import check_claims
    failed = [name for name, ok, _ in check_claims.checks() if not ok]
    problems += [f'claim no longer holds: {name}' for name in failed]
    for p in problems:
        print('WARNING:', p)
    if problems and '--draft' not in sys.argv:
        raise SystemExit('final build refused (use --draft to build anyway): ' + '; '.join(problems))

    d = base_document()
    # title page
    p = d.add_paragraph()
    add_inline(p, f'**{TITLE}**')
    p = d.add_paragraph()
    p.add_run(', '.join(f'{n}' for n, a in AUTHORS))
    for k, v in AFFILIATIONS.items():
        d.add_paragraph(v)
    p = d.add_paragraph()
    add_inline(p, '**Corresponding author:** ' + CORRESPONDING)
    p = d.add_paragraph()
    add_inline(p, '**Funding:** ' + FUNDING)
    p = d.add_paragraph()
    add_inline(p, '**Conflict of interest:** ' + COI)
    p = d.add_paragraph()
    add_inline(p, '**Keywords:** ' + '; '.join(KEYWORDS))
    p = d.add_paragraph()
    add_inline(p, f'**Word count:** main text {n_words:,}; abstract {n_abstract}. **References:** {n_refs}. '
                  f'**Figures:** {n_fig}. **Tables:** {n_tab}.')
    page_break(d)

    for lvl, title, paras in body:
        if title in ('FIGURE LEGENDS', 'SUPPLEMENTARY MATERIAL'):
            continue
        if title == 'ABSTRACT':
            heading(d, 'ABSTRACT', 2)
            for t in paras:
                add_inline(d.add_paragraph(), t)
            page_break(d)
            continue
        if title == 'STUDY HIGHLIGHTS':
            page_break(d)
        heading(d, title, lvl)
        for t in paras:
            add_inline(d.add_paragraph(), t)

    # references
    page_break(d)
    heading(d, 'REFERENCES', 2)
    for i, k in enumerate(cites.order, 1):
        p = d.add_paragraph()
        p.paragraph_format.left_indent = Inches(.35)
        p.paragraph_format.first_line_indent = Inches(-.35)
        add_inline(p, f'{i}.\t' + REFERENCES[k].replace('<', '‹').replace('>', '›').replace('‹', '<').replace('›', '>'))

    # tables (landscape, one per page)
    landscape_section(d)
    add_table(d, '**Table 1.** Benchmark datasets, reference models, and the expert statement of the expert-knowledge '
                 'condition',
              ['Dataset', 'Design', 'Reference model', 'Expert statement'], table1_rows(),
              'CL, clearance; IIV, interindividual variability; IV, intravenous; Ka, absorption rate constant; KM, '
              'Michaelis constant; LBM, lean body mass; MM, Michaelis–Menten; Q2 and Q3, intercompartmental '
              'clearances; V, V1, V2, V3, volumes of distribution; VMAX, maximum elimination rate.',
              [1.3, 2.6, 2.6, 2.5])
    page_break(d)
    header = ['Dataset, condition', 'Runs (GPT/Claude)', 'Reference structure',
              'Reference relationships present (reference form)', 'Other relationships per run',
              'Typical values agree', 'Reproduced', 'ΔOFV vs. reference fit', 'ΔAIC vs. reference fit',
              'Models fitted', 'Hours', 'Fees (USD)']
    rows = [[f"{r['dataset']}, {r['condition'].lower()}", r['runs'], r['structure'],
             'NA' if r['relationships'] == 'NA' else f"{r['relationships']} ({r['forms']})", r['extra'],
             r['typical'], r['reproduced'], r['delta_ofv'], r['delta_aic'], r['fits'], r['hours'], r['cost']]
            for r in table2]
    add_table(d, '**Table 2.** Final models of the agent runs compared with the reference models',
              header, rows,
              'Values are counts over runs, or medians (ranges); with two runs, both values are given. Reference '
              'relationships: parameter–covariate pairs of the reference model present in the final model, summed over '
              'runs, with the number in the reference form family in parentheses (NA, the reference model has no '
              'covariates). Other relationships: pairs in the final model that are not in the reference model. '
              'Typical values agree: the median ratio of the subject-level typical values (final/reference model) lay '
              'within 0.80 to 1.25 for every reference parameter. Reproduced: reference structure, exactly the '
              'reference relationships, and agreement of typical values. ΔOFV and ΔAIC: final model minus the PKPy2 '
              'fit of the reference model, whose stochastic model is given in Table S2. Fees: language model fees. AIC, '
              'Akaike information criterion; Claude, Claude Opus 5.5; GPT, GPT-6.1 Sol; MM, Michaelis–Menten; NA, not '
              'applicable; OFV, objective function value; USD, US dollars.',
              [1.4, .6, .6, .95, .7, .6, .6, 1.05, 1.0, .75, .75, .8], size=8)

    # figure legends and supplementary material
    portrait_section(d)
    for lvl, title, paras in body:
        if title in ('FIGURE LEGENDS', 'SUPPLEMENTARY MATERIAL'):
            heading(d, title, 2)
            for t in paras:
                add_inline(d.add_paragraph(), t)
            if title == 'FIGURE LEGENDS':
                d.add_paragraph()

    OUT.mkdir(exist_ok=True)
    d.save(OUT / 'PKAgent_CPT_manuscript.docx')
    cover_letter()
    for n in range(1, n_fig + 1):
        for ext in ('pdf', 'tiff'):
            src = HERE / 'figures' / f'Figure_{n}.{ext}'
            if src.exists():
                shutil.copy(src, OUT / src.name)
    (BUILD / 'manuscript_cpt_filled.md').write_text(text, encoding='utf-8')
    (BUILD / 'reference_order.json').write_text(json.dumps(cites.order, indent=1), encoding='utf-8')
    unused = sorted(set(REFERENCES) - set(cites.order))
    if unused:
        print('references not cited:', unused)
    print('wrote', OUT / 'PKAgent_CPT_manuscript.docx')


def cover_letter():
    import datetime as dt
    today = dt.date.today()
    values = dict(date=f'{today:%B} {today.day}, {today.year}', title=TITLE, ai_disclosure=f'⟦HL⟧{AI_DISCLOSURE}⟦/HL⟧',
                  corresponding=CORRESPONDING.replace('. Email:', '\nEmail:'))
    text = (HERE / 'cover_letter_cpt.md').read_text(encoding='utf-8')
    text = re.sub(r'\{\{(\w+)\}\}', lambda m: values[m.group(1)], text)
    d = docx.Document()
    st = d.styles['Normal']
    st.font.name, st.font.size = 'Times New Roman', Pt(12)
    for block in text.split('\n\n'):
        p = d.add_paragraph()
        for k, line in enumerate(block.strip().splitlines()):
            if k:
                p.add_run().add_break()
            add_inline(p, line)
    d.save(OUT / 'PKAgent_CPT_cover_letter.docx')


def _between(body, start, stop):
    out, on = [], False
    for b in body:
        if b[1] == start:
            on = True
        if b[1] == stop:
            break
        if on:
            out.append(b)
    return out


if __name__ == '__main__':
    main()
