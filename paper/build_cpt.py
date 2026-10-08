"""Build the Clinical Pharmacology & Therapeutics submission of the PKAgent manuscript.

Input: paper/manuscript_cpt.md (body), paper/build/numbers.json and table2.json (python paper/manuscript_numbers.py),
paper/misleading_text.json (prose about the misleading-sentence runs, once they exist), paper/figures/Figure_<n>.pdf and
.tiff, paper/figures/alt_text.txt.
Output: paper/submission_cpt/ with the manuscript (DOCX: title page, abstract, text, study highlights, statements,
references, tables, figure legends), the figures and their alternative text as separate files, and the cover letter.

CPT conventions applied: 12-point Times New Roman, double spacing, 1-inch margins, US Letter, page and line numbers;
citations as superscript numbers after punctuation, numbered in order of first citation; tables after the references,
one per page, with at most 130 characters per row (the final build is refused otherwise); figure legends after the
tables. Headings use the Word styles Heading 1 (main headings, capitals) and Heading 2 (subheadings, sentence case),
restyled to the body font, so that the navigation pane and PDF bookmarks work.
"""
import json
import re
import shutil
import sys
from pathlib import Path

import docx
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_COLOR_INDEX, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from references_cpt import REFERENCES                      # noqa: E402

BUILD = HERE / 'build'
OUT = HERE / 'submission_cpt'
MAX_TABLE_ROW = 130                              # CPT: 'restrict the number of characters per row to 130'

TITLE = 'PKAgent: Expert Knowledge Versus Data in Population Pharmacokinetic Modeling by a Large Language Model Agent'
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
AI_DISCLOSURE = ('[To be completed by the author: disclosure of any use of artificial intelligence tools in preparing '
                 'this manuscript, as required by the journal (tool name and version, date of use, role, and how the '
                 'author reviewed the output). The language models evaluated in this study are described in Methods.]')

NBSP = '\u00a0'                                 # no-break space
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
    style_headings(d)
    sec = d.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    for side in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'):
        setattr(sec, side, Inches(1))
    line_numbers(sec)
    page_numbers(sec)
    return d


def style_headings(d):
    """Restyle the built-in Heading 1 and Heading 2 (which keep their outline levels) to Times New Roman 12 pt bold
    black, double spaced, with no space before or after; the template's theme fonts and colors would override the
    font name and color, so their attributes are removed."""
    for name in ('Heading 1', 'Heading 2'):
        st = d.styles[name]
        st.font.name = 'Times New Roman'
        st.font.size = Pt(12)
        st.font.bold = True
        st.font.italic = False
        st.font.color.rgb = RGBColor(0, 0, 0)
        rpr = st.element.rPr
        fonts = rpr.rFonts
        for attr in ('w:asciiTheme', 'w:hAnsiTheme', 'w:eastAsiaTheme', 'w:cstheme'):
            fonts.attrib.pop(qn(attr), None)
        for attr in ('w:ascii', 'w:hAnsi', 'w:eastAsia', 'w:cs'):
            fonts.set(qn(attr), 'Times New Roman')
        color = rpr.find(qn('w:color'))
        for attr in ('w:themeColor', 'w:themeShade', 'w:themeTint'):
            color.attrib.pop(qn(attr), None)
        szcs = rpr.find(qn('w:szCs'))
        if szcs is not None:
            szcs.set(qn('w:val'), '24')
        pf = st.paragraph_format
        pf.space_before = pf.space_after = Pt(0)
        pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
        pf.keep_with_next = True


def word_2013_layout(d):
    """Compatibility mode 15 (Word 2013 and later) instead of the template's mode 14."""
    for cs in d.settings.element.find(qn('w:compat')).findall(qn('w:compatSetting')):
        if cs.get(qn('w:name')) == 'compatibilityMode':
            cs.set(qn('w:val'), '15')


def line_numbers(section):
    """Continuous line numbers; w:lnNumType goes before w:pgNumType, w:cols and w:docGrid (schema order)."""
    ln = OxmlElement('w:lnNumType')
    ln.set(qn('w:countBy'), '1')
    ln.set(qn('w:restart'), 'continuous')
    ln.set(qn('w:distance'), '360')
    section._sectPr.insert_element_before(
        ln, 'w:pgNumType', 'w:cols', 'w:formProt', 'w:vAlign', 'w:noEndnote', 'w:titlePg', 'w:textDirection',
        'w:bidi', 'w:rtlGutter', 'w:docGrid', 'w:printerSettings', 'w:sectPrChange')


def page_numbers(section):
    p = section.footer.paragraphs[0]
    p.alignment = 1
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    p.paragraph_format.space_after = Pt(0)
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


def add_inline(p, text, size=None, bold=False, literal=False):
    """Add text with **bold**, *italic*, ^{superscript} and ⟦HL⟧highlight⟦/HL⟧ markup to a paragraph.

    literal=True adds the text as one plain run (verbatim text such as equations with '*' or '**')."""
    for part in [text] if literal else INLINE.split(text):
        if not part:
            continue
        kw = {}
        if literal:
            pass
        elif part.startswith('**') and part.endswith('**'):
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


def set_properties(d, title):
    """Document properties of a submission file (python-docx otherwise keeps those of its template)."""
    import datetime as dt
    cp = d.core_properties
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    cp.author = cp.last_modified_by = AUTHORS[0][0]
    cp.title = title
    cp.comments = cp.subject = cp.keywords = ''
    cp.created = cp.modified = now
    cp.revision = 1
    return d


def heading(d, text, level, new_page=False):
    """Level 2 ('## ', main headings in capitals) uses Heading 1, level 3 ('### ', subheadings in sentence case)
    Heading 2. new_page starts the heading on a new page (instead of an empty paragraph with a page break)."""
    p = d.add_paragraph(text, style='Heading 1' if level == 2 else 'Heading 2')
    p.paragraph_format.keep_with_next = True
    if new_page:
        p.paragraph_format.page_break_before = True
    return p


# ------------------------------------------------------------------ tables
def table1_rows(cites):
    """Key facts only, so that a row stays within the CPT limit of 130 characters; the details are in Methods and
    Table S2, and the expert statements, verbatim, in Supplementary Material S1.5. Called after the body has been
    numbered, so the cited sources keep the numbers of their first citation in the text."""
    def cite(*keys):
        return '^{' + cites.numbers(list(keys)) + '}'
    design = {
        'pheno': f'59 preterm neonates{cite("Grasela1985")}; IV doses; 155 concentrations',
        'remifentanil': '65 adults aged 20–85; IV infusion; 1,992 concentrations',
        'oral_mm': f'Oral_1CPTMM{cite("nlmixr2data", "Schoemaker2019")} subset: 40 subjects; 10–80' + NBSP + 'mg orally',
    }
    reference = {
        'pheno': f'One compartment; birth weight on CL and V; Apgar score on V{cite("Boeckmann1994")}',
        'remifentanil': f'Three compartments; age and LBM on V1, V2, CL; age on Q2, Q3{cite("Minto1997")}',
        'oral_mm': 'One compartment; first-order absorption; MM elimination',
    }
    label = dict(pheno='Phenobarbital', remifentanil='Remifentanil', oral_mm='Oral MM (simulated)')
    return [[label[k], design[k], reference[k]] for k in label]


def add_table(d, title, header, rows, footnote, widths, size=9, literal=False, note_size=10,
              note_spacing=WD_LINE_SPACING.SINGLE, fixed=False, new_page=False):
    """literal=True adds the body cells verbatim (no markup); title, header and footnote keep the markup.
    fixed=True fixes the column widths (autofit off, tblGrid equal to the cell widths) within the 9-inch text width of
    a landscape page. new_page starts the title on a new page. Returns the table. (The title is not set to keep with
    next: Word in compatibility mode 15 hangs exporting a PDF when it is, before a fixed-width table.)"""
    p = d.add_paragraph()
    add_inline(p, title)
    if new_page:
        p.paragraph_format.page_break_before = True
    t = d.add_table(rows=1, cols=len(header))
    t.style = 'Table Grid'
    for cell, h in zip(t.rows[0].cells, header):
        cell.paragraphs[0].text = ''
        add_inline(cell.paragraphs[0], h, size=size, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for cell, v in zip(cells, row):
            add_inline(cell.paragraphs[0], str(v), size=size, literal=literal)
    for row in t.rows:
        for cell, w in zip(row.cells, widths):
            cell.width = Inches(w)
            for par in cell.paragraphs:
                par.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    if fixed:
        if sum(widths) > 9.0 + 1e-9:
            raise ValueError(f'column widths sum to {sum(widths):.2f} in > 9.0 in')
        t.autofit = False
        for gc, w in zip(t._tbl.tblGrid.findall(qn('w:gridCol')), widths):
            gc.set(qn('w:w'), str(round(w * 1440)))
        tblw = t._tbl.tblPr.find(qn('w:tblW'))
        tblw.set(qn('w:type'), 'dxa')
        tblw.set(qn('w:w'), str(sum(round(w * 1440) for w in widths)))
    p = d.add_paragraph()
    add_inline(p, footnote, size=note_size)
    p.paragraph_format.line_spacing_rule = note_spacing
    return t


def chars_per_row(t):
    """Characters of cell text per table row (line breaks inside a cell count as one space)."""
    return [sum(len(c.text.replace('\n', ' ')) for c in r.cells) for r in t.rows]


def _new_section(d):
    """python-docx ends the current section with a new empty paragraph; move its sectPr into the preceding paragraph,
    so that a section break adds no (line-numbered) blank line."""
    sec = d.add_section()
    sect_p = d.element.body[-2]                      # the new paragraph; body[-1] is the document's final sectPr
    prev = sect_p.getprevious()
    if prev is not None and prev.tag == qn('w:p') and not prev.xpath('./w:pPr/w:sectPr'):
        prev.set_sectPr(sect_p.pPr.sectPr)
        sect_p.getparent().remove(sect_p)
    return sec


def landscape_section(d):
    sec = _new_section(d)
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = Inches(11), Inches(8.5)
    return sec


def portrait_section(d):
    sec = _new_section(d)
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
    numbers['funding'] = FUNDING                         # end-matter FUNDING and CONFLICT OF INTEREST sections repeat
    numbers['coi'] = COI                                 # the title-page sentences

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
    n_words_h = n_words + sum(words(title) for lvl, title, paras in _between(body, 'INTRODUCTION', 'STUDY HIGHLIGHTS'))
    print(f'main text including headings {n_words_h} words')
    if n_words_h > 4000:                                 # what an editor's word count of the section shows
        problems.append(f'main text including headings {n_words_h} > 4000 words')
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
    if AI_DISCLOSURE.lstrip().startswith('[To be completed'):          # reported, but the build is not refused
        print('WARNING: the AI-use disclosure is still a placeholder (AI_DISCLOSURE in paper/build_cpt.py; printed '
              'highlighted in ACKNOWLEDGMENTS and in the cover letter); the author must write it before submission.')
    links = re.findall(r'(https://github\.com/\S+) \(tag ([^)]+)\)', text)
    print('REMINDER: before submission, these Data Availability links must exist (not checked here): '
          + '; '.join(f'{url} tag {tag}' for url, tag in links)
          + ('; the release of the first tag must carry the run logs, reference fits, and evaluation outputs.'
             if 'attached to that release' in text else '.'))

    d = base_document()
    word_2013_layout(d)
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

    for lvl, title, paras in body:
        if title in ('FIGURE LEGENDS', 'SUPPLEMENTARY MATERIAL'):
            continue
        heading(d, title, lvl, new_page=title in ('ABSTRACT', 'INTRODUCTION'))
        for t in paras:
            p = add_inline(d.add_paragraph(), t)
            if title == 'STUDY HIGHLIGHTS' and t.startswith('**'):
                p.paragraph_format.keep_with_next = True             # keep each question with its answer

    # references
    heading(d, 'REFERENCES', 2, new_page=True)
    for i, k in enumerate(cites.order, 1):
        p = d.add_paragraph()
        p.paragraph_format.left_indent = Inches(.35)
        p.paragraph_format.first_line_indent = Inches(-.35)
        add_inline(p, f'{i}.\t' + REFERENCES[k].replace('<', '‹').replace('>', '›').replace('‹', '<').replace('›', '>'))

    # tables (landscape, one per page); the sources cited in Table 1 keep their numbers from the text
    landscape_section(d)
    statements = json.loads((HERE.parent / 'benchmarks' / 'datasets.json').read_text(encoding='utf-8'))
    statement_note = '; '.join(f'{name}, "{statements[k]["knowledge"]}"' for k, name in
                               (('pheno', 'phenobarbital'), ('remifentanil', 'remifentanil'), ('oral_mm', 'oral MM')))
    t1 = add_table(d, '**Table 1.** Benchmark datasets, reference models, and expert statements',
                   ['Dataset', 'Design', 'Reference model'], table1_rows(cites),
                   f'Expert statements given in the expert-statement condition (verbatim): {statement_note} '
                   'CL, clearance; IV, intravenous; LBM, lean body mass; MM, Michaelis–Menten; Q2 and Q3, '
                   'intercompartmental clearances; V, V1, V2, volumes of distribution. '
                   'Details of the designs and reference models are given in Methods and Table S2.',
                   [1.65, 3.6, 3.75], size=12, note_size=12, note_spacing=WD_LINE_SPACING.DOUBLE, fixed=True)
    if len(cites.order) != n_refs:
        raise SystemExit(f'Table 1 cites a reference not cited in the text: {cites.order[n_refs:]}')
    header = ['Dataset, condition', 'Runs', 'Reference structure', 'Reference relationships (form)',
              'Other relationships', 'Typical values agree', 'Reproduced', 'ΔOFV', 'ΔAIC']

    def two_lines(v):                                    # median (range): the range on its own line
        return str(v).replace(' (', '\n(', 1)
    rows = [[f"{r['dataset']}, {r['condition'].lower()}", r['runs'], r['structure'],
             'NA' if r['relationships'] == 'NA' else f"{r['relationships']} ({r['forms']})", two_lines(r['extra']),
             r['typical'], r['reproduced'], two_lines(r['delta_ofv']), two_lines(r['delta_aic'])]
            for r in table2]
    t2 = add_table(d, '**Table 2.** Final models of the agent runs compared with the reference models',
                   header, rows,
                   'AIC, Akaike information criterion; Claude, Claude Opus 5.5; GPT, GPT-6.1 Sol; MM, '
                   'Michaelis–Menten; NA, not applicable; OFV, objective function value. '
                   'Values are counts over runs, or medians (ranges). Runs: number of runs (GPT/Claude). '
                   + ('Misleading statement: deliberately wrong statement (Supplementary Material S1); counts refer '
                      'to the true reference model. ' if misleading else '') +
                   'Reference structure: runs whose final model had the structure of the reference model. Reference '
                   'relationships (form): parameter–covariate pairs of the reference model present in the final model, '
                   'summed over runs, with the number in the reference form family in parentheses (NA, the reference '
                   'model has no covariates). Other relationships: pairs per run in the final model that are not in '
                   'the reference model. Typical values agree: the median ratio of the subject-level typical values '
                   '(final/reference model) lay within 0.80 to 1.25 for every reference parameter. Reproduced: '
                   'reference structure, exactly the reference relationships, and agreement of typical values '
                   '(stochastic models not compared). ΔOFV and ΔAIC: final model minus the PKPy2 fit of the reference '
                   'model, whose stochastic model is given in Table S2 (data scale for log-normal error).',
                   [1.5, .6, .84, 1.0, 1.0, .75, .95, 1.18, 1.18], size=10, note_size=12, fixed=True, new_page=True)
    print('characters per row: Table 1 ' + ', '.join(map(str, chars_per_row(t1))) + '; Table 2 '
          + ', '.join(map(str, chars_per_row(t2))))
    long_rows = [f'Table {k} row {i} ({c} characters)' for k, t in ((1, t1), (2, t2))
                 for i, c in enumerate(chars_per_row(t)) if c > MAX_TABLE_ROW]
    for r in long_rows:
        print(f'WARNING: {r} > {MAX_TABLE_ROW} characters per row')
    if long_rows and '--draft' not in sys.argv:
        raise SystemExit(f'final build refused (use --draft to build anyway): table rows over {MAX_TABLE_ROW} '
                         'characters: ' + '; '.join(long_rows))

    # figure legends and supplementary material
    portrait_section(d)
    for lvl, title, paras in body:
        if title in ('FIGURE LEGENDS', 'SUPPLEMENTARY MATERIAL'):
            heading(d, title, 2)
            for t in paras:
                add_inline(d.add_paragraph(), t)

    OUT.mkdir(exist_ok=True)
    set_properties(d, TITLE)
    d.save(OUT / 'PKAgent_CPT_manuscript.docx')
    cover_letter()
    for n in range(1, n_fig + 1):
        for ext in ('pdf', 'tiff'):
            src = HERE / 'figures' / f'Figure_{n}.{ext}'
            if src.exists():
                shutil.copy(src, OUT / src.name)
    shutil.copy(HERE / 'figures' / 'alt_text.txt', OUT / 'alt_text.txt')     # checked by check_claims.alt_text_checks
    (BUILD / 'manuscript_cpt_filled.md').write_text(text, encoding='utf-8')
    (BUILD / 'reference_order.json').write_text(json.dumps(cites.order, indent=1), encoding='utf-8')
    unused = sorted(set(REFERENCES) - set(cites.order))
    if unused:
        print('references not cited:', unused)
    print('wrote', OUT / 'PKAgent_CPT_manuscript.docx')


def signature_lines():
    """Name, program, university, postal address, and email of the corresponding author, one per line."""
    contact, email = CORRESPONDING.rsplit('. Email: ', 1)
    name, program, university, address = contact.split(', ', 3)
    return [name, program, university, address, f'Email: {email}']


def cover_letter():
    """Business-letter layout: Letter page, 1-inch margins, Times New Roman 12 pt, single spacing with 12 pt after
    each paragraph, page numbers in the footer, signature block on separate lines."""
    import datetime as dt
    today = dt.date.today()
    numbers = json.loads((BUILD / 'numbers.json').read_text(encoding='utf-8'))     # run counts as in the manuscript
    values = dict(numbers, date=f'{today:%B} {today.day}, {today.year}', title=TITLE,
                  ai_disclosure=f'⟦HL⟧{AI_DISCLOSURE}⟦/HL⟧', corresponding='\n'.join(signature_lines()))
    text = (HERE / 'cover_letter_cpt.md').read_text(encoding='utf-8')
    text = re.sub(r'\{\{(\w+)\}\}', lambda m: str(values[m.group(1)]), text)
    d = docx.Document()
    st = d.styles['Normal']
    st.font.name, st.font.size = 'Times New Roman', Pt(12)
    st.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    pf = st.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    pf.space_before, pf.space_after = Pt(0), Pt(12)
    sec = d.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    for side in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'):
        setattr(sec, side, Inches(1))
    page_numbers(sec)
    word_2013_layout(d)
    for block in text.split('\n\n'):
        p = d.add_paragraph()
        for k, line in enumerate(block.strip().splitlines()):
            if k:
                p.add_run().add_break()
            add_inline(p, line)
    set_properties(d, 'Cover letter: ' + TITLE)
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
