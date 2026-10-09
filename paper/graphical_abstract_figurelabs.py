"""Graphical abstract for CPT, from the FigureLabs vector export.

The drawing was made with FigureLabs on 2026-10-09 (project https://chat.figurelabs.ai/project/2108416590436233218;
illustration, style Flat, ratio 3:2; generation job_2108417024764801027 and one edit, job_2108417584842797058,
that removed three column headers and two pairs of quotation marks), opened in the FigureLabs vector canvas ("Edit in
Canvas") and exported as SVG, so every shape, icon and text line is a FigureLabs vector element. The raw export, the
native rasters (1,264 x 848 JPEG), the prompt and the edit instruction are kept in paper/figures/figurelabs/
(git-ignored; ga2_*).

Steps:
1. Copies the raw export without the editor's scene metadata (which holds signed links to the source raster) to
   paper/figures/Graphical_abstract_source.svg, the input of the next steps (used as is when the raw export is absent).
2. Sets the size, weight, color (#1F2933) and alignment of each of the 44 text lines (the export dropped bold and gave
   lines of one block different sizes; the strings are not changed), recolors the condition and language-model colors
   to the exact palette of Figures 2 to 4, and crops the page to the CPT print size, 7.0 x 4.375 in (width = 1.6 x
   height), with even margins: paper/figures/Graphical_abstract.svg.
3. Checks that the text lines are exactly the expected ones, that the claims on the cards hold in
   paper/build/numbers.json (run paper/manuscript_numbers.py first), that the banner is the last sentence of the
   abstract in paper/manuscript_cpt.md, and that no fee, time or logging appears.
4. Prints the PDF with headless Chromium (playwright; Arial embedded, no raster image), renders the PNG from that PDF
   at 600 dpi (PyMuPDF) and the CMYK TIFF from the PNG (benchmarks/figures.py tiff_cmyk), and checks the words, page
   size, fonts and smallest text size of the PDF.
5. Writes the graphical abstract text (ga_text of paper/graphical_abstract.py, which also checks its values against
   numbers.json) to Graphical_abstract.txt, and copies the .pdf, .tiff and .txt to paper/submission_cpt/.

Usage: python paper/graphical_abstract_figurelabs.py [RAW_SVG]   (default: paper/figures/figurelabs/ga2_v2_vector.svg)
"""
import html
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'benchmarks'))
sys.path.insert(0, str(HERE))
from figures import COLOR, LLM, tiff_cmyk  # noqa: E402
from graphical_abstract import ga_text, load_numbers, runs_of, word  # noqa: E402

FIG, SUB = HERE / 'figures', HERE / 'submission_cpt'
RAW = FIG / 'figurelabs' / 'ga2_v2_vector.svg'
SOURCE = FIG / 'Graphical_abstract_source.svg'
STEM = 'Graphical_abstract'
NUMBERS = HERE / 'build' / 'numbers.json'
W_IN, H_IN, DPI = 7.0, 4.375, 600
PAGE = (-18.9, 58.9, 1301.8, 813.6)      # page in canvas units (x, y, width, height): ratio 1.6, even margins
PT = W_IN * 72 / PAGE[2]                 # pt per canvas unit at the print size
MIN_PT = 7.0                             # smallest text at the print size (pt)

# the take-home banner: the last sentence of the abstract, in two lines
BANNER = ('Where data are strong, PKAgent recovers the model; where they are weak, the knowledge',
          'an analyst states shapes the model and should be reported with it.')

# exact paper palette (FigureLabs' colors were close but not identical): Figures 2 to 4 and the earlier vector GA
PALETTE = {'rgb(168,107,13)': 'rgb(200,128,30)',    # orange #C8801E, expert statement
           'rgb(156,53,39)': 'rgb(181,71,58)',      # red #B5473A, misleading statement
           'rgb(130,137,151)': 'rgb(138,148,166)',  # gray #8A94A6, no knowledge
           'rgb(42,108,120)': 'rgb(43,122,140)'}    # teal #2B7A8C, language model
assert (COLOR['none'], COLOR['knowledge']) == ('#8A94A6', '#C8801E'), 'condition colors differ from Figures 2 to 4'

# every text line of the export: (string, center y of its group in the export, font size in canvas units, weight,
# 'L' left edge or 'C' center at anchor x, optional vertical shift that evens out the line pitch)
B, R = '700', '400'
SPEC = [
    ('Three public datasets', 106, 24, B, 'C', 182.2),
    ('Neonatal', 165, 23, R, 'L', 147),
    ('phenobarbital', 194, 23, R, 'L', 147),
    ('Adult', 267, 23, R, 'L', 147),
    ('remifentanil', 295, 23, R, 'L', 147),
    ('Simulated', 370, 23, R, 'L', 147),
    ('oral drug', 399, 23, R, 'L', 147),
    ('Three conditions', 485, 24, B, 'C', 182.2),
    ('no knowledge', 543, 23, R, 'C', 182.2),
    ('expert statement', 609, 23, R, 'C', 182.2),
    ('misleading statement', 679, 23, R, 'C', 182.2),
    ('PKAgent', 126, 38, B, 'C', 614.9),
    ('LLM', 205, 34, B, 'L', 573),
    ('GPT-6.1 Sol', 244, 22, R, 'L', 567),
    ('Claude Opus 5.5', 273, 22, R, 'L', 567),
    ('only', 391, 22, R, 'C', 556),
    ('through', 417, 22, R, 'C', 556),
    ('tools', 443, 22, R, 'C', 556),
    ('PKPy2', 574, 36, B, 'L', 569),
    ('open-source', 617, 22, R, 'C', 625),
    ('estimation engine', 641, 22, R, 'C', 625),
    ('Strong evidence: recovered', 121, 24, B, 'C', 1069.9),
    ('reference structure in all 36 runs', 160, 19, R, 'L', 937),
    ('saturable elimination inferred', 193, 19, R, 'L', 937),
    ('from the data', 216, 19, R, 'L', 937),
    ('strongly supported covariate', 249, 19, R, 'L', 937),
    ('effects kept in all but one run', 272, 19, R, 'L', 937),
    ('Weak evidence', 342, 24, B, 'C', 1069.9),
    ('Apgar score on phenobarbital volume,', 376, 19, R, 'C', 1069.9),
    ('age on remifentanil central volume', 399, 19, R, 'C', 1069.9),
    ('no knowledge', 446, 18.5, B, 'L', 935),
    ('expert', 447, 18.5, B, 'L', 1127),
    ('statement', 468, 18.5, B, 'L', 1127),
    ('left out, like', 474, 18.5, R, 'C', 983.4, -1.0),
    ('stepwise covariate', 496, 18.5, R, 'C', 983.4),
    ('selection', 516, 18.5, R, 'C', 983.4, 1.5),
    ('included', 495, 18.5, R, 'L', 1127),
    ('Misleading statement', 585, 24, B, 'C', 1069.9),
    ('claims the data contradicted:', 619, 19, R, 'L', 950),
    ('rejected', 643, 19, R, 'L', 950),
    ('unsupported second', 672, 19, R, 'L', 950),
    ('compartment: adopted', 694, 19, R, 'L', 950),
    (BANNER[0], 776, 24, B, 'L', 125),
    (BANNER[1], 808, 24, B, 'L', 125),
]

TEXT_RE = re.compile(
    r'(<g transform="matrix\(1 0 0 1 ([-\d.]+) ([-\d.]+)\)" style=""\s*>\s*)'
    r'<text([^>]*)font-size="([\d.]+)"([^>]*)font-weight="(\d+)"([^>]*)>\s*'
    r'<tspan x="([-\d.]+)" y="([-\d.]+)" >([^<]*)</tspan></text>', re.S)
FORBIDDEN = re.compile(r'logged|\blog\b|\bfees?\b|\$|USD|\bhours?\b|\bh\b|\bcosts?\b|\btime\b|\bminutes?\b', re.I)


def strip_metadata(svg):
    """The SVG without the editor's scene metadata (signed links to the source raster)."""
    out = re.sub(r'<metadata.*?</metadata>', '', svg, flags=re.S)
    assert 'amazonaws' not in out and 'X-Amz' not in out, 'a signed link is left in the SVG'
    return out


def normalize(svg):
    """Typography of the 44 text lines, exact palette, page cropped to 7.0 x 4.375 in."""
    used = set()

    def repl(m):
        head, cx, cy = m.group(1), float(m.group(2)), float(m.group(3))
        pre, fs0, mid, post = m.group(4), float(m.group(5)), m.group(6), m.group(8)
        y0, text = float(m.group(10)), m.group(11)
        hits = [i for i, sp in enumerate(SPEC) if sp[0] == html.unescape(text) and abs(sp[1] - cy) < 6
                and i not in used]
        if len(hits) != 1:
            raise SystemExit(f'no unique text specification for {text!r} at y = {cy:.1f}')
        i = hits[0]
        used.add(i)
        _, _, fs, wt, align, ax = SPEC[i][:6]
        dy = SPEC[i][6] if len(SPEC[i]) > 6 else 0.0
        y = y0 * fs / fs0 + dy                  # keeps the vertical center of the line box
        post = post.replace('fill: rgb(0,0,0)', 'fill: rgb(31,41,51)')
        anchor = ' text-anchor="middle"' if align == 'C' else ''
        return (f'{head}<text{pre}font-size="{fs}"{mid}font-weight="{wt}"{anchor}{post}>'
                f'<tspan x="{ax - cx:.3f}" y="{y:.3f}" >{text}</tspan></text>')

    svg, n = TEXT_RE.subn(repl, svg)
    assert n == len(SPEC) == len(used), f'{n} text lines in the export, {len(SPEC)} specified, {len(used)} matched'
    for old, new in PALETTE.items():
        svg = svg.replace(f'fill: {old}', f'fill: {new}').replace(f'stroke: {old}', f'stroke: {new}')
    x0, y0, w, h = PAGE
    assert abs(w / h - W_IN / H_IN) < 1e-3
    svg, k = re.subn(r'width="1265" height="849" viewBox="[^"]*"',
                     f'width="{W_IN:g}in" height="{H_IN:g}in" viewBox="{x0} {y0} {w} {h}"', svg, count=1)
    assert k == 1 and '</defs>' in svg, 'unexpected SVG header'
    return svg.replace('</defs>', f'</defs>\n<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#FFFFFF" />', 1)


def abstract_last_sentence():
    md = (HERE / 'manuscript_cpt.md').read_text(encoding='utf-8')
    abstract = re.search(r'^## ABSTRACT\s*\n(.*?)\n## ', md, re.S | re.M).group(1)
    abstract = re.sub(r'<!--.*?-->|\{\{\w+\}\}', '', abstract).strip()     # conditional markers, inserted prose
    return re.split(r'(?<=[.])\s+(?=[A-Z])', abstract)[-1].strip()


def check_claims(n, v):
    """The card lines that carry counts, rebuilt from numbers.json, and the claims of the other lines. load_numbers()
    in paper/graphical_abstract.py has already checked the expert-statement, weak-effect and misleading-statement
    claims (every reference relationship with the statement; Apgar on V and age on V1 in no run without knowledge;
    weight kept, Apgar on CL and linear elimination rejected, second compartment adopted)."""
    expected = Counter(sp[0] for sp in SPEC)
    k, t = runs_of(n['structure_all'])
    assert k == t, 'the card says "in all runs": not every run had the reference structure'
    assert f'reference structure in all {t} runs' in expected
    # strongly supported covariate effects: kept in every expert-statement run, and in strong_kept without knowledge
    ks, ts = runs_of(n['strong_kept'])
    missed = ts - ks
    line = ('effects kept in every run' if missed == 0
            else f"effects kept in all but {word(missed)} run{'s' if missed > 1 else ''}")
    assert line in expected, f'card line should read {line!r}'
    # saturable elimination chosen from the data: MM elimination in every oral-drug run without knowledge
    k, t = runs_of(n['oral_mm_none_mm'])
    assert k == t, 'a run without knowledge did not select saturable elimination'
    # weak effects left out 'like stepwise covariate selection': the stepwise baselines removed the same two effects
    assert n['scm_pheno_removed'] == 'V–APGR', n['scm_pheno_removed']
    assert n['backward_remifentanil_removed'] == 'the age effect on V1', n['backward_remifentanil_removed']
    assert {'GPT-6.1 Sol', 'Claude Opus 5.5'} == set(LLM.values()) and v['n_llm'] == 2
    assert word(v['n_datasets']).capitalize() + ' public datasets' in expected
    banner = ' '.join(BANNER)
    assert banner == abstract_last_sentence(), f'banner differs from the last sentence of the abstract: {banner!r}'
    found = [m for sp in SPEC for m in FORBIDDEN.findall(sp[0])]
    assert not found, f'fees, time or logging on the image: {found}'


def print_pdf(svg, pdf):
    from playwright.sync_api import sync_playwright
    body = re.sub(r'<\?xml[^>]*\?>|<!DOCTYPE[^>]*>', '', svg)
    css = (f'@page{{size:{W_IN:g}in {H_IN:g}in;margin:0}} html,body{{margin:0;padding:0;background:#fff}} '
           f'svg{{display:block;width:{W_IN:g}in;height:{H_IN:g}in}}')
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(java_script_enabled=False)
            page.set_content(f'<html><head><style>{css}</style></head><body>{body}</body></html>')
            page.pdf(path=str(pdf), width=f'{W_IN:g}in', height=f'{H_IN:g}in', print_background=True,
                     prefer_css_page_size=True)
        finally:
            browser.close()
    # Chromium's page box is about 0.12 pt taller than asked: keep the top 7.0 x 4.375 in (the extra strip is white)
    import pymupdf
    with pymupdf.open(pdf) as doc:
        mb = doc[0].mediabox                    # PDF coordinates, y upward
        assert abs(mb.width - W_IN * 72) < 0.01 and 0 <= mb.height - H_IN * 72 < 0.5, mb
        doc[0].set_mediabox(pymupdf.Rect(mb.x0, mb.y1 - H_IN * 72, mb.x1, mb.y1))
        data = doc.tobytes(garbage=3, deflate=True)
    Path(pdf).write_bytes(data)


def check_pdf_and_render_png(pdf, png):
    """Page size, embedded fonts, no raster image, the words of the text lines, the smallest text; PNG at 600 dpi."""
    import pymupdf
    with pymupdf.open(pdf) as doc:
        assert doc.page_count == 1, f'{doc.page_count} pages'
        page = doc[0]
        assert abs(page.rect.width - W_IN * 72) < 0.01 and abs(page.rect.height - H_IN * 72) < 0.01, page.rect
        assert not page.get_images(), 'the PDF holds a raster image'
        fonts = page.get_fonts()
        assert fonts and all(f[1] != 'n/a' for f in fonts), f'a font is not embedded: {fonts}'
        assert all('Arial' in f[3] for f in fonts), [f[3] for f in fonts]
        words = Counter(w[4] for w in page.get_text('words'))
        want = Counter(w for sp in SPEC for w in sp[0].split())
        assert words == want, f'PDF words differ: missing {want - words}, extra {words - want}'
        sizes = [s['size'] for b in page.get_text('dict')['blocks'] for ln in b.get('lines', [])
                 for s in ln['spans'] if s['text'].strip()]
        pix = page.get_pixmap(matrix=pymupdf.Matrix(DPI / 72, DPI / 72), alpha=False)
        assert (pix.width, pix.height) == (round(W_IN * DPI), round(H_IN * DPI)), (pix.width, pix.height)
        pix.save(png)
    from PIL import Image
    with Image.open(png) as im:                 # record the resolution in the PNG
        im.load()
        im.save(png, dpi=(DPI, DPI))
    return sorted({f[3] for f in fonts}), min(sizes)


def main():
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW
    if raw.exists():
        SOURCE.write_text(strip_metadata(raw.read_text(encoding='utf-8')), encoding='utf-8')
    elif not SOURCE.exists():
        raise SystemExit(f'neither {raw} nor {SOURCE} exists')
    n = json.loads(NUMBERS.read_text(encoding='utf-8'))
    v = load_numbers()                          # also checks the claims on the cards (see check_claims)
    check_claims(n, v)
    svg = normalize(SOURCE.read_text(encoding='utf-8'))
    lines = [html.unescape(t) for t in re.findall(r'<tspan[^>]*>([^<]*)</tspan>', svg)]
    assert Counter(lines) == Counter(sp[0] for sp in SPEC), 'text lines differ from the specification'
    sizes = {sp[0]: sp[2] * PT for sp in SPEC}
    assert min(sizes.values()) >= MIN_PT, f'text below {MIN_PT} pt'
    text_ = ga_text(v)
    assert 50 <= len(text_.split()) <= 80 and '—' not in text_ and not FORBIDDEN.search(text_)

    out = {ext: FIG / f'{STEM}.{ext}' for ext in ('svg', 'pdf', 'png', 'tiff', 'txt')}
    out['svg'].write_text(svg, encoding='utf-8')
    print_pdf(svg, out['pdf'])
    fonts, min_pdf = check_pdf_and_render_png(out['pdf'], out['png'])
    tiff_cmyk(out['png'], out['tiff'])
    out['txt'].write_text(text_ + '\n', encoding='utf-8')
    SUB.mkdir(exist_ok=True)
    for ext in ('pdf', 'tiff', 'txt'):
        shutil.copy(out[ext], SUB / f'{STEM}.{ext}')

    small = sorted(sizes.items(), key=lambda kv: kv[1])[:3]
    print(f'source: {SOURCE.relative_to(HERE.parent)} (from {raw.name if raw.exists() else "the copy"})')
    print(f'{len(lines)} text lines as specified; sizes {min(sizes.values()):.2f} to {max(sizes.values()):.2f} pt '
          f'(smallest: {", ".join(f"{s!r} {p:.2f}" for s, p in small)}); PDF smallest span {min_pdf:.2f} pt; '
          f'fonts {", ".join(fonts)}')
    print('banner:', ' '.join(BANNER))
    print('text:', text_)
    print(f'Graphical abstract: {W_IN} x {H_IN} in (ratio {W_IN / H_IN:.3f}), vector PDF, PNG and CMYK TIFF at '
          f'{DPI} dpi; text {len(text_.split())} words')


if __name__ == '__main__':
    main()
