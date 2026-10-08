"""Make the empty Word templates of the submission from the author's earlier paper (its format only).

The manuscript and the supplementary file of the author's earlier paper (read, never changed) define the format:
Letter page, margins 1.25 in left and right and 1.0 in top and bottom, no header, footer, page or line numbers, Times
New Roman 11 pt with 1.15 line spacing and 10 pt after each paragraph (Normal style and document defaults), and the
grid table style. This script keeps their styles, settings, theme, numbering, font table, web settings and section
properties, and drops everything that carries content: the body (one empty paragraph is left), images, footnotes and
endnotes (separators only), custom XML (bibliography), comments, headers and footers, hyperlinks and their
relationships, and the revision identifiers and document ids in the settings. Document properties are reset (author
Hyunseung Kong, no title, no statistics). It then checks that no text node is left in any part and, optionally, that
none of the words given after --forbid occurs anywhere (case-insensitive).

Usage: python paper/templates/make_templates.py MANUSCRIPT.docx SUPPLEMENT.docx [--forbid WORD ...]
Writes paper/templates/manuscript_template.docx and paper/templates/supplement_template.docx, which build_cpt.py and
supplement_cpt.py open as the base of every file they write.
"""
import re
import sys
import zipfile
from pathlib import Path

from lxml import etree

HERE = Path(__file__).resolve().parent
TARGETS = [HERE / 'manuscript_template.docx', HERE / 'supplement_template.docx']
AUTHOR = 'Hyunseung Kong'
CREATED = '2026-10-08T00:00:00Z'

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
W14 = 'http://schemas.microsoft.com/office/word/2010/wordml'
W15 = 'http://schemas.microsoft.com/office/word/2012/wordml'
PKG_REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
OFFICE_REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
KEPT = {                                                  # part name -> (relationship type, content type)
    'word/styles.xml': ('styles', 'wordprocessingml.styles+xml'),
    'word/settings.xml': ('settings', 'wordprocessingml.settings+xml'),
    'word/webSettings.xml': ('webSettings', 'wordprocessingml.webSettings+xml'),
    'word/fontTable.xml': ('fontTable', 'wordprocessingml.fontTable+xml'),
    'word/numbering.xml': ('numbering', 'wordprocessingml.numbering+xml'),
    'word/theme/theme1.xml': ('theme', 'theme+xml'),
}
OOXML = 'application/vnd.openxmlformats-officedocument.'


def w(tag):
    return f'{{{W}}}{tag}'


def clean_document(xml):
    """Body reduced to one empty paragraph and the final section properties (page size and margins), without
    revision ids and without header or footer references."""
    root = etree.fromstring(xml)
    body = root.find(w('body'))
    sect = body.find(w('sectPr'))
    for el in list(body):
        body.remove(el)
    for ref in sect.findall(w('headerReference')) + sect.findall(w('footerReference')):
        sect.remove(ref)
    for attr in list(sect.attrib):
        if attr.startswith(f'{{{W}}}rsid'):
            del sect.attrib[attr]
    etree.SubElement(body, w('p'))
    body.append(sect)
    return etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)


def clean_settings(xml):
    """Settings without revision ids, document ids, footnote and endnote references (those parts are dropped), an
    attached template, protection, or mail merge."""
    root = etree.fromstring(xml)
    for tag in (w('rsids'), w('footnotePr'), w('endnotePr'), w('attachedTemplate'), w('documentProtection'),
                w('writeProtection'), w('mailMerge'), f'{{{W14}}}docId', f'{{{W15}}}docId'):
        for el in root.findall(tag):
            root.remove(el)
    return etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)


def content_types():
    over = ''.join(f'<Override PartName="/{name}" ContentType="{OOXML}{ct}"/>' for name, (_, ct) in KEPT.items())
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            f'<Override PartName="/word/document.xml" ContentType="{OOXML}wordprocessingml.document.main+xml"/>'
            + over +
            '<Override PartName="/docProps/core.xml" '
            'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            f'<Override PartName="/docProps/app.xml" ContentType="{OOXML}extended-properties+xml"/>'
            '</Types>').encode()


def package_rels():
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            f'<Relationships xmlns="{PKG_REL}">'
            f'<Relationship Id="rId1" Type="{OFFICE_REL}/officeDocument" Target="word/document.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/'
            'core-properties" Target="docProps/core.xml"/>'
            f'<Relationship Id="rId3" Type="{OFFICE_REL}/extended-properties" Target="docProps/app.xml"/>'
            '</Relationships>').encode()


def document_rels():
    rels = ''.join(f'<Relationship Id="rId{i}" Type="{OFFICE_REL}/{kind}" Target="{name[len("word/"):]}"/>'
                   for i, (name, (kind, _)) in enumerate(KEPT.items(), 1))
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            f'<Relationships xmlns="{PKG_REL}">{rels}</Relationships>').encode()


def core():
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            f'<dc:title></dc:title><dc:subject></dc:subject><dc:creator>{AUTHOR}</dc:creator>'
            f'<cp:keywords></cp:keywords><dc:description></dc:description><cp:lastModifiedBy>{AUTHOR}'
            '</cp:lastModifiedBy><cp:revision>1</cp:revision>'
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{CREATED}</dcterms:created>'
            f'<dcterms:modified xsi:type="dcterms:W3CDTF">{CREATED}</dcterms:modified>'
            '</cp:coreProperties>').encode()


def app():
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
            'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
            '<Template>Normal</Template><TotalTime>0</TotalTime><Application>Microsoft Office Word</Application>'
            '<DocSecurity>0</DocSecurity><ScaleCrop>false</ScaleCrop><Company></Company>'
            '<LinksUpToDate>false</LinksUpToDate><SharedDoc>false</SharedDoc>'
            '<HyperlinksChanged>false</HyperlinksChanged><AppVersion>16.0000</AppVersion></Properties>').encode()


def make(source, target, forbid):
    with zipfile.ZipFile(source) as z:
        parts = {name: z.read(name) for name in KEPT}
        document = z.read('word/document.xml')
    parts['word/document.xml'] = clean_document(document)
    parts['word/settings.xml'] = clean_settings(parts['word/settings.xml'])
    parts.update({'[Content_Types].xml': content_types(), '_rels/.rels': package_rels(),
                  'word/_rels/document.xml.rels': document_rels(), 'docProps/core.xml': core(),
                  'docProps/app.xml': app()})
    order = ['[Content_Types].xml', '_rels/.rels', 'word/document.xml', 'word/_rels/document.xml.rels',
             *[n for n in KEPT], 'docProps/core.xml', 'docProps/app.xml']
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in order:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, parts[name])
    check(target, forbid)


def check(target, forbid):
    """No part other than the kept ones, no image, no text node in any part, and none of the forbidden words."""
    allowed = set(KEPT) | {'[Content_Types].xml', '_rels/.rels', 'word/document.xml', 'word/_rels/document.xml.rels',
                           'docProps/core.xml', 'docProps/app.xml'}
    with zipfile.ZipFile(target) as z:
        names = z.namelist()
        extra = sorted(set(names) - allowed)
        media = [n for n in names if n.startswith('word/media/')]
        found = {(n, word) for n in names for word in forbid
                 if re.search(re.escape(word), z.read(n).decode('utf-8', 'replace'), re.I)}
        texts = [n for n in names if re.search(rb'<w:t[ >]', z.read(n))]
    problems = ([f'unexpected parts {extra}'] if extra else []) + ([f'images {media}'] if media else []) + \
               ([f'forbidden words: {sorted(found)}'] if found else []) + \
               ([f'text nodes left in {texts}'] if texts else [])
    if problems:
        raise SystemExit(f'{target.name}: ' + '; '.join(problems))
    print(f'wrote {target.relative_to(HERE.parent.parent).as_posix()}: {len(names)} parts, no images, no text'
          + (f', none of the {len(forbid)} forbidden words' if forbid else ''))


def main():
    args = sys.argv[1:]
    cut = args.index('--forbid') if '--forbid' in args else len(args)
    sources, forbid = [Path(p) for p in args[:cut]], args[cut + 1:]
    if len(sources) != 2:
        raise SystemExit('usage: python paper/templates/make_templates.py MANUSCRIPT.docx SUPPLEMENT.docx '
                         '[--forbid WORD ...]')
    for source, target in zip(sources, TARGETS):
        make(source, target, forbid)


if __name__ == '__main__':
    main()
