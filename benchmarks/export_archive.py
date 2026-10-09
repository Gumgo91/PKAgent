"""One archive of the benchmark outputs (runs, reference fits, evaluation) without data files or local paths.

Contents: runs (every run: prompts, transcripts with reasoning summaries, tool calls and results, fitted models,
plots, reports), reference_fits and evaluation (incl. agent tests, stepwise baseline, drop-one evidence). The data
files are not redistributed (folders named 'data' are left out), although the per-model diagnostics include the
observations; benchmarks/export_data.R and prepare_data.py recreate them from the R packages nlme and nlmixr2data.
Local absolute paths are removed from all text files.
Writes dist/PKAgent_benchmark_archive.zip.
Usage: python benchmarks/export_archive.py
"""
import re
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FOLDERS = ['runs', 'reference_fits', 'evaluation']
TEXT = {'.json', '.jsonl', '.csv', '.md', '.log', '.txt'}
PREFIX = re.compile(r'[A-Za-z]:(?:\\\\|\\|/)Users(?:\\\\|\\|/)[^\\/"]+(?:\\\\|\\|/)Desktop(?:\\\\|\\|/)PKAgent(?:\\\\|\\|/)')
HOME = re.compile(r'[A-Za-z]:(?:\\\\|\\|/)Users(?:\\\\|\\|/)[^\\/"\s]+')


LOCAL = re.compile(r'~(?:\\\\|\\|/)Desktop(?:\\\\|\\|/)(?:pkpy2(?:\\\\|\\|/)packages(?:\\\\|\\|/)pkpy2(?:\\\\|\\|/)src(?:\\\\|\\|/))?')


def sanitize(text):
    text = PREFIX.sub('', text)
    return LOCAL.sub('', HOME.sub('~', text))       # PKPy2 source paths in console logs become 'pkpy2\...'


def main():
    out = ROOT / 'dist' / 'PKAgent_benchmark_archive.zip'
    out.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for folder in FOLDERS:
            for p in sorted((HERE / folder).rglob('*')):
                if not p.is_file() or p.suffix == '.pyc' or 'data' in p.relative_to(HERE).parts[1:-1]:
                    continue
                arc = Path('benchmarks') / p.relative_to(HERE)
                if p.suffix in TEXT:
                    z.writestr(str(arc).replace('\\', '/'), sanitize(p.read_text(encoding='utf-8', errors='replace')))
                else:
                    z.write(p, str(arc).replace('\\', '/'))
                n += 1
    with zipfile.ZipFile(out) as z:                       # check that no local path is left
        left = [i.filename for i in z.infolist() if Path(i.filename).suffix in TEXT
                and re.search(r'Users[\\/]+[^\\/]+[\\/]+Desktop|~[\\/]+Desktop', z.read(i).decode('utf-8', 'replace'))]
    print(f'wrote {out} ({n} files, {out.stat().st_size / 1e6:.1f} MB); files with local paths: {len(left)}')


if __name__ == '__main__':
    main()
