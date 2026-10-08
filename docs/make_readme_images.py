"""Web-sized copies of paper figures for README.md.

Reads the 600-dpi PNGs in paper/figures/ (made by paper/graphical_abstract.py, paper/figure1_architecture.py and
benchmarks/figures.py) and writes lossless RGB PNGs on a white background to docs/images/. Each image is resampled
(Lanczos) to the largest of the widths in WIDTHS at which the file stays within MAX_BYTES. The output depends only on
the source PNG and the Pillow version (no metadata, no random elements).

Run from any directory: python docs/make_readme_images.py
"""
import hashlib
import io
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'paper' / 'figures'
OUT = ROOT / 'docs' / 'images'

IMAGES = {                                   # output name: source PNG in paper/figures/
    'graphical_abstract.png': 'Graphical_abstract.png',
    'architecture.png': 'Figure_1.png',
    'covariate_relationships.png': 'Figure_3.png',
}
WIDTHS = (2000, 1800, 1600)                  # pixels, tried in this order
MAX_BYTES = 600 * 1024


def on_white(im):
    """RGB image with any transparency composited on white."""
    rgba = im.convert('RGBA')
    white = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, rgba).convert('RGB')


def png_bytes(im, width):
    height = round(im.height * width / im.width)
    small = im.resize((width, height), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, format='PNG', optimize=True)
    return small.size, buf.getvalue()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [s for s in IMAGES.values() if not (SRC / s).exists()]
    if missing:
        sys.exit(f'missing in {SRC}: {missing}; run the figure scripts first (see paper/README.md)')
    for name, source in IMAGES.items():
        im = on_white(Image.open(SRC / source))
        for width in WIDTHS:
            size, data = png_bytes(im, width)
            if len(data) <= MAX_BYTES:
                break
        else:
            sys.exit(f'{name}: {len(data)} bytes at {WIDTHS[-1]} px, above {MAX_BYTES}')
        (OUT / name).write_bytes(data)
        print(f'{name}: {size[0]} x {size[1]} px, {len(data) / 1024:.0f} KB, '
              f'sha256 {hashlib.sha256(data).hexdigest()[:12]} (from {source})')


if __name__ == '__main__':
    main()
