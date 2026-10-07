"""Graphical abstract (optional for CPT: PDF, TIFF or EPS, width 1.6 times the height, at least 300 dpi).

Source: paper/figures/figurelabs/ga_final_up4k.png (FigureLabs illustration upscaled on the FigureLabs canvas; see
paper/figures/figurelabs/README.md). The image is padded with white to the 1.6 aspect ratio and written at 7 inches
wide at 600 dpi to paper/figures/ and paper/submission_cpt/, together with the graphical abstract text.
Usage: python paper/graphical_abstract.py
"""
import shutil
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
SRC = HERE / 'figures' / 'figurelabs' / 'ga_final_up4k.png'
OUT, SUB = HERE / 'figures', HERE / 'submission_cpt'
RATIO, WIDTH_IN, DPI, MARGIN = 1.6, 7.0, 600, 0.015            # margin: fraction of the width on each side

TEXT = ('PKAgent lets a large language model develop population pharmacokinetic models only through the tools of an '
        'open-source estimation engine, with every step logged. On three public datasets, two language models found '
        'the reference structures in all 36 runs and kept strongly supported covariate effects in 11 of 12 runs '
        'without expert knowledge. The knowledge an analyst states shapes weakly supported effects and should be '
        'reported with the model.')


def main():
    im = Image.open(SRC).convert('RGB')
    width = round(im.width * (1 + 2 * MARGIN))
    height = round(width / RATIO)
    if height < im.height:                                  # wider than 1.6: pad the width instead
        height = round(im.height * (1 + 2 * MARGIN))
        width = round(height * RATIO)
    canvas = Image.new('RGB', (width, height), 'white')
    canvas.paste(im, ((width - im.width) // 2, (height - im.height) // 2))
    dpi = DPI
    width, height = round(WIDTH_IN * DPI), round(WIDTH_IN * DPI / RATIO)
    canvas = canvas.resize((width, height), Image.LANCZOS)
    canvas.save(OUT / 'Graphical_abstract.png', dpi=(dpi, dpi))
    canvas.convert('CMYK').save(OUT / 'Graphical_abstract.tiff', dpi=(dpi, dpi), compression='tiff_lzw')
    canvas.save(OUT / 'Graphical_abstract.pdf', resolution=dpi)
    (OUT / 'Graphical_abstract.txt').write_text(TEXT + '\n', encoding='utf-8')
    SUB.mkdir(exist_ok=True)
    for ext in ('tiff', 'pdf', 'txt'):
        shutil.copy(OUT / f'Graphical_abstract.{ext}', SUB / f'Graphical_abstract.{ext}')
    print(f'Graphical abstract: {width} x {height} px (ratio {width / height:.3f}), {dpi} dpi at {WIDTH_IN} in; '
          f'text {len(TEXT.split())} words')


if __name__ == '__main__':
    main()
