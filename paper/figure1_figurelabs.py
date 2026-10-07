"""Figure 1 from the FigureLabs schematic: resampled to the CPT print size (178 mm wide, 600 dpi; CPT asks
for at least 400 dpi for images containing text) and written as PNG, CMYK TIFF and PDF.

Source: paper/figures/figurelabs/figure1_v3_up4k.png (FigureLabs illustration upscaled on the FigureLabs canvas
to 5,504 x 3,072 px, with one local correction; see paper/figures/figurelabs/README.md). The schematic is not
generated from data; paper/figure1_architecture.py draws the earlier vector version (Figure_1_script.*).
Usage: python paper/figure1_figurelabs.py
"""
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
SRC = HERE / 'figures' / 'figurelabs' / 'figure1_v3_up4k.png'
OUT = HERE / 'figures'
WIDTH_MM, DPI = 178, 600


def main():
    im = Image.open(SRC).convert('RGB')
    width = round(WIDTH_MM / 25.4 * DPI)                      # 4,205 px
    height = round(im.height * width / im.width)
    big = im.resize((width, height), Image.LANCZOS)             # Lanczos downsampling from the 4K upscale
    big.save(OUT / 'Figure_1.png', dpi=(DPI, DPI))
    big.convert('CMYK').save(OUT / 'Figure_1.tiff', dpi=(DPI, DPI), compression='tiff_lzw')
    big.save(OUT / 'Figure_1.pdf', resolution=DPI)
    print(f'Figure_1: {width} x {height} px at {DPI} dpi ({WIDTH_MM} x {height / DPI * 25.4:.0f} mm)')


if __name__ == '__main__':
    main()
