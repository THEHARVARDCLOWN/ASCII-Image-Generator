"""Measure the ink coverage of each printable ASCII glyph in a monospace font.

The output is the COVERAGE table embedded in asciify/glyphs.py. Regenerate it for your terminal font with:
    python tools/measure_glyphs.py path/to/YourMono.ttf
Coverage = fraction of the full character *cell* (advance width x line height) that is inked.
"""
import math
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PRINTABLE = "".join(chr(c) for c in range(32, 127))


def measure(font_path, size=96):
    font = ImageFont.truetype(font_path, size)
    ascent, descent = font.getmetrics()
    w, h = math.ceil(font.getlength("M")), ascent + descent
    out = {}
    for ch in PRINTABLE:
        im = Image.new("L", (w, h), 0)
        ImageDraw.Draw(im).text((0, 0), ch, fill=255, font=font)
        out[ch] = float(np.asarray(im, dtype=np.float32).mean() / 255)
    return out, (w, h)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "consola.ttf"
    cov, cell = measure(path)
    print(f"# measured from {path} at 96px, cell {cell[0]}x{cell[1]}")
    items = sorted(cov.items(), key=lambda kv: kv[1])
    print("COVERAGE = {")
    for i in range(0, len(items), 6):
        print("    " + " ".join(f"{k!r}: {v:.4f}," for k, v in items[i:i + 6]))
    print("}")
