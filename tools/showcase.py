"""Generate the comparison images in docs/images/ from any photo.

Usage:  python tools/showcase.py photo.jpg [second_photo.jpg]
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "v1")]

import asciify_v1 as v1  # noqa: E402
from asciify import Options, convert, render  # noqa: E402
from asciify.engine import Frame  # noqa: E402

OUT = os.path.join(ROOT, "docs", "images")


def label(img, text):
    d = ImageDraw.Draw(img)
    w = int(d.textlength(text)) + 12
    d.rectangle((0, 0, w, 16), fill=(0, 0, 0))
    d.text((6, 2), text, fill=(255, 214, 102))
    return img


def sheet(tiles, cols, gap=6):
    w = max(t.width for t in tiles)
    h = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap), (40, 40, 44))
    for i, t in enumerate(tiles):
        out.paste(t, ((i % cols) * (w + gap), (i // cols) * (h + gap)))
    return out


def v1_frame(path, width):
    rows = v1.convert(path, width)
    fg = np.array([[c for _, c in r] for r in rows], np.uint8)
    return Frame(chars=["".join(ch for ch, _ in r) for r in rows], fg=fg, lower=None, background=(12, 12, 14),
                 foreground=(214, 214, 214), mode="ascii", colored=True, cell_ratio=0.5, source_size=(0, 0))


def main(photo, second=None):
    os.makedirs(OUT, exist_ok=True)
    fs, width = 11, 96
    before = label(render.to_image(v1_frame(photo, width), font_size=fs), "v1.0  (index ramp, raw colors, 0.55 aspect)")
    after = label(render.to_image(convert(photo, Options(width=width)), font_size=fs), "v2.0  default")
    sheet([before, after], 2).save(os.path.join(OUT, "v1_vs_v2.png"), optimize=True)

    subject = second or photo
    modes = [label(render.to_image(convert(subject, Options(width=width, mode=m)), font_size=fs), m)
             for m in ("ascii", "edges", "blocks")]
    light = render.to_image(convert(photo, Options(width=width, light=True)), font_size=fs)
    modes.append(label(light.crop((0, 0, modes[0].width, min(light.height, modes[0].height))), "ascii --light"))
    sheet(modes, 2).save(os.path.join(OUT, "modes.png"), optimize=True)

    names = ["vivid", "true", "matrix", "amber", "synthwave", "inferno", "ice", "sepia"]
    pals = [label(render.to_image(convert(photo, Options(width=72, palette=n)), font_size=9), n) for n in names]
    sheet(pals, 2).save(os.path.join(OUT, "palettes.png"), optimize=True)
    for name in ("v1_vs_v2.png", "modes.png", "palettes.png"):
        p = os.path.join(OUT, name)
        print(f"{p}  {Image.open(p).size}  {os.path.getsize(p) // 1024} KB")


if __name__ == "__main__":
    main(*sys.argv[1:3])
