"""asciify v1.0 — first functional draft (kept for the review record; superseded by the asciify package).

Usage:
    python asciify_v1.py photo.jpg [width] [--html out.html]
"""
import sys

import numpy as np
from PIL import Image

# Paul Bourke's 70-level ramp, dense -> sparse
RAMP = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
CHAR_ASPECT = 0.55  # terminal cells are ~2x taller than wide


def convert(path, width=100):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    height = int(h / w * width * CHAR_ASPECT)
    img = img.resize((width, height))
    px = np.asarray(img).astype(np.float32)
    lum = 0.299 * px[..., 0] + 0.587 * px[..., 1] + 0.114 * px[..., 2]
    idx = ((255 - lum) / 255 * (len(RAMP) - 1)).astype(int)
    rows = []
    for y in range(height):
        rows.append([(RAMP[idx[y, x]], tuple(int(c) for c in px[y, x])) for x in range(width)])
    return rows


def to_ansi(rows):
    out = []
    for row in rows:
        line = ""
        for ch, (r, g, b) in row:
            line += f"\x1b[38;2;{r};{g};{b}m{ch}"
        out.append(line)
    return "\n".join(out) + "\x1b[0m"


def to_html(rows):
    body = ""
    for row in rows:
        for ch, (r, g, b) in row:
            body += f'<span style="color:rgb({r},{g},{b})">{ch}</span>'
        body += "\n"
    return (
        "<html><body style='background:#000'>"
        f"<pre style='font-family:monospace;line-height:1'>{body}</pre></body></html>"
    )


if __name__ == "__main__":
    path = sys.argv[1]
    width = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 100
    rows = convert(path, width)
    if "--html" in sys.argv:
        with open(sys.argv[sys.argv.index("--html") + 1], "w") as f:
            f.write(to_html(rows))
    else:
        print(to_ansi(rows))
