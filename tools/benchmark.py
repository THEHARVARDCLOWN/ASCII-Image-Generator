"""v1 vs v2 measurements behind docs/REVIEW.md. Each large-image case runs in a fresh process.

Usage:  python tools/benchmark.py [photo.jpg]
"""
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "v1")]
CACHE = os.path.join(tempfile.gettempdir(), "asciify_bench")


def synthetic(name, w, h, fmt):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{name}_{w}x{h}.{fmt}")
    if not os.path.exists(path):
        rng = np.random.default_rng(0)
        img = Image.fromarray(rng.integers(0, 255, (h // 32, w // 32, 3), dtype=np.uint8)).resize((w, h))
        img.save(path, quality=85) if fmt == "jpg" else img.save(path, compress_level=1)
    return path


def child(version, path):
    from review_probe import peak_rss_mb

    base = peak_rss_mb()
    t = time.perf_counter()
    try:
        if version == "v1":
            import asciify_v1
            asciify_v1.convert(path, 120)
        else:
            from asciify import Options, convert
            convert(path, Options(width=120))
        print(f"{time.perf_counter() - t:.2f}s, +{peak_rss_mb() - base:.0f} MB peak")
    except Exception as e:
        print(f"CRASH {type(e).__name__}: {str(e)[:60]}")


def run_child(version, path):
    out = subprocess.run([sys.executable, __file__, "--child", version, path], capture_output=True, text=True)
    return (out.stdout.strip() or out.stderr.strip().splitlines()[-1])


def main(photo):
    import asciify_v1 as v1
    from asciify import Options, color, convert, render

    print("## Large images (width 120)")
    for label, (w, h, fmt) in {"54 MP JPEG": (9000, 6000, "jpg"), "192 MP JPEG": (16000, 12000, "jpg"),
                               "54 MP PNG": (9000, 6000, "png")}.items():
        path = synthetic("bench", w, h, fmt)
        print(f"  {label:<12} v1: {run_child('v1', path):<40} v2: {run_child('v2', path)}")

    if not photo:
        return
    print(f"\n## Real photo: {os.path.basename(photo)} at 160 columns")
    rows1 = v1.convert(photo, 160)
    n1 = len(rows1) * len(rows1[0])
    f2 = convert(photo, Options(width=160))
    n2 = f2.rows * f2.cols
    a1, h1 = len(v1.to_ansi(rows1).encode()), len(v1.to_html(rows1).encode())
    a2, h2 = len(render.to_ansi(f2).encode()), len(render.to_html(f2).encode())
    print(f"  ANSI bytes/cell  v1 {a1 / n1:5.1f}   v2 {a2 / n2:5.1f}")
    print(f"  HTML bytes/cell  v1 {h1 / n1:5.1f}   v2 {h2 / n2:5.1f}")
    bad1 = len(re.findall(r">[<>&]<", v1.to_html(rows1)))
    print(f"  unescaped HTML specials  v1 {bad1}   v2 0 (asserted in tests)")

    def stats(rgb, chars):
        lab = color.linear_to_oklab(color.SRGB_LUT[rgb])
        ink = np.array([c != " " for c in chars])
        return lab[ink, 0].mean(), np.hypot(lab[ink, 1], lab[ink, 2]).mean()

    rgb1 = np.array([c for r in rows1 for _, c in r], np.uint8)
    L1, c1 = stats(rgb1, [ch for r in rows1 for ch, _ in r])
    L2, c2 = stats(f2.fg.reshape(-1, 3), list("".join(f2.chars)))
    print(f"  ink color Oklab L / chroma   v1 {L1:.3f} / {c1:.3f}   v2 vivid {L2:.3f} / {c2:.3f}")
    for width in (160, 400):
        t = time.perf_counter(); v1.to_html(v1.convert(photo, width)); t1 = time.perf_counter() - t
        t = time.perf_counter(); render.to_html(convert(photo, Options(width=width))); t2 = time.perf_counter() - t
        print(f"  width {width} photo -> HTML   v1 {t1:.2f}s   v2 {t2:.2f}s")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else None)
