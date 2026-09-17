"""Adversarial probe for asciify v1.0. Every finding in docs/REVIEW.md is backed by a measurement here.

Run:  python v1/review_probe.py [path-to-a-real-photo]
"""
import ctypes
import io
import os
import re
import sys
import tempfile
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
import asciify_v1 as v1  # noqa: E402

TMP = tempfile.mkdtemp(prefix="asciify_probe_")


def peak_rss_mb():
    if os.name != "nt":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 2**20


def save(img, name, **kw):
    p = os.path.join(TMP, name)
    img.save(p, **kw)
    return p


def probe(title, fn):
    print(f"\n## {title}")
    try:
        fn()
    except Exception as e:  # the probe records crashes as findings
        print(f"  CRASH: {type(e).__name__}: {e}")
        traceback.print_exc(limit=1, file=sys.stdout)


def aspect():
    img = Image.new("RGB", (400, 400), "black")
    ImageDraw.Draw(img).ellipse((0, 0, 399, 399), fill="white")
    rows = v1.convert(save(img, "circle.png"), 80)
    h, w = len(rows), len(rows[0])
    # Rendered shape = rows*cell_h : cols*cell_w. Real cell width/height ratios:
    for name, ratio in [("Windows Terminal, Cascadia Mono 12pt (9x19 px cell)", 9 / 19),
                        ("HTML <pre> line-height:1, 0.6em-advance font", 0.6),
                        ("HTML default line-height:normal (~1.17em), Consolas 0.55em", 0.55 / 1.17)]:
        print(f"  {name}: circle renders {h / (w * ratio):.2f}x taller than wide (1.00 is correct)")


def panorama():
    rows = v1.convert(save(Image.new("RGB", (4000, 20), "gray"), "pano.png"), 100)
    print(f"  rows={len(rows)}")


def exif_rotation():
    img = Image.new("RGB", (300, 100), "white")  # stored landscape
    exif = Image.Exif(); exif[0x0112] = 6        # 'rotate 90 CW on display' => portrait photo
    rows = v1.convert(save(img, "phone.jpg", exif=exif), 60)
    print(f"  photo displays as 100x300 portrait, v1 output grid = {len(rows[0])}x{len(rows)} (landscape => sideways)")


def transparency():
    img = Image.new("RGBA", (200, 200), (255, 0, 0, 0))  # fully transparent, hidden RGB = red
    ImageDraw.Draw(img).ellipse((50, 50, 150, 150), fill=(255, 255, 255, 255))
    rows = v1.convert(save(img, "logo.png"), 40)
    corner = rows[0][0]
    print(f"  transparent corner rendered as char={corner[0]!r} color={corner[1]} (should be blank background)")


def sixteen_bit():
    arr = (np.linspace(0, 65535, 256)[None, :].repeat(64, 0)).astype(np.uint16)
    img = Image.fromarray(arr)  # uint16 -> mode I;16
    rows = v1.convert(save(img, "scan16.png"), 64)
    chars = "".join(ch for ch, _ in rows[len(rows) // 2])
    print(f"  16-bit 0..65535 gradient middle row: {chars!r}")
    print(f"  distinct chars in a full gradient: {len(set(chars))}")


def huge():
    Image.MAX_IMAGE_PIXELS = Image.MAX_IMAGE_PIXELS  # library default
    for w, h in [(9000, 6000), (16000, 12000)]:
        rng = np.random.default_rng(0)
        base = Image.fromarray(rng.integers(0, 255, (h // 16, w // 16, 3), dtype=np.uint8)).resize((w, h))
        p = save(base, f"huge_{w}x{h}.jpg", quality=80)
        del base
        before = peak_rss_mb()
        t = time.perf_counter()
        try:
            rows = v1.convert(p, 120)
            print(f"  {w}x{h} ({w*h/1e6:.0f} MP): {time.perf_counter()-t:.2f}s, peak RSS {peak_rss_mb():.0f} MB "
                  f"(was {before:.0f} MB) -> {len(rows[0])}x{len(rows)}")
        except Exception as e:
            print(f"  {w}x{h} ({w*h/1e6:.0f} MP): CRASH {type(e).__name__}: {str(e)[:90]}")


def gamma():
    # 1px black/white checkerboard: physically 50% light => sRGB ~188 when averaged in linear light.
    board = (np.indices((400, 400)).sum(0) % 2 * 255).astype(np.uint8)
    rows = v1.convert(save(Image.fromarray(board).convert("RGB"), "checker.png"), 40)
    print(f"  v1 averaged color {rows[5][5][1]} (gamma-space); physically correct ~(188,188,188)")


def ramp_order():
    font = None
    for f in ["consola.ttf", "DejaVuSansMono.ttf", "Menlo.ttc"]:
        try:
            font = ImageFont.truetype(f, 64); break
        except OSError:
            pass
    if font is None:
        print("  (no monospace font found; skipped)"); return
    cov = []
    for ch in v1.RAMP:
        im = Image.new("L", (40, 80), 0)
        ImageDraw.Draw(im).text((2, 0), ch, fill=255, font=font)
        cov.append(np.asarray(im).mean() / 255)
    cov = np.array(cov)
    inversions = int(np.sum(np.diff(cov) > 0.002))  # ramp claims monotonically decreasing density
    print(f"  measured coverage '$'={cov[0]:.3f} ... ' '={cov[-1]:.3f}; darkest glyph actually "
          f"{v1.RAMP[int(cov.argmax())]!r}={cov.max():.3f}")
    print(f"  order inversions along the 70-char ramp: {inversions}")
    steps = -np.diff(cov)
    print(f"  step sizes: min {steps.min():+.4f}, max {steps.max():+.4f} (index mapping assumes all equal)")


def html_and_size(photo):
    rows = v1.convert(photo, 160)
    html = v1.to_html(rows)
    ansi = v1.to_ansi(rows)
    n = len(rows) * len(rows[0])
    bad = len(re.findall(r">[<>&]<", html))
    print(f"  cells={n}: ANSI {len(ansi.encode())/1e3:.0f} KB ({len(ansi.encode())/n:.1f} B/char), "
          f"HTML {len(html.encode())/1e3:.0f} KB ({len(html.encode())/n:.1f} B/char)")
    print(f"  unescaped '<', '>' or '&' spans in HTML: {bad}")
    print(f"  ANSI resets per line: {ansi.count(chr(27) + '[0m')} for {len(rows)} lines (color bleeds on wrap/resize)")
    t = time.perf_counter(); v1.to_html(v1.convert(photo, 400)); print(f"  width 400 end-to-end: {time.perf_counter()-t:.2f}s")


def muddiness(photo):
    rows = v1.convert(photo, 160)
    cols = np.array([c for r in rows for _, c in r], dtype=np.float32) / 255
    chars = [ch for r in rows for ch, _ in r]
    lin = np.where(cols <= 0.04045, cols / 12.92, ((cols + 0.055) / 1.055) ** 2.4)
    y = lin @ np.array([0.2126, 0.7152, 0.0722])
    mx, mn = cols.max(1), cols.min(1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    print(f"  ramp levels used: {len(set(chars))}/70; mean fg luminance Y={y.mean():.3f}; mean HSV saturation={sat.mean():.3f}")
    print("  on a dark page a glyph lights at most ~36% of its cell (@), so perceived cell brightness ~= coverage * Y")


def flat():
    rows = v1.convert(save(Image.new("RGB", (640, 400), (0, 120, 215)), "flat.png"), 40)
    print(f"  flat Windows-blue image -> chars {set(ch for r in rows for ch, _ in r)} (ok, but no contrast control at all)")


if __name__ == "__main__":
    photo = sys.argv[1] if len(sys.argv) > 1 else None
    probe("1. Aspect ratio (400x400 circle, width 80)", aspect)
    probe("2. Extreme panorama 4000x20", panorama)
    probe("3. EXIF orientation (phone photo)", exif_rotation)
    probe("4. Transparent PNG", transparency)
    probe("5. 16-bit grayscale PNG", sixteen_bit)
    probe("6. Gamma-incorrect downsampling", gamma)
    probe("7. Ramp density ordering vs. measured glyph coverage (Consolas)", ramp_order)
    probe("8. Flat-color image", flat)
    if photo:
        probe("9. Output size, escaping, speed (real photo)", lambda: html_and_size(photo))
        probe("10. Color muddiness (real photo)", lambda: muddiness(photo))
    probe("11. Huge images (memory / decompression bomb)", huge)
