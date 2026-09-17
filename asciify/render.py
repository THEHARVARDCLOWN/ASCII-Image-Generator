"""Frame renderers: plain text, ANSI escape codes, standalone HTML and raster images."""
from __future__ import annotations

import base64
import html
import io
import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import color as C
from .engine import Frame

RESET = "\x1b[0m"
COLOR_DEPTHS = ("truecolor", "256", "16", "none")
IMAGE_EXTS = (".png", ".webp", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff")
_FONT_CANDIDATES = ("consola.ttf", "CascadiaMono.ttf", "DejaVuSansMono.ttf", "Menlo.ttc", "SFNSMono.ttf",
                    "LiberationMono-Regular.ttf", "NotoSansMono-Regular.ttf", "cour.ttf")
# Cached fonts are shared across web-server threads; a FreeType face is not safe to rasterize concurrently.
_FONT_LOCK = threading.Lock()


# --------------------------------------------------------------------------- text / ANSI
def to_text(frame: Frame) -> str:
    return "\n".join(frame.chars)


def quantize6(rgb: np.ndarray) -> np.ndarray:
    """Round to 6 bits per channel (error <= 2/255, invisible) so neighboring cells share codes."""
    return np.minimum((rgb.astype(np.uint16) + 2) >> 2 << 2, 255).astype(np.uint8)


def _codes(rgb: np.ndarray, depth: str, background: bool) -> list:
    """Per-cell SGR parameter strings, computed once per unique color."""
    flat = rgb.reshape(-1, 3)
    keys = (flat[:, 0].astype(np.int32) << 16) | (flat[:, 1].astype(np.int32) << 8) | flat[:, 2]
    uniq, inverse = np.unique(keys, return_inverse=True)
    urgb = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(np.uint8)
    if depth == "truecolor":
        prefix = "48;2;" if background else "38;2;"
        strs = [f"{prefix}{r};{g};{b}" for r, g, b in urgb.tolist()]
    elif depth == "256":
        prefix = "48;5;" if background else "38;5;"
        strs = [f"{prefix}{i}" for i in C.to_ansi256(urgb).tolist()]
    else:
        base, bright = (40, 100) if background else (30, 90)
        strs = [str(base + i if i < 8 else bright + i - 8) for i in C.to_ansi16(urgb).tolist()]
    return np.array(strs, dtype=object)[inverse.reshape(-1)].reshape(rgb.shape[:2]).tolist()


def to_ansi(frame: Frame, depth: str = "truecolor", fill_background: bool = False) -> str:
    """Escape codes are emitted only when the color changes, and every line ends with a reset so
    colors never bleed into wrapped lines or the shell prompt."""
    if depth not in COLOR_DEPTHS:
        raise ValueError(f"unknown color depth {depth!r}")
    if depth == "none" or (not frame.colored and not fill_background):
        return to_text(frame)
    lines = []
    prep = quantize6 if depth == "truecolor" else (lambda a: a)
    if frame.mode == "blocks":
        top, bottom = prep(frame.fg), prep(frame.lower)
        upper, lower = _codes(top, depth, False), _codes(bottom, depth, True)
        same = (top == bottom).all(axis=-1).tolist()
        for r in range(frame.rows):
            parts, prev = [], None
            for c in range(frame.cols):
                seq = lower[r][c] if same[r][c] else f"{upper[r][c]};{lower[r][c]}"
                if seq != prev:
                    parts.append(f"\x1b[{seq}m")
                    prev = seq
                parts.append(" " if same[r][c] else "▀")
            parts.append(RESET)
            lines.append("".join(parts))
        return "\n".join(lines)

    ink = _codes(prep(frame.fg), depth, False) if frame.colored else None
    page = _codes(np.array([[frame.background]], np.uint8), depth, True)[0][0] if fill_background else None
    for r, row in enumerate(frame.chars):
        parts, prev = [], None
        if page:
            parts.append(f"\x1b[{page}m")
            if not frame.colored:
                parts.append(f"\x1b[{_codes(np.array([[frame.foreground]], np.uint8), depth, False)[0][0]}m")
        for c, ch in enumerate(row):
            if ink is not None and ch != " " and ink[r][c] != prev:  # spaces show no ink: skip codes
                prev = ink[r][c]
                parts.append(f"\x1b[{prev}m")
            parts.append(ch)
        if prev is not None or page:
            parts.append(RESET)
        lines.append("".join(parts))
    return "\n".join(lines)


# --------------------------------------------------------------------------- HTML
def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="generator" content="asciify">
<title>{title}</title>
<style>
:root{{color-scheme:{scheme}}}
html,body{{margin:0;background:{bg}}}
body{{min-height:100vh;display:flex;align-items:center;justify-content:center;padding:16px;box-sizing:border-box}}
{css}
</style></head><body>
{body}
</body></html>
"""


def to_html(frame: Frame, title: str = "ASCII art") -> str:
    """Standalone page. The font size fits the viewport width, and the line-height is set in `ch`
    units so each cell is exactly cell_ratio wide:tall in any monospace font."""
    bg = frame.background
    scheme = "light" if (0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]) > 128 else "dark"
    safe_title = html.escape(title)
    if frame.mode == "blocks":
        pix = np.empty((frame.rows * 2, frame.cols, 3), np.uint8)
        pix[0::2], pix[1::2] = frame.fg, frame.lower
        buf = io.BytesIO()
        Image.fromarray(pix).save(buf, "PNG", optimize=True)
        css = ("img{display:block;image-rendering:pixelated;height:auto;"
               f"width:min(100%,{frame.cols * 8}px);aspect-ratio:{frame.cols}/{frame.rows / frame.cell_ratio:.3f}}}")
        body = (f'<img alt="{safe_title}" src="data:image/png;base64,'
                f'{base64.b64encode(buf.getvalue()).decode("ascii")}">')
        return _PAGE.format(title=safe_title, scheme=scheme, bg=_hex(bg), css=css, body=body)

    pre_css = ('pre{margin:0;overflow:auto;font-family:ui-monospace,"Cascadia Mono",Consolas,Menlo,'
               '"DejaVu Sans Mono","Liberation Mono",monospace;'
               f"font-size:min(14px,calc((100vw - 32px) / {frame.cols * 0.6:.1f}));"
               f"line-height:calc(1ch / {frame.cell_ratio:g});color:{_hex(frame.foreground)};"
               "font-kerning:none;font-variant-ligatures:none;letter-spacing:0}"
               "pre i{font-style:normal}")
    rules, lines = [], []
    if frame.colored:
        grid = np.array([list(r) for r in frame.chars])
        inked = grid != " "
        q = quantize6(frame.fg)
        keys = (q[..., 0].astype(np.int32) << 16) | (q[..., 1].astype(np.int32) << 8) | q[..., 2]
        uniq, inverse = np.unique(keys[inked], return_inverse=True)
        cls = np.full(keys.shape, -1, np.int64)
        cls[inked] = inverse.reshape(-1)
        rules = [f".c{np.base_repr(i, 36).lower()}{{color:#{int(k):06x}}}" for i, k in enumerate(uniq.tolist())]
        names = [f"c{np.base_repr(i, 36).lower()}" for i in range(len(uniq))]
        for r, row in enumerate(frame.chars):
            parts, cur = [], -1
            ids = cls[r].tolist()
            for c, ch in enumerate(row):
                if ch != " " and ids[c] != cur:
                    if cur >= 0:
                        parts.append("</i>")
                    cur = ids[c]
                    parts.append(f"<i class={names[cur]}>")
                parts.append(html.escape(ch, quote=False))
            if cur >= 0:
                parts.append("</i>")
            lines.append("".join(parts))
    else:
        lines = [html.escape(r, quote=False) for r in frame.chars]
    body = f'<pre role="img" aria-label="{safe_title}">' + "\n".join(lines) + "</pre>"
    return _PAGE.format(title=safe_title, scheme=scheme, bg=_hex(bg), css=pre_css + "\n" + "\n".join(rules),
                        body=body)


# --------------------------------------------------------------------------- raster
@lru_cache(maxsize=8)
def load_font(path: Optional[str], size: int):
    """Return (font, is_monospace). An explicit path must exist; otherwise try common monospace fonts."""
    if path:
        try:
            return ImageFont.truetype(path, size), True
        except OSError:
            raise ValueError(f"cannot load font: {path}") from None
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size), True
        except OSError:
            continue
    try:
        return ImageFont.load_default(size), False
    except TypeError:  # very old Pillow without FreeType sizing
        return ImageFont.load_default(), False


def to_image(frame: Frame, font_path: Optional[str] = None, font_size: int = 16,
             max_output_pixels: int = 60_000_000) -> Image.Image:
    """Rasterize with a glyph atlas: each distinct character is drawn once, then composited per row."""
    if frame.mode == "blocks":
        pix = np.empty((frame.rows * 2, frame.cols, 3), np.uint8)
        pix[0::2], pix[1::2] = frame.fg, frame.lower
        px_w = max(1, font_size // 2)
        px_h = max(1, round(px_w / (2 * frame.cell_ratio)))
        return Image.fromarray(pix).resize((frame.cols * px_w, frame.rows * 2 * px_h), Image.Resampling.NEAREST)

    font, mono = load_font(font_path, font_size)
    cw = max(1, round(font.getlength("M")))
    ch = max(1, round(cw / frame.cell_ratio))
    if frame.rows * ch * frame.cols * cw > max_output_pixels and font_size > 4:
        shrink = (max_output_pixels / (frame.rows * ch * frame.cols * cw)) ** 0.5
        return to_image(frame, font_path, max(4, int(font_size * shrink)), max_output_pixels)
    ascent, descent = font.getmetrics() if hasattr(font, "getmetrics") else (font_size, 0)
    y0 = (ch - (ascent + descent)) // 2

    uniq = sorted(set("".join(frame.chars)))
    lookup = {c: i for i, c in enumerate(uniq)}
    atlas = np.zeros((len(uniq), ch, cw), np.float32)
    with _FONT_LOCK:
        for i, c in enumerate(uniq):
            if c == " ":
                continue
            glyph = Image.new("L", (cw, ch), 0)
            x0 = 0 if mono else (cw - font.getlength(c)) / 2
            ImageDraw.Draw(glyph).text((x0, y0), c, fill=255, font=font)
            atlas[i] = np.asarray(glyph, np.float32) / 255.0

    bg = np.array(frame.background, np.float32)
    fg = frame.fg.astype(np.float32) if frame.colored else np.broadcast_to(
        np.array(frame.foreground, np.float32), frame.fg.shape)
    out = np.empty((frame.rows * ch, frame.cols * cw, 3), np.uint8)
    for r, row in enumerate(frame.chars):
        mask = atlas[[lookup[c] for c in row]][..., None]              # (cols, ch, cw, 1)
        cells = bg + mask * (fg[r][:, None, None, :] - bg)             # (cols, ch, cw, 3)
        out[r * ch:(r + 1) * ch] = (cells.transpose(1, 0, 2, 3).reshape(ch, frame.cols * cw, 3) + 0.5
                                    ).astype(np.uint8)
    return Image.fromarray(out)


# --------------------------------------------------------------------------- files
def save(frame: Frame, path, depth: str = "truecolor", font_path: Optional[str] = None,
         font_size: int = 16, title: Optional[str] = None) -> None:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".html", ".htm"):
        path.write_text(to_html(frame, title or path.stem), encoding="utf-8")
    elif ext in IMAGE_EXTS:
        to_image(frame, font_path, font_size).save(path)
    elif ext in (".ans", ".ansi"):
        path.write_text(to_ansi(frame, depth) + RESET + "\n", encoding="utf-8")
    elif ext in (".txt", ""):
        path.write_text(to_text(frame) + "\n", encoding="utf-8")
    else:
        raise ValueError(f"unsupported output type {ext!r}: use .txt, .ans, .html or an image extension")
