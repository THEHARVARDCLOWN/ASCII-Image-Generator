"""Core conversion: image -> Frame (a grid of glyphs plus per-cell ink colors)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from . import color as C
from . import glyphs, loader

MODES = ("ascii", "edges", "blocks")
MAX_COLS = 1000
MAX_ROWS = 1000


@dataclass
class Options:
    width: Optional[int] = None         # columns; None = derive from height / max_width
    height: Optional[int] = None        # rows; None = derive from width and the aspect ratio
    max_width: Optional[int] = None     # fit-inside box used when width/height are not given
    max_height: Optional[int] = None
    mode: str = "ascii"                 # ascii | edges | blocks
    ramp: str = "detailed"              # ramp name or a literal string of characters
    palette: str = "vivid"
    light: bool = False                 # render for a light page: dense glyphs mean dark
    background: Optional[C.RGB] = None  # None = palette/theme default
    contrast: float = 1.0               # 0..1 strength of the 1st-99th percentile tone stretch
    gamma: Optional[float] = None       # >1 darkens mid-tones; None = auto (see effective_gamma)
    saturation: float = 1.25            # Oklab chroma multiplier (vivid palette / blocks)
    lift: float = 0.5                   # 0..1 how far ink lightness is pulled toward readable
    dither: bool = False                # Floyd-Steinberg error diffusion across glyph levels
    edge_threshold: float = 0.6         # edges mode: gradient strength (tone per cell width)
    cell_ratio: float = 0.5             # character cell width / height of the target display
    max_pixels: Optional[int] = loader.DEFAULT_MAX_PIXELS


@dataclass
class Frame:
    chars: List[str]           # one string per row (blocks mode: shade-glyph fallback for plain text)
    fg: np.ndarray             # (rows, cols, 3) uint8 ink colors; blocks mode: upper half-cell color
    lower: Optional[np.ndarray]  # blocks mode only: (rows, cols, 3) lower half-cell color
    background: C.RGB
    foreground: C.RGB          # text color used when colored is False
    mode: str
    colored: bool
    cell_ratio: float
    source_size: Tuple[int, int]

    @property
    def rows(self) -> int:
        return len(self.chars)

    @property
    def cols(self) -> int:
        return len(self.chars[0]) if self.chars else 0


def grid_size(img_w: int, img_h: int, width: Optional[int] = None, height: Optional[int] = None,
              max_width: Optional[int] = None, max_height: Optional[int] = None,
              cell_ratio: float = 0.5) -> Tuple[int, int]:
    """Columns and rows that preserve the image's shape on cells that are cell_ratio as wide as tall.

    rows = cols * (img_h / img_w) * cell_ratio. A square image at 80 columns on 1:2 cells is
    therefore 40 rows, which displays square.
    """
    rows_per_col = img_h / img_w * cell_ratio
    if width and height:
        cols, rows = float(width), float(height)
    elif width:
        cols, rows = float(width), width * rows_per_col
    elif height:
        cols, rows = height / rows_per_col, float(height)
    else:
        cols = float(max_width or 100)
        rows = cols * rows_per_col
        if max_height and rows > max_height:
            rows, cols = float(max_height), max_height / rows_per_col
    if not (width and height):  # keep aspect when clamping to hard limits
        scale = min(1.0, MAX_COLS / cols, MAX_ROWS / rows)
        cols, rows = cols * scale, rows * scale
    return (int(min(MAX_COLS, max(1, round(cols)))), int(min(MAX_ROWS, max(1, round(rows)))))


def effective_gamma(o: Options) -> float:
    """Tone gamma: 1.5 for glyphs on dark pages, 1.0 otherwise.

    Light glyphs on a dark page look bolder than their ink coverage (irradiation), so a straight
    mapping washes bright regions out into walls of '@'. Tuned by eye on photographs; see docs/REVIEW.md.
    """
    if o.gamma is not None:
        return o.gamma
    return 1.0 if (o.light or o.mode == "blocks") else 1.5


def _block_mean(a: np.ndarray, sy: int, sx: int) -> np.ndarray:
    if sy == 1 and sx == 1:
        return a
    h, w = a.shape[0] // sy, a.shape[1] // sx
    return a[:h * sy, :w * sx].reshape(h, sy, w, sx, *a.shape[2:]).mean(axis=(1, 3))


def _tone_mapper(lightness: np.ndarray, opaque: Optional[np.ndarray], contrast: float, gamma: float):
    """Robust auto-levels: stretch the 1st-99th percentile of opaque lightness to [0, 1]."""
    values = lightness[opaque] if opaque is not None and opaque.sum() >= 16 else lightness.ravel()
    lo, hi = (np.percentile(values, [1, 99]) if values.size else (0.0, 1.0))
    stretch = contrast > 0 and (hi - lo) > 0.04  # flat images: leave tones alone

    def apply(L: np.ndarray) -> np.ndarray:
        t = L
        if stretch:
            t = L + (np.clip((L - lo) / (hi - lo), 0.0, 1.0) - L) * contrast
        t = np.clip(t, 0.0, 1.0)
        if gamma != 1.0:
            t = t ** gamma
        return t.astype(np.float32)

    return apply


def _edges(tone: np.ndarray, sx: int, sy: int, cell_ratio: float, threshold: float):
    """Directional glyphs from a per-cell structure tensor of the Sobel gradient.

    Gradients are measured in display units (tone change per cell width), so orientation is right
    even though subpixels are sampled on a non-square character grid.
    """
    p = np.pad(tone, 1, mode="edge")
    gx = ((p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])) * (sx / 8)
    gy = ((p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])) * (
        sy * cell_ratio / 8)
    jxx, jyy, jxy = (_block_mean(g, sy, sx) for g in (gx * gx, gy * gy, gx * gy))
    energy = jxx + jyy
    magnitude = np.sqrt(energy)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / np.maximum(energy, 1e-12)
    # Dominant gradient angle (image y points down), rotated 90 degrees to follow the edge itself.
    theta = (np.degrees(0.5 * np.arctan2(2 * jxy, jxx - jyy)) + 90.0) % 180.0
    g = math.degrees(math.atan(1 / cell_ratio))  # on-screen angle of '\' in a cell
    b1, b2, b3, b4 = g / 2, (g + 90) / 2, (270 - g) / 2, 180 - g / 2
    chars = np.full(theta.shape, "-", dtype="<U1")
    chars[(theta >= b1) & (theta < b2)] = "\\"
    chars[(theta >= b2) & (theta < b3)] = "|"
    chars[(theta >= b3) & (theta < b4)] = "/"
    return chars, (magnitude > threshold) & (coherence > 0.5)


def _validate(o: Options) -> C.Palette:
    if o.mode not in MODES:
        raise ValueError(f"unknown mode {o.mode!r}; choose from {', '.join(MODES)}")
    if o.palette not in C.PALETTES:
        raise ValueError(f"unknown palette {o.palette!r}; choose from {', '.join(C.PALETTE_NAMES)}")
    for name, lo, hi in (("contrast", 0, 1), ("saturation", 0, 4), ("lift", 0, 1),
                         ("edge_threshold", 0, 20), ("cell_ratio", 0.1, 2)):
        value = getattr(o, name)
        if not (lo <= value <= hi):
            raise ValueError(f"{name} must be between {lo} and {hi} (got {value})")
    if o.gamma is not None and not (0.1 <= o.gamma <= 10):
        raise ValueError(f"gamma must be between 0.1 and 10 (got {o.gamma})")
    for name in ("width", "height", "max_width", "max_height"):
        value = getattr(o, name)
        if value is not None and value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return C.PALETTES[o.palette]


def convert(source, options: Optional[Options] = None) -> Frame:
    """Convert an image (path, bytes or file object) into a Frame."""
    o = options or Options()
    palette = _validate(o)
    ramp = glyphs.get_ramp("blocks" if o.mode == "blocks" else o.ramp)

    with loader.pixel_limit(o.max_pixels):
        im = loader.open_image(source, o.max_pixels)
        try:
            src_w, src_h = loader.oriented_size(im)
            cols, rows = grid_size(src_w, src_h, o.width, o.height, o.max_width, o.max_height, o.cell_ratio)
            if o.mode == "blocks":
                sx, sy = 1, 2                                  # two square pixels per 1:2 cell
            elif o.mode == "edges":
                sx = 3
                sy = max(1, round(sx / o.cell_ratio))         # ~square subpixels for gradients
            else:
                sx = sy = 1                                    # BOX resample = exact per-cell area mean
            need_w, need_h = cols * sx, rows * sy
            lin, alpha = loader.load_linear(im, need_w, need_h)
        finally:
            im.close()

    lin = loader.resample(lin, need_w, need_h)
    alpha = None if alpha is None else np.clip(loader.resample(alpha, need_w, need_h), 0.0, 1.0)

    if o.background is not None:
        background = tuple(o.background)
    elif o.light:
        background = C.LIGHT_BG
    else:
        background = palette.background or C.DARK_BG
    foreground = C.LIGHT_FG if o.light else C.DARK_FG
    bg_lin = C.srgb_to_linear(np.array(background, np.float32) / 255.0)

    blank = 1.0 if o.light else 0.0  # tone that produces no ink on this page

    if o.mode == "blocks":  # every half-cell is a solid pixel
        color_lin = _unpremultiply(lin, alpha, bg_lin)
        lightness = C.linear_to_oklab(color_lin)[..., 0]
        opaque = None if alpha is None else alpha > 0.5
        tone = _tone_mapper(lightness, opaque, o.contrast, effective_gamma(o))(lightness)
        pix = C.pixel_colors(palette, color_lin, tone, saturation=o.saturation)
        if alpha is not None:  # composite over the page in linear light
            a = alpha[..., None]
            pix = C.linear_to_srgb8(C.SRGB_LUT[pix] * a + bg_lin * (1.0 - a))
        shade_tone = tone if alpha is None else tone * alpha
        shade = ramp.index((shade_tone[0::2] + shade_tone[1::2]) / 2)
        table = np.array(list(ramp.chars))
        return Frame(chars=["".join(r) for r in table[shade]], fg=pix[0::2], lower=pix[1::2],
                     background=background, foreground=foreground, mode=o.mode, colored=True,
                     cell_ratio=o.cell_ratio, source_size=(src_w, src_h))

    cell_lin = _block_mean(lin, sy, sx)
    cell_alpha = None if alpha is None else _block_mean(alpha, sy, sx)
    color_lin = _unpremultiply(cell_lin, cell_alpha, bg_lin)
    lightness = C.linear_to_oklab(color_lin)[..., 0]
    mapper = _tone_mapper(lightness, None if cell_alpha is None else cell_alpha > 0.5, o.contrast,
                          effective_gamma(o))
    # Alpha behaves like ink coverage: a half-transparent cell gets half the glyph density, and a
    # fully transparent one is always blank, whether or not the tone stretch kicked in.
    tone = _with_alpha(mapper(lightness), cell_alpha, blank)
    target = 1.0 - tone if o.light else tone

    idx = ramp.dither(target) if o.dither else ramp.index(target)
    grid = np.array(list(ramp.chars))[idx]
    if o.mode == "edges":
        sub_tone = _with_alpha(mapper(C.linear_to_oklab(_unpremultiply(lin, alpha, bg_lin))[..., 0]), alpha, blank)
        edge_chars, mask = _edges(sub_tone, sx, sy, o.cell_ratio, o.edge_threshold)
        grid = np.where(mask, edge_chars, grid)

    colored = palette.name != "none"
    if colored:
        fg = C.ink_colors(palette, color_lin, tone, light=o.light, saturation=o.saturation, lift=o.lift)
    else:
        fg = np.broadcast_to(np.array(foreground, np.uint8), (rows, cols, 3)).copy()
    return Frame(chars=["".join(r) for r in grid], fg=fg, lower=None, background=background,
                 foreground=foreground, mode=o.mode, colored=colored, cell_ratio=o.cell_ratio,
                 source_size=(src_w, src_h))


def _unpremultiply(lin: np.ndarray, alpha: Optional[np.ndarray], fallback: np.ndarray) -> np.ndarray:
    if alpha is None:
        return np.clip(lin, 0.0, 1.0)
    a = alpha[..., None]
    return np.clip(np.where(a > 1e-3, lin / np.maximum(a, 1e-3), fallback), 0.0, 1.0)


def _with_alpha(tone: np.ndarray, alpha: Optional[np.ndarray], blank: float) -> np.ndarray:
    return tone if alpha is None else (tone * alpha + blank * (1.0 - alpha)).astype(np.float32)
