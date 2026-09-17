"""Character ramps ordered by *measured* ink coverage rather than by folklore.

Classic ramps map luminance to string index, which assumes each character is exactly one equal
density step darker than the last. Measured in Consolas, Paul Bourke's 70-character ramp has 30
order inversions and step sizes from -0.07 to +0.11. Here each glyph's real coverage decides which
tone it represents. Regenerate the table for another font with tools/measure_glyphs.py.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

# Fraction of the character cell that is inked, measured from consola.ttf at 96px (cell 53x97).
COVERAGE = {
    ' ': 0.0000, '`': 0.0214, '.': 0.0367, "'": 0.0402, '-': 0.0423, ':': 0.0630,
    ',': 0.0682, '_': 0.0693, '"': 0.0772, '^': 0.0814, '~': 0.0865, ';': 0.0997,
    '>': 0.1037, '<': 0.1041, '!': 0.1042, '=': 0.1098, '*': 0.1137, '\\': 0.1151,
    '/': 0.1151, '+': 0.1230, 'r': 0.1309, '?': 0.1313, '|': 0.1361, 'c': 0.1371,
    'L': 0.1384, ')': 0.1462, '(': 0.1464, 'v': 0.1508, 'T': 0.1514, '7': 0.1524,
    'i': 0.1575, 's': 0.1590, 'z': 0.1596, 'J': 0.1610, 'l': 0.1630, '}': 0.1659,
    '{': 0.1660, 't': 0.1665, 'x': 0.1680, 'Y': 0.1693, '[': 0.1706, 'F': 0.1715,
    ']': 0.1719, '1': 0.1749, 'n': 0.1752, 'u': 0.1753, 'f': 0.1766, 'C': 0.1782,
    'I': 0.1784, '3': 0.1829, 'j': 0.1882, 'o': 0.1898, '2': 0.1910, '5': 0.1919,
    'e': 0.1946, 'a': 0.1972, 'S': 0.2001, 'y': 0.2005, 'k': 0.2008, 'V': 0.2022,
    'h': 0.2062, 'E': 0.2071, 'Z': 0.2078, 'P': 0.2081, 'w': 0.2127, 'K': 0.2141,
    'X': 0.2171, 'U': 0.2173, '4': 0.2175, '9': 0.2242, '6': 0.2246, 'd': 0.2281,
    'b': 0.2281, 'p': 0.2287, 'q': 0.2294, 'm': 0.2315, 'A': 0.2340, 'H': 0.2344,
    'G': 0.2371, '#': 0.2399, 'R': 0.2417, 'O': 0.2431, 'D': 0.2487, '8': 0.2622,
    '%': 0.2624, 'W': 0.2676, 'B': 0.2711, 'M': 0.2712, 'N': 0.2715, '$': 0.2739,
    '0': 0.2756, 'Q': 0.2923, 'g': 0.2952, '&': 0.3011, '@': 0.3630,
}
# Shade blocks are drawn procedurally by most terminals, so their nominal coverage is reliable.
BLOCK_COVERAGE = {' ': 0.0, '░': 0.25, '▒': 0.5, '▓': 0.75, '█': 1.0}

RAMPS = {
    # Evenly spaced in measured coverage and built from shape-neutral glyphs. Directional glyphs
    # (/ \ | - _) would streak flat areas and are reserved for edge mode.
    "detailed": " .:;=+cvxoawdO8MQ&@",
    # Paul Bourke's classic 70-level ramp, used here by measured coverage instead of string order.
    "standard": " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
    "simple": " .:-=+*#%@",
    "blocks": " ░▒▓█",
}
EDGE_CHARS = "-\\|/"


class Ramp:
    """An ordered set of glyphs with normalized density levels in [0, 1]."""

    def __init__(self, spec: str):
        chars = RAMPS.get(spec, spec)
        chars = "".join(dict.fromkeys(chars.replace("\n", "").replace("\t", "")))  # dedupe, keep order
        if len(chars) < 2:
            raise ValueError("a ramp needs at least 2 distinct characters")
        table = {**COVERAGE, **BLOCK_COVERAGE}
        if all(c in table for c in chars):
            order = sorted(chars, key=lambda c: table[c])
            levels = np.array([table[c] for c in order], np.float32)
        else:  # unknown glyphs: trust the user's order (sparse -> dense), assume even steps
            order = list(chars)
            levels = np.linspace(0.0, 1.0, len(order), dtype=np.float32)
        span = float(levels[-1] - levels[0])
        if span <= 0:
            raise ValueError("ramp characters all have the same density")
        self.chars = "".join(order)
        self.levels = (levels - levels[0]) / span
        # Nearest-level lookup table; ties resolve to the sparser glyph.
        grid = np.linspace(0.0, 1.0, 1024, dtype=np.float32)
        self._lut = np.abs(grid[:, None] - self.levels[None, :]).argmin(axis=1)

    def index(self, target: np.ndarray) -> np.ndarray:
        return self._lut[np.clip(np.rint(target * 1023), 0, 1023).astype(np.intp)]

    def dither(self, target: np.ndarray) -> np.ndarray:
        """Serpentine Floyd-Steinberg error diffusion in *coverage* space.

        Uses each glyph's real level, not an idealized uniform step, as the quantization value.
        """
        levels = self.levels.tolist()
        mids = [(a + b) / 2 for a, b in zip(levels, levels[1:])]
        from bisect import bisect_left

        h, w = target.shape
        t = target.astype(np.float64).tolist()
        out = np.empty((h, w), np.intp)
        for y in range(h):
            row, nxt = t[y], (t[y + 1] if y + 1 < h else None)
            step = 1 if y % 2 == 0 else -1
            for x in (range(w) if step == 1 else range(w - 1, -1, -1)):
                i = bisect_left(mids, row[x])
                out[y, x] = i
                err = row[x] - levels[i]
                xn, xp = x + step, x - step
                if 0 <= xn < w:
                    row[xn] += err * 0.4375
                if nxt is not None:
                    nxt[x] += err * 0.3125
                    if 0 <= xp < w:
                        nxt[xp] += err * 0.1875
                    if 0 <= xn < w:
                        nxt[xn] += err * 0.0625
        return out


@lru_cache(maxsize=32)
def get_ramp(spec: str) -> Ramp:
    return Ramp(spec)
