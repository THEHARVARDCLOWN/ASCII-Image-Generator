"""Color science: sRGB transfer curves, Oklab, gamut mapping, palettes and ANSI quantization.

All image math happens in *linear light* (averaging) or *Oklab* (perceptual edits). Gamma-encoded
sRGB only appears at the edges: when decoding pixels and when emitting final colors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

RGB = Tuple[int, int, int]


# --------------------------------------------------------------------------- transfer curves
def srgb_to_linear(x) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055).astype(np.float32)


#: uint8 sRGB code value -> linear light, for fast decoding via fancy indexing.
SRGB_LUT = srgb_to_linear(np.arange(256, dtype=np.float32) / 255.0)


def linear_to_srgb8(x) -> np.ndarray:
    return (linear_to_srgb(x) * 255.0 + 0.5).astype(np.uint8)


def hex_to_rgb(value: str) -> RGB:
    s = value.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6 or any(c not in "0123456789abcdefABCDEF" for c in s):
        raise ValueError(f"invalid color {value!r}; expected #rrggbb")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


# --------------------------------------------------------------------------- Oklab (Ottosson 2020)
_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]], dtype=np.float32)
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]], dtype=np.float32)
_M2_INV = np.array([[1.0, 0.3963377774, 0.2158037573],
                    [1.0, -0.1055613458, -0.0638541728],
                    [1.0, -0.0894841775, -1.2914855480]], dtype=np.float32)
_M1_INV = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                    [-1.2684380046, 2.6097574011, -0.3413193965],
                    [-0.0041960863, -0.7034186147, 1.7076147010]], dtype=np.float32)


def linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    return np.cbrt(rgb @ _M1.T) @ _M2.T


def oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    return ((lab @ _M2_INV.T) ** 3) @ _M1_INV.T


def gamut_map(lab: np.ndarray, iterations: int = 10) -> np.ndarray:
    """Oklab -> linear sRGB, pulling out-of-gamut colors toward gray at constant lightness and hue.

    Naive per-channel clipping shifts hue (saturated blues turn purple); shrinking chroma does not.
    """
    lab = np.array(lab, dtype=np.float32, copy=True)
    lab[..., 0] = np.clip(lab[..., 0], 0.0, 1.0)
    rgb = oklab_to_linear(lab)
    eps = 1e-4
    bad = ((rgb < -eps) | (rgb > 1 + eps)).any(axis=-1)
    if bad.any():
        sub = lab[bad]
        lo = np.zeros(len(sub), np.float32)
        hi = np.ones(len(sub), np.float32)
        for _ in range(iterations):
            mid = (lo + hi) / 2
            test = sub.copy()
            test[:, 1:] *= mid[:, None]
            r = oklab_to_linear(test)
            ok = ((r >= -eps) & (r <= 1 + eps)).all(axis=-1)
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        sub[:, 1:] *= lo[:, None]
        rgb[bad] = oklab_to_linear(sub)
    return np.clip(rgb, 0.0, 1.0)


# --------------------------------------------------------------------------- palettes
@dataclass(frozen=True)
class Palette:
    name: str
    description: str
    stops: Tuple[str, ...] = ()        # gradient stops, dark -> light; empty = derived from the image
    background: Optional[RGB] = None  # preferred page background in dark mode

    def gradient(self, t: np.ndarray) -> np.ndarray:
        """Map tone t in [0,1] to linear RGB by interpolating the stops in Oklab."""
        stops = linear_to_oklab(srgb_to_linear(np.array([hex_to_rgb(s) for s in self.stops]) / 255.0))
        pos = np.linspace(0.0, 1.0, len(stops))
        flat = np.clip(t, 0, 1).ravel()
        lab = np.stack([np.interp(flat, pos, stops[:, i]) for i in range(3)], axis=-1).astype(np.float32)
        return gamut_map(lab).reshape(*np.shape(t), 3)


PALETTES = {p.name: p for p in [
    Palette("vivid", "Image colors with ink compensation: lightness lifted and chroma boosted in Oklab (default)"),
    Palette("true", "Exact mean color of each cell: faithful, but thin glyphs make it look dim"),
    Palette("none", "No color, plain foreground text"),
    Palette("mono", "Grayscale shading", ("#303030", "#8c8c8c", "#f4f4f4")),
    Palette("matrix", "Green phosphor terminal", ("#003b12", "#008f33", "#2bff6a", "#caffd6"), (1, 8, 3)),
    Palette("amber", "Amber CRT monitor", ("#4a2300", "#a65a00", "#ffae1a", "#ffe9b8"), (12, 7, 0)),
    Palette("synthwave", "Retro sunset: violet, magenta, orange, gold",
            ("#3b1370", "#9b1fa8", "#ff2a85", "#ff8a3d", "#ffe66d"), (13, 4, 27)),
    Palette("inferno", "Perceptually uniform heat map",
            ("#1b0c41", "#4a0c6b", "#932667", "#dd513a", "#fca50a", "#fcffa4"), (0, 0, 4)),
    Palette("ice", "Glacial blues", ("#0b2a5b", "#1667a8", "#3fb6e8", "#e8fbff"), (2, 8, 20)),
    Palette("sepia", "Faded photograph", ("#4a3220", "#8c6443", "#d2ad85", "#fff3dc"), (18, 12, 7)),
]}
PALETTE_NAMES = tuple(PALETTES)

DARK_BG: RGB = (12, 12, 14)
LIGHT_BG: RGB = (250, 249, 245)
DARK_FG: RGB = (214, 214, 214)
LIGHT_FG: RGB = (28, 28, 28)


def ink_colors(palette: Palette, color_lin: np.ndarray, tone: np.ndarray, *, light: bool,
               saturation: float, lift: float) -> np.ndarray:
    """Choose the glyph (ink) color for each cell, returned as uint8 sRGB.

    The glyph itself already encodes brightness through its ink coverage, so reusing the cell's
    mean color darkens twice: perceived brightness ~= coverage x color. "vivid" gives brightness to
    the glyph and hue to the color. It pulls lightness toward a readable target and boosts chroma.
    """
    if palette.stops:
        return linear_to_srgb8(palette.gradient(tone))
    if palette.name == "true":
        return linear_to_srgb8(color_lin)
    lab = linear_to_oklab(color_lin)
    target = 0.40 if light else 0.90
    lab[..., 0] += (target - lab[..., 0]) * lift
    lab[..., 1:] *= saturation
    return linear_to_srgb8(gamut_map(lab))


def pixel_colors(palette: Palette, comp_lin: np.ndarray, tone: np.ndarray, *, saturation: float) -> np.ndarray:
    """Colors for block mode, where each half-cell is a solid pixel: tone drives lightness directly."""
    if palette.stops:
        return linear_to_srgb8(palette.gradient(tone))
    if palette.name == "true":
        return linear_to_srgb8(comp_lin)
    lab = linear_to_oklab(comp_lin)
    lab[..., 0] = tone
    lab[..., 1:] *= 0.0 if palette.name == "none" else saturation
    return linear_to_srgb8(gamut_map(lab))


# --------------------------------------------------------------------------- ANSI quantization
XTERM16 = np.array([(0, 0, 0), (205, 0, 0), (0, 205, 0), (205, 205, 0), (0, 0, 238), (205, 0, 205),
                    (0, 205, 205), (229, 229, 229), (127, 127, 127), (255, 0, 0), (0, 255, 0),
                    (255, 255, 0), (92, 92, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255)], np.uint8)
_CUBE = (0, 95, 135, 175, 215, 255)
XTERM256 = np.concatenate([
    XTERM16,
    np.array([(r, g, b) for r in _CUBE for g in _CUBE for b in _CUBE], np.uint8),
    np.array([(v, v, v) for v in range(8, 248, 10)], np.uint8),
])


def nearest_index(rgb: np.ndarray, table: np.ndarray, start: int = 0) -> np.ndarray:
    """Perceptually nearest (Oklab distance) palette index for each uint8 RGB row."""
    lab = linear_to_oklab(SRGB_LUT[rgb])
    tlab = linear_to_oklab(SRGB_LUT[table[start:]])
    out = np.empty(len(rgb), np.int32)
    for i in range(0, len(rgb), 2048):
        d = ((lab[i:i + 2048, None, :] - tlab[None, :, :]) ** 2).sum(axis=-1)
        out[i:i + 2048] = d.argmin(axis=1) + start
    return out


def to_ansi256(rgb: np.ndarray) -> np.ndarray:
    # Skip 0-15: terminal themes redefine them, so only the 6x6x6 cube and gray ramp are predictable.
    return nearest_index(rgb, XTERM256, start=16)


def to_ansi16(rgb: np.ndarray) -> np.ndarray:
    return nearest_index(rgb, XTERM16)
