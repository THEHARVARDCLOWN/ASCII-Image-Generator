"""Safe, memory-bounded image decoding into linear-light float arrays.

Pipeline: validate header size -> JPEG DCT-domain downscale (draft) -> normalize exotic modes ->
ICC to sRGB -> strip-wise *linear-light* box reduction -> EXIF orientation -> premultiplied alpha.
Peak memory tracks the decoded source, never a float copy of it.
"""
from __future__ import annotations

import io
import os
import threading
import warnings
from contextlib import contextmanager
from typing import Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

from .color import SRGB_LUT, srgb_to_linear

try:  # optional HEIC/HEIF support (iPhone photos): pip install pillow-heif
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
except ImportError:
    pass

DEFAULT_MAX_PIXELS = 200_000_000
_HIGH_BIT_MODES = ("I;16", "I;16L", "I;16B", "I;16N", "I", "F")
_STRIP_PIXELS = 4_000_000  # source pixels decoded into floats at a time

_LIMIT_LOCK = threading.Lock()
_active_limits: list = []
_pillow_default = None


class ImageLoadError(ValueError):
    """The input could not be read as an image (shown to users verbatim)."""


@contextmanager
def pixel_limit(max_pixels: Optional[int]):
    """Make Pillow's decompression-bomb checks follow *our* limit for the duration of a conversion.

    Pillow re-checks its global MAX_IMAGE_PIXELS during decode (GIF frames, TIFF tiles, crop), not
    just in Image.open, so swapping it only around open() is not enough. Thread-safe: overlapping
    conversions share the most permissive active limit and the original value is restored at the end.
    """
    global _pillow_default
    value = max_pixels or None
    with _LIMIT_LOCK:
        if not _active_limits:
            _pillow_default = Image.MAX_IMAGE_PIXELS
        _active_limits.append(value)
        Image.MAX_IMAGE_PIXELS = None if None in _active_limits else max(_active_limits)
    try:
        yield
    finally:
        with _LIMIT_LOCK:
            _active_limits.remove(value)
            if _active_limits:
                Image.MAX_IMAGE_PIXELS = None if None in _active_limits else max(_active_limits)
            else:
                Image.MAX_IMAGE_PIXELS = _pillow_default


def open_image(source, max_pixels: Optional[int] = DEFAULT_MAX_PIXELS) -> Image.Image:
    """Lazily open a path, bytes or file object. Only the header is read here."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        if not source:
            raise ImageLoadError("empty input")
        fp = io.BytesIO(source)
    elif isinstance(source, (str, os.PathLike)):
        if not os.path.isfile(source):
            raise ImageLoadError(f"file not found: {source}")
        fp = source
    else:
        fp = source
    try:  # the header size is checked below against max_pixels, with a clearer message than Pillow's
        with pixel_limit(None), warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            im = Image.open(fp)
    except UnidentifiedImageError:
        raise ImageLoadError("not a recognized image format") from None
    except OSError as e:
        raise ImageLoadError(f"cannot read image: {e}") from None
    w, h = im.size
    if w < 1 or h < 1:
        raise ImageLoadError("image has no pixels")
    if max_pixels and w * h > max_pixels:
        im.close()
        raise ImageLoadError(
            f"image is {w}x{h} ({w * h / 1e6:.0f} MP), above the {max_pixels / 1e6:.0f} MP safety limit "
            "(raise it with --max-pixels)")
    return im


def exif_orientation(im: Image.Image) -> int:
    try:
        value = int(im.getexif().get(0x0112, 1))
    except Exception:
        return 1
    return value if 1 <= value <= 8 else 1


def oriented_size(im: Image.Image) -> Tuple[int, int]:
    w, h = im.size
    return (h, w) if exif_orientation(im) in (5, 6, 7, 8) else (w, h)


def _apply_orientation(a: np.ndarray, orientation: int) -> np.ndarray:
    """Numpy equivalent of ImageOps.exif_transpose, applied after reduction (cheap)."""
    if orientation == 2:
        return a[:, ::-1]
    if orientation == 3:
        return a[::-1, ::-1]
    if orientation == 4:
        return a[::-1]
    if orientation == 5:
        return a.transpose(1, 0, 2)
    if orientation == 6:
        return np.rot90(a, k=-1)
    if orientation == 7:
        return a[::-1, ::-1].transpose(1, 0, 2)
    if orientation == 8:
        return np.rot90(a, k=1)
    return a


def _to_srgb(im: Image.Image) -> Image.Image:
    """Convert embedded ICC profiles (Display P3, Adobe RGB, ...) to sRGB, or wide-gamut photos look dull."""
    icc = im.info.get("icc_profile")
    if not icc or im.mode not in ("RGB", "RGBA"):
        return im
    try:
        from PIL import ImageCms

        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        if "srgb" in (ImageCms.getProfileDescription(src) or "").lower():
            return im
        return ImageCms.profileToProfile(im, src, ImageCms.createProfile("sRGB"), outputMode=im.mode)
    except Exception:  # broken or unsupported profile: fall back to assuming sRGB
        return im


def _normalize_mode(im: Image.Image) -> Image.Image:
    mode = im.mode
    if mode in _HIGH_BIT_MODES or mode in ("L", "LA", "RGB", "RGBA"):
        return im
    try:
        if mode == "P":
            return im.convert("RGBA" if "transparency" in im.info else "RGB")
        if mode in ("PA", "RGBa"):
            return im.convert("RGBA")
        if mode == "La":
            return im.convert("LA")
        if mode == "1":
            return im.convert("L")
        return im.convert("RGB")  # CMYK, YCbCr, HSV, ...
    except (ValueError, OSError) as e:
        raise ImageLoadError(f"unsupported color mode {mode}: {e}") from None


def _reduce_uint8(im: Image.Image, f: int) -> np.ndarray:
    """Box-average an 8-bit image by integer factor f *in linear light*, strip by strip.

    Output channels: color (premultiplied if alpha present) followed by alpha.
    """
    w, h = im.size
    bands = len(im.getbands())
    has_alpha = im.mode in ("LA", "RGBA")
    cb = bands - has_alpha
    ow, oh = max(1, w // f), max(1, h // f)
    fx, fy = min(f, w), min(f, h)
    out = np.empty((oh, ow, bands), np.float32)
    rows_per_strip = max(1, _STRIP_PIXELS // max(1, ow * fx * fy))
    for oy in range(0, oh, rows_per_strip):
        n = min(rows_per_strip, oh - oy)
        a = np.asarray(im.crop((0, oy * fy, ow * fx, (oy + n) * fy)))
        a = a.reshape(n * fy, ow * fx, bands)
        color = SRGB_LUT[a[..., :cb]]
        if has_alpha:
            alpha = a[..., cb:].astype(np.float32) / 255.0
            color *= alpha
            out[oy:oy + n, :, cb:] = alpha.reshape(n, fy, ow, fx, 1).mean(axis=(1, 3))
        out[oy:oy + n, :, :cb] = color.reshape(n, fy, ow, fx, cb).mean(axis=(1, 3))
    return out


def _high_bit_to_linear(im: Image.Image, f: int) -> np.ndarray:
    a = np.asarray(im).astype(np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    if im.mode.startswith("I;16") or (im.mode == "I" and a.max(initial=0) > 255):
        a /= 65535.0
    elif im.mode == "I":
        a /= 255.0
    else:  # F: unknown range
        peak = float(a.max(initial=0))
        a = a / peak if peak > 1.0 else a
    lin = srgb_to_linear(a)
    h, w = lin.shape
    oh, ow = max(1, h // f), max(1, w // f)
    fy, fx = min(f, h), min(f, w)
    return lin[:oh * fy, :ow * fx].reshape(oh, fy, ow, fx).mean(axis=(1, 3))[..., None]


def load_linear(im: Image.Image, need_w: int, need_h: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Decode to (premultiplied linear RGB float32 [H,W,3], alpha [H,W] or None).

    The result keeps at least 2x oversampling relative to (need_w, need_h) in display orientation,
    so the final resample to the exact grid still averages many source pixels.
    """
    orientation = exif_orientation(im)
    sw, sh = (need_h, need_w) if orientation in (5, 6, 7, 8) else (need_w, need_h)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im.draft(None, (sw * 2, sh * 2))  # JPEG: decode at 1/2, 1/4 or 1/8 scale in the DCT domain
            im.load()
        im = _normalize_mode(im)
        im = _to_srgb(im)
        w, h = im.size
        f = max(1, min(w // (sw * 2), h // (sh * 2)))
        arr = _high_bit_to_linear(im, f) if im.mode in _HIGH_BIT_MODES else _reduce_uint8(im, f)
    except ImageLoadError:
        raise
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as e:
        raise ImageLoadError(f"cannot decode image: {e}") from None

    arr = _apply_orientation(arr, orientation)
    bands = arr.shape[2]
    if bands in (2, 4):
        color, alpha = arr[..., :-1], arr[..., -1]
        if alpha.min() >= 0.999:
            alpha = None
    else:
        color, alpha = arr, None
    if color.shape[2] == 1:
        color = np.repeat(color, 3, axis=2)
    return np.ascontiguousarray(color, np.float32), (None if alpha is None else np.ascontiguousarray(alpha))


def resample(a: np.ndarray, w: int, h: int) -> np.ndarray:
    """Exact float resize per channel: area-average (BOX) when shrinking, bilinear when enlarging."""
    if a.shape[0] == h and a.shape[1] == w:
        return a
    shrinking = w <= a.shape[1] and h <= a.shape[0]
    filt = Image.Resampling.BOX if shrinking else Image.Resampling.BILINEAR
    squeeze = a.ndim == 2
    if squeeze:
        a = a[..., None]
    chans = [np.asarray(Image.fromarray(np.ascontiguousarray(a[..., c])).resize((w, h), filt))
             for c in range(a.shape[2])]
    out = np.stack(chans, axis=-1).astype(np.float32)
    return out[..., 0] if squeeze else out
