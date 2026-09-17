"""asciify: image to ASCII art converter with perceptual color, for terminals, HTML and PNG.

    >>> from asciify import convert, Options, render
    >>> frame = convert("photo.jpg", Options(width=120, palette="vivid"))
    >>> print(render.to_ansi(frame))
"""
__version__ = "2.1.0"

from .engine import Frame, Options, convert, grid_size  # noqa: E402
from .loader import ImageLoadError  # noqa: E402

__all__ = ["Frame", "Options", "convert", "grid_size", "ImageLoadError", "__version__"]
