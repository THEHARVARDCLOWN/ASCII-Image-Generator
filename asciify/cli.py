"""Command-line interface: `asciify IMAGE [options]` and `asciify serve`."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import List, Optional

from . import __version__, color, engine, glyphs, loader, render, terminal

EXAMPLES = """examples:
  asciify photo.jpg                          fit the terminal, auto-detected color depth
  asciify photo.jpg -w 200 -o art.html -o art.png
  asciify photo.jpg -m edges -p synthwave    outline glyphs (| / - \\) plus a gradient palette
  asciify photo.jpg -m blocks                half-block pixels, 2x vertical resolution
  asciify photo.jpg --light -p none -o art.txt
  cat photo.png | asciify - -w 100
  asciify serve                              local drag-and-drop web UI
"""


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="asciify", epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Convert images to ASCII art with perceptual color for terminals, HTML and PNG.")
    p.add_argument("image", nargs="?", help="image file, or '-' to read from stdin")
    p.add_argument("--version", action="version", version=f"asciify {__version__}")
    p.add_argument("--list", action="store_true", help="list palettes and ramps, then exit")

    size = p.add_argument_group("size")
    size.add_argument("-w", "--width", type=_positive_int, help="columns (default: fit terminal; 160 for files)")
    size.add_argument("-H", "--height", type=_positive_int, help="rows (with --width too: stretch to exactly this)")
    size.add_argument("--cell-ratio", type=float, default=0.5,
                      help="character cell width/height of your display (default 0.5)")

    look = p.add_argument_group("look")
    look.add_argument("-m", "--mode", choices=engine.MODES, default="ascii")
    look.add_argument("-r", "--ramp", default="detailed",
                      help=f"{', '.join(glyphs.RAMPS)} or custom characters, sparse to dense (default detailed)")
    look.add_argument("-p", "--palette", choices=color.PALETTE_NAMES, default="vivid", metavar="PALETTE",
                      help=f"{', '.join(color.PALETTE_NAMES)} (default vivid)")
    look.add_argument("--light", action="store_true", help="target a light background (dense glyphs = dark)")
    look.add_argument("--bg", help="page background #rrggbb for HTML/PNG and transparency")
    look.add_argument("--contrast", type=float, default=1.0, help="auto-levels strength 0..1 (default 1)")
    look.add_argument("--gamma", type=float, default=None,
                      help="tone gamma; >1 darker mid-tones (default auto: 1.5 on dark pages, 1.0 on light)")
    look.add_argument("--saturation", type=float, default=1.25, help="chroma multiplier (default 1.25)")
    look.add_argument("--lift", type=float, default=0.5, help="ink lightness lift 0..1 for vivid (default 0.5)")
    look.add_argument("--dither", action="store_true", help="Floyd-Steinberg dithering across glyph levels")
    look.add_argument("--edge-threshold", type=float, default=engine.Options.edge_threshold,
                      help=f"edges mode sensitivity; lower = more edges (default {engine.Options.edge_threshold})")

    out = p.add_argument_group("output")
    out.add_argument("-o", "--output", action="append", default=[], metavar="FILE",
                     help="write .txt, .ans, .html or .png/.webp/.jpg (repeatable)")
    out.add_argument("--show", action="store_true", help="also print to the terminal when using -o")
    out.add_argument("--color", choices=("auto",) + render.COLOR_DEPTHS, default="auto",
                     help="terminal color depth (default auto; honors NO_COLOR/FORCE_COLOR)")
    out.add_argument("--fill-bg", action="store_true", help="paint the page background in the terminal too")
    out.add_argument("--font", help="TTF/OTF font for image output (default: a system monospace font)")
    out.add_argument("--font-size", type=_positive_int, default=16, help="font size in px for image output")
    p.add_argument("--max-pixels", type=float, default=loader.DEFAULT_MAX_PIXELS / 1e6, metavar="MP",
                   help=f"refuse larger images, in megapixels (default {loader.DEFAULT_MAX_PIXELS // 10**6})")
    return p


def _list() -> None:
    print("palettes:")
    for pal in color.PALETTES.values():
        print(f"  {pal.name:<10} {pal.description}")
    print("ramps:")
    for name, chars in glyphs.RAMPS.items():
        print(f"  {name:<10} {chars if len(chars) < 40 else chars[:37] + '...'}")


def _run(args: argparse.Namespace) -> int:
    if args.list:
        _list()
        return 0
    if not args.image:
        build_parser().error("the image argument is required (or use --list / serve)")

    to_terminal = not args.output or args.show
    depth = args.color
    if to_terminal:
        terminal.enable_vt_mode(sys.stdout)
        terminal.ensure_utf8(sys.stdout)
        if depth == "auto":
            depth = terminal.detect_color_depth(sys.stdout)

    max_w = max_h = None
    if not (args.width or args.height):
        if args.output and not args.show:
            max_w = 160
        else:
            size = shutil.get_terminal_size((100, 40))
            max_w, max_h = max(20, size.columns - 1), max(10, size.lines - 2)

    source = sys.stdin.buffer.read() if args.image == "-" else os.path.expanduser(args.image)
    opts = engine.Options(
        width=args.width, height=args.height, max_width=max_w, max_height=max_h, mode=args.mode,
        ramp=args.ramp, palette=args.palette, light=args.light,
        background=color.hex_to_rgb(args.bg) if args.bg else None, contrast=args.contrast, gamma=args.gamma,
        saturation=args.saturation, lift=args.lift, dither=args.dither, edge_threshold=args.edge_threshold,
        cell_ratio=args.cell_ratio, max_pixels=int(args.max_pixels * 1e6) if args.max_pixels > 0 else None)
    frame = engine.convert(source, opts)

    file_depth = args.color if args.color not in ("auto", "none") else "truecolor"
    for path in args.output:
        title = "asciify" if args.image == "-" else os.path.basename(args.image)
        render.save(frame, os.path.expanduser(path), depth=file_depth,
                    font_path=os.path.expanduser(args.font) if args.font else None,
                    font_size=args.font_size, title=title)
        print(f"wrote {path} ({frame.cols}x{frame.rows})", file=sys.stderr)
    if to_terminal:
        sys.stdout.write(render.to_ansi(frame, depth, fill_background=args.fill_bg) + "\n")
        sys.stdout.flush()
    return 0


def _serve(argv: List[str]) -> int:
    p = argparse.ArgumentParser(prog="asciify serve", description="Run the local drag-and-drop web UI.")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = p.parse_args(argv)
    from . import server

    server.serve(port=args.port, open_browser=not args.no_browser)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        if argv[:1] == ["serve"]:
            return _serve(argv[1:])
        return _run(build_parser().parse_args(argv))
    except BrokenPipeError:  # `asciify img.jpg | head`: the reader left; not an error. Must precede OSError.
        try:  # keep the interpreter's final flush from raising again
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except (OSError, ValueError):
            pass
        return 0
    except (loader.ImageLoadError, ValueError, OSError) as e:
        print(f"asciify: error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        try:  # an interrupt mid-print must not leave the shell painted in the last color
            if sys.stdout.isatty():
                sys.stdout.write(render.RESET + "\n")
                sys.stdout.flush()
        except (OSError, ValueError):
            pass
        return 130
