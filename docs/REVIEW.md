# asciify: engineering log (research → v1.0 → self-review → v2.0)

Every number below was measured on Windows 11, Python 3.12.10, Pillow 12.3, NumPy 2.5. The
scripts that produced them are in the repo:
[`v1/review_probe.py`](../v1/review_probe.py) (attacks v1) and
[`tools/benchmark.py`](../tools/benchmark.py) (v1 vs v2). The real photo used is a bundled
Windows 11 wallpaper (`ThemeC/img29.jpg`, 3840×2400, pastel lake and forest).

---

## 1. Research findings

| Topic | Finding | Consequence for the design |
|---|---|---|
| Character ramps | Paul Bourke's 70-level `$@B%8&WM#*oahk…` and 10-level `@%#*+=-:. ` are the de-facto standards; converters map luminance to *string index*. | Index mapping assumes equal density steps. **Measured** in Consolas: 30 order inversions, steps from −0.066 to +0.113, and `@` (0.363 coverage) is far denser than `$` (0.274), even though Bourke lists `$` first. → Rank glyphs by *measured* ink coverage (`tools/measure_glyphs.py`). |
| Aspect ratio | Pixels are square; terminal cells are ~2:1 tall. Common fixes divide rows by 2, 2.2 or use 0.55. | One magic constant can't be right for terminal, HTML and PNG at once. → A single `cell_ratio` drives sampling, and **each renderer enforces it** (CSS `line-height: calc(1ch / ratio)`, PNG cell = `advance / ratio`). |
| Luminance / color math | BT.601 luma on gamma-encoded RGB is what most tutorials use. Oklab (Ottosson 2020) is the modern perceptual space: two 3×3 matrices + cube root. | Average in **linear light**, pick tones in **Oklab L**, edit color (lift, chroma, gradients, gamut mapping) in **Oklab**. |
| Large images | `Image.draft()` decodes JPEG at 1/2, 1/4 or 1/8 scale in the DCT domain; `MAX_IMAGE_PIXELS` guards decompression bombs (error at 2× the limit, ~179 MP by default). | Draft-decode, validate size from the header before decoding, reduce strip-wise. |
| Terminal color | Truecolor via `COLORTERM`/`WT_SESSION`, 256-color cube 16–231 + gray ramp 232–255; `NO_COLOR` convention; Windows needs `ENABLE_VIRTUAL_TERMINAL_PROCESSING`. | Capability detection with graceful 256/16 fallback, per-line resets. |

Sources: [Paul Bourke ramps via Wikipedia: ASCII art](https://en.wikipedia.org/wiki/ASCII_art) ·
[Modulate: ASCII art effect explained](https://modulate.to/effects/ascii/) ·
[movq: (almost) square pixels in the terminal](https://movq.de/blog/postings/2016-12-17/0/POSTING-en.html) ·
[DEV: true-color ASCII generator in Python](https://dev.to/madhura_ravishan/painting-the-terminal-building-a-true-color-ascii-art-generator-in-python-2abj) ·
[Björn Ottosson: A perceptual color space for image processing](https://bottosson.github.io/posts/oklab/) ·
[Wikipedia: Oklab](https://en.wikipedia.org/wiki/Oklab_color_space) ·
[Pillow Image module docs (draft, reduce, MAX_IMAGE_PIXELS)](https://pillow.readthedocs.io/en/stable/reference/Image.html) ·
[42 Astounding Scripts: densitySort](https://www.astoundingscripts.com/art/create-your-own-ascii-art-palettes-densitysort/)

## 2. v1.0 draft

[`v1/asciify_v1.py`](../v1/asciify_v1.py), 60 lines: `Image.open().convert("RGB")` → `resize((w, int(h/w·w·0.55)))`
→ BT.601 luma → index into Bourke's ramp → truecolor escape per character / `<span style>` per
character. It works on a typical JPEG, which is exactly why each flaw below needed a deliberate
probe to find.

## 3. Self-review: vulnerabilities found in v1.0 (measured)

| # | Flaw | Evidence | Severity |
|---|---|---|---|
| 1 | **Aspect ratio is wrong everywhere** | One constant (0.55) for all targets. A circle renders **1.16×** too tall in Windows Terminal (9×19 px cells), **0.92×** in HTML with `line-height:1`, **1.17×** in HTML at default line-height. | High |
| 2 | **Crash on panoramas** | 4000×20 image → `ValueError: height and width must be > 0` (`int()` truncates rows to 0). | High |
| 3 | **Huge images crash or balloon memory** | 54 MP JPEG: +413 MB peak. 192 MP JPEG: `DecompressionBombError` traceback. With the guard disabled: 2.9 s, **1.48 GB** peak. Full-resolution decode for a 120-column result. | High |
| 4 | **Phone photos come out sideways** | EXIF orientation ignored: a portrait photo (orientation 6) renders as a 60×11 landscape grid. | High |
| 5 | **Transparent PNGs show hidden garbage** | Fully transparent pixel (hidden RGB = red) rendered as `'['` in **red**. | Medium |
| 6 | **16-bit images collapse** | `I;16` → RGB conversion clips: a full 0–65535 gradient yields **2** distinct characters. | Medium |
| 7 | **Ramp density is not monotonic** | 30 inversions in 70 characters (Consolas coverage). Result: speckled noise and `[ ] | /` streaks in smooth areas (see `docs/images/v1_vs_v2.png`). | High (quality) |
| 8 | **Gamma-incorrect averaging** | 1-px black/white checkerboard averages to **(128,128,128)**; physically correct is ~188. Fine detail and edges darken. | Medium |
| 9 | **Muddy, dim color** | Color = raw mean of the cell, but a glyph lights at most ~36% of its cell (`@` in Consolas), so perceived brightness ≈ coverage × color: dark twice over. Pastel photo: mean ink Oklab L 0.692, chroma 0.035. No contrast control. | High (the "muddy color" risk) |
| 10 | **Broken HTML** | 175 unescaped `<`, `>` or `&` characters from the ramp inside `<pre>` in one image. | High |
| 11 | **Bloated, bleeding output** | ANSI 19.3 bytes/cell (escape on every character); HTML 44.3 bytes/cell; only one reset for the whole block → color bleeds into wrapped lines and the prompt. No 256/16-color fallback, no `NO_COLOR`, no VT enable on legacy Windows consoles, `open(..., "w")` uses the locale encoding. | Medium |
| 12 | **No size-to-terminal** | Default width 100 regardless of terminal; wider art wraps and garbles. | Low |

## 4. v2.0 corrections and verification

| # | Fix | Verified |
|---|---|---|
| 1 | `grid_size()`: `rows = cols·(h/w)·cell_ratio`. HTML sets `line-height: calc(1ch / ratio)`, exact in *any* monospace font; PNG cell height = `advance / ratio`. Terminal default 0.5, tunable `--cell-ratio`. | Circle test: extent ratio 1.00 ± 0.06; PNG cell ratio 0.5 ± 0.03 (`GeometryTests`, `test_image_output_geometry`) |
| 2 | Rows/cols clamped to ≥1 and hard maxima of 1000, preserving aspect. | `test_extreme_panorama_does_not_crash`, `test_extreme_tall_image` |
| 3 | Header-only size check (friendly error, `--max-pixels`, default 200 MP) → `draft()` DCT downscale → strip-wise reduction (4 MP of floats at a time). | 54 MP JPEG: **0.10 s, +18 MB** (v1 0.50 s, +413 MB). 192 MP JPEG: **0.29 s, +56 MB** (v1 crash). |
| 4 | EXIF orientation applied with NumPy *after* reduction (cheap). | Pixel-exact vs `ImageOps.exif_transpose` for all 8 orientations |
| 5 | Premultiplied-alpha resampling; **alpha acts as ink coverage** (`tone·α`), so transparency is always blank, and anti-aliased edges thin out smoothly. Blocks mode composites over the page in linear light. | `test_transparent_pixels_are_background`, `test_palette_image_with_transparency` |
| 6 | High-bit-depth modes scaled by their true range (`I;16`, `I`, `F`). | 16-bit gradient uses ≥17 of 19 levels |
| 7 | Glyph coverage **measured** (Consolas, 96 px) and embedded; every ramp is sorted by it and mapped to the *nearest measured level*. New curated `detailed` ramp ` .:;=+cvxoawdO8MQ&@`: evenly spaced, shape-neutral (directional glyphs reserved for edges). | `test_ramps_are_sorted_by_measured_coverage`, max step < 0.18 |
| 8 | Decode through an sRGB→linear LUT, box-average in linear light, area-exact `BOX` resample to the grid; ICC profiles (Display P3, Adobe RGB) converted to sRGB first. | Checkerboard → **188** (`test_gamma_correct_downsampling`) |
| 9 | **Ink compensation** (`vivid`, default): in Oklab, lift ink lightness toward a readable target (0.90 on dark pages, 0.40 on light) by `--lift`, scale chroma by `--saturation`, then **gamut-map by chroma reduction at constant L and hue** (no hue shifts from clipping). Robust auto-levels (1st–99th percentile of opaque pixels, skipped for flat images). | Same photo: ink L **0.809**, chroma **0.044** (v1 0.692 / 0.035). `test_vivid_is_brighter_and_more_saturated_than_true`, gamut test checks L and hue preserved |
| 10 | `html.escape` on every glyph; standalone page with `aria-label`. | `test_html_is_escaped` |
| 11 | Escape emitted only on color change, skipped for spaces, colors rounded to 6 bits/channel (≤2/255 error) for longer runs; reset at every line end; truecolor → 256 → 16 fallback by nearest **Oklab** distance; `NO_COLOR`/`FORCE_COLOR`; VT mode enabled on Windows; UTF-8 file I/O. HTML uses deduplicated CSS classes. | ANSI **9.5** bytes/cell (v1 19.3); HTML **13.4** bytes/cell (v1 44.3); `test_ansi_*` |
| 12 | Terminal output fits both terminal width−1 and height−2 by default. | manual |

Beyond fixing the flaws, v2 also adds:
- **Three render modes:** `ascii`; `edges`, with Sobel structure-tensor orientation mapped to `| / - \`, where angles are computed in display units so the non-square cell grid doesn't skew them; and `blocks`, with truecolor half-blocks `▀` for 2× vertical resolution.
- **Error-diffusion dithering** in measured-coverage space.
- **Ten palettes:** image-derived `vivid` and `true`, plain `none`, and seven gradient palettes interpolated in Oklab (`mono`, `matrix`, `amber`, `synthwave`, `inferno`, `ice`, `sepia`).
- **A light-page mode.**
- **A local web UI:** drag-and-drop, paste, and file picker, with a sandboxed preview iframe, Host-header check against DNS rebinding, a 40 MB upload cap, and bounded parameters.

**Tests:** 47 unit tests, all passing (44 in v2.0, +3 in v2.1) (`python -m unittest discover -s tests`).

## 5. Decisions tuned by eye

* **Tone gamma 1.5 on dark pages, 1.0 on light pages / blocks.** Light glyphs on black look bolder
  than their ink coverage (irradiation), so a straight Oklab-L mapping produced walls of `@` in skies
  and dotted noise in dark vignettes. Compared 1.0 / 1.6 / 2.2 side by side on two photos; 1.5 kept
  sky gradients and water detail. Override with `--gamma`.
* **Ink lift 0.5, saturation 1.25.** Enough to fix the "dim" look without turning pastels neon.
* **Edge threshold 0.6** (RMS tone change per cell width) with coherence > 0.5. Outlines contours
  without sprouting slashes in texture.

## 6. Second review pass (v2.1)

A fresh adversarial pass over the v2.0 code. Each defect was reproduced before it was fixed, and
each fix has a regression test. The pipe test was verified to fail on the v2.0 exception ordering.

| # | Defect in v2.0 | Reproduction | Fix |
|---|---|---|---|
| 13 | **`asciify img \| head` reported an error.** `BrokenPipeError` is a subclass of `OSError`, so the generic handler caught it first. | `asciify: error: [Errno 32] Broken pipe`, exit 1 | Handle `BrokenPipeError` first and redirect stdout to devnull so the final flush stays silent. Exit 0. (`test_closed_pipe_is_not_an_error`) |
| 14 | **`--max-pixels` was only honored at `Image.open`.** Pillow re-checks its *global* limit during decode (GIF frames, TIFF tiles, `crop()`), after v2.0 had restored the default. A 190 MP TIFF would fail despite the 200 MP limit, and raising `--max-pixels` could not help. | With Pillow's global at 1000 px, a 100×100 GIF, TIFF and PNG all failed at `max_pixels=1e6` | `loader.pixel_limit()`: a thread-safe context manager covering the whole open+decode span, restoring Pillow's value afterward even when conversions overlap. (`test_decode_time_checks_follow_our_limit`, `test_pixel_limit_restored_after_concurrent_conversions`) |
| 15 | Ctrl+C during terminal output could leave the shell painted in the last ANSI color. | Code inspection | Emit a reset on `KeyboardInterrupt` when stdout is a TTY. |
| 16 | The web server shared cached FreeType faces across threads during PNG export. | Code inspection (FreeType faces are not thread-safe) | Serialize glyph-atlas rasterization with a lock (a few ms). |
| 17 | The web UI rejected dropped files with an empty MIME type, which some platforms report for AVIF/HEIC/TIFF. | Code inspection | Reject only files with a declared non-image type; the server validates the bytes regardless. |
| 18 | `~/photo.jpg` was not expanded when the shell does not expand it (for example, PowerShell with native commands). | Code inspection | `os.path.expanduser` for input, outputs and `--font`. |

Also checked and found fine: Python 3.9 grammar compatibility (`ast.parse(..., feature_version=(3, 9))`
over every file), a clean non-editable `pip install .` into a fresh venv (the `asciify` entry point
works and `webui.html` is packaged), and `NO_COLOR` precedence over `FORCE_COLOR` (it matches CPython).

## 7. Known limitations and trade-offs (honest list)

* **Terminal cell ratio can't be detected portably.** 0.5 is a good default; real terminals range
  roughly 0.45–0.55 depending on font, size and line spacing. Pass `--cell-ratio` if circles look oval.
* **Glyph coverage is Consolas-measured.** Other monospace fonts order glyphs similarly but not
  identically; regenerate with `tools/measure_glyphs.py YourFont.ttf`.
* **Huge non-JPEG images cost CPU for correctness.** A 54 MP PNG takes 2.1 s vs v1's 1.3 s (≈1 s of
  that is PNG decode, ≈1 s the linear-light reduction). Memory is still lower (+311 MB vs +413 MB).
  PNG can't be decoded at reduced scale; JPEG, the common case, is 5× faster than v1.
* **Edges mode is the slowest path** (≈0.56 s at 400 columns, 18 subpixels per cell). Soft edges
  that straddle a cell boundary can mark two adjacent cells.
* Animated GIF/WebP/APNG: first frame only. HEIC needs `pip install pillow-heif`.
* The web UI is a single-user local tool with no authentication; it binds to 127.0.0.1 only.
