"""Regression tests. Each v1.0 finding in docs/REVIEW.md has at least one test here.

Run from the project root:  python -m unittest -v
"""
import http.client
import io
import json
import os
import re
import sys
import tempfile
import threading
import time
import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asciify import Options, cli, color, convert, engine, glyphs, grid_size, loader, render  # noqa: E402
from asciify.loader import ImageLoadError  # noqa: E402


def png_bytes(img, **kw):
    buf = io.BytesIO()
    img.save(buf, "PNG", **kw)
    return buf.getvalue()


class GeometryTests(unittest.TestCase):
    def test_square_image_displays_square(self):
        self.assertEqual(grid_size(400, 400, width=80, cell_ratio=0.5), (80, 40))
        self.assertEqual(grid_size(400, 400, width=80, cell_ratio=0.6), (80, 48))

    def test_circle_stays_round(self):
        img = Image.new("L", (600, 600), 0)
        ImageDraw.Draw(img).ellipse((0, 0, 599, 599), fill=255)
        f = convert(png_bytes(img), Options(width=80, palette="none", gamma=1.0))
        inked = np.array([[c != " " for c in row] for row in f.chars])
        h_extent = inked.any(axis=1).sum() / f.cell_ratio  # rows in cell-width units
        w_extent = inked.any(axis=0).sum()
        self.assertAlmostEqual(h_extent / w_extent, 1.0, delta=0.06)

    def test_extreme_panorama_does_not_crash(self):  # v1: ValueError height must be > 0
        f = convert(png_bytes(Image.new("RGB", (4000, 20), "gray")), Options(width=100))
        self.assertEqual((f.cols, f.rows), (100, 1))

    def test_extreme_tall_image(self):
        f = convert(png_bytes(Image.new("RGB", (10, 5000), "gray")), Options(max_width=100, max_height=40))
        self.assertEqual(f.rows, 40)
        self.assertGreaterEqual(f.cols, 1)

    def test_fit_box_respects_both_limits(self):
        cols, rows = grid_size(1000, 3000, max_width=120, max_height=30)
        self.assertLessEqual(cols, 120)
        self.assertLessEqual(rows, 30)

    def test_hard_limits_preserve_aspect(self):
        cols, rows = grid_size(100, 100000, width=5000)
        self.assertLessEqual(cols, engine.MAX_COLS)
        self.assertLessEqual(rows, engine.MAX_ROWS)


class LoaderTests(unittest.TestCase):
    def test_exif_orientation_matches_pillow_for_all_8_values(self):
        rng = np.random.default_rng(1)
        base = Image.fromarray(rng.integers(0, 255, (6, 10, 3), dtype=np.uint8))
        for orientation in range(1, 9):
            exif = Image.Exif()
            exif[0x0112] = orientation
            buf = io.BytesIO()
            base.save(buf, "PNG", exif=exif)
            im = loader.open_image(buf.getvalue())
            lin, _ = loader.load_linear(im, *loader.oriented_size(im))
            ours = color.linear_to_srgb8(lin)
            expected = np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")))
            self.assertEqual(ours.shape, expected.shape, f"orientation {orientation}")
            self.assertLessEqual(int(np.abs(ours.astype(int) - expected).max()), 1, f"orientation {orientation}")

    def test_phone_photo_is_upright(self):  # v1: sideways
        exif = Image.Exif()
        exif[0x0112] = 6
        buf = io.BytesIO()
        Image.new("RGB", (300, 100), "white").save(buf, "JPEG", exif=exif)
        f = convert(buf.getvalue(), Options(width=50))
        self.assertEqual(f.source_size, (100, 300))
        self.assertGreater(f.rows, f.cols / 2)

    def test_transparent_pixels_are_background(self):  # v1: hidden red pixels rendered
        img = Image.new("RGBA", (200, 200), (255, 0, 0, 0))
        ImageDraw.Draw(img).ellipse((50, 50, 150, 150), fill=(255, 255, 255, 255))
        f = convert(png_bytes(img), Options(width=40))
        self.assertEqual(f.chars[0][0], " ")
        self.assertNotEqual(f.chars[f.rows // 2][20], " ")
        r, g, _ = f.fg[0, 0].astype(int)
        self.assertLess(abs(r - g), 12)  # neutral, not the hidden red

    def test_palette_image_with_transparency(self):
        img = Image.new("P", (64, 64), 0)
        img.putpalette([255, 0, 0, 0, 255, 0] + [0] * 762)
        ImageDraw.Draw(img).rectangle((16, 16, 47, 47), fill=1)
        f = convert(png_bytes(img, transparency=0), Options(width=16))
        self.assertEqual(f.chars[0][0], " ")

    def test_16_bit_gradient_keeps_its_levels(self):  # v1: 2 distinct chars
        arr = np.linspace(0, 65535, 512)[None, :].repeat(32, 0).astype(np.uint16)
        f = convert(png_bytes(Image.fromarray(arr)), Options(width=64, palette="none", contrast=0, gamma=1.0))
        self.assertGreaterEqual(len(set(f.chars[f.rows // 2])), len(glyphs.RAMPS["detailed"]) - 2)

    def test_gamma_correct_downsampling(self):  # v1: (128,128,128)
        board = (np.indices((400, 400)).sum(0) % 2 * 255).astype(np.uint8)
        f = convert(png_bytes(Image.fromarray(board).convert("RGB")), Options(width=40, palette="true"))
        self.assertAlmostEqual(int(f.fg[5, 5, 0]), 188, delta=2)

    def test_pixel_limit_is_a_friendly_error(self):  # v1: DecompressionBombError traceback
        before = Image.MAX_IMAGE_PIXELS
        data = png_bytes(Image.new("L", (3000, 3000)))
        with self.assertRaisesRegex(ImageLoadError, "safety limit"):
            convert(data, Options(max_pixels=1_000_000))
        self.assertEqual(Image.MAX_IMAGE_PIXELS, before)  # Pillow's global is restored

    def test_decode_time_checks_follow_our_limit(self):  # v2.0: GIF/TIFF/crop re-checked Pillow's global
        before = Image.MAX_IMAGE_PIXELS
        img = Image.linear_gradient("L").resize((100, 100))
        for fmt, kw in (("GIF", {}), ("TIFF", {"tile": True, "tile_width": 64, "tile_length": 64}), ("PNG", {})):
            buf = io.BytesIO()
            img.save(buf, fmt, **kw)
            Image.MAX_IMAGE_PIXELS = 1000  # a host app with a stricter global must not break our explicit limit
            try:
                f = convert(buf.getvalue(), Options(width=20, max_pixels=1_000_000))
                self.assertEqual(f.cols, 20, fmt)
                self.assertEqual(Image.MAX_IMAGE_PIXELS, 1000, fmt)
            finally:
                Image.MAX_IMAGE_PIXELS = before

    def test_pixel_limit_restored_after_concurrent_conversions(self):
        before = Image.MAX_IMAGE_PIXELS
        data = png_bytes(Image.linear_gradient("L").resize((600, 600)))
        errors = []

        def work(limit):
            try:
                for _ in range(5):
                    convert(data, Options(width=40, max_pixels=limit))
            except Exception as e:  # pragma: no cover - reported below
                errors.append(e)

        threads = [threading.Thread(target=work, args=(lim,)) for lim in (None, 5_000_000, 300_000_000, 1_000_000)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(Image.MAX_IMAGE_PIXELS, before)

    def test_large_jpeg_is_decoded_at_reduced_scale(self):
        rng = np.random.default_rng(0)
        big = Image.fromarray(rng.integers(0, 255, (375, 500, 3), dtype=np.uint8)).resize((8000, 6000))
        buf = io.BytesIO()
        big.save(buf, "JPEG", quality=70)
        im = loader.open_image(buf.getvalue())
        im.draft(None, (240, 180))
        self.assertLessEqual(im.size[0], 1000)  # DCT scaling kicked in (1/8)
        t = time.perf_counter()
        f = convert(buf.getvalue(), Options(width=120))
        self.assertLess(time.perf_counter() - t, 3.0)
        self.assertEqual(f.cols, 120)

    def test_garbage_input(self):
        with self.assertRaisesRegex(ImageLoadError, "not a recognized image"):
            convert(b"definitely not an image")
        with self.assertRaisesRegex(ImageLoadError, "empty"):
            convert(b"")
        with self.assertRaisesRegex(ImageLoadError, "not found"):
            convert("no/such/file.png")

    def test_truncated_file(self):
        buf = io.BytesIO()
        Image.new("RGB", (400, 400), "red").save(buf, "PNG")
        with self.assertRaises(ImageLoadError):
            convert(buf.getvalue()[:200])

    def test_exotic_modes(self):
        for mode in ("1", "L", "LA", "P", "CMYK", "I", "I;16", "F"):
            img = Image.new(mode, (40, 40))
            data = png_bytes(img) if mode in ("1", "L", "LA", "P") else _tiff(img)
            f = convert(data, Options(width=10))
            self.assertEqual(f.cols, 10, mode)


def _tiff(img):
    buf = io.BytesIO()
    img.save(buf, "TIFF")
    return buf.getvalue()


class GlyphTests(unittest.TestCase):
    def test_ramps_are_sorted_by_measured_coverage(self):
        for name in glyphs.RAMPS:
            r = glyphs.get_ramp(name)
            self.assertTrue(np.all(np.diff(r.levels) >= 0), name)
            self.assertEqual((r.levels[0], r.levels[-1]), (0.0, 1.0))

    def test_detailed_ramp_is_evenly_spaced(self):
        steps = np.diff(glyphs.get_ramp("detailed").levels)
        self.assertLess(steps.max(), 0.18)

    def test_custom_unknown_glyphs_keep_user_order(self):
        r = glyphs.Ramp(" ·•●")
        self.assertEqual(r.chars, " ·•●")

    def test_dither_preserves_mean_density(self):
        r = glyphs.get_ramp("simple")
        target = np.full((40, 80), 0.37, np.float32)
        idx = r.dither(target)
        self.assertAlmostEqual(float(r.levels[idx].mean()), 0.37, delta=0.02)

    def test_invalid_ramp(self):
        with self.assertRaises(ValueError):
            glyphs.Ramp("aaaa")


class EdgeTests(unittest.TestCase):
    def _line(self, xy):
        img = Image.new("L", (480, 480), 0)
        ImageDraw.Draw(img).line(xy, fill=255, width=14)
        f = convert(png_bytes(img), Options(width=48, mode="edges", palette="none"))
        counts = {c: sum(row.count(c) for row in f.chars) for c in glyphs.EDGE_CHARS}
        return max(counts, key=counts.get), counts

    def test_vertical(self):
        self.assertEqual(self._line((240, 0, 240, 479))[0], "|")

    def test_horizontal(self):
        self.assertEqual(self._line((0, 240, 479, 240))[0], "-")

    def test_diagonals(self):
        self.assertEqual(self._line((0, 0, 479, 479))[0], "\\")
        self.assertEqual(self._line((0, 479, 479, 0))[0], "/")


class ColorTests(unittest.TestCase):
    def test_oklab_roundtrip(self):
        rgb = np.random.default_rng(2).random((500, 3)).astype(np.float32)
        back = color.oklab_to_linear(color.linear_to_oklab(rgb))
        self.assertLess(float(np.abs(back - rgb).max()), 1e-4)

    def test_oklab_reference_white(self):
        lab = color.linear_to_oklab(np.array([1.0, 1.0, 1.0], np.float32))
        np.testing.assert_allclose(lab, [1.0, 0.0, 0.0], atol=1e-3)

    def test_gamut_map_preserves_hue_and_lightness(self):
        lab = np.array([[0.6, 0.3, -0.3]], np.float32)  # far outside sRGB
        rgb = color.gamut_map(lab)
        self.assertTrue(((rgb >= 0) & (rgb <= 1)).all())
        back = color.linear_to_oklab(rgb)
        self.assertAlmostEqual(float(back[0, 0]), 0.6, delta=0.01)
        self.assertAlmostEqual(float(np.arctan2(back[0, 2], back[0, 1])), float(np.arctan2(-0.3, 0.3)), delta=0.03)

    def test_vivid_is_brighter_and_more_saturated_than_true(self):  # v1: muddy colors
        rng = np.random.default_rng(3)
        img = Image.fromarray((rng.random((60, 90, 3)) * 140).astype(np.uint8)).resize((600, 400))
        data = png_bytes(img)
        vivid, true = (convert(data, Options(width=60, palette=p)) for p in ("vivid", "true"))
        lab_v = color.linear_to_oklab(color.SRGB_LUT[vivid.fg])
        lab_t = color.linear_to_oklab(color.SRGB_LUT[true.fg])
        self.assertGreater(lab_v[..., 0].mean(), lab_t[..., 0].mean() + 0.1)
        self.assertGreater(np.hypot(lab_v[..., 1], lab_v[..., 2]).mean(), np.hypot(lab_t[..., 1], lab_t[..., 2]).mean())

    def test_ansi256_and_16_mapping(self):
        rgb = np.array([[255, 0, 0], [0, 0, 0], [255, 255, 255], [128, 128, 128]], np.uint8)
        self.assertEqual(color.to_ansi256(rgb).tolist()[:3], [196, 16, 231])
        self.assertEqual(color.to_ansi16(rgb).tolist()[:3], [9, 0, 15])

    def test_all_palettes_render(self):
        data = png_bytes(Image.linear_gradient("L").convert("RGB"))
        for name in color.PALETTE_NAMES:
            for light in (False, True):
                f = convert(data, Options(width=30, palette=name, light=light))
                self.assertEqual(f.fg.dtype, np.uint8, name)


class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        img = Image.linear_gradient("L").rotate(45).convert("RGB")
        cls.data = png_bytes(img)
        cls.frame = convert(cls.data, Options(width=60, ramp=" <&>@"))

    def test_html_is_escaped(self):  # v1: 175 raw '<' '>' '&' characters
        html = render.to_html(self.frame)
        pre = html.split("<pre", 1)[1].split("</pre>", 1)[0]
        text = re.sub(r"</?i[^>]*>", "", pre.split(">", 1)[1])
        self.assertNotRegex(text, r"[<>]")
        self.assertNotRegex(text, r"&(?!lt;|gt;|amp;)")

    def test_html_line_height_matches_cell_ratio(self):
        self.assertIn("line-height:calc(1ch / 0.5)", render.to_html(self.frame))

    def test_ansi_resets_every_line_and_skips_redundant_codes(self):  # v1: one reset, 19 B/char
        ansi = render.to_ansi(self.frame)
        lines = ansi.split("\n")
        self.assertEqual(len(lines), self.frame.rows)
        for line in lines:
            if "\x1b[" in line:
                self.assertTrue(line.endswith("\x1b[0m"))
        flat = convert(png_bytes(Image.new("RGB", (300, 300), (40, 160, 90))), Options(width=50, contrast=0))
        for line in render.to_ansi(flat).split("\n"):
            self.assertLessEqual(line.count("\x1b["), 2)  # one color code + one reset per line

    def test_ansi_depths(self):
        self.assertIn("38;5;", render.to_ansi(self.frame, "256"))
        self.assertRegex(render.to_ansi(self.frame, "16"), r"\x1b\[(3|9)\dm")
        self.assertNotIn("\x1b", render.to_ansi(self.frame, "none"))

    def test_blocks_mode_outputs(self):
        f = convert(self.data, Options(width=40, mode="blocks"))
        self.assertEqual(f.lower.shape, f.fg.shape)
        self.assertIn("▀", render.to_ansi(f))
        self.assertIn("data:image/png;base64,", render.to_html(f))
        im = render.to_image(f, font_size=8)
        self.assertEqual(im.size, (40 * 4, f.rows * 2 * 4))

    def test_image_output_geometry(self):
        im = render.to_image(self.frame, font_size=16)
        cell_w = im.width / self.frame.cols
        cell_h = im.height / self.frame.rows
        self.assertAlmostEqual(cell_w / cell_h, 0.5, delta=0.03)

    def test_save_all_formats(self):
        with tempfile.TemporaryDirectory() as d:
            for ext in ("txt", "ans", "html", "png", "webp"):
                path = os.path.join(d, f"out.{ext}")
                render.save(self.frame, path)
                self.assertGreater(os.path.getsize(path), 100, ext)
            with self.assertRaises(ValueError):
                render.save(self.frame, os.path.join(d, "out.docx"))


class CliTests(unittest.TestCase):
    def test_cli_writes_files(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "in.png")
            Image.linear_gradient("L").save(src)
            out = os.path.join(d, "out.html")
            self.assertEqual(cli.main([src, "-w", "40", "-o", out, "-p", "amber"]), 0)
            with open(out, encoding="utf-8") as fh:
                self.assertIn("<pre", fh.read())

    def test_cli_errors_are_friendly(self):
        self.assertEqual(cli.main(["missing.png", "-o", "x.txt"]), 1)
        self.assertEqual(cli.main([__file__, "-o", "x.txt"]), 1)

    def test_closed_pipe_is_not_an_error(self):  # v2.0: "asciify: error: [Errno 32] Broken pipe", exit 1
        import subprocess

        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "in.png")  # noise: ~1.6 MB of escapes, far beyond any pipe buffer
            Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)).save(src)
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            proc = subprocess.Popen(
                [sys.executable, "-m", "asciify", src, "-w", "300", "-m", "blocks", "--color", "truecolor"],
                cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            proc.stdout.read(64)  # like `| head -c 64`
            proc.stdout.close()
            stderr = proc.stderr.read()
            proc.stderr.close()
            self.assertEqual(proc.wait(timeout=60), 0, stderr)
            self.assertNotIn(b"error", stderr.lower())


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from asciify import server

        cls.server_mod = server
        cls.httpd = server.make_server(port=0)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.image = png_bytes(Image.linear_gradient("L").convert("RGB"))

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def request(self, method, path, body=None, host=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        headers = {"Host": host or f"127.0.0.1:{self.port}"}
        conn.request(method, path, body=body, headers=headers)
        res = conn.getresponse()
        data = res.read()
        conn.close()
        return res, data

    def test_index_and_meta(self):
        res, body = self.request("GET", "/")
        self.assertEqual(res.status, 200)
        self.assertIn(b"asciify", body)
        res, body = self.request("GET", "/api/meta")
        self.assertIn("vivid", [p["name"] for p in json.loads(body)["palettes"]])

    def test_convert_formats(self):
        for fmt, marker in (("html", b"<pre"), ("txt", b"@"), ("png", b"\x89PNG")):
            res, body = self.request("POST", f"/api/convert?format={fmt}&width=50", self.image)
            self.assertEqual(res.status, 200, body[:200])
            self.assertIn(marker, body[:4000] if fmt != "png" else body[:8])
            self.assertEqual(res.getheader("X-Asciify-Grid").split("x")[0], "50")

    def test_rejects_bad_input(self):
        res, _ = self.request("POST", "/api/convert?width=99999", self.image)
        self.assertEqual(res.status, 400)
        res, _ = self.request("POST", "/api/convert?palette=<script>", self.image)
        self.assertEqual(res.status, 400)
        res, body = self.request("POST", "/api/convert", b"not an image")
        self.assertEqual(res.status, 400)
        self.assertIn(b"not a recognized image", body)

    def test_dns_rebinding_host_is_refused(self):
        res, _ = self.request("GET", "/", host="evil.example:80")
        self.assertEqual(res.status, 403)


if __name__ == "__main__":
    unittest.main(verbosity=2)
