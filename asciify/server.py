"""Local web UI: drop, pick or paste an image, tune settings live, download HTML/PNG/TXT.

Standard library only. It binds to 127.0.0.1, checks the Host header against DNS rebinding,
caps upload size and validates every parameter before it reaches the engine.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import __version__, color, engine, glyphs, loader, render

MAX_UPLOAD_BYTES = 40 * 1024 * 1024
FORMATS = {
    "html": "text/html; charset=utf-8",
    "png": "image/png",
    "txt": "text/plain; charset=utf-8",
    "ans": "text/plain; charset=utf-8",
}
PAGE = Path(__file__).with_name("webui.html")


def options_from_query(query: dict):
    """Parse and bound-check request parameters. Raises ValueError with a user-facing message."""
    def get(name, default=None):
        return query.get(name, [default])[0]

    def num(name, default, lo, hi, cast=float):
        raw = get(name)
        if raw in (None, ""):
            return default
        try:
            value = cast(raw)
        except ValueError:
            raise ValueError(f"invalid {name}: {raw!r}") from None
        if not (lo <= value <= hi):
            raise ValueError(f"{name} must be between {lo} and {hi}")
        return value

    def choice(name, default, allowed):
        value = get(name, default)
        if value not in allowed:
            raise ValueError(f"invalid {name}: {value!r}")
        return value

    ramp = get("ramp", "detailed") or "detailed"
    if len(ramp) > 128:
        raise ValueError("custom ramp is too long (max 128 characters)")
    gamma = get("gamma")
    opts = engine.Options(
        width=num("width", 120, 8, 400, int),
        mode=choice("mode", "ascii", engine.MODES),
        ramp=ramp,
        palette=choice("palette", "vivid", color.PALETTE_NAMES),
        light=get("light", "0") == "1",
        dither=get("dither", "0") == "1",
        contrast=num("contrast", 1.0, 0.0, 1.0),
        gamma=None if gamma in (None, "", "auto") else num("gamma", None, 0.2, 5.0),
        saturation=num("saturation", 1.25, 0.0, 3.0),
        lift=num("lift", 0.5, 0.0, 1.0),
        edge_threshold=num("edge", engine.Options.edge_threshold, 0.05, 5.0),
        cell_ratio=num("ratio", 0.5, 0.25, 1.0),
    )
    return opts, choice("format", "html", FORMATS)


class Handler(BaseHTTPRequestHandler):
    server_version = f"asciify/{__version__}"
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, body: bytes, content_type: str, extra: dict = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _error(self, status: int, message: str, close: bool = False) -> None:
        extra = {"Connection": "close"} if close else None
        if close:
            self.close_connection = True
        self._send(status, json.dumps({"error": message}).encode(), "application/json", extra)

    def _host_ok(self) -> bool:
        return self.headers.get("Host", "") in self.server.allowed_hosts  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_ok():
            return self._error(403, "forbidden host")
        path = urlsplit(self.path).path
        if path == "/":
            csp = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                   "img-src 'self' data: blob:; frame-src 'self' about:; base-uri 'none'; form-action 'none'")
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8", {"Content-Security-Policy": csp})
        if path == "/api/meta":
            meta = {
                "version": __version__,
                "modes": list(engine.MODES),
                "palettes": [{"name": p.name, "description": p.description} for p in color.PALETTES.values()],
                "ramps": glyphs.RAMPS,
                "defaults": {"edge": engine.Options.edge_threshold},
                "maxUploadBytes": MAX_UPLOAD_BYTES,
            }
            return self._send(200, json.dumps(meta).encode(), "application/json")
        self._error(404, "not found")

    do_HEAD = do_GET

    def do_POST(self) -> None:  # noqa: N802
        if not self._host_ok():
            return self._error(403, "forbidden host", close=True)
        parts = urlsplit(self.path)
        if parts.path != "/api/convert":
            return self._error(404, "not found", close=True)
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self._error(411, "Content-Length required", close=True)
        if length > MAX_UPLOAD_BYTES:
            return self._error(413, f"upload exceeds {MAX_UPLOAD_BYTES // 2**20} MB", close=True)
        if length <= 0:
            return self._error(400, "empty upload")
        data = self.rfile.read(length)

        try:
            opts, fmt = options_from_query(parse_qs(parts.query))
            started = time.perf_counter()
            frame = engine.convert(data, opts)
            if fmt == "html":
                body = render.to_html(frame, title="asciify").encode("utf-8")
            elif fmt == "png":
                import io

                buf = io.BytesIO()
                render.to_image(frame).save(buf, "PNG")
                body = buf.getvalue()
            elif fmt == "ans":
                body = (render.to_ansi(frame, "truecolor") + render.RESET + "\n").encode("utf-8")
            else:
                body = (render.to_text(frame) + "\n").encode("utf-8")
            elapsed = (time.perf_counter() - started) * 1000
        except (ValueError, loader.ImageLoadError) as e:
            return self._error(400, str(e))
        except MemoryError:
            return self._error(413, "image too large to process")
        except Exception:  # never leak a traceback to the client
            traceback.print_exc()
            return self._error(500, "conversion failed; see server log")
        self._send(200, body, FORMATS[fmt], {
            "X-Asciify-Grid": f"{frame.cols}x{frame.rows}",
            "X-Asciify-Source": f"{frame.source_size[0]}x{frame.source_size[1]}",
            "X-Asciify-Ms": f"{elapsed:.0f}",
        })

    def log_message(self, fmt: str, *args) -> None:
        if getattr(self.server, "verbose", False):
            sys.stderr.write(f"[asciify] {self.address_string()} {fmt % args}\n")


def make_server(port: int = 8765, attempts: int = 20) -> ThreadingHTTPServer:
    """Bind to localhost on the first free port starting at `port` (0 = any free port)."""
    last_error = None
    for candidate in ([0] if port == 0 else range(port, port + attempts)):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", candidate), Handler)
            break
        except OSError as e:
            last_error = e
    else:
        raise OSError(f"no free port in {port}-{port + attempts - 1}: {last_error}")
    bound = httpd.server_address[1]
    httpd.daemon_threads = True
    httpd.allowed_hosts = {f"127.0.0.1:{bound}", f"localhost:{bound}"}  # type: ignore[attr-defined]
    return httpd


def serve(port: int = 8765, open_browser: bool = True, verbose: bool = False) -> None:
    httpd = make_server(port)
    httpd.verbose = verbose  # type: ignore[attr-defined]
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"asciify web UI: {url}   (Ctrl+C to stop)", file=sys.stderr)
    if open_browser:
        threading.Timer(0.4, webbrowser.open, (url,)).start()
    try:
        httpd.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
