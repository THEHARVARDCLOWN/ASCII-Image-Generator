"""Terminal capability detection: VT escape support on Windows, color depth and UTF-8 output."""
from __future__ import annotations

import os
import sys


def enable_vt_mode(stream=None) -> bool:
    """Turn on ANSI escape processing for this process's Windows console (no-op elsewhere)."""
    if os.name != "nt":
        return True
    stream = stream or sys.stdout
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = msvcrt.get_osfhandle(stream.fileno())
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False  # not a console (redirected)
        enable_virtual_terminal_processing = 0x0004
        if mode.value & enable_virtual_terminal_processing:
            return True
        return bool(kernel32.SetConsoleMode(handle, mode.value | enable_virtual_terminal_processing))
    except Exception:
        return False


def detect_color_depth(stream=None) -> str:
    """Return 'truecolor', '256', '16' or 'none', following NO_COLOR / FORCE_COLOR conventions."""
    stream = stream or sys.stdout
    env = os.environ
    if env.get("NO_COLOR"):
        return "none"
    is_tty = hasattr(stream, "isatty") and stream.isatty()
    if not is_tty and not env.get("FORCE_COLOR"):
        return "none"
    term = env.get("TERM", "")
    if term == "dumb":
        return "none"
    if env.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return "truecolor"
    if (env.get("WT_SESSION") or env.get("KITTY_WINDOW_ID") or env.get("TERM_PROGRAM") in
            ("iTerm.app", "WezTerm", "vscode", "Hyper", "ghostty", "Tabby")):
        return "truecolor"
    if os.name == "nt":  # Windows 10 1703+ consoles support 24-bit color once VT mode is on
        return "truecolor" if enable_vt_mode(stream) else "16"
    if "256" in term:
        return "256"
    return "16"


def ensure_utf8(stream=None) -> None:
    """Avoid UnicodeEncodeError for block glyphs when output is redirected on legacy code pages."""
    stream = stream or sys.stdout
    encoding = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
    if encoding != "utf8" and hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
