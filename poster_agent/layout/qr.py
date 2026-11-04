"""QR codes as vector files: the payload is text from the brief (a URL), the code is generated deterministically (never an AI image), written
once to data/cache/qr/<hash>.svg and placed like an icon (an editable vector in VectorCraft)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import segno

from poster_agent.spec.brand import ROOT

QR_DIR = ROOT / "data" / "cache" / "qr"


def qr_svg(payload: str, dark: str = "#000000", light: str = "#ffffff", border: int = 2) -> Path:
    """SVG of a QR code for `payload` (error correction M, a light quiet zone so it scans on any ground). Cached by content."""
    key = hashlib.sha1(f"{payload}|{dark}|{light}|{border}".encode()).hexdigest()[:16]
    path = QR_DIR / f"{key}.svg"
    if not path.exists():
        QR_DIR.mkdir(parents=True, exist_ok=True)
        segno.make(payload, error="m").save(str(path), kind="svg", scale=10, dark=dark, light=light, border=border, xmldecl=False,
                                           svgns=True, nl=False)
    return path
