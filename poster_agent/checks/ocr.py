"""OCR helpers (RapidOCR, local CPU). Used to verify the *rendered* poster, not the plan.

Matching is character-level and space/punctuation-insensitive (the engine often drops spaces), via fuzzy partial ratio.
Non-Latin copy (e.g. Tamil) cannot be verified by this engine: such strings are reported as unverifiable, never as failures.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image
from rapidfuzz import fuzz

MIN_CROP_HEIGHT_PX = 64  # crops shorter than this are upscaled before OCR (small text recall)


@dataclass
class OcrLine:
    text: str
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def alnum(s: str) -> str:
    return re.sub(r"[\W_]+", "", s.lower())


def verifiable(s: str) -> bool:
    """True if the string is mostly ASCII letters/digits (what the engine can read) and long enough to be meaningful."""
    a = alnum(s)
    return len(a) >= 3 and sum(c.isascii() for c in a) / len(a) >= 0.8


def read_lines(img: Image.Image | np.ndarray, offset: tuple[float, float] = (0.0, 0.0), scale: float = 1.0) -> list[OcrLine]:
    arr = np.array(img.convert("RGB")) if isinstance(img, Image.Image) else img
    res, _ = _engine()(arr)
    out: list[OcrLine] = []
    for box, text, conf in res or []:
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        out.append(OcrLine(text, float(conf), offset[0] + min(xs) / scale, offset[1] + min(ys) / scale,
                           offset[0] + max(xs) / scale, offset[1] + max(ys) / scale))
    return out


def _with_margin(crop: Image.Image) -> Image.Image:
    """The text detector fails on thin, wide crops with no room above/below the text. Add a margin in the crop's own background
    colour (not by widening the crop, which would pull in neighbouring lines)."""
    a = np.array(crop)
    edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = tuple(int(v) for v in np.median(edge, axis=0))
    mx, my = 24, max(32, crop.height // 2)
    out = Image.new("RGB", (crop.width + 2 * mx, crop.height + 2 * my), bg)
    out.paste(crop, (mx, my))
    return out


def _read_text(crop: Image.Image, upscale: float = 1.0) -> str:
    crop = _with_margin(crop.convert("RGB"))
    k = upscale
    if crop.height < MIN_CROP_HEIGHT_PX:
        k = max(k, MIN_CROP_HEIGHT_PX / max(crop.height, 1) * 1.5)
    if k != 1.0:
        crop = crop.resize((max(1, round(crop.width * k)), max(1, round(crop.height * k))), Image.LANCZOS)
    lines = sorted(read_lines(crop), key=lambda ln: (round(ln.cy / 20), ln.x0))
    return " ".join(ln.text for ln in lines)


def read_crop(img: Image.Image, box: tuple[int, int, int, int]) -> str:
    """OCR one region (pixel box) and return its text in reading order.

    The detector is sensitive to crop shape, so try the tight crop first, then progressively wider source crops, then an
    upscale, and return the first attempt that reads anything. A neighbouring line leaking into a wider crop is harmless:
    matching is a partial ratio of the expected text inside what was read."""
    x0, y0, x1, y1 = box
    text = _read_text(img.crop(box))
    for grow in (12, 30):
        if len(alnum(text)) >= 3:
            return text
        w, h = img.size
        text = _read_text(img.crop((max(x0 - grow, 0), max(y0 - grow, 0), min(x1 + grow, w), min(y1 + grow, h))))
    if len(alnum(text)) < 3:
        text = _read_text(img.crop(box), upscale=1.6)
    return text


_LOOKALIKE = str.maketrans({"o": "0", "i": "1", "l": "1"})  # O/0 and I/l/1 are what OCR confuses most in times, years and numbers


def fold(s: str) -> str:
    """Lower-cased alphanumerics with look-alike characters merged (applied to both texts being compared, never to one alone)."""
    return alnum(s).lower().translate(_LOOKALIKE)


def match_score(expected: str, observed: str) -> float:
    """0..100: how well `expected` appears inside `observed`, ignoring case, spaces, punctuation and O/0, I/l/1 confusions."""
    e, o = alnum(expected), alnum(observed)
    if not e:
        return 100.0
    if not o:
        return 0.0
    return float(max(fuzz.partial_ratio(e, o), fuzz.partial_ratio(fold(expected), fold(observed))))


def line_score(expected: str, observed: str) -> float:
    """Like match_score but for a multi-line block: the WORST line decides, so a read that dropped a line cannot score 100 (the plain
    score only asks whether what was read appears inside the expected text)."""
    lines = [ln for ln in expected.split("\n") if alnum(ln)]
    return min((match_score(ln, observed) for ln in lines), default=100.0)
