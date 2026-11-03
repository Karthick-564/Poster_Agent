"""Font lookup and text measurement for the layout solver.

Measures with the same TTF files VectorCraft renders with (brands/fonts + Windows system fonts). The compile step
reads back VectorCraft's real overflow/bounds, so any residual difference surfaces as a `fit` issue, never silently.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

from poster_agent.spec.brand import BRANDS_DIR

_REF_PX = 200  # measure at a large size, scale down: keeps hinting from distorting widths
SYSTEM_FONT_DIRS = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"]


class FontBook:
    def __init__(self, dirs: list[Path] | None = None):
        self._files: dict[tuple[str, str], Path] = {}
        for d in dirs or [BRANDS_DIR / "fonts", *SYSTEM_FONT_DIRS]:
            for f in Path(d).glob("*.ttf"):
                try:
                    family, style = ImageFont.truetype(str(f), 12).getname()
                except OSError:
                    continue
                self._files.setdefault((family.casefold(), style.casefold()), f)  # first dir wins (brand fonts)

    def path(self, family: str, style: str = "Regular") -> Path:
        fam, sty = family.casefold(), style.casefold()
        for key in ((fam, sty), (fam, "regular")):
            if key in self._files:
                return self._files[key]
        raise KeyError(f"font {family!r} {style!r} not available (have: {sorted({f for f, _ in self._files})})")

    @lru_cache(maxsize=256)
    def font(self, family: str, style: str = "Regular") -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(str(self.path(family, style)), _REF_PX)


class TextMeasurer:
    def __init__(self, book: FontBook | None = None):
        self.book = book or FontBook()

    def width(self, text: str, family: str, style: str, size: float, tracking: float = 0.0) -> float:
        """Advance width in pt; `tracking` is in 1/1000 em (VectorCraft's unit), applied between characters."""
        if not text:
            return 0.0
        w = self.book.font(family, style).getlength(text) * size / _REF_PX
        return w + tracking / 1000.0 * size * max(len(text) - 1, 0)

    def wrap(self, text: str, family: str, style: str, size: float, max_w: float, tracking: float = 0.0) -> list[str]:
        """Greedy word wrap. Existing newlines are kept; a single word wider than max_w gets its own line."""
        lines: list[str] = []
        for para in text.split("\n"):
            words = para.split()
            if not words:
                lines.append("")
                continue
            cur = words[0]
            for word in words[1:]:
                trial = f"{cur} {word}"
                if self.width(trial, family, style, size, tracking) <= max_w:
                    cur = trial
                else:
                    lines.append(cur)
                    cur = word
            lines.append(cur)
        return lines

    def widest(self, lines: list[str], family: str, style: str, size: float, tracking: float = 0.0) -> float:
        return max((self.width(ln, family, style, size, tracking) for ln in lines), default=0.0)
