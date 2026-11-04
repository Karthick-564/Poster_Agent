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


def _glue_numbers(words: list[str]) -> list[str]:
    """Consecutive digit groups (a phone number: '+91 81220 91077') are one unbreakable word."""
    import re
    out: list[str] = []
    for w in words:
        if out and re.fullmatch(r"\+?\d+", w) and re.fullmatch(r"[+\d ]+", out[-1]):
            out[-1] = f"{out[-1]} {w}"
        else:
            out.append(w)
    return out


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
            words = _glue_numbers(para.split())
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

    def wrap_balanced(self, text: str, family: str, style: str, size: float, max_w: float, tracking: float = 0.0) -> list[str]:
        """Same line count as the greedy wrap, but the breaks are chosen so the lines are as even as possible (no long line over a
        one-word stub, no orphan on the last line). Headlines and subtitles read better balanced. Falls back to the greedy lines when
        they cannot be improved, and never produces a line wider than max_w."""
        greedy = self.wrap(text, family, style, size, max_w, tracking)
        if len(greedy) < 2 or any("\n" in p for p in [text]) and len(text.split("\n")) > 1:
            return greedy
        words = _glue_numbers(text.split())
        n, k = len(words), len(greedy)
        if n < k + 1 or n > 40:
            return greedy
        width = lambda i, j: self.width(" ".join(words[i:j]), family, style, size, tracking)
        total = self.width(" ".join(words), family, style, size, tracking)
        target = total / k
        inf = float("inf")
        cost = [[inf] * (n + 1) for _ in range(k + 1)]
        back = [[0] * (n + 1) for _ in range(k + 1)]
        cost[0][0] = 0.0
        for line in range(1, k + 1):
            for j in range(line, n + 1):
                for i in range(line - 1, j):
                    if cost[line - 1][i] == inf:
                        continue
                    w = width(i, j)
                    if w > max_w:
                        continue
                    c = cost[line - 1][i] + (target - w) ** 2 * (1.5 if line == k else 1.0)  # the last line matters most (no stub)
                    if c < cost[line][j]:
                        cost[line][j], back[line][j] = c, i
        if cost[k][n] == inf:
            return greedy
        out, j = [], n
        for line in range(k, 0, -1):
            i = back[line][j]
            out.append(" ".join(words[i:j]))
            j = i
        out.reverse()
        return out

    def widest(self, lines: list[str], family: str, style: str, size: float, tracking: float = 0.0) -> float:
        return max((self.width(ln, family, style, size, tracking) for ln in lines), default=0.0)
