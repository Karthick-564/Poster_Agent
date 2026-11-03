"""Poster formats. Document units are points (VectorCraft's unit); 1 pt == 1 px at export scale 1."""
from __future__ import annotations

from dataclasses import dataclass

MM_TO_PT = 72.0 / 25.4


@dataclass(frozen=True)
class FormatSpec:
    name: str
    width_pt: float
    height_pt: float
    kind: str  # "digital" | "print"
    export_scale: float  # pixels per point for the PNG export
    bleed_mm: float = 0.0

    @property
    def dpi(self) -> float:
        return 72.0 * self.export_scale


# Digital: 2x export (e.g. story -> 2160x3840, matching the designers' masters). Print: 300 DPI + 3 mm bleed.
FORMATS: dict[str, FormatSpec] = {
    "story": FormatSpec("story", 1080, 1920, "digital", 2.0),
    "fourfive": FormatSpec("fourfive", 1080, 1350, "digital", 2.0),
    "square": FormatSpec("square", 1080, 1080, "digital", 2.0),
    "a3": FormatSpec("a3", 841.89, 1190.55, "print", 300 / 72, bleed_mm=3.0),
    "a4": FormatSpec("a4", 595.28, 841.89, "print", 300 / 72, bleed_mm=3.0),
}


def get_format(name: str) -> FormatSpec:
    try:
        return FORMATS[name.lower()]
    except KeyError:
        raise ValueError(f"unknown format {name!r}; known: {sorted(FORMATS)}") from None
