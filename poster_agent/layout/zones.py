"""Layout templates: named zones (fractions of the free area between the top and bottom bands) that components are assigned to, so
posters get different structures instead of one top-to-bottom stack. Zones may overlap on purpose (a title sitting on the lower edge
of the art). Components without a zone go to the template's default zone. `classic` is the single full-area stack."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Zone:
    x: float
    y: float
    w: float
    h: float
    valign: str = "top"  # where the zone's content sits in spare height: top | center | bottom | balanced; "flow" = the poster's flow_valign


@dataclass(frozen=True)
class Template:
    zones: dict[str, Zone]
    default: str
    about: str
    region: tuple[float, float, float, float] | None = None  # default flow region (fractions of the safe box) when the plan sets none
    gap_scale: float = 1.0  # < 1: the designers' tight rhythm between blocks


SKELETONS = ("house", "split", "giant_word", "greeting", "photo_led")  # single-column structures taken from the institution's posters


TEMPLATES: dict[str, Template] = {
    "classic": Template({"main": Zone(0, 0, 1, 1, "flow")}, "main", "one full-width stack, top to bottom"),
    "stage": Template({
        "head": Zone(0, 0, 1, 0.14, "top"), "art": Zone(0, 0.12, 1, 0.40, "center"),
        "title": Zone(0, 0.48, 1, 0.26, "center"), "details": Zone(0, 0.76, 1, 0.24, "top"),
    }, "details", "logo/header strip, a tall art zone, the title laid over the lower edge of the art, details below"),
    "two_col": Template({
        "head": Zone(0, 0, 1, 0.14, "top"), "title": Zone(0, 0.14, 1, 0.32, "center"),
        "left": Zone(0, 0.48, 0.47, 0.52, "top"), "right": Zone(0.53, 0.48, 0.47, 0.52, "top"),
    }, "left", "header strip, a wide title row, then two columns (details left, art / list right)"),
    "house": Template({"main": Zone(0, 0, 1, 1, "top")}, "main",
                      "header strip, wide headline, kicker, fact line, content block, and an anchor at the bottom that takes the rest",
                      gap_scale=0.8),
    "split": Template({"main": Zone(0, 0, 1, 1, "top")}, "main",
                      "text column on the left 58%, a cutout (BottomCutout beside: true) standing on the bottom edge on the right",
                      region=(0.0, 0.0, 0.58, 1.0), gap_scale=0.85),
    "giant_word": Template({"main": Zone(0, 0, 1, 1, "center")}, "main", "one giant word or number as the hero, the facts around it"),
    "greeting": Template({"main": Zone(0, 0, 1, 1, "center")}, "main", "a centred image with one greeting line and the logo"),
    "photo_led": Template({"main": Zone(0, 0, 1, 1, "top")}, "main",
                          "a large photo (PhotoFade, anchor bottom) takes the lower part; the text sits above it", gap_scale=0.8),
    "top_art": Template({
        "head": Zone(0, 0, 1, 0.14, "top"), "art": Zone(0, 0.12, 1, 0.36, "center"), "main": Zone(0, 0.50, 1, 0.50, "flow"),
    }, "main", "header strip, art across the upper half, the title and details together in the lower half"),
}


def template(name: str) -> Template:
    return TEMPLATES.get(name) or TEMPLATES["classic"]
