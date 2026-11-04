"""Prompt builders for the diffusion models.

Ideogram 4 is trained ONLY on structured JSON captions (docs/prompting.md in ideogram-oss/ideogram4). A prompt that does not
follow the schema is much more likely to be blocked ("Image blocked by safety filter"). So the caption is built to the exact
schema and checked by `validate_ideogram_caption`, a mirror of the official CaptionVerifier rules:

  top level: high_level_description, style_description, compositional_deconstruction (required)
  style_description: aesthetics, lighting, then EXACTLY ONE of photo / art_style (with medium in its fixed place), color_palette last
      photo caption:      aesthetics, lighting, photo, medium, color_palette
      non-photo caption:  aesthetics, lighting, medium, art_style, color_palette
  compositional_deconstruction: background first, then elements (both required);
      obj element: type, bbox, desc, color_palette    bbox = [y_min, x_min, y_max, x_max] in 0-1000
  colours: uppercase #RRGGBB (<=16 overall, <=5 per element); serialised compactly with ensure_ascii=False.

The layout's text zones are not asked for with instructions: they are expressed as a *described scene* (open, smooth space where the
text goes) plus an element whose bbox covers where the visual interest should live. Art prompts describe imagery only — never text.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from poster_agent.spec import Box, ElementKind, PlacedElement

BAND_NAMES = ["very top", "upper", "upper middle", "lower middle", "lower", "very bottom"]
CALM_THRESHOLD = 0.10  # a band with >=10% of its area under text must stay calm
HEX_UPPER = re.compile(r"^#[0-9A-F]{6}$")
MEDIUMS = ("photograph", "illustration", "3d_render", "painting", "graphic_design")


class ArtDirection(BaseModel):
    description: str  # imagery only: what the picture shows
    style: str = ""  # aesthetics, e.g. "clean minimal geometric, soft gradients"
    lighting: str = ""
    medium: str = "digital illustration"
    tone: str = "light"  # light | dark — which text colour the art must support
    palette: list[str] = Field(default_factory=list)  # #hex


@dataclass
class CalmLayout:
    """Where the poster's text sits, expressed for the image model."""
    text: str  # a described scene sentence ("smooth open space fills the upper part ...")
    busy_ranges: list[tuple[int, int]] = field(default_factory=list)  # (y_min, y_max) in 0-1000 where visual interest may live
    all_calm: bool = False


def text_coverage_bands(placed: list[PlacedElement], canvas_w: float, canvas_h: float, n: int = len(BAND_NAMES)) -> list[float]:
    """Fraction of each horizontal band covered by text boxes (images/shapes are ignored)."""
    cover = [0.0] * n
    bh = canvas_h / n
    for el in placed:
        if el.kind != ElementKind.text:
            continue
        for i in range(n):
            band = Box(x=0, y=i * bh, w=canvas_w, h=bh)
            cover[i] += el.box.intersection_area(band) / (canvas_w * bh)
    return [min(c, 1.0) for c in cover]


def calm_layout(placed: list[PlacedElement], canvas_w: float, canvas_h: float) -> CalmLayout:
    cover = text_coverage_bands(placed, canvas_w, canvas_h)
    n = len(cover)
    calm = [BAND_NAMES[i] for i, c in enumerate(cover) if c >= CALM_THRESHOLD]
    busy_idx = [i for i, c in enumerate(cover) if c < CALM_THRESHOLD]
    if not calm:
        return CalmLayout("The visual interest can spread across the whole image.", [(0, 1000)])
    if not busy_idx:
        return CalmLayout("A calm, softly graded field with only subtle texture, evenly toned from edge to edge.", [], all_calm=True)
    ranges: list[tuple[int, int]] = []
    start = prev = busy_idx[0]
    for i in busy_idx[1:] + [None]:
        if i is not None and i == prev + 1:
            prev = i
            continue
        ranges.append((round(start / n * 1000), round((prev + 1) / n * 1000)))
        if i is not None:
            start = prev = i
    return CalmLayout(f"Smooth, soft, low-detail open space fills the {', '.join(calm)} part of the frame.",
                      ranges)


def calm_description(placed: list[PlacedElement], canvas_w: float, canvas_h: float) -> str:
    return calm_layout(placed, canvas_w, canvas_h).text


def normalize_medium(free_text: str) -> str:
    t = free_text.lower()
    if "photo" in t:
        return "photograph"
    if "3d" in t or "render" in t:
        return "3d_render"
    if "paint" in t:
        return "painting"
    if any(w in t for w in ("graphic", "vector", "design", "geometric")):
        return "graphic_design"
    return "illustration"


def _hex(c: str) -> str:
    return c.upper()


def ideogram_caption(art: ArtDirection, calm: CalmLayout | str) -> dict:
    """The structured caption, built in the exact key order the model was trained on."""
    if isinstance(calm, str):
        calm = CalmLayout(text=calm, busy_ranges=[])
    desc = art.description.strip().rstrip(".")
    medium = normalize_medium(art.medium)
    style: dict = {
        "aesthetics": art.style.strip() or "clean, refined, professional",
        "lighting": art.lighting.strip() or "soft, even, diffused lighting",
    }
    if medium == "photograph":
        style["photo"] = "wide shot, soft focus, shallow depth of field"
        style["medium"] = medium
    else:
        style["medium"] = medium
        style["art_style"] = art.style.strip() or "refined, minimal, professional"
    if art.palette:
        style["color_palette"] = [_hex(c) for c in art.palette][:16]

    elements: list[dict] = []
    for y0, y1 in calm.busy_ranges:
        el: dict = {"type": "obj"}
        if (y0, y1) != (0, 1000):
            el["bbox"] = [y0, 0, y1, 1000]
        el["desc"] = f"The main visual interest of the scene: {desc[:1].lower() + desc[1:]}."
        elements.append(el)
    if not elements:
        elements.append({"type": "obj", "desc": "A soft, smooth wash of colour with faint, subtle texture across the frame."})
    return {
        "high_level_description": f"{desc}.",
        "style_description": style,
        "compositional_deconstruction": {"background": calm.text, "elements": elements},
    }


def ideogram_prompt(art: ArtDirection, calm: CalmLayout | str) -> dict:
    return ideogram_caption(art, calm)


def caption_text(caption: dict) -> str:
    """Serialisation the model expects: compact separators, literal (non-escaped) unicode."""
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def validate_ideogram_caption(c: dict) -> list[str]:
    """Problems that would make Ideogram treat the prompt as off-schema (and risk a false 'blocked by safety filter')."""
    errs: list[str] = []

    def order(keys: list[str], expected: list[str], where: str) -> None:
        if [k for k in expected if k in keys] != keys:
            errs.append(f"{where}: keys {keys} are not in the required order {expected}")

    order(list(c), ["high_level_description", "style_description", "compositional_deconstruction"], "top level")
    if "compositional_deconstruction" not in c:
        errs.append("compositional_deconstruction is required")
        return errs
    if not isinstance(c.get("high_level_description"), str) or not c.get("high_level_description"):
        errs.append("high_level_description should be a non-empty string")
    sd = c.get("style_description")
    if sd is not None:
        has_photo, has_art = "photo" in sd, "art_style" in sd
        if has_photo == has_art:
            errs.append("style_description must contain exactly one of photo / art_style")
        for req in ("aesthetics", "lighting", "medium"):
            if req not in sd:
                errs.append(f"style_description is missing {req}")
        order(list(sd), ["aesthetics", "lighting", "photo", "medium", "color_palette"] if has_photo
              else ["aesthetics", "lighting", "medium", "art_style", "color_palette"], "style_description")
        if sd.get("medium") not in MEDIUMS:
            errs.append(f"style_description.medium {sd.get('medium')!r} is not one of {MEDIUMS}")
        pal = sd.get("color_palette", [])
        if len(pal) > 16 or any(not HEX_UPPER.match(p) for p in pal):
            errs.append("style_description.color_palette must be <=16 uppercase #RRGGBB entries")
    cd = c["compositional_deconstruction"]
    order(list(cd), ["background", "elements"], "compositional_deconstruction")
    if not isinstance(cd.get("background"), str) or "elements" not in cd:
        errs.append("compositional_deconstruction needs background (string) and elements (list)")
    for i, el in enumerate(cd.get("elements", [])):
        order(list(el), ["type", "bbox", "text", "desc", "color_palette"] if el.get("type") == "text"
              else ["type", "bbox", "desc", "color_palette"], f"element {i}")
        if el.get("type") not in ("obj", "text"):
            errs.append(f"element {i}: type must be obj or text")
        bb = el.get("bbox")
        if bb is not None and not (len(bb) == 4 and all(isinstance(v, int) and 0 <= v <= 1000 for v in bb)
                                   and bb[0] < bb[2] and bb[1] < bb[3]):
            errs.append(f"element {i}: bbox must be [y_min, x_min, y_max, x_max] ints in 0-1000 with min < max")
        if len(el.get("color_palette", [])) > 5 or any(not HEX_UPPER.match(p) for p in el.get("color_palette", [])):
            errs.append(f"element {i}: color_palette must be <=5 uppercase #RRGGBB entries")
    return errs


def qwen_subject_prompt(subject: str) -> str:
    return (f"{subject.rstrip('. ')}. Single subject, centered, fully visible with generous margin, isolated on a plain flat "
            "light grey studio background, soft even lighting, crisp clean edges, high detail, photorealistic render.")


# ---- FLUX 3 Klein: plain prose, no negative prompt (cfg 1) -----------------------------------------------------------------

NAMED_COLORS = {
    "navy": (30, 40, 66), "deep blue": (0, 79, 146), "royal blue": (27, 73, 144), "sky blue": (135, 190, 235), "teal": (0, 125, 130),
    "mint": (133, 206, 189), "aqua": (64, 200, 200), "green": (60, 140, 80), "olive": (143, 138, 51), "gold": (200, 160, 40),
    "orange": (230, 130, 40), "red": (200, 40, 40), "burgundy": (126, 2, 70), "maroon": (112, 83, 85), "mauve": (176, 160, 161),
    "purple": (110, 60, 160), "white": (255, 255, 255), "soft white": (245, 245, 248), "pale grey-blue": (221, 227, 237),
    "light grey": (200, 199, 204), "silver grey": (160, 162, 168), "charcoal": (65, 64, 66), "black": (10, 10, 14),
}


def color_words(palette: list[str], limit: int = 4) -> str:
    """Hex colours -> colour names a diffusion model understands ('deep navy, mint and soft white')."""
    names: list[str] = []
    for hx in palette:
        h = hx.lstrip("#")
        if len(h) != 6:
            continue
        rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        name = min(NAMED_COLORS, key=lambda n: sum((a - b) ** 2 for a, b in zip(NAMED_COLORS[n], rgb)))
        if name not in names:
            names.append(name)
    names = names[:limit]
    if not names:
        return ""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def focus_phrase(ranges: list[tuple[int, int]]) -> str:
    """Where the visual interest should sit, in words ('the lower third of the frame')."""
    if not ranges:
        return ""
    y0, y1 = min(r[0] for r in ranges), max(r[1] for r in ranges)
    span, mid = y1 - y0, (y0 + y1) / 2
    if span >= 750:
        return "across most of the frame"
    where = "upper" if mid < 340 else ("middle" if mid < 660 else "lower")
    size = "third" if span <= 400 else "half"
    return f"in the {where} {size} of the frame"


def flux_prompt(art: ArtDirection, calm: CalmLayout | str) -> str:
    """Prose caption for FLUX. Describes the scene and where it is busy/calm as a picture, never as a rule or a negation."""
    if isinstance(calm, str):
        calm = CalmLayout(text=calm)
    parts = [art.description.strip().rstrip(".") + "."]
    if art.style.strip():
        parts.append(art.style.strip().rstrip(".").capitalize() + ".")
    if art.lighting.strip():
        parts.append(art.lighting.strip().rstrip(".").capitalize() + ".")
    colors = color_words(art.palette)
    if colors:
        parts.append(f"The colour palette is {colors}.")
    parts.append("The whole image is a soft, muted backdrop with gentle gradients and low contrast, quiet enough for text to sit on it.")
    if calm.text:
        parts.append(calm.text.rstrip(".") + ".")
    focus = focus_phrase(calm.busy_ranges)
    if focus and not calm.all_calm:
        parts.append(f"The main visual interest sits {focus}.")
    return " ".join(parts)
